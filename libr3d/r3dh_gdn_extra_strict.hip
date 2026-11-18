#include "r3dh_gdn_extra.h"
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstring>
#include <cstdint>
#if defined(__FAST_MATH__)
#error r3dh_gdn_extra requires -fno-fast-math -ffp-contract=off
#endif
#pragma clang fp contract(off)

namespace {
struct CompactTensor {
    void* data;
    int64_t stride[2];
};
struct Call {
    CompactTensor tensor[18];
    int rows, heads, keys, sequences, bound, group, anchor, act;
    float scale, eps, l2eps, threshold;
    int64_t cache_slots, cache_width, index_width;
};
static_assert(((sizeof(Call) + sizeof(bool) + 7) & ~size_t(7)) + 256 <= 768,
              "GDN convolution arguments plus implicit ABI must fit the AQL slot");
long long integer(const RadArgs* args, const char* name, long long fallback) {
    long long value = fallback;
    rad_args_geti(args, name, &value);
    return value;
}
float real(const RadArgs* args, const char* name, float fallback) {
    for (int index = 0; index < args->n_p; ++index)
        if (args->p[index].kind == RAD_P_F64 && !std::strcmp(args->p[index].key, name))
            return static_cast<float>(args->p[index].dval);
    return fallback;
}
bool matrix(const RadTensor& tensor, uint32_t type, int rows, int columns) {
    return tensor.data && tensor.dtype == type && tensor.rank == 2
        && tensor.shape[0] == rows && tensor.shape[1] == columns
        && tensor.stride[1] == 1 && tensor.stride[0] >= columns
        && tensor.stride[0] <= (1ll << 36);
}
bool cube(const RadTensor& tensor, uint32_t type, int rows, int heads, int width) {
    return tensor.data && tensor.dtype == type && tensor.rank == 3
        && tensor.shape[0] == rows && tensor.shape[1] == heads && tensor.shape[2] == width
        && tensor.stride[2] == 1 && tensor.stride[1] >= width
        && tensor.stride[0] >= heads * tensor.stride[1] && tensor.stride[0] <= (1ll << 36);
}
bool vector(const RadTensor& tensor, uint32_t type, int length) {
    return tensor.data && tensor.dtype == type && tensor.rank == 1
        && tensor.shape[0] == length && tensor.stride[0] == 1;
}
bool index_rows(const RadTensor& tensor, int rows, int columns) {
    return tensor.data && tensor.dtype == RAD_I32
        && ((tensor.rank == 1 && columns == 1 && tensor.shape[0] == rows && tensor.stride[0] == 1)
            || (tensor.rank == 2 && tensor.shape[0] == rows && tensor.shape[1] >= columns
                && tensor.stride[1] == 1 && tensor.stride[0] >= tensor.shape[1]));
}
bool state(const RadTensor& tensor, int heads) {
    return tensor.data && tensor.dtype == RAD_F32 && tensor.rank == 4 && tensor.shape[0] > 0
        && tensor.shape[1] == heads && tensor.shape[2] == 128 && tensor.shape[3] == 128
        && tensor.stride[3] == 1 && tensor.stride[2] == 128 && tensor.stride[1] >= 16384
        && tensor.stride[0] >= heads * tensor.stride[1] && tensor.stride[0] <= (1ll << 36);
}
bool geometry(int keys, int heads) {
    return (keys == 8 && heads == 24) || (keys == 16 && heads == 48);
}
int prepare(const RadArgs* args, int operation, Call* call) {
    if (!args || !args->t || !args->p || args->n_p < 0 || args->n_t > 18) return RAD_E_INVAL;
    const int count = operation == 0 ? 10 : operation == 1 ? 16 : operation == 2 ? 14
        : operation == 3 ? 18 : 3;
    if (args->n_t < count) return RAD_E_INVAL;
    *call = Call{};
    for (int index = 0; index < args->n_t; ++index) {
        call->tensor[index].data = args->t[index].data;
        call->tensor[index].stride[0] = args->t[index].stride[0];
        call->tensor[index].stride[1] = args->t[index].stride[1];
    }
    call->group = integer(args, "group", 128);
    if (call->group != 128 && call->group != 256) return RAD_E_UNSUPPORTED;
    if (operation == 4) {
        const RadTensor& input = args->t[0];
        if (input.rank != 3 || input.shape[2] != 128) return RAD_E_SHAPE;
        call->rows = input.shape[0]; call->heads = input.shape[1];
        if (!geometry(call->heads / 3, call->heads)) return RAD_E_SHAPE;
        const int width = call->heads * 128;
        if (!cube(input, RAD_BF16, call->rows, call->heads, 128)
            || input.stride[1] != 128
            || !matrix(args->t[1], RAD_I8, call->rows, width)
            || !matrix(args->t[2], RAD_F32, call->rows, width / call->group)) return RAD_E_SHAPE;
        return call->rows > 0 && call->rows <= 2048 ? RAD_OK : RAD_E_SHAPE;
    }
    if (integer(args, "head_k", 0) != 128 || integer(args, "head_v", 0) != 128)
        return RAD_E_UNSUPPORTED;
    if (operation != 2 && integer(args, "conv_width", 0) != 4) return RAD_E_UNSUPPORTED;
    const int output_index = operation == 0 ? 9 : operation == 1 ? 13 : operation == 2 ? 13 : 15;
    const RadTensor& output = args->t[output_index];
    if (output.rank != 3 || output.shape[2] != 128) return RAD_E_SHAPE;
    call->rows = output.shape[0]; call->heads = output.shape[1]; call->keys = call->heads / 3;
    if (!geometry(call->keys, call->heads) || call->rows <= 0 || call->rows > 2048)
        return RAD_E_SHAPE;
    if (integer(args, "n_head_v", call->heads) != call->heads) return RAD_E_SHAPE;
    if (!cube(output, RAD_BF16, call->rows, call->heads, 128)) return RAD_E_SHAPE;
    const int cu_index = operation == 2 ? 10 : operation == 3 ? 12 : 6;
    const RadTensor& cumulative = args->t[cu_index];
    if (cumulative.rank != 1 || cumulative.shape[0] < 2 || cumulative.shape[0] > 65)
        return RAD_E_SHAPE;
    call->sequences = cumulative.shape[0] - 1;
    if (!vector(cumulative, RAD_I32, call->sequences + 1)) return RAD_E_DTYPE;
    call->bound = integer(args, operation == 1 ? "M" : "q_len", operation == 1 ? call->rows : 0);
    if (call->bound <= 0 || call->bound > (operation == 1 ? 2048 : operation == 2 ? 16 : 8)
        || call->rows > call->bound * call->sequences) return RAD_E_SHAPE;
    if (call->bound > call->rows) call->bound = call->rows;
    call->scale = real(args, "scale", 1.0f);
    call->eps = real(args, "eps", 1e-6f);
    call->l2eps = real(args, "l2_eps", 1e-6f);
    call->threshold = real(args, "softplus_thr", 20.0f);
    const char* activation = rad_args_gets(args, "act");
    if (activation && std::strcmp(activation, "silu") && std::strcmp(activation, "sigmoid"))
        return RAD_E_UNSUPPORTED;
    call->act = activation && !std::strcmp(activation, "sigmoid");
    const char* form = rad_args_gets(args, "state_form");
    if (form && std::strcmp(form, "anchor") && std::strcmp(form, "per_candidate"))
        return RAD_E_UNSUPPORTED;
    call->anchor = form && !std::strcmp(form, "anchor");
    if (!std::isfinite(call->scale) || !std::isfinite(call->eps) || call->eps <= 0
        || !std::isfinite(call->l2eps) || call->l2eps <= 0 || !std::isfinite(call->threshold))
        return RAD_E_INVAL;
    if (operation == 0 || operation == 1 || operation == 3) {
        const int channels = (2 * call->keys + call->heads) * 128;
        const RadTensor& input = args->t[0];
        if (input.rank != 2 || input.shape[1] < channels
            || !matrix(input, RAD_BF16, call->rows, input.shape[1])
            || !matrix(args->t[1], RAD_BF16, channels, 4)) return RAD_E_SHAPE;
        if (args->t[2].data && !vector(args->t[2], RAD_BF16, channels)) return RAD_E_DTYPE;
        const int cache_index = operation == 1 ? 5 : 3;
        const RadTensor& cache = args->t[cache_index];
        call->cache_slots = cache.shape[0];
        call->cache_width = cache.shape[2];
        if (!cache.data || cache.dtype != RAD_BF16 || cache.rank != 3 || cache.shape[0] < 1
            || cache.shape[1] != channels || cache.shape[2] < 3
            || cache.shape[2] > 2050 || cache.stride[2] != 1 || cache.stride[1] < cache.shape[2]
            || cache.stride[0] < channels * cache.stride[1]) return RAD_E_SHAPE;
        if (!index_rows(args->t[operation == 1 ? 9 : 4], call->sequences, 1)) return RAD_E_SHAPE;
        if (operation == 0 && call->bound > call->cache_width - 2)
            call->bound = call->cache_width - 2;
        if (operation != 1 && args->t[operation == 3 ? 11 : 5].data
            && !vector(args->t[operation == 3 ? 11 : 5], RAD_I32, call->sequences))
            return RAD_E_SHAPE;
        if (operation != 3) {
            const int first = operation == 1 ? 11 : 7;
            if (!cube(args->t[first], RAD_BF16, call->rows, call->keys, 128)
                || !cube(args->t[first + 1], RAD_BF16, call->rows, call->keys, 128))
                return RAD_E_SHAPE;
        }
        if (operation == 1) {
            if (integer(args, "chunk", 0) != 64) return RAD_E_UNSUPPORTED;
            if (!vector(args->t[3], RAD_F32, call->heads) || !vector(args->t[4], RAD_F32, call->heads)
                || !matrix(args->t[7], RAD_BF16, call->rows, call->heads)
                || !matrix(args->t[8], RAD_BF16, call->rows, call->heads)
                || !matrix(args->t[14], RAD_F32, call->rows, call->heads)
                || !matrix(args->t[15], RAD_F32, call->rows, call->heads)
                || (args->t[10].data && !vector(args->t[10], RAD_I32, call->sequences)))
                return RAD_E_SHAPE;
        }
    }
    if (operation == 2 || operation == 3) {
        const int shift = operation == 3 ? 2 : 0;
        call->cache_slots = args->t[7 + shift].shape[0];
        const RadTensor& indices = args->t[8 + shift];
        call->index_width = indices.rank == 2 ? indices.shape[1] : 1;
        if (!state(args->t[7 + shift], call->heads)
            || !index_rows(args->t[8 + shift], call->sequences, call->anchor ? 6 : 1)
            || (args->t[9 + shift].data && !vector(args->t[9 + shift], RAD_I32, call->sequences))
            || !matrix(args->t[3 + shift], RAD_BF16, call->rows, call->heads)
            || !matrix(args->t[4 + shift], RAD_BF16, call->rows, call->heads)
            || !vector(args->t[5 + shift], RAD_F32, call->heads)
            || !vector(args->t[6 + shift], RAD_F32, call->heads)
            || !vector(args->t[12 + shift], RAD_F32, 128)) return RAD_E_SHAPE;
        if (!call->anchor && call->bound > call->index_width) call->bound = call->index_width;
        const RadTensor& gate = args->t[11 + shift];
        if (!(matrix(gate, RAD_BF16, call->rows, call->heads * 128)
            || cube(gate, RAD_BF16, call->rows, call->heads, 128))) return RAD_E_SHAPE;
        if (gate.rank == 2) call->tensor[11 + shift].stride[1] = 128;
        if (operation == 2) {
            if (args->n_t != 14 || !cube(args->t[0], RAD_BF16, call->rows, call->keys, 128)
                || !cube(args->t[1], RAD_BF16, call->rows, call->keys, 128)
                || !cube(args->t[2], RAD_BF16, call->rows, call->heads, 128)) return RAD_E_SHAPE;
        } else if (!matrix(args->t[16], RAD_I8, call->rows, call->heads * 128)
            || !matrix(args->t[17], RAD_F32, call->rows, call->heads * 128 / call->group))
            return RAD_E_SHAPE;
    }
    if (operation == 3 && call->cache_width < call->bound + 2) return RAD_E_SHAPE;
    return call->rows <= call->bound * call->sequences ? RAD_OK : RAD_E_SHAPE;
}
__device__ float widen(uint16_t value) { return __uint_as_float(static_cast<uint32_t>(value) << 16); }
__device__ uint16_t narrow(float value) {
    const uint32_t bits = __float_as_uint(value);
    if ((bits & 0x7fffffffu) > 0x7f800000u) return (bits >> 16) | 64u;
    return (bits + 32767u + ((bits >> 16) & 1u)) >> 16;
}
__device__ int64_t offset3(const CompactTensor& tensor, int row, int head, int column) {
    return row * tensor.stride[0] + head * tensor.stride[1] + column;
}
__device__ float read3(const CompactTensor& tensor, int row, int head, int column) {
    return widen(static_cast<const uint16_t*>(tensor.data)[offset3(tensor, row, head, column)]);
}
__device__ void write3(const CompactTensor& tensor, int row, int head, int column, float value) {
    static_cast<uint16_t*>(tensor.data)[offset3(tensor, row, head, column)] = narrow(value);
}
__device__ float exponential(float value) { return static_cast<float>(exp(static_cast<double>(value))); }
__device__ float sigmoid(float value) { return 1.0f / (1.0f + exponential(-value)); }
__device__ float softplus(float value, float threshold) {
    if (value > threshold) return value;
    const float inner = exponential(value > 0 ? -value : value);
    const float result = static_cast<float>(log1p(static_cast<double>(inner)));
    return value > 0 ? value + result : result;
}
__global__ void convolution(Call call, bool prefill) {
    const int channel = blockIdx.x * blockDim.x + threadIdx.x;
    const int sequence = blockIdx.y;
    const int channels = (2 * call.keys + call.heads) * 128;
    if (channel >= channels) return;
    const auto* tensor = call.tensor;
    const auto* cumulative = static_cast<const int*>(tensor[6].data);
    const int begin = cumulative[sequence], end = cumulative[sequence + 1];
    if (begin < 0 || end < begin || end > call.rows || end - begin > call.bound) return;
    const auto& cache = tensor[prefill ? 5 : 3];
    const auto& indices = tensor[prefill ? 9 : 4];
    const int slot = static_cast<const int*>(indices.data)[sequence * indices.stride[0]];
    if (slot < 0 || slot >= call.cache_slots || begin == end) return;
    const int accepted = prefill || !tensor[5].data ? 1 : static_cast<const int*>(tensor[5].data)[sequence];
    const int history_start = accepted - 1;
    if (history_start < 0 || history_start + 3 > call.cache_width
        || (!prefill && end - begin + 2 > call.cache_width)) return;
    const bool warm = !prefill || !tensor[10].data || static_cast<const int*>(tensor[10].data)[sequence] != 0;
    const int64_t cache_base = slot * cache.stride[0] + channel * cache.stride[1];
    float history[3];
    for (int index = 0; index < 3; ++index)
        history[index] = warm ? widen(static_cast<const uint16_t*>(cache.data)[cache_base + history_start + index]) : 0;
    const int first = prefill ? 11 : 7;
    const int output_kind = channel < call.keys * 128 ? 0 : channel < call.keys * 256 ? 1 : 2;
    const int local = channel - (output_kind == 1 ? call.keys * 128 : output_kind == 2 ? call.keys * 256 : 0);
    for (int token = begin; token < end; ++token) {
        float accumulator = tensor[2].data ? widen(static_cast<const uint16_t*>(tensor[2].data)[channel]) : 0;
        for (int tap = 0; tap < 4; ++tap) {
            const int source = token - begin - 3 + tap;
            const float value = source < 0 ? history[source + 3]
                : widen(static_cast<const uint16_t*>(tensor[0].data)[(begin + source) * tensor[0].stride[0] + channel]);
            accumulator = accumulator + value * widen(static_cast<const uint16_t*>(tensor[1].data)[channel * tensor[1].stride[0] + tap]);
        }
        write3(tensor[first + output_kind], token, local / 128, local % 128, accumulator * sigmoid(accumulator));
    }
    auto* destination = static_cast<uint16_t*>(cache.data);
    if (prefill) {
        uint16_t keep[3];
        for (int index = 0; index < 3; ++index) {
            const int source = end - begin - 3 + index;
            keep[index] = source >= 0 ? static_cast<const uint16_t*>(tensor[0].data)[(begin + source) * tensor[0].stride[0] + channel]
                : narrow(history[source + 3]);
        }
        for (int index = 0; index < 3; ++index) destination[cache_base + index] = keep[index];
    } else {
        destination[cache_base] = narrow(history[1]); destination[cache_base + 1] = narrow(history[2]);
        for (int token = begin; token < end; ++token)
            destination[cache_base + token - begin + 2] = static_cast<const uint16_t*>(tensor[0].data)[token * tensor[0].stride[0] + channel];
    }
}
__global__ void prepare_prefill(Call call) {
    const int sequence = blockIdx.x, head = blockIdx.y;
    const auto* tensor = call.tensor;
    const auto* cumulative = static_cast<const int*>(tensor[6].data);
    const int begin = cumulative[sequence], end = cumulative[sequence + 1];
    if (begin < 0 || end < begin || end > call.rows || end - begin > call.bound) return;
    if (threadIdx.x == 0 && head < call.keys)
        for (int token = begin; token < end; ++token)
            for (int kind = 11; kind <= 12; ++kind) {
                float sum = 0;
                for (int column = 0; column < 128; ++column) {
                    const float value = read3(tensor[kind], token, head, column);
                    sum = sum + value * value;
                }
                const float inverse = 1.0f / sqrtf(sum + call.l2eps);
                for (int column = 0; column < 128; ++column)
                    write3(tensor[kind], token, head, column, read3(tensor[kind], token, head, column) * inverse);
            }
    if (threadIdx.x == 1) {
        float running = 0;
        for (int token = begin; token < end; ++token) {
            if ((token - begin) % 64 == 0) running = 0;
            const float decay_input = widen(static_cast<const uint16_t*>(tensor[7].data)[token * tensor[7].stride[0] + head]);
            const float beta_input = widen(static_cast<const uint16_t*>(tensor[8].data)[token * tensor[8].stride[0] + head]);
            const float decay = -exponential(static_cast<const float*>(tensor[3].data)[head])
                * softplus(decay_input + static_cast<const float*>(tensor[4].data)[head], call.threshold);
            running = running + decay;
            static_cast<float*>(tensor[14].data)[token * tensor[14].stride[0] + head] = running;
            static_cast<float*>(tensor[15].data)[token * tensor[15].stride[0] + head] = sigmoid(beta_input);
        }
    }
}
__global__ __launch_bounds__(128) void recurrent(Call call) {
    const int sequence = blockIdx.x, head = blockIdx.y, value_row = threadIdx.x;
    const auto* tensor = call.tensor;
    const auto* cumulative = static_cast<const int*>(tensor[10].data);
    const int begin = cumulative[sequence], end = cumulative[sequence + 1];
    if (begin < 0 || end < begin || end > call.rows || end - begin > call.bound) return;
    const auto& indices = tensor[8]; const auto& cache = tensor[7];
    const auto* index = static_cast<const int*>(indices.data) + sequence * indices.stride[0];
    const int accepted = tensor[9].data ? static_cast<const int*>(tensor[9].data)[sequence] : 1;
    const int64_t width = call.index_width;
    if (accepted < (call.anchor ? 0 : 1) || accepted > (call.anchor ? 16 : width)) return;
    for (int64_t column = 0; column < (call.anchor ? 6 : width); ++column)
        if (index[column] < 0 || index[column] >= call.cache_slots) return;
    auto* pool = static_cast<float*>(cache.data);
    const int64_t head_base = head * cache.stride[1];
    constexpr int factor_stride = 648, factor_mark = 16 * factor_stride;
    int source_slot = index[accepted > 0 ? accepted - 1 : 0];
    if (call.anchor) {
        const int mark = static_cast<int>(pool[index[4] * cache.stride[0] + head_base + factor_mark] + 0.5f) - 1;
        source_slot = index[3 - ((mark >= 0 && mark <= 2) ? mark : 1)];
    }
    float state_row[128];
    for (int column = 0; column < 128; ++column)
        state_row[column] = pool[source_slot * cache.stride[0] + head_base + value_row * 128 + column];
    if (call.anchor) {
        for (int token = 0; token < accepted; ++token) {
            const float* factor = pool + index[4] * cache.stride[0] + head_base + token * factor_stride;
            const int group = value_row / 32;
            const float decay = exponential(factor[640 + 2 * group]);
            float dot = 0;
            for (int column = 0; column < 128; ++column) {
                state_row[column] = state_row[column] * decay;
                dot = dot + state_row[column] * factor[group * 128 + column];
            }
            const float delta = factor[641 + 2 * group] * (factor[512 + value_row] - dot);
            for (int column = 0; column < 128; ++column)
                state_row[column] = state_row[column] + delta * factor[group * 128 + column];
        }
        for (int column = 0; column < 128; ++column)
            pool[index[2] * cache.stride[0] + head_base + value_row * 128 + column] = state_row[column];
    }
    __shared__ float query[128], key[128], result[128], norm, decay, beta, gate_value;
    for (int token = begin; token < end; ++token) {
        if (value_row == 0) {
            float query_sum = 0, key_sum = 0;
            for (int column = 0; column < 128; ++column) {
                const float query_value = read3(tensor[0], token, head / 3, column);
                const float key_value = read3(tensor[1], token, head / 3, column);
                query_sum = query_sum + query_value * query_value;
                key_sum = key_sum + key_value * key_value;
            }
            const float query_scale = 1.0f / sqrtf(query_sum + call.l2eps) * call.scale;
            const float key_scale = 1.0f / sqrtf(key_sum + call.l2eps);
            for (int column = 0; column < 128; ++column) {
                query[column] = read3(tensor[0], token, head / 3, column) * query_scale;
                key[column] = read3(tensor[1], token, head / 3, column) * key_scale;
            }
            const float avalue = widen(static_cast<const uint16_t*>(tensor[3].data)[token * tensor[3].stride[0] + head]);
            const float bvalue = widen(static_cast<const uint16_t*>(tensor[4].data)[token * tensor[4].stride[0] + head]);
            gate_value = -exponential(static_cast<const float*>(tensor[5].data)[head])
                * softplus(avalue + static_cast<const float*>(tensor[6].data)[head], call.threshold);
            decay = exponential(gate_value); beta = sigmoid(bvalue);
        }
        __syncthreads();
        float dot = 0;
        for (int column = 0; column < 128; ++column) {
            state_row[column] = state_row[column] * decay;
            dot = dot + state_row[column] * key[column];
        }
        const float delta = beta * (read3(tensor[2], token, head, value_row) - dot);
        float output = 0;
        for (int column = 0; column < 128; ++column) {
            state_row[column] = state_row[column] + delta * key[column];
            output = output + state_row[column] * query[column];
        }
        result[value_row] = output;
        if (!call.anchor)
            for (int column = 0; column < 128; ++column)
                pool[index[token - begin] * cache.stride[0] + head_base + value_row * 128 + column] = state_row[column];
        __syncthreads();
        if (value_row == 0) {
            float sum = 0;
            for (int column = 0; column < 128; ++column) sum = sum + result[column] * result[column];
            norm = 1.0f / sqrtf(sum / 128.0f + call.eps);
        }
        if (call.anchor) {
            for (int copy = 0; copy < 3; ++copy) {
                if (copy != 1 && index[3] == index[4] && index[5] == index[4]) continue;
                float* factor = pool + index[3 + copy] * cache.stride[0] + head_base + (token - begin) * factor_stride;
                for (int group = 0; group < 4; ++group) factor[group * 128 + value_row] = key[value_row];
                factor[512 + value_row] = read3(tensor[2], token, head, value_row);
                if (value_row < 4) { factor[640 + 2 * value_row] = gate_value; factor[641 + 2 * value_row] = beta; }
            }
        }
        __syncthreads();
        const float zvalue = read3(tensor[11], token, head, value_row);
        float value = result[value_row] * norm;
        value = value * static_cast<const float*>(tensor[12].data)[value_row];
        value = value * (call.act ? sigmoid(zvalue) : zvalue * sigmoid(zvalue));
        write3(tensor[13], token, head, value_row, value);
        __syncthreads();
    }
    if (call.anchor && value_row == 0)
        for (int copy = 0; copy < 3; ++copy)
            pool[index[3 + copy] * cache.stride[0] + head_base + factor_mark] = static_cast<float>(copy + 1);
}
__global__ void quantize(Call call) {
    const int token = blockIdx.x, group = blockIdx.y;
    if (threadIdx.x != 0) return;
    const auto* tensor = call.tensor;
    float values[256];
    for (int column = 0; column < call.group; ++column) {
        const int source = group * call.group + column;
        values[column] = read3(tensor[0], token, source / 128, source % 128);
    }
    for (int block = 0; block < call.group; block += 128)
        for (int stride = 1; stride < 128; stride *= 2)
            for (int base = 0; base < 128; base += 2 * stride)
                for (int column = base; column < base + stride; ++column) {
                    const float lower = values[block + column], upper = values[block + column + stride];
                    values[block + column] = lower + upper;
                    values[block + column + stride] = lower - upper;
                }
    float maximum = 0;
    for (int column = 0; column < call.group; ++column) maximum = fmaxf(maximum, fabsf(values[column]));
    const float scale = maximum > 0 ? maximum * (1.0f / 127.0f) : 1.0f;
    const float inverse = 1.0f / scale;
    for (int column = 0; column < call.group; ++column) {
        const float value = fminf(127.0f, fmaxf(-127.0f, values[column] * inverse));
        static_cast<int8_t*>(tensor[1].data)[token * tensor[1].stride[0] + group * call.group + column]
            = static_cast<int8_t>(__float2int_rn(value));
    }
    static_cast<float*>(tensor[2].data)[token * tensor[2].stride[0] + group] = scale;
}
int launch_status() { return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE; }
RadTensor dense3(void* pointer, int rows, int heads) {
    RadTensor result{}; result.data = pointer; result.dtype = RAD_BF16; result.rank = 3;
    result.shape[0] = rows; result.shape[1] = heads; result.shape[2] = 128;
    result.stride[0] = heads * 128; result.stride[1] = 128; result.stride[2] = 1;
    return result;
}
}
extern "C" int r3dh_gdn_extra_validate(const RadArgs* args, int operation) {
    if (operation < 0 || operation > 4) return RAD_E_INVAL;
    Call call{}; return prepare(args, operation, &call);
}
extern "C" void r3dh_gdn_extra_gdn_dims(int* key, int* value, int* chunk) {
    if (key) *key = 128; if (value) *value = 128; if (chunk) *chunk = 64;
}
extern "C" int r3dh_gdn_extra_conv_update_strict(const RadArgs* args, RadStream stream) {
    Call call{}; const int status = prepare(args, 0, &call); if (status != RAD_OK) return status;
    hipLaunchKernelGGL(convolution, dim3(((2 * call.keys + call.heads) * 128 + 127) / 128, call.sequences), dim3(128), 0,
        reinterpret_cast<hipStream_t>(stream), call, false);
    return launch_status();
}
extern "C" int r3dh_gdn_extra_conv_prep_strict(const RadArgs* args, RadStream stream) {
    Call call{}; const int status = prepare(args, 1, &call); if (status != RAD_OK) return status;
    hipLaunchKernelGGL(convolution, dim3(((2 * call.keys + call.heads) * 128 + 127) / 128, call.sequences), dim3(128), 0,
        reinterpret_cast<hipStream_t>(stream), call, true);
    int result = launch_status(); if (result != RAD_OK) return result;
    hipLaunchKernelGGL(prepare_prefill, dim3(call.sequences, call.heads), dim3(32), 0,
        reinterpret_cast<hipStream_t>(stream), call);
    return launch_status();
}
extern "C" int r3dh_gdn_extra_recurrent_strict(const RadArgs* args, RadStream stream) {
    Call call{}; const int status = prepare(args, 2, &call); if (status != RAD_OK) return status;
    hipLaunchKernelGGL(recurrent, dim3(call.sequences, call.heads), dim3(128), 0,
        reinterpret_cast<hipStream_t>(stream), call);
    return launch_status();
}
extern "C" int r3dh_gdn_extra_had_quant_strict(const RadArgs* args, RadStream stream) {
    Call call{}; const int status = prepare(args, 4, &call); if (status != RAD_OK) return status;
    hipLaunchKernelGGL(quantize, dim3(call.rows, call.heads * 128 / call.group), dim3(32), 0,
        reinterpret_cast<hipStream_t>(stream), call);
    return launch_status();
}
extern "C" int64_t r3dh_gdn_extra_scratch(const RadArgs* args) {
    if (args && !args->t && args->n_t == 0) {
        const int64_t heads = integer(args, "n_head_v", 0);
        if ((heads != 24 && heads != 48) || integer(args, "head_k", 0) != 128
            || integer(args, "head_v", 0) != 128 || integer(args, "conv_width", 0) != 4
            || integer(args, "q_len", 0) < 1 || integer(args, "q_len", 0) > 8)
            return RAD_E_SHAPE;
        return int64_t(2048) * (2 * (heads / 3) + heads) * 256;
    }
    if (!args || !args->t || args->n_t != 18) return RAD_E_INVAL;
    const RadTensor& output = args->t[15];
    if (output.rank != 3 || output.dtype != RAD_BF16 || output.shape[2] != 128
        || output.shape[0] < 1 || output.shape[0] > 2048) return RAD_E_SHAPE;
    const int64_t heads = output.shape[1];
    if ((heads != 24 && heads != 48)
        || integer(args, "head_k", 0) != 128 || integer(args, "head_v", 0) != 128
        || integer(args, "conv_width", 0) != 4
        || integer(args, "n_head_v", heads) != heads) return RAD_E_SHAPE;
    const int64_t keys = heads / 3;
    return (output.shape[0] * (2 * keys + heads) * 256 + 255) & ~255ll;
}
extern "C" int r3dh_gdn_extra_conv_recurrent_update_had_strict(const RadArgs* args, RadStream stream) {
    Call call{}; int status = prepare(args, 3, &call); if (status != RAD_OK) return status;
    const int64_t needed = r3dh_gdn_extra_scratch(args);
    if (!args->scratch || args->scratch_bytes < needed || reinterpret_cast<uintptr_t>(args->scratch) % 16)
        return RAD_E_INVAL;
    auto* scratch = static_cast<uint16_t*>(args->scratch);
    const RadTensor query = dense3(scratch, call.rows, call.keys);
    const RadTensor key = dense3(scratch + call.rows * call.keys * 128, call.rows, call.keys);
    const RadTensor value = dense3(scratch + call.rows * call.keys * 256, call.rows, call.heads);
    const RadTensor* source = args->t;
    RadTensor convolution_tensors[10] = {source[0], source[1], source[2], source[3], source[4], source[11], source[12], query, key, value};
    RadTensor recurrent_tensors[14] = {query, key, value, source[5], source[6], source[7], source[8], source[9], source[10], source[11], source[12], source[13], source[14], source[15]};
    RadTensor quantizer_tensors[3] = {source[15], source[16], source[17]};
    RadArgs convolution_args = *args, recurrent_args = *args, quantizer_args = *args;
    convolution_args.t = convolution_tensors; convolution_args.n_t = 10;
    recurrent_args.t = recurrent_tensors; recurrent_args.n_t = 14;
    quantizer_args.t = quantizer_tensors; quantizer_args.n_t = 3;
    if ((status = r3dh_gdn_extra_validate(&convolution_args, 0)) != RAD_OK
        || (status = r3dh_gdn_extra_validate(&recurrent_args, 2)) != RAD_OK
        || (status = r3dh_gdn_extra_validate(&quantizer_args, 4)) != RAD_OK) return status;
    if ((status = r3dh_gdn_extra_conv_update_strict(&convolution_args, stream)) != RAD_OK) return status;
    if ((status = r3dh_gdn_extra_recurrent_strict(&recurrent_args, stream)) != RAD_OK) return status;
    return r3dh_gdn_extra_had_quant_strict(&quantizer_args, stream);
}
