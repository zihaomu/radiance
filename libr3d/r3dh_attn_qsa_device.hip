#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "r3dh_attn_qsa.h"
#include <cmath>
#include <cstring>
#if defined(__FAST_MATH__)
#error r3dh_attn_qsa requires strict floating point and -ffp-contract=off
#endif
namespace {
long integer(const RadArgs* args, const char* key, long fallback) {
    long long value = fallback;
    rad_args_geti(args, key, &value);
    return value;
}
float real(const RadArgs* args, const char* key, float fallback) {
    for (int index = 0; index < args->n_p; ++index)
        if (args->p[index].kind == RAD_P_F64 && !std::strcmp(args->p[index].key, key))
            return float(args->p[index].dval);
    return fallback;
}
bool named(const RadArgs* args, const char* key, const char* expected) {
    const char* value = rad_args_gets(args, key);
    return !value || !std::strcmp(value, expected);
}
bool vector(const RadTensor& tensor, uint32_t type, long size) {
    return tensor.data && tensor.dtype == type && tensor.rank == 1
        && tensor.shape[0] == size && tensor.stride[0] == 1;
}
bool matrix(const RadTensor& tensor, uint32_t type, long rows, long width) {
    if (!tensor.data || tensor.dtype != type || tensor.shape[0] != rows) return false;
    if (tensor.rank == 2) return tensor.shape[1] == width && tensor.stride[1] == 1
        && tensor.stride[0] >= width && tensor.stride[0] <= (1ll << 30);
    return tensor.rank == 3 && tensor.shape[1] * tensor.shape[2] == width
        && tensor.stride[2] == 1 && tensor.stride[1] == tensor.shape[2]
        && tensor.stride[0] >= width && tensor.stride[0] <= (1ll << 30);
}
bool cache(const RadTensor& tensor) {
    return tensor.data && tensor.dtype == RAD_I8 && tensor.rank == 4
        && tensor.shape[0] > 0 && tensor.shape[0] <= r3dh_attn_qsa_max_cache_pages && tensor.shape[1] == 1
        && tensor.shape[2] == 4 && tensor.shape[3] == 516 && tensor.stride[3] == 1
        && tensor.stride[2] == 516 && tensor.stride[1] == 2064 && tensor.stride[0] == 2064;
}
__device__ float load(const uint16_t* pointer, long offset) {
    return __uint_as_float(uint32_t(pointer[offset]) << 16);
}
__device__ uint16_t rounded(float value) {
    const uint32_t word = __float_as_uint(value);
    if ((word & 0x7fffffffu) > 0x7f800000u) return uint16_t((word >> 16) | 64u);
    return uint16_t((word + 0x7fffu + ((word >> 16) & 1u)) >> 16);
}
struct Prep {
    const uint16_t *qg, *key;
    const float *table, *wq, *wk;
    const int* positions;
    uint16_t *query, *outkey, *gate;
    long qpitch, kpitch, opitch, okpitch, gpitch, table_rows;
    int rows, heads, keyheads, dim;
    float eps, wadd;
};
__global__ void prepare(Prep call) {
    const int row = blockIdx.x, head = blockIdx.y;
    const bool key = head >= call.heads;
    const int local = key ? head - call.heads : head;
    const uint16_t* input = key ? call.key + row * call.kpitch + local * call.dim
        : call.qg + row * call.qpitch + local * 2 * call.dim;
    const float* weight = key ? call.wk : call.wq;
    uint16_t* output = key ? call.outkey + row * call.okpitch + local * call.dim
        : call.query + row * call.opitch + local * call.dim;
    float values[256];
    uint16_t gates[256];
    float sum = 0;
    for (int column = 0; column < call.dim; ++column) {
        values[column] = load(input, column);
        sum = sum + values[column] * values[column];
        if (!key && call.gate) gates[column] = input[call.dim + column];
    }
    const float inverse = 1.0f / sqrtf(sum / float(call.dim) + call.eps);
    for (int column = 0; column < call.dim; ++column)
        values[column] = __uint_as_float(uint32_t(rounded((values[column] * inverse)
            * (weight[column] + call.wadd))) << 16);
    const int position = call.positions[row];
    for (int column = 0; column < call.dim; ++column) {
        float result = values[column];
        if (column < 64 && position >= 0 && position < call.table_rows) {
            const int pair = column % 32;
            const float cosine = call.table[long(position) * 64 + pair];
            const float sine = call.table[long(position) * 64 + 32 + pair];
            result = column < 32 ? values[column] * cosine - values[column + 32] * sine
                : values[column - 32] * sine + values[column] * cosine;
        }
        output[column] = rounded(result);
        if (!key && call.gate) call.gate[row * call.gpitch + local * call.dim + column] = gates[column];
    }
}
struct Rope {
    uint16_t* data; const int* positions;
    long pitch, plane; int heads, dim; bool multi;
    float theta, scale;
};
__global__ void rope(Rope call) {
    uint16_t* values = call.data + blockIdx.x * call.pitch + blockIdx.y * call.dim;
    const int pair = threadIdx.x;
    const int component = pair % 3 == 1 ? 1 : (pair % 3 == 2 && pair < 30 ? 2 : 0);
    const int position = call.positions[blockIdx.x + (call.multi ? component * call.plane : 0)];
    if (position < 0) return;
    const float angle = (float(position) / call.scale)
        * (1.0f / float(pow(double(call.theta), double(float(2 * pair) / 64.0f))));
    const float cosine = float(cos(double(angle))), sine = float(sin(double(angle)));
    const float first = load(values, pair), second = load(values, pair + 32);
    values[pair] = rounded(first * cosine - second * sine);
    values[pair + 32] = rounded(first * sine + second * cosine);
}
__global__ void store(const uint16_t* key, const uint16_t* value, const int* slots,
                       int8_t* output, long kpitch, long vpitch, long blocks) {
    const int token = blockIdx.x, half = blockIdx.y;
    const int slot = slots[token];
    if (slot < 0 || slot >= blocks * 4) return;
    const uint16_t* input = half ? value + token * vpitch : key + token * kpitch;
    int8_t* row = output + long(slot) * 516;
    float maximum = 0;
    for (int column = 0; column < 256; ++column) maximum = fmaxf(maximum, fabsf(load(input, column)));
    const __half encoded = __float2half_rn(fminf(maximum / 127.0f, 515.0f));
    const float scale = __half2float(encoded);
    for (int column = 0; column < 256; ++column) {
        const float quantized = scale > 0 ? nearbyintf(load(input, column) / scale) : 0;
        row[half * 256 + column] = int8_t(fminf(fmaxf(quantized, -127.0f), 127.0f));
    }
    reinterpret_cast<__half*>(row + 512)[half] = encoded;
}
struct Attention {
    const uint16_t *query, *key, *value;
    const int8_t* kv;
    const int *cu, *used, *table;
    uint16_t* output;
    long qpitch, kpitch, vpitch, opitch, table_pitch, pages;
    int rows, seqs, table_width;
    float scale;
};
__device__ float cached(const Attention& call, int seq, int token, int half, int column) {
    const int page = call.table[seq * call.table_pitch + token / 4];
    if (page < 0 || page >= call.pages) return 0;
    const int8_t* row = call.kv + (long(page) * 4 + token % 4) * 516;
    return float(row[half * 256 + column]) * __half2float(reinterpret_cast<const __half*>(row + 512)[half]);
}
template<bool PAGED> __global__ void attention(Attention call) {
    const int row = blockIdx.x, head = blockIdx.y;
    int sequence = -1, begin = 0, end = 0;
    for (int candidate = 0; candidate < call.seqs; ++candidate) {
        const int lower = call.cu ? call.cu[candidate] : candidate * (call.rows / call.seqs);
        const int upper = call.cu ? call.cu[candidate + 1] : (candidate + 1) * (call.rows / call.seqs);
        if (lower >= 0 && upper <= call.rows && row >= lower && row < upper) {
            sequence = candidate; begin = lower; end = upper; break;
        }
    }
    int keys = row - begin + 1;
    if (PAGED && sequence >= 0) keys = call.used[sequence] - (end - begin) + row - begin + 1;
    if (sequence < 0 || keys < 0 || (PAGED && (call.used[sequence] > call.table_width * 4
        || call.used[sequence] < end - begin))) keys = 0;
    float query[256], accumulator[256];
    for (int column = 0; column < 256; ++column) {
        query[column] = load(call.query, row * call.qpitch + head * 256 + column);
        accumulator[column] = 0;
    }
    float maximum = -INFINITY;
    for (int token = 0; token < keys; ++token) {
        float dot = 0;
        for (int column = 0; column < 256; ++column) {
            const float key = PAGED ? cached(call, sequence, token, 0, column)
                : load(call.key, (begin + token) * call.kpitch + column);
            dot = dot + query[column] * key;
        }
        maximum = fmaxf(maximum, dot * call.scale);
    }
    float denominator = 0;
    for (int token = 0; token < keys; ++token) {
        float dot = 0;
        for (int column = 0; column < 256; ++column) {
            const float key = PAGED ? cached(call, sequence, token, 0, column)
                : load(call.key, (begin + token) * call.kpitch + column);
            dot = dot + query[column] * key;
        }
        const float probability = float(exp(double(dot * call.scale - maximum)));
        denominator = denominator + probability;
        for (int column = 0; column < 256; ++column) {
            const float value = PAGED ? cached(call, sequence, token, 1, column)
                : load(call.value, (begin + token) * call.vpitch + column);
            accumulator[column] = accumulator[column] + probability * value;
        }
    }
    const float inverse = denominator > 0 ? 1.0f / denominator : 0;
    for (int column = 0; column < 256; ++column)
        call.output[row * call.opitch + head * 256 + column] = rounded(accumulator[column] * inverse);
}
int launched() { return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE; }
}
extern "C" int r3dh_attn_qsa_prepare(const RadArgs* args, RadStream stream) {
    if (!args || !args->t || args->n_t != 9) return RAD_E_INVAL;
    const int dim = integer(args, "head_dim", 0), heads = integer(args, "n_head", -1);
    const int keyheads = integer(args, "n_head_kv", 0);
    if (!((dim == 256 && heads == 12 && keyheads == 1) || (dim == 128 && heads == 0 && keyheads == 4))
        || integer(args, "rot", 0) != 64 || integer(args, "cs_f32", 0) != 1
        || !named(args, "q_dtype", "bf16")) return RAD_E_UNSUPPORTED;
    const RadTensor* tensors = args->t;
    const long rows = tensors[1].shape[0];
    const bool indexer_projection = heads == 0 && dim == 128 && keyheads == 4
        && tensors[1].rank == 2 && matrix(tensors[1], RAD_BF16, rows, 640);
    if (rows < 1 || rows > 2048
        || (!matrix(tensors[1], RAD_BF16, rows, keyheads * dim) && !indexer_projection)
        || !matrix(tensors[7], RAD_BF16, rows, keyheads * dim)
        || !vector(tensors[3], RAD_I32, rows) || !vector(tensors[5], RAD_F32, dim)
        || !tensors[2].data || tensors[2].dtype != RAD_F32 || tensors[2].rank != 2
        || tensors[2].shape[0] < 1 || tensors[2].shape[1] != 64 || tensors[2].stride[0] != 64
        || tensors[2].stride[1] != 1) return RAD_E_SHAPE;
    if (!heads && (tensors[0].data || tensors[4].data || tensors[6].data || tensors[8].data))
        return RAD_E_SHAPE;
    if (heads && (!matrix(tensors[0], RAD_BF16, rows, 2 * heads * dim)
        || !matrix(tensors[6], RAD_BF16, rows, heads * dim) || !vector(tensors[4], RAD_F32, dim)
        || (tensors[8].data && !matrix(tensors[8], RAD_BF16, rows, heads * dim)))) return RAD_E_SHAPE;
    const float eps = real(args, "eps", 1e-6f), wadd = real(args, "wadd", 0);
    if (!std::isfinite(eps) || eps <= 0 || !std::isfinite(wadd)) return RAD_E_INVAL;
    Prep call{static_cast<const uint16_t*>(tensors[0].data), static_cast<const uint16_t*>(tensors[1].data),
        static_cast<const float*>(tensors[2].data), static_cast<const float*>(tensors[4].data),
        static_cast<const float*>(tensors[5].data), static_cast<const int*>(tensors[3].data),
        static_cast<uint16_t*>(tensors[6].data), static_cast<uint16_t*>(tensors[7].data),
        static_cast<uint16_t*>(tensors[8].data), tensors[0].stride[0], tensors[1].stride[0],
        tensors[6].stride[0], tensors[7].stride[0], tensors[8].stride[0], tensors[2].shape[0],
        int(rows), heads, keyheads, dim, eps, wadd};
    hipLaunchKernelGGL(prepare, dim3(rows, heads + keyheads), dim3(1), 0,
        reinterpret_cast<hipStream_t>(stream), call);
    return launched();
}
extern "C" int r3dh_attn_qsa_rope(const RadArgs* args, RadStream stream) {
    if (!args || !args->t || args->n_t != 2) return RAD_E_INVAL;
    const int dim = integer(args, "head_dim", 0), heads = integer(args, "n_head", 0);
    const bool multi = !named(args, "mode", "neox");
    if (!((dim == 256 && (heads == 12 || heads == 1)) || (dim == 128 && heads == 4))
        || integer(args, "rotary_dim", 0) != 64 || integer(args, "n_head_kv", 0) != 0
        || (multi && (!named(args, "mode", "imrope") || !rad_args_gets(args, "sections")
            || !named(args, "sections", "11 11 10")))
        || !named(args, "ladder", "rotary") || real(args, "yarn_factor", 0) != 0
        || real(args, "llama3_factor", 0) != 0
        || !named(args, "dtype", "bf16")) return RAD_E_UNSUPPORTED;
    const long rows = args->t[0].shape[0];
    const float theta = real(args, "theta", 10000), scale = real(args, "scale", 1);
    const RadTensor& positions = args->t[1];
    const bool positions_ok = vector(positions, RAD_I32, rows) || (multi && positions.data
        && positions.dtype == RAD_I32 && positions.rank == 2 && positions.shape[0] == 3
        && positions.shape[1] == rows && positions.stride[1] == 1 && positions.stride[0] >= rows);
    if (rows < 1 || rows > 2048 || !matrix(args->t[0], RAD_BF16, rows, heads * dim)
        || !positions_ok) return RAD_E_SHAPE;
    if (!std::isfinite(theta) || theta <= 0 || !std::isfinite(scale) || scale <= 0) return RAD_E_INVAL;
    Rope call{static_cast<uint16_t*>(args->t[0].data), static_cast<const int*>(args->t[1].data),
        args->t[0].stride[0], positions.stride[0], heads, dim, positions.rank == 2, theta, scale};
    hipLaunchKernelGGL(rope, dim3(rows, heads), dim3(32), 0, reinterpret_cast<hipStream_t>(stream), call);
    return launched();
}
extern "C" int r3dh_attn_qsa_kv_store(const RadArgs* args, RadStream stream) {
    if (!args || !args->t || args->n_t != 4) return RAD_E_INVAL;
    if (integer(args, "head_dim", 0) != 256 || integer(args, "n_head_kv", 1) != 1
        || integer(args, "block_size", 0) != 4 || !named(args, "kv_dtype", "i8")) return RAD_E_UNSUPPORTED;
    const RadTensor* tensors = args->t;
    const long rows = tensors[0].shape[0];
    if (rows < 1 || rows > 2048 || !matrix(tensors[0], RAD_BF16, rows, 256)
        || !matrix(tensors[1], RAD_BF16, rows, 256) || !vector(tensors[2], RAD_I32, rows)
        || !cache(tensors[3])) return RAD_E_SHAPE;
    hipLaunchKernelGGL(store, dim3(rows, 2), dim3(1), 0, reinterpret_cast<hipStream_t>(stream),
        static_cast<const uint16_t*>(tensors[0].data), static_cast<const uint16_t*>(tensors[1].data),
        static_cast<const int*>(tensors[2].data), static_cast<int8_t*>(tensors[3].data),
        tensors[0].stride[0], tensors[1].stride[0], tensors[3].shape[0]);
    return launched();
}
template<bool PAGED> int launch_attention(const RadArgs* args, RadStream stream) {
    if (!args || !args->t || args->n_t != (PAGED ? 8 : 5)) return RAD_E_INVAL;
    if (integer(args, "head_dim", 0) != 256 || integer(args, "gqa", 0) != 12
        || integer(args, "causal", PAGED ? 1 : 0) != 1 || integer(args, "window", 0) != 0
        || integer(args, "block", 0) != 0 || integer(args, "cu_div", 1) != 1
        || integer(args, "window_right", 0) != 0 || !named(args, "q_dtype", "bf16")
        || !named(args, "dtype", "bf16") || integer(args, "n_head", 12) != 12
        || (PAGED && (integer(args, "block_size", 0) != 4 || !named(args, "kv_dtype", "i8"))))
        return RAD_E_UNSUPPORTED;
    const RadTensor* tensors = args->t;
    const RadTensor& output = tensors[PAGED ? 7 : 4];
    const RadTensor& cu = tensors[PAGED ? 6 : 3];
    const long rows = tensors[0].shape[0];
    if (rows < 1 || rows > 2048 || !matrix(tensors[0], RAD_BF16, rows, 3072)
        || !matrix(output, RAD_BF16, rows, 3072)) return RAD_E_SHAPE;
    const long sequences = PAGED ? tensors[3].shape[0] : cu.shape[0] - 1;
    if (sequences < 1 || sequences > 2048 || (cu.data && !vector(cu, RAD_I32, sequences + 1))
        || (!cu.data && (!PAGED || rows % sequences))) return RAD_E_SHAPE;
    Attention call{};
    call.query = static_cast<const uint16_t*>(tensors[0].data);
    call.output = static_cast<uint16_t*>(output.data);
    call.qpitch = tensors[0].stride[0]; call.opitch = output.stride[0];
    call.rows = int(rows); call.seqs = int(sequences); call.cu = static_cast<const int*>(cu.data);
    call.scale = real(args, "scale", 0.0625f);
    if (!std::isfinite(call.scale) || call.scale <= 0) return RAD_E_INVAL;
    if (PAGED) {
        if (!cache(tensors[1]) || !vector(tensors[3], RAD_I32, sequences)
            || tensors[4].data || tensors[5].data || !tensors[2].data || tensors[2].dtype != RAD_I32
            || tensors[2].rank != 2 || tensors[2].shape[0] != sequences || tensors[2].shape[1] < 1
            || tensors[2].shape[1] > 50000 || tensors[2].stride[1] != 1
            || tensors[2].stride[0] < tensors[2].shape[1]) return RAD_E_SHAPE;
        call.kv = static_cast<const int8_t*>(tensors[1].data);
        call.table = static_cast<const int*>(tensors[2].data); call.pages = tensors[1].shape[0];
        call.used = static_cast<const int*>(tensors[3].data); call.table_pitch = tensors[2].stride[0];
        call.table_width = int(tensors[2].shape[1]);
    } else {
        if (!matrix(tensors[1], RAD_BF16, rows, 256) || !matrix(tensors[2], RAD_BF16, rows, 256))
            return RAD_E_SHAPE;
        call.key = static_cast<const uint16_t*>(tensors[1].data);
        call.value = static_cast<const uint16_t*>(tensors[2].data);
        call.kpitch = tensors[1].stride[0]; call.vpitch = tensors[2].stride[0];
    }
    hipLaunchKernelGGL(attention<PAGED>, dim3(rows, 12), dim3(1), 0, reinterpret_cast<hipStream_t>(stream), call);
    return launched();
}
extern "C" int r3dh_attn_qsa_dense(const RadArgs* args, RadStream stream) { return launch_attention<false>(args, stream); }
extern "C" int r3dh_attn_qsa_paged(const RadArgs* args, RadStream stream) { return launch_attention<true>(args, stream); }
