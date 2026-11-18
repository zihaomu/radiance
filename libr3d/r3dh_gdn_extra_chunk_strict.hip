#include "r3dh_gdn_extra.h"
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstring>
#if defined(__FAST_MATH__)
#error r3dh_gdn_extra requires -fno-fast-math -ffp-contract=off
#endif
#pragma clang fp contract(off)
namespace {
struct ChunkTensor { void* data; int64_t stride[2]; };
struct ChunkCall {
    ChunkTensor tensor[11];
    int rows, heads, keys, sequences;
    float scale;
    int64_t cache_slots;
};
static_assert(((sizeof(ChunkCall) + 7) & ~size_t(7)) + 256 <= 768,
              "GDN chunk arguments plus implicit ABI must fit the AQL slot");
bool plane(const RadTensor& tensor, uint32_t type, int rows, int heads, int width) {
    return tensor.data && tensor.dtype == type && tensor.rank == 3 && tensor.shape[0] == rows
        && tensor.shape[1] == heads && tensor.shape[2] == width && tensor.stride[2] == 1
        && tensor.stride[1] >= width && tensor.stride[0] >= heads * tensor.stride[1];
}
bool matrix(const RadTensor& tensor, int rows, int heads) {
    return tensor.data && tensor.dtype == RAD_F32 && tensor.rank == 2 && tensor.shape[0] == rows
        && tensor.shape[1] == heads && tensor.stride[1] == 1 && tensor.stride[0] >= heads;
}
bool state(const RadTensor& tensor, int heads) {
    return tensor.data && tensor.dtype == RAD_F32 && tensor.rank == 4 && tensor.shape[0] > 0
        && tensor.shape[1] == heads && tensor.shape[2] == 128 && tensor.shape[3] == 128
        && tensor.stride[3] == 1 && tensor.stride[2] == 128 && tensor.stride[1] >= 16384
        && tensor.stride[0] >= heads * tensor.stride[1];
}
int prepare(const RadArgs* args, bool solve, ChunkCall* call) {
    if (!args || !args->t || !args->p || args->n_t != (solve ? 5 : 11)) return RAD_E_INVAL;
    long long key_width = 0, value_width = 128, chunk = 64;
    rad_args_geti(args, "head_k", &key_width); rad_args_geti(args, "head_v", &value_width);
    rad_args_geti(args, "chunk", &chunk);
    if (key_width != 128 || value_width != 128 || chunk != 64) return RAD_E_UNSUPPORTED;
    *call = ChunkCall{};
    for (int index = 0; index < args->n_t; ++index) {
        call->tensor[index].data = args->t[index].data;
        call->tensor[index].stride[0] = args->t[index].stride[0];
        call->tensor[index].stride[1] = args->t[index].stride[1];
    }
    const RadTensor& inverse = args->t[solve ? 4 : 3];
    if (inverse.rank != 3 || inverse.shape[2] != 64 || inverse.shape[0] < 1
        || inverse.shape[0] > 2048 || (inverse.shape[1] != 24 && inverse.shape[1] != 48))
        return RAD_E_SHAPE;
    call->rows = inverse.shape[0]; call->heads = inverse.shape[1]; call->keys = call->heads / 3;
    if ((call->heads != 24 && call->heads != 48) || call->rows < 1 || call->rows > 2048)
        return RAD_E_SHAPE;
    if (!plane(inverse, RAD_BF16, call->rows, call->heads, 64)) return RAD_E_SHAPE;
    const RadTensor& cumulative = args->t[solve ? 3 : 7];
    if (!cumulative.data || cumulative.dtype != RAD_I32 || cumulative.rank != 1
        || cumulative.shape[0] < 2 || cumulative.shape[0] > 65 || cumulative.stride[0] != 1)
        return RAD_E_SHAPE;
    call->sequences = cumulative.shape[0] - 1;
    if (solve) {
        if (!plane(args->t[0], RAD_BF16, call->rows, call->keys, 128)
            || !matrix(args->t[1], call->rows, call->heads)
            || !matrix(args->t[2], call->rows, call->heads)) return RAD_E_SHAPE;
    } else {
        if (!plane(args->t[0], RAD_BF16, call->rows, call->keys, 128)
            || !plane(args->t[1], RAD_BF16, call->rows, call->keys, 128)
            || !plane(args->t[2], RAD_BF16, call->rows, call->heads, 128)
            || !plane(args->t[8], RAD_BF16, call->rows, call->heads, 128)
            || !matrix(args->t[4], call->rows, call->heads)
            || !matrix(args->t[5], call->rows, call->heads)
            || (args->t[6].data && !state(args->t[6], call->heads))
            || !state(args->t[9], call->heads)) return RAD_E_SHAPE;
        if (args->t[6].data && args->t[6].shape[0] != args->t[9].shape[0]) return RAD_E_SHAPE;
        call->cache_slots = args->t[9].shape[0];
        const RadTensor& index = args->t[10];
        if (!index.data || index.dtype != RAD_I32 || index.shape[0] != call->sequences
            || (index.rank != 1 && index.rank != 2) || index.stride[0] < 1
            || (index.rank == 2 && (index.shape[1] < 1 || index.stride[1] != 1
                || index.stride[0] < index.shape[1]))) return RAD_E_SHAPE;
    }
    call->scale = 1;
    for (int index = 0; index < args->n_p; ++index)
        if (args->p[index].kind == RAD_P_F64 && !std::strcmp(args->p[index].key, "scale"))
            call->scale = args->p[index].dval;
    return std::isfinite(call->scale) ? RAD_OK : RAD_E_INVAL;
}
__device__ int64_t offset(const ChunkTensor& tensor, int row, int head, int column) {
    return row * tensor.stride[0] + head * tensor.stride[1] + column;
}
__device__ float load(const ChunkTensor& tensor, int row, int head, int column) {
    return __uint_as_float(static_cast<unsigned>(static_cast<const uint16_t*>(tensor.data)[offset(tensor, row, head, column)]) << 16);
}
__device__ void store(const ChunkTensor& tensor, int row, int head, int column, float value) {
    const unsigned bits = __float_as_uint(value);
    const uint16_t converted = (bits & 0x7fffffffu) > 0x7f800000u ? (bits >> 16) | 64u
        : (bits + 32767u + ((bits >> 16) & 1u)) >> 16;
    static_cast<uint16_t*>(tensor.data)[offset(tensor, row, head, column)] = converted;
}
__device__ float factor(float value) { return static_cast<float>(exp(static_cast<double>(value))); }
__device__ float scalar(const ChunkTensor& tensor, int token, int head) {
    return static_cast<const float*>(tensor.data)[token * tensor.stride[0] + head];
}
__global__ void solve_kernel(ChunkCall call) {
    if (threadIdx.x != 0) return;
    const int sequence = blockIdx.x, head = blockIdx.y;
    const auto* tensor = call.tensor; const auto* cumulative = static_cast<const int*>(tensor[3].data);
    const int begin = cumulative[sequence], end = cumulative[sequence + 1];
    if (begin < 0 || end < begin || end > call.rows) return;
    __shared__ float inverse[4096];
    float row[64], result[64];
    for (int chunk = begin; chunk < end; chunk += 64) {
        const int length = end - chunk < 64 ? end - chunk : 64;
        for (int token = 0; token < length; ++token) {
            for (int previous = 0; previous < token; ++previous) {
                float dot = 0;
                for (int column = 0; column < 128; ++column)
                    dot = dot + load(tensor[0], chunk + token, head / 3, column)
                        * load(tensor[0], chunk + previous, head / 3, column);
                row[previous] = scalar(tensor[1], chunk + token, head) * dot
                    * factor(scalar(tensor[2], chunk + token, head) - scalar(tensor[2], chunk + previous, head));
            }
            for (int previous = 0; previous <= token; ++previous) result[previous] = previous == token ? 1 : 0;
            for (int intermediate = 0; intermediate < token; ++intermediate)
                for (int previous = 0; previous <= intermediate; ++previous)
                    result[previous] = result[previous] - row[intermediate] * inverse[intermediate * 64 + previous];
            for (int previous = 0; previous < 64; ++previous) {
                const float value = previous <= token ? result[previous] : 0;
                inverse[token * 64 + previous] = value;
                store(tensor[4], chunk + token, head, previous, value);
            }
        }
    }
}
__global__ __launch_bounds__(128) void scan_kernel(ChunkCall call) {
    const int sequence = blockIdx.x, head = blockIdx.y, value_row = threadIdx.x;
    const auto* tensor = call.tensor; const auto* cumulative = static_cast<const int*>(tensor[7].data);
    const int begin = cumulative[sequence], end = cumulative[sequence + 1];
    if (begin < 0 || end < begin || end > call.rows) return;
    const int slot = static_cast<const int*>(tensor[10].data)[sequence * tensor[10].stride[0]];
    if (slot < 0 || slot >= call.cache_slots) return;
    float state_row[128], work[64], transformed[64];
    const int64_t initial_base = slot * tensor[6].stride[0] + head * tensor[6].stride[1] + value_row * 128;
    for (int column = 0; column < 128; ++column)
        state_row[column] = tensor[6].data ? static_cast<const float*>(tensor[6].data)[initial_base + column] : 0;
    __shared__ float products[4096], gates[64], decays[64];
    for (int chunk = begin; chunk < end; chunk += 64) {
        const int length = end - chunk < 64 ? end - chunk : 64;
        if (value_row < length) {
            gates[value_row] = scalar(tensor[4], chunk + value_row, head);
            decays[value_row] = factor(scalar(tensor[4], chunk + length - 1, head) - gates[value_row]);
            for (int previous = 0; previous <= value_row; ++previous) {
                float dot = 0;
                for (int column = 0; column < 128; ++column)
                    dot = dot + load(tensor[0], chunk + value_row, head / 3, column)
                        * load(tensor[1], chunk + previous, head / 3, column);
                products[value_row * 64 + previous] = call.scale
                    * factor(gates[value_row] - scalar(tensor[4], chunk + previous, head)) * dot;
            }
        }
        __syncthreads();
        for (int token = 0; token < length; ++token) {
            float dot = 0;
            for (int column = 0; column < 128; ++column)
                dot = dot + load(tensor[1], chunk + token, head / 3, column) * state_row[column];
            work[token] = scalar(tensor[5], chunk + token, head)
                * (load(tensor[2], chunk + token, head, value_row) - factor(gates[token]) * dot);
        }
        for (int token = 0; token < length; ++token) {
            float value = 0;
            for (int previous = 0; previous <= token; ++previous)
                value = value + load(tensor[3], chunk + token, head, previous) * work[previous];
            transformed[token] = value;
        }
        for (int token = 0; token < length; ++token) {
            float dot = 0;
            for (int column = 0; column < 128; ++column)
                dot = dot + load(tensor[0], chunk + token, head / 3, column) * state_row[column];
            const float coefficient = call.scale * factor(gates[token]);
            float value = coefficient * dot;
            for (int previous = 0; previous <= token; ++previous)
                value = value + products[token * 64 + previous] * transformed[previous];
            store(tensor[8], chunk + token, head, value_row, value);
        }
        const float last_decay = factor(gates[length - 1]);
        for (int column = 0; column < 128; ++column) {
            state_row[column] = last_decay * state_row[column];
            for (int token = 0; token < length; ++token) {
                const float scaled_key = decays[token] * load(tensor[1], chunk + token, head / 3, column);
                state_row[column] = state_row[column] + transformed[token] * scaled_key;
            }
        }
        __syncthreads();
    }
    const int64_t final_base = slot * tensor[9].stride[0] + head * tensor[9].stride[1] + value_row * 128;
    for (int column = 0; column < 128; ++column)
        static_cast<float*>(tensor[9].data)[final_base + column] = state_row[column];
}
}
extern "C" int r3dh_gdn_extra_kkt_strict(const RadArgs* args, RadStream stream) {
    ChunkCall call{}; const int status = prepare(args, true, &call); if (status != RAD_OK) return status;
    hipLaunchKernelGGL(solve_kernel, dim3(call.sequences, call.heads), dim3(32), 0,
        reinterpret_cast<hipStream_t>(stream), call);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
extern "C" int r3dh_gdn_extra_chunk_strict(const RadArgs* args, RadStream stream) {
    ChunkCall call{}; const int status = prepare(args, false, &call); if (status != RAD_OK) return status;
    hipLaunchKernelGGL(scan_kernel, dim3(call.sequences, call.heads), dim3(128), 0,
        reinterpret_cast<hipStream_t>(stream), call);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
