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
__device__ float widen(uint16_t bits) { return __uint_as_float(static_cast<unsigned>(bits) << 16); }
__device__ uint16_t raw(const ChunkTensor& tensor, int row, int head, int column) {
    return static_cast<const uint16_t*>(tensor.data)[offset(tensor, row, head, column)];
}
// Largest row with row * (row + 1) / 2 <= element: walks a packed lower triangle.
__device__ int triangle_row(int element) {
    int row = static_cast<int>((sqrtf(8.0f * element + 1.0f) - 1.0f) * 0.5f);
    while (row * (row + 1) / 2 > element) --row;
    while ((row + 1) * (row + 2) / 2 <= element) ++row;
    return row;
}
// The sequential dot of one staged BF16 row (16-byte aligned) with a register-resident state row.
__device__ __forceinline__ float row_dot(const uint16_t* row, const float (&state)[128]) {
    const uint4* packed = reinterpret_cast<const uint4*>(row);
    float dot = 0;
#pragma unroll
    for (int group = 0; group < 16; ++group) {
        const uint4 bits = packed[group];
        const unsigned word[4] = {bits.x, bits.y, bits.z, bits.w};
#pragma unroll
        for (int half = 0; half < 8; ++half) {
            const unsigned value = half & 1 ? word[half >> 1] & 0xffff0000u : word[half >> 1] << 16;
            dot = dot + __uint_as_float(value) * state[group * 8 + half];
        }
    }
    return dot;
}
__global__ __launch_bounds__(128) void solve_kernel(ChunkCall call) {
    const int lane = threadIdx.x, sequence = blockIdx.y, head = blockIdx.z;
    const auto* tensor = call.tensor; const auto* cumulative = static_cast<const int*>(tensor[3].data);
    const int begin = cumulative[sequence], end = cumulative[sequence + 1];
    if (begin < 0 || end < begin || end > call.rows) return;
    const int chunk = begin + static_cast<int>(blockIdx.x) * 64;
    if (chunk >= end) return;
    const int length = end - chunk < 64 ? end - chunk : 64;
    // The keys are dead once the products are formed; the inverse reuses their storage.
    __shared__ float keys[64 * 129], matrix[64 * 64], gates[64], betas[64];
    float* const inverse = keys;
    for (int element = lane; element < length * 128; element += 128)
        keys[(element >> 7) * 129 + (element & 127)] = load(tensor[0], chunk + (element >> 7), head / 3, element & 127);
    if (lane < length) {
        gates[lane] = scalar(tensor[2], chunk + lane, head);
        betas[lane] = scalar(tensor[1], chunk + lane, head);
    }
    __syncthreads();
    for (int pair = lane; pair < 64 * 63 / 2; pair += 128) {
        const int token = triangle_row(pair) + 1, previous = pair - token * (token - 1) / 2;
        if (token >= length) continue;
        float dot = 0;
        for (int column = 0; column < 128; ++column)
            dot = dot + keys[token * 129 + column] * keys[previous * 129 + column];
        matrix[token * 64 + previous] = betas[token] * dot * factor(gates[token] - gates[previous]);
    }
    __syncthreads();
    // Each inverse column is an independent forward substitution, summed in the serial order.
    if (lane < 64) {
        const int previous = lane;
        for (int token = 0; token < length; ++token) {
            float value = previous == token ? 1 : 0;
            for (int intermediate = previous; intermediate < token; ++intermediate)
                value = value - matrix[token * 64 + intermediate] * inverse[intermediate * 64 + previous];
            inverse[token * 64 + previous] = previous <= token ? value : 0;
        }
    }
    __syncthreads();
    for (int element = lane; element < length * 64; element += 128)
        store(tensor[4], chunk + (element >> 6), head, element & 63, inverse[element]);
}
__global__ __launch_bounds__(128) void scan_kernel(ChunkCall call) {
    const int sequence = blockIdx.x, head = blockIdx.y, value_row = threadIdx.x;
    const auto* tensor = call.tensor; const auto* cumulative = static_cast<const int*>(tensor[7].data);
    const int begin = cumulative[sequence], end = cumulative[sequence + 1];
    if (begin < 0 || end < begin || end > call.rows) return;
    const int slot = static_cast<const int*>(tensor[10].data)[sequence * tensor[10].stride[0]];
    if (slot < 0 || slot >= call.cache_slots) return;
    float state_row[128];
    const int64_t initial_base = slot * tensor[6].stride[0] + head * tensor[6].stride[1] + value_row * 128;
#pragma unroll
    for (int column = 0; column < 128; ++column)
        state_row[column] = tensor[6].data ? static_cast<const float*>(tensor[6].data)[initial_base + column] : 0;
    // Packed lower triangles; the staged queries are dead after the products and their storage
    // then holds each value row's work, transformed in place.
    __shared__ float products[64 * 65 / 2], gates[64], decays[64], growth[64], rows[64 * 128];
    __shared__ __align__(16) uint16_t keys[64 * 136];
    __shared__ uint16_t inverse[64 * 65 / 2];
    uint16_t* const queries = reinterpret_cast<uint16_t*>(rows);
    for (int chunk = begin; chunk < end; chunk += 64) {
        const int length = end - chunk < 64 ? end - chunk : 64, triangle = length * (length + 1) / 2;
        for (int element = value_row; element < length * 128; element += 128) {
            const int token = element >> 7, column = element & 127;
            keys[token * 136 + column] = raw(tensor[1], chunk + token, head / 3, column);
            queries[token * 136 + column] = raw(tensor[0], chunk + token, head / 3, column);
        }
        for (int element = value_row; element < triangle; element += 128) {
            const int token = triangle_row(element);
            inverse[element] = raw(tensor[3], chunk + token, head, element - token * (token + 1) / 2);
        }
        if (value_row < length) {
            gates[value_row] = scalar(tensor[4], chunk + value_row, head);
            decays[value_row] = factor(scalar(tensor[4], chunk + length - 1, head) - gates[value_row]);
            growth[value_row] = factor(gates[value_row]);
        }
        __syncthreads();
        for (int element = value_row; element < triangle; element += 128) {
            const int token = triangle_row(element), previous = element - token * (token + 1) / 2;
            float dot = 0;
            for (int column = 0; column < 128; ++column)
                dot = dot + widen(queries[token * 136 + column]) * widen(keys[previous * 136 + column]);
            products[element] = call.scale * factor(gates[token] - gates[previous]) * dot;
        }
        __syncthreads();
        for (int token = 0; token < length; ++token) {
            const float dot = row_dot(keys + token * 136, state_row);
            rows[token * 128 + value_row] = scalar(tensor[5], chunk + token, head)
                * (load(tensor[2], chunk + token, head, value_row) - growth[token] * dot);
        }
        for (int token = length - 1; token >= 0; --token) {
            float value = 0;
            for (int previous = 0; previous <= token; ++previous)
                value = value + widen(inverse[token * (token + 1) / 2 + previous]) * rows[previous * 128 + value_row];
            rows[token * 128 + value_row] = value;
        }
        for (int token = 0; token < length; ++token) {
            const uint16_t* query = static_cast<const uint16_t*>(tensor[0].data) + offset(tensor[0], chunk + token, head / 3, 0);
            float dot = 0;
            if ((reinterpret_cast<uintptr_t>(query) & 15) == 0) dot = row_dot(query, state_row);
            else {
#pragma unroll
                for (int column = 0; column < 128; ++column) dot = dot + widen(query[column]) * state_row[column];
            }
            const float coefficient = call.scale * growth[token];
            float value = coefficient * dot;
            for (int previous = 0; previous <= token; ++previous)
                value = value + products[token * (token + 1) / 2 + previous] * rows[previous * 128 + value_row];
            store(tensor[8], chunk + token, head, value_row, value);
        }
        const float last_decay = growth[length - 1];
#pragma unroll
        for (int column = 0; column < 128; ++column) state_row[column] = last_decay * state_row[column];
        for (int token = 0; token < length; ++token) {
            const float transformed = rows[token * 128 + value_row], decay = decays[token];
            const uint4* packed = reinterpret_cast<const uint4*>(keys + token * 136);
#pragma unroll
            for (int group = 0; group < 16; ++group) {
                const uint4 bits = packed[group];
                const unsigned word[4] = {bits.x, bits.y, bits.z, bits.w};
#pragma unroll
                for (int half = 0; half < 8; ++half) {
                    const unsigned key = half & 1 ? word[half >> 1] & 0xffff0000u : word[half >> 1] << 16;
                    const float scaled_key = decay * __uint_as_float(key);
                    state_row[group * 8 + half] = state_row[group * 8 + half] + transformed * scaled_key;
                }
            }
        }
        __syncthreads();
    }
    const int64_t final_base = slot * tensor[9].stride[0] + head * tensor[9].stride[1] + value_row * 128;
#pragma unroll
    for (int column = 0; column < 128; ++column)
        static_cast<float*>(tensor[9].data)[final_base + column] = state_row[column];
}
}
extern "C" int r3dh_gdn_extra_kkt_strict(const RadArgs* args, RadStream stream) {
    ChunkCall call{}; const int status = prepare(args, true, &call); if (status != RAD_OK) return status;
    hipLaunchKernelGGL(solve_kernel, dim3((call.rows + 63) / 64, call.sequences, call.heads), dim3(128), 0,
        reinterpret_cast<hipStream_t>(stream), call);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
extern "C" int r3dh_gdn_extra_chunk_strict(const RadArgs* args, RadStream stream) {
    ChunkCall call{}; const int status = prepare(args, false, &call); if (status != RAD_OK) return status;
    hipLaunchKernelGGL(scan_kernel, dim3(call.sequences, call.heads), dim3(128), 0,
        reinterpret_cast<hipStream_t>(stream), call);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
