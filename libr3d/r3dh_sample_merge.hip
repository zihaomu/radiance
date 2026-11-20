#include "r3dh_sample_merge.h"
#include <hip/hip_runtime.h>
#include <cstdint>
#include <limits>
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__)
#error sample_merge requires strict NaN and infinity semantics
#endif
namespace {
struct Entry { uint32_t id, bits, position; };
__device__ bool before(const Entry& left, const Entry& right) {
    if (left.id == UINT32_MAX || right.id == UINT32_MAX)
        return left.id != UINT32_MAX;
    const float lhs = __uint_as_float(left.bits), rhs = __uint_as_float(right.bits);
    if (lhs != rhs) return lhs > rhs;
    if (left.id != right.id) return left.id < right.id;
    return left.position < right.position;
}
template<int Capacity>
__global__ __launch_bounds__(256) void merge(const uint32_t* gathered, const RadSampleParams* params,
    int32_t* indices, uint32_t* values, int width, int inputs, int64_t gather_pitch,
    int64_t index_pitch, int64_t value_pitch) {
    __shared__ Entry entries[Capacity];
    const int row = blockIdx.x;
    for (int item = threadIdx.x; item < Capacity; item += blockDim.x) {
        Entry entry{UINT32_MAX, 0xff800000u, uint32_t(item)};
        if (item < inputs) {
            const uint32_t id = gathered[row * gather_pitch + item * 2];
            const uint32_t bits = gathered[row * gather_pitch + item * 2 + 1];
            if (id <= INT32_MAX && (bits & 0x7fffffffu) <= 0x7f800000u)
                entry = Entry{id, bits, uint32_t(item)};
        }
        entries[item] = entry;
    }
    __syncthreads();
    for (int span = 2; span <= Capacity; span *= 2) {
        for (int distance = span / 2; distance; distance /= 2) {
            for (int item = threadIdx.x; item < Capacity; item += blockDim.x) {
                const int peer = item ^ distance;
                if (peer > item) {
                    const Entry left = entries[item], right = entries[peer];
                    if ((item & span) ? before(left, right) : before(right, left)) {
                        entries[item] = right; entries[peer] = left;
                    }
                }
            }
            __syncthreads();
        }
    }
    if (threadIdx.x == 0) {
        const RadSampleParams parameter = params[row];
        const int keep = parameter.flags & RAD_SP_GREEDY ? 0
            : parameter.top_k > 0 && parameter.top_k < width ? parameter.top_k : width;
        int emitted = 0;
        for (int item = 0; item < inputs && emitted < keep; ++item) {
            const Entry entry = entries[item];
            if (entry.id == UINT32_MAX) break;
            if (item && entry.id == entries[item - 1].id
                && __uint_as_float(entry.bits) == __uint_as_float(entries[item - 1].bits)) continue;
            indices[row * index_pitch + emitted] = int32_t(entry.id);
            values[row * value_pitch + emitted] = entry.bits;
            ++emitted;
        }
        for (; emitted < width; ++emitted) {
            indices[row * index_pitch + emitted] = -1;
            values[row * value_pitch + emitted] = 0xff800000u;
        }
    }
}
bool plane(const RadTensor& tensor, int64_t rows, int64_t width) {
    return tensor.rank == 2 && tensor.shape[0] == rows && tensor.shape[1] == width;
}
bool pitch(const RadTensor& tensor) {
    return tensor.stride[1] == 1 && tensor.stride[0] >= tensor.shape[1]
        && tensor.stride[0] <= INT32_MAX;
}
bool overlap(const RadTensor& left, const RadTensor& right) {
    const uintptr_t lhs = reinterpret_cast<uintptr_t>(left.data), rhs = reinterpret_cast<uintptr_t>(right.data);
    const uint64_t lbytes = uint64_t((left.shape[0] - 1) * left.stride[0] + left.shape[1]) * 4;
    const uint64_t rbytes = uint64_t((right.shape[0] - 1) * right.stride[0] + right.shape[1]) * 4;
    return lhs > UINTPTR_MAX - lbytes || rhs > UINTPTR_MAX - rbytes
        || (lhs < rhs + rbytes && rhs < lhs + lbytes);
}
}
extern "C" int r3dh_sample_merge_topk(const RadArgs* args, RadStream stream) {
    if (!args || args->n_t != 4 || !args->t || args->n_p < 0 || (args->n_p && !args->p)) return RAD_E_INVAL;
    const int64_t band = rad_param_geti(args->p, args->n_p, "M", 0);
    const int64_t rows = args->t[2].rank == 2 ? args->t[2].shape[0] : 0;
    const int64_t width = rad_param_geti(args->p, args->n_p, "n_cand", 0);
    if (rows < 1 || rows > band || band > 32 || width < 1 || width > 2048)
        return RAD_E_SHAPE;
    const RadTensor& gathered = args->t[0]; const RadTensor& params = args->t[1];
    const RadTensor& indices = args->t[2]; const RadTensor& values = args->t[3];
    for (int operand = 0; operand < 4; ++operand) {
        if (!args->t[operand].data || reinterpret_cast<uintptr_t>(args->t[operand].data) % (operand == 1 ? 8 : 4))
            return RAD_E_INVAL;
    }
    if ((gathered.dtype != RAD_U32 && gathered.dtype != RAD_I32)
        || (params.dtype != RAD_U32 && params.dtype != RAD_I32 && params.dtype != RAD_F32)
        || indices.dtype != RAD_I32 || values.dtype != RAD_F32) return RAD_E_DTYPE;
    if (!plane(indices, rows, width) || !plane(values, rows, width) || !plane(params, rows, 24)
        || gathered.rank != 2 || gathered.shape[0] != rows || gathered.shape[1] < 2
        || gathered.shape[1] > 8192 || gathered.shape[1] % 2) return RAD_E_SHAPE;
    for (int operand = 0; operand < 4; ++operand) if (!pitch(args->t[operand])) return RAD_E_STRIDE;
    if (params.stride[0] != 24) return RAD_E_STRIDE;
    const int64_t world = rad_param_geti(args->p, args->n_p, "world_size", 0);
    if (world && (world < 1 || world > 16 || world * width * 2 != gathered.shape[1])) return RAD_E_SHAPE;
    for (int output = 2; output < 4; ++output)
        for (int other = 0; other < output; ++other)
            if (overlap(args->t[output], args->t[other])) return RAD_E_INVAL;
    const int inputs = int(gathered.shape[1] / 2);
    const auto launch = [&](auto capacity) {
        hipLaunchKernelGGL((merge<decltype(capacity)::value>), dim3(unsigned(rows)), dim3(256), 0,
            reinterpret_cast<hipStream_t>(stream), static_cast<const uint32_t*>(gathered.data),
            static_cast<const RadSampleParams*>(params.data), static_cast<int32_t*>(indices.data),
            static_cast<uint32_t*>(values.data), int(width), inputs, gathered.stride[0], indices.stride[0], values.stride[0]);
    };
    if (inputs <= 256) launch(std::integral_constant<int,256>{});
    else if (inputs <= 512) launch(std::integral_constant<int,512>{});
    else if (inputs <= 1024) launch(std::integral_constant<int,1024>{});
    else if (inputs <= 2048) launch(std::integral_constant<int,2048>{});
    else launch(std::integral_constant<int,4096>{});
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
