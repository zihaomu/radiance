#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "r3dh_sparse_qsa_internal.h"
namespace {
__device__ float bf_load(const RadTensor& value, int64_t offset) {
    return __uint_as_float(unsigned(static_cast<const uint16_t*>(value.data)[offset]) << 16);
}
__device__ uint16_t bf_round(float value) {
    unsigned bits = __float_as_uint(value);
    if ((bits & 0x7fffffffU) > 0x7f800000U) return uint16_t((bits >> 16) | 64);
    return uint16_t((bits + 0x7fffU + ((bits >> 16) & 1)) >> 16);
}
__device__ float bf_value(float value) { return __uint_as_float(unsigned(bf_round(value)) << 16); }
struct KeyCall {
    RadTensor keys, weights, pages, positions, output;
    int blocks, rot, mode, sections[3];
    int64_t pos0;
    float eps, wadd, theta, scale;
};
__global__ void key_kernel(KeyCall call) {
    const int block = blockIdx.x;
    const int page = call.pages.data ? static_cast<const int*>(call.pages.data)[block * call.pages.stride[0]] : block;
    if (page < 0 || page >= call.output.shape[0]) return;
    float pooled[128], normed[128];
    double square = 0;
    for (int lane = 0; lane < 128; ++lane) {
        float sum = 0;
        for (int token = 0; token < 4; ++token) sum += bf_load(call.keys, (block * 4 + token) * call.keys.stride[0] + lane);
        pooled[lane] = bf_value(sum / 4.0f);
        square += double(pooled[lane]) * double(pooled[lane]);
    }
    const float inverse = 1.0f / sqrtf(float(square / 128.0) + call.eps);
    for (int lane = 0; lane < 128; ++lane)
        normed[lane] = pooled[lane] * inverse * (call.wadd + static_cast<const float*>(call.weights.data)[lane]);
    const int half = call.rot / 2;
    for (int lane = 0; lane < 128; ++lane) {
        float value = normed[lane];
        if (lane < call.rot) {
            const int frequency = lane % half;
            int component = 0;
            if (call.mode == 2) component = frequency < call.sections[0] ? 0 : frequency < call.sections[0] + call.sections[1] ? 1 : 2;
            if (call.mode == 3) {
                if (frequency % 3 == 1 && frequency < 3 * call.sections[1]) component = 1;
                if (frequency % 3 == 2 && frequency < 3 * call.sections[2]) component = 2;
            }
            const int64_t position = call.positions.data
                ? static_cast<const int*>(call.positions.data)[block * call.positions.stride[0] + component]
                : call.pos0 + block * 4;
            const float inverse_frequency = 1.0f / float(pow(double(call.theta), double(float(2 * frequency) / float(call.rot))));
            const float angle = (float(position) / call.scale) * inverse_frequency;
            const float cosine = float(cos(double(angle))), sine = float(sin(double(angle)));
            value = lane < half ? normed[lane] * cosine - normed[lane + half] * sine
                                : normed[lane - half] * sine + normed[lane] * cosine;
        }
        static_cast<uint16_t*>(call.output.data)[page * call.output.stride[0] + lane] = bf_round(value);
    }
}
struct AttnCall {
    RadTensor query, cache, table, used, cu, output;
    int dim, heads, gqa, sequences, rows;
    float scale;
};
__device__ int64_t cache_row(const AttnCall& call, int sequence, int key, int head) {
    const int page = static_cast<const int*>(call.table.data)[sequence * call.table.stride[0] + (key / 4) * call.table.stride[1]];
    if (page < 0 || page >= call.cache.shape[0]) return -1;
    return page * call.cache.stride[0] + (head / call.gqa) * call.cache.stride[1] + (key % 4) * call.cache.stride[2];
}
__device__ float cache_value(const AttnCall& call, int64_t row, int lane, int half) {
    if (call.cache.dtype == RAD_BF16) return bf_load(call.cache, row + half * call.dim + lane);
    const auto* data = static_cast<const unsigned char*>(call.cache.data);
    const int scale_offset = 2 * call.dim + 2 * half;
    const uint16_t bits = uint16_t(data[row + scale_offset]) | (uint16_t(data[row + scale_offset + 1]) << 8);
    const float scale = __half2float(__ushort_as_half(bits));
    return float(static_cast<const signed char*>(call.cache.data)[row + half * call.dim + lane]) * scale;
}
__device__ float dot(const AttnCall& call, int64_t query, int64_t cache) {
    float value = 0;
    for (int lane = 0; lane < call.dim; ++lane)
        value += bf_load(call.query, query + lane) * cache_value(call, cache, lane, 0);
    return value * call.scale;
}
__global__ __launch_bounds__(256) void attention_kernel(AttnCall call) {
    const int token = blockIdx.x, head = blockIdx.y;
    const int lane = threadIdx.x;
    if (lane >= call.dim) return;
    int sequence = -1, first = 0, last = 0;
    for (int index = 0; index < call.sequences; ++index) {
        const int begin = call.cu.data ? static_cast<const int*>(call.cu.data)[index * call.cu.stride[0]] : index * (call.rows / call.sequences);
        const int end = call.cu.data ? static_cast<const int*>(call.cu.data)[(index + 1) * call.cu.stride[0]] : (index + 1) * (call.rows / call.sequences);
        if (token >= begin && token < end) { sequence = index; first = begin; last = end; break; }
    }
    if (sequence < 0) return;
    const int context = static_cast<const int*>(call.used.data)[sequence * call.used.stride[0]];
    const int visible = min(context - (last - first) + token - first + 1, int(call.table.shape[1]) * 4);
    const int64_t query_offset = token * call.query.stride[0] + head * call.dim;
    const int64_t out_offset = token * call.output.stride[0] + head * call.dim + lane;
    __shared__ float probability[256], reduce[256];
    __shared__ int64_t rows[256];
    // Parallel over keys/columns only; every per-output sum keeps the sequential key order.
    float maximum = -INFINITY;
    for (int key = lane; key < visible; key += call.dim) {
        const int64_t row = cache_row(call, sequence, key, head);
        if (row >= 0) { const float score = dot(call, query_offset, row); if (score > maximum) maximum = score; }
    }
    reduce[lane] = maximum;
    __syncthreads();
    for (int span = call.dim / 2; span > 0; span >>= 1) {
        if (lane < span && reduce[lane + span] > reduce[lane]) reduce[lane] = reduce[lane + span];
        __syncthreads();
    }
    maximum = reduce[0];
    __syncthreads();
    float accumulator = 0, denominator = 0;
    for (int base = 0; base < visible; base += call.dim) {
        if (base + lane < visible) {
            const int64_t row = cache_row(call, sequence, base + lane, head);
            rows[lane] = row;
            probability[lane] = row >= 0 && maximum != -INFINITY
                ? float(exp(double(dot(call, query_offset, row) - maximum))) : 0;
        }
        __syncthreads();
        const int count = min(call.dim, visible - base);
        for (int index = 0; index < count; ++index) {
            if (rows[index] < 0) continue;
            if (lane == 0) denominator += probability[index];
            accumulator += probability[index] * cache_value(call, rows[index], lane, 1);
        }
        __syncthreads();
    }
    if (lane == 0) reduce[0] = denominator;
    __syncthreads();
    denominator = reduce[0];
    const float inverse = denominator > 0 ? 1.0f / denominator : 0;
    static_cast<uint16_t*>(call.output.data)[out_offset] = bf_round(accumulator * inverse);
}
}
extern "C" int r3dh_sparse_qsa_block_key(const RadArgs* args, RadStream stream) {
    const int status = r3dh_sparse_qsa_shape("qsa_block_key", args);
    if (status) return status;
    KeyCall call{};
    call.keys = args->t[0]; call.weights = args->t[1]; call.pages = args->t[2];
    call.positions = args->t[3]; call.output = args->t[4];
    call.blocks = call.pages.data ? call.pages.shape[0] : call.output.shape[0];
    call.rot = r3dh_sparse::integer(args, "rotary_dim", 0);
    call.pos0 = r3dh_sparse::integer(args, "pos0", -1);
    call.eps = r3dh_sparse::real(args, "eps", 1e-6); call.wadd = r3dh_sparse::real(args, "wadd", 1);
    call.theta = r3dh_sparse::real(args, "theta", 10000); call.scale = r3dh_sparse::real(args, "scale", 1);
    r3dh_sparse::sections(args, call.sections);
    const char* mode = rad_args_gets(args, "mode");
    if (call.positions.rank == 2 && call.positions.shape[1] == 3 && mode && rad_args_gets(args, "sections"))
        call.mode = !std::strcmp(mode, "mrope") ? 2 : !std::strcmp(mode, "imrope") ? 3 : 0;
    hipLaunchKernelGGL(key_kernel, dim3(call.blocks), dim3(1), 0, reinterpret_cast<hipStream_t>(stream), call);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
extern "C" int r3dh_sparse_qsa_paged(const RadArgs* args, RadStream stream) {
    const int status = r3dh_sparse_qsa_shape("attn_paged", args);
    if (status) return status;
    AttnCall call{};
    call.query = args->t[0]; call.cache = args->t[1]; call.table = args->t[2]; call.used = args->t[3];
    call.cu = args->t[6]; call.output = args->t[7];
    call.dim = r3dh_sparse::integer(args, "head_dim", 0); call.gqa = r3dh_sparse::integer(args, "gqa", 1);
    call.heads = call.cache.shape[1] * call.gqa; call.rows = call.query.shape[0]; call.sequences = call.used.shape[0];
    call.scale = r3dh_sparse::real(args, "scale", 1.0f / std::sqrt(float(call.dim)));
    hipLaunchKernelGGL(attention_kernel, dim3(call.rows, call.heads), dim3(call.dim), 0, reinterpret_cast<hipStream_t>(stream), call);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
