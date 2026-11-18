#include "r3dh_hc_api.h"
#include "r3dh_hc_args.h"
#include <hip/hip_runtime.h>
#include <climits>

#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ > 0)
#error "Strict HC requires -fno-fast-math -ffp-contract=off -fno-associative-math"
#endif

#pragma clang fp contract(off)
#pragma clang fp reassociate(off)

namespace {
constexpr int width = 2560;
constexpr int streams = 4;
constexpr int wide = width * streams;
constexpr int rank = 320;
constexpr int down_pitch = 10560;
constexpr int up_pitch = 336;

size_t aligned(size_t bytes) { return (bytes + 255) & ~size_t(255); }

__device__ float widen_bf16(uint16_t value) {
    return __uint_as_float(uint32_t(value) << 16);
}
__device__ uint16_t narrow_bf16(float value) {
    const uint32_t bits = __float_as_uint(value);
    if ((bits & 0x7fffffffU) > 0x7f800000U) return uint16_t((bits >> 16) | 64);
    return uint16_t((bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16);
}
__device__ float e4m3(uint8_t code) {
    const unsigned exponent = (code >> 3) & 15, mantissa = code & 7;
    const float magnitude = exponent ? __uint_as_float(((exponent + 120) << 23) | (mantissa << 20))
                                     : float(mantissa) * 0x1p-9f;
    return code & 128 ? -magnitude : magnitude;
}
__device__ float sigmoid_exact(float value) {
    const float exponential = float(::exp(double(-value)));
    return __fdiv_rn(1.0f, 1.0f + exponential);
}
__device__ float matrix_value(const uint8_t* matrix, int row, int column,
                              int columns, int group, int pitch) {
    const uint8_t* codes = matrix + size_t(row) * pitch;
    const float* scales = reinterpret_cast<const float*>(codes + columns);
    return e4m3(codes[column]) * scales[column / group];
}

__global__ void strict_norm(const uint16_t* hidden, const float* weights, uint16_t* normalized,
                             int rows, float eps, float wadd) {
    const int row = blockIdx.x, channel_stream = threadIdx.x;
    if (row >= rows || channel_stream >= streams) return;
    const size_t base = size_t(row) * wide + channel_stream * width;
    double squares = 0;
    for (int channel = 0; channel < width; ++channel) {
        const double value = double(widen_bf16(hidden[base + channel]));
        squares += value * value;
    }
    const float scale = __fdiv_rn(1.0f, sqrtf(float(squares / double(width)) + eps));
    for (int channel = 0; channel < width; ++channel)
        normalized[base + channel] = narrow_bf16((widen_bf16(hidden[base + channel]) * scale) *
            (weights[channel_stream * width + channel] + wadd));
}

__global__ void strict_down(const uint16_t* normalized, const uint8_t* weights,
                             uint16_t* intermediate, int rows, uint16_t* trace) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x, row = blockIdx.y;
    if (column >= rank || row >= rows) return;
    float sum = 0;
    for (int channel = 0; channel < wide; ++channel)
        sum += widen_bf16(normalized[size_t(row) * wide + channel]) *
               matrix_value(weights, column, channel, wide, 128, down_pitch);
    const float divided = sum / float(streams);
    const uint16_t value = narrow_bf16(divided * sigmoid_exact(divided));
    intermediate[size_t(row) * rank + column] = value;
    if (trace) trace[size_t(row) * rank + column] = value;
}

__global__ void strict_up(const uint16_t* intermediate, const uint8_t* weights,
                           uint16_t* gates, int rows, float* trace_logits, uint16_t* trace_gates) {
    const int channel = blockIdx.x * blockDim.x + threadIdx.x, row = blockIdx.y;
    if (channel >= wide || row >= rows) return;
    float sum = 0;
    for (int column = 0; column < rank; ++column)
        sum += widen_bf16(intermediate[size_t(row) * rank + column]) *
               matrix_value(weights, channel, column, rank, 80, up_pitch);
    const size_t index = size_t(row) * wide + channel;
    gates[index] = narrow_bf16(sigmoid_exact(sum));
    if (trace_logits) trace_logits[index] = sum;
    if (trace_gates) trace_gates[index] = gates[index];
}

__global__ void strict_mix(const uint16_t* normalized, const uint16_t* gates,
                            uint16_t* output, int rows) {
    const int channel = blockIdx.x * blockDim.x + threadIdx.x, row = blockIdx.y;
    if (channel >= width || row >= rows) return;
    float sum = 0;
    for (int channel_stream = 0; channel_stream < streams; ++channel_stream) {
        const size_t index = size_t(row) * wide + channel_stream * width + channel;
        sum += widen_bf16(gates[index]) * widen_bf16(normalized[index]);
    }
    output[size_t(row) * width + channel] = narrow_bf16(sum / float(streams));
}

__global__ void strict_inject(const uint16_t* normalized, const uint16_t* weights,
                               uint16_t* output, int rows) {
    const int row = blockIdx.x, channel_stream = threadIdx.x;
    if (row >= rows || channel_stream >= streams) return;
    float sum = 0;
    for (int channel = 0; channel < wide; ++channel)
        sum += widen_bf16(normalized[size_t(row) * wide + channel]) *
               widen_bf16(weights[channel_stream * wide + channel]);
    output[size_t(row) * streams + channel_stream] =
        narrow_bf16(2.0f * sigmoid_exact(sum / float(streams)));
}
__device__ uint8_t encode_e4m3(float value) {
    const unsigned sign = (__float_as_uint(value) >> 24) & 128;
    const float magnitude = fminf(fabsf(value), 448.0f);
    int lower = 0, upper = 126;
    while (lower + 1 < upper) {
        const int middle = (lower + upper) / 2;
        if (e4m3(uint8_t(middle)) <= magnitude) lower = middle;
        else upper = middle;
    }
    const float lower_distance = magnitude - e4m3(uint8_t(lower));
    const float upper_distance = e4m3(uint8_t(upper)) - magnitude;
    const int selected = lower_distance < upper_distance ||
        (lower_distance == upper_distance && !(lower & 1)) ? lower : upper;
    return uint8_t(sign | selected);
}

__global__ void strict_quant(const uint16_t* output, uint8_t* codes, float* scales,
                              bool rotated, bool int8_codes) {
    __shared__ float values[128];
    __shared__ float scale;
    const int lane = threadIdx.x;
    const size_t group_index = size_t(blockIdx.y) * (width / 128) + blockIdx.x;
    const size_t base = group_index * 128;
    values[lane] = widen_bf16(output[base + lane]);
    __syncthreads();
    if (rotated) {
        for (int step = 1; step < 128; step *= 2) {
            const float first = values[lane & ~step], second = values[lane | step];
            __syncthreads();
            values[lane] = lane & step ? first - second : first + second;
            __syncthreads();
        }
    }
    const float top = int8_codes ? 127.0f : 448.0f;
    if (lane == 0) {
        float maximum = 0;
        for (int index = 0; index < 128; ++index)
            if (fabsf(values[index]) > maximum) maximum = fabsf(values[index]);
        scale = maximum > 0 ? maximum * __fdiv_rn(1.0f, top) : 1.0f;
        scales[group_index] = scale;
    }
    __syncthreads();
    const float value = fminf(fmaxf(values[lane] * __fdiv_rn(1.0f, scale), -top), top);
    codes[base + lane] = int8_codes ? uint8_t(__float2int_rn(value)) : encode_e4m3(value);
}


bool dense(const RadTensor* tensor, uint32_t dtype, int64_t count) {
    if (!tensor || tensor->dtype != dtype || tensor->rank < 1 || tensor->rank > RAD_MAX_RANK)
        return false;
    int64_t stride = 1;
    for (int axis = int(tensor->rank) - 1; axis >= 0; --axis) {
        if (tensor->shape[axis] <= 0 || tensor->stride[axis] != stride) return false;
        if (tensor->shape[axis] > INT64_MAX / stride) return false;
        stride *= tensor->shape[axis];
    }
    return stride >= count;
}

int strict_launch(const RadArgs* args, RadStream stream, const R3dhHcTrace* trace) {
    if (!args || !args->t || args->n_t < 6) return RAD_E_INVAL;
    const int64_t bytes = r3dh_hc_rad_hc_read_strict_scratch(args);
    if (!bytes) return RAD_E_UNSUPPORTED;
    const RadTensor* hidden = rad_arg_in(args, 0), *weights = rad_arg_in(args, 1);
    const RadTensor* down = rad_arg_in(args, 2), *up = rad_arg_in(args, 3);
    const RadTensor* inject_weights = rad_arg_in(args, 4), *output = rad_arg_in(args, 5);
    const RadTensor* inject = rad_arg_in(args, 6);
    if (!hidden || !weights || !down || !up || !output) return RAD_E_INVAL;
    if (!dense(hidden, RAD_BF16, wide)) return RAD_E_DTYPE;
    const int64_t elements = rad_tensor_numel(hidden), rows = elements / wide;
    if (rows < 1 || elements % wide || rows > r3dh_hc_geti_or(args, "M", 0)) return RAD_E_SHAPE;
    if (!dense(hidden, RAD_BF16, rows * wide) || !dense(weights, RAD_F32, wide) ||
        !dense(down, RAD_F8E4M3, rank * wide) || !dense(up, RAD_F8E4M3, wide * rank) ||
        !dense(output, RAD_BF16, rows * width)) return RAD_E_DTYPE;
    if (down->rank != 2 || down->shape[0] != rank || down->shape[1] != wide ||
        up->rank != 2 || up->shape[0] != wide || up->shape[1] != rank) return RAD_E_SHAPE;
    if (bool(inject_weights) != bool(inject)) return RAD_E_INVAL;
    if (inject_weights && (!dense(inject_weights, RAD_BF16, streams * wide) ||
                           !dense(inject, RAD_BF16, rows * streams))) return RAD_E_DTYPE;
    const RadTensor* code_outputs[] = {rad_arg_in(args, 7), rad_arg_in(args, 9)};
    const RadTensor* scale_outputs[] = {rad_arg_in(args, 8), rad_arg_in(args, 10)};
    for (int pair = 0; pair < 2; ++pair) {
        if (bool(code_outputs[pair]) != bool(scale_outputs[pair])) return RAD_E_INVAL;
        if (!code_outputs[pair]) continue;
        if (r3dh_hc_geti_or(args, "group", 0) != 128) return RAD_E_UNSUPPORTED;
        const uint32_t dtype = code_outputs[pair]->dtype;
        if ((dtype != RAD_I8 && dtype != RAD_F8E4M3) ||
            !dense(code_outputs[pair], dtype, rows * width) ||
            !dense(scale_outputs[pair], RAD_F32, rows * (width / 128))) return RAD_E_DTYPE;
    }
    if (trace && (!trace->down || !trace->logits || !trace->gates || trace->rows != rows ||
                  trace->n != width || trace->hc != streams || trace->lowrank != rank)) return RAD_E_SHAPE;
    const float eps = float(r3dh_hc_getf(args, "eps", 1e-6));
    const float wadd = float(r3dh_hc_getf(args, "wadd", 0));
    if (!std::isfinite(eps) || eps < 0 || !std::isfinite(wadd)) return RAD_E_INVAL;
    if (!args->scratch || args->scratch_bytes < bytes || (uintptr_t(args->scratch) & 255)) return RAD_E_SCRATCH;
    auto* normalized = static_cast<uint16_t*>(args->scratch);
    const size_t band = size_t(r3dh_hc_geti_or(args, "M", 0));
    auto* intermediate = reinterpret_cast<uint16_t*>(static_cast<uint8_t*>(args->scratch) + aligned(band * wide * 2));
    auto* gates = reinterpret_cast<uint16_t*>(reinterpret_cast<uint8_t*>(intermediate) + aligned(band * rank * 2));
    const auto device_stream = reinterpret_cast<hipStream_t>(stream);
    strict_norm<<<dim3(rows), 4, 0, device_stream>>>(static_cast<const uint16_t*>(hidden->data),
        static_cast<const float*>(weights->data), normalized, int(rows), eps, wadd);
    strict_down<<<dim3((rank + 63) / 64, rows), 64, 0, device_stream>>>(normalized,
        static_cast<const uint8_t*>(down->data), intermediate, int(rows), trace ? trace->down : nullptr);
    strict_up<<<dim3((wide + 63) / 64, rows), 64, 0, device_stream>>>(intermediate,
        static_cast<const uint8_t*>(up->data), gates, int(rows), trace ? trace->logits : nullptr,
        trace ? trace->gates : nullptr);
    strict_mix<<<dim3((width + 127) / 128, rows), 128, 0, device_stream>>>(normalized, gates,
        static_cast<uint16_t*>(output->data), int(rows));
    if (inject_weights)
        strict_inject<<<dim3(rows), 4, 0, device_stream>>>(normalized,
            static_cast<const uint16_t*>(inject_weights->data), static_cast<uint16_t*>(inject->data), int(rows));
    for (int pair = 0; pair < 2; ++pair)
        if (code_outputs[pair])
            strict_quant<<<dim3(width / 128, rows), 128, 0, device_stream>>>(
                static_cast<const uint16_t*>(output->data), static_cast<uint8_t*>(code_outputs[pair]->data),
                static_cast<float*>(scale_outputs[pair]->data), pair == 1, code_outputs[pair]->dtype == RAD_I8);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
}

extern "C" int64_t r3dh_hc_rad_hc_read_strict_scratch(const RadArgs* args) {
    if (!args || r3dh_hc_geti_or(args, "n", 0) != width || r3dh_hc_geti_or(args, "hc", 0) != streams ||
        r3dh_hc_geti_or(args, "lowrank", 0) != rank) return 0;
    const char* mix = r3dh_hc_gets(args, "mix"), *dtype = r3dh_hc_gets(args, "dtype");
    if (!mix || std::strcmp(mix, "e4m3") || (dtype && std::strcmp(dtype, "bf16"))) return 0;
    const int64_t band = r3dh_hc_geti_or(args, "M", 0);
    if (band < 1 || band > 65535) return 0;
    return int64_t(aligned(size_t(band) * wide * 2) * 2 + aligned(size_t(band) * rank * 2));
}
extern "C" int r3dh_hc_rad_hc_read_strict(const RadArgs* args, RadStream stream) {
    return strict_launch(args, stream, nullptr);
}
extern "C" int r3dh_hc_rad_hc_read_strict_trace(const RadArgs* args, RadStream stream, const R3dhHcTrace* trace) {
    if (!trace) return RAD_E_INVAL;
    return strict_launch(args, stream, trace);
}
extern "C" int r3dh_hc_rad_hc_read(const RadArgs* args, RadStream stream) {
    return r3dh_hc_rad_hc_read_strict(args, stream);
}
extern "C" int r3dh_hc_rad_hc_read_trace(const RadArgs* args, RadStream stream, const R3dhHcTrace* trace) {
    return r3dh_hc_rad_hc_read_strict_trace(args, stream, trace);
}
