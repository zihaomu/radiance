#include "r3dh_qwen_ops.h"
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)

namespace {
__device__ float widen(uint16_t value) {
    return __uint_as_float(static_cast<uint32_t>(value) << 16);
}
__device__ uint16_t narrow(float value) {
    const uint32_t bits = __float_as_uint(value);
    if ((bits & 0x7fffffffu) > 0x7f800000u) return (bits >> 16) | 0x40u;
    return (bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16;
}
template<bool Separate>
__global__ void quantize(const uint16_t* gate, const uint16_t* input, int8_t* codes,
                        float* scales, uint16_t* output, int mode,
                        int64_t gate_pitch, int64_t input_pitch, int64_t code_pitch,
                        int64_t scale_pitch, int64_t output_pitch) {
    __shared__ float values[128];
    __shared__ float maxima[128];
    const unsigned position = threadIdx.x;
    const int64_t row = blockIdx.y, group = blockIdx.x;
    const int64_t column = group * 128 + position;
    const float gate_value = widen(gate[row * gate_pitch + column]);
    const float up_value = widen(input[row * input_pitch + column]);
    float activation = gate_value;
    if (mode != 2) {
        const float sigmoid = __fdiv_rn(1.0f, 1.0f + expf(-gate_value));
        activation = mode == 0 ? gate_value * sigmoid : sigmoid;
        if constexpr (Separate) activation = widen(narrow(activation));
    }
    const uint16_t product = narrow(activation * up_value);
    if (output) output[row * output_pitch + column] = product;
    float value = widen(product);
    values[position] = value;
    __syncthreads();
    for (unsigned distance = 1; distance < 128; distance <<= 1) {
        const float partner = values[position ^ distance];
        value = position & distance ? partner - value : value + partner;
        __syncthreads();
        values[position] = value;
        __syncthreads();
    }
    maxima[position] = fabsf(value);
    __syncthreads();
    for (unsigned distance = 64; distance; distance >>= 1) {
        if (position < distance)
            maxima[position] = fmaxf(maxima[position], maxima[position + distance]);
        __syncthreads();
    }
    const float scale = maxima[0] > 0.0f ? maxima[0] * (1.0f / 127.0f) : 1.0f;
    const float inverse = __fdiv_rn(1.0f, scale);
    const float rounded = fminf(127.0f, fmaxf(-127.0f, value * inverse));
    codes[row * code_pitch + column] = static_cast<int8_t>(__float2int_rn(rounded));
    if (position == 0) scales[row * scale_pitch + group] = scale;
}
int matrix(const RadTensor& tensor, uint32_t dtype, int64_t rows, int64_t columns) {
    if (!tensor.data) return RAD_E_INVAL;
    if (tensor.dtype != dtype) return RAD_E_DTYPE;
    if (tensor.rank != 2 || tensor.shape[0] != rows || tensor.shape[1] != columns)
        return RAD_E_SHAPE;
    const int bytes = dtype == RAD_F32 ? 4 : dtype == RAD_BF16 ? 2 : 1;
    if (tensor.stride[1] != 1 || tensor.stride[0] < columns
        || tensor.stride[0] > (std::numeric_limits<int64_t>::max() / bytes - columns) / rows
        || reinterpret_cast<uintptr_t>(tensor.data) % bytes) return RAD_E_STRIDE;
    return RAD_OK;
}
int flatten(const RadTensor& tensor, int64_t width, RadTensor* result) {
    if (tensor.rank < 1 || tensor.rank > RAD_MAX_RANK) return RAD_E_SHAPE;
    int64_t count = 1;
    for (unsigned axis = 0; axis < tensor.rank; ++axis) {
        if (tensor.shape[axis] < 1 || count > std::numeric_limits<int64_t>::max() / tensor.shape[axis])
            return RAD_E_SHAPE;
        count *= tensor.shape[axis];
    }
    if (count % width) return RAD_E_SHAPE;
    int64_t pitch = width;
    if (tensor.shape[tensor.rank - 1] == width) {
        if (tensor.stride[tensor.rank - 1] != 1) return RAD_E_STRIDE;
        if (tensor.rank > 1) pitch = tensor.stride[tensor.rank - 2];
        for (int axis = static_cast<int>(tensor.rank) - 2; axis > 0; --axis) {
            if (tensor.stride[axis] < 1 || tensor.stride[axis] > std::numeric_limits<int64_t>::max() / tensor.shape[axis]
                || tensor.stride[axis - 1] != tensor.shape[axis] * tensor.stride[axis]) return RAD_E_STRIDE;
        }
    } else {
        int64_t stride = 1;
        for (int axis = static_cast<int>(tensor.rank) - 1; axis >= 0; --axis) {
            if (tensor.shape[axis] != 1 && tensor.stride[axis] != stride) return RAD_E_STRIDE;
            stride *= tensor.shape[axis];
        }
    }
    *result = tensor; result->rank = 2;
    result->shape[0] = count / width; result->shape[1] = width;
    result->stride[0] = pitch; result->stride[1] = 1;
    return RAD_OK;
}
int launch(const RadArgs* args, RadStream stream, bool separate) {
    const int required = separate ? 4 : 3;
    if (!args || !args->t || args->n_t < required || args->n_t > required + 1
        || args->n_p < 0 || (args->n_p && !args->p)) return RAD_E_INVAL;
    const int code_index = separate ? 2 : 1;
    const int64_t width = rad_args_geti_or(args, "n", 0);
    if (width < 128 || width > 1048576 || width % 128
        || rad_args_geti_or(args, "group", 128) != 128) return RAD_E_SHAPE;
    RadTensor gate{}, codes{}, scales{}, input_view{}, output_view{};
    int status = flatten(args->t[0], separate ? width : width * 2, &gate);
    if (status) return status;
    if ((status = flatten(args->t[code_index], width, &codes))) return status;
    if ((status = flatten(args->t[code_index + 1], width / 128, &scales))) return status;
    if (separate && (status = flatten(args->t[1], width, &input_view))) return status;
    const bool have_output = args->n_t > required && args->t[required].data;
    if (have_output && (status = flatten(args->t[required], width, &output_view))) return status;
    const RadTensor* output = have_output ? &output_view : nullptr;
    const int64_t rows = codes.shape[0];
    if (rows < 1 || rows > (separate ? 65535 : 20480)) return RAD_E_SHAPE;
    const char* dtype = rad_args_gets(args, "dtype");
    if (dtype && std::strcmp(dtype, "bf16")) return RAD_E_DTYPE;
    const char* activation = rad_args_gets(args, "act");
    int mode = static_cast<int>(rad_args_geti_or(args, "mode", separate ? 1 : 0));
    if (activation) {
        if (!std::strcmp(activation, "silu")) mode = 0;
        else if (!std::strcmp(activation, "sigmoid")) mode = 1;
        else if (!std::strcmp(activation, "none")) mode = 2;
        else return RAD_E_UNSUPPORTED;
    }
    if (mode < 0 || mode > 2 || rad_args_geti_or(args, "chain", 0)) return RAD_E_UNSUPPORTED;
    status = matrix(gate, RAD_BF16, rows, separate ? width : width * 2);
    if (status) return status;
    status = matrix(codes, RAD_I8, rows, width);
    if (status) return status;
    status = matrix(scales, RAD_F32, rows, width / 128);
    if (status) return status;
    const RadTensor& input = separate ? input_view : gate;
    if (separate && (status = matrix(input, RAD_BF16, rows, width))) return status;
    if (output && (status = matrix(*output, RAD_BF16, rows, width))) return status;
    const auto* gate_data = static_cast<const uint16_t*>(gate.data);
    const auto* input_data = static_cast<const uint16_t*>(input.data) + (separate ? 0 : width);
    const dim3 grid(width / 128, rows);
    const auto native_stream = reinterpret_cast<hipStream_t>(stream);
    auto* output_data = output ? static_cast<uint16_t*>(output->data) : nullptr;
    if (separate)
        quantize<true><<<grid, 128, 0, native_stream>>>(gate_data, input_data,
            static_cast<int8_t*>(codes.data), static_cast<float*>(scales.data), output_data,
            mode, gate.stride[0], input.stride[0], codes.stride[0], scales.stride[0],
            output ? output->stride[0] : 0);
    else
        quantize<false><<<grid, 128, 0, native_stream>>>(gate_data, input_data,
            static_cast<int8_t*>(codes.data), static_cast<float*>(scales.data), output_data,
            mode, gate.stride[0], input.stride[0], codes.stride[0], scales.stride[0],
            output ? output->stride[0] : 0);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
}
extern "C" int r3dh_qwen_ops_gated_had_quant_i8g(const RadArgs* args, RadStream stream) {
    return launch(args, stream, false);
}
extern "C" int r3dh_qwen_ops_gate_had_quant_i8g(const RadArgs* args, RadStream stream) {
    return launch(args, stream, true);
}
