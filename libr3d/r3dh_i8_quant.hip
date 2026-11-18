#include <hip/hip_runtime.h>
#include "r3dh_i8.h"
#include <cmath>
#include <limits>
#ifdef __FAST_MATH__
#error r3dh_i8 requires strict floating point compilation
#endif
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)

namespace {
template<bool Rotate>
__global__ void quantize(const uint16_t* input, int8_t* codes, float* scale,
    int64_t input_pitch, int64_t code_pitch, int64_t scale_pitch) {
    __shared__ float values[128];
    __shared__ float maxima[128];
    const unsigned position = threadIdx.x;
    const int64_t row = blockIdx.y, group = blockIdx.x;
    float value = __uint_as_float(static_cast<uint32_t>(input[row * input_pitch + group * 128 + position]) << 16);
    values[position] = value;
    __syncthreads();
    if constexpr (Rotate) {
        for (unsigned distance = 1; distance < 128; distance <<= 1) {
            const float partner = values[position ^ distance];
            value = position & distance ? partner - value : value + partner;
            __syncthreads();
            values[position] = value;
            __syncthreads();
        }
    }
    maxima[position] = fabsf(value);
    __syncthreads();
    for (unsigned distance = 64; distance; distance >>= 1) {
        if (position < distance)
            maxima[position] = fmaxf(maxima[position], maxima[position + distance]);
        __syncthreads();
    }
    const float maximum = maxima[0] > 0.0f ? maxima[0] : 1e-8f;
    const float inverse = __fdiv_rn(127.0f, maximum);
    const float quantized = fmaxf(-127.0f, fminf(127.0f, value * inverse));
    codes[row * code_pitch + group * 128 + position] = static_cast<int8_t>(__float2int_rn(quantized));
    if (position == 0) scale[row * scale_pitch + group] = __fdiv_rn(maximum, 127.0f);
}

int launch(const RadArgs* arguments, RadStream stream, bool rotate) {
    if (!arguments || !arguments->t || arguments->n_t != 3 || arguments->n_p < 0
        || (arguments->n_p && !arguments->p)) return RAD_E_INVAL;
    RadTensor tensor[] = {arguments->t[0], arguments->t[1], arguments->t[2]};
    RadTensor& input = tensor[0];
    if (input.rank < 2 || input.rank > sizeof(input.shape) / sizeof(input.shape[0]))
        return RAD_E_SHAPE;
    int64_t folded = 1;
    for (int axis = input.rank - 1; axis > 0; --axis) {
        if (input.shape[axis] < 1 || input.shape[axis] > 2147483520 / folded)
            return RAD_E_SHAPE;
        if (input.stride[axis] != folded) return RAD_E_STRIDE;
        folded *= input.shape[axis];
    }
    const int64_t rows = input.shape[0];
    const int64_t band = rad_args_geti_or(arguments, "M", rows);
    const int64_t width = rad_args_geti_or(arguments, "n", folded);
    if (band < 1 || band > 2048 || rows < 1 || rows > band
        || width != folded || width < 128 || width > 2147483520 || width % 128
        || rad_args_geti_or(arguments, "group", 128) != 128) return RAD_E_SHAPE;
    input.rank = 2;
    input.shape[1] = folded;
    input.stride[1] = 1;
    const uint32_t types[] = {RAD_BF16, RAD_I8, RAD_F32};
    for (int index = 0; index < 3; ++index) {
        const RadTensor& view = tensor[index];
        const int64_t columns = index == 2 ? width / 128 : width;
        if (!view.data || view.dtype != types[index]) return RAD_E_DTYPE;
        if (view.rank != 2 || view.shape[0] != rows || view.shape[1] != columns)
            return RAD_E_SHAPE;
        if (view.stride[1] != 1 || view.stride[0] < columns
            || rows - 1 > (std::numeric_limits<int64_t>::max() - columns) / view.stride[0])
            return RAD_E_STRIDE;
        if (index != 1 && reinterpret_cast<uintptr_t>(view.data) % (index == 0 ? 2 : 4))
            return RAD_E_STRIDE;
    }
    const dim3 grid(width / 128, rows);
    const auto native_stream = reinterpret_cast<hipStream_t>(stream);
    if (rotate)
        quantize<true><<<grid, dim3(128), 0, native_stream>>>(
            static_cast<const uint16_t*>(tensor[0].data), static_cast<int8_t*>(tensor[1].data),
            static_cast<float*>(tensor[2].data), tensor[0].stride[0], tensor[1].stride[0], tensor[2].stride[0]);
    else
        quantize<false><<<grid, dim3(128), 0, native_stream>>>(
            static_cast<const uint16_t*>(tensor[0].data), static_cast<int8_t*>(tensor[1].data),
            static_cast<float*>(tensor[2].data), tensor[0].stride[0], tensor[1].stride[0], tensor[2].stride[0]);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
}

extern "C" int r3dh_i8_rad_quant_act_i8g(const RadArgs* arguments, RadStream stream) {
    return launch(arguments, stream, false);
}
extern "C" int r3dh_i8_rad_had_quant_act_i8g(const RadArgs* arguments, RadStream stream) {
    return launch(arguments, stream, true);
}
