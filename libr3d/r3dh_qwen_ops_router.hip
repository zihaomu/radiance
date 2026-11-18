#include "r3dh_qwen_ops.h"
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstring>
#include <limits>
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)

namespace {
__global__ void project(const uint16_t* input, const uint16_t* weight, uint16_t* logits,
                        int64_t input_pitch, int64_t weight_pitch, int64_t output_pitch,
                        int columns, int rows) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y;
    if (column >= columns || row >= rows) return;
    float sum = 0.0f;
    for (int position = 0; position < 2560; ++position) {
        const float left = __uint_as_float(static_cast<uint32_t>(input[row * input_pitch + position]) << 16);
        const float right = __uint_as_float(static_cast<uint32_t>(weight[column * weight_pitch + position]) << 16);
        const float product = left * right;
        sum = sum + product;
    }
    const uint32_t bits = __float_as_uint(sum);
    logits[row * output_pitch + column] = (bits & 0x7fffffffu) > 0x7f800000u
        ? (bits >> 16) | 0x40u : (bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16;
}
}
extern "C" int r3dh_qwen_ops_router_gemm(const RadArgs* args, RadStream stream) {
    if (!args || !args->t || args->n_t != 3 || args->n_p < 0
        || (args->n_p && !args->p)) return RAD_E_INVAL;
    const int64_t rows = args->t[0].shape[0];
    const int64_t columns = rad_args_geti_or(args, "N", 512);
    const int64_t band = rad_args_geti_or(args, "M", rows);
    if (rows < 1 || rows > 2048 || columns < 1 || columns > 512
        || rad_args_geti_or(args, "K", 2560) != 2560
        || band < rows || band > 2048) return RAD_E_SHAPE;
    const char* dtype = rad_args_gets(args, "dtype");
    if (dtype && std::strcmp(dtype, "bf16")) return RAD_E_DTYPE;
    for (int index = 0; index < 3; ++index) {
        const RadTensor& tensor = args->t[index];
        if (!tensor.data) return RAD_E_INVAL;
        if (tensor.dtype != RAD_BF16) return RAD_E_DTYPE;
        const int64_t height = index == 1 ? columns : rows;
        const int64_t width = index == 2 ? columns : 2560;
        if (tensor.rank != 2 || tensor.shape[0] != height || tensor.shape[1] != width)
            return RAD_E_SHAPE;
        if (tensor.stride[1] != 1 || tensor.stride[0] < width
            || tensor.stride[0] > (std::numeric_limits<int64_t>::max() / 2 - width) / height
            || reinterpret_cast<uintptr_t>(tensor.data) % 2) return RAD_E_STRIDE;
    }
    project<<<dim3((columns + 127) / 128, rows), 128, 0, reinterpret_cast<hipStream_t>(stream)>>>(
        static_cast<const uint16_t*>(args->t[0].data), static_cast<const uint16_t*>(args->t[1].data),
        static_cast<uint16_t*>(args->t[2].data), args->t[0].stride[0], args->t[1].stride[0],
        args->t[2].stride[0], columns, rows);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
