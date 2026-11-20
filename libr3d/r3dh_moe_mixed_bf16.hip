#include <hip/hip_runtime.h>
#include "r3dh_moe_mixed.h"
#include <cstring>
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__)
#error Mixed BF16 requires strict floating point
#endif
namespace {
struct Geometry { int tokens, columns, inner, sorted, experts; };
int geometry(const RadParam* params, int count, Geometry* output) {
    if (!output || count < 0 || (count && !params)) return RAD_E_INVAL;
    if (std::strcmp(rad_param_gets(params, count, "dtype", ""), "bf16"))
        return RAD_E_DTYPE;
    const char* order = rad_param_gets(params, count, "a_order", "");
    const bool sorted = !std::strcmp(order, "sorted");
    if (!sorted && std::strcmp(order, "token")) return RAD_E_UNSUPPORTED;
    const auto tokens = rad_param_geti(params, count, "M", 0);
    const auto columns = rad_param_geti(params, count, "N", 0);
    const auto inner = rad_param_geti(params, count, "K", 0);
    if (tokens < 1 || tokens > 2048 || (rad_param_geti(params, count, "n_expert", 0) != 2 && rad_param_geti(params, count, "n_expert", 0) != 4)
        || rad_param_geti(params, count, "top_k", 0) != 10
        || rad_param_geti(params, count, "parts", sorted ? 1 : 2) != (sorted ? 1 : 2)
        || (sorted ? columns != 2560 || (inner != 320 && inner != 640)
                   : (columns != 640 && columns != 1280) || inner != 2560))
        return RAD_E_SHAPE;
    *output = Geometry{int(tokens), int(columns), int(inner), int(sorted), int(rad_param_geti(params, count, "n_expert", 0))};
    return RAD_OK;
}
bool matrix(const RadTensor& value, uint32_t type, int rows, int columns) {
    return value.data && value.dtype == type && value.rank == 2
        && value.shape[0] == rows && value.shape[1] == columns
        && value.stride[0] >= columns && value.stride[0] <= (1ll << 30)
        && value.stride[1] == 1;
}
__device__ float unpack(uint16_t value) { return __uint_as_float(uint32_t(value) << 16); }
__device__ uint16_t pack(float value) {
    const uint32_t bits = __float_as_uint(value);
    if ((bits & 0x7fffffffu) > 0x7f800000u) return uint16_t((bits >> 16) | 64u);
    return uint16_t((bits + 32767u + ((bits >> 16) & 1u)) >> 16);
}
__global__ void protected_gemm(Geometry dims, const uint16_t* input, int64_t input_pitch,
    const uint16_t* table, int64_t entry_bytes, const int32_t* sorted,
    const int32_t* offsets, uint16_t* output, int64_t output_pitch) {
    const int row = blockIdx.y, column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= dims.columns || row < offsets[0] || row >= offsets[dims.experts]
        || offsets[0] < 0 || offsets[dims.experts] > dims.tokens * 10) return;
    int expert = -1;
    for (int index = 0; index < dims.experts; ++index)
        if (offsets[index] <= row && row < offsets[index + 1]) expert = index;
    if (expert < 0) return;
    const int flat = sorted[row];
    const int source = dims.sorted ? row : flat / 10;
    float sum = 0.0f;
    if (flat >= 0 && flat < dims.tokens * 10) {
        const auto* weight = entry_bytes
            ? reinterpret_cast<const uint16_t*>(reinterpret_cast<const uint8_t*>(table)
                + int64_t(expert) * entry_bytes)
            : reinterpret_cast<const uint16_t* const*>(table)[expert];
        for (int inner = 0; inner < dims.inner; ++inner)
            sum = fmaf(unpack(input[int64_t(source) * input_pitch + inner]),
                unpack(weight[int64_t(column) * dims.inner + inner]), sum);
    }
    output[int64_t(row) * output_pitch + column] = pack(sum);
}
}
extern "C" int r3dh_moe_mixed_bf16_shape(const RadParam* params, int count,
                                          int operand, RadOpdDesc* output) {
    if (!output || operand < 0 || operand > 4) return RAD_E_INVAL;
    Geometry dims{};
    int status = geometry(params, count, &dims);
    if (status != RAD_OK) return status;
    *output = RadOpdDesc{}; output->idx_const = -1;
    output->dtype = operand == 2 || operand == 3 ? RAD_I32 : RAD_BF16;
    output->rank = operand == 2 || operand == 3 ? 1 : 2;
    output->shape[0] = operand == 0 ? dims.tokens * (dims.sorted ? 10 : 1)
        : operand == 1 ? dims.columns : operand == 3 ? dims.experts + 1 : dims.tokens * 10;
    output->shape[1] = operand == 4 ? dims.columns : dims.inner;
    return RAD_OK;
}
extern "C" int r3dh_moe_mixed_bf16_launch(const RadArgs* args, RadStream stream) {
    if (!args || args->n_t != 5 || !args->t) return RAD_E_INVAL;
    Geometry dims{};
    int status = geometry(args->p, args->n_p, &dims);
    if (status != RAD_OK) return status;
    const RadTensor* tensors = args->t;
    if (tensors[4].rank != 2 || tensors[4].shape[0] < 10
        || tensors[4].shape[0] % 10 || tensors[4].shape[0] / 10 > dims.tokens)
        return RAD_E_SHAPE;
    dims.tokens = static_cast<int>(tensors[4].shape[0] / 10);
    if (!matrix(tensors[0], RAD_BF16, dims.tokens * (dims.sorted ? 10 : 1), dims.inner)
        || !matrix(tensors[4], RAD_BF16, dims.tokens * 10, dims.columns)) return RAD_E_SHAPE;
    for (int operand : {2, 3}) {
        const auto& value = tensors[operand];
        if (!value.data || value.dtype != RAD_I32 || value.rank != 1
            || value.shape[0] != (operand == 2 ? dims.tokens * 10 : dims.experts + 1)
            || value.stride[0] != 1) return RAD_E_SHAPE;
    }
    const auto& weight = tensors[1];
    const int64_t bytes = int64_t(dims.columns) * dims.inner * 2;
    if (!weight.data || weight.dtype != RAD_BF16 || weight.rank != 3
        || weight.shape[0] != dims.experts || weight.shape[1] != dims.columns
        || weight.shape[2] != dims.inner) return RAD_E_SHAPE;
    if (weight.stride[0] < 0 || (weight.stride[0] && weight.stride[0] < bytes)
        || weight.stride[1] != dims.inner || weight.stride[2] != 1) return RAD_E_STRIDE;
    hipLaunchKernelGGL(protected_gemm, dim3((dims.columns + 63) / 64, dims.tokens * 10),
        dim3(64), 0, reinterpret_cast<hipStream_t>(stream), dims,
        static_cast<const uint16_t*>(tensors[0].data), tensors[0].stride[0],
        static_cast<const uint16_t*>(weight.data), weight.stride[0],
        static_cast<const int32_t*>(tensors[2].data), static_cast<const int32_t*>(tensors[3].data),
        static_cast<uint16_t*>(tensors[4].data), tensors[4].stride[0]);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
