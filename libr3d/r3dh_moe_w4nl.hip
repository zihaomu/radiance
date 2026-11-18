#include <hip/hip_runtime.h>
#include "r3dh_moe_w4nl.h"
#include <cstring>
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__)
#error W4NL64 requires -ffp-contract=off -fno-fast-math
#endif
#pragma clang fp contract(off)

namespace {
using namespace r3dh_moe_w4nl;
struct Table {
    const void* data;
    uint64_t stride;
};
struct Call {
    const int8_t* activation;
    const float* activation_scale;
    const int32_t* sorted;
    const int32_t* offsets;
    uint16_t* output;
    Table codes[2], scales[2];
    Config geometry;
    int activation_stride, scale_stride, output_stride;
};
__device__ const uint8_t* entry(Table table, int index) {
    return table.stride ? static_cast<const uint8_t*>(table.data) + index * table.stride
        : static_cast<const uint8_t*>(static_cast<const void* const*>(table.data)[index]);
}
__device__ float scale_value(uint8_t raw) {
    const uint32_t exponent = (raw >> 3) & 15;
    const uint32_t mantissa = raw & 7;
    const float positive = exponent ? __builtin_bit_cast(float, ((exponent + 120) << 23)
        | (mantissa << 20)) : float(mantissa) * 0x1p-9f;
    return raw & 128 ? -positive : positive;
}
__device__ uint16_t bf16(float value) {
    uint32_t bits = __builtin_bit_cast(uint32_t, value);
    if ((bits & 0x7fffffffu) > 0x7f800000u) return uint16_t((bits >> 16) | 64u);
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return uint16_t(bits >> 16);
}
__global__ __launch_bounds__(256) void down(Call call) {
    const int column = int(blockIdx.x) * 64 + int(threadIdx.x) % 64;
    const int row = int(blockIdx.y) * 4 + int(threadIdx.x) / 64;
    const Config geometry = call.geometry;
    const int rows = geometry.tokens * geometry.top_k;
    if (column >= geometry.columns || row >= rows) return;
    uint16_t result = 0;
    const int flat = call.sorted[row];
    if (flat >= 0 && flat < rows && call.offsets[0] == 0
        && row < call.offsets[geometry.experts] && call.offsets[geometry.experts] <= rows) {
        int lower = 0, upper = geometry.experts;
        while (lower + 1 < upper) {
            const int middle = (lower + upper) / 2;
            if (call.offsets[middle] <= row) lower = middle;
            else upper = middle;
        }
        const int expert = lower;
        const int parity = expert & 1;
        const int inner = geometry.inner[parity];
        const int columns = geometry.gate ? geometry.gate_columns[parity] : geometry.columns;
        const int half = geometry.columns / 2;
        const int weight_column = geometry.gate
            ? (column / half) * (columns / 2) + column % half : column;
        const uint8_t* code_entry = entry(call.codes[parity], expert / 2);
        const uint8_t* scale_entry = entry(call.scales[parity], expert / 2);
        if ((geometry.gate ? column % half < columns / 2 : column < columns)
            && code_entry && scale_entry && call.offsets[expert] <= row
            && row < call.offsets[expert + 1]) {
            const auto* code_header = reinterpret_cast<const CodesHeader*>(code_entry);
            const auto* scale_header = reinterpret_cast<const ScalesHeader*>(scale_entry);
            if (code_header->magic == codes_magic && code_header->version == 1
                && code_header->rows == uint32_t(columns)
                && code_header->columns == uint32_t(inner)
                && scale_header->magic == scales_magic && scale_header->version == 1
                && scale_header->rows == uint32_t(columns)
                && scale_header->columns == uint32_t(inner)
                && scale_header->level == 0x1p-13f) {
                const uint8_t* codes = code_entry + sizeof(CodesHeader)
                    + uint64_t(weight_column) * (inner / 2);
                const uint8_t* scales = scale_entry + sizeof(ScalesHeader)
                    + uint64_t(weight_column) * (inner / 64);
                const int source = geometry.gate ? flat / geometry.top_k : row;
                const int8_t* activation = call.activation + uint64_t(source) * call.activation_stride;
                const float* activation_scale = call.activation_scale
                    + uint64_t(source) * call.scale_stride;
                float accumulator = 0.0f;
                for (int group = 0; group < inner / 64; ++group) {
                    float subtotal = 0.0f;
                    for (int offset = 0; offset < 64; ++offset) {
                        const int position = group * 64 + offset;
                        const uint32_t code = (codes[position / 2] >> ((position & 1) * 4)) & 15;
                        const float product = float(activation[position]) * code_header->table[code];
                        subtotal = subtotal + product;
                    }
                    const float weight_scale = scale_value(scales[group]) * scale_header->level;
                    const float first = subtotal * activation_scale[group / 2];
                    const float second = first * weight_scale;
                    accumulator = accumulator + second;
                }
                result = bf16(accumulator);
            }
        }
    }
    call.output[uint64_t(row) * call.output_stride + column] = result;
}
int table(const RadTensor& value, uint32_t dtype, int entries, int columns, int inner,
          int64_t bytes, Table* output) {
    if (!value.data) return RAD_E_INVAL;
    if (value.dtype != dtype) return RAD_E_DTYPE;
    const bool serialized = value.rank == 2 && value.shape[1] == bytes;
    const bool logical = value.rank == 3 && value.shape[1] == columns
        && value.shape[2] == inner;
    if (value.shape[0] != entries || (!serialized && !logical))
        return RAD_E_SHAPE;
    if (value.stride[1] != (logical ? inner : 1)
        || (logical && value.stride[2] != 1) || value.stride[0] < 0
        || (value.stride[0] && (value.stride[0] < bytes || value.stride[0] % 16)))
        return RAD_E_STRIDE;
    *output = Table{value.data, uint64_t(value.stride[0])};
    return RAD_OK;
}
int matrix(const RadTensor& value, uint32_t dtype, int rows, int columns) {
    if (!value.data) return RAD_E_INVAL;
    if (value.dtype != dtype) return RAD_E_DTYPE;
    if (value.rank != 2 || value.shape[0] != rows || value.shape[1] != columns)
        return RAD_E_SHAPE;
    if (value.stride[1] != 1 || value.stride[0] < columns || value.stride[0] > INT32_MAX)
        return RAD_E_STRIDE;
    return RAD_OK;
}
}
extern "C" int r3dh_moe_w4nl_launch(const RadArgs* arguments, RadStream stream) {
    if (!arguments || !arguments->t || arguments->n_t != 9) return RAD_E_INVAL;
    Call call{};
    int status = config(arguments->p, arguments->n_p, &call.geometry);
    if (status != RAD_OK) return status;
    const RadTensor* tensors = arguments->t;
    if (tensors[8].rank != 2 || tensors[8].shape[0] < call.geometry.top_k
        || tensors[8].shape[0] % call.geometry.top_k
        || tensors[8].shape[0] / call.geometry.top_k > call.geometry.tokens) return RAD_E_SHAPE;
    call.geometry.tokens = static_cast<int>(tensors[8].shape[0] / call.geometry.top_k);
    const Config geometry = call.geometry;
    const int rows = geometry.tokens * geometry.top_k;
    const int activation_rows = geometry.gate ? geometry.tokens : rows;
    status = matrix(tensors[0], RAD_I8, activation_rows, geometry.activation_columns);
    if (status != RAD_OK) return status;
    status = matrix(tensors[1], RAD_F32, activation_rows, geometry.activation_columns / 128);
    if (status != RAD_OK) return status;
    status = matrix(tensors[8], RAD_BF16, rows, geometry.columns);
    if (status != RAD_OK) return status;
    for (int operand : {4, 5}) {
        const RadTensor& value = tensors[operand];
        if (!value.data) return RAD_E_INVAL;
        if (value.dtype != RAD_I32) return RAD_E_DTYPE;
        if (value.rank != 1 || value.shape[0] != (operand == 4 ? rows : geometry.experts + 1))
            return RAD_E_SHAPE;
        if (value.stride[0] != 1) return RAD_E_STRIDE;
    }
    for (int parity = 0; parity < 2; ++parity) {
        const int entries = (geometry.experts + 1 - parity) / 2;
        if (!entries) continue;
        const int inner = geometry.inner[parity];
        const int columns = geometry.gate ? geometry.gate_columns[parity] : geometry.columns;
        const int operand = parity ? 6 : 2;
        status = table(tensors[operand], codes_type, entries, columns, inner,
            sizeof(CodesHeader) + int64_t(columns) * inner / 2, &call.codes[parity]);
        if (status != RAD_OK) return status;
        status = table(tensors[operand + 1], scales_type, entries, columns, inner,
            sizeof(ScalesHeader) + int64_t(columns) * (inner / 64), &call.scales[parity]);
        if (status != RAD_OK) return status;
    }
    int device = 0;
    hipDeviceProp_t properties{};
    if (hipGetDevice(&device) != hipSuccess || hipGetDeviceProperties(&properties, device) != hipSuccess)
        return RAD_E_DEVICE;
    if (std::strncmp(properties.gcnArchName, "gfx1100", 7)) return RAD_E_UNSUPPORTED;
    call.activation = static_cast<const int8_t*>(tensors[0].data);
    call.activation_scale = static_cast<const float*>(tensors[1].data);
    call.sorted = static_cast<const int32_t*>(tensors[4].data);
    call.offsets = static_cast<const int32_t*>(tensors[5].data);
    call.output = static_cast<uint16_t*>(tensors[8].data);
    call.activation_stride = int(tensors[0].stride[0]);
    call.scale_stride = int(tensors[1].stride[0]);
    call.output_stride = int(tensors[8].stride[0]);
    hipLaunchKernelGGL(down, dim3((geometry.columns + 63) / 64, (rows + 3) / 4), dim3(256),
        0, reinterpret_cast<hipStream_t>(stream), call);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
