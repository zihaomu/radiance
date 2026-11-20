#include <hip/hip_runtime.h>
#include "r3dh_moe_w4nl.h"
#include <cstdlib>
#include <cstring>
#include "r3d_flags.h"
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
    bool vec;
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
__device__ __forceinline__ int2 integer_codes(uint16_t codes) {
    const uint32_t indices = (codes & 15u) | ((codes & 240u) << 4)
        | ((codes & 3840u) << 8) | ((codes & 61440u) << 12);
    const uint32_t magnitude = indices & 0x07070707u;
    const uint32_t selector = 0x03020100u + ((indices & 0x08080808u) >> 1);
    const uint32_t positive = __builtin_amdgcn_perm(0x5848382cu, 0x20160d04u, magnitude);
    const uint32_t negative = __builtin_amdgcn_perm(0xa8b8c8d4u, 0xe0eaf3fcu, magnitude);
    const uint32_t residual_positive = __builtin_amdgcn_perm(0u, 1u, magnitude);
    const uint32_t residual_negative = __builtin_amdgcn_perm(0u, 255u, magnitude);
    return make_int2(int(__builtin_amdgcn_perm(negative, positive, selector)),
        int(__builtin_amdgcn_perm(residual_negative, residual_positive, selector)));
}
__global__ __launch_bounds__(128) void down_wave(Call call) {
    const int column = int(blockIdx.x) * 32 + int(threadIdx.x) / 4;
    const int component = int(threadIdx.x) & 3;
    const int row = int(blockIdx.y);
    const Config geometry = call.geometry;
    const int rows = geometry.tokens * geometry.top_k;
    if (column >= geometry.columns) return;
    const auto destination = uint64_t(row) * call.output_stride + column;
    const int flat = call.sorted[row];
    if (flat < 0 || flat >= rows || call.offsets[0] != 0
        || row >= call.offsets[geometry.experts] || call.offsets[geometry.experts] > rows) {
        if (!component) call.output[destination] = 0;
        return;
    }
    int lower = 0, upper = geometry.experts;
    while (lower + 1 < upper) {
        const int middle = (lower + upper) / 2;
        if (call.offsets[middle] <= row) lower = middle; else upper = middle;
    }
    const int parity = lower & 1, inner = geometry.inner[parity];
    const auto* code_entry = entry(call.codes[parity], lower / 2);
    const auto* scale_entry = entry(call.scales[parity], lower / 2);
    if (!code_entry || !scale_entry || call.offsets[lower] > row
        || row >= call.offsets[lower + 1]) {
        if (!component) call.output[destination] = 0;
        return;
    }
    const auto* code_header = reinterpret_cast<const CodesHeader*>(code_entry);
    const auto* scale_header = reinterpret_cast<const ScalesHeader*>(scale_entry);
    if (code_header->magic != codes_magic || code_header->version != 1
        || code_header->rows != uint32_t(geometry.columns) || code_header->columns != uint32_t(inner)
        || scale_header->magic != scales_magic || scale_header->version != 1
        || scale_header->rows != uint32_t(geometry.columns) || scale_header->columns != uint32_t(inner)
        || scale_header->level != 0x1p-13f) {
        if (!component) call.output[destination] = 0;
        return;
    }
    int canonical = 1;
    if (!component)
        for (int index = 0; index < 16; ++index) {
            const int magnitude = int((0x5848382c20160d04ull >> ((index & 7) * 8)) & 255u);
            const int residual = (index & 7) == 0 ? 1 : 0;
            canonical &= code_header->table[index]
                == float((index & 8 ? -1 : 1) * (2 * magnitude + residual)) * 0.125f;
        }
    canonical = __shfl(canonical, 0, 4);
    if (!canonical && component) return;
    const uint8_t* codes = code_entry + sizeof(CodesHeader) + uint64_t(column) * (inner / 2);
    const uint8_t* scales = scale_entry + sizeof(ScalesHeader) + uint64_t(column) * (inner / 64);
    const int8_t* activation = call.activation + uint64_t(row) * call.activation_stride;
    const float* activation_scale = call.activation_scale + uint64_t(row) * call.scale_stride;
    float accumulator = 0;
    for (int group = 0; group < inner / 64; ++group) {
        float subtotal = 0;
        if (canonical) {
            int dot = 0;
#pragma unroll
            for (int element = 0; element < 4; ++element) {
                const int position = group * 64 + component * 16 + element * 4;
                uint32_t input = 0;
                uint16_t packed_codes = 0;
                __builtin_memcpy(&input, activation + position, sizeof(input));
                __builtin_memcpy(&packed_codes, codes + position / 2, sizeof(packed_codes));
                const int2 weights = integer_codes(packed_codes);
                const int first = __builtin_amdgcn_sudot4(true, weights.x, true, int(input), 0, false);
                const int second = __builtin_amdgcn_sudot4(true, weights.y, true, int(input), 0, false);
                dot += 2 * first + second;
            }
            dot += __shfl_xor(dot, 1, 4);
            dot += __shfl_xor(dot, 2, 4);
            subtotal = float(dot) * 0.125f;
        } else {
            for (int offset = 0; offset < 64; ++offset) {
                const int position = group * 64 + offset;
                const unsigned code = (codes[position / 2] >> ((position & 1) * 4)) & 15u;
                subtotal = subtotal + float(activation[position]) * code_header->table[code];
            }
        }
        if (!component) {
            const float weight_scale = scale_value(scales[group]) * scale_header->level;
            const float first = subtotal * activation_scale[group / 2];
            accumulator = accumulator + first * weight_scale;
        }
    }
    if (!component) call.output[destination] = bf16(accumulator);
}
constexpr int run_rows = 8;
// down_wave's arithmetic per row; the first row of each 8-row chunk of an expert's run serves the chunk,
// so the expert's codes are read once per chunk instead of once per routed row.
__global__ __launch_bounds__(128) void down_run(Call call) {
    const int column = int(blockIdx.x) * 32 + int(threadIdx.x) / 4;
    const int component = int(threadIdx.x) & 3;
    const int row = int(blockIdx.y);
    const Config geometry = call.geometry;
    const int rows = geometry.tokens * geometry.top_k;
    if (column >= geometry.columns) return;
    if (call.offsets[0] != 0 || row >= call.offsets[geometry.experts] || call.offsets[geometry.experts] > rows) {
        if (!component) call.output[uint64_t(row) * call.output_stride + column] = 0;
        return;
    }
    int lower = 0, upper = geometry.experts;
    while (lower + 1 < upper) {
        const int middle = (lower + upper) / 2;
        if (call.offsets[middle] <= row) lower = middle; else upper = middle;
    }
    const int begin = call.offsets[lower], end = call.offsets[lower + 1];
    if (begin > row || row >= end) {
        if (!component) call.output[uint64_t(row) * call.output_stride + column] = 0;
        return;
    }
    if ((row - begin) % run_rows) return;
    const int count = end - row < run_rows ? end - row : run_rows;
    const int parity = lower & 1, inner = geometry.inner[parity];
    const auto* code_entry = entry(call.codes[parity], lower / 2);
    const auto* scale_entry = entry(call.scales[parity], lower / 2);
    const auto* code_header = reinterpret_cast<const CodesHeader*>(code_entry);
    const auto* scale_header = reinterpret_cast<const ScalesHeader*>(scale_entry);
    if (!code_entry || !scale_entry || code_header->magic != codes_magic || code_header->version != 1
        || code_header->rows != uint32_t(geometry.columns) || code_header->columns != uint32_t(inner)
        || scale_header->magic != scales_magic || scale_header->version != 1
        || scale_header->rows != uint32_t(geometry.columns) || scale_header->columns != uint32_t(inner)
        || scale_header->level != 0x1p-13f) {
        if (!component)
            for (int index = 0; index < count; ++index)
                call.output[uint64_t(row + index) * call.output_stride + column] = 0;
        return;
    }
    int canonical = 1;
    if (!component)
        for (int index = 0; index < 16; ++index) {
            const int magnitude = int((0x5848382c20160d04ull >> ((index & 7) * 8)) & 255u);
            const int residual = (index & 7) == 0 ? 1 : 0;
            canonical &= code_header->table[index]
                == float((index & 8 ? -1 : 1) * (2 * magnitude + residual)) * 0.125f;
        }
    canonical = __shfl(canonical, 0, 4);
    if (!canonical && component) return;
    const uint8_t* codes = code_entry + sizeof(CodesHeader) + uint64_t(column) * (inner / 2);
    const uint8_t* scales = scale_entry + sizeof(ScalesHeader) + uint64_t(column) * (inner / 64);
    bool live[run_rows];
#pragma unroll
    for (int index = 0; index < run_rows; ++index) {
        const int flat = index < count ? call.sorted[row + index] : -1;
        live[index] = flat >= 0 && flat < rows;
    }
    float accumulator[run_rows] = {};
    // Same bytes as the per-element loads: every group's codes are issued at once, then read back in order.
    __shared__ uint2 hoisted[16][128];
    const bool vec = call.vec && canonical && inner / 64 <= 16 && !(reinterpret_cast<uintptr_t>(codes) & 7);
    if (vec) {
#pragma unroll
        for (int group = 0; group < 16; ++group)
            if (group < inner / 64)
                hoisted[group][threadIdx.x] = *reinterpret_cast<const uint2*>(codes + group * 32 + component * 8);
    }
    for (int group = 0; group < inner / 64; ++group) {
        float subtotal[run_rows] = {};
        if (canonical) {
            int2 weights[4];
            if (vec) {
                const uint2 packed = hoisted[group][threadIdx.x];
                weights[0] = integer_codes(uint16_t(packed.x)); weights[1] = integer_codes(uint16_t(packed.x >> 16));
                weights[2] = integer_codes(uint16_t(packed.y)); weights[3] = integer_codes(uint16_t(packed.y >> 16));
            } else {
#pragma unroll
                for (int element = 0; element < 4; ++element) {
                    uint16_t packed_codes = 0;
                    __builtin_memcpy(&packed_codes, codes + (group * 64 + component * 16 + element * 4) / 2,
                        sizeof(packed_codes));
                    weights[element] = integer_codes(packed_codes);
                }
            }
#pragma unroll
            for (int index = 0; index < run_rows; ++index) {
                if (index >= count) break;
                const int8_t* activation = call.activation + uint64_t(row + index) * call.activation_stride;
                int dot = 0;
                uint4 inputs{};
                if (vec) inputs = *reinterpret_cast<const uint4*>(activation + group * 64 + component * 16);
#pragma unroll
                for (int element = 0; element < 4; ++element) {
                    uint32_t input = 0;
                    if (vec) input = element == 0 ? inputs.x : element == 1 ? inputs.y : element == 2 ? inputs.z : inputs.w;
                    else __builtin_memcpy(&input, activation + group * 64 + component * 16 + element * 4, sizeof(input));
                    const int first = __builtin_amdgcn_sudot4(true, weights[element].x, true, int(input), 0, false);
                    const int second = __builtin_amdgcn_sudot4(true, weights[element].y, true, int(input), 0, false);
                    dot += 2 * first + second;
                }
                dot += __shfl_xor(dot, 1, 4);
                dot += __shfl_xor(dot, 2, 4);
                subtotal[index] = float(dot) * 0.125f;
            }
        } else {
#pragma unroll
            for (int index = 0; index < run_rows; ++index) {
                if (index >= count) break;
                const int8_t* activation = call.activation + uint64_t(row + index) * call.activation_stride;
                for (int offset = 0; offset < 64; ++offset) {
                    const int position = group * 64 + offset;
                    const unsigned code = (codes[position / 2] >> ((position & 1) * 4)) & 15u;
                    subtotal[index] = subtotal[index] + float(activation[position]) * code_header->table[code];
                }
            }
        }
        if (!component) {
            const float weight_scale = scale_value(scales[group]) * scale_header->level;
#pragma unroll
            for (int index = 0; index < run_rows; ++index) {
                if (index >= count) break;
                const float* activation_scale = call.activation_scale + uint64_t(row + index) * call.scale_stride;
                const float first = subtotal[index] * activation_scale[group / 2];
                accumulator[index] = accumulator[index] + first * weight_scale;
            }
        }
    }
    if (!component)
#pragma unroll
        for (int index = 0; index < run_rows; ++index) {
            if (index >= count) break;
            call.output[uint64_t(row + index) * call.output_stride + column] = live[index] ? bf16(accumulator[index]) : 0;
        }
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
    // A tenth operand, `split`, marks an in-step checkpoint; every row here is computed alone, so it changes nothing.
    if (!arguments || !arguments->t || (arguments->n_t != 9 && arguments->n_t != 10)) return RAD_E_INVAL;
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
    const bool dot4 = r3d_exact_path("RAD_W4NL_DOT4");
    const bool run = r3d_exact_path("RAD_W4NL_RUN");
    const bool vec = r3d_exact_path("RAD_W4NL_VEC");
    call.vec = vec && call.activation_stride % 16 == 0
        && !(reinterpret_cast<uintptr_t>(call.activation) & 15);
    if (!geometry.gate && run)
        hipLaunchKernelGGL(down_run, dim3((geometry.columns + 31) / 32, rows), dim3(128),
            0, reinterpret_cast<hipStream_t>(stream), call);
    else if (!geometry.gate && dot4)
        hipLaunchKernelGGL(down_wave, dim3((geometry.columns + 31) / 32, rows), dim3(128),
            0, reinterpret_cast<hipStream_t>(stream), call);
    else
        hipLaunchKernelGGL(down, dim3((geometry.columns + 63) / 64, (rows + 3) / 4), dim3(256),
            0, reinterpret_cast<hipStream_t>(stream), call);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
