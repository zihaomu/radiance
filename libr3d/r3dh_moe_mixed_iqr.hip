#include <hip/hip_runtime.h>
#include "r3dh_moe_mixed.h"
#include <cstring>
#include <limits>
#if defined(__FAST_MATH__)
#error r3dh_moe_iqr requires strict floating point and -ffp-contract=off
#endif

namespace {
using namespace r3dh_moe_iqr;
using Int4 = int __attribute__((ext_vector_type(4)));
using Int8 = int __attribute__((ext_vector_type(8)));
struct Table {
    const uint8_t* base;
    uint64_t entry_bytes;
};
struct Call {
    Config geometry;
    Table weight[2], scale[2];
    const int8_t* activation;
    const float* activation_scale;
    const int32_t* sorted;
    const int32_t* offsets;
    uint16_t* output;
    int64_t activation_pitch, scale_pitch, output_pitch;
};
__device__ const uint8_t* entry(Table table, int index) {
    return table.entry_bytes ? table.base + index * table.entry_bytes
        : static_cast<const uint8_t*>(reinterpret_cast<const void* const*>(table.base)[index]);
}
__device__ uint16_t bf16(float value) {
    const uint32_t word = __float_as_uint(value);
    if ((word & 0x7fffffffu) > 0x7f800000u) return static_cast<uint16_t>((word >> 16) | 64u);
    return static_cast<uint16_t>((word + 0x7fffu + ((word >> 16) & 1u)) >> 16);
}
__device__ int source(const Call& call, int row, int begin, int end) {
    if (row < begin || row >= end) return -1;
    const int slot = call.sorted[row];
    return slot >= 0 && slot < call.geometry.tokens * 10 ? slot / 10 : -1;
}
__global__ __launch_bounds__(32) void gate_up(Call call) {
    const unsigned lane = threadIdx.x;
    const int first = blockIdx.y * 16;
    const int total = call.geometry.tokens * 10;
    const int limit = first + 16 < total ? first + 16 : total;
    const int column = blockIdx.x * 16 + (lane & 15u);
    for (int begin = first; begin < limit;) {
        int lower = 0, upper = call.geometry.experts + 1;
        while (lower < upper) {
            const int middle = (lower + upper) / 2;
            if (call.offsets[middle] <= begin) lower = middle + 1;
            else upper = middle;
        }
        const int expert = lower - 1;
        const bool routed = expert >= 0 && expert < call.geometry.experts;
        const int boundary = routed ? call.offsets[expert + 1] : total;
        const int end = routed && boundary > begin && boundary < limit ? boundary : limit;
        const int parity = routed ? expert & 1 : 0;
        const int width = call.geometry.width[parity];
        const int part_width = call.geometry.output_width / 2;
        const int local = column % part_width;
        const int weight_row = (column / part_width) * (width / 2) + local;
        const bool live_column = column < call.geometry.output_width && local < width / 2;
        float accumulator[8] = {};
        const uint8_t* blob = routed ? entry(call.weight[parity], expert / 2) : nullptr;
        const auto* header = reinterpret_cast<const Header*>(blob);
        const auto* scales = routed ? reinterpret_cast<const float*>(entry(call.scale[parity], expert / 2)) : nullptr;
        for (int group = 0; routed && group < 20; ++group) {
            Int8 subtotal{};
            for (int step = 0; step < 128; step += 16) {
                Int4 activation_fragment{}, weight_fragment{};
                const int input_row = source(call, first + (lane & 15u), begin, end);
                for (unsigned word = 0; word < 4; ++word) {
                    uint32_t input_word = 0, weight_word = 0;
                    for (unsigned byte = 0; byte < 4; ++byte) {
                        const int position = group * 128 + step + word * 4 + byte;
                        const int8_t input = input_row >= 0
                            ? call.activation[input_row * call.activation_pitch + position] : 0;
                        const int8_t weight = live_column ? mad(state(blob + sizeof(Header), 2560,
                            weight_row, position), header->multiplier, header->increment) : 0;
                        input_word |= static_cast<uint32_t>(static_cast<uint8_t>(input)) << (byte * 8);
                        weight_word |= static_cast<uint32_t>(static_cast<uint8_t>(weight)) << (byte * 8);
                    }
                    activation_fragment[word] = static_cast<int>(input_word);
                    weight_fragment[word] = static_cast<int>(weight_word);
                }
                subtotal = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
                    true, activation_fragment, true, weight_fragment, subtotal, false);
            }
            for (unsigned index = 0; index < 8; ++index) {
                const int row = first + index * 2 + (lane >> 4);
                const int input_row = source(call, row, begin, end);
                const float factor = input_row >= 0
                    ? call.activation_scale[input_row * call.scale_pitch + group] : 0.0f;
                const float scaled = static_cast<float>(subtotal[index]) * factor;
                accumulator[index] = accumulator[index] + scaled * scales[width + group];
            }
        }
        for (unsigned index = 0; index < 8; ++index) {
            const int row = first + index * 2 + (lane >> 4);
            if (row >= begin && row < end && column < call.geometry.output_width) {
                const bool valid = routed && live_column && source(call, row, begin, end) >= 0;
                call.output[row * call.output_pitch + column]
                    = valid ? bf16(accumulator[index] * scales[weight_row]) : 0;
            }
        }
        begin = end;
    }
}
bool matrix(const RadTensor& tensor, uint32_t type, int rows, int columns) {
    return tensor.data && tensor.dtype == type && tensor.rank == 2
        && tensor.shape[0] == rows && tensor.shape[1] == columns
        && tensor.stride[0] >= columns && tensor.stride[0] <= (1ll << 40)
        && tensor.stride[1] == 1;
}
int table(const RadTensor& tensor, bool weight, int rows, int entries, Table* output) {
    if (!tensor.data || tensor.rank != 3 || tensor.shape[0] != entries
        || tensor.shape[1] != rows
        || (tensor.shape[2] != 2560 && (weight || tensor.shape[2] != 20))) return RAD_E_SHAPE;
    if (tensor.dtype != (weight ? weight_dtype : scale_dtype)) return RAD_E_DTYPE;
    const int64_t bytes = weight ? sizeof(Header) + rows * 1280 : (rows + 20) * sizeof(float);
    if (tensor.stride[0] < 0 || (tensor.stride[0] && tensor.stride[0] < bytes)
        || tensor.stride[1] != tensor.shape[2] || tensor.stride[2] != 1)
        return RAD_E_STRIDE;
    *output = Table{static_cast<const uint8_t*>(tensor.data), static_cast<uint64_t>(tensor.stride[0])};
    return RAD_OK;
}
}

extern "C" int r3dh_moe_mixed_iqr_launch(const RadArgs* arguments, RadStream stream) {
    // A tenth operand, `split`, marks an in-step checkpoint; every row here is computed alone, so it changes nothing.
    if (!arguments || (arguments->n_t != 9 && arguments->n_t != 10) || !arguments->t) return RAD_E_INVAL;
    Call call{};
    int status = r3dh_moe_mixed::config(arguments->p, arguments->n_p, &call.geometry);
    if (status != RAD_OK) return status;
    if (arguments->scratch) return r3dh_moe_iqr_launch(arguments, stream);
    const RadTensor* tensors = arguments->t;
    if (tensors[8].rank != 2 || tensors[8].shape[0] < 10
        || tensors[8].shape[0] % 10 || tensors[8].shape[0] / 10 > call.geometry.tokens)
        return RAD_E_SHAPE;
    call.geometry.tokens = static_cast<int>(tensors[8].shape[0] / 10);
    const Config& geometry = call.geometry;
    if (!matrix(tensors[0], RAD_I8, geometry.tokens, 2560)
        || !matrix(tensors[1], RAD_F32, geometry.tokens, 20)
        || !matrix(tensors[8], RAD_BF16, geometry.tokens * 10, geometry.output_width))
        return RAD_E_SHAPE;
    for (int operand : {4, 5}) {
        const RadTensor& tensor = tensors[operand];
        if (!tensor.data || tensor.dtype != RAD_I32 || tensor.rank != 1
            || tensor.shape[0] != (operand == 4 ? geometry.tokens * 10 : geometry.experts + 1)
            || tensor.stride[0] != 1) return RAD_E_SHAPE;
    }
    for (int parity = 0; parity < 2; ++parity) {
        status = table(tensors[2 + parity * 4], true, geometry.width[parity], geometry.experts / 2, &call.weight[parity]);
        if (status != RAD_OK) return status;
        status = table(tensors[3 + parity * 4], false, geometry.width[parity], geometry.experts / 2, &call.scale[parity]);
        if (status != RAD_OK) return status;
    }
    call.activation = static_cast<const int8_t*>(tensors[0].data);
    call.activation_scale = static_cast<const float*>(tensors[1].data);
    call.sorted = static_cast<const int32_t*>(tensors[4].data);
    call.offsets = static_cast<const int32_t*>(tensors[5].data);
    call.output = static_cast<uint16_t*>(tensors[8].data);
    call.activation_pitch = tensors[0].stride[0];
    call.scale_pitch = tensors[1].stride[0];
    call.output_pitch = tensors[8].stride[0];
    hipLaunchKernelGGL(gate_up,
        dim3((geometry.output_width + 15) / 16, (geometry.tokens * 10 + 15) / 16),
        dim3(32), 0, reinterpret_cast<hipStream_t>(stream), call);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
