#include "r3d_i8_gated.h"
#include <hip/hip_runtime.h>

#include "r3d_iqr.h"
#include "rad_plugin.h"

#include <cstring>
#include <cstdlib>
#include <limits>

namespace {

__device__ uint16_t narrow_bf16(float value) {
    const uint32_t word = __float_as_uint(value);
    if ((word & 0x7fffffffu) > 0x7f800000u) return static_cast<uint16_t>((word >> 16) | 0x40u);
    return static_cast<uint16_t>((word + 0x7fffu + ((word >> 16) & 1u)) >> 16);
}

__global__ void iqr_scalar(const int8_t* activation, const float* activation_scale,
    const uint8_t* weight, const float* scales, uint16_t* output,
    int rows, int columns, int reduction, int64_t activation_pitch,
    int64_t activation_scale_pitch, int64_t output_pitch) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y;
    if (column >= columns || row >= rows) return;
    const auto* header = reinterpret_cast<const r3d_iqr::Header*>(weight);
    const uint8_t* codes = weight + sizeof(r3d_iqr::Header);
    const uint64_t code_pitch = static_cast<uint64_t>(reduction) * header->bits / 8;
    const float* column_scale = scales + columns;
    float accumulator = 0.0f;
    for (int group = 0; group < reduction / 128; ++group) {
        int32_t subtotal = 0;
        for (int offset = 0; offset < 128; ++offset) {
            const int position = group * 128 + offset;
            const uint32_t state = r3d_iqr::trellis_state(codes, code_pitch, column,
                position, header->bits, header->state_bits, header->run_rows);
            const int code = r3d_iqr::trellis_mad(state, header->multiplier, header->increment);
            subtotal += static_cast<int>(activation[row * activation_pitch + position]) * code;
        }
        const float scaled = static_cast<float>(subtotal)
                           * activation_scale[row * activation_scale_pitch + group];
        accumulator += scaled * column_scale[group];
    }
    output[row * output_pitch + column] = narrow_bf16(accumulator * scales[column]);
}

}

using Int4 = int __attribute__((ext_vector_type(4)));
using Int8 = int __attribute__((ext_vector_type(8)));

template<bool Packed>
__global__ void iqr_wave(const int8_t* activation, const float* activation_scale,
    const uint8_t* weight, const float* scales, uint16_t* output,
    int rows, int columns, int reduction, int64_t activation_pitch,
    int64_t activation_scale_pitch, int64_t output_pitch) {
    const unsigned lane = threadIdx.x & 31u;
    const int column = blockIdx.x * 4 + threadIdx.x / 32;
    const int row = blockIdx.y;
    if (column >= columns || row >= rows) return;
    const auto* header = reinterpret_cast<const r3d_iqr::Header*>(weight);
    const uint8_t* codes = weight + sizeof(r3d_iqr::Header);
    const uint64_t pitch = static_cast<uint64_t>(reduction) * header->bits / 8;
    float accumulator = 0.0f;
    for (int group = 0; group < reduction / 128; ++group) {
        int32_t subtotal = 0;
        uint32_t input_word = 0, weight_word = 0;
        for (unsigned offset = 0; offset < 4; ++offset) {
            const int position = group * 128 + lane * 4 + offset;
            uint32_t state;
            if constexpr (Packed) {
                state = header->bits == 4 ? r3d_iqr::trellis_state(codes, pitch, column, position, 4, 16, header->run_rows)
                    : header->bits == 3 ? r3d_iqr::trellis_state(codes, pitch, column, position, 3, 16, header->run_rows)
                    : r3d_iqr::trellis_state(codes, pitch, column, position, 5, 16, header->run_rows);
            } else {
                state = r3d_iqr::trellis_state(codes, pitch, column, position,
                    header->bits, header->state_bits, header->run_rows);
            }
            const int code = r3d_iqr::trellis_mad(state, header->multiplier, header->increment);
            const int8_t input = activation[row * activation_pitch + position];
            if constexpr (Packed) {
                input_word |= uint32_t(uint8_t(input)) << (offset * 8);
                weight_word |= uint32_t(uint8_t(code)) << (offset * 8);
            } else {
                subtotal += int(input) * code;
            }
        }
        if constexpr (Packed)
            subtotal = __builtin_amdgcn_sudot4(true, int(weight_word), true, int(input_word), 0, false);
        for (unsigned distance = 16; distance; distance >>= 1)
            subtotal += __shfl_xor(subtotal, distance, 32);
        if (lane == 0) {
            const float scaled = static_cast<float>(subtotal)
                               * activation_scale[row * activation_scale_pitch + group];
            accumulator += scaled * scales[columns + group];
        }
    }
    if (lane == 0)
        output[row * output_pitch + column] = narrow_bf16(accumulator * scales[column]);
}

__global__ void iqr_wmma(const int8_t* activation, const float* activation_scale,
    const uint8_t* weight, const float* scales, uint16_t* output,
    int rows, int columns, int reduction, int64_t activation_pitch,
    int64_t activation_scale_pitch, int64_t output_pitch) {
    const unsigned lane = threadIdx.x;
    const int column = blockIdx.x * 16 + (lane & 15u);
    const int activation_row = blockIdx.y * 16 + (lane & 15u);
    const auto* header = reinterpret_cast<const r3d_iqr::Header*>(weight);
    const uint8_t* codes = weight + sizeof(r3d_iqr::Header);
    const uint64_t code_pitch = static_cast<uint64_t>(reduction) * header->bits / 8;
    float accumulator[8] = {};
    for (int group = 0; group < reduction / 128; ++group) {
        Int8 subtotal{};
        for (int step = 0; step < 128; step += 16) {
            Int4 input_fragment{};
            Int4 weight_fragment{};
            for (unsigned word = 0; word < 4; ++word) {
                uint32_t input_word = 0;
                uint32_t weight_word = 0;
                for (unsigned byte = 0; byte < 4; ++byte) {
                    const int position = group * 128 + step + word * 4 + byte;
                    const int input_code = activation_row < rows
                        ? activation[activation_row * activation_pitch + position] : 0;
                    const uint32_t state = r3d_iqr::trellis_state(codes, code_pitch,
                        column, position, header->bits, header->state_bits, header->run_rows);
                    const int weight_code = r3d_iqr::trellis_mad(state, header->multiplier, header->increment);
                    input_word |= static_cast<uint32_t>(static_cast<uint8_t>(input_code)) << (byte * 8);
                    weight_word |= static_cast<uint32_t>(static_cast<uint8_t>(weight_code)) << (byte * 8);
                }
                input_fragment[word] = static_cast<int>(input_word);
                weight_fragment[word] = static_cast<int>(weight_word);
            }
            subtotal = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
                true, input_fragment, true, weight_fragment, subtotal, false);
        }
        for (unsigned index = 0; index < 8; ++index) {
            const int output_row = blockIdx.y * 16 + index * 2 + (lane >> 4);
            const float activation_factor = output_row < rows
                ? activation_scale[output_row * activation_scale_pitch + group] : 0.0f;
            const float scaled = static_cast<float>(subtotal[index]) * activation_factor;
            accumulator[index] += scaled * scales[columns + group];
        }
    }
    for (unsigned index = 0; index < 8; ++index) {
        const int output_row = blockIdx.y * 16 + index * 2 + (lane >> 4);
        if (output_row < rows)
            output[output_row * output_pitch + column] = narrow_bf16(accumulator[index] * scales[column]);
    }
}

template<bool Wave>
__global__ void iqr_group_bf16(const int8_t* activation, const float* activation_scale,
    const uint8_t* weight, const uint16_t* scales, uint16_t* output,
    int rows, int columns, int reduction, int64_t activation_pitch,
    int64_t activation_scale_pitch, int64_t output_pitch) {
    const unsigned lane = threadIdx.x & 31u;
    const int column = Wave ? blockIdx.x * 4 + threadIdx.x / 32
                            : blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y;
    if (column >= columns || row >= rows) return;
    const auto* header = reinterpret_cast<const r3d_iqr::Header*>(weight);
    if (header->magic != r3d_iqr::weight_magic || header->state_bits != 16
        || (header->bits != 3 && header->bits != 4 && header->bits != 5)
        || (header->run_rows != 128 && header->run_rows != 256)
        || header->rows != uint32_t(columns) || header->columns != uint32_t(reduction)
        || columns % header->run_rows) return;
    const auto* codes = weight + sizeof(r3d_iqr::Header);
    const uint64_t pitch = uint64_t(reduction) * header->bits / 8;
    const int groups = reduction / 128;
    float accumulator = 0.0f;
    for (int group = 0; group < groups; ++group) {
        int32_t subtotal = 0;
        for (int offset = Wave ? int(lane) * 4 : 0; offset < (Wave ? int(lane) * 4 + 4 : 128); ++offset) {
            const int position = group * 128 + offset;
            const uint32_t state = r3d_iqr::trellis_state(codes, pitch, column,
                position, header->bits, 16, header->run_rows);
            subtotal += int(activation[row * activation_pitch + position])
                * int(r3d_iqr::trellis_mad(state, header->multiplier, header->increment));
        }
        if constexpr (Wave) {
            for (unsigned distance = 16; distance; distance >>= 1)
                subtotal += __shfl_xor(subtotal, distance, 32);
            if (lane != 0) continue;
        }
        const float scaled = __fmul_rn(float(subtotal),
            activation_scale[row * activation_scale_pitch + group]);
        const float weight_scale = __uint_as_float(uint32_t(scales[column * groups + group]) << 16);
        accumulator = __fadd_rn(accumulator, __fmul_rn(scaled, weight_scale));
    }
    if (!Wave || lane == 0) output[row * output_pitch + column] = narrow_bf16(accumulator);
}

extern "C" int r3d_rad_gemm_iqr_scalar(const RadArgs* arguments, RadStream stream) {
    if (!arguments || arguments->n_t < 5 || arguments->n_t > 7 || !arguments->t) return RAD_E_INVAL;
    for (int index = 5; index < arguments->n_t; ++index)
        if (arguments->t[index].data || arguments->t[index].rank) return RAD_E_UNSUPPORTED;
    const int64_t implementation = rad_args_geti_or(arguments, "impl", 0);
    if (implementation < 0 || implementation > 2) return RAD_E_INVAL;
    RadTensor views[5];
    std::memcpy(views, arguments->t, sizeof(views));
    const RadTensor* tensor = views;
    const int64_t band_rows = rad_args_geti_or(arguments, "M", 0);
    const int64_t rows = tensor[0].rank == 2 ? tensor[0].shape[0] : 0;
    const int64_t columns = rad_args_geti_or(arguments, "N", 0);
    const int64_t reduction = rad_args_geti_or(arguments, "K", 0);
    if (rows < 1 || rows > 2048 || band_rows < rows || band_rows > 2048
        || columns < 1 || columns > 2147483647
        || reduction < 128 || reduction > 2147483647 || reduction % 128
        || rad_args_geti_or(arguments, "group", 128) != 128
        || rad_args_geti_or(arguments, "w_rb", 1) != 1) return RAD_E_SHAPE;
    auto& output = views[4];
    if (output.rank < 2 || output.rank > sizeof(output.shape) / sizeof(output.shape[0]))
        return RAD_E_SHAPE;
    int64_t width = 1;
    for (int axis = output.rank - 1; axis > 0; --axis) {
        if (output.shape[axis] < 1
            || output.shape[axis] > std::numeric_limits<int64_t>::max() / width)
            return RAD_E_SHAPE;
        if (output.stride[axis] != width) return RAD_E_STRIDE;
        width *= output.shape[axis];
    }
    if (output.shape[0] != rows || width != columns) return RAD_E_SHAPE;
    const int64_t max_elements = std::numeric_limits<int64_t>::max() / sizeof(uint16_t);
    if (output.stride[0] < width
        || output.stride[0] > max_elements
        || rows - 1 > (max_elements - width) / output.stride[0]) return RAD_E_STRIDE;
    const uint64_t bytes = ((rows - 1) * output.stride[0] + width) * sizeof(uint16_t);
    if (reinterpret_cast<uintptr_t>(output.data) > std::numeric_limits<uintptr_t>::max() - bytes)
        return RAD_E_STRIDE;
    output.rank = 2;
    output.shape[1] = columns;
    output.stride[1] = 1;
    const bool group_scale = tensor[3].dtype == r3d_iqr::group_scale_dtype;
    if (group_scale && std::strcmp(rad_args_gets_or(arguments, "dtype", ""), "i8a8h"))
        return RAD_E_DTYPE;
    const uint32_t types[5] = {RAD_I8, RAD_F32, r3d_iqr::weight_dtype,
        group_scale ? r3d_iqr::group_scale_dtype : r3d_iqr::scale_dtype, RAD_BF16};
    for (int index = 0; index < 5; ++index)
        if (!tensor[index].data || tensor[index].dtype != types[index]
            || tensor[index].rank != 2 || tensor[index].stride[1] != 1) return RAD_E_DTYPE;
    if (tensor[0].shape[0] != rows || tensor[0].shape[1] != reduction
        || tensor[1].shape[0] != rows || tensor[1].shape[1] != reduction / 128
        || tensor[2].shape[0] != columns || tensor[2].shape[1] != reduction
        || tensor[3].shape[0] != columns
        || (tensor[3].shape[1] != reduction / 128 && tensor[3].shape[1] != reduction)
        || tensor[4].shape[0] != rows || tensor[4].shape[1] != columns
        || tensor[0].stride[0] < reduction || tensor[1].stride[0] < reduction / 128
        || tensor[4].stride[0] < columns) return RAD_E_SHAPE;
    if (group_scale) {
        const char* wave = std::getenv("RAD_IQR_DENSE_WAVE");
        if (rows <= 16 && wave && std::strcmp(wave, "0") != 0) {
            iqr_group_bf16<true><<<dim3((columns + 3) / 4, rows), dim3(128), 0,
                reinterpret_cast<hipStream_t>(stream)>>>(
                static_cast<const int8_t*>(tensor[0].data), static_cast<const float*>(tensor[1].data),
                static_cast<const uint8_t*>(tensor[2].data), static_cast<const uint16_t*>(tensor[3].data),
                static_cast<uint16_t*>(tensor[4].data), int(rows), int(columns), int(reduction),
                tensor[0].stride[0], tensor[1].stride[0], tensor[4].stride[0]);
            return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
        }
        iqr_group_bf16<false><<<dim3((columns + 127) / 128, rows), dim3(128), 0,
            reinterpret_cast<hipStream_t>(stream)>>>(
            static_cast<const int8_t*>(tensor[0].data), static_cast<const float*>(tensor[1].data),
            static_cast<const uint8_t*>(tensor[2].data), static_cast<const uint16_t*>(tensor[3].data),
            static_cast<uint16_t*>(tensor[4].data), int(rows), int(columns), int(reduction),
            tensor[0].stride[0], tensor[1].stride[0], tensor[4].stride[0]);
        return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
    }
    if (implementation == 2) {
        const char* wave = std::getenv("RAD_IQR_DENSE_WAVE");
        if (wave && std::strcmp(wave, "0") != 0) {
            iqr_wave<true><<<dim3((columns + 3) / 4, rows), dim3(128), 0,
                reinterpret_cast<hipStream_t>(stream)>>>(
                static_cast<const int8_t*>(tensor[0].data), static_cast<const float*>(tensor[1].data),
                static_cast<const uint8_t*>(tensor[2].data), static_cast<const float*>(tensor[3].data),
                static_cast<uint16_t*>(tensor[4].data), int(rows), int(columns), int(reduction),
                tensor[0].stride[0], tensor[1].stride[0], tensor[4].stride[0]);
            return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
        }
        iqr_wave<false><<<dim3((columns + 3) / 4, rows), dim3(128), 0,
            reinterpret_cast<hipStream_t>(stream)>>>(
            static_cast<const int8_t*>(tensor[0].data), static_cast<const float*>(tensor[1].data),
            static_cast<const uint8_t*>(tensor[2].data), static_cast<const float*>(tensor[3].data),
            static_cast<uint16_t*>(tensor[4].data), static_cast<int>(rows),
            static_cast<int>(columns), static_cast<int>(reduction),
            tensor[0].stride[0], tensor[1].stride[0], tensor[4].stride[0]);
        return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
    }
    if (implementation == 1) {
        if (columns % 16) return RAD_E_SHAPE;
        iqr_wmma<<<dim3(columns / 16, (rows + 15) / 16), dim3(32), 0,
            reinterpret_cast<hipStream_t>(stream)>>>(
            static_cast<const int8_t*>(tensor[0].data), static_cast<const float*>(tensor[1].data),
            static_cast<const uint8_t*>(tensor[2].data), static_cast<const float*>(tensor[3].data),
            static_cast<uint16_t*>(tensor[4].data), static_cast<int>(rows),
            static_cast<int>(columns), static_cast<int>(reduction),
            tensor[0].stride[0], tensor[1].stride[0], tensor[4].stride[0]);
        return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
    }
    iqr_scalar<<<dim3((columns + 127) / 128, rows), dim3(128), 0,
        reinterpret_cast<hipStream_t>(stream)>>>(
        static_cast<const int8_t*>(tensor[0].data), static_cast<const float*>(tensor[1].data),
        static_cast<const uint8_t*>(tensor[2].data), static_cast<const float*>(tensor[3].data),
        static_cast<uint16_t*>(tensor[4].data), static_cast<int>(rows),
        static_cast<int>(columns), static_cast<int>(reduction),
        tensor[0].stride[0], tensor[1].stride[0], tensor[4].stride[0]);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
