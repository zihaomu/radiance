#include "r3d_i8_gated.h"
#include <hip/hip_runtime.h>

#include "r3d_iqr.h"
#include "rad_plugin.h"

#include <cstring>
#include <cstdlib>
#include <limits>
#include "r3d_flags.h"

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

__device__ __forceinline__ uint32_t iqr_u5_codes(const uint8_t* codes, uint64_t pitch,
    unsigned column, unsigned position, uint32_t multiplier, uint32_t increment) {
    uint32_t packed[4];
#pragma unroll
    for (unsigned back = 0; back < 4; ++back) {
        const unsigned previous = (column & ~255u) | ((column - back) & 255u);
        const unsigned bit = position * 5;
        const uint8_t* source = codes + uint64_t(previous) * pitch + bit / 8;
        packed[back] = (uint32_t(source[0]) | (uint32_t(source[1]) << 8)
            | (uint32_t(source[2]) << 16)) >> (bit & 7u);
    }
    uint32_t result = 0;
#pragma unroll
    for (unsigned element = 0; element < 4; ++element) {
        const unsigned shift = element * 5;
        const uint32_t state = ((packed[0] >> shift) & 31u)
            | (((packed[1] >> shift) & 31u) << 5)
            | (((packed[2] >> shift) & 31u) << 10)
            | (((packed[3] >> shift) & 1u) << 15);
        result |= uint32_t(uint8_t(r3d_iqr::trellis_mad(state, multiplier, increment))) << (element * 8);
    }
    return result;
}

template<bool Packed, bool U5 = false>
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
        if (U5 && header->bits == 5 && header->state_bits == 16 && header->run_rows == 256) {
            const int position = group * 128 + lane * 4;
            __builtin_memcpy(&input_word, activation + row * activation_pitch + position, sizeof(input_word));
            weight_word = iqr_u5_codes(codes, pitch, unsigned(column), unsigned(position),
                header->multiplier, header->increment);
        } else {
            for (unsigned offset = 0; offset < 4; ++offset) {
                const int position = group * 128 + lane * 4 + offset;
                uint32_t state;
                if constexpr (Packed && !U5) {
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

constexpr unsigned fused_waves = 4;
constexpr unsigned fused_pitch = 136;

// Lane L decodes K positions [4L, 4L+4) of all 16 tile columns; the trellis window is shared.
template<unsigned Bits>
__device__ __forceinline__ void iqr_fused_decode(const uint8_t* codes, uint64_t pitch,
    unsigned tile_column, unsigned run_mask, unsigned position, uint32_t multiplier,
    uint32_t increment, uint32_t (&words)[16]) {
    constexpr unsigned backs = (16 + Bits - 1) / Bits;
    constexpr unsigned span = 15 + backs;
    const unsigned bit = position * Bits;
    const unsigned shift = bit & 7u;
    const unsigned run_start = tile_column & ~run_mask;
    uint32_t chunk[span];
#pragma unroll
    for (unsigned index = 0; index < span; ++index) {
        const unsigned row = run_start | ((tile_column + index - (backs - 1)) & run_mask);
        const uint8_t* source = codes + uint64_t(row) * pitch + bit / 8;
        uint32_t word = uint32_t(source[0]) | (uint32_t(source[1]) << 8);
        if constexpr (Bits == 5) word |= uint32_t(source[2]) << 16;
        chunk[index] = word >> shift;
    }
#pragma unroll
    for (unsigned column = 0; column < 16; ++column) {
        uint32_t packed = 0;
#pragma unroll
        for (unsigned element = 0; element < 4; ++element) {
            uint32_t state = 0;
#pragma unroll
            for (unsigned back = 0; back < backs; ++back)
                state |= ((chunk[column + backs - 1 - back] >> (element * Bits)) & ((1u << Bits) - 1u))
                    << (back * Bits);
            // trellis_mad's byte sum as one udot4: the same integer.
            const uint32_t masked = ((state & 0xffffu) * (multiplier & 0x00ffffffu) + increment) & 0x3f3f3f3fu;
            const int value = int(__builtin_amdgcn_udot4(masked, 0x01010101u, 0u, false)) - 126;
            packed |= uint32_t(uint8_t(value)) << (element * 8);
        }
        words[column] = packed;
    }
}

// Same arithmetic as iqr_scalar; each weight is decoded once per call per band of Tiles*16 rows.
template<int Tiles>
__global__ __launch_bounds__(fused_waves * 32) void iqr_wmma_fused(const int8_t* activation,
    const float* activation_scale, const uint8_t* weight, const float* scales, uint16_t* output,
    int rows, int columns, int reduction, int64_t activation_pitch,
    int64_t activation_scale_pitch, int64_t output_pitch) {
    __shared__ __align__(16) uint8_t transposed[fused_waves][16 * fused_pitch];
    const unsigned lane = threadIdx.x & 31u;
    const unsigned wave = threadIdx.x / 32u;
    const int tile_column = int(blockIdx.x * fused_waves + wave) * 16;
    if (tile_column >= columns) return;
    const unsigned slot = lane & 15u;
    const unsigned half = lane >> 4;
    const int column = tile_column + int(slot);
    const int band = int(blockIdx.y) * Tiles * 16;
    const int band_rows = rows - band < Tiles * 16 ? rows - band : Tiles * 16;
    const auto* header = reinterpret_cast<const r3d_iqr::Header*>(weight);
    const uint8_t* codes = weight + sizeof(r3d_iqr::Header);
    const unsigned bits = header->bits;
    const unsigned run_rows = header->run_rows;
    const uint64_t code_pitch = static_cast<uint64_t>(reduction) * bits / 8;
    const bool windowed = bits >= 3 && bits <= 5 && header->state_bits == 16
        && run_rows >= 16 && (run_rows & (run_rows - 1)) == 0;
    const float* column_scale = scales + columns;
    const int8_t* input[Tiles];
#pragma unroll
    for (int tile = 0; tile < Tiles; ++tile) {
        const int row = band + tile * 16 + int(slot);
        input[tile] = activation + int64_t(row < rows ? row : rows - 1) * activation_pitch;
    }
    float accumulator[Tiles][8] = {};
    uint8_t* transposed_tile = transposed[wave];
    for (int group = 0; group < reduction / 128; ++group) {
        const unsigned position = unsigned(group) * 128u + lane * 4u;
        uint32_t words[16];
        if (windowed && bits == 5)
            iqr_fused_decode<5>(codes, code_pitch, unsigned(tile_column), run_rows - 1u, position,
                header->multiplier, header->increment, words);
        else if (windowed && bits == 4)
            iqr_fused_decode<4>(codes, code_pitch, unsigned(tile_column), run_rows - 1u, position,
                header->multiplier, header->increment, words);
        else if (windowed)
            iqr_fused_decode<3>(codes, code_pitch, unsigned(tile_column), run_rows - 1u, position,
                header->multiplier, header->increment, words);
        else
            for (unsigned index = 0; index < 16; ++index) {
                uint32_t packed = 0;
                for (unsigned element = 0; element < 4; ++element) {
                    const uint32_t state = r3d_iqr::trellis_state(codes, code_pitch,
                        unsigned(tile_column) + index, position + element, bits,
                        header->state_bits, run_rows);
                    packed |= uint32_t(uint8_t(r3d_iqr::trellis_mad(state, header->multiplier,
                        header->increment))) << (element * 8);
                }
                words[index] = packed;
            }
#pragma unroll
        for (unsigned index = 0; index < 16; ++index)
            *reinterpret_cast<uint32_t*>(transposed_tile + index * fused_pitch + lane * 4u) = words[index];
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
        __builtin_amdgcn_wave_barrier();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
        Int4 weight_fragment[8];
#pragma unroll
        for (unsigned step = 0; step < 8; ++step) {
            const auto* source = reinterpret_cast<const uint2*>(transposed_tile + slot * fused_pitch + step * 16u);
            const uint2 low = source[0], high = source[1];
            weight_fragment[step] = Int4{int(low.x), int(low.y), int(high.x), int(high.y)};
        }
        const float group_scale = column_scale[group];
#pragma unroll
        for (int tile = 0; tile < Tiles; ++tile) {
            if (tile * 16 >= band_rows) break;
            Int8 subtotal{};
#pragma unroll
            for (unsigned step = 0; step < 8; ++step) {
                const uint4 value = *reinterpret_cast<const uint4*>(input[tile] + group * 128 + step * 16);
                const Int4 input_fragment{int(value.x), int(value.y), int(value.z), int(value.w)};
                subtotal = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
                    true, input_fragment, true, weight_fragment[step], subtotal, false);
            }
#pragma unroll
            for (unsigned index = 0; index < 8; ++index) {
                const int output_row = band + tile * 16 + int(index * 2 + half);
                if (output_row >= rows) continue;
                const float scaled = static_cast<float>(subtotal[index])
                    * activation_scale[output_row * activation_scale_pitch + group];
                accumulator[tile][index] = accumulator[tile][index] + scaled * group_scale;
            }
        }
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
        __builtin_amdgcn_wave_barrier();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
    }
    const float row_scale = scales[column];
#pragma unroll
    for (int tile = 0; tile < Tiles; ++tile)
#pragma unroll
        for (unsigned index = 0; index < 8; ++index) {
            const int output_row = band + tile * 16 + int(index * 2 + half);
            if (output_row < rows)
                output[output_row * output_pitch + column] = narrow_bf16(accumulator[tile][index] * row_scale);
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
    if (implementation < 0 || implementation > 3) return RAD_E_INVAL;
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
    const bool fused = r3d_exact_path("RAD_IQR_FUSED_WMMA");
    const bool fused_decode = r3d_exact_path("RAD_IQR_FUSED_DECODE");
    const bool fusable = columns % 16 == 0 && reinterpret_cast<uintptr_t>(tensor[0].data) % 16 == 0
        && tensor[0].stride[0] % 16 == 0;
    const bool use_fused = fusable && (implementation == 3
        || (implementation == 1 && fused)
        || (implementation == 2 && fused_decode));
    if (implementation == 2 && !use_fused) {
        const bool u5 = r3d_exact_path("RAD_IQR_U5_WAVE");
        const bool qg = r3d_exact_path("RAD_IQR_U5_QG");
        if (rows <= 2 && (columns == 16384 || (columns == 12288 && qg))
            && reduction == 2560 && u5) {
            iqr_wave<true, true><<<dim3((columns + 3) / 4, rows), dim3(128), 0,
                reinterpret_cast<hipStream_t>(stream)>>>(
                static_cast<const int8_t*>(tensor[0].data), static_cast<const float*>(tensor[1].data),
                static_cast<const uint8_t*>(tensor[2].data), static_cast<const float*>(tensor[3].data),
                static_cast<uint16_t*>(tensor[4].data), int(rows), int(columns), int(reduction),
                tensor[0].stride[0], tensor[1].stride[0], tensor[4].stride[0]);
            return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
        }
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
    if (use_fused) {
        const int tiles = rows <= 16 ? 1 : rows <= 32 ? 2 : 4;
        const dim3 grid(unsigned((columns / 16 + fused_waves - 1) / fused_waves),
            unsigned((rows + tiles * 16 - 1) / (tiles * 16)));
        const auto native_stream = reinterpret_cast<hipStream_t>(stream);
        const auto* input = static_cast<const int8_t*>(tensor[0].data);
        const auto* input_scale = static_cast<const float*>(tensor[1].data);
        const auto* weight = static_cast<const uint8_t*>(tensor[2].data);
        const auto* scale = static_cast<const float*>(tensor[3].data);
        auto* result = static_cast<uint16_t*>(tensor[4].data);
        if (tiles == 1)
            iqr_wmma_fused<1><<<grid, fused_waves * 32, 0, native_stream>>>(input, input_scale, weight, scale,
                result, int(rows), int(columns), int(reduction), tensor[0].stride[0], tensor[1].stride[0], tensor[4].stride[0]);
        else if (tiles == 2)
            iqr_wmma_fused<2><<<grid, fused_waves * 32, 0, native_stream>>>(input, input_scale, weight, scale,
                result, int(rows), int(columns), int(reduction), tensor[0].stride[0], tensor[1].stride[0], tensor[4].stride[0]);
        else
            iqr_wmma_fused<4><<<grid, fused_waves * 32, 0, native_stream>>>(input, input_scale, weight, scale,
                result, int(rows), int(columns), int(reduction), tensor[0].stride[0], tensor[1].stride[0], tensor[4].stride[0]);
        return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
    }
    if (implementation == 1 || implementation == 3) {
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
