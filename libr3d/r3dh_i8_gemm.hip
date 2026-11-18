#include <hip/hip_runtime.h>
#include "r3dh_i8.h"
#include <cstring>
#include <cstdlib>
#include <limits>
#ifdef __FAST_MATH__
#error r3dh_i8 requires strict floating point compilation
#endif
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
namespace {
using Int4 = int __attribute__((ext_vector_type(4)));
using Int8 = int __attribute__((ext_vector_type(8)));
__device__ uint16_t narrow(float value) {
    const uint32_t word = __float_as_uint(value);
    if ((word & 0x7fffffffu) > 0x7f800000u)
        return static_cast<uint16_t>((word >> 16) | 0x40u);
    return static_cast<uint16_t>((word + 0x7fffu + ((word >> 16) & 1u)) >> 16);
}
__global__ void wave_gemm(const int8_t* activation, const float* activation_scale,
    const int8_t* weight, const float* scale, uint16_t* output,
    int rows, int columns, int reduction, int64_t activation_pitch,
    int64_t scale_pitch, int64_t output_pitch) {
    const unsigned lane = threadIdx.x & 31u;
    const int column = blockIdx.x * 4 + threadIdx.x / 32;
    const int row = blockIdx.y;
    float accumulator = 0.0f;
    for (int group = 0; group < reduction / 128; ++group) {
        int subtotal = 0;
        const int position = group * 128 + lane * 4;
        const auto* packed = reinterpret_cast<const uint32_t*>(weight
            + (((int64_t(column / 16) * (reduction / 128) + group) * 8
                 + (lane * 4) / 16) * 16 + column % 16) * 16 + (lane * 4) % 16);
        const uint32_t input = *reinterpret_cast<const uint32_t*>(
            activation + row * activation_pitch + position);
        const uint32_t weight_word = *packed;
        for (unsigned byte = 0; byte < 4; ++byte)
            subtotal += static_cast<int8_t>(input >> (byte * 8))
                      * static_cast<int8_t>(weight_word >> (byte * 8));
        for (unsigned distance = 16; distance; distance >>= 1)
            subtotal += __shfl_xor(subtotal, distance, 32);
        if (lane == 0) {
            const float scaled = static_cast<float>(subtotal)
                               * activation_scale[row * scale_pitch + group];
            accumulator = accumulator + scaled * scale[int64_t(column) * (reduction / 128) + group];
        }
    }
    if (lane == 0 && column < columns && row < rows)
        output[row * output_pitch + column] = narrow(accumulator);
}
template<int Count, int Mask>
__device__ __forceinline__ void halve_columns(int (&values)[16], int lane) {
    const bool high = (lane & Mask) != 0;
#pragma unroll
    for (int index = 0; index < Count / 2; ++index) {
        const int low = values[index], upper = values[index + Count / 2];
        values[index] = (high ? upper : low) + __shfl_xor(high ? low : upper, Mask, 32);
    }
}
__global__ __launch_bounds__(128) void dot4_gemv(const int8_t* activation, const float* activation_scale,
    const int8_t* weight, const float* scale, uint16_t* output,
    int reduction, int64_t activation_pitch, int64_t scale_pitch, int64_t output_pitch) {
    __shared__ uint32_t cached_activation[4096];
    __shared__ float cached_scale[128];
    const int tid = threadIdx.x, lane = tid & 31;
    const int row = blockIdx.y, groups = reduction / 128;
    const int first = blockIdx.x * 64 + (tid / 32) * 16;
    for (int index = tid; index < reduction / 4; index += 128)
        cached_activation[index] = reinterpret_cast<const uint32_t*>(activation + row * activation_pitch)[index];
    for (int index = tid; index < groups; index += 128)
        cached_scale[index] = activation_scale[row * scale_pitch + index];
    __syncthreads();
    float accumulator = 0;
    for (int group = 0; group < groups; ++group) {
        const int input = int(cached_activation[group * 32 + lane]);
        int sums[16];
#pragma unroll
        for (int index = 0; index < 16; ++index) {
            const int64_t offset = (((int64_t(first / 16) * groups + group) * 8
                + lane / 4) * 16 + index) * 16 + (lane % 4) * 4;
            const int packed = *reinterpret_cast<const int*>(weight + offset);
            sums[index] = __builtin_amdgcn_sudot4(true, packed, true, input, 0, false);
        }
        halve_columns<16, 16>(sums, lane);
        halve_columns<8, 8>(sums, lane);
        halve_columns<4, 4>(sums, lane);
        halve_columns<2, 2>(sums, lane);
        const int subtotal = sums[0] + __shfl_xor(sums[0], 1, 32);
        const float scaled = float(subtotal) * cached_scale[group];
        accumulator = accumulator + scaled * scale[int64_t(first + lane / 2) * groups + group];
    }
    if ((lane & 1) == 0) output[row * output_pitch + first + lane / 2] = narrow(accumulator);
}
__global__ void wmma_gemm(const int8_t* activation, const float* activation_scale,
    const int8_t* weight, const float* scale, uint16_t* output,
    int rows, int columns, int reduction, int64_t activation_pitch,
    int64_t scale_pitch, int64_t output_pitch) {
    const unsigned lane = threadIdx.x;
    const int column = blockIdx.x * 16 + (lane & 15u);
    const int input_row = blockIdx.y * 16 + (lane & 15u);
    float accumulator[8] = {};
    for (int group = 0; group < reduction / 128; ++group) {
        Int8 subtotal{};
        for (int step = 0; step < 8; ++step) {
            Int4 input_fragment{};
            Int4 weight_fragment{};
            const auto* packed = reinterpret_cast<const uint32_t*>(weight
                + (((int64_t(blockIdx.x) * (reduction / 128) + group) * 8 + step)
                   * 16 + (lane & 15u)) * 16);
            for (unsigned word = 0; word < 4; ++word) {
                if (input_row < rows)
                    input_fragment[word] = *reinterpret_cast<const uint32_t*>(activation
                        + input_row * activation_pitch + group * 128 + step * 16 + word * 4);
                weight_fragment[word] = packed[word];
            }
            subtotal = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
                true, input_fragment, true, weight_fragment, subtotal, false);
        }
        for (unsigned index = 0; index < 8; ++index) {
            const int output_row = blockIdx.y * 16 + index * 2 + (lane >> 4);
            if (output_row < rows) {
                const float scaled = static_cast<float>(subtotal[index])
                                   * activation_scale[output_row * scale_pitch + group];
                accumulator[index] = accumulator[index] + scaled * scale[int64_t(column) * (reduction / 128) + group];
            }
        }
    }
    for (unsigned index = 0; index < 8; ++index) {
        const int output_row = blockIdx.y * 16 + index * 2 + (lane >> 4);
        if (output_row < rows)
            output[output_row * output_pitch + column] = narrow(accumulator[index]);
    }
}
}
extern "C" int r3dh_i8_rad_gemm_nt_q(const RadArgs* arguments, RadStream stream) {
    if (!arguments || !arguments->t || arguments->n_t < 5 || arguments->n_t > 7
        || arguments->n_p < 0 || (arguments->n_p && !arguments->p))
        return RAD_E_INVAL;
    const char* dtype = rad_args_gets_or(arguments, "dtype", "");
    if (std::strcmp(dtype, "i8a8h") && std::strcmp(dtype, "i8a8")
        && std::strcmp(dtype, "i8")) return RAD_E_DTYPE;
    const RadTensor* tensor = arguments->t;
    const int64_t band = rad_args_geti_or(arguments, "M", 0);
    const int64_t rows = tensor[0].rank == 2 ? tensor[0].shape[0] : 0;
    const int64_t columns = rad_args_geti_or(arguments, "N", 0);
    const int64_t reduction = rad_args_geti_or(arguments, "K", 0);
    if (rows < 1 || rows > 2048 || band < rows || band > 2048
        || columns < 128 || columns > 2147483520 || columns % 128
        || reduction < 128 || reduction > 2147483520 || reduction % 128
        || rad_args_geti_or(arguments, "group", 128) != 128
        || rad_args_geti_or(arguments, "w_rb", 1) != 1
        || rad_args_geti_or(arguments, "pb1", 0)
        || rad_args_geti_or(arguments, "pb2", 0)) return RAD_E_SHAPE;
    for (int index = 5; index < arguments->n_t; ++index)
        if (arguments->t[index].data || arguments->t[index].rank) return RAD_E_UNSUPPORTED;
    const uint32_t types[] = {RAD_I8, RAD_F32, r3dh_i8::weight_dtype,
                             r3dh_i8::scale_dtype, RAD_BF16};
    const int64_t heights[] = {rows, rows, columns, columns, rows};
    const int64_t widths[] = {reduction, reduction / 128, reduction, reduction / 128, columns};
    for (int index = 0; index < 5; ++index) {
        if (!tensor[index].data || tensor[index].dtype != types[index]) return RAD_E_DTYPE;
        if (tensor[index].rank != 2 || tensor[index].shape[0] != heights[index]
            || tensor[index].shape[1] != widths[index]) return RAD_E_SHAPE;
        if (tensor[index].stride[1] != 1 || tensor[index].stride[0] < widths[index]
            || ((index == 2 || index == 3) && tensor[index].stride[0] != widths[index])
            || heights[index] - 1 > (std::numeric_limits<int64_t>::max() - widths[index]) / tensor[index].stride[0])
            return RAD_E_STRIDE;
        const unsigned alignment = index == 4 ? 2 : 4;
        if (reinterpret_cast<uintptr_t>(tensor[index].data) % alignment) return RAD_E_STRIDE;
    }
    if (tensor[0].stride[0] % 4 || reinterpret_cast<uintptr_t>(tensor[0].data) % 4
        || reinterpret_cast<uintptr_t>(tensor[2].data) % 4) return RAD_E_STRIDE;
    const dim3 grid(rows <= 2 ? columns / 4 : columns / 16, rows <= 2 ? rows : (rows + 15) / 16);
    const auto native_stream = reinterpret_cast<hipStream_t>(stream);
    const char* dot4 = std::getenv("RAD_I8_DOT4");
    if (rows <= 2 && reduction <= 16384 && dot4 && std::strcmp(dot4, "0") != 0) {
        dot4_gemv<<<dim3(columns / 64, rows), dim3(128), 0, native_stream>>>(
            static_cast<const int8_t*>(tensor[0].data), static_cast<const float*>(tensor[1].data),
            static_cast<const int8_t*>(tensor[2].data), static_cast<const float*>(tensor[3].data),
            static_cast<uint16_t*>(tensor[4].data), reduction,
            tensor[0].stride[0], tensor[1].stride[0], tensor[4].stride[0]);
        return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
    }
    if (rows <= 2)
        wave_gemm<<<grid, dim3(128), 0, native_stream>>>(
            static_cast<const int8_t*>(tensor[0].data), static_cast<const float*>(tensor[1].data),
            static_cast<const int8_t*>(tensor[2].data), static_cast<const float*>(tensor[3].data),
            static_cast<uint16_t*>(tensor[4].data), rows, columns, reduction,
            tensor[0].stride[0], tensor[1].stride[0], tensor[4].stride[0]);
    else
        wmma_gemm<<<grid, dim3(32), 0, native_stream>>>(
            static_cast<const int8_t*>(tensor[0].data), static_cast<const float*>(tensor[1].data),
            static_cast<const int8_t*>(tensor[2].data), static_cast<const float*>(tensor[3].data),
            static_cast<uint16_t*>(tensor[4].data), rows, columns, reduction,
            tensor[0].stride[0], tensor[1].stride[0], tensor[4].stride[0]);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
