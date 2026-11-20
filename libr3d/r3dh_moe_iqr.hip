#include <hip/hip_runtime.h>
#include "r3dh_moe_iqr.h"
#include <cstring>
#include <cstdlib>
#include <limits>
#include "r3d_flags.h"
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
    bool dot4, window;
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
        const int input_row = source(call, first + (lane & 15u), begin, end);
        for (int group = 0; routed && group < 20; ++group) {
            Int8 subtotal{};
            for (int step = 0; step < 128; step += 16) {
                Int4 activation_fragment{}, weight_fragment{};
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
constexpr int mapped_columns = 64;
constexpr int mapped_pitch = 132;
constexpr int mapped_rows = 64;
constexpr int mapped_tiles = mapped_rows / 16;
int map_capacity(int rows, int experts) {
    return (experts < rows ? experts : rows) + (rows + mapped_rows - 1) / mapped_rows + 1;
}
__global__ __launch_bounds__(256) void build_run_map(Call call, int2* map, int capacity) {
    __shared__ int counts[1024];
    const int total = call.geometry.tokens * 10;
    const int experts = call.geometry.experts;
    for (int index = threadIdx.x; index < capacity; index += 256)
        map[index] = make_int2(-2, 0);
    for (int expert = threadIdx.x; expert < 1024; expert += 256) {
        const int begin = expert <= experts ? call.offsets[expert] : total;
        const int end = expert < experts ? call.offsets[expert + 1] : total;
        counts[expert] = expert <= experts ? (end - begin + mapped_rows - 1) / mapped_rows : 0;
    }
    __syncthreads();
    for (int distance = 1; distance < 1024; distance *= 2) {
        int previous[4];
        for (int slot = 0; slot < 4; ++slot) {
            const int index = threadIdx.x + slot * 256;
            previous[slot] = index >= distance ? counts[index - distance] : 0;
        }
        __syncthreads();
        for (int slot = 0; slot < 4; ++slot)
            counts[threadIdx.x + slot * 256] += previous[slot];
        __syncthreads();
    }
    for (int expert = threadIdx.x; expert <= experts; expert += 256) {
        const int begin = call.offsets[expert];
        const int end = expert < experts ? call.offsets[expert + 1] : total;
        int destination = expert ? counts[expert - 1] : 0;
        for (int row = begin; row < end; row += mapped_rows)
            map[destination++] = make_int2(expert < experts ? expert : -1, row);
    }
}
template<int Count, int Mask>
__device__ __forceinline__ void dot4_halve(int (&values)[16], int lane) {
    const bool high = (lane & Mask) != 0;
#pragma unroll
    for (int index = 0; index < Count / 2; ++index) {
        const int low = values[index], upper = values[index + Count / 2];
        values[index] = (high ? upper : low)
            + __shfl_xor(high ? low : upper, Mask, 32);
    }
}
struct Dot4Codes {
    uint32_t packed[16][2];
};
__device__ __forceinline__ Dot4Codes fetch_dot4_codes(const uint8_t* codes, int base,
    int part_width, int width, int group, int lane) {
    Dot4Codes result{};
#pragma unroll
    for (int index = 0; index < 16; ++index) {
        const int column = base + index, local = column % part_width;
        const int row = (column / part_width) * (width / 2) + local;
        if (column < 2 * part_width && local < width / 2) {
#pragma unroll
            for (unsigned back = 0; back < 4; ++back) {
                const unsigned previous = (unsigned(row) & ~127u) | ((unsigned(row) - back) & 127u);
                const uint16_t value = *reinterpret_cast<const uint16_t*>(codes
                    + uint64_t(previous) * 1280 + group * 64 + lane * 2);
                result.packed[index][back / 2] |= uint32_t(value) << ((back & 1u) * 16);
            }
        }
    }
    return result;
}
// Same codes as fetch_dot4_codes for 16 contiguous rows in one run: each of the 19 rows is read once.
__device__ __forceinline__ Dot4Codes fetch_dot4_window(const uint8_t* codes, unsigned first_row,
    int group, int lane) {
    uint32_t window[19];
#pragma unroll
    for (unsigned index = 0; index < 19; ++index) {
        const unsigned row = (first_row & ~127u) | ((first_row + index - 3u) & 127u);
        window[index] = *reinterpret_cast<const uint16_t*>(codes + uint64_t(row) * 1280 + group * 64 + lane * 2);
    }
    Dot4Codes result{};
#pragma unroll
    for (unsigned index = 0; index < 16; ++index)
#pragma unroll
        for (unsigned back = 0; back < 4; ++back)
            result.packed[index][back / 2] |= window[index + 3 - back] << ((back & 1u) * 16);
    return result;
}
template<bool Prefetch>
__device__ void gate_up_dot4(Call call, int expert, int first, int end, int8_t* staging) {
    const int tid = threadIdx.x, lane = tid & 31;
    const int count = end - first;
    const int width = call.geometry.width[expert & 1];
    const int part_width = call.geometry.output_width / 2;
    const int base = blockIdx.x * mapped_columns + (tid / 32) * 16;
    const auto* blob = entry(call.weight[expert & 1], expert / 2);
    const auto* header = reinterpret_cast<const Header*>(blob);
    const auto* codes = blob + sizeof(Header);
    const auto* scales = reinterpret_cast<const float*>(entry(call.scale[expert & 1], expert / 2));
    auto* activation_scales = reinterpret_cast<float*>(staging + 4 * 2560);
    for (int index = tid; index < count * 2560; index += 128) {
        const int input_row = source(call, first + index / 2560, first, end);
        staging[index] = input_row >= 0
            ? call.activation[input_row * call.activation_pitch + index % 2560] : 0;
    }
    for (int index = tid; index < count * 20; index += 128) {
        const int input_row = source(call, first + index / 20, first, end);
        activation_scales[index] = input_row >= 0
            ? call.activation_scale[input_row * call.scale_pitch + index % 20] : 0.0f;
    }
    __syncthreads();
    const int column = base + (lane >> 1);
    const int local = column % part_width;
    const int weight_row = (column / part_width) * (width / 2) + local;
    const bool valid = column < call.geometry.output_width && local < width / 2;
    float accumulator[4] = {};
    const int base_local = base % part_width;
    const unsigned first_row = unsigned((base / part_width) * (width / 2) + base_local);
    const bool contiguous = call.window && base + 16 <= 2 * part_width && base_local + 16 <= part_width
        && base_local + 16 <= width / 2 && (first_row & 127u) <= 112u;
    auto fetch = [&](int group) {
        return contiguous ? fetch_dot4_window(codes, first_row, group, lane)
            : fetch_dot4_codes(codes, base, part_width, width, group, lane);
    };
    Dot4Codes current{};
    float current_scale = 0;
    if constexpr (Prefetch) {
        current = fetch(0);
        current_scale = scales[width];
    }
    for (int group = 0; group < 20; ++group) {
        Dot4Codes next{};
        float next_scale = 0;
        if constexpr (Prefetch) {
            if (group + 1 < 20) {
                next = fetch(group + 1);
                next_scale = scales[width + group + 1];
            }
        } else {
            current = fetch(group);
            current_scale = scales[width + group];
        }
        int weights[16];
#pragma unroll
        for (int index = 0; index < 16; ++index) {
            uint32_t states[4] = {};
#pragma unroll
            for (unsigned back = 0; back < 4; ++back) {
#pragma unroll
                for (int byte = 0; byte < 4; ++byte)
                    states[byte] |= ((current.packed[index][back / 2]
                        >> ((back & 1u) * 16 + byte * 4)) & 15u) << (back * 4);
            }
            uint32_t word = 0;
#pragma unroll
            for (int byte = 0; byte < 4; ++byte)
                word |= uint32_t(uint8_t(mad(states[byte], header->multiplier, header->increment))) << (byte * 8);
            weights[index] = int(word);
        }
#pragma unroll
        for (int row = 0; row < 4; ++row) {
            if (row >= count) break;
            const int input = *reinterpret_cast<const int*>(staging + row * 2560 + group * 128 + lane * 4);
            int sums[16];
#pragma unroll
            for (int index = 0; index < 16; ++index)
                sums[index] = __builtin_amdgcn_sudot4(true, weights[index], true, input, 0, false);
            dot4_halve<16, 16>(sums, lane);
            dot4_halve<8, 8>(sums, lane);
            dot4_halve<4, 4>(sums, lane);
            dot4_halve<2, 2>(sums, lane);
            const int sum = sums[0] + __shfl_xor(sums[0], 1, 32);
            const float scaled = float(sum) * activation_scales[row * 20 + group];
            accumulator[row] = accumulator[row] + scaled * current_scale;
        }
        if constexpr (Prefetch) {
            current = next;
            current_scale = next_scale;
        }
    }
    if ((lane & 1) == 0 && column < call.geometry.output_width) {
        for (int row = 0; row < count; ++row)
            call.output[(first + row) * call.output_pitch + column]
                = valid && source(call, first + row, first, end) >= 0
                ? bf16(accumulator[row] * scales[weight_row]) : 0;
    }
}
template<bool Prefetch>
__global__ __launch_bounds__(128) void gate_up_mapped(Call call, const int2* map) {
    const int2 task = map[blockIdx.y];
    if (task.x == -2) return;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int first = task.y;
    const int total = call.geometry.tokens * 10;
    const bool routed = task.x >= 0;
    const int boundary = routed ? call.offsets[task.x + 1] : total;
    const int end = first + mapped_rows < boundary ? first + mapped_rows : boundary;
    const int tiles = (end - first + 15) / 16;
    const int column_base = blockIdx.x * mapped_columns;
    const int column = column_base + (tid / 32) * 16 + (lane & 15);
    if (!routed) {
        for (int index = tid; index < mapped_rows * mapped_columns; index += 128) {
            const int row = first + index / mapped_columns;
            const int output_column = column_base + index % mapped_columns;
            if (row < end && output_column < call.geometry.output_width)
                call.output[row * call.output_pitch + output_column] = 0;
        }
        return;
    }
    __shared__ int sources[mapped_rows];
    __shared__ __align__(16) int8_t staging[(mapped_rows + mapped_columns) * mapped_pitch];
    if (call.dot4 && end - first <= 4) {
        gate_up_dot4<Prefetch>(call, task.x, first, end, staging);
        return;
    }
    int8_t* activation_tile = staging;
    int8_t* weight_tile = staging + mapped_rows * mapped_pitch;
    if (tid < mapped_rows) sources[tid] = source(call, first + tid, first, end);
    __syncthreads();
    const int parity = task.x & 1;
    const int width = call.geometry.width[parity];
    const int part_width = call.geometry.output_width / 2;
    const int local = column % part_width;
    const int weight_row = (column / part_width) * (width / 2) + local;
    const bool live_column = column < call.geometry.output_width && local < width / 2;
    const uint8_t* blob = entry(call.weight[parity], task.x / 2);
    const auto* header = reinterpret_cast<const Header*>(blob);
    const auto* codes = blob + sizeof(Header);
    const auto* scales = reinterpret_cast<const float*>(entry(call.scale[parity], task.x / 2));
    float accumulator[mapped_tiles][8] = {};
    __shared__ __align__(16) uint8_t raw_codes[(mapped_columns + 3) * 64];
    const int base_local = column_base % part_width;
    const unsigned first_row = unsigned((column_base / part_width) * (width / 2) + base_local);
    const bool windowed = call.window && column_base + mapped_columns <= 2 * part_width
        && base_local + mapped_columns <= part_width && base_local + mapped_columns <= width / 2
        && (first_row & 127u) + mapped_columns <= 128u && (reinterpret_cast<uintptr_t>(codes) & 3u) == 0;
    for (int group = 0; group < 20; ++group) {
        if (windowed) {
            for (int index = tid; index < (mapped_columns + 3) * 16; index += 128) {
                const unsigned row = (first_row & ~127u) | ((first_row + unsigned(index / 16) - 3u) & 127u);
                *reinterpret_cast<uint32_t*>(raw_codes + index * 4) = *reinterpret_cast<const uint32_t*>(
                    codes + uint64_t(row) * 1280 + group * 64 + (index % 16) * 4);
            }
            __syncthreads();
        }
        for (int index = tid; index < tiles * 16 * 128; index += 128) {
            const int row = index / 128;
            const int position = index % 128;
            const int input_row = sources[row];
            activation_tile[row * mapped_pitch + position] = input_row >= 0
                ? call.activation[input_row * call.activation_pitch + group * 128 + position] : 0;
        }
        for (int index = tid; index < mapped_columns * 64; index += 128) {
            const int tile_column = index / 64;
            const int position = (index % 64) * 2;
            const int output_column = column_base + tile_column;
            const int local_column = output_column % part_width;
            const int decoded_row = (output_column / part_width) * (width / 2) + local_column;
            uint32_t low_state = 0, high_state = 0;
            const bool valid = output_column < call.geometry.output_width && local_column < width / 2;
            if (valid) {
                for (unsigned back = 0; back < 4; ++back) {
                    const unsigned previous = (static_cast<unsigned>(decoded_row) & ~127u)
                        | ((static_cast<unsigned>(decoded_row) - back) & 127u);
                    const uint8_t packed = windowed ? raw_codes[(tile_column + 3 - back) * 64 + position / 2]
                        : codes[static_cast<uint64_t>(previous) * 1280 + group * 64 + position / 2];
                    low_state |= (packed & 15u) << (back * 4);
                    high_state |= (packed >> 4) << (back * 4);
                }
            }
            weight_tile[tile_column * mapped_pitch + position] = valid
                ? mad(low_state, header->multiplier, header->increment) : 0;
            weight_tile[tile_column * mapped_pitch + position + 1] = valid
                ? mad(high_state, header->multiplier, header->increment) : 0;
        }
        __syncthreads();
        Int8 subtotal[mapped_tiles]{};
        for (int step = 0; step < 128; step += 16) {
            Int4 weight_fragment{};
            for (unsigned word = 0; word < 4; ++word)
                weight_fragment[word] = *reinterpret_cast<const int*>(weight_tile
                    + ((tid / 32) * 16 + (lane & 15)) * mapped_pitch + step + word * 4);
#pragma unroll
            for (int tile = 0; tile < mapped_tiles; ++tile) {
                if (tile >= tiles) break;
                Int4 activation_fragment{};
                for (unsigned word = 0; word < 4; ++word)
                    activation_fragment[word] = *reinterpret_cast<const int*>(activation_tile
                        + (tile * 16 + (lane & 15)) * mapped_pitch + step + word * 4);
                subtotal[tile] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
                    true, activation_fragment, true, weight_fragment, subtotal[tile], false);
            }
        }
#pragma unroll
        for (int tile = 0; tile < mapped_tiles; ++tile) {
            if (tile >= tiles) break;
            for (unsigned index = 0; index < 8; ++index) {
                const int input_row = sources[tile * 16 + index * 2 + (lane >> 4)];
                const float factor = input_row >= 0
                    ? call.activation_scale[input_row * call.scale_pitch + group] : 0.0f;
                const float scaled = static_cast<float>(subtotal[tile][index]) * factor;
                accumulator[tile][index] = accumulator[tile][index] + scaled * scales[width + group];
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int tile = 0; tile < mapped_tiles; ++tile) {
        if (tile >= tiles) break;
        for (unsigned index = 0; index < 8; ++index) {
            const int tile_row = tile * 16 + index * 2 + (lane >> 4);
            const int row = first + tile_row;
            if (row < end && column < call.geometry.output_width)
                call.output[row * call.output_pitch + column] = live_column && sources[tile_row] >= 0
                    ? bf16(accumulator[tile][index] * scales[weight_row]) : 0;
        }
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

// Decode gate_up: the same integer group sums as gate_up_dot4, with K interleaved over four waves per row half.
// weight = bytesum((state * multiplier + increment) & 0x3f3f3f3f) - 126, so a group's sum is
// sum_k x_k * bytesum_k - 126 * sum_k x_k; the float chain over the 20 groups keeps its order.
constexpr int split_columns = 64, split_pitch = 68;
__global__ __launch_bounds__(256) void gate_up_split(Call call, const int2* map) {
    const int2 task = map[blockIdx.y];
    if (task.x == -2) return;
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, half = wave & 1, quarter = wave >> 1;
    const int first = task.y, total = call.geometry.tokens * 10;
    const bool routed = task.x >= 0;
    const int boundary = routed ? call.offsets[task.x + 1] : total;
    const int end = first + mapped_rows < boundary ? first + mapped_rows : boundary;
    const int column_base = blockIdx.x * split_columns;
    if (!routed) {
        for (int index = tid; index < (end - first) * split_columns; index += 256) {
            const int column = column_base + index % split_columns;
            if (column < call.geometry.output_width)
                call.output[(first + index / split_columns) * call.output_pitch + column] = 0;
        }
        return;
    }
    __shared__ uint32_t inputs[640];
    __shared__ int group_inputs[20];
    __shared__ float input_factors[20], group_factors[20];
    __shared__ int sums[split_columns][21];
    __shared__ uint32_t tiles[8][35 * split_pitch / 4];
    const int parity = task.x & 1, width = call.geometry.width[parity];
    const int part_width = call.geometry.output_width / 2;
    const int first_row = (column_base / part_width) * (width / 2) + column_base % part_width;
    const unsigned tile_row = unsigned(first_row + half * 32);
    const auto* blob = entry(call.weight[parity], task.x / 2);
    const auto* header = reinterpret_cast<const Header*>(blob);
    const auto* codes = blob + sizeof(Header);
    const auto* scales = reinterpret_cast<const float*>(entry(call.scale[parity], task.x / 2));
    const uint32_t multiplier = header->multiplier & 0x00ffffffu, increment = header->increment;
    uint32_t* tile = tiles[wave];
    uint4 staged[5];
    auto fetch = [&](int group) {
#pragma unroll
        for (int index = 0; index < 5; ++index) {
            const int slot = index * 32 + lane, row = slot / 4;
            uint4 packed{};
            if (row < 35) {
                const unsigned source_row = (tile_row & ~127u) | ((tile_row + unsigned(row) - 3u) & 127u);
                __builtin_memcpy(&packed, codes + uint64_t(source_row) * 1280 + group * 64 + (slot % 4) * 16, 16);
            }
            staged[index] = packed;
        }
    };
    for (int row = first; row < end; ++row) {
        const int input_row = source(call, row, first, end);
        __syncthreads();
        if (tid < 160) {
            uint4 packed{};
            if (input_row >= 0) __builtin_memcpy(&packed, call.activation + input_row * call.activation_pitch + tid * 16, 16);
            const uint32_t words[4]{packed.x, packed.y, packed.z, packed.w};
            int partial = 0;
#pragma unroll
            for (int word = 0; word < 4; ++word) {
                inputs[tid * 4 + word] = words[word];
#pragma unroll
                for (int byte = 0; byte < 4; ++byte) partial += int(int8_t(words[word] >> (byte * 8)));
            }
            partial += __shfl_xor(partial, 1, 8);
            partial += __shfl_xor(partial, 2, 8);
            partial += __shfl_xor(partial, 4, 8);
            if ((tid & 7) == 0) group_inputs[tid / 8] = partial;
        } else if (tid < 180) {
            input_factors[tid - 160] = input_row >= 0 ? call.activation_scale[input_row * call.scale_pitch + tid - 160] : 0.0f;
        } else if (tid < 200) {
            group_factors[tid - 180] = scales[width + tid - 180];
        }
        fetch(quarter);
        __syncthreads();
        for (int step = 0; step < 5; ++step) {
            const int group = quarter + 4 * step;
#pragma unroll
            for (int index = 0; index < 5; ++index) {
                const int slot = index * 32 + lane, row_index = slot / 4;
                if (row_index < 35) {
                    uint32_t* target = tile + row_index * (split_pitch / 4) + (slot % 4) * 4;
                    target[0] = staged[index].x; target[1] = staged[index].y;
                    target[2] = staged[index].z; target[3] = staged[index].w;
                }
            }
            __syncthreads();
            if (step + 1 < 5) fetch(group + 4);
            const uint32_t* xs = inputs + group * 32;
            int sum = 0;
#pragma unroll 4
            for (int word = 0; word < 16; ++word) {
                const uint32_t w0 = tile[(lane + 3) * (split_pitch / 4) + word];
                const uint32_t w1 = tile[(lane + 2) * (split_pitch / 4) + word];
                const uint32_t w2 = tile[(lane + 1) * (split_pitch / 4) + word];
                const uint32_t w3 = tile[lane * (split_pitch / 4) + word];
                constexpr uint32_t low = 0x0f0f0f0fu;
                const uint32_t even_low = (w0 & low) | ((w1 & low) << 4);
                const uint32_t even_high = (w2 & low) | ((w3 & low) << 4);
                const uint32_t odd_low = ((w0 >> 4) & low) | (w1 & ~low);
                const uint32_t odd_high = ((w2 >> 4) & low) | (w3 & ~low);
                const uint32_t pairs[4]{__builtin_amdgcn_perm(even_high, even_low, 0x05010400u),
                    __builtin_amdgcn_perm(odd_high, odd_low, 0x05010400u),
                    __builtin_amdgcn_perm(even_high, even_low, 0x07030602u),
                    __builtin_amdgcn_perm(odd_high, odd_low, 0x07030602u)};
                const uint32_t x_words[2]{xs[word * 2], xs[word * 2 + 1]};
#pragma unroll
                for (int pair = 0; pair < 4; ++pair) {
                    // pair p holds k = 4 * (p / 2) + (p & 1) in its low half and that k + 2 in its high half
                    const uint32_t x_word = x_words[pair >> 1];
                    const uint32_t low_byte = pair & 1, high_byte = low_byte + 2;
                    const int low_input = int(__builtin_amdgcn_perm(x_word, x_word, low_byte * 0x01010101u));
                    const int high_input = int(__builtin_amdgcn_perm(x_word, x_word, high_byte * 0x01010101u));
                    const uint32_t low_value = ((pairs[pair] & 0xffffu) * multiplier + increment) & 0x3f3f3f3fu;
                    const uint32_t high_value = ((pairs[pair] >> 16) * multiplier + increment) & 0x3f3f3f3fu;
                    sum = __builtin_amdgcn_sudot4(false, int(low_value), true, low_input, sum, false);
                    sum = __builtin_amdgcn_sudot4(false, int(high_value), true, high_input, sum, false);
                }
            }
            sums[half * 32 + lane][group] = sum - 126 * group_inputs[group];
            __syncthreads();
        }
        __syncthreads();
        if (tid < split_columns) {
            const int column = column_base + tid, local = column % part_width;
            const int weight_row = (column / part_width) * (width / 2) + local;
            const bool valid = column < call.geometry.output_width && local < width / 2;
            float accumulator = 0;
#pragma unroll
            for (int group = 0; group < 20; ++group) {
                const float scaled = static_cast<float>(sums[tid][group]) * input_factors[group];
                accumulator = accumulator + scaled * group_factors[group];
            }
            if (column < call.geometry.output_width)
                call.output[row * call.output_pitch + column]
                    = valid && input_row >= 0 ? bf16(accumulator * scales[weight_row]) : 0;
        }
    }
}

}

extern "C" int r3dh_moe_iqr_launch(const RadArgs* arguments, RadStream stream) {
    // A tenth operand, `split`, marks an in-step checkpoint; every row here is computed alone, so it changes nothing.
    if (!arguments || (arguments->n_t != 9 && arguments->n_t != 10) || !arguments->t) return RAD_E_INVAL;
    Call call{};
    int status = config(arguments->p, arguments->n_p, &call.geometry);
    if (status != RAD_OK) return status;
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
    const bool dot4 = r3d_exact_path("RAD_IQR_DOT4");
    call.dot4 = dot4;
    const bool window = r3d_exact_path("RAD_IQR_DOT4_WINDOW");
    call.window = window;
    const int capacity = map_capacity(geometry.tokens * 10, geometry.experts);
    if (arguments->scratch && arguments->scratch_bytes >= int64_t(capacity) * int64_t(sizeof(int2))
        && reinterpret_cast<uintptr_t>(arguments->scratch) % alignof(int2) == 0) {
        auto* map = static_cast<int2*>(arguments->scratch);
        const auto native_stream = reinterpret_cast<hipStream_t>(stream);
        build_run_map<<<1, 256, 0, native_stream>>>(call, map, capacity);
        if (hipGetLastError() != hipSuccess) return RAD_E_DEVICE;
        const dim3 grid((geometry.output_width + mapped_columns - 1) / mapped_columns, capacity);
        const bool even = geometry.width[0] == geometry.output_width && geometry.width[1] == geometry.output_width
            && (geometry.output_width / 2) % split_columns == 0;
        if (geometry.tokens == 1 && even && r3d_exact_path("RAD_IQR_SPLIT")) {
            gate_up_split<<<dim3(geometry.output_width / split_columns, capacity), 256, 0, native_stream>>>(call, map);
            return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
        }
        const char* prefetch = std::getenv("RAD_IQR_PREFETCH");
        if (call.dot4 && prefetch && std::strcmp(prefetch, "0") != 0)
            gate_up_mapped<true><<<grid, 128, 0, native_stream>>>(call, map);
        else
            gate_up_mapped<false><<<grid, 128, 0, native_stream>>>(call, map);
        return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
    }
    hipLaunchKernelGGL(gate_up,
        dim3((geometry.output_width + 15) / 16, (geometry.tokens * 10 + 15) / 16),
        dim3(32), 0, reinterpret_cast<hipStream_t>(stream), call);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

extern "C" int64_t r3dh_moe_iqr_scratch(const RadArgs* arguments) {
    Config geometry{};
    if (!arguments || config(arguments->p, arguments->n_p, &geometry) != RAD_OK) return 0;
    return int64_t(map_capacity(geometry.tokens * 10, geometry.experts)) * sizeof(int2);
}
