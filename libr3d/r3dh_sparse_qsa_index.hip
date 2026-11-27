#include <hip/hip_runtime.h>
#include "r3dh_sparse_qsa.h"
#include <cmath>
#include <cstring>
#include <cstdlib>
#include "r3d_flags.h"
namespace {
template<int Strides> struct DeviceTensor {
    void* data;
    int64_t stride[Strides];
};
template<int Strides> DeviceTensor<Strides> compact(const RadTensor& tensor) {
    DeviceTensor<Strides> result{tensor.data, {}};
    for (int index = 0; index < Strides; ++index) result.stride[index] = tensor.stride[index];
    return result;
}
struct ScoreCall {
    DeviceTensor<2> query;
    DeviceTensor<1> keys;
    DeviceTensor<2> table;
    DeviceTensor<1> cu, count;
    DeviceTensor<2> output;
    int64_t key_rows, table_columns, cu_rows, output_columns;
};
struct SelectCall {
    DeviceTensor<2> scores, table;
    DeviceTensor<1> count, positions, cu;
    DeviceTensor<2> selected;
    DeviceTensor<1> nselected, used;
    int64_t score_columns, table_columns, cu_rows;
    int topk;
    bool signed_score;
};
static_assert(((sizeof(ScoreCall) + 7) & ~size_t(7)) + 256 <= 768);
static_assert(((sizeof(SelectCall) + 7) & ~size_t(7)) + 256 <= 768);
long long param(const RadArgs* args, const char* key, long long fallback) {
    rad_args_geti(args, key, &fallback);
    return fallback;
}
template<int Strides> __device__ float load_bf(const DeviceTensor<Strides>& tensor, long long offset) {
    return __uint_as_float(unsigned(static_cast<const unsigned short*>(tensor.data)[offset]) << 16);
}
__device__ int seq(const DeviceTensor<1>& cu, int64_t cu_rows, int row, int fallback) {
    const int* offsets = static_cast<const int*>(cu.data);
    for (int index = 0; index + 1 < cu_rows; ++index)
        if (row >= offsets[index * cu.stride[0]] && row < offsets[(index + 1) * cu.stride[0]]) return index;
    return fallback;
}
__global__ void score_kernel(ScoreCall call) {
    const auto& query = call.query;
    const auto& keys = call.keys;
    const auto& table = call.table;
    const auto& cu = call.cu;
    const auto& count = call.count;
    const auto& output = call.output;
    const int row = blockIdx.x;
    const int column = blockIdx.y * blockDim.x + threadIdx.x;
    const int sequence = seq(cu, call.cu_rows, row, -1);
    if (sequence < 0 || column >= call.table_columns || column >= call.output_columns
        || column >= static_cast<const int*>(count.data)[sequence * count.stride[0]]) return;
    const int page = static_cast<const int*>(table.data)[sequence * table.stride[0] + column * table.stride[1]];
    float sum = 0;
    if (page >= 0 && page < call.key_rows) {
        for (int head = 0; head < 4; ++head) {
            float dot = 0;
            for (int lane = 0; lane < 128; ++lane)
                dot += load_bf(query, row * query.stride[0] + (head * 128 + lane) * query.stride[1])
                     * load_bf(keys, page * keys.stride[0] + lane);
            sum += fmaxf(dot, 0.0f);
        }
        sum /= sqrtf(128.0f);
    } else sum = -INFINITY;
    static_cast<float*>(output.data)[row * output.stride[0] + column * output.stride[1]] = sum;
}
// score_kernel's arithmetic with one thread per (page, head) dot chain; the head-0 thread of each page adds the
// four ReLU'd dots in head order. Vector loads read a key row's 128 values as sixteen 16-byte pieces.
template<bool Vector> __device__ void score_part(const ScoreCall& call, int row, int part) {
    __shared__ float query[512];
    const int tid = threadIdx.x, head = tid & 3;
    const int column = part * 64 + (tid >> 2);
    const int sequence = seq(call.cu, call.cu_rows, row, -1);
    for (int index = tid; index < 512; index += 256)
        query[index] = load_bf(call.query, row * call.query.stride[0] + index * call.query.stride[1]);
    __syncthreads();
    const bool live = sequence >= 0 && column < call.table_columns && column < call.output_columns
        && column < static_cast<const int*>(call.count.data)[sequence * call.count.stride[0]];
    const int page = live ? static_cast<const int*>(call.table.data)[sequence * call.table.stride[0]
        + column * call.table.stride[1]] : -1;
    const bool valid = page >= 0 && page < call.key_rows;
    float dot = 0;
    if (valid) {
        const unsigned short* keys = static_cast<const unsigned short*>(call.keys.data) + page * call.keys.stride[0];
        const float* q = query + head * 128;
        if (Vector) {
            for (int piece = 0; piece < 16; ++piece) {
                const uint4 packed = reinterpret_cast<const uint4*>(keys)[piece];
                const uint32_t words[4]{packed.x, packed.y, packed.z, packed.w};
#pragma unroll
                for (int half = 0; half < 8; ++half) {
                    const float key = __uint_as_float(half & 1 ? words[half / 2] & 0xffff0000u : words[half / 2] << 16);
                    dot += q[piece * 8 + half] * key;
                }
            }
        } else {
            for (int lane = 0; lane < 128; ++lane)
                dot += q[lane] * __uint_as_float(unsigned(keys[lane]) << 16);
        }
    }
    const int base = threadIdx.x & ~3;
    const float d0 = __shfl(dot, base), d1 = __shfl(dot, base + 1), d2 = __shfl(dot, base + 2), d3 = __shfl(dot, base + 3);
    if (!live || head) return;
    float sum = 0;
    if (valid) {
        sum += fmaxf(d0, 0.0f);
        sum += fmaxf(d1, 0.0f);
        sum += fmaxf(d2, 0.0f);
        sum += fmaxf(d3, 0.0f);
        sum /= sqrtf(128.0f);
    } else sum = -INFINITY;
    static_cast<float*>(call.output.data)[row * call.output.stride[0] + column * call.output.stride[1]] = sum;
}
template<bool Vector> __global__ __launch_bounds__(256) void score_kernel_parallel(ScoreCall call) {
    score_part<Vector>(call, blockIdx.x, blockIdx.y);
}
__device__ bool candidate(DeviceTensor<2> scores, DeviceTensor<2> table, int row, int sequence,
                          int column, bool signed_score, float& value) {
    value = static_cast<const float*>(scores.data)[row * scores.stride[0] + column * scores.stride[1]];
    if (static_cast<const int*>(table.data)[sequence * table.stride[0] + column * table.stride[1]] < 0) return false;
    if (signed_score ? !isfinite(value) : !(value > -1e30f)) return false;
    if (!signed_score) value = value > 0 ? value : 0;
    return true;
}
__device__ void select_row(const SelectCall& call, int row, float* heap) {
    const auto& scores = call.scores;
    const auto& table = call.table;
    const auto& count = call.count;
    const auto& positions = call.positions;
    const auto& cu = call.cu;
    const auto& selected = call.selected;
    const auto& nselected = call.nselected;
    const auto& used = call.used;
    const int topk = call.topk;
    const bool signed_score = call.signed_score;
    const int sequence = seq(cu, call.cu_rows, row, 0);
    const int position = static_cast<const int*>(positions.data)[row * positions.stride[0]];
    const int width = min(int(call.score_columns), int(call.table_columns));
    const int complete = min(min(static_cast<const int*>(count.data)[sequence * count.stride[0]], width), (position + 1) / 4);
    int live = 0;
    float threshold = -INFINITY;
    for (int column = 0; column < complete; ++column) {
        float value;
        if (!candidate(scores, table, row, sequence, column, signed_score, value)) continue;
        ++live;
        if (live <= topk) {
            int child = live - 1;
            while (child > 0 && value < heap[(child - 1) / 2]) {
                heap[child] = heap[(child - 1) / 2];
                child = (child - 1) / 2;
            }
            heap[child] = value;
        } else if (value > heap[0]) {
            int parent = 0;
            while (parent * 2 + 1 < topk) {
                int child = parent * 2 + 1;
                if (child + 1 < topk && heap[child + 1] < heap[child]) ++child;
                if (!(heap[child] < value)) break;
                heap[parent] = heap[child];
                parent = child;
            }
            heap[parent] = value;
        }
    }
    if (live >= topk) threshold = heap[0];
    int written = 0;
    for (int pass = 0; pass < (live <= topk ? 1 : 2); ++pass)
        for (int column = 0; column < complete && written < topk; ++column) {
            float value;
            if (!candidate(scores, table, row, sequence, column, signed_score, value)) continue;
            if (live > topk && !(pass == 0 ? value > threshold : value == threshold)) continue;
            static_cast<int*>(selected.data)[row * selected.stride[0] + written++ * selected.stride[1]] =
                static_cast<const int*>(table.data)[sequence * table.stride[0] + column * table.stride[1]];
        }
    if (complete < width && (position + 1) % 4) {
        const int page = static_cast<const int*>(table.data)[sequence * table.stride[0] + complete * table.stride[1]];
        if (page >= 0) static_cast<int*>(selected.data)[row * selected.stride[0] + written++ * selected.stride[1]] = page;
    }
    for (int column = written; column <= topk; ++column)
        static_cast<int*>(selected.data)[row * selected.stride[0] + column * selected.stride[1]] = -1;
    static_cast<int*>(nselected.data)[row * nselected.stride[0]] = written;
    static_cast<int*>(used.data)[row * used.stride[0]] = written ? (written - 1) * 4 + position % 4 + 1 : 0;
}
__global__ void select_kernel(SelectCall call) {
    float heap[512];
    select_row(call, blockIdx.x, heap);
}
// Without pruning (live <= topk) the original emits every live column in order; do that in parallel.
__device__ void select_part(const SelectCall& call, int row) {
    __shared__ int counts[256];
    __shared__ float heap[512];
    __shared__ int total, written;
    const int tid = threadIdx.x;
    const int sequence = seq(call.cu, call.cu_rows, row, 0);
    const int position = static_cast<const int*>(call.positions.data)[row * call.positions.stride[0]];
    const int width = min(int(call.score_columns), int(call.table_columns));
    const int complete = min(min(static_cast<const int*>(call.count.data)[sequence * call.count.stride[0]], width),
        (position + 1) / 4);
    const int chunk = (max(complete, 0) + 255) / 256, first = tid * chunk, last = min(first + chunk, complete);
    int live = 0;
    float value;
    for (int column = first; column < last; ++column)
        live += candidate(call.scores, call.table, row, sequence, column, call.signed_score, value);
    counts[tid] = live;
    __syncthreads();
    if (tid == 0) {
        int sum = 0;
        for (int index = 0; index < 256; ++index) { const int count = counts[index]; counts[index] = sum; sum += count; }
        total = sum;
    }
    __syncthreads();
    if (total > call.topk) {
        if (tid == 0) select_row(call, row, heap);
        return;
    }
    int* selected = static_cast<int*>(call.selected.data) + row * call.selected.stride[0];
    const int* table = static_cast<const int*>(call.table.data) + sequence * call.table.stride[0];
    int offset = counts[tid];
    for (int column = first; column < last; ++column)
        if (candidate(call.scores, call.table, row, sequence, column, call.signed_score, value))
            selected[offset++ * call.selected.stride[1]] = table[column * call.table.stride[1]];
    if (tid == 0) {
        int count = total;
        if (complete < width && (position + 1) % 4) {
            const int page = table[complete * call.table.stride[1]];
            if (page >= 0) selected[count++ * call.selected.stride[1]] = page;
        }
        written = count;
        static_cast<int*>(call.nselected.data)[row * call.nselected.stride[0]] = count;
        static_cast<int*>(call.used.data)[row * call.used.stride[0]] = count ? (count - 1) * 4 + position % 4 + 1 : 0;
    }
    __syncthreads();
    for (int column = written + tid; column <= call.topk; column += 256)
        selected[column * call.selected.stride[1]] = -1;
}
__global__ __launch_bounds__(256) void select_kernel_parallel(SelectCall call) { select_part(call, blockIdx.x); }
constexpr int radix_capacity = 9216;
template<bool four_rows>
__device__ __forceinline__ void select_radix_row(SelectCall call) {
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1100__)
    select_part(call, blockIdx.x);
#else
#if defined(__HIP_DEVICE_COMPILE__)
    if (__builtin_amdgcn_wavefrontsize() != 32) { select_part(call, blockIdx.x); return; }
#endif
    const int tid = threadIdx.x, lane = tid % 32, wave = tid / 32;
    const int row = four_rows ? int(blockIdx.x) : 0;
    const int position = static_cast<const int*>(call.positions.data)[row * call.positions.stride[0]];
    const int width = min(int(call.score_columns), int(call.table_columns));
    const int complete = min(min(static_cast<const int*>(call.count.data)[0], width), (position + 1) / 4);
    const int* table = static_cast<const int*>(call.table.data);
    int* selected = static_cast<int*>(call.selected.data) + row * call.selected.stride[0];
    __shared__ uint32_t keys[radix_capacity];
    __shared__ int hist[8][16], bins[16], rank, live, high_base, equal_base;
    __shared__ uint32_t prefix;
    int valid_count = 0;
    for (int column = tid; column < complete; column += 256) {
        float value;
        const bool valid = candidate(call.scores, call.table, row, 0, column, call.signed_score, value);
        const uint32_t raw = value == 0.0f ? 0u : __float_as_uint(value);
        keys[column] = valid ? (raw & 0x80000000u ? ~raw : raw ^ 0x80000000u) : 0u;
        valid_count += valid;
    }
    for (int distance = 16; distance; distance >>= 1) valid_count += __shfl_down(valid_count, distance, 32);
    if (lane == 0) hist[wave][0] = valid_count;
    __syncthreads();
    if (tid == 0) {
        live = 0;
        for (int group = 0; group < 8; ++group) live += hist[group][0];
        rank = 512; prefix = 0; high_base = 0; equal_base = 0;
    }
    __syncthreads();
    const bool prune = live > 512;
    if (prune) {
        uint32_t mask = 0;
        for (int shift = 28; shift >= 0; shift -= 4) {
            const uint32_t target = prefix;
            int count = 0;
            for (int base = 0; base < complete; base += 256) {
                const uint32_t key = base + tid < complete ? keys[base + tid] : 0u;
                uint32_t members = uint32_t(__ballot(key != 0u && (key & mask) == target));
                const unsigned digit = (key >> shift) & 15u;
#pragma unroll
                for (int bit = 0; bit < 4; ++bit) {
                    const uint32_t ones = uint32_t(__ballot((digit >> bit) & 1u));
                    members &= (lane >> bit) & 1 ? ones : ~ones;
                }
                count += __popc(members);
            }
            if (lane < 16) hist[wave][lane] = count;
            __syncthreads();
            if (tid < 16) {
                int total = 0;
                for (int group = 0; group < 8; ++group) total += hist[group][tid];
                bins[tid] = total;
            }
            __syncthreads();
            if (tid == 0) {
                int bucket = 15, remaining = rank;
                while (bucket > 0 && remaining > bins[bucket]) remaining -= bins[bucket--];
                rank = remaining; prefix = target | (uint32_t(bucket) << shift);
            }
            mask |= 15u << shift;
            __syncthreads();
        }
    }
    const uint32_t threshold = prefix;
    const int quota = rank;
    for (int base = 0; base < complete; base += 256) {
        const int column = base + tid;
        const uint32_t key = column < complete ? keys[column] : 0u;
        const bool above = key > threshold, equal = prune && key == threshold;
        const uint32_t high_mask = uint32_t(__ballot(above)), equal_mask = uint32_t(__ballot(equal));
        if (lane == 0) { hist[wave][0] = __popc(high_mask); hist[wave][1] = __popc(equal_mask); }
        __syncthreads();
        if (tid == 0) {
            int high = high_base, tied = equal_base;
            for (int group = 0; group < 8; ++group) {
                const int add_high = hist[group][0], add_equal = hist[group][1];
                hist[group][0] = high; hist[group][1] = tied;
                high += add_high; tied += add_equal;
            }
            high_base = high; equal_base = tied;
        }
        __syncthreads();
        const uint32_t earlier = (1u << lane) - 1u;
        if (above) selected[(hist[wave][0] + __popc(high_mask & earlier)) * call.selected.stride[1]] = table[column * call.table.stride[1]];
        if (equal) {
            const int index = hist[wave][1] + __popc(equal_mask & earlier);
            if (index < quota) selected[(512 - quota + index) * call.selected.stride[1]] = table[column * call.table.stride[1]];
        }
        __syncthreads();
    }
    int written = min(512, live);
    if (complete >= 0 && complete < width && (position + 1) % 4 && table[complete * call.table.stride[1]] >= 0) {
        if (tid == 0) selected[written * call.selected.stride[1]] = table[complete * call.table.stride[1]];
        ++written;
    }
    for (int column = written + tid; column <= 512; column += 256) selected[column * call.selected.stride[1]] = -1;
    if (tid == 0) {
        static_cast<int*>(call.nselected.data)[row * call.nselected.stride[0]] = written;
        static_cast<int*>(call.used.data)[row * call.used.stride[0]] = written ? (written - 1) * 4 + position % 4 + 1 : 0;
    }
#endif
}
    __global__ __launch_bounds__(256) void select_kernel_radix(SelectCall call) { select_radix_row<false>(call); }
    __global__ __launch_bounds__(256) void select_kernel_radix4(SelectCall call) { select_radix_row<true>(call); }
ScoreCall score_call(const RadTensor* tensors) {
    return ScoreCall{compact<2>(tensors[0]), compact<1>(tensors[1]), compact<2>(tensors[2]),
        compact<1>(tensors[3]), compact<1>(tensors[4]), compact<2>(tensors[5]),
        tensors[1].shape[0], tensors[2].shape[1], tensors[3].shape[0], tensors[5].shape[1]};
}
SelectCall select_call(const RadArgs* args) {
    const RadTensor* tensors = args->t;
    return SelectCall{compact<2>(tensors[0]), compact<2>(tensors[1]), compact<1>(tensors[2]),
        compact<1>(tensors[3]), compact<1>(tensors[4]), compact<2>(tensors[5]),
        compact<1>(tensors[6]), compact<1>(tensors[7]), tensors[0].shape[1], tensors[1].shape[1],
        tensors[4].shape[0], int(param(args, "topk", 0)), bool(param(args, "signed", 0))};
}
bool aligned_keys(const RadTensor& keys) {
    return reinterpret_cast<uintptr_t>(keys.data) % 16 == 0 && keys.stride[0] % 8 == 0;
}
}
extern "C" int r3dh_sparse_qsa_score(const RadArgs* args, RadStream stream) {
    const int status = r3dh_sparse_qsa_shape("qsa_score", args);
    if (status) return status;
    const RadTensor* tensors = args->t;
    const ScoreCall call = score_call(tensors);
    const bool parallel = r3d_exact_path("RAD_QSA_SCORE_PARALLEL");
    if (parallel) {
        const bool vector = aligned_keys(tensors[1]);
        const dim3 grid(tensors[5].shape[0], (tensors[5].shape[1] + 63) / 64);
        if (vector) hipLaunchKernelGGL(score_kernel_parallel<true>, grid, dim3(256), 0, reinterpret_cast<hipStream_t>(stream), call);
        else hipLaunchKernelGGL(score_kernel_parallel<false>, grid, dim3(256), 0, reinterpret_cast<hipStream_t>(stream), call);
    } else {
        hipLaunchKernelGGL(score_kernel, dim3(tensors[5].shape[0], (tensors[5].shape[1] + 63) / 64), dim3(64), 0,
            reinterpret_cast<hipStream_t>(stream), call);
    }
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
extern "C" int r3dh_sparse_qsa_select_observed(const RadArgs* args, RadStream stream, int* path) {
    const int status = r3dh_sparse_qsa_shape("qsa_select", args);
    if (status) return status;
    const RadTensor* tensors = args->t;
    const SelectCall call = select_call(args);
    const bool parallel = r3d_exact_path("RAD_QSA_SELECT_PARALLEL");
    const char* enabled = std::getenv("RAD_QSA_SELECT_RADIX");
    const long width = std::min(call.score_columns, call.table_columns);
    const bool eligible = !std::strcmp(R3D_BUILD_TARGET, "gfx1100") && parallel && enabled && !std::strcmp(enabled, "1") && call.topk == 512 &&
        call.cu_rows == 2 && width > 512 && width <= radix_capacity;
    const bool radix = eligible && tensors[0].shape[0] == 1;
    const char* four = eligible && tensors[0].shape[0] == 4 ? std::getenv("RAD_QSA_SELECT_RADIX4") : nullptr;
    const bool radix4 = four && !std::strcmp(four, "1");
    if (path) *path = radix ? 2 : radix4 ? 3 : parallel ? 1 : 0;
    if (radix)
        hipLaunchKernelGGL(select_kernel_radix, dim3(1), dim3(256), 0, reinterpret_cast<hipStream_t>(stream), call);
    else if (radix4)
        hipLaunchKernelGGL(select_kernel_radix4, dim3(4), dim3(256), 0, reinterpret_cast<hipStream_t>(stream), call);
    else if (parallel)
        hipLaunchKernelGGL(select_kernel_parallel, dim3(tensors[0].shape[0]), dim3(256), 0,
            reinterpret_cast<hipStream_t>(stream), call);
    else
        hipLaunchKernelGGL(select_kernel, dim3(tensors[0].shape[0]), dim3(1), 0,
            reinterpret_cast<hipStream_t>(stream), call);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
extern "C" int r3dh_sparse_qsa_select(const RadArgs* args, RadStream stream) {
    return r3dh_sparse_qsa_select_observed(args, stream, nullptr);
}
