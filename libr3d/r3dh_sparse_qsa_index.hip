#include <hip/hip_runtime.h>
#include "r3dh_sparse_qsa.h"
#include <cmath>
#include <cstring>
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
__device__ bool candidate(DeviceTensor<2> scores, DeviceTensor<2> table, int row, int sequence,
                          int column, bool signed_score, float& value) {
    value = static_cast<const float*>(scores.data)[row * scores.stride[0] + column * scores.stride[1]];
    if (static_cast<const int*>(table.data)[sequence * table.stride[0] + column * table.stride[1]] < 0) return false;
    if (signed_score ? !isfinite(value) : !(value > -1e30f)) return false;
    if (!signed_score) value = value > 0 ? value : 0;
    return true;
}
__global__ void select_kernel(SelectCall call) {
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
    const int row = blockIdx.x;
    const int sequence = seq(cu, call.cu_rows, row, 0);
    const int position = static_cast<const int*>(positions.data)[row * positions.stride[0]];
    const int width = min(int(call.score_columns), int(call.table_columns));
    const int complete = min(min(static_cast<const int*>(count.data)[sequence * count.stride[0]], width), (position + 1) / 4);
    int live = 0;
    float heap[512];
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
}
extern "C" int r3dh_sparse_qsa_score(const RadArgs* args, RadStream stream) {
    const int status = r3dh_sparse_qsa_shape("qsa_score", args);
    if (status) return status;
    const RadTensor* tensors = args->t;
    const ScoreCall call{compact<2>(tensors[0]), compact<1>(tensors[1]), compact<2>(tensors[2]),
        compact<1>(tensors[3]), compact<1>(tensors[4]), compact<2>(tensors[5]),
        tensors[1].shape[0], tensors[2].shape[1], tensors[3].shape[0], tensors[5].shape[1]};
    hipLaunchKernelGGL(score_kernel, dim3(tensors[5].shape[0], (tensors[5].shape[1] + 63) / 64), dim3(64), 0,
        reinterpret_cast<hipStream_t>(stream), call);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
extern "C" int r3dh_sparse_qsa_select(const RadArgs* args, RadStream stream) {
    const int status = r3dh_sparse_qsa_shape("qsa_select", args);
    if (status) return status;
    const RadTensor* tensors = args->t;
    const SelectCall call{compact<2>(tensors[0]), compact<2>(tensors[1]), compact<1>(tensors[2]),
        compact<1>(tensors[3]), compact<1>(tensors[4]), compact<2>(tensors[5]),
        compact<1>(tensors[6]), compact<1>(tensors[7]), tensors[0].shape[1], tensors[1].shape[1],
        tensors[4].shape[0], int(param(args, "topk", 0)), bool(param(args, "signed", 0))};
    hipLaunchKernelGGL(select_kernel, dim3(tensors[0].shape[0]), dim3(1), 0,
        reinterpret_cast<hipStream_t>(stream), call);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
