#include <hip/hip_runtime.h>
#include "r3dh_vision_strict.h"
#include <cstring>
#include <climits>
extern "C" int rad_stream_device_access(RadStream, int, const void*) __attribute__((weak));
namespace {
using namespace r3dh_vision_strict;
int64_t integer(const RadArgs* args, const char* name, int64_t fallback = 0) {
    return rad_param_geti(args->p, args->n_p, name, fallback);
}
bool matrix(const RadTensor& tensor, int64_t rows, int64_t columns) {
    return tensor.data && tensor.dtype == RAD_BF16 && tensor.rank == 2 &&
        tensor.shape[0] == rows && tensor.shape[1] == columns && tensor.stride[1] == 1 &&
        tensor.stride[0] >= columns && tensor.stride[0] <= INT_MAX;
}
int device_operand(const RadTensor& tensor, RadStream stream) {
    if (!tensor.data) return RAD_E_INVAL;
    int current = -1;
    if (hipGetDevice(&current) != hipSuccess) return RAD_E_DEVICE;
    if (stream && rad_stream_device_access) {
        const int status = rad_stream_device_access(stream, current, nullptr);
        if (status == RAD_OK) return rad_stream_device_access(stream, current, tensor.data);
        if (status != RAD_E_UNSUPPORTED) return status;
    }
    hipPointerAttribute_t attributes{};
    if (hipPointerGetAttributes(&attributes, tensor.data) != hipSuccess) {
        static_cast<void>(hipGetLastError()); return RAD_E_UNSUPPORTED;
    }
    if (attributes.type != hipMemoryTypeDevice ||
        attributes.device != current) return RAD_E_UNSUPPORTED;
    return RAD_OK;
}
bool overlap(const RadTensor& left, const RadTensor& right) {
    if (!left.data || !right.data) return false;
    auto end = [](const RadTensor& tensor) {
        uint64_t count = 1;
        for (uint32_t axis = 0; axis < tensor.rank; ++axis)
            count += uint64_t(tensor.shape[axis] - 1) * uint64_t(tensor.stride[axis]);
        return uintptr_t(tensor.data) + count * uint64_t(rad_dtype_bits(tensor.dtype) / 8);
    };
    return uintptr_t(left.data) < end(right) && uintptr_t(right.data) < end(left);
}
__device__ uint16_t epilogue(uint16_t rounded, int activation, uint16_t residual, bool has_residual) {
    const float value = widen(rounded);
    if (activation == 1) rounded = narrow(0.5f * value * (1.0f + float(erf(double(value * 0.70710678118654752f)))));
    if (activation == 2) rounded = narrow(0.5f * value * (1.0f + float(tanh(double(
        0.7978845608028654f * (value + 0.044715f * value * value * value))))));
    if (has_residual) rounded = narrow(widen(residual) + widen(rounded));
    return rounded;
}
__global__ void gemm(const uint16_t* left, const uint16_t* right, const uint16_t* bias,
                    const uint16_t* residual, uint16_t* output, int64_t left_pitch,
                    int64_t right_pitch, int64_t bias_pitch, int64_t residual_pitch,
                    int64_t output_pitch, int64_t rows, int64_t columns, int64_t depth, int activation) {
    const int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= rows * columns) return;
    const int64_t row = index / columns, column = index % columns;
    const uint16_t old = residual ? residual[row * residual_pitch + column] : 0;
    const float offset = bias ? widen(bias[column * bias_pitch]) : 0.0f;
    const uint16_t value = product(left + row * left_pitch,
        right + column * right_pitch, depth, 1, 1, offset);
    output[row * output_pitch + column] = epilogue(value, activation, old, residual != nullptr);
}
int gemm_entry(const RadArgs* args, RadStream stream, bool with_bias) {
    if (!args || !args->t || args->n_t != (with_bias ? 5 : 3) || args->n_p < 0 ||
        (args->n_p && !args->p)) return RAD_E_INVAL;
    const int64_t band = integer(args, "M"), columns = integer(args, "N"), depth = integer(args, "K");
    if (band < 1 || band > 256) return RAD_E_SHAPE;
    if (!r3dh_vision_strict_geometry(columns, depth)) return RAD_E_UNSUPPORTED;
    if (std::strcmp(rad_param_gets(args->p, args->n_p, "dtype", "bf16"), "bf16")) return RAD_E_DTYPE;
    const char* name = with_bias ? rad_param_gets(args->p, args->n_p, "act", "none") : "none";
    const int activation = !std::strcmp(name, "none") ? 0 : !std::strcmp(name, "gelu") ? 1 : !std::strcmp(name, "gelu_tanh") ? 2 : -1;
    if (activation < 0) return RAD_E_UNSUPPORTED;
    const RadTensor empty{}, &left = args->t[0], &right = args->t[1], &output = args->t[with_bias ? 4 : 2];
    const int64_t rows = output.rank == 2 ? output.shape[0] : 0;
    if (rows < 1 || rows > band) return RAD_E_SHAPE;
    const RadTensor& bias = with_bias ? args->t[2] : empty;
    const RadTensor& residual = with_bias ? args->t[3] : empty;
    if (!matrix(left, rows, depth) || !matrix(right, columns, depth) || !matrix(output, rows, columns) ||
        (residual.data && !matrix(residual, rows, columns))) return RAD_E_SHAPE;
    if (with_bias && (!bias.data || bias.dtype != RAD_BF16 || bias.rank != 1 || bias.shape[0] != columns || bias.stride[0] < 1)) return RAD_E_SHAPE;
    for (int operand = 0; operand < args->n_t; ++operand) {
        if (!args->t[operand].data && with_bias && operand == 3) continue;
        const int status = device_operand(args->t[operand], stream);
        if (status != RAD_OK) return status;
    }
    if (overlap(output, left) || overlap(output, right) || overlap(output, bias)) return RAD_E_UNSUPPORTED;
    if (overlap(output, residual) && (output.data != residual.data || output.stride[0] != residual.stride[0]))
        return RAD_E_UNSUPPORTED;
    gemm<<<(rows * columns + 127) / 128, 128, 0, reinterpret_cast<hipStream_t>(stream)>>>(
        static_cast<const uint16_t*>(left.data), static_cast<const uint16_t*>(right.data),
        static_cast<const uint16_t*>(bias.data), static_cast<const uint16_t*>(residual.data),
        static_cast<uint16_t*>(output.data), left.stride[0], right.stride[0], bias.stride[0],
        residual.stride[0], output.stride[0], rows, columns, depth, activation);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_STATE;
}
struct HeadTensor {
    uint16_t* data;
    int64_t row, head;
};
bool heads(const RadTensor& tensor, int64_t rows, HeadTensor& result) {
    if (!tensor.data || tensor.dtype != RAD_BF16) return false;
    if (tensor.rank == 2 && tensor.shape[0] == rows && tensor.shape[1] == 1152 &&
        tensor.stride[1] == 1 && tensor.stride[0] >= 1152) {
        result = {static_cast<uint16_t*>(tensor.data), tensor.stride[0], 72}; return true;
    }
    if (tensor.rank == 3 && tensor.shape[0] == rows && tensor.shape[1] == 16 && tensor.shape[2] == 72 &&
        tensor.stride[2] == 1 && tensor.stride[1] >= 72 && tensor.stride[0] >= 16 * tensor.stride[1]) {
        result = {static_cast<uint16_t*>(tensor.data), tensor.stride[0], tensor.stride[1]}; return true;
    }
    return false;
}
__device__ float score(HeadTensor query, HeadTensor key, int64_t row, int64_t column, int64_t head, float scale) {
    float sum = 0.0f;
    for (int64_t dimension = 0; dimension < 72; ++dimension)
        sum = add_product(sum, widen(query.data[row * query.row + head * query.head + dimension]),
            widen(key.data[column * key.row + head * key.head + dimension]));
    return sum * scale;
}
__global__ void attention(HeadTensor query, HeadTensor key, HeadTensor value, HeadTensor output,
                          const int32_t* boundaries, int64_t sequences, int64_t rows, float scale) {
    const int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= rows * 16 * 72) return;
    const int64_t dimension = index % 72, head = (index / 72) % 16, row = index / 1152;
    int64_t begin = -1, end = -1;
    if (boundaries[0] != 0 || boundaries[sequences] != rows) return;
    for (int64_t sequence = 0; sequence < sequences; ++sequence) {
        const int64_t lower = boundaries[sequence], upper = boundaries[sequence + 1];
        if (lower < 0 || upper < lower || upper > rows) return;
        if (row >= lower && row < upper) { begin = lower; end = upper; }
    }
    if (begin < 0) return;
    float maximum = -INFINITY;
    for (int64_t column = begin; column < end; ++column) {
        const float dot = score(query, key, row, column, head, scale);
        if (dot > maximum) maximum = dot;
    }
    if (maximum == -INFINITY) {
        output.data[row * output.row + head * output.head + dimension] = 0;
        return;
    }
    float denominator = 0.0f, sum = 0.0f;
    for (int64_t column = begin; column < end; ++column) {
        const float weight = float(exp(double(score(query, key, row, column, head, scale) - maximum)));
        denominator += weight;
        sum = add_product(sum, weight, widen(value.data[column * value.row + head * value.head + dimension]));
    }
    output.data[row * output.row + head * output.head + dimension] = narrow(sum * (denominator > 0.0f ? 1.0f / denominator : 0.0f));
}
}
extern "C" int r3dh_vision_strict_gemm_nt(const RadArgs* args, RadStream stream) { return gemm_entry(args, stream, false); }
extern "C" int r3dh_vision_strict_gemm_nt_bias(const RadArgs* args, RadStream stream) { return gemm_entry(args, stream, true); }
extern "C" int r3dh_vision_strict_attn_dense(const RadArgs* args, RadStream stream) {
    if (!args || !args->t || args->n_t != 5 || args->n_p < 0 || (args->n_p && !args->p)) return RAD_E_INVAL;
    const int64_t band = integer(args, "M");
    const int64_t rows = args->t[4].rank >= 2 ? args->t[4].shape[0] : 0;
    if (band < 1 || band > 256 || rows < 1 || rows > band) return RAD_E_SHAPE;
    if (integer(args, "head_dim") != 72 || integer(args, "gqa", 1) != 1 || integer(args, "causal") ||
        integer(args, "window") || integer(args, "window_right") || integer(args, "block") || integer(args, "cu_div", 1) != 1)
        return RAD_E_UNSUPPORTED;
    if (std::strcmp(rad_param_gets(args->p, args->n_p, "dtype", "bf16"), "bf16")) return RAD_E_DTYPE;
    HeadTensor query{}, key{}, value{}, output{};
    if (!heads(args->t[0], rows, query) || !heads(args->t[1], rows, key) || !heads(args->t[2], rows, value) ||
        !heads(args->t[4], rows, output)) return RAD_E_SHAPE;
    const RadTensor& cu = args->t[3];
    if (!cu.data || cu.dtype != RAD_I32 || cu.rank != 1 || cu.stride[0] != 1 || cu.shape[0] < 2 || cu.shape[0] > rows + 2)
        return RAD_E_SHAPE;
    for (int operand = 0; operand < 5; ++operand) {
        const int status = device_operand(args->t[operand], stream); if (status != RAD_OK) return status;
    }
    for (int operand = 0; operand < 4; ++operand)
        if (overlap(args->t[4], args->t[operand])) return RAD_E_UNSUPPORTED;
    const float scale = float(rad_param_getf(args->p, args->n_p, "scale", 1.0f / std::sqrt(72.0f)));
    if (!std::isfinite(scale)) return RAD_E_SHAPE;
    attention<<<(rows * 1152 + 127) / 128, 128, 0, reinterpret_cast<hipStream_t>(stream)>>>(
        query, key, value, output, static_cast<const int32_t*>(cu.data), cu.shape[0] - 1, rows, scale);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_STATE;
}
