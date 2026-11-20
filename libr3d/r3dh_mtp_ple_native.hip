#include <hip/hip_runtime.h>
#include "r3dh_mtp_ple.h"
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include "r3d_flags.h"


namespace r3dh_mtp_ple_detail {
using Index = int64_t;
Index integer(const RadArgs* args, const char* key, Index fallback = 0) {
    return rad_args_geti_or(args, key, fallback);
}
float real(const RadArgs* args, const char* key, float fallback) {
    return float(rad_args_getf_or(args, key, fallback));
}
const RadTensor* operand(const RadArgs* args, int index) {
    return args && args->t && index < args->n_t && args->t[index].data &&
        args->t[index].rank ? args->t + index : nullptr;
}
Index pitch(const RadTensor* tensor, Index width) {
    if (!tensor || !width || tensor->rank > RAD_MAX_RANK) return -1;
    if (tensor->rank == 2 && tensor->shape[1] == width && tensor->stride[1] == 1 &&
        tensor->stride[0] >= width) return tensor->stride[0];
    return rad_tensor_is_contiguous(tensor) && rad_tensor_numel(tensor) % width == 0 ? width : -1;
}
Index rows(const RadTensor* tensor, Index width) {
    return pitch(tensor, width) > 0 ? rad_tensor_numel(tensor) / width : 0;
}
__device__ float bf(uint16_t value) { return __uint_as_float(uint32_t(value) << 16); }
__device__ uint16_t narrow(float value) {
    uint32_t bits = __float_as_uint(value);
    if ((bits & 0x7fffffffu) > 0x7f800000u) return uint16_t((bits >> 16) | 0x40);
    return uint16_t((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}
__device__ float fp8(uint8_t code) {
    const int magnitude = code & 127, exponent = magnitude >> 3, mantissa = magnitude & 7;
    float value = exponent == 0 ? float(mantissa) * 0.001953125f :
        ldexpf(1.0f + float(mantissa) * 0.125f, exponent - 7);
    if (magnitude == 127) value = __uint_as_float(0x7fc00000u);
    return code & 128 ? -value : value;
}
struct DataView {
    void* data;
    uint32_t dtype;
    DataView(const RadTensor& tensor) : data(tensor.data), dtype(tensor.dtype) {}
};
static_assert(sizeof(DataView) == 16);
template <class Tensor>
__device__ float load(Tensor tensor, Index offset) {
    if (tensor.dtype == RAD_BF16) return bf(static_cast<const uint16_t*>(tensor.data)[offset]);
    if (tensor.dtype == RAD_F8E4M3) return fp8(static_cast<const uint8_t*>(tensor.data)[offset]);
    return static_cast<const float*>(tensor.data)[offset];
}
__device__ float weight(RadTensor tensor, Index row, Index column, Index width) {
    if (tensor.dtype != RAD_F8E4M3 && tensor.dtype != r3dh_mtp_fc_e4m3_dtype)
        return load(tensor, row * tensor.stride[0] + column);
    const auto* base = static_cast<const uint8_t*>(tensor.data) + row * tensor.stride[0];
    return fp8(base[column]) * reinterpret_cast<const float*>(base + width)[column / 128];
}
__global__ void norm(RadTensor hidden, RadTensor embedding, RadTensor wh, RadTensor we,
                     uint16_t* normalized, Index width, Index streams, Index hp, Index ep,
                     Index groups, float eps, float wadd) {
    if (threadIdx.x) return;
    const Index row = blockIdx.x, group = blockIdx.y, wide = width * streams;
    const bool is_embedding = group == streams;
    const auto* source = static_cast<const uint16_t*>(is_embedding ? embedding.data : hidden.data);
    const Index base = row * (is_embedding ? ep : hp);
    const Index start = is_embedding || groups == 1 ? 0 : group * width;
    const Index count = !is_embedding && groups == 1 ? wide : width;
    double sum = 0;
    for (Index column = 0; column < count; ++column) {
        const double value = bf(source[base + start + column]); sum += value * value;
    }
    const float reciprocal = 1.0f / sqrtf(float(sum / double(count)) + eps);
    const auto* gain = static_cast<const float*>(is_embedding ? we.data : wh.data);
    const Index destination = row * (wide + width) + (is_embedding ? wide : group * width);
    for (Index column = 0; column < width; ++column) {
        const Index channel = is_embedding ? column : group * width + column;
        normalized[destination + column] = narrow(bf(source[base + channel]) * reciprocal * (gain[channel] + wadd));
    }
}
__global__ void project(const uint16_t* normalized, RadTensor matrix, RadTensor output,
                        float* embedding_product, Index width, Index streams, Index op, bool hidden) {
    const Index column = Index(blockIdx.x) * blockDim.x + threadIdx.x;
    const Index row = blockIdx.y, group = blockIdx.z, wide = width * streams;
    if (column >= width) return;
    const Index start = row * (wide + width) + (hidden ? group * width : wide);
    float accumulator = 0;
    for (Index inner = 0; inner < width; ++inner)
        accumulator += bf(normalized[start + inner]) * weight(matrix, column, inner, width);
    if (hidden) static_cast<uint16_t*>(output.data)[row * op + group * width + column] =
        narrow(accumulator + embedding_product[row * width + column]);
    else embedding_product[row * width + column] = accumulator;
}
__global__ void gate(DataView key, DataView query, DataView value, DataView wk,
                     DataView wq, DataView wc, DataView gv, DataView gvn,
                     Index width, Index streams, Index kp, Index qp, Index vp,
                     Index gp, Index np, float eps, float wadd, float gate_eps) {
    if (threadIdx.x) return;
    const Index row = blockIdx.x, group = blockIdx.y, start = group * width;
    double sk = 0, sq = 0;
    for (Index column = 0; column < width; ++column) {
        const double kval = load(key, row * kp + start + column);
        const double qval = load(query, row * qp + start + column);
        sk += kval * kval; sq += qval * qval;
    }
    const float rk = float(1.0 / sqrt(sk / double(width) + double(eps)));
    const float rq = float(1.0 / sqrt(sq / double(width) + double(eps)));
    double dot = 0;
    for (Index column = 0; column < width; ++column) {
        const Index channel = start + column;
        const float kval = bf(narrow(load(key, row * kp + channel) * rk * (load(wk, channel) + wadd)));
        const float qval = bf(narrow(load(query, row * qp + channel) * rq * (load(wq, channel) + wadd)));
        dot += double(kval) * double(qval);
    }
    float score = float(dot / sqrt(double(width)));
    const float magnitude = sqrtf(fmaxf(fabsf(score), gate_eps));
    score = score > 0 ? magnitude : (score < 0 ? -magnitude : 0);
    const float coefficient = 1.0f / (1.0f + expf(-score));
    double sum = 0;
    for (Index column = 0; column < width; ++column) {
        const uint16_t stored = narrow(coefficient * load(value, row * vp + column));
        static_cast<uint16_t*>(gv.data)[row * gp + start + column] = stored;
        const double rounded = bf(stored); sum += rounded * rounded;
    }
    const float reciprocal = float(1.0 / sqrt(sum / double(width) + double(eps)));
    for (Index column = 0; column < width; ++column) {
        const Index channel = start + column;
        static_cast<uint16_t*>(gvn.data)[row * np + channel] = narrow(
            load(gv, row * gp + channel) * reciprocal * (load(wc, channel) + wadd));
    }
}
// Thread 0 receives the block totals of a and b.
__device__ void block_sum(double& a, double& b, double* sa, double* sb) {
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
    for (int distance = 16; distance; distance >>= 1) {
        a += __shfl_xor(a, distance, 32); b += __shfl_xor(b, distance, 32);
    }
    __syncthreads();
    if (!lane) { sa[wave] = a; sb[wave] = b; }
    __syncthreads();
    if (threadIdx.x) return;
    a = 0; b = 0;
    for (int index = 0; index < 8; ++index) { a += sa[index]; b += sb[index]; }
}
__device__ float norm_factor(double sum, Index width, float eps) {
    return float(1.0 / sqrt(sum / double(width) + double(eps)));
}
// Serial and parallel FP64 sums of <= 4096 non-negative terms differ by < 2^-40 relative; the factor is
// monotone in the sum, so equal floats at sum*(1 -+ 2^-36) fix the serial result.
__device__ bool norm_certified(double sum, Index width, float eps, float& factor) {
    if (!(sum < 0x1p1000)) return false;
    factor = norm_factor(sum * (1 - 0x1p-36), width, eps);
    return __float_as_uint(factor) == __float_as_uint(norm_factor(sum * (1 + 0x1p-36), width, eps));
}
__device__ float score_of(double dot, Index width) { return float(dot / sqrt(double(width))); }
// gate with one 256-thread block per (row, stream); each FP64 sum is accepted only when certified, else
// thread 0 reruns gate's serial chain.
__global__ __launch_bounds__(256) void gate_parallel(DataView key, DataView query, DataView value, DataView wk,
                     DataView wq, DataView wc, DataView gv, DataView gvn,
                     Index width, Index streams, Index kp, Index qp, Index vp,
                     Index gp, Index np, float eps, float wadd, float gate_eps) {
    __shared__ double sa[8], sb[8];
    __shared__ float shared_rk, shared_rq, shared_coefficient, shared_reciprocal;
    const Index row = blockIdx.x, group = blockIdx.y, start = group * width;
    const int tid = threadIdx.x;
    double sk = 0, sq = 0;
    for (Index column = tid; column < width; column += 256) {
        const double kval = load(key, row * kp + start + column), qval = load(query, row * qp + start + column);
        sk += kval * kval; sq += qval * qval;
    }
    block_sum(sk, sq, sa, sb);
    if (tid == 0) {
        float rk = 0, rq = 0;
        if (!norm_certified(sk, width, eps, rk) || !norm_certified(sq, width, eps, rq)) {
            sk = 0; sq = 0;
            for (Index column = 0; column < width; ++column) {
                const double kval = load(key, row * kp + start + column);
                const double qval = load(query, row * qp + start + column);
                sk += kval * kval; sq += qval * qval;
            }
            rk = norm_factor(sk, width, eps); rq = norm_factor(sq, width, eps);
        }
        shared_rk = rk; shared_rq = rq;
    }
    __syncthreads();
    const float rk = shared_rk, rq = shared_rq;
    double dot = 0, magnitude = 0;
    for (Index column = tid; column < width; column += 256) {
        const Index channel = start + column;
        const float kval = bf(narrow(load(key, row * kp + channel) * rk * (load(wk, channel) + wadd)));
        const float qval = bf(narrow(load(query, row * qp + channel) * rq * (load(wq, channel) + wadd)));
        const double term = double(kval) * double(qval);
        dot += term; magnitude += fabs(term);
    }
    block_sum(dot, magnitude, sa, sb);
    if (tid == 0) {
        const double error = magnitude * 0x1p-38;
        float score = score_of(dot - error, width);
        if (!(magnitude < 0x1p1000) || __float_as_uint(score) != __float_as_uint(score_of(dot + error, width))) {
            dot = 0;
            for (Index column = 0; column < width; ++column) {
                const Index channel = start + column;
                const float kval = bf(narrow(load(key, row * kp + channel) * rk * (load(wk, channel) + wadd)));
                const float qval = bf(narrow(load(query, row * qp + channel) * rq * (load(wq, channel) + wadd)));
                dot += double(kval) * double(qval);
            }
            score = score_of(dot, width);
        }
        const float magnitude_score = sqrtf(fmaxf(fabsf(score), gate_eps));
        score = score > 0 ? magnitude_score : (score < 0 ? -magnitude_score : 0);
        shared_coefficient = 1.0f / (1.0f + expf(-score));
    }
    __syncthreads();
    const float coefficient = shared_coefficient;
    double sum = 0, unused = 0;
    for (Index column = tid; column < width; column += 256) {
        const uint16_t stored = narrow(coefficient * load(value, row * vp + column));
        static_cast<uint16_t*>(gv.data)[row * gp + start + column] = stored;
        const double rounded = bf(stored);
        sum += rounded * rounded;
    }
    block_sum(sum, unused, sa, sb);
    if (tid == 0) {
        float reciprocal = 0;
        if (!norm_certified(sum, width, eps, reciprocal)) {
            sum = 0;
            for (Index column = 0; column < width; ++column) {
                const double rounded = bf(narrow(coefficient * load(value, row * vp + column)));
                sum += rounded * rounded;
            }
            reciprocal = norm_factor(sum, width, eps);
        }
        shared_reciprocal = reciprocal;
    }
    __syncthreads();
    const float reciprocal = shared_reciprocal;
    for (Index column = tid; column < width; column += 256) {
        const Index channel = start + column;
        const float stored = bf(narrow(coefficient * load(value, row * vp + column)));
        static_cast<uint16_t*>(gvn.data)[row * np + channel] = narrow(stored * reciprocal * (load(wc, channel) + wadd));
    }
}
__global__ void grid(RadTensor output, RadTensor table, RadTensor coordinates,
                     Index width, Index side, Index count, Index op, Index tp) {
    const Index index = Index(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count * width) return;
    const Index row = index / width, column = index % width;
    const auto* coord = static_cast<const int32_t*>(coordinates.data);
    Index taps[2][2]; float weights[2][2];
    for (int axis = 0; axis < 2; ++axis) {
        const int position = coord[axis * coordinates.stride[0] + row * coordinates.stride[1]];
        const int extent = coord[(axis + 2) * coordinates.stride[0] + row * coordinates.stride[1]];
        const float source = float(position) * float(side - 1) / float(extent - 1 > 1 ? extent - 1 : 1);
        const float floor = floorf(source);
        for (int tap = 0; tap < 2; ++tap) {
            const Index at = Index(floor) + tap;
            taps[axis][tap] = at < 0 ? 0 : (at >= side ? side - 1 : at);
            weights[axis][tap] = fmaxf(1.0f - fabsf(source - floor - float(tap)), 0.0f);
        }
    }
    float accumulator = 0;
    for (int vertical = 0; vertical < 2; ++vertical)
        for (int horizontal = 0; horizontal < 2; ++horizontal)
            accumulator += load(table, (taps[0][vertical] * side + taps[1][horizontal]) * tp + column) *
                (weights[0][vertical] * weights[1][horizontal]);
    static_cast<uint16_t*>(output.data)[row * op + column] = narrow(load(output, row * op + column) + bf(narrow(accumulator)));
}
__global__ void gather(RadTensor tokens, RadTensor table, RadTensor scale, RadTensor output,
                       Index width, Index vocabulary, Index offset, Index count, Index op, Index tp) {
    const Index index = Index(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count * width) return;
    const Index row = index / width, column = index % width;
    const Index token = Index(static_cast<const int32_t*>(tokens.data)[row]) - offset;
    const float multiplier = scale.data ? load(scale, 0) : 1.0f;
    const float result = token >= 0 && token < vocabulary ? load(table, token * tp + column) * multiplier : 0;
    static_cast<uint16_t*>(output.data)[row * op + column] = narrow(result);
}
int launched() { return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE; }
bool weight_ok(RadTensor* tensor, Index width, bool packed) {
    if (!tensor || tensor->rank != 2 || tensor->shape[0] != width || tensor->stride[1] != 1) return false;
    if (!packed) return tensor->dtype == RAD_BF16 && tensor->shape[1] == width && tensor->stride[0] >= width;
    const Index bytes = (width + 4 * (width / 128) + 15) & ~Index(15);
    if (width % 128 || uintptr_t(tensor->data) % 4) return false;
    if (tensor->dtype == r3dh_mtp_fc_e4m3_dtype && tensor->shape[1] == width && tensor->stride[0] == width) {
        tensor->shape[1] = bytes;
        tensor->stride[0] = bytes;
        return true;
    }
    return (tensor->dtype == RAD_F8E4M3 || tensor->dtype == r3dh_mtp_fc_e4m3_dtype) &&
        tensor->shape[1] == bytes && tensor->stride[0] == bytes;
}
}
using namespace r3dh_mtp_ple_detail;
extern "C" int r3dh_mtp_ple_device(const RadArgs* args, RadStream stream) {
    if (!args || !args->t || args->n_t <= 0 || args->n_t > 16) return RAD_E_INVAL;
    int current = -1;
    hipDeviceProp_t properties{};
    if (hipGetDevice(&current) != hipSuccess ||
        hipGetDeviceProperties(&properties, current) != hipSuccess) return RAD_E_DEVICE;
    if (std::strncmp(properties.gcnArchName, "gfx1100", 7) ||
        (properties.gcnArchName[7] && properties.gcnArchName[7] != ':')) return RAD_E_DEVICE;
    return RAD_OK;
}
extern "C" int64_t r3dh_mtp_ple_mtp_enter_scratch(const RadArgs* args) {
    const Index width = integer(args, "n"), streams = integer(args, "hc"), band = integer(args, "M");
    if (width <= 0 || width > 4096 || streams < 2 || streams > 8 || band <= 0 || band > 1048576) return 0;
    return band * width * ((streams + 1) * 2 + 4);
}
extern "C" int r3dh_mtp_ple_mtp_enter(const RadArgs* args, RadStream stream) {
    const auto *hidden = operand(args, 0), *embedding = operand(args, 1), *wh = operand(args, 2),
        *we = operand(args, 3), *fh = operand(args, 4), *fe = operand(args, 5), *output = operand(args, 6);
    if (!hidden || !embedding || !wh || !we || !fh || !fe || !output) return RAD_E_INVAL;
    const Index width = integer(args, "n"), streams = integer(args, "hc"), groups = integer(args, "ngroup", streams);
    if (width <= 0 || width > 4096 || streams < 2 || streams > 8 || (groups != 1 && groups != streams)) return RAD_E_SHAPE;
    const Index wide = width * streams, count = rows(hidden, wide);
    if (count <= 0 || rows(embedding, width) != count || rows(output, wide) != count ||
        rad_tensor_numel(wh) != wide || rad_tensor_numel(we) != width) return RAD_E_SHAPE;
    if (hidden->dtype != RAD_BF16 || embedding->dtype != RAD_BF16 || output->dtype != RAD_BF16 ||
        wh->dtype != RAD_F32 || we->dtype != RAD_F32) return RAD_E_DTYPE;
    if (!rad_tensor_is_contiguous(wh) || !rad_tensor_is_contiguous(we)) return RAD_E_STRIDE;
    const char* fc = rad_args_gets_or(args, "fc", "");
    const bool packed = !std::strcmp(fc, "e4m3");
    if (!packed && std::strcmp(fc, "bf16")) return RAD_E_UNSUPPORTED;
    RadTensor fc_hidden = *fh, fc_embedding = *fe;
    if (!weight_ok(&fc_hidden, width, packed) || !weight_ok(&fc_embedding, width, packed)) return RAD_E_DTYPE;
    const Index bytes = count * width * ((streams + 1) * 2 + 4);
    if (!args->scratch || args->scratch_bytes < bytes || uintptr_t(args->scratch) % 4) return RAD_E_SCRATCH;
    auto* normalized = static_cast<uint16_t*>(args->scratch);
    auto* product = reinterpret_cast<float*>(normalized + count * width * (streams + 1));
    const auto queue = reinterpret_cast<hipStream_t>(stream);
    const int device = r3dh_mtp_ple_device(args, stream);
    if (device != RAD_OK) return device;
    norm<<<dim3(count, streams + 1), 1, 0, queue>>>(*hidden, *embedding, *wh, *we, normalized, width,
        streams, pitch(hidden, wide), pitch(embedding, width), groups, real(args, "eps", 1e-6f), real(args, "wadd", 0));
    if (launched() != RAD_OK) return RAD_E_DEVICE;
    project<<<dim3((width + 127) / 128, count), 128, 0, queue>>>(normalized, fc_embedding, *output, product, width, streams, pitch(output, wide), false);
    if (launched() != RAD_OK) return RAD_E_DEVICE;
    project<<<dim3((width + 127) / 128, count, streams), 128, 0, queue>>>(normalized, fc_hidden, *output, product, width, streams, pitch(output, wide), true);
    return launched();
}
extern "C" int r3dh_mtp_ple_ple_gate(const RadArgs* args, RadStream stream) {
    const Index width = integer(args, "n"), streams = integer(args, "hc");
    if (width <= 0 || width > 4096 || streams <= 0 || streams > 8) return RAD_E_SHAPE;
    for (int index = 0; index < 8; ++index) {
        const auto* tensor = operand(args, index);
        if (!tensor) return RAD_E_INVAL;
        if (tensor->dtype != (index >= 3 && index <= 5 ? RAD_F32 : RAD_BF16)) return RAD_E_DTYPE;
    }
    const auto* tensors = args->t; const Index wide = width * streams, count = rows(tensors + 2, width);
    if (count <= 0) return RAD_E_SHAPE;
    for (int index : {0, 1, 6, 7}) if (rows(tensors + index, wide) != count) return RAD_E_SHAPE;
    for (int index : {3, 4, 5}) if (rad_tensor_numel(tensors + index) != wide || !rad_tensor_is_contiguous(tensors + index)) return RAD_E_SHAPE;
    const int device = r3dh_mtp_ple_device(args, stream);
    if (device != RAD_OK) return device;
    const bool parallel = r3d_exact_path("RAD_PLE_GATE_PARALLEL");
    if (parallel)
        gate_parallel<<<dim3(count, streams), 256, 0, reinterpret_cast<hipStream_t>(stream)>>>(tensors[0], tensors[1], tensors[2], tensors[3], tensors[4], tensors[5], tensors[6], tensors[7],
            width, streams, pitch(tensors, wide), pitch(tensors + 1, wide), pitch(tensors + 2, width), pitch(tensors + 6, wide), pitch(tensors + 7, wide),
            real(args, "eps", 1e-6f), real(args, "wadd", 0), real(args, "gate_eps", 1e-6f));
    else
        gate<<<dim3(count, streams), 1, 0, reinterpret_cast<hipStream_t>(stream)>>>(tensors[0], tensors[1], tensors[2], tensors[3], tensors[4], tensors[5], tensors[6], tensors[7],
            width, streams, pitch(tensors, wide), pitch(tensors + 1, wide), pitch(tensors + 2, width), pitch(tensors + 6, wide), pitch(tensors + 7, wide),
            real(args, "eps", 1e-6f), real(args, "wadd", 0), real(args, "gate_eps", 1e-6f));
    return launched();
}
extern "C" int r3dh_mtp_ple_grid_embed(const RadArgs* args, RadStream stream) {
    const auto *output = operand(args, 0), *table = operand(args, 1), *coord = operand(args, 2);
    if (!output || !table || !coord) return RAD_E_INVAL;
    const Index width = integer(args, "n"), side = integer(args, "side");
    if (width <= 0 || side <= 0 || side > 4096) return RAD_E_SHAPE;
    const Index count = rows(output, width);
    if (count <= 0 || rows(table, width) != side * side || coord->rank != 2 || coord->shape[0] != 4 || coord->shape[1] < count) return RAD_E_SHAPE;
    if (output->dtype != RAD_BF16 || table->dtype != RAD_BF16 || coord->dtype != RAD_I32) return RAD_E_DTYPE;
    const int device = r3dh_mtp_ple_device(args, stream);
    if (device != RAD_OK) return device;
    grid<<<(count * width + 255) / 256, 256, 0, reinterpret_cast<hipStream_t>(stream)>>>(*output, *table, *coord, width, side, count, pitch(output, width), pitch(table, width));
    return launched();
}
extern "C" int r3dh_mtp_ple_embed_lookup_q(const RadArgs* args, RadStream stream) {
    const auto *tokens = operand(args, 0), *table = operand(args, 1), *scale = operand(args, 2), *output = operand(args, 3);
    if (!tokens || !table || !output) return RAD_E_INVAL;
    const Index width = integer(args, "n_embd"), vocabulary = integer(args, "n_vocab"), offset = integer(args, "vocab_offset");
    if (width <= 0 || vocabulary <= 0 || offset < 0) return RAD_E_SHAPE;
    const Index count = rows(output, width);
    if (count <= 0 || rows(table, width) != vocabulary || rad_tensor_numel(tokens) < count) return RAD_E_SHAPE;
    if (!rad_tensor_is_contiguous(tokens)) return RAD_E_STRIDE;
    if (tokens->dtype != RAD_I32 || output->dtype != RAD_BF16 || (table->dtype != RAD_BF16 && table->dtype != RAD_F8E4M3)) return RAD_E_DTYPE;
    if (scale && (rad_tensor_numel(scale) != 1 || (scale->dtype != RAD_BF16 && scale->dtype != RAD_F32))) return RAD_E_DTYPE;
    RadTensor absent{};
    const int device = r3dh_mtp_ple_device(args, stream);
    if (device != RAD_OK) return device;
    gather<<<(count * width + 255) / 256, 256, 0, reinterpret_cast<hipStream_t>(stream)>>>(*tokens, *table, scale ? *scale : absent, *output,
        width, vocabulary, offset, count, pitch(output, width), pitch(table, width));
    return launched();
}
