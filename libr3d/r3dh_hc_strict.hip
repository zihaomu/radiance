#include "r3dh_hc_api.h"
#include "r3dh_hc_args.h"
#include <hip/hip_runtime.h>
#include <climits>

#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ > 0)
#error "Strict HC requires -fno-fast-math -ffp-contract=off -fno-associative-math"
#endif

#pragma clang fp contract(off)
#pragma clang fp reassociate(off)

namespace {
constexpr int width = 2560;
constexpr int streams = 4;
constexpr int wide = width * streams;
constexpr int rank = 320;
constexpr int down_pitch = 10560;
constexpr int up_pitch = 336;

size_t aligned(size_t bytes) { return (bytes + 255) & ~size_t(255); }

__device__ float widen_bf16(uint16_t value) {
    return __uint_as_float(uint32_t(value) << 16);
}
__device__ uint16_t narrow_bf16(float value) {
    const uint32_t bits = __float_as_uint(value);
    if ((bits & 0x7fffffffU) > 0x7f800000U) return uint16_t((bits >> 16) | 64);
    return uint16_t((bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16);
}
__device__ float e4m3(uint8_t code) {
    const unsigned exponent = (code >> 3) & 15, mantissa = code & 7;
    const float magnitude = exponent ? __uint_as_float(((exponent + 120) << 23) | (mantissa << 20))
                                     : float(mantissa) * 0x1p-9f;
    return code & 128 ? -magnitude : magnitude;
}
__device__ float sigmoid_exact(float value) {
    const float exponential = float(::exp(double(-value)));
    return __fdiv_rn(1.0f, 1.0f + exponential);
}

// One block per row: the row is staged once, each stream's double sum stays a single sequential chain.
__global__ __launch_bounds__(256) void strict_norm(const uint16_t* hidden, const float* weights,
                                                   uint16_t* normalized, int rows, float eps, float wadd) {
    __shared__ __attribute__((aligned(16))) uint16_t staged[wide];
    __shared__ float stream_scale[streams];
    const int row = blockIdx.x, lane = threadIdx.x;
    if (row >= rows) return;
    const uint16_t* source = hidden + size_t(row) * wide;
    for (int index = lane; index < wide; index += 256) staged[index] = source[index];
    __syncthreads();
    if ((lane & 31) == 0 && lane < 32 * streams) {
        const int channel_stream = lane >> 5;
        const uint4* words = reinterpret_cast<const uint4*>(staged) + channel_stream * (width / 8);
        double squares = 0;
        for (int block = 0; block < width / 8; ++block) {
            const uint4 packed = words[block];
            const uint32_t pair[4] = {packed.x, packed.y, packed.z, packed.w};
#pragma unroll
            for (int half = 0; half < 4; ++half) {
                const double low = double(__uint_as_float(pair[half] << 16));
                squares += low * low;
                const double high = double(__uint_as_float(pair[half] & 0xffff0000U));
                squares += high * high;
            }
        }
        stream_scale[channel_stream] = __fdiv_rn(1.0f, sqrtf(float(squares / double(width)) + eps));
    }
    __syncthreads();
    uint16_t* target = normalized + size_t(row) * wide;
    for (int index = lane; index < wide; index += 256)
        target[index] = narrow_bf16((widen_bf16(staged[index]) * stream_scale[index / width]) *
                                    (weights[index] + wadd));
}

// MODE 0 is down (K=10240), MODE 1 is up (K=320). A block owns a tile of rows x outputs; each E4M3 weight
// is decoded once per tile into LDS. Every output keeps one register accumulator fed in ascending k order.
template <int MODE, int KC, int NTN, int NTM, int RN, int RM>
__global__ __launch_bounds__(NTN * NTM) void strict_tiled(const uint16_t* activations, const uint8_t* matrix,
                                                         int rows, uint16_t* result, uint16_t* trace_values,
                                                         float* trace_logits) {
    constexpr bool down = MODE == 0;
    constexpr int depth = down ? wide : rank, outputs = down ? rank : wide;
    constexpr int group = down ? 128 : 80, pitch = down ? down_pitch : up_pitch;
    constexpr int threads = NTN * NTM, tile_n = NTN * RN, tile_m = NTM * RM;
    constexpr int w_total = tile_n * KC / 4, x_total = tile_m * KC / 2;
    constexpr int w_items = (w_total + threads - 1) / threads, x_items = (x_total + threads - 1) / threads;
    static_assert(depth % KC == 0 && outputs % tile_n == 0 && KC % 4 == 0 && group % 4 == 0);
    __shared__ float lut[256];
    __shared__ float wtile[KC][tile_n + 1];
    __shared__ __attribute__((aligned(16))) float xtile[tile_m][KC + 4];
    const int lane = threadIdx.x, tn = lane % NTN, tm = lane / NTN;
    const int n0 = blockIdx.x * tile_n, m0 = blockIdx.y * tile_m;
    for (int code = lane; code < 256; code += threads) lut[code] = e4m3(uint8_t(code));
    uint32_t codes[w_items], pairs[x_items];
    float scales[w_items];
    auto fetch = [&](int k0) {
#pragma unroll
        for (int slot = 0; slot < w_items; ++slot) {
            const int item = lane + threads * slot;
            if (w_total % threads == 0 || item < w_total) {
                const int k = k0 + 4 * (item % (KC / 4));
                const uint8_t* line = matrix + size_t(n0 + item / (KC / 4)) * pitch;
                codes[slot] = *reinterpret_cast<const uint32_t*>(line + k);
                scales[slot] = reinterpret_cast<const float*>(line + depth)[k / group];
            }
        }
#pragma unroll
        for (int slot = 0; slot < x_items; ++slot) {
            const int item = lane + threads * slot, m = m0 + item / (KC / 2);
            if (x_total % threads == 0 || item < x_total)
                pairs[slot] = m < rows ? *reinterpret_cast<const uint32_t*>(
                    activations + size_t(m) * depth + k0 + 2 * (item % (KC / 2))) : 0U;
        }
    };
    auto store = [&]() {
#pragma unroll
        for (int slot = 0; slot < w_items; ++slot) {
            const int item = lane + threads * slot;
            if (w_total % threads == 0 || item < w_total) {
                const int n = item / (KC / 4), k = 4 * (item % (KC / 4));
#pragma unroll
                for (int byte = 0; byte < 4; ++byte)
                    wtile[k + byte][n] = lut[(codes[slot] >> (8 * byte)) & 255U] * scales[slot];
            }
        }
#pragma unroll
        for (int slot = 0; slot < x_items; ++slot) {
            const int item = lane + threads * slot;
            if (x_total % threads == 0 || item < x_total) {
                const int m = item / (KC / 2), k = 2 * (item % (KC / 2));
                xtile[m][k] = __uint_as_float(pairs[slot] << 16);
                xtile[m][k + 1] = __uint_as_float(pairs[slot] & 0xffff0000U);
            }
        }
    };
    float sums[RN][RM];
#pragma unroll
    for (int i = 0; i < RN; ++i)
#pragma unroll
        for (int j = 0; j < RM; ++j) sums[i][j] = 0;
    fetch(0);
    for (int k0 = 0; k0 < depth; k0 += KC) {
        __syncthreads();
        store();
        __syncthreads();
        if (k0 + KC < depth) fetch(k0 + KC);
#pragma unroll 4
        for (int k = 0; k < KC; k += 4) {
            float xs[RM][4];
#pragma unroll
            for (int j = 0; j < RM; ++j) {
                const float4 packed = *reinterpret_cast<const float4*>(&xtile[tm + NTM * j][k]);
                xs[j][0] = packed.x; xs[j][1] = packed.y; xs[j][2] = packed.z; xs[j][3] = packed.w;
            }
#pragma unroll
            for (int step = 0; step < 4; ++step)
#pragma unroll
                for (int i = 0; i < RN; ++i) {
                    const float weight = wtile[k + step][tn + NTN * i];
#pragma unroll
                    for (int j = 0; j < RM; ++j) sums[i][j] += xs[j][step] * weight;
                }
        }
    }
#pragma unroll
    for (int i = 0; i < RN; ++i)
#pragma unroll
        for (int j = 0; j < RM; ++j) {
            const int m = m0 + tm + NTM * j, n = n0 + tn + NTN * i;
            if (m >= rows) continue;
            const float sum = sums[i][j];
            if constexpr (down) {
                const float divided = sum / float(streams);
                const uint16_t value = narrow_bf16(divided * sigmoid_exact(divided));
                result[size_t(m) * rank + n] = value;
                if (trace_values) trace_values[size_t(m) * rank + n] = value;
            } else {
                const size_t index = size_t(m) * wide + n;
                const uint16_t gate = narrow_bf16(sigmoid_exact(sum));
                result[index] = gate;
                if (trace_logits) trace_logits[index] = sum;
                if (trace_values) trace_values[index] = gate;
            }
        }
}
template <int MODE, int KC, int NTN, int NTM, int RN, int RM>
void launch_tiled(hipStream_t stream, int rows, const uint16_t* activations, const uint8_t* matrix,
                  uint16_t* result, uint16_t* trace_values, float* trace_logits) {
    constexpr int outputs = MODE == 0 ? rank : wide, tile_m = NTM * RM;
    strict_tiled<MODE, KC, NTN, NTM, RN, RM><<<dim3(outputs / (NTN * RN), (rows + tile_m - 1) / tile_m),
        NTN * NTM, 0, stream>>>(activations, matrix, rows, result, trace_values, trace_logits);
}

__global__ void strict_mix(const uint16_t* normalized, const uint16_t* gates,
                            uint16_t* output, int rows) {
    const int channel = blockIdx.x * blockDim.x + threadIdx.x, row = blockIdx.y;
    if (channel >= width || row >= rows) return;
    float sum = 0;
    for (int channel_stream = 0; channel_stream < streams; ++channel_stream) {
        const size_t index = size_t(row) * wide + channel_stream * width + channel;
        sum += widen_bf16(gates[index]) * widen_bf16(normalized[index]);
    }
    output[size_t(row) * width + channel] = narrow_bf16(sum / float(streams));
}

__global__ void strict_inject_serial(const uint16_t* normalized, const uint16_t* weights,
                                     uint16_t* output, int rows) {
    const int row = blockIdx.x, channel_stream = threadIdx.x;
    if (row >= rows || channel_stream >= streams) return;
    float sum = 0;
    for (int channel = 0; channel < wide; ++channel)
        sum += widen_bf16(normalized[size_t(row) * wide + channel]) *
               widen_bf16(weights[channel_stream * wide + channel]);
    output[size_t(row) * streams + channel_stream] =
        narrow_bf16(2.0f * sigmoid_exact(sum / float(streams)));
}
// Same per-(row, stream) sequential sum as the serial form, with operands staged through LDS in chunks.
template <int KC, int TM, int THREADS>
__global__ __launch_bounds__(THREADS) void strict_inject(const uint16_t* normalized, const uint16_t* weights,
                                                         uint16_t* output, int rows) {
    constexpr int w_total = streams * KC / 2, x_total = TM * KC / 2;
    constexpr int w_items = (w_total + THREADS - 1) / THREADS, x_items = (x_total + THREADS - 1) / THREADS;
    static_assert(wide % KC == 0 && KC % 4 == 0 && streams * TM <= THREADS);
    __shared__ __attribute__((aligned(16))) float wtile[streams][KC + 4];
    __shared__ __attribute__((aligned(16))) float xtile[TM][KC + 4];
    const int lane = threadIdx.x, m0 = blockIdx.x * TM;
    const int channel_stream = lane % streams, local = lane / streams;
    uint32_t wpairs[w_items], xpairs[x_items];
    auto fetch = [&](int k0) {
#pragma unroll
        for (int slot = 0; slot < w_items; ++slot) {
            const int item = lane + THREADS * slot;
            if (w_total % THREADS == 0 || item < w_total)
                wpairs[slot] = *reinterpret_cast<const uint32_t*>(
                    weights + size_t(item / (KC / 2)) * wide + k0 + 2 * (item % (KC / 2)));
        }
#pragma unroll
        for (int slot = 0; slot < x_items; ++slot) {
            const int item = lane + THREADS * slot, m = m0 + item / (KC / 2);
            if (x_total % THREADS == 0 || item < x_total)
                xpairs[slot] = m < rows ? *reinterpret_cast<const uint32_t*>(
                    normalized + size_t(m) * wide + k0 + 2 * (item % (KC / 2))) : 0U;
        }
    };
    auto store = [&]() {
#pragma unroll
        for (int slot = 0; slot < w_items; ++slot) {
            const int item = lane + THREADS * slot;
            if (w_total % THREADS == 0 || item < w_total) {
                const int s = item / (KC / 2), k = 2 * (item % (KC / 2));
                wtile[s][k] = __uint_as_float(wpairs[slot] << 16);
                wtile[s][k + 1] = __uint_as_float(wpairs[slot] & 0xffff0000U);
            }
        }
#pragma unroll
        for (int slot = 0; slot < x_items; ++slot) {
            const int item = lane + THREADS * slot;
            if (x_total % THREADS == 0 || item < x_total) {
                const int m = item / (KC / 2), k = 2 * (item % (KC / 2));
                xtile[m][k] = __uint_as_float(xpairs[slot] << 16);
                xtile[m][k + 1] = __uint_as_float(xpairs[slot] & 0xffff0000U);
            }
        }
    };
    float sum = 0;
    fetch(0);
    for (int k0 = 0; k0 < wide; k0 += KC) {
        __syncthreads();
        store();
        __syncthreads();
        if (k0 + KC < wide) fetch(k0 + KC);
        if (local < TM) {
#pragma unroll 8
            for (int k = 0; k < KC; k += 4) {
                const float4 x = *reinterpret_cast<const float4*>(&xtile[local][k]);
                const float4 w = *reinterpret_cast<const float4*>(&wtile[channel_stream][k]);
                sum += x.x * w.x;
                sum += x.y * w.y;
                sum += x.z * w.z;
                sum += x.w * w.w;
            }
        }
    }
    if (local < TM && m0 + local < rows)
        output[size_t(m0 + local) * streams + channel_stream] =
            narrow_bf16(2.0f * sigmoid_exact(sum / float(streams)));
}
__device__ uint8_t encode_e4m3(float value) {
    const unsigned sign = (__float_as_uint(value) >> 24) & 128;
    const float magnitude = fminf(fabsf(value), 448.0f);
    int lower = 0, upper = 126;
    while (lower + 1 < upper) {
        const int middle = (lower + upper) / 2;
        if (e4m3(uint8_t(middle)) <= magnitude) lower = middle;
        else upper = middle;
    }
    const float lower_distance = magnitude - e4m3(uint8_t(lower));
    const float upper_distance = e4m3(uint8_t(upper)) - magnitude;
    const int selected = lower_distance < upper_distance ||
        (lower_distance == upper_distance && !(lower & 1)) ? lower : upper;
    return uint8_t(sign | selected);
}

__global__ void strict_quant(const uint16_t* output, uint8_t* codes, float* scales,
                              bool rotated, bool int8_codes) {
    __shared__ float values[128];
    __shared__ float scale;
    const int lane = threadIdx.x;
    const size_t group_index = size_t(blockIdx.y) * (width / 128) + blockIdx.x;
    const size_t base = group_index * 128;
    values[lane] = widen_bf16(output[base + lane]);
    __syncthreads();
    if (rotated) {
        for (int step = 1; step < 128; step *= 2) {
            const float first = values[lane & ~step], second = values[lane | step];
            __syncthreads();
            values[lane] = lane & step ? first - second : first + second;
            __syncthreads();
        }
    }
    const float top = int8_codes ? 127.0f : 448.0f;
    if (lane == 0) {
        float maximum = 0;
        for (int index = 0; index < 128; ++index)
            if (fabsf(values[index]) > maximum) maximum = fabsf(values[index]);
        scale = maximum > 0 ? maximum * __fdiv_rn(1.0f, top) : 1.0f;
        scales[group_index] = scale;
    }
    __syncthreads();
    const float value = fminf(fmaxf(values[lane] * __fdiv_rn(1.0f, scale), -top), top);
    codes[base + lane] = int8_codes ? uint8_t(__float2int_rn(value)) : encode_e4m3(value);
}


bool dense(const RadTensor* tensor, uint32_t dtype, int64_t count) {
    if (!tensor || tensor->dtype != dtype || tensor->rank < 1 || tensor->rank > RAD_MAX_RANK)
        return false;
    int64_t stride = 1;
    for (int axis = int(tensor->rank) - 1; axis >= 0; --axis) {
        if (tensor->shape[axis] <= 0 || tensor->stride[axis] != stride) return false;
        if (tensor->shape[axis] > INT64_MAX / stride) return false;
        stride *= tensor->shape[axis];
    }
    return stride >= count;
}

int strict_launch(const RadArgs* args, RadStream stream, const R3dhHcTrace* trace) {
    if (!args || !args->t || args->n_t < 6) return RAD_E_INVAL;
    const int64_t bytes = r3dh_hc_rad_hc_read_strict_scratch(args);
    if (!bytes) return RAD_E_UNSUPPORTED;
    const RadTensor* hidden = rad_arg_in(args, 0), *weights = rad_arg_in(args, 1);
    const RadTensor* down = rad_arg_in(args, 2), *up = rad_arg_in(args, 3);
    const RadTensor* inject_weights = rad_arg_in(args, 4), *output = rad_arg_in(args, 5);
    const RadTensor* inject = rad_arg_in(args, 6);
    if (!hidden || !weights || !down || !up || !output) return RAD_E_INVAL;
    if (!dense(hidden, RAD_BF16, wide)) return RAD_E_DTYPE;
    const int64_t elements = rad_tensor_numel(hidden), rows = elements / wide;
    if (rows < 1 || elements % wide || rows > r3dh_hc_geti_or(args, "M", 0)) return RAD_E_SHAPE;
    if (!dense(hidden, RAD_BF16, rows * wide) || !dense(weights, RAD_F32, wide) ||
        !dense(down, RAD_F8E4M3, rank * wide) || !dense(up, RAD_F8E4M3, wide * rank) ||
        !dense(output, RAD_BF16, rows * width)) return RAD_E_DTYPE;
    if (down->rank != 2 || down->shape[0] != rank || down->shape[1] != wide ||
        up->rank != 2 || up->shape[0] != wide || up->shape[1] != rank) return RAD_E_SHAPE;
    if (bool(inject_weights) != bool(inject)) return RAD_E_INVAL;
    if (inject_weights && (!dense(inject_weights, RAD_BF16, streams * wide) ||
                           !dense(inject, RAD_BF16, rows * streams))) return RAD_E_DTYPE;
    const RadTensor* code_outputs[] = {rad_arg_in(args, 7), rad_arg_in(args, 9)};
    const RadTensor* scale_outputs[] = {rad_arg_in(args, 8), rad_arg_in(args, 10)};
    for (int pair = 0; pair < 2; ++pair) {
        if (bool(code_outputs[pair]) != bool(scale_outputs[pair])) return RAD_E_INVAL;
        if (!code_outputs[pair]) continue;
        if (r3dh_hc_geti_or(args, "group", 0) != 128) return RAD_E_UNSUPPORTED;
        const uint32_t dtype = code_outputs[pair]->dtype;
        if ((dtype != RAD_I8 && dtype != RAD_F8E4M3) ||
            !dense(code_outputs[pair], dtype, rows * width) ||
            !dense(scale_outputs[pair], RAD_F32, rows * (width / 128))) return RAD_E_DTYPE;
    }
    if (trace && (!trace->down || !trace->logits || !trace->gates || trace->rows != rows ||
                  trace->n != width || trace->hc != streams || trace->lowrank != rank)) return RAD_E_SHAPE;
    const float eps = float(r3dh_hc_getf(args, "eps", 1e-6));
    const float wadd = float(r3dh_hc_getf(args, "wadd", 0));
    if (!std::isfinite(eps) || eps < 0 || !std::isfinite(wadd)) return RAD_E_INVAL;
    if (!args->scratch || args->scratch_bytes < bytes || (uintptr_t(args->scratch) & 255)) return RAD_E_SCRATCH;
    auto* normalized = static_cast<uint16_t*>(args->scratch);
    const size_t band = size_t(r3dh_hc_geti_or(args, "M", 0));
    auto* intermediate = reinterpret_cast<uint16_t*>(static_cast<uint8_t*>(args->scratch) + aligned(band * wide * 2));
    auto* gates = reinterpret_cast<uint16_t*>(reinterpret_cast<uint8_t*>(intermediate) + aligned(band * rank * 2));
    const auto device_stream = reinterpret_cast<hipStream_t>(stream);
    const int count = int(rows);
    const auto* down_data = static_cast<const uint8_t*>(down->data);
    const auto* up_data = static_cast<const uint8_t*>(up->data);
    uint16_t* down_trace = trace ? trace->down : nullptr;
    float* up_logits = trace ? trace->logits : nullptr;
    uint16_t* up_gates = trace ? trace->gates : nullptr;
    strict_norm<<<dim3(rows), 256, 0, device_stream>>>(static_cast<const uint16_t*>(hidden->data),
        static_cast<const float*>(weights->data), normalized, count, eps, wadd);
    if (rows <= 4) {
        launch_tiled<0, 128, 16, 4, 1, 1>(device_stream, count, normalized, down_data, intermediate, down_trace, nullptr);
        launch_tiled<1, 64, 64, 4, 1, 1>(device_stream, count, intermediate, up_data, gates, up_gates, up_logits);
    } else if (rows <= 8) {
        launch_tiled<0, 128, 16, 8, 1, 1>(device_stream, count, normalized, down_data, intermediate, down_trace, nullptr);
        launch_tiled<1, 64, 32, 8, 1, 1>(device_stream, count, intermediate, up_data, gates, up_gates, up_logits);
    } else if (rows <= 16) {
        launch_tiled<0, 128, 16, 16, 1, 1>(device_stream, count, normalized, down_data, intermediate, down_trace, nullptr);
        launch_tiled<1, 64, 16, 16, 1, 1>(device_stream, count, intermediate, up_data, gates, up_gates, up_logits);
    } else if (rows <= 32) {
        launch_tiled<0, 128, 16, 16, 1, 2>(device_stream, count, normalized, down_data, intermediate, down_trace, nullptr);
        launch_tiled<1, 64, 16, 16, 1, 2>(device_stream, count, intermediate, up_data, gates, up_gates, up_logits);
    } else {
        launch_tiled<0, 64, 16, 16, 4, 2>(device_stream, count, normalized, down_data, intermediate, down_trace, nullptr);
        launch_tiled<1, 64, 16, 16, 4, 2>(device_stream, count, intermediate, up_data, gates, up_gates, up_logits);
    }
    strict_mix<<<dim3((width + 127) / 128, rows), 128, 0, device_stream>>>(normalized, gates,
        static_cast<uint16_t*>(output->data), count);
    if (inject_weights) {
        const auto* inject_data = static_cast<const uint16_t*>(inject_weights->data);
        auto* inject_out = static_cast<uint16_t*>(inject->data);
        if (uintptr_t(inject_data) & 3)
            strict_inject_serial<<<dim3(rows), 4, 0, device_stream>>>(normalized, inject_data, inject_out, count);
        else
            strict_inject<256, 16, 128><<<dim3((rows + 15) / 16), 128, 0, device_stream>>>(normalized,
                inject_data, inject_out, count);
    }
    for (int pair = 0; pair < 2; ++pair)
        if (code_outputs[pair])
            strict_quant<<<dim3(width / 128, rows), 128, 0, device_stream>>>(
                static_cast<const uint16_t*>(output->data), static_cast<uint8_t*>(code_outputs[pair]->data),
                static_cast<float*>(scale_outputs[pair]->data), pair == 1, code_outputs[pair]->dtype == RAD_I8);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
}

extern "C" int64_t r3dh_hc_rad_hc_read_strict_scratch(const RadArgs* args) {
    if (!args || r3dh_hc_geti_or(args, "n", 0) != width || r3dh_hc_geti_or(args, "hc", 0) != streams ||
        r3dh_hc_geti_or(args, "lowrank", 0) != rank) return 0;
    const char* mix = r3dh_hc_gets(args, "mix"), *dtype = r3dh_hc_gets(args, "dtype");
    if (!mix || std::strcmp(mix, "e4m3") || (dtype && std::strcmp(dtype, "bf16"))) return 0;
    const int64_t band = r3dh_hc_geti_or(args, "M", 0);
    if (band < 1 || band > 65535) return 0;
    return int64_t(aligned(size_t(band) * wide * 2) * 2 + aligned(size_t(band) * rank * 2));
}
extern "C" int r3dh_hc_rad_hc_read_strict(const RadArgs* args, RadStream stream) {
    return strict_launch(args, stream, nullptr);
}
extern "C" int r3dh_hc_rad_hc_read_strict_trace(const RadArgs* args, RadStream stream, const R3dhHcTrace* trace) {
    if (!trace) return RAD_E_INVAL;
    return strict_launch(args, stream, trace);
}
extern "C" int r3dh_hc_rad_hc_read(const RadArgs* args, RadStream stream) {
    return r3dh_hc_rad_hc_read_strict(args, stream);
}
extern "C" int r3dh_hc_rad_hc_read_trace(const RadArgs* args, RadStream stream, const R3dhHcTrace* trace) {
    return r3dh_hc_rad_hc_read_strict_trace(args, stream, trace);
}
