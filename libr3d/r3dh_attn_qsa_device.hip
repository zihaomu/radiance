#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "r3dh_attn_qsa.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include "r3d_flags.h"
#if defined(__FAST_MATH__)
#error r3dh_attn_qsa requires strict floating point and -ffp-contract=off
#endif
namespace {
long integer(const RadArgs* args, const char* key, long fallback) {
    long long value = fallback;
    rad_args_geti(args, key, &value);
    return value;
}
float real(const RadArgs* args, const char* key, float fallback) {
    for (int index = 0; index < args->n_p; ++index)
        if (args->p[index].kind == RAD_P_F64 && !std::strcmp(args->p[index].key, key))
            return float(args->p[index].dval);
    return fallback;
}
bool named(const RadArgs* args, const char* key, const char* expected) {
    const char* value = rad_args_gets(args, key);
    return !value || !std::strcmp(value, expected);
}
bool vector(const RadTensor& tensor, uint32_t type, long size) {
    return tensor.data && tensor.dtype == type && tensor.rank == 1
        && tensor.shape[0] == size && tensor.stride[0] == 1;
}
bool matrix(const RadTensor& tensor, uint32_t type, long rows, long width) {
    if (!tensor.data || tensor.dtype != type || tensor.shape[0] != rows) return false;
    if (tensor.rank == 2) return tensor.shape[1] == width && tensor.stride[1] == 1
        && tensor.stride[0] >= width && tensor.stride[0] <= (1ll << 30);
    return tensor.rank == 3 && tensor.shape[1] * tensor.shape[2] == width
        && tensor.stride[2] == 1 && tensor.stride[1] == tensor.shape[2]
        && tensor.stride[0] >= width && tensor.stride[0] <= (1ll << 30);
}
bool cache(const RadTensor& tensor, int heads = 1) {
    return tensor.data && tensor.dtype == RAD_I8 && tensor.rank == 4
        && tensor.shape[0] > 0 && tensor.shape[0] <= r3dh_attn_qsa_max_cache_pages && tensor.shape[1] == heads
        && tensor.shape[2] == 4 && tensor.shape[3] == 516 && tensor.stride[3] == 1
        && tensor.stride[2] == 516 && tensor.stride[1] == 2064 && tensor.stride[0] == heads * 2064;
}
__device__ float load(const uint16_t* pointer, long offset) {
    return __uint_as_float(uint32_t(pointer[offset]) << 16);
}
__device__ uint16_t rounded(float value) {
    const uint32_t word = __float_as_uint(value);
    if ((word & 0x7fffffffu) > 0x7f800000u) return uint16_t((word >> 16) | 64u);
    return uint16_t((word + 0x7fffu + ((word >> 16) & 1u)) >> 16);
}
struct Prep {
    const uint16_t *qg, *key;
    const float *table, *wq, *wk;
    const int* positions;
    uint16_t *query, *outkey, *gate;
    long qpitch, kpitch, opitch, okpitch, gpitch, table_rows;
    int rows, heads, keyheads, dim;
    float eps, wadd;
};
__global__ void prepare(Prep call) {
    const int row = blockIdx.x, head = blockIdx.y;
    const bool key = head >= call.heads;
    const int local = key ? head - call.heads : head;
    const uint16_t* input = key ? call.key + row * call.kpitch + local * call.dim
        : call.qg + row * call.qpitch + local * 2 * call.dim;
    const float* weight = key ? call.wk : call.wq;
    uint16_t* output = key ? call.outkey + row * call.okpitch + local * call.dim
        : call.query + row * call.opitch + local * call.dim;
    float values[256];
    uint16_t gates[256];
    float sum = 0;
    for (int column = 0; column < call.dim; ++column) {
        values[column] = load(input, column);
        sum = sum + values[column] * values[column];
        if (!key && call.gate) gates[column] = input[call.dim + column];
    }
    const float inverse = 1.0f / sqrtf(sum / float(call.dim) + call.eps);
    for (int column = 0; column < call.dim; ++column)
        values[column] = __uint_as_float(uint32_t(rounded((values[column] * inverse)
            * (weight[column] + call.wadd))) << 16);
    const int position = call.positions[row];
    for (int column = 0; column < call.dim; ++column) {
        float result = values[column];
        if (column < 64 && position >= 0 && position < call.table_rows) {
            const int pair = column % 32;
            const float cosine = call.table[long(position) * 64 + pair];
            const float sine = call.table[long(position) * 64 + 32 + pair];
            result = column < 32 ? values[column] * cosine - values[column + 32] * sine
                : values[column - 32] * sine + values[column] * cosine;
        }
        output[column] = rounded(result);
        if (!key && call.gate) call.gate[row * call.gpitch + local * call.dim + column] = gates[column];
    }
}
// Same arithmetic as prepare: per-column products, one ordered FP32 sum, per-column round and RoPE.
// Returns the rounded output column (written as well); `active` is false past head_dim.
__device__ uint16_t prepare_column(const Prep& call, int row, int head, int column, bool& active) {
    const bool key = head >= call.heads;
    const int local = key ? head - call.heads : head;
    const uint16_t* input = key ? call.key + row * call.kpitch + local * call.dim
        : call.qg + row * call.qpitch + local * 2 * call.dim;
    const float* weight = key ? call.wk : call.wq;
    uint16_t* output = key ? call.outkey + row * call.okpitch + local * call.dim
        : call.query + row * call.opitch + local * call.dim;
    __shared__ float squares[256], values[256];
    __shared__ float inverse;
    active = column < call.dim;
    const float value = active ? load(input, column) : 0.0f;
    if (active) squares[column] = value * value;
    __syncthreads();
    if (column == 0) {
        float sum = 0;
        for (int index = 0; index < call.dim; ++index) sum = sum + squares[index];
        inverse = 1.0f / sqrtf(sum / float(call.dim) + call.eps);
    }
    __syncthreads();
    if (active) values[column] = __uint_as_float(uint32_t(rounded((value * inverse)
        * (weight[column] + call.wadd))) << 16);
    __syncthreads();
    if (!active) return 0;
    float result = values[column];
    const int position = call.positions[row];
    if (column < 64 && position >= 0 && position < call.table_rows) {
        const int pair = column % 32;
        const float cosine = call.table[long(position) * 64 + pair];
        const float sine = call.table[long(position) * 64 + 32 + pair];
        result = column < 32 ? values[column] * cosine - values[column + 32] * sine
            : values[column - 32] * sine + values[column] * cosine;
    }
    const uint16_t bits = rounded(result);
    output[column] = bits;
    if (!key && call.gate) call.gate[row * call.gpitch + local * call.dim + column] = input[call.dim + column];
    return bits;
}
__global__ __launch_bounds__(256) void prepare_parallel(Prep call) {
    bool active;
    prepare_column(call, blockIdx.x, blockIdx.y, threadIdx.x, active);
}
struct Rope {
    uint16_t* data; const int* positions;
    long pitch, plane; int heads, dim; bool multi;
    float theta, scale;
};
__global__ void rope(Rope call) {
    uint16_t* values = call.data + blockIdx.x * call.pitch + blockIdx.y * call.dim;
    const int pair = threadIdx.x;
    const int component = pair % 3 == 1 ? 1 : (pair % 3 == 2 && pair < 30 ? 2 : 0);
    const int position = call.positions[blockIdx.x + (call.multi ? component * call.plane : 0)];
    if (position < 0) return;
    const float angle = (float(position) / call.scale)
        * (1.0f / float(pow(double(call.theta), double(float(2 * pair) / 64.0f))));
    const float cosine = float(cos(double(angle))), sine = float(sin(double(angle)));
    const float first = load(values, pair), second = load(values, pair + 32);
    values[pair] = rounded(first * cosine - second * sine);
    values[pair + 32] = rounded(first * sine + second * cosine);
}
__global__ void store(const uint16_t* key, const uint16_t* value, const int* slots,
                       int8_t* output, long kpitch, long vpitch, long blocks, int heads) {
    const int token = blockIdx.x, half = blockIdx.y % 2, head = blockIdx.y / 2;
    const int slot = slots[token];
    if (slot < 0 || slot >= blocks * 4) return;
    const uint16_t* input = (half ? value + token * vpitch : key + token * kpitch) + head * 256;
    int8_t* row = output + ((long(slot / 4) * heads + head) * 4 + slot % 4) * 516;
    float maximum = 0;
    for (int column = 0; column < 256; ++column) maximum = fmaxf(maximum, fabsf(load(input, column)));
    const __half encoded = __float2half_rn(fminf(maximum / 127.0f, 515.0f));
    const float scale = __half2float(encoded);
    for (int column = 0; column < 256; ++column) {
        const float quantized = scale > 0 ? nearbyintf(load(input, column) / scale) : 0;
        row[half * 256 + column] = int8_t(fminf(fmaxf(quantized, -127.0f), 127.0f));
    }
    reinterpret_cast<__half*>(row + 512)[half] = encoded;
}
struct Attention {
    const uint16_t *query, *key, *value;
    const int8_t* kv;
    const int *cu, *used, *table;
    uint16_t* output;
    long qpitch, kpitch, vpitch, opitch, table_pitch, pages;
    int rows, seqs, table_width, keyheads;
    float scale;
    // attn_paged_gate_had_quant's epilogue: gate head h of a row at gate + row * gpitch + h * ghead.
    const uint16_t* gate;
    int8_t* codes;
    float* scales;
    long gpitch, ghead, cpitch, spitch;
};
__device__ float cached(const Attention& call, int seq, int token, int keyhead, int half, int column) {
    const int page = call.table[seq * call.table_pitch + token / 4];
    if (page < 0 || page >= call.pages) return 0;
    const int8_t* row = call.kv + ((long(page) * call.keyheads + keyhead) * 4 + token % 4) * 516;
    return float(row[half * 256 + column]) * __half2float(reinterpret_cast<const __half*>(row + 512)[half]);
}
template<bool PAGED> __global__ __launch_bounds__(256) void attention(Attention call) {
    const int row = blockIdx.x, head = blockIdx.y;
    int sequence = -1, begin = 0, end = 0;
    for (int candidate = 0; candidate < call.seqs; ++candidate) {
        const int lower = call.cu ? call.cu[candidate] : candidate * (call.rows / call.seqs);
        const int upper = call.cu ? call.cu[candidate + 1] : (candidate + 1) * (call.rows / call.seqs);
        if (lower >= 0 && upper <= call.rows && row >= lower && row < upper) {
            sequence = candidate; begin = lower; end = upper; break;
        }
    }
    int keys = row - begin + 1;
    if (PAGED && sequence >= 0) keys = call.used[sequence] - (end - begin) + row - begin + 1;
    if (sequence < 0 || keys < 0 || (PAGED && (call.used[sequence] > call.table_width * 4
        || call.used[sequence] < end - begin))) keys = 0;
    __shared__ float query[256], probability[256], reduce[256];
    const int lane = threadIdx.x;
    query[lane] = load(call.query, row * call.qpitch + head * 256 + lane);
    __syncthreads();
    // Parallel over keys/columns only; every per-output sum keeps the sequential order.
    auto score = [&](int token) {
        float dot = 0;
        for (int column = 0; column < 256; ++column) {
            const float key = PAGED ? cached(call, sequence, token, head / 12, 0, column)
                : load(call.key, (begin + token) * call.kpitch + (head / 12) * 256 + column);
            dot = dot + query[column] * key;
        }
        return dot;
    };
    float maximum = -INFINITY;
    for (int token = lane; token < keys; token += 256) maximum = fmaxf(maximum, score(token) * call.scale);
    reduce[lane] = maximum;
    __syncthreads();
    for (int span = 128; span > 0; span >>= 1) {
        if (lane < span) reduce[lane] = fmaxf(reduce[lane], reduce[lane + span]);
        __syncthreads();
    }
    maximum = reduce[0];
    float denominator = 0, accumulator = 0;
    for (int base = 0; base < keys; base += 256) {
        if (base + lane < keys)
            probability[lane] = float(exp(double(score(base + lane) * call.scale - maximum)));
        __syncthreads();
        const int count = min(256, keys - base);
        if (lane == 0) for (int index = 0; index < count; ++index) denominator = denominator + probability[index];
        for (int index = 0; index < count; ++index) {
            const float value = PAGED ? cached(call, sequence, base + index, head / 12, 1, lane)
                : load(call.value, (begin + base + index) * call.vpitch + (head / 12) * 256 + lane);
            accumulator = accumulator + probability[index] * value;
        }
        __syncthreads();
    }
    if (lane == 0) reduce[0] = denominator;
    __syncthreads();
    denominator = reduce[0];
    const float inverse = denominator > 0 ? 1.0f / denominator : 0;
    call.output[row * call.opitch + head * 256 + lane] = rounded(accumulator * inverse);
}
// store's arithmetic with one thread per column (max is order-free; each column quantizes alone).
__device__ void store_column(float element, int column, int8_t* row, int half) {
    __shared__ float reduce[256];
    reduce[column] = fabsf(element);
    __syncthreads();
    for (int span = 128; span > 0; span >>= 1) {
        if (column < span) reduce[column] = fmaxf(reduce[column], reduce[column + span]);
        __syncthreads();
    }
    const float maximum = fmaxf(0.0f, reduce[0]);
    const __half encoded = __float2half_rn(fminf(maximum / 127.0f, 515.0f));
    const float scale = __half2float(encoded);
    const float quantized = scale > 0 ? nearbyintf(element / scale) : 0;
    row[half * 256 + column] = int8_t(fminf(fmaxf(quantized, -127.0f), 127.0f));
    if (column == 0) reinterpret_cast<__half*>(row + 512)[half] = encoded;
}
__global__ __launch_bounds__(256) void store_parallel(const uint16_t* key, const uint16_t* value, const int* slots,
                       int8_t* output, long kpitch, long vpitch, long blocks, int heads) {
    const int token = blockIdx.x, half = blockIdx.y % 2, head = blockIdx.y / 2, column = threadIdx.x;
    const int slot = slots[token];
    if (slot < 0 || slot >= blocks * 4) return;
    const uint16_t* input = (half ? value + token * vpitch : key + token * kpitch) + head * 256;
    store_column(load(input, column), column, output + ((long(slot / 4) * heads + head) * 4 + slot % 4) * 516, half);
}
// prepare_parallel then store_parallel in one launch: a key head's block quantizes the bits it just wrote, and
// keyheads more blocks store V. Byte-identical to the pair (the store reads exactly those bits back).
struct Store { const uint16_t* value; const int* slots; int8_t* cache; long vpitch, blocks; };
__global__ __launch_bounds__(256) void prepare_store(Prep call, Store store) {
    const int row = blockIdx.x, head = blockIdx.y, column = threadIdx.x;
    const int slot = store.slots[row];
    const bool stored = slot >= 0 && slot < store.blocks * 4;
    if (head < call.heads + call.keyheads) {
        bool active;
        const uint16_t bits = prepare_column(call, row, head, column, active);
        if (head < call.heads || !stored) return;
        const int keyhead = head - call.heads;
        store_column(__uint_as_float(uint32_t(bits) << 16), column,
            store.cache + ((long(slot / 4) * call.keyheads + keyhead) * 4 + slot % 4) * 516, 0);
        return;
    }
    if (!stored) return;
    const int keyhead = head - call.heads - call.keyheads;
    store_column(load(store.value + row * store.vpitch + keyhead * 256, column), column,
        store.cache + ((long(slot / 4) * call.keyheads + keyhead) * 4 + slot % 4) * 516, 1);
}
// gate_had_quant_i8g (r3dh_qwen_ops_quant.hip quantize<true>, sigmoid) on one head's two 128-column groups, by
// the workgroup's first 256 threads (`column`); every thread takes the barriers. `reduce` and `scores` hold 256 floats.
__device__ void gate_epilogue(const Attention& call, int row, int head, uint16_t attended, bool column,
                              float* reduce, float* scores) {
    const int lane = threadIdx.x;
    const int position = lane & 127, group = lane & 128;
    float value = 0;
    __syncthreads();
    if (column) {
        const float gate = load(call.gate, row * call.gpitch + head * call.ghead + lane);
        const float sigmoid = __fdiv_rn(1.0f, 1.0f + expf(-gate));
        const float activation = __uint_as_float(uint32_t(rounded(sigmoid)) << 16);
        const uint16_t product = rounded(activation * __uint_as_float(uint32_t(attended) << 16));
        call.output[row * call.opitch + head * 256 + lane] = product;
        value = __uint_as_float(uint32_t(product) << 16);
        reduce[lane] = value;
    }
    __syncthreads();
    for (int distance = 1; distance < 128; distance <<= 1) {
        const float partner = column ? reduce[group + (position ^ distance)] : 0.0f;
        value = position & distance ? partner - value : value + partner;
        __syncthreads();
        if (column) reduce[lane] = value;
        __syncthreads();
    }
    if (column) scores[lane] = fabsf(value);
    __syncthreads();
    for (int distance = 64; distance; distance >>= 1) {
        if (column && position < distance) scores[lane] = fmaxf(scores[lane], scores[lane + distance]);
        __syncthreads();
    }
    if (!column) return;
    const float peak = scores[group];
    const float scale = peak > 0.0f ? peak * (1.0f / 127.0f) : 1.0f;
    const float reciprocal = __fdiv_rn(1.0f, scale);
    const float clamped = fminf(127.0f, fmaxf(-127.0f, value * reciprocal));
    call.codes[row * call.cpitch + head * 256 + lane] = int8_t(__float2int_rn(clamped));
    if (position == 0) call.scales[row * call.spitch + head * 2 + (lane >> 7)] = scale;
}
constexpr int fast_keys = 4096, value_chunk = 128, chunk_words = value_chunk * 64 / 256;
// attention<true>'s arithmetic: each token's dot, the max, the ordered denominator and every output's ordered
// sum over keys are unchanged. Page ids are read once into LDS, keys are read as words, values are staged in
// LDS per chunk, and a ninth wave sums the denominator (in key order, one chunk per pass) beside the P.V.
template<bool Gated> __global__ __launch_bounds__(288) void attention_paged_fast(Attention call) {
    const int row = blockIdx.x, head = blockIdx.y, keyhead = head / 12;
    int sequence = -1, begin = 0, end = 0;
    for (int candidate = 0; candidate < call.seqs; ++candidate) {
        const int lower = call.cu ? call.cu[candidate] : candidate * (call.rows / call.seqs);
        const int upper = call.cu ? call.cu[candidate + 1] : (candidate + 1) * (call.rows / call.seqs);
        if (lower >= 0 && upper <= call.rows && row >= lower && row < upper) {
            sequence = candidate; begin = lower; end = upper; break;
        }
    }
    int keys = sequence >= 0 ? call.used[sequence] - (end - begin) + row - begin + 1 : 0;
    if (sequence < 0 || keys < 0 || call.used[sequence] > call.table_width * 4
        || call.used[sequence] < end - begin) keys = 0;
    __shared__ float query[256], scores[fast_keys], reduce[256], value_scale[value_chunk];
    __shared__ __align__(16) int8_t values[value_chunk][256];
    __shared__ int page_ids[fast_keys / 4];
    const int lane = threadIdx.x;
    const bool column = lane < 256;
    if (column) query[lane] = load(call.query, row * call.qpitch + head * 256 + lane);
    for (int entry = lane; column && entry < (keys + 3) / 4; entry += 256) {
        const int page = call.table[sequence * call.table_pitch + entry];
        page_ids[entry] = page >= 0 && page < call.pages ? page : -1;
    }
    __syncthreads();
    uint32_t staged[chunk_words];
    float staged_scale = 0;
    // Branch-free: an absent key loads from page 0 and is masked to the strict path's zero.
    auto fetch = [&](int base) {
        const int count = min(value_chunk, keys - base);
#pragma unroll
        for (int slot = 0; slot < chunk_words; ++slot) {
            const int item = slot * 256 + lane, index = item / 64, word = item % 64, token = base + index;
            const int page = index < count ? page_ids[token / 4] : -1;
            const uint32_t loaded = reinterpret_cast<const uint32_t*>(
                call.kv + ((long(max(page, 0)) * call.keyheads + keyhead) * 4 + token % 4) * 516 + 256)[word];
            staged[slot] = page >= 0 ? loaded : 0u;
        }
        if (lane < value_chunk) {
            const int token = base + lane;
            const int page = lane < count ? page_ids[token / 4] : -1;
            const float loaded = __half2float(reinterpret_cast<const __half*>(
                call.kv + ((long(max(page, 0)) * call.keyheads + keyhead) * 4 + token % 4) * 516 + 512)[1]);
            staged_scale = page >= 0 ? loaded : 0.0f;
        }
    };
    if (column && keys > 0) fetch(0);
    float maximum = -INFINITY;
    for (int token = lane; column && token < keys; token += 256) {
        const int page = page_ids[token / 4];
        float dot = 0;
        if (page < 0) {
            for (int index = 0; index < 256; ++index) dot = dot + query[index] * 0.0f;
        } else {
            const int8_t* line = call.kv + ((long(page) * call.keyheads + keyhead) * 4 + token % 4) * 516;
            const float scale = __half2float(reinterpret_cast<const __half*>(line + 512)[0]);
            const uint32_t* words = reinterpret_cast<const uint32_t*>(line);
#pragma unroll 8
            for (int word = 0; word < 64; ++word) {
                const uint32_t packed = words[word];
#pragma unroll
                for (int byte = 0; byte < 4; ++byte) {
                    const float key = float(int8_t(packed >> (8 * byte))) * scale;
                    dot = dot + query[word * 4 + byte] * key;
                }
            }
        }
        const float scaled = dot * call.scale;
        scores[token] = scaled;
        maximum = fmaxf(maximum, scaled);
    }
    if (column) reduce[lane] = maximum;
    __syncthreads();
    for (int span = 128; span > 0; span >>= 1) {
        if (lane < span) reduce[lane] = fmaxf(reduce[lane], reduce[lane + span]);
        __syncthreads();
    }
    maximum = reduce[0];
    __syncthreads();
    for (int token = lane; column && token < keys; token += 256)
        scores[token] = float(exp(double(scores[token] - maximum)));
    float accumulator = 0, denominator = 0;
    for (int base = 0; base < keys; base += value_chunk) {
        const int count = min(value_chunk, keys - base);
        __syncthreads();
        if (column) {
#pragma unroll
            for (int slot = 0; slot < chunk_words; ++slot) {
                const int item = slot * 256 + lane;
                reinterpret_cast<uint32_t*>(values[item / 64])[item % 64] = staged[slot];
            }
            if (lane < value_chunk) value_scale[lane] = staged_scale;
        }
        __syncthreads();
        if (!column) {
            if (lane == 256) {
#pragma unroll 16
                for (int index = 0; index < count; ++index) denominator = denominator + scores[base + index];
            }
            continue;
        }
        if (base + value_chunk < keys) fetch(base + value_chunk);
#pragma unroll 16
        for (int index = 0; index < count; ++index) {
            const float value = float(values[index][lane]) * value_scale[index];
            accumulator = accumulator + scores[base + index] * value;
        }
    }
    __syncthreads();
    if (lane == 256) reduce[0] = denominator;
    __syncthreads();
    const float total = reduce[0];
    const float inverse = total > 0 ? 1.0f / total : 0;
    const uint16_t attended = rounded(accumulator * inverse);
    if (!Gated) {
        if (column) call.output[row * call.opitch + head * 256 + lane] = attended;
        return;
    }
    gate_epilogue(call, row, head, attended, column, reduce, scores);
}
// Split-KV (tolerance, RAD_ATTN_SPLIT): a workgroup owns one KV head's twelve query heads over one key segment
// (rows of q_len 1, each its own sequence), so a key is read once for all twelve. Scores and P.V run in FP32 with
// FMA, an online softmax per head, and the partial (max, sum, unnormalised output) goes to scratch; the merge
// rescales the segments and runs gate_epilogue. Not bit-identical to the strict path.
constexpr int split_tile = 32, split_min = 64, split_heads = 12, split_partial = split_heads * (2 + 256);
__device__ int split_keys(const Attention& call, int row) {
    const int used = call.used[row];
    return used >= 1 && used <= call.table_width * 4 ? used : 0;
}
__device__ int split_length(int keys, int splits) {
    const int even = (keys + splits - 1) / splits;
    return max(split_min, (even + split_tile - 1) / split_tile * split_tile);
}
// One halving step: a lane keeps the half of `values` its `Step` bit selects, plus its partner's copy of it.
template<int Step, int Half> __device__ inline void halve(float* values, int lane) {
    const bool upper = lane & Step;
#pragma unroll
    for (int index = 0; index < Half; ++index) {
        const float send = upper ? values[index] : values[index + Half];
        const float keep = upper ? values[index + Half] : values[index];
        values[index] = keep + __shfl_xor(send, Step);
    }
}
__global__ __launch_bounds__(256) void attention_split(Attention call, float* partial, int splits) {
    const int row = blockIdx.x, keyhead = blockIdx.y, split = blockIdx.z, tid = threadIdx.x;
    const int keys = split_keys(call, row), length = split_length(keys, splits);
    const int first = split * length, last = min(keys, first + length);
    float* out = partial + (long(row * call.keyheads + keyhead) * splits + split) * split_partial;
    if (first >= last) {
        if (tid < split_heads) { out[tid] = -INFINITY; out[split_heads + tid] = 0.0f; }
        return;
    }
    // Scores: sixteen lanes a head (192 threads), each holding sixteen query dims in registers.
    __shared__ __align__(16) float key_values[split_tile][256];
    __shared__ __align__(16) uint32_t value_words[split_tile][64];
    __shared__ float key_scale[split_tile], value_scale[split_tile], correction[split_heads];
    __shared__ __align__(16) float weights[split_tile][split_heads];
    __shared__ int page_ids[fast_keys / 4];
    for (int entry = first / 4 + tid; entry < (last + 3) / 4; entry += 256) {
        const int page = call.table[row * call.table_pitch + entry];
        page_ids[entry - first / 4] = page >= 0 && page < call.pages ? page : -1;
    }
    const bool scorer = tid < split_heads * 16;
    const int head = tid >> 4, slice = tid & 15;
    float query[16];
    const uint16_t* query_row = call.query + row * call.qpitch + (keyhead * split_heads + min(head, split_heads - 1)) * 256
        + slice * 16;
#pragma unroll
    for (int index = 0; index < 16; ++index) query[index] = load(query_row, index) * call.scale;
    float maximum = -INFINITY, sum = 0.0f, accumulator[split_heads];
    for (int index = 0; index < split_heads; ++index) accumulator[index] = 0.0f;
    // A tile's loads, issued a tile ahead into registers; an absent key reads page 0 and is zeroed.
    constexpr int items = split_tile * 64 / 256;
    uint32_t keys_loaded[items], values_loaded[items], scales_loaded = 0;
    bool valid[items], scale_valid = false;
    auto fetch = [&](int base) {
        const int count = min(split_tile, last - base);
#pragma unroll
        for (int slot = 0; slot < items; ++slot) {
            const int item = slot * 256 + tid, index = item / 64, word = item % 64, token = base + index;
            const int page = index < count ? page_ids[token / 4 - first / 4] : -1;
            valid[slot] = page >= 0;
            const uint32_t* line = reinterpret_cast<const uint32_t*>(
                call.kv + ((long(max(page, 0)) * call.keyheads + keyhead) * 4 + token % 4) * 516);
            keys_loaded[slot] = line[word];
            values_loaded[slot] = line[64 + word];
        }
        const int lane = tid & (split_tile - 1), token = base + lane;
        const int page = lane < count ? page_ids[token / 4 - first / 4] : -1;
        scale_valid = page >= 0;
        scales_loaded = reinterpret_cast<const uint32_t*>(
            call.kv + ((long(max(page, 0)) * call.keyheads + keyhead) * 4 + token % 4) * 516)[128];
    };
    __syncthreads();
    fetch(first);
    for (int base = first; base < last; base += split_tile) {
        const int count = min(split_tile, last - base);
        __syncthreads();
#pragma unroll
        for (int slot = 0; slot < items; ++slot) {
            const int item = slot * 256 + tid, index = item / 64, word = item % 64;
            const uint32_t packed = valid[slot] ? keys_loaded[slot] : 0u;
            *reinterpret_cast<float4*>(&key_values[index][word * 4]) = make_float4(float(int8_t(packed)),
                float(int8_t(packed >> 8)), float(int8_t(packed >> 16)), float(int8_t(packed >> 24)));
            value_words[index][word] = valid[slot] ? values_loaded[slot] : 0u;
        }
        if (tid < split_tile) {
            const uint32_t scales = scale_valid ? scales_loaded : 0u;
            key_scale[tid] = __half2float(__ushort_as_half(uint16_t(scales)));
            value_scale[tid] = __half2float(__ushort_as_half(uint16_t(scales >> 16)));
        }
        __syncthreads();
        if (base + split_tile < last) fetch(base + split_tile);
        if (scorer) {
            float dot[split_tile];
#pragma unroll
            for (int key = 0; key < split_tile; ++key) {
                const float4* row_values = reinterpret_cast<const float4*>(&key_values[key][slice * 16]);
                float total = 0.0f;
#pragma unroll
                for (int quad = 0; quad < 4; ++quad) {
                    const float4 value = row_values[quad];
                    total = fmaf(query[quad * 4], value.x, total);
                    total = fmaf(query[quad * 4 + 1], value.y, total);
                    total = fmaf(query[quad * 4 + 2], value.z, total);
                    total = fmaf(query[quad * 4 + 3], value.w, total);
                }
                dot[key] = total;
            }
            // Halving exchange over the sixteen lanes: lane `slice` ends with keys 2 slice and 2 slice + 1.
            halve<8, 16>(dot, slice);
            halve<4, 8>(dot, slice);
            halve<2, 4>(dot, slice);
            halve<1, 2>(dot, slice);
            const int key0 = 2 * slice, key1 = key0 + 1;
            const float score0 = key0 < count ? dot[0] * key_scale[key0] : -INFINITY;
            const float score1 = key1 < count ? dot[1] * key_scale[key1] : -INFINITY;
            float tile = fmaxf(score0, score1);
            for (int mask = 1; mask < 16; mask <<= 1) tile = fmaxf(tile, __shfl_xor(tile, mask));
            const float next = fmaxf(maximum, tile);
            const float scale = expf(maximum - next);
            const float p0 = expf(score0 - next), p1 = expf(score1 - next);
            float part = p0 + p1;
            for (int mask = 1; mask < 16; mask <<= 1) part += __shfl_xor(part, mask);
            sum = sum * scale + part;
            maximum = next;
            weights[key0][head] = p0;
            weights[key1][head] = p1;
            if (slice == 0) correction[head] = scale;
        }
        __syncthreads();
#pragma unroll
        for (int index = 0; index < split_heads; ++index) accumulator[index] *= correction[index];
        const uint8_t* bytes = reinterpret_cast<const uint8_t*>(value_words);
        for (int key = 0; key < count; ++key) {
            const float value = float(int8_t(bytes[key * 256 + tid])) * value_scale[key];
            const float4* p = reinterpret_cast<const float4*>(weights[key]);
            const float4 p0 = p[0], p1 = p[1], p2 = p[2];
            accumulator[0] = fmaf(p0.x, value, accumulator[0]); accumulator[1] = fmaf(p0.y, value, accumulator[1]);
            accumulator[2] = fmaf(p0.z, value, accumulator[2]); accumulator[3] = fmaf(p0.w, value, accumulator[3]);
            accumulator[4] = fmaf(p1.x, value, accumulator[4]); accumulator[5] = fmaf(p1.y, value, accumulator[5]);
            accumulator[6] = fmaf(p1.z, value, accumulator[6]); accumulator[7] = fmaf(p1.w, value, accumulator[7]);
            accumulator[8] = fmaf(p2.x, value, accumulator[8]); accumulator[9] = fmaf(p2.y, value, accumulator[9]);
            accumulator[10] = fmaf(p2.z, value, accumulator[10]); accumulator[11] = fmaf(p2.w, value, accumulator[11]);
        }
    }
    if (scorer && slice == 0) { out[head] = maximum; out[split_heads + head] = sum; }
#pragma unroll
    for (int index = 0; index < split_heads; ++index) out[2 * split_heads + index * 256 + tid] = accumulator[index];
}
// The segments' partials rescaled to one maximum, summed, normalised and rounded, then gate_epilogue.
__global__ __launch_bounds__(256) void attention_merge(Attention call, const float* partial, int splits) {
    const int row = blockIdx.x, head = blockIdx.y, tid = threadIdx.x;
    const int keyhead = head / split_heads, local = head % split_heads;
    const int keys = split_keys(call, row), length = split_length(keys, splits);
    const int active = keys ? (keys + length - 1) / length : 0;
    const float* base = partial + long(row * call.keyheads + keyhead) * splits * split_partial;
    float maximum = -INFINITY;
    for (int split = 0; split < active; ++split) maximum = fmaxf(maximum, base[split * split_partial + local]);
    float numerator = 0.0f, denominator = 0.0f;
    for (int split = 0; split < active; ++split) {
        const float* segment = base + split * split_partial;
        const float weight = expf(segment[local] - maximum);
        denominator = fmaf(segment[split_heads + local], weight, denominator);
        numerator = fmaf(segment[2 * split_heads + local * 256 + tid], weight, numerator);
    }
    __shared__ float reduce[256], maxima[256];
    const uint16_t attended = rounded(denominator > 0 ? numerator / denominator : 0.0f);
    gate_epilogue(call, row, head, attended, true, reduce, maxima);
}
int launched() { return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE; }
}
namespace {
int build_prep(const RadArgs* args, Prep& prep) {
    const int dim = integer(args, "head_dim", 0), heads = integer(args, "n_head", -1);
    const int keyheads = integer(args, "n_head_kv", 0);
    if (!((dim == 256 && ((heads == 12 && keyheads == 1) || (heads == 24 && keyheads == 2)))
        || (dim == 128 && heads == 0 && keyheads == 4))
        || integer(args, "rot", 0) != 64 || integer(args, "cs_f32", 0) != 1
        || !named(args, "q_dtype", "bf16")) return RAD_E_UNSUPPORTED;
    const RadTensor* tensors = args->t;
    const long rows = tensors[1].shape[0];
    const bool indexer_projection = heads == 0 && dim == 128 && keyheads == 4
        && tensors[1].rank == 2 && matrix(tensors[1], RAD_BF16, rows, 640);
    if (rows < 1 || rows > 2048
        || (!matrix(tensors[1], RAD_BF16, rows, keyheads * dim) && !indexer_projection)
        || !matrix(tensors[7], RAD_BF16, rows, keyheads * dim)
        || !vector(tensors[3], RAD_I32, rows) || !vector(tensors[5], RAD_F32, dim)
        || !tensors[2].data || tensors[2].dtype != RAD_F32 || tensors[2].rank != 2
        || tensors[2].shape[0] < 1 || tensors[2].shape[1] != 64 || tensors[2].stride[0] != 64
        || tensors[2].stride[1] != 1) return RAD_E_SHAPE;
    if (!heads && (tensors[0].data || tensors[4].data || tensors[6].data || tensors[8].data))
        return RAD_E_SHAPE;
    if (heads && (!matrix(tensors[0], RAD_BF16, rows, 2 * heads * dim)
        || !matrix(tensors[6], RAD_BF16, rows, heads * dim) || !vector(tensors[4], RAD_F32, dim)
        || (tensors[8].data && !matrix(tensors[8], RAD_BF16, rows, heads * dim)))) return RAD_E_SHAPE;
    const float eps = real(args, "eps", 1e-6f), wadd = real(args, "wadd", 0);
    if (!std::isfinite(eps) || eps <= 0 || !std::isfinite(wadd)) return RAD_E_INVAL;
    Prep call{static_cast<const uint16_t*>(tensors[0].data), static_cast<const uint16_t*>(tensors[1].data),
        static_cast<const float*>(tensors[2].data), static_cast<const float*>(tensors[4].data),
        static_cast<const float*>(tensors[5].data), static_cast<const int*>(tensors[3].data),
        static_cast<uint16_t*>(tensors[6].data), static_cast<uint16_t*>(tensors[7].data),
        static_cast<uint16_t*>(tensors[8].data), tensors[0].stride[0], tensors[1].stride[0],
        tensors[6].stride[0], tensors[7].stride[0], tensors[8].stride[0], tensors[2].shape[0],
        int(rows), heads, keyheads, dim, eps, wadd};
    prep = call;
    return RAD_OK;
}
}
extern "C" int r3dh_attn_qsa_prepare(const RadArgs* args, RadStream stream) {
    if (!args || !args->t || args->n_t != 9) return RAD_E_INVAL;
    Prep call{};
    const int status = build_prep(args, call);
    if (status) return status;
    const long rows = call.rows;
    const int heads = call.heads, keyheads = call.keyheads;
    const bool parallel = r3d_exact_path("RAD_QK_PREPARE_PARALLEL");
    if (parallel)
        hipLaunchKernelGGL(prepare_parallel, dim3(rows, heads + keyheads), dim3(256), 0,
            reinterpret_cast<hipStream_t>(stream), call);
    else
        hipLaunchKernelGGL(prepare, dim3(rows, heads + keyheads), dim3(1), 0,
            reinterpret_cast<hipStream_t>(stream), call);
    return launched();
}
extern "C" int r3dh_attn_qsa_rope(const RadArgs* args, RadStream stream) {
    if (!args || !args->t || args->n_t != 2) return RAD_E_INVAL;
    const int dim = integer(args, "head_dim", 0), heads = integer(args, "n_head", 0);
    const bool multi = !named(args, "mode", "neox");
    if (!((dim == 256 && (heads == 12 || heads == 1 || heads == 24 || heads == 2)) || (dim == 128 && heads == 4))
        || integer(args, "rotary_dim", 0) != 64 || integer(args, "n_head_kv", 0) != 0
        || (multi && (!named(args, "mode", "imrope") || !rad_args_gets(args, "sections")
            || !named(args, "sections", "11 11 10")))
        || !named(args, "ladder", "rotary") || real(args, "yarn_factor", 0) != 0
        || real(args, "llama3_factor", 0) != 0
        || !named(args, "dtype", "bf16")) return RAD_E_UNSUPPORTED;
    const long rows = args->t[0].shape[0];
    const float theta = real(args, "theta", 10000), scale = real(args, "scale", 1);
    const RadTensor& positions = args->t[1];
    const bool positions_ok = vector(positions, RAD_I32, rows) || (multi && positions.data
        && positions.dtype == RAD_I32 && positions.rank == 2 && positions.shape[0] == 3
        && positions.shape[1] == rows && positions.stride[1] == 1 && positions.stride[0] >= rows);
    if (rows < 1 || rows > 2048 || !matrix(args->t[0], RAD_BF16, rows, heads * dim)
        || !positions_ok) return RAD_E_SHAPE;
    if (!std::isfinite(theta) || theta <= 0 || !std::isfinite(scale) || scale <= 0) return RAD_E_INVAL;
    Rope call{static_cast<uint16_t*>(args->t[0].data), static_cast<const int*>(args->t[1].data),
        args->t[0].stride[0], positions.stride[0], heads, dim, positions.rank == 2, theta, scale};
    hipLaunchKernelGGL(rope, dim3(rows, heads), dim3(32), 0, reinterpret_cast<hipStream_t>(stream), call);
    return launched();
}
extern "C" int r3dh_attn_qsa_kv_store(const RadArgs* args, RadStream stream) {
    if (!args || !args->t || args->n_t != 4) return RAD_E_INVAL;
    const int heads = integer(args, "n_head_kv", 1);
    if (integer(args, "head_dim", 0) != 256 || (heads != 1 && heads != 2)
        || integer(args, "block_size", 0) != 4 || !named(args, "kv_dtype", "i8")) return RAD_E_UNSUPPORTED;
    const RadTensor* tensors = args->t;
    const long rows = tensors[0].shape[0];
    if (rows < 1 || rows > 2048 || !matrix(tensors[0], RAD_BF16, rows, heads * 256)
        || !matrix(tensors[1], RAD_BF16, rows, heads * 256) || !vector(tensors[2], RAD_I32, rows)
        || !cache(tensors[3], heads)) return RAD_E_SHAPE;
    const bool parallel = r3d_exact_path("RAD_KV_STORE_PARALLEL");
    if (parallel) {
        hipLaunchKernelGGL(store_parallel, dim3(rows, heads * 2), dim3(256), 0, reinterpret_cast<hipStream_t>(stream),
            static_cast<const uint16_t*>(tensors[0].data), static_cast<const uint16_t*>(tensors[1].data),
            static_cast<const int*>(tensors[2].data), static_cast<int8_t*>(tensors[3].data),
            tensors[0].stride[0], tensors[1].stride[0], tensors[3].shape[0], heads);
        return launched();
    }
    hipLaunchKernelGGL(store, dim3(rows, heads * 2), dim3(1), 0, reinterpret_cast<hipStream_t>(stream),
        static_cast<const uint16_t*>(tensors[0].data), static_cast<const uint16_t*>(tensors[1].data),
        static_cast<const int*>(tensors[2].data), static_cast<int8_t*>(tensors[3].data),
        tensors[0].stride[0], tensors[1].stride[0], tensors[3].shape[0], heads);
    return launched();
}
struct GateOut { const uint16_t* gate; int8_t* codes; float* scales; long gpitch, ghead, cpitch, spitch; };
// Partials the split-KV scratch holds: (row, segment) pairs, each a KV head's twelve heads.
constexpr long split_capacity = 512;
long flag(const char* name, long fallback) {
    const char* value = std::getenv(name);
    return value && *value ? std::strtol(value, nullptr, 10) : fallback;
}
// Segments a row's keys are split into under RAD_ATTN_SPLIT (0: the exact kernel). About
// RAD_ATTN_SPLIT_TARGET (192) workgroups a launch; a row of one query and its own selection only.
int split_count(const RadArgs* args, long rows, long sequences, bool prefix) {
    if (r3d_precision_exact() || !flag("RAD_ATTN_SPLIT", 0) || prefix || sequences != rows || rows > split_capacity) return 0;
    const long target = flag("RAD_ATTN_SPLIT_TARGET", 192);
    const long splits = std::max(1L, std::min(std::min(64L, split_capacity / rows), target / (2 * rows)));
    const uint64_t bytes = uint64_t(rows) * 2 * splits * split_partial * sizeof(float);
    if (!args->scratch || args->scratch_bytes < 0 || uint64_t(args->scratch_bytes) < bytes) return 0;
    return int(splits);
}
template<bool PAGED> int launch_attention(const RadArgs* args, RadStream stream, const GateOut* gated = nullptr) {
    if (!args || !args->t || args->n_t != (PAGED ? 8 : 5)) return RAD_E_INVAL;
    const auto& query = args->t[0];
    if ((query.rank != 2 && query.rank != 3) || query.shape[1] < 1
        || query.shape[1] > 6144 || (query.rank == 3 && (query.shape[2] < 1
        || query.shape[2] > 6144 || query.shape[1] > 6144 / query.shape[2]))) return RAD_E_SHAPE;
    const long query_width = query.rank == 3 ? query.shape[1] * query.shape[2] : query.shape[1];
    const int heads = int(query_width / 256);
    if (integer(args, "head_dim", 0) != 256 || integer(args, "gqa", 0) != 12
        || integer(args, "causal", PAGED ? 1 : 0) != 1 || integer(args, "window", 0) != 0
        || integer(args, "block", 0) != 0 || integer(args, "cu_div", 1) != 1
        || integer(args, "window_right", 0) != 0 || !named(args, "q_dtype", "bf16")
        || !named(args, "dtype", "bf16") || (heads != 12 && heads != 24)
        || integer(args, "n_head", heads) != heads
        || (PAGED && (integer(args, "block_size", 0) != 4 || !named(args, "kv_dtype", "i8"))))
        return RAD_E_UNSUPPORTED;
    const RadTensor* tensors = args->t;
    const RadTensor& output = tensors[PAGED ? 7 : 4];
    const RadTensor& cu = tensors[PAGED ? 6 : 3];
    const long rows = tensors[0].shape[0];
    if (rows < 1 || rows > 2048 || !matrix(tensors[0], RAD_BF16, rows, heads * 256)
        || !matrix(output, RAD_BF16, rows, heads * 256)) return RAD_E_SHAPE;
    const long sequences = PAGED ? tensors[3].shape[0] : cu.shape[0] - 1;
    if (sequences < 1 || sequences > 2048 || (cu.data && !vector(cu, RAD_I32, sequences + 1))
        || (!cu.data && (!PAGED || rows % sequences))) return RAD_E_SHAPE;
    Attention call{};
    call.keyheads = heads / 12;
    call.query = static_cast<const uint16_t*>(tensors[0].data);
    call.output = static_cast<uint16_t*>(output.data);
    call.qpitch = tensors[0].stride[0]; call.opitch = output.stride[0];
    call.rows = int(rows); call.seqs = int(sequences); call.cu = static_cast<const int*>(cu.data);
    call.scale = real(args, "scale", 0.0625f);
    if (!std::isfinite(call.scale) || call.scale <= 0) return RAD_E_INVAL;
    if (PAGED) {
        if (!cache(tensors[1], call.keyheads) || !vector(tensors[3], RAD_I32, sequences)
            || tensors[4].data || tensors[5].data || !tensors[2].data || tensors[2].dtype != RAD_I32
            || tensors[2].rank != 2 || tensors[2].shape[0] != sequences || tensors[2].shape[1] < 1
            || tensors[2].shape[1] > 50000 || tensors[2].stride[1] != 1
            || tensors[2].stride[0] < tensors[2].shape[1]) return RAD_E_SHAPE;
        call.kv = static_cast<const int8_t*>(tensors[1].data);
        call.table = static_cast<const int*>(tensors[2].data); call.pages = tensors[1].shape[0];
        call.used = static_cast<const int*>(tensors[3].data); call.table_pitch = tensors[2].stride[0];
        call.table_width = int(tensors[2].shape[1]);
    } else {
        if (!matrix(tensors[1], RAD_BF16, rows, call.keyheads * 256) || !matrix(tensors[2], RAD_BF16, rows, call.keyheads * 256))
            return RAD_E_SHAPE;
        call.key = static_cast<const uint16_t*>(tensors[1].data);
        call.value = static_cast<const uint16_t*>(tensors[2].data);
        call.kpitch = tensors[1].stride[0]; call.vpitch = tensors[2].stride[0];
    }
    if (gated) {
        if (!PAGED || call.table_width > fast_keys / 4) return RAD_E_UNSUPPORTED;
        call.gate = gated->gate; call.codes = gated->codes; call.scales = gated->scales;
        call.gpitch = gated->gpitch; call.ghead = gated->ghead; call.cpitch = gated->cpitch; call.spitch = gated->spitch;
        const int splits = split_count(args, rows, sequences, call.cu != nullptr);
        if (splits > 0) {
            auto* partial = static_cast<float*>(args->scratch);
            hipLaunchKernelGGL(attention_split, dim3(rows, call.keyheads, splits), dim3(256), 0,
                reinterpret_cast<hipStream_t>(stream), call, partial, splits);
            if (hipGetLastError() != hipSuccess) return RAD_E_DEVICE;
            hipLaunchKernelGGL(attention_merge, dim3(rows, heads), dim3(256), 0,
                reinterpret_cast<hipStream_t>(stream), call, partial, splits);
            return launched();
        }
        hipLaunchKernelGGL(attention_paged_fast<true>, dim3(rows, heads), dim3(288), 0,
            reinterpret_cast<hipStream_t>(stream), call);
        return launched();
    }
    const bool fast = r3d_exact_path("RAD_ATTN_PAGED_FAST");
    if (PAGED && fast && call.table_width <= fast_keys / 4) {
        hipLaunchKernelGGL(attention_paged_fast<false>, dim3(rows, heads), dim3(288), 0,
            reinterpret_cast<hipStream_t>(stream), call);
        return launched();
    }
    hipLaunchKernelGGL(attention<PAGED>, dim3(rows, heads), dim3(256), 0, reinterpret_cast<hipStream_t>(stream), call);
    return launched();
}
extern "C" int r3dh_attn_qsa_dense(const RadArgs* args, RadStream stream) { return launch_attention<false>(args, stream); }
extern "C" int r3dh_attn_qsa_paged(const RadArgs* args, RadStream stream) { return launch_attention<true>(args, stream); }

// attn_paged_gate_had_quant at n_head 24: attn_paged's eight operands, then the gate, the codes and their
// per-128-column scales. One launch: the fast paged kernel with gate_had_quant_i8g as its epilogue.
extern "C" int r3dh_attn_qsa_paged_gate(const RadArgs* args, RadStream stream) {
    if (!args || !args->t || args->n_t != 11) return RAD_E_INVAL;
    if (integer(args, "group", 128) != 128 || !named(args, "act", "sigmoid")) return RAD_E_UNSUPPORTED;
    const RadTensor* tensors = args->t;
    const RadTensor& query = tensors[0];
    if (query.rank != 2 || query.shape[1] % 256) return RAD_E_SHAPE;
    const long rows = query.shape[0], heads = query.shape[1] / 256;
    const RadTensor &gate = tensors[8], &codes = tensors[9], &scales = tensors[10];
    GateOut out{};
    if (!matrix(codes, RAD_I8, rows, heads * 256) || codes.rank != 2
        || !matrix(scales, RAD_F32, rows, heads * 2) || scales.rank != 2
        || !gate.data || gate.dtype != RAD_BF16 || gate.stride[gate.rank - 1] != 1) return RAD_E_SHAPE;
    if (gate.rank == 2 && gate.shape[0] == rows && gate.shape[1] == 256 && gate.stride[0] >= heads * 512)
        out.ghead = 512;
    else if (gate.rank == 3 && gate.shape[0] == rows && gate.shape[1] == heads && gate.shape[2] == 256
        && gate.stride[1] >= 256 && gate.stride[0] >= heads * gate.stride[1])
        out.ghead = gate.stride[1];
    else if (gate.rank == 2 && gate.shape[0] == rows && gate.shape[1] == heads * 256)
        out.ghead = 256;
    else return RAD_E_SHAPE;
    out.gate = static_cast<const uint16_t*>(gate.data); out.gpitch = gate.stride[0];
    out.codes = static_cast<int8_t*>(codes.data); out.cpitch = codes.stride[0];
    out.scales = static_cast<float*>(scales.data); out.spitch = scales.stride[0];
    RadTensor attention[8];
    for (int index = 0; index < 8; ++index) attention[index] = tensors[index];
    RadArgs call = *args; call.t = attention; call.n_t = 8;
    return launch_attention<true>(&call, stream, &out);
}
// qk_norm_rope_gate_kv_store: qk_norm_rope_gate's nine operands, then v, the slots and the cache.
extern "C" int r3dh_attn_qsa_prepare_store(const RadArgs* args, RadStream stream) {
    if (!args || !args->t || args->n_t != 12) return RAD_E_INVAL;
    const RadTensor* tensors = args->t;
    const int heads = integer(args, "n_head", -1), keyheads = integer(args, "n_head_kv", 0);
    if (integer(args, "head_dim", 0) != 256 || (heads != 24 && heads != 12) || keyheads * 12 != heads
        || integer(args, "block_size", 0) != 4
        || !named(args, "kv_dtype", "i8") || integer(args, "pos_i64", 0) != 0) return RAD_E_UNSUPPORTED;
    const long rows = tensors[1].shape[0];
    if (!tensors[6].data || !matrix(tensors[9], RAD_BF16, rows, keyheads * 256)
        || !vector(tensors[10], RAD_I32, rows) || !cache(tensors[11], keyheads)) return RAD_E_SHAPE;
    Prep prep{};
    const int status = build_prep(args, prep);
    if (status) return status;
    Store store{static_cast<const uint16_t*>(tensors[9].data), static_cast<const int*>(tensors[10].data),
        static_cast<int8_t*>(tensors[11].data), tensors[9].stride[0], tensors[11].shape[0]};
    hipLaunchKernelGGL(prepare_store, dim3(rows, heads + 2 * keyheads), dim3(256), 0,
        reinterpret_cast<hipStream_t>(stream), prep, store);
    return launched();
}
// The split-KV partial buffer (RAD_ATTN_SPLIT), zero when the flag is off.
extern "C" int64_t r3dh_attn_qsa_paged_gate_scratch(const RadArgs*) {
    return flag("RAD_ATTN_SPLIT", 0) ? int64_t(split_capacity * 2 * split_partial * sizeof(float)) : 0;
}
