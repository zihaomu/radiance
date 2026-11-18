#include <hip/hip_runtime.h>
#include "r3dh_mtp_ple.h"
#include "r3d_logits_planes.h"
#include <hip/hip_fp16.h>
#include <cmath>
#include <cstdint>

namespace r3dh_mtp_ple_select {
struct StrictRerank {
    const uint16_t* input;
    const RadPlanesHeader* weight;
    const int32_t* candidates;
    int32_t* indices;
    float* values;
    float* pairs;
    int64_t pitch[5];
    int64_t available;
    int offset;
};
static_assert(sizeof(StrictRerank) + 256 <= 768);
__global__ void joined_rerank(StrictRerank call) {
    __shared__ int32_t ids[32];
    __shared__ float scores[32];
    if (r3d_head::header(call.weight, 124160, 2560, call.available) != RAD_OK) return;
    const auto* base = reinterpret_cast<const uint8_t*>(call.weight);
    if (uint64_t(call.weight->bytes) > UINTPTR_MAX - reinterpret_cast<uintptr_t>(base)) return;
    const int candidate = threadIdx.x;
    const int row = blockIdx.x;
    if (candidate < 8) ids[candidate] = call.candidates[int64_t(row) * call.pitch[1] + candidate];
    __syncthreads();
    if (candidate < 8) {
        const int64_t local = int64_t(ids[candidate]) - call.offset;
        const bool owned = ids[candidate] >= 0 && local >= 0 && local < 124160;
        float score = 0.0f;
        if (owned) {
            const auto* entry = reinterpret_cast<const RadPlanesEntry*>(call.weight + 1);
            const auto& codes = entry[0];
            const auto& scale = entry[1];
            const int64_t scale_row = scale.block[0] ? local / scale.block[0] : 0;
            for (int position = 0; position < 2560; ++position) {
                const uint8_t bits = base[codes.offset + local * codes.row_bytes + position];
                const int magnitude = bits & 127;
                const float decoded = magnitude == 127 ? __uint_as_float(0x7fc00000u)
                    : magnitude >= 8 ? ldexpf(1.0f + float(magnitude & 7) / 8.0f, (magnitude >> 3) - 7)
                    : float(magnitude) / 512.0f;
                const float code = codes.dtype == RAD_I8 ? float(static_cast<int8_t>(bits))
                    : bits & 128 ? -decoded : decoded;
                const int64_t scale_col = scale.block[1] ? position / scale.block[1] : 0;
                const auto* address = base + scale.offset + scale_row * scale.row_bytes +
                    scale_col * r3d_head::scale_bytes(scale.dtype);
                const float multiplier = scale.dtype == RAD_F32 ? *reinterpret_cast<const float*>(address)
                    : scale.dtype == RAD_BF16 ? __uint_as_float(uint32_t(*reinterpret_cast<const uint16_t*>(address)) << 16)
                    : __half2float(*reinterpret_cast<const __half*>(address));
                const float weight = __fmul_rn(code, multiplier);
                const float activation = __uint_as_float(uint32_t(call.input[int64_t(row) * call.pitch[0] + position]) << 16);
                score = __fadd_rn(score, __fmul_rn(activation, weight));
            }
        }
        scores[candidate] = owned ? score : -INFINITY;
        if (!owned) ids[candidate] = -1;
    }
    __syncthreads();
    if (candidate) return;
    unsigned used = 0;
    for (int output = 0; output < 8; ++output) {
        int best = -1;
        for (int current = 0; current < 8; ++current) {
            if ((used & (1u << current)) || ids[current] < 0) continue;
            if (best < 0 || scores[current] > scores[best] ||
                (scores[current] == scores[best] && ids[current] < ids[best])) best = current;
        }
        const int32_t id = best < 0 ? -1 : ids[best];
        const float score = best < 0 ? -INFINITY : scores[best];
        if (best >= 0) used |= 1u << best;
        call.indices[int64_t(row) * call.pitch[2] + output] = id;
        call.values[int64_t(row) * call.pitch[3] + output] = score;
        if (call.pairs) {
            call.pairs[int64_t(row) * call.pitch[4] + output * 2] = __int_as_float(id);
            call.pairs[int64_t(row) * call.pitch[4] + output * 2 + 1] = score;
        }
    }
}
using Index = int64_t;
const RadTensor* at(const RadArgs* args, int index) {
    return args && args->t && index < args->n_t && args->t[index].data && args->t[index].rank ? args->t + index : nullptr;
}
Index parameter(const RadArgs* args, const char* key, Index fallback = 0) { return rad_args_geti_or(args, key, fallback); }
bool dense(const RadTensor* tensor, uint32_t type, Index count) {
    return tensor && tensor->dtype == type && rad_tensor_is_contiguous(tensor) && rad_tensor_numel(tensor) >= count;
}
__device__ float bf16(uint16_t bits) { return __uint_as_float(uint32_t(bits) << 16); }
__device__ float e4m3(uint8_t bits) {
    const int magnitude = bits & 127, exponent = magnitude >> 3, fraction = magnitude & 7;
    const float value = magnitude == 127 ? __uint_as_float(0x7fc00000u) :
        (exponent ? ldexpf(1.0f + float(fraction) / 8.0f, exponent - 7) : float(fraction) / 512.0f);
    return bits & 128 ? -value : value;
}
__device__ float head(RadTensor tensor, Index row, Index column, Index width) {
    if (tensor.dtype == RAD_BF16) return bf16(static_cast<const uint16_t*>(tensor.data)[row * width + column]);
    if (tensor.dtype == RAD_F32) return static_cast<const float*>(tensor.data)[row * width + column];
    const auto* packed = static_cast<const uint8_t*>(tensor.data) + row * tensor.stride[0];
    const float code = tensor.dtype == RAD_I8 ? float(reinterpret_cast<const int8_t*>(packed)[column]) : e4m3(packed[column]);
    return code * bf16(reinterpret_cast<const uint16_t*>(packed + width)[column / 128]);
}
__global__ void rerank(RadTensor input, RadTensor weights, const int32_t* candidates,
                       int32_t* indices, float* values, float* pairs, Index width, Index vocabulary, Index offset, int shortlist) {
    if (threadIdx.x) return;
    const Index row = blockIdx.x;
    int32_t ids[32]; float scores[32]; bool used[32]{};
    for (int candidate = 0; candidate < shortlist; ++candidate) {
        const int32_t global = candidates[row * shortlist + candidate];
        const Index local = Index(global) - offset;
        const bool owned = global >= 0 && local >= 0 && local < vocabulary;
        double sum = 0;
        if (owned) for (Index column = 0; column < width; ++column)
            sum += double(bf16(static_cast<const uint16_t*>(input.data)[row * width + column])) * double(head(weights, local, column, width));
        ids[candidate] = owned ? global : -1; scores[candidate] = owned ? float(sum) : -INFINITY;
    }
    for (int output = 0; output < shortlist; ++output) {
        int best = -1;
        for (int candidate = 0; candidate < shortlist; ++candidate) {
            if (used[candidate] || ids[candidate] < 0) continue;
            if (best < 0 || scores[candidate] > scores[best] ||
                (scores[candidate] == scores[best] && ids[candidate] < ids[best])) best = candidate;
        }
        const int32_t id = best >= 0 ? ids[best] : -1;
        const float score = best >= 0 ? scores[best] : -INFINITY;
        if (best >= 0) used[best] = true;
        indices[row * shortlist + output] = id; values[row * shortlist + output] = score;
        if (pairs) { pairs[(row * shortlist + output) * 2] = __int_as_float(id); pairs[(row * shortlist + output) * 2 + 1] = score; }
    }
}
__global__ void merge(const float* pairs, int32_t* indices, float* values, int shortlist, int world) {
    if (threadIdx.x) return;
    const Index row = blockIdx.x; const int total = shortlist * world;
    bool used[256]{};
    for (int output = 0; output < shortlist; ++output) {
        int best = -1;
        for (int candidate = 0; candidate < total; ++candidate) {
            const int32_t id = __float_as_int(pairs[(row * total + candidate) * 2]);
            if (used[candidate] || id < 0) continue;
            const float score = pairs[(row * total + candidate) * 2 + 1];
            const float previous = best >= 0 ? pairs[(row * total + best) * 2 + 1] : 0;
            if (best < 0 || score > previous || (score == previous && id < __float_as_int(pairs[(row * total + best) * 2]))) best = candidate;
        }
        if (best >= 0) used[best] = true;
        indices[row * shortlist + output] = best >= 0 ? __float_as_int(pairs[(row * total + best) * 2]) : -1;
        values[row * shortlist + output] = best >= 0 ? pairs[(row * total + best) * 2 + 1] : -INFINITY;
    }
}
__global__ void hint(const int32_t* tokens, const int32_t* positions, const int32_t* cu,
                     const int64_t* multipliers, const int64_t* sizes, const int64_t* offsets,
                     const int32_t* state_indices, int32_t* tail, int32_t* ids,
                     Index sequences, Index heads, Index order, Index eos, Index history, Index slots, Index pitch) {
    if (blockIdx.x || threadIdx.x) return;
    const Index context = order - 1, per_order = heads / context;
    for (Index sequence = 0; sequence < sequences; ++sequence) {
        const Index begin = cu[sequence], end = cu[sequence + 1];
        const int32_t slot = state_indices ? state_indices[sequence] : int32_t(sequence);
        if (end <= begin || slot < 0) {
            for (Index head = 0; head < heads; ++head) ids[sequence * pitch + head] = int32_t(offsets[head]);
            continue;
        }
        int32_t* committed = tail + (Index(uint32_t(slot)) % slots) * 2 * context;
        int32_t* chain = committed + context;
        const Index last = end - 1; int64_t back[8];
        Index first = last;
        if (history) while (first > begin && positions[first - 1] == positions[last]) --first;
        const Index available = first - begin;
        for (Index shift = 0; shift < context; ++shift)
            back[shift] = history ? (shift < available ? tokens[first - 1 - shift] : committed[context - 1 - (shift - available)]) : chain[context - 1 - shift];
        for (Index shift = 0; shift < context; ++shift) {
            const int32_t token = shift + 1 < context ? int32_t(back[context - 2 - shift]) : tokens[last];
            if (history) committed[shift] = token;
            chain[shift] = token;
        }
        uint64_t mixed = uint64_t(int64_t(tokens[last])) * uint64_t(multipliers[0]);
        int64_t previous = tokens[last]; bool cut = false;
        for (Index shift = 1; shift < order; ++shift) {
            cut = cut || previous == eos;
            const int64_t token = cut ? eos : back[shift - 1]; previous = token;
            mixed ^= uint64_t(token) * uint64_t(multipliers[shift]);
            for (Index subhead = 0; subhead < per_order; ++subhead) {
                const Index head = (shift - 1) * per_order + subhead;
                const int64_t remainder = int64_t(mixed) % sizes[head];
                ids[sequence * pitch + head] = int32_t((remainder < 0 ? remainder + sizes[head] : remainder) + offsets[head]);
            }
        }
    }
}
int status() { return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE; }
}
using namespace r3dh_mtp_ple_select;
extern "C" int r3d_rad_logit_rerank_strict(const RadArgs* args, RadStream stream) {
    const auto* input = at(args, 0);
    const auto* weight = at(args, 1);
    const auto* candidates = at(args, 2);
    const auto* indices = at(args, 3);
    const auto* values = at(args, 4);
    const auto* pairs = at(args, 5);
    if (!input || !weight || !candidates || !indices || !values) return RAD_E_INVAL;
    RadTensor pair_view{};
    if (pairs && pairs->rank == 3) {
        if (pairs->shape[1] != 8 || pairs->shape[2] != 2) return RAD_E_SHAPE;
        if (pairs->stride[1] != 2 || pairs->stride[2] != 1) return RAD_E_STRIDE;
        pair_view = *pairs;
        pair_view.rank = 2;
        pair_view.shape[1] = 16;
        pair_view.stride[1] = 1;
        pairs = &pair_view;
    }
    const Index band = parameter(args, "M"), offset = parameter(args, "vocab_off");
    if (parameter(args, "n_embd") != 2560 || parameter(args, "n_vocab") != 124160 ||
        parameter(args, "R") != 8 || band < 1 || band > 32 || (offset != 0 && offset != 124160)) return RAD_E_SHAPE;
    if (input->dtype != RAD_BF16 || weight->dtype != r3d_head::dtype ||
        candidates->dtype != RAD_I32 || indices->dtype != RAD_I32 || values->dtype != RAD_F32 ||
        (pairs && pairs->dtype != RAD_F32)) return RAD_E_DTYPE;
    const Index rows = input->shape[0];
    if (rows < 1 || rows > band) return RAD_E_SHAPE;
    const RadTensor* matrices[]{input, candidates, indices, values, pairs};
    const Index columns[]{2560, 8, 8, 8, 16};
    uint64_t spans[5]{};
    for (int index = 0; index < 5; ++index) {
        const auto* tensor = matrices[index];
        if (!tensor) continue;
        if (tensor->rank != 2 || tensor->shape[0] != rows || tensor->shape[1] != columns[index]) return RAD_E_SHAPE;
        if (tensor->stride[1] != 1 || tensor->stride[0] < columns[index] ||
            tensor->stride[0] > (INT64_MAX / 4 - columns[index]) / rows) return RAD_E_STRIDE;
        spans[index] = uint64_t((rows - 1) * tensor->stride[0] + columns[index]) * (index ? 4 : 2);
        if (spans[index] > UINTPTR_MAX - reinterpret_cast<uintptr_t>(tensor->data)) return RAD_E_STRIDE;
        if (reinterpret_cast<uintptr_t>(tensor->data) % (index ? 4 : 2)) return RAD_E_ALIGN;
    }
    Index available = 0;
    if (weight->rank == 1) {
        if (weight->stride[0] != 1 || weight->shape[0] < r3d_head::rounded(sizeof(RadPlanesHeader) + 2 * sizeof(RadPlanesEntry))) return RAD_E_SHAPE;
        available = weight->shape[0];
    } else if (weight->rank != 2 || weight->shape[0] != 124160 || weight->shape[1] != 2560 ||
               weight->stride[0] != 2560 || weight->stride[1] != 1) return RAD_E_SHAPE;
    if (reinterpret_cast<uintptr_t>(weight->data) % 16) return RAD_E_ALIGN;
    for (int output = 2; output < 5; ++output) {
        if (!matrices[output]) continue;
        const uintptr_t begin = reinterpret_cast<uintptr_t>(matrices[output]->data);
        for (int other = 0; other < output; ++other) {
            const uintptr_t previous = reinterpret_cast<uintptr_t>(matrices[other]->data);
            if (begin < previous + spans[other] && previous < begin + spans[output] &&
                !(output == 2 && other == 1 && begin == previous && indices->stride[0] == candidates->stride[0])) return RAD_E_INVAL;
        }
        const uintptr_t head = reinterpret_cast<uintptr_t>(weight->data);
        if (begin >= head && begin - head < uint64_t(available ? available : 124160LL * 2560)) return RAD_E_INVAL;
    }
    const int device = r3dh_mtp_ple_device(args, stream);
    if (device != RAD_OK) return device;
    StrictRerank call{static_cast<const uint16_t*>(input->data), static_cast<const RadPlanesHeader*>(weight->data),
        static_cast<const int32_t*>(candidates->data), static_cast<int32_t*>(indices->data),
        static_cast<float*>(values->data), pairs ? static_cast<float*>(pairs->data) : nullptr,
        {input->stride[0], candidates->stride[0], indices->stride[0], values->stride[0], pairs ? pairs->stride[0] : 0}, available, int(offset)};
    joined_rerank<<<rows, 32, 0, reinterpret_cast<hipStream_t>(stream)>>>(call);
    return status();
}
extern "C" int r3dh_mtp_ple_logit_rerank(const RadArgs* args, RadStream stream) {
    const auto *input = at(args, 0), *weights = at(args, 1), *candidates = at(args, 2), *indices = at(args, 3), *values = at(args, 4), *pairs = at(args, 5);
    if (!input || !weights || !candidates || !indices || !values) return RAD_E_INVAL;
    const Index width = parameter(args, "n_embd"), vocabulary = parameter(args, "n_vocab"), offset = parameter(args, "vocab_off"), shortlist = parameter(args, "R");
    if (width <= 0 || vocabulary <= 0 || offset < 0 || shortlist <= 0 || shortlist > 32) return RAD_E_SHAPE;
    const Index count = rad_tensor_numel(input) / width;
    if (count < 1 || rad_tensor_numel(input) % width) return RAD_E_SHAPE;
    if (!dense(input, RAD_BF16, count * width) || !dense(candidates, RAD_I32, count * shortlist) ||
        !dense(indices, RAD_I32, count * shortlist) || !dense(values, RAD_F32, count * shortlist) ||
        (pairs && !dense(pairs, RAD_F32, count * shortlist * 2))) return RAD_E_DTYPE;
    if (weights->rank != 2 || weights->shape[0] != vocabulary) return RAD_E_SHAPE;
    if (weights->dtype == RAD_BF16 || weights->dtype == RAD_F32) {
        if (weights->shape[1] != width || !rad_tensor_is_contiguous(weights)) return RAD_E_STRIDE;
    } else {
        const Index row_bytes = (width + 2 * (width / 128) + 15) & ~Index(15);
        if (width % 128 || (weights->dtype != RAD_F8E4M3 && weights->dtype != RAD_I8) ||
            weights->shape[1] != row_bytes || weights->stride[0] != row_bytes || weights->stride[1] != 1) return RAD_E_DTYPE;
    }
    const int device = r3dh_mtp_ple_device(args, stream);
    if (device != RAD_OK) return device;
    rerank<<<count, 1, 0, reinterpret_cast<hipStream_t>(stream)>>>(*input, *weights, static_cast<const int32_t*>(candidates->data),
        static_cast<int32_t*>(indices->data), static_cast<float*>(values->data), pairs ? static_cast<float*>(pairs->data) : nullptr, width, vocabulary, offset, int(shortlist));
    return status();
}
extern "C" int r3dh_mtp_ple_rowtopkmerge(const RadArgs* args, RadStream stream) {
    const auto *pairs = at(args, 0), *indices = at(args, 1), *values = at(args, 2);
    if (!pairs || !indices || !values) return RAD_E_INVAL;
    const Index shortlist = parameter(args, "R"), world = parameter(args, "world_size", 1);
    if (shortlist <= 0 || shortlist > 32 || world <= 0 || world > 8) return RAD_E_SHAPE;
    const Index count = rad_tensor_numel(pairs) / (world * shortlist * 2);
    if (count <= 0 || rad_tensor_numel(pairs) % (world * shortlist * 2)) return RAD_E_SHAPE;
    if (!dense(pairs, RAD_F32, count * world * shortlist * 2) || !dense(indices, RAD_I32, count * shortlist) || !dense(values, RAD_F32, count * shortlist)) return RAD_E_DTYPE;
    const int device = r3dh_mtp_ple_device(args, stream);
    if (device != RAD_OK) return device;
    merge<<<count, 1, 0, reinterpret_cast<hipStream_t>(stream)>>>(static_cast<const float*>(pairs->data), static_cast<int32_t*>(indices->data), static_cast<float*>(values->data), int(shortlist), int(world));
    return status();
}
extern "C" int r3dh_mtp_ple_ngram_hint(const RadArgs* args, RadStream stream) {
    const auto *tokens = at(args, 0), *positions = at(args, 1), *cu = at(args, 2), *multipliers = at(args, 3), *sizes = at(args, 4), *offsets = at(args, 5), *state_indices = at(args, 6), *tail = at(args, 7), *ids = at(args, 8);
    if (!tokens || !positions || !cu || !multipliers || !sizes || !offsets || !tail || !ids) return RAD_E_INVAL;
    const Index order = parameter(args, "ngram"), heads = parameter(args, "heads"), eos = parameter(args, "eos", -1), history = parameter(args, "hist");
    if (order < 2 || order > 8 || heads < 1 || heads % (order - 1) || eos < 0 || (history != 0 && history != 1)) return RAD_E_SHAPE;
    const Index sequences = rad_tensor_numel(cu) - 1, slots = rad_tensor_numel(tail) / (2 * (order - 1));
    if (sequences < 1 || slots < 1 || ids->rank != 2 || ids->shape[0] < sequences || ids->shape[1] != heads || ids->stride[1] != 1 || ids->stride[0] < heads) return RAD_E_SHAPE;
    if (!dense(tokens, RAD_I32, 1) || !dense(positions, RAD_I32, rad_tensor_numel(tokens)) || !dense(cu, RAD_I32, sequences + 1) ||
        !dense(multipliers, RAD_I64, order) || !dense(sizes, RAD_I64, heads) || !dense(offsets, RAD_I64, heads) ||
        !dense(tail, RAD_I32, slots * 2 * (order - 1)) || ids->dtype != RAD_I32 || (state_indices && !dense(state_indices, RAD_I32, sequences))) return RAD_E_DTYPE;
    const int device = r3dh_mtp_ple_device(args, stream);
    if (device != RAD_OK) return device;
    hint<<<1, 1, 0, reinterpret_cast<hipStream_t>(stream)>>>(static_cast<const int32_t*>(tokens->data), static_cast<const int32_t*>(positions->data), static_cast<const int32_t*>(cu->data),
        static_cast<const int64_t*>(multipliers->data), static_cast<const int64_t*>(sizes->data), static_cast<const int64_t*>(offsets->data), state_indices ? static_cast<const int32_t*>(state_indices->data) : nullptr,
        static_cast<int32_t*>(tail->data), static_cast<int32_t*>(ids->data), sequences, heads, order, eos, history, slots, ids->stride[0]);
    return status();
}
