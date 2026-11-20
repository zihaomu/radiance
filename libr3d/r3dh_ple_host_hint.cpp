#include "r3dh_ple_host.h"

#include <bit>
#include <climits>
#include <cstdint>

namespace {
bool dense(const RadTensor& value, uint32_t dtype, int64_t elements) {
    return value.data && value.dtype == dtype && value.rank >= 1 && value.rank <= 2
        && rad_tensor_is_contiguous(&value) && rad_tensor_numel(&value) == elements;
}
}

extern "C" int r3dh_ple_host_ngram_hint(const RadArgs* args, RadStream) {
    if (!args || !args->t || args->n_t != 9 || args->n_p < 0 || (args->n_p && !args->p))
        return RAD_E_INVAL;
    const int64_t order = rad_param_geti(args->p, args->n_p, "ngram", 0);
    const int64_t heads = rad_param_geti(args->p, args->n_p, "heads", 0);
    const int64_t eos = rad_param_geti(args->p, args->n_p, "eos", -1);
    const int64_t history = rad_param_geti(args->p, args->n_p, "hist", -1);
    if (order < 2 || order > 8 || heads < 1 || heads > 256 || heads % (order - 1)
        || eos < 0 || eos > INT32_MAX || (history != 0 && history != 1)) return RAD_E_SHAPE;
    const auto &tokens = args->t[0], &positions = args->t[1], &cu = args->t[2];
    const auto &multipliers = args->t[3], &sizes = args->t[4], &offsets = args->t[5];
    const auto &indices = args->t[6], &tail = args->t[7], &ids = args->t[8];
    const int64_t context = order - 1;
    if (tokens.rank != 1 || positions.rank != 1 || cu.rank != 1
        || cu.shape[0] < 2 || cu.shape[0] > 2049
        || tail.rank != 2 || tail.shape[0] < 1 || tail.shape[0] > INT32_MAX
        || tail.shape[1] != 2 * context || ids.rank != 2
        || ids.shape[0] < cu.shape[0] - 1 || ids.shape[1] != heads
        || ids.stride[1] != 1 || ids.stride[0] < heads) return RAD_E_SHAPE;
    const int64_t sequences = cu.shape[0] - 1, slots = tail.shape[0];
    const int64_t token_count = tokens.shape[0];
    if (token_count < 1 || token_count > INT32_MAX) return RAD_E_SHAPE;
    if (!dense(tokens, RAD_I32, token_count) || !dense(positions, RAD_I32, token_count)
        || !dense(cu, RAD_I32, sequences + 1) || !dense(multipliers, RAD_I64, order)
        || !dense(sizes, RAD_I64, heads) || !dense(offsets, RAD_I64, heads)
        || !dense(tail, RAD_I32, slots * 2 * context) || !ids.data || ids.dtype != RAD_I32
        || (indices.data && !dense(indices, RAD_I32, sequences))) return RAD_E_DTYPE;
    const auto* token = static_cast<const int32_t*>(tokens.data);
    const auto* position = static_cast<const int32_t*>(positions.data);
    const auto* cumulative = static_cast<const int32_t*>(cu.data);
    const auto* multiplier = static_cast<const int64_t*>(multipliers.data);
    const auto* size = static_cast<const int64_t*>(sizes.data);
    const auto* offset = static_cast<const int64_t*>(offsets.data);
    const auto* state_index = static_cast<const int32_t*>(indices.data);
    auto* state = static_cast<int32_t*>(tail.data);
    auto* output = static_cast<int32_t*>(ids.data);
    if (cumulative[0] != 0 || cumulative[sequences] < 0
        || cumulative[sequences] > token_count) return RAD_E_SHAPE;
    for (int64_t sequence = 0; sequence < sequences; ++sequence)
        if (cumulative[sequence] < 0 || cumulative[sequence] > cumulative[sequence + 1])
            return RAD_E_SHAPE;
    for (int64_t head = 0; head < heads; ++head)
        if (size[head] < 1 || size[head] > INT32_MAX || offset[head] < 0
            || offset[head] > INT32_MAX - (size[head] - 1)) return RAD_E_SHAPE;
    for (int64_t sequence = 0; sequence < sequences; ++sequence) {
        const int64_t begin = cumulative[sequence], end = cumulative[sequence + 1];
        const int32_t slot = state_index ? state_index[sequence] : int32_t(sequence);
        if (end <= begin || slot < 0) {
            for (int64_t head = 0; head < heads; ++head)
                output[sequence * ids.stride[0] + head] = int32_t(offset[head]);
            continue;
        }
        int32_t* committed = state + (int64_t(slot) % slots) * 2 * context;
        int32_t* chain = committed + context;
        const int64_t last = end - 1;
        int64_t first = last;
        if (history)
            while (first > begin && position[first - 1] == position[last]) --first;
        const int64_t available = first - begin;
        int64_t back[7]{};
        for (int64_t shift = 0; shift < context; ++shift)
            back[shift] = history ? (shift < available ? token[first - 1 - shift]
                : committed[context - 1 - (shift - available)]) : chain[context - 1 - shift];
        for (int64_t shift = 0; shift < context; ++shift) {
            const int32_t value = shift + 1 < context ? int32_t(back[context - 2 - shift]) : token[last];
            if (history) committed[shift] = value;
            chain[shift] = value;
        }
        uint64_t mixed = uint64_t(int64_t(token[last])) * uint64_t(multiplier[0]);
        int64_t previous = token[last];
        bool cut = false;
        for (int64_t shift = 1; shift < order; ++shift) {
            cut = cut || previous == eos;
            const int64_t value = cut ? eos : back[shift - 1];
            previous = value;
            mixed ^= uint64_t(value) * uint64_t(multiplier[shift]);
            for (int64_t subhead = 0; subhead < heads / context; ++subhead) {
                const int64_t head = (shift - 1) * (heads / context) + subhead;
                const int64_t remainder = std::bit_cast<int64_t>(mixed) % size[head];
                output[sequence * ids.stride[0] + head] = int32_t(
                    (remainder < 0 ? remainder + size[head] : remainder) + offset[head]);
            }
        }
    }
    return RAD_OK;
}

extern "C" const RadKernelInfo* r3dh_ple_host_hint_kernel() {
    static const RadConstraint limits[]{
        {"M", RAD_C_GE, 1, nullptr}, {"M", RAD_C_LE, 2048, nullptr},
        {"heads", RAD_C_GE, 1, nullptr}, {"heads", RAD_C_LE, 256, nullptr},
        {"ngram", RAD_C_GE, 2, nullptr}, {"ngram", RAD_C_LE, 8, nullptr},
        {"hist", RAD_C_IN, 0, "0,1"}};
    static const RadKernelInfo result = [] {
        RadKernelInfo value{};
        value.name = "r3dh_ple_host_ngram_hint";
        value.op = "ngram_hint";
        value.family = "ple";
        value.computes = "HOST committed/draft ngram hint ids";
        value.domain = RAD_DOMAIN_HOST;
        value.priority = 100;
        value.launch = r3dh_ple_host_ngram_hint;
        value.constraints = limits;
        value.n_constraints = int(sizeof(limits) / sizeof(limits[0]));
        return value;
    }();
    return &result;
}
