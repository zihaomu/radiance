#include "r3dh_ple_host.h"
#include <array>
#include <bit>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {
uint64_t checks = 0;
void require(bool condition, const char* message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}
RadParam integer(const char* key, int64_t value) {
    RadParam result{}; result.key = key; result.kind = RAD_P_INT; result.ival = value; return result;
}
RadTensor tensor(void* data, uint32_t type, int64_t rows, int64_t columns = 0) {
    RadTensor result{}; result.data = data; result.dtype = type; result.rank = columns ? 2 : 1;
    result.shape[0] = rows; result.shape[1] = columns;
    result.stride[0] = columns ? columns : 1; result.stride[1] = 1; return result;
}
void hint_cases() {
    for (int order = 2; order <= 8; ++order) {
        const int context = order - 1, heads = context * 2;
        int32_t tokens[]{5, -9, 0, 0, 7, 11}, positions[]{0, 1, 2, 2, 4, 5};
        int32_t cumulative[]{0, 4, 4, 6}, indices[]{5, -1, 0};
        int64_t multipliers[8], sizes[14], offsets[14];
        for (int index = 0; index < order; ++index) multipliers[index] = INT64_MAX - index * 6;
        for (int head = 0; head < heads; ++head) { sizes[head] = 13 + head; offsets[head] = head * 100; }
        std::vector<int32_t> state(3 * 2 * context), expected;
        for (size_t index = 0; index < state.size(); ++index) state[index] = int32_t(index) - 5;
        std::vector<int32_t> output(3 * (heads + 2), -987), oracle(output);
        RadParam params[]{integer("M", 3), integer("heads", heads), integer("ngram", order), integer("eos", 0), integer("hist", 1)};
        RadTensor values[]{tensor(tokens, RAD_I32, 6), tensor(positions, RAD_I32, 6), tensor(cumulative, RAD_I32, 4),
            tensor(multipliers, RAD_I64, order), tensor(sizes, RAD_I64, heads), tensor(offsets, RAD_I64, heads),
            tensor(indices, RAD_I32, 3), tensor(state.data(), RAD_I32, 3, 2 * context), tensor(output.data(), RAD_I32, 3, heads)};
        values[8].stride[0] = heads + 2;
        RadArgs args{}; args.t = values; args.n_t = 9; args.p = params; args.n_p = 5; args.world_size = 1;
        for (int repeat = 0; repeat < 4; ++repeat) {
            params[4].ival = repeat == 0;
            tokens[5] = repeat ? 0 : 11;
            expected = state;
            for (int sequence = 0; sequence < 3; ++sequence) {
                const int begin = cumulative[sequence], end = cumulative[sequence + 1];
                if (begin == end || indices[sequence] < 0) {
                    for (int head = 0; head < heads; ++head) oracle[sequence * (heads + 2) + head] = int32_t(offsets[head]);
                    continue;
                }
                const int slot = indices[sequence] % 3, base = slot * 2 * context;
                std::vector<int32_t> history(context + 1);
                int first = end - 1;
                if (!repeat) while (first > begin && positions[first - 1] == positions[end - 1]) --first;
                for (int distance = 1; distance <= context; ++distance)
                    history[context - distance] = !repeat
                        ? (first - distance >= begin ? tokens[first - distance]
                            : expected[base + context + first - begin - distance])
                        : expected[base + 2 * context - distance];
                history[context] = tokens[end - 1];
                for (int index = 0; index < context; ++index) {
                    expected[base + context + index] = history[index + 1];
                    if (!repeat) expected[base + index] = history[index + 1];
                }
                for (int head = 0; head < heads; ++head) {
                    const int length = head / 2 + 2;
                    uint64_t hash = 0;
                    bool eos = false;
                    for (int distance = 0; distance < length; ++distance) {
                        const int64_t token = eos ? 0 : history[context - distance];
                        hash ^= uint64_t(token) * uint64_t(multipliers[distance]);
                        eos = eos || token == 0;
                    }
                    const __int128 signed_hash = (hash >> 63) ? __int128(hash) - (__int128(1) << 64) : __int128(hash);
                    const int64_t residue = int64_t(signed_hash % sizes[head]);
                    oracle[sequence * (heads + 2) + head] = int32_t((residue + sizes[head]) % sizes[head] + offsets[head]);
                }
            }
            require(r3dh_ple_host_ngram_hint(&args, nullptr) == RAD_OK, "hint launch");
            for (size_t index = 0; index < state.size(); ++index) require(state[index] == expected[index], "hint tail oracle");
            for (size_t index = 0; index < output.size(); ++index) require(output[index] == oracle[index], "hint ids/guards oracle");
        }
        expected = state;
        sizes[heads - 1] = 0;
        require(r3dh_ple_host_ngram_hint(&args, nullptr) == RAD_E_SHAPE && state == expected, "reject sizes atomically");
        sizes[heads - 1] = 13 + heads - 1; cumulative[3] = 7;
        require(r3dh_ple_host_ngram_hint(&args, nullptr) == RAD_E_SHAPE && state == expected, "reject cumulative bounds");
        cumulative[3] = 6; params[4].ival = 2;
        require(r3dh_ple_host_ngram_hint(&args, nullptr) == RAD_E_SHAPE, "reject history");
        params[4].ival = 0; values[7].dtype = RAD_F32;
        require(r3dh_ple_host_ngram_hint(&args, nullptr) == RAD_E_DTYPE, "reject tail dtype");
    }
    require(r3dh_ple_host_ngram_hint(nullptr, nullptr) == RAD_E_INVAL, "null hint");
}
void registry_and_bf16(const char* path) {
    require(r3dh_ple_host_registry_count() == 3 && r3dh_ple_host_registry_at(0),
        "closed provider keeps static enumeration");
    require(r3dh_ple_host_open("/missing/r3dh_ple_host_provider.so") == RAD_E_NOTFOUND, "missing provider");
    require(r3dh_ple_host_open(path) == RAD_OK && r3dh_ple_host_open(path) == RAD_OK, "open provider");
    require(r3dh_ple_host_registry_count() == 3 && !r3dh_ple_host_registry_at(-1)
        && !r3dh_ple_host_registry_at(3), "private registry bounds");
    const char* operations[]{"ngram_ids", "embed_lookup_q", "ngram_hint"};
    for (int index = 0; index < 3; ++index) {
        const auto* row = r3dh_ple_host_registry_at(index);
        require(row && row->domain == RAD_DOMAIN_HOST && !std::strcmp(row->op, operations[index])
            && !std::strncmp(row->name, "r3dh_ple_host_", 14), "only scoped HOST rows");
    }
    require(r3dh_ple_host_open("/different/provider.so") == RAD_E_STATE, "cannot replace live provider");
    const auto* row = r3dh_ple_host_registry_at(1);
    std::vector<uint16_t> table(64 * 160), output(4 * 160 + 2, 0xa5a5);
    for (size_t index = 0; index < table.size(); ++index) table[index] = uint16_t(index * 17);
    int32_t tokens[]{0, 63, 17, -1}, ahead[]{25, 39};
    RadParam params[]{integer("M", 4), integer("n_embd", 160), integer("n_vocab", 64), {}};
    params[3].key = "dtype"; params[3].kind = RAD_P_STR; params[3].sval = "bf16";
    RadTensor values[]{tensor(tokens, RAD_I32, 4), tensor(table.data(), RAD_BF16, 64, 160), {},
        tensor(output.data() + 1, RAD_BF16, 1, 640), tensor(ahead, RAD_I32, 2)};
    RadArgs args{}; args.t = values; args.n_t = 5; args.p = params; args.n_p = 4; args.world_size = 1;
    require(row->launch(&args, nullptr) == RAD_OK, "stored BF16 gather");
    for (int index = 0; index < 640; ++index)
        require(output[index + 1] == (tokens[index / 160] < 0 ? 0 : table[tokens[index / 160] * 160 + index % 160]), "stored BF16 oracle");
    require(output.front() == 0xa5a5 && output.back() == 0xa5a5, "BF16 guards");
    require(r3dh_ple_host_schema_count() == 7, "schemas count");
    const auto* schema = r3dh_ple_host_schema_at(6);
    require(!std::strcmp(schema->op, "ngram_hint") && schema->n_params == 5 && schema->n_operands == 9, "hint ABI schema");
}
}
int main(int argc, char** argv) {
    try {
        require(argc == 2, "usage: cpu_check libavx.so");
        hint_cases(); registry_and_bf16(argv[1]);
        std::printf("PLE_HOST_CPU_PASS checks=%llu hint_cases=28 rejection_cases=29 bf16_values=640 HOST=3 GPU_used=0\n", static_cast<unsigned long long>(checks));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "PLE_HOST_CPU_FAIL %s checks=%llu\n", error.what(), static_cast<unsigned long long>(checks)); return 1;
    }
}
