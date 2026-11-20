#include "rad_sample.h"
#include <array>
#include <functional>

namespace sample_pair_aql {
struct Counts { uint64_t calls = 0, bytes = 0, rejects = 0, draws = 0; };
uint32_t bits(float value) { uint32_t word; std::memcpy(&word, &value, 4); return word; }
float value(uint32_t word) { float result; std::memcpy(&result, &word, 4); return result; }
void equal(const std::vector<uint8_t>& expected, const std::vector<uint8_t>& actual,
           const char* stage, int operand) {
    if (expected != actual) {
        size_t offset = 0;
        while (offset < expected.size() && expected[offset] == actual[offset]) ++offset;
        std::printf("SAMPLE_PAIR_DIFF stage=%s operand=%d byte=%zu expected=%u actual=%u\n",
                    stage, operand, offset, unsigned(expected[offset]), unsigned(actual[offset]));
    }
    REQUIRE(expected == actual);
}
using Oracle = std::function<void(std::vector<RadTensor>&)>;
void run(const RadKernelInfo* native, const RadKernelInfo* reference,
         const std::vector<Buffer*>& buffers, std::vector<RadParam> parameters,
         RadStream stream, const Oracle& oracle, Counts& counts, bool null_pairs = false,
         int resident_prefix = 0) {
    std::vector<RadTensor> host, device, ref;
    std::vector<std::vector<uint8_t>> copies;
    for (auto* buffer : buffers) {
        if (int(host.size()) >= resident_prefix) buffer->push(stream);
        host.push_back(buffer->host);
        device.push_back(buffer->device);
        copies.push_back(buffer->expected);
    }
    REQUIRE_EQ(rad_stream_sync(stream), RAD_OK);
    if (null_pairs) { host[4].data = nullptr; device[4].data = nullptr; device[4].dtype = RAD_U8; }
    oracle(host);
    for (size_t index = 0; index < buffers.size(); ++index) {
        ref.push_back(buffers[index]->host);
        ref.back().data = copies[index].data() + 64;
    }
    if (!std::strcmp(native->op, "sample_topk") && ref.size() == 5) ref[4].dtype = RAD_U32;
    if (null_pairs) ref[4].data = nullptr;
    RadArgs args{};
    args.p = parameters.data(); args.n_p = int(parameters.size());
    args.t = ref.data(); args.n_t = int(ref.size());
    REQUIRE_EQ(reference->launch(&args, nullptr), RAD_OK);
    for (size_t index = 0; index < buffers.size(); ++index)
        equal(buffers[index]->expected, copies[index], "independent-vs-libref", int(index));
    args.t = device.data();
    REQUIRE_EQ(native->launch(&args, stream), RAD_OK);
    REQUIRE_EQ(rad_stream_sync(stream), RAD_OK);
    for (size_t index = 0; index < buffers.size(); ++index) {
        equal(buffers[index]->expected, buffers[index]->read(stream), native->op, int(index));
        counts.bytes += buffers[index]->expected.size();
    }
    ++counts.calls;
}
void topk(std::vector<RadTensor>& tensors, int offset) {
    const auto& logits = tensors[0];
    const auto* params = static_cast<const RadSampleParams*>(tensors[1].data);
    const int width = int(tensors[2].shape[1]);
    for (int row = 0; row < logits.shape[0]; ++row) {
        const auto* input = static_cast<const float*>(logits.data) + row * logits.stride[0];
        auto* indices = static_cast<int32_t*>(tensors[2].data) + row * tensors[2].stride[0];
        auto* values = static_cast<float*>(tensors[3].data) + row * tensors[3].stride[0];
        auto* pairs = tensors.size() > 4 && tensors[4].data
            ? static_cast<uint32_t*>(tensors[4].data) + row * tensors[4].stride[0] : nullptr;
        const int keep = params[row].flags & RAD_SP_GREEDY ? 0
            : params[row].top_k > 0 ? std::min(params[row].top_k, width) : width;
        std::vector<int> ordered;
        bool nonfinite = false;
        for (int column = 0; column < logits.shape[1]; ++column) {
            ordered.push_back(column);
            nonfinite |= std::isnan(input[column]);
        }
        if (!nonfinite) std::stable_sort(ordered.begin(), ordered.end(), [&](int left, int right) {
            return input[left] > input[right];
        });
        float previous = INFINITY;
        int previous_id = -1;
        for (int item = 0; item < width; ++item) {
            int chosen = -1;
            if (item < keep && !nonfinite && item < int(ordered.size())) chosen = ordered[item];
            if (item < keep && nonfinite) {
                for (int column = 0; column < logits.shape[1]; ++column) {
                    const float candidate = input[column];
                    if (candidate > previous || (candidate == previous && column <= previous_id)) continue;
                    if (chosen < 0 || candidate > input[chosen]) chosen = column;
                }
            }
            const float selected = chosen < 0 ? -INFINITY : input[chosen];
            indices[item] = chosen; values[item] = selected;
            if (pairs) { pairs[item * 2] = chosen < 0 ? UINT32_MAX : uint32_t(chosen + offset);
                         pairs[item * 2 + 1] = bits(selected); }
            if (chosen >= 0) { previous = selected; previous_id = chosen; }
        }
    }
}
void merge(std::vector<RadTensor>& tensors) {
    const auto* params = static_cast<const RadSampleParams*>(tensors[1].data);
    for (int row = 0; row < tensors[2].shape[0]; ++row) {
        std::vector<std::pair<float, int32_t>> ranked;
        const auto* words = static_cast<const uint32_t*>(tensors[0].data) + row * tensors[0].stride[0];
        for (int item = 0; item < tensors[0].shape[1] / 2; ++item) {
            const int32_t id = int32_t(words[item * 2]);
            const float logit = value(words[item * 2 + 1]);
            if (id >= 0 && !std::isnan(logit)) ranked.emplace_back(logit, id);
        }
        std::stable_sort(ranked.begin(), ranked.end(), [](const auto& left, const auto& right) {
            return left.first != right.first ? left.first > right.first : left.second < right.second;
        });
        ranked.erase(std::unique(ranked.begin(), ranked.end()), ranked.end());
        const int width = int(tensors[2].shape[1]);
        const int keep = params[row].flags & RAD_SP_GREEDY ? 0
            : params[row].top_k > 0 ? std::min(params[row].top_k, width) : width;
        for (int item = 0; item < width; ++item) {
            const bool live = item < keep && item < int(ranked.size());
            static_cast<int32_t*>(tensors[2].data)[row * tensors[2].stride[0] + item] =
                live ? ranked[item].second : -1;
            static_cast<float*>(tensors[3].data)[row * tensors[3].stride[0] + item] =
                live ? ranked[item].first : -INFINITY;
        }
    }
}
void pick(std::vector<RadTensor>& tensors) {
    const auto* params = static_cast<const RadSampleParams*>(tensors[2].data);
    for (int row = 0; row < tensors[0].shape[0]; ++row) {
        if (params[row].flags & RAD_SP_GREEDY) continue;
        const auto* indices = static_cast<const int32_t*>(tensors[0].data) + row * tensors[0].stride[0];
        const auto* values = static_cast<const float*>(tensors[1].data) + row * tensors[1].stride[0];
        const int width = int(tensors[0].shape[1]);
        float maximum = -INFINITY;
        for (int item = 0; item < width; ++item)
            if (indices[item] >= 0 && values[item] > maximum) maximum = values[item];
        int32_t chosen = -1, last = -1;
        if (maximum != -INFINITY) {
            float sum = 0;
            for (int item = 0; item < width; ++item)
                if (indices[item] >= 0) sum += float(std::exp(double(values[item] - maximum)));
            const float inverse = sum > 0 ? 1.0f / sum : 0;
            const float draw = rad_sample_uniform01(params[row].seed, params[row].pos);
            float cumulative = 0;
            for (int item = 0; item < width; ++item) {
                if (indices[item] < 0) continue;
                if (values[item] > -INFINITY) last = indices[item];
                volatile float probability = float(std::exp(double(values[item] - maximum))) * inverse;
                cumulative += probability;
                if (draw < cumulative) { chosen = indices[item]; break; }
            }
            if (chosen < 0) chosen = last;
        }
        static_cast<int32_t*>(tensors[3].data)[row] = chosen;
    }
}
void parameters(Buffer& buffer, int rows, int keep, int step, bool greedy = false) {
    for (int row = 0; row < rows; ++row) {
        RadSampleParams params{};
        params.seed = UINT64_C(0xfedcba9876543210) + uint64_t(step * 7919 + row);
        params.pos = UINT64_C(0x1234567800000000) + uint64_t(step * 8 + row);
        params.temp = keep == 1 ? 0 : 1;
        params.top_k = keep;
        params.top_p = params.typical_p = params.rep_penalty = 1;
        params.mask_row = -1;
        params.flags = greedy ? RAD_SP_GREEDY : 0;
        std::memcpy(buffer.data<uint8_t>() + row * sizeof(params), &params, sizeof(params));
    }
}
}

TEST(r3d_sample_pair_actual_aql) {
    using namespace sample_pair_aql;
    RadDeviceProps props{};
    REQUIRE_EQ(rad_dev_props(0, &props), RAD_OK);
    if (props.is_host_backend) { REQUIRE(!std::getenv("RAD_R3D_REQUIRE_DEVICE")); SKIP("gfx1100 AQL required"); }
    REQUIRE_EQ(std::strcmp(props.arch, "gfx1100"), 0);
    REQUIRE_EQ(rad_dev_set(0), RAD_OK);
    Library native("libr3d"), reference("libref");
    const auto* select = native.ngram("sample_topk");
    const auto* combine = native.ngram("sample_merge_topk");
    const auto* draw = native.ngram("sample_pick");
    const auto* ref_select = reference.ngram("sample_topk");
    const auto* ref_combine = reference.ngram("sample_merge_topk");
    const auto* ref_draw = reference.ngram("sample_pick");
    REQUIRE_EQ(std::strcmp(select->name, "sample_topk_f32"), 0);
    REQUIRE_EQ(std::strcmp(draw->name, "sample_pick_f32"), 0);
    std::printf("SAMPLE_PAIR_ROWS topk=%s merge=%s pick=%s params=%zu seed=%zu pos=%zu\n",
        select->name, combine->name, draw->name, sizeof(RadSampleParams),
        offsetof(RadSampleParams, seed), offsetof(RadSampleParams, pos));
    Queue first(0), second(1);
    Counts counts;
    auto chain = [&](int rows, int step, int keep, bool padded, bool greedy, int pair_type) {
        Buffer params(RAD_U32, {rows, 24}, RAD_MEM_DEVICE);
        parameters(params, rows, keep, step, greedy);
        Buffer gathered(RAD_U32, {rows, 4096}, RAD_MEM_DEVICE);
        for (int rank = 0; rank < 2; ++rank) {
            RadStream stream = rank ? second.stream : first.stream;
            Buffer logits(RAD_F32, {rows, 124160}, RAD_MEM_DEVICE, padded ? 124167 : 0);
            Buffer indices(RAD_I32, {rows, 1024}, RAD_MEM_DEVICE, padded ? 1031 : 0);
            Buffer values(RAD_F32, {rows, 1024}, RAD_MEM_DEVICE, padded ? 1033 : 0);
            Buffer pairs(pair_type, {rows, 2048}, RAD_MEM_DEVICE, padded ? 2055 : 0);
            for (int row = 0; row < rows; ++row) for (int column = 0; column < 124160; ++column)
                logits.data<float>()[row * logits.host.stride[0] + column] = rank && column >= 123908 ? -INFINITY :
                    float(int((uint32_t(column) * 7919u + uint32_t(row * 71 + step * 131 + rank * 23)) % 4093u) - 2046) / 256;
            const int offset = rank * 124160;
            run(select, ref_select, {&logits, &params, &indices, &values, &pairs},
                {integer("M", 8), integer("n_vocab", 124160), integer("vocab_off", offset)}, stream,
                [&](auto& tensors) { topk(tensors, offset); }, counts);
            const auto produced = pairs.read(stream);
            for (int row = 0; row < rows; ++row)
                std::memcpy(gathered.data<uint32_t>() + row * 4096 + rank * 2048,
                            produced.data() + 64 + row * pairs.host.stride[0] * 4, 8192);
        }
        Buffer indices(RAD_I32, {rows, 1024}, RAD_MEM_DEVICE);
        Buffer values(RAD_F32, {rows, 1024}, RAD_MEM_DEVICE);
        Buffer tokens(RAD_I32, {rows}, RAD_MEM_DEVICE);
        run(combine, ref_combine, {&gathered, &params, &indices, &values},
            {integer("M", rows), integer("n_cand", 1024), integer("world_size", 2)}, first.stream, merge, counts);
        run(draw, ref_draw, {&indices, &values, &params, &tokens},
            {integer("M", 8), integer("n_cand", 1024)}, second.stream, pick, counts, false, 2);
        counts.draws += rows;
        std::printf("PASS SAMPLE_PAIR_CHAIN rows=%d band=8 vocab=124160 ncand=1024 topk=%d greedy=%d pitch=%d pair_dtype=%d step=%d\n",
                    rows, keep, int(greedy), int(padded), pair_type, step);
        std::fflush(stdout);
    };
    for (int rows : {1, 3, 8}) for (int step = 0; step < 2; ++step)
        chain(rows, step, 1, bool(step), false, RAD_U32);
    for (int rows : {1, 3, 8}) chain(rows, 19, 1024, true, false, RAD_U32);
    for (int step = 0; step < 16; ++step) chain(8, step + 100, 64, bool(step % 2), false, RAD_U32);
    chain(3, 501, 17, true, true, RAD_U32);
    chain(3, 502, 17, true, false, RAD_I32);
    chain(3, 503, 17, false, false, RAD_F32);
    std::printf("PASS SAMPLE_PAIR_FIXED_SEED checks=128 seed_bits=64 pos_bits=64\n");
    std::printf("PASS SAMPLE_PAIR_AQL calls=%llu exact_bytes=%llu draws=%llu rejects=%llu tolerance=0 queues=2\n",
        (unsigned long long)counts.calls, (unsigned long long)counts.bytes,
        (unsigned long long)counts.draws, (unsigned long long)counts.rejects);
}

TEST(r3d_sample_pair_edges_actual_aql) {
    using namespace sample_pair_aql;
    RadDeviceProps props{};
    REQUIRE_EQ(rad_dev_props(0, &props), RAD_OK);
    if (props.is_host_backend) { REQUIRE(!std::getenv("RAD_R3D_REQUIRE_DEVICE")); SKIP("gfx1100 AQL required"); }
    REQUIRE_EQ(rad_dev_set(0), RAD_OK);
    Library native("libr3d"), reference("libref");
    const auto* select = native.ngram("sample_topk");
    const auto* ref_select = reference.ngram("sample_topk");
    Queue first(0), second(1);
    Counts counts;
    for (int scenario = 0; scenario < 6; ++scenario) for (int step = 0; step < 2; ++step) {
        const int vocab = scenario == 0 ? 17 : scenario == 1 ? 257 : 1025;
        const int offset = step * 124160;
        const int pair_type = scenario % 3 == 0 ? RAD_U32 : scenario % 3 == 1 ? RAD_I32 : RAD_F32;
        RadStream stream = step ? second.stream : first.stream;
        Buffer logits(RAD_F32, {3, vocab}, RAD_MEM_DEVICE, vocab + 7);
        Buffer params(RAD_U32, {3, 24}, RAD_MEM_DEVICE);
        Buffer indices(RAD_I32, {3, 1024}, RAD_MEM_DEVICE, 1031);
        Buffer values(RAD_F32, {3, 1024}, RAD_MEM_DEVICE, 1033);
        Buffer pairs(pair_type, {3, 2048}, RAD_MEM_DEVICE, 2055);
        parameters(params, 3, 0, step + scenario * 2);
        for (int row = 0; row < 3; ++row) {
            auto* input = logits.data<float>() + row * logits.host.stride[0];
            for (int column = 0; column < vocab; ++column)
                input[column] = float((column * 19 + row * 7 + step * 11) % 31 - 15) / 8;
            input[1] = -0.0f; input[2] = 0.0f;
            if (scenario == 1) { input[0] = INFINITY; input[vocab - 1] = INFINITY; }
            if (scenario == 2) input[0] = value(0x7fc12345u + uint32_t(row));
            if (scenario == 3) input[5 + row * 2] = value(0xffc12345u + uint32_t(row));
            if (scenario == 4) std::fill(input, input + vocab, -INFINITY);
            if (scenario == 5) { input[0] = -INFINITY; input[17] = INFINITY; input[vocab - 1] = -INFINITY; }
            if (row == 2) params.data<RadSampleParams>()[row].flags = RAD_SP_GREEDY;
        }
        const std::vector<Buffer*> buffers{&logits, &params, &indices, &values, &pairs};
        const std::vector<RadParam> keys{integer("M", 8), integer("n_vocab", vocab), integer("vocab_off", offset)};
        std::printf("SAMPLE_PAIR_EDGE scenario=%d step=%d vocab=%d pair_dtype=%d\n", scenario, step, vocab, pair_type);
        std::fflush(stdout);
        run(select, ref_select, buffers, keys, stream, [&](auto& tensors) { topk(tensors, offset); }, counts);
        if (scenario == 0) {
            run(select, ref_select, {&logits, &params, &indices, &values}, keys, stream,
                [&](auto& tensors) { topk(tensors, offset); }, counts);
            run(select, ref_select, buffers, keys, stream,
                [&](auto& tensors) { topk(tensors, offset); }, counts, true);
        }
        std::vector<RadTensor> original;
        for (auto* buffer : buffers) original.push_back(buffer->device);
        auto reject = [&](int expected, const std::function<void(std::vector<RadTensor>&)>& change) {
            auto tensors = original;
            change(tensors);
            RadArgs args{}; args.t = tensors.data(); args.n_t = int(tensors.size());
            args.p = keys.data(); args.n_p = int(keys.size());
            REQUIRE_EQ(select->launch(&args, reinterpret_cast<RadStream>(0x12345)), expected);
            ++counts.rejects;
        };
        for (int dtype : {RAD_U8, RAD_I16}) reject(RAD_E_DTYPE, [&](auto& tensors) { tensors[4].dtype = dtype; });
        reject(RAD_E_ALIGN, [](auto& tensors) { tensors[4].data = static_cast<uint8_t*>(tensors[4].data) + 1; });
        reject(RAD_E_STRIDE, [](auto& tensors) { tensors[4].stride[1] = 2; });
        reject(RAD_E_STRIDE, [](auto& tensors) { tensors[4].stride[0] = 2047; });
        reject(RAD_E_SHAPE, [](auto& tensors) { tensors[4].shape[0] = 1; });
        reject(RAD_E_ALIGN, [](auto& tensors) { tensors[1].data = static_cast<uint8_t*>(tensors[1].data) + 4; });
        reject(RAD_E_DTYPE, [](auto& tensors) { tensors[1].dtype = RAD_U8; });
        reject(RAD_E_STRIDE, [](auto& tensors) { tensors[1].stride[0] = 25; });
        reject(RAD_E_INVAL, [](auto& tensors) { tensors[1].data = nullptr; });
        for (auto* buffer : buffers) equal(buffer->expected, buffer->read(stream), "rejected-unchanged", 0);
        Buffer gathered(RAD_U32, {3, 4096}, RAD_MEM_DEVICE, 4103);
        Buffer merged_indices(RAD_I32, {3, 1024}, RAD_MEM_DEVICE, 1031);
        Buffer merged_values(RAD_F32, {3, 1024}, RAD_MEM_DEVICE, 1033);
        Buffer tokens(RAD_I32, {3}, RAD_MEM_DEVICE);
        const auto produced = pairs.read(stream);
        for (int row = 0; row < 3; ++row) for (int rank = 0; rank < 2; ++rank)
            std::memcpy(gathered.data<uint32_t>() + row * gathered.host.stride[0] + rank * 2048,
                        produced.data() + 64 + row * pairs.host.stride[0] * 4, 8192);
        run(native.ngram("sample_merge_topk"), reference.ngram("sample_merge_topk"),
            {&gathered, &params, &merged_indices, &merged_values},
            {integer("M", 3), integer("n_cand", 1024), integer("world_size", 2)}, stream, merge, counts);
        run(native.ngram("sample_pick"), reference.ngram("sample_pick"),
            {&merged_indices, &merged_values, &params, &tokens},
            {integer("M", 8), integer("n_cand", 1024)}, stream, pick, counts, false, 2);
    }
    for (int step = 0; step < 2; ++step) {
        Buffer gathered(RAD_U32, {3, 4096}, RAD_MEM_DEVICE, 4103);
        Buffer params(RAD_U32, {3, 24}, RAD_MEM_DEVICE);
        Buffer indices(RAD_I32, {3, 1024}, RAD_MEM_DEVICE, 1031);
        Buffer values(RAD_F32, {3, 1024}, RAD_MEM_DEVICE, 1033);
        Buffer tokens(RAD_I32, {3}, RAD_MEM_DEVICE);
        parameters(params, 3, 0, 900 + step);
        params.data<RadSampleParams>()[2].flags = RAD_SP_GREEDY;
        for (int row = 0; row < 3; ++row) {
            auto* words = gathered.data<uint32_t>() + row * gathered.host.stride[0];
            for (int item = 0; item < 2048; ++item) {
                const int local = item % 1024;
                words[2 * item] = local < 19 ? uint32_t(local + row * 124160) : UINT32_MAX;
                words[2 * item + 1] = bits(local < 19 ? float(9 - local / 2 + step) / 8 : -INFINITY);
            }
            words[0] = 0xfffffffeu;
            words[3] = 0x7fc12345u;
            words[5] = 0xff800000u;
            words[7] = 0x80000000u;
            words[9] = 0;
            words[2050] = words[10]; words[2051] = words[11];
        }
        RadStream stream = step ? second.stream : first.stream;
        run(native.ngram("sample_merge_topk"), reference.ngram("sample_merge_topk"),
            {&gathered, &params, &indices, &values},
            {integer("M", 3), integer("n_cand", 1024), integer("world_size", 2)}, stream, merge, counts);
        run(native.ngram("sample_pick"), reference.ngram("sample_pick"),
            {&indices, &values, &params, &tokens}, {integer("M", 8), integer("n_cand", 1024)}, stream, pick, counts, false, 2);
    }
    std::printf("PASS SAMPLE_PAIR_EDGES calls=%llu rejects=%llu exact_bytes=%llu nan_payloads=1 infinities=1 signed_zero=1 optional=2 duplicate_merge=1 tolerance=0\n",
        (unsigned long long)counts.calls, (unsigned long long)counts.rejects, (unsigned long long)counts.bytes);
}
