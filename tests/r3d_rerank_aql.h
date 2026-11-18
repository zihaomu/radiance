#include <array>
#include "build/startup.h"

namespace rerank_aql {
using namespace joined_head_test;
struct Counts { uint64_t calls = 0, bytes = 0, rejected = 0, references = 0, scored = 0, no_write = 0, head_witnesses = 0; };
void equal(const std::vector<uint8_t>& expected, const std::vector<uint8_t>& actual, const char* label) {
    if (expected != actual) {
        size_t byte = 0;
        while (byte < expected.size() && expected[byte] == actual[byte]) ++byte;
        std::printf("RERANK_DIFF %s byte=%zu expected=%u actual=%u\n", label, byte, unsigned(expected[byte]), unsigned(actual[byte]));
    }
    REQUIRE(expected == actual);
}
std::vector<RadParam> parameters(int offset) {
    return {RAD_INT("M", 8), RAD_INT("R", 8), RAD_INT("n_vocab", 124160),
        RAD_INT("n_embd", 2560), RAD_INT("vocab_off", offset), RAD_STR("dtype", "bf16")};
}
void oracle(Fixture& fixture, RadTensor* tensors, int rows, int offset, Counts& counts) {
    for (int row = 0; row < rows; ++row) {
        std::array<int32_t, 8> ids{};
        std::array<float, 8> scores{};
        unsigned used = 0;
        for (int candidate = 0; candidate < 8; ++candidate) {
            const int32_t global = static_cast<const int32_t*>(tensors[2].data)[row * tensors[2].stride[0] + candidate];
            const int64_t local = int64_t(global) - offset;
            const bool owned = global >= 0 && local >= 0 && local < 124160;
            ids[candidate] = owned ? global : -1;
            float score = 0.0f;
            if (owned) for (int position = 0; position < 2560; ++position) {
                const float weight = multiply(decode(fixture.codes[size_t(local) * 2560 + position], fixture.planes[0].dtype),
                    fixture.factor(int(local), position));
                const auto bits = static_cast<const uint16_t*>(tensors[0].data)[row * tensors[0].stride[0] + position];
                const float activation = rad_load_f32(&bits, RAD_BF16, 0);
                score = add(score, multiply(activation, weight));
            }
            scores[candidate] = owned ? score : -INFINITY;
            counts.scored += owned;
        }
        for (int output = 0; output < 8; ++output) {
            int best = -1;
            for (int candidate = 0; candidate < 8; ++candidate) {
                if ((used & (1u << candidate)) || ids[candidate] < 0) continue;
                if (best < 0 || scores[candidate] > scores[best] ||
                    (scores[candidate] == scores[best] && ids[candidate] < ids[best])) best = candidate;
            }
            const int32_t id = best < 0 ? -1 : ids[best];
            const float score = best < 0 ? -INFINITY : scores[best];
            if (best >= 0) used |= 1u << best;
            static_cast<int32_t*>(tensors[3].data)[row * tensors[3].stride[0] + output] = id;
            static_cast<float*>(tensors[4].data)[row * tensors[4].stride[0] + output] = score;
            if (tensors[5].data) {
                auto* pair = static_cast<uint8_t*>(tensors[5].data) + (row * tensors[5].stride[0] + output * 2) * 4;
                std::memcpy(pair, &id, 4); std::memcpy(pair + 4, &score, 4);
            }
        }
    }
}
void exercise(Api& api, Fixture& fixture, int offset, bool edge, Counts& counts, int band = 8) {
    const auto* native = api.native.ngram("logit_rerank");
    const auto* reference = api.reference.ngram("logit_rerank");
    REQUIRE_EQ(std::strcmp(native->name, "r3d_logit_rerank_joined_strict"), 0);
    REQUIRE_EQ(native->domain, RAD_DOMAIN_DEVICE);
    REQUIRE(!native->scratch);
    auto params = parameters(offset);
    RadLayout layout{};
    params[0] = RAD_INT("M", band);
    REQUIRE_EQ(native->layout(params.data(), int(params.size()), 1, &fixture.encoding, Fixture::selected,
        fixture.planes, 2, &layout), RAD_OK);
    CHECK_EQ(layout.dtype, fixture.stored.dtype);
    CHECK_EQ(layout.bytes, fixture.stored.bytes);
    CHECK_EQ(layout.rank, fixture.stored.rank);
    CHECK_EQ(std::strcmp(layout.tag, fixture.stored.tag), 0);
    CHECK(std::equal(layout.shape, layout.shape + layout.rank, fixture.stored.shape));
    std::vector<uint8_t> packed(layout.bytes);
    REQUIRE_EQ(native->relayout(params.data(), int(params.size()), 1, &fixture.encoding, Fixture::selected,
        fixture.planes, 2, packed.data(), packed.size()), RAD_OK);
    REQUIRE(packed == fixture.joined);
    CHECK(native->layout == api.layout);
    CHECK(native->relayout == api.relayout);
    RadOpdDesc descriptor{};
    params[0] = RAD_RANGE("M", 1, band);
    for (int operand = 0; operand < 6; ++operand) REQUIRE_EQ(native->opd_shape(params.data(), int(params.size()), operand, &descriptor), RAD_OK);
    params[0] = RAD_INT("M", band);
    const auto* head = api.native.ngram("logits_gemm");
    RadOpdDesc logical_head{};
    REQUIRE_EQ(head->opd_shape(params.data(), int(params.size()), 1, &logical_head), RAD_OK);
    REQUIRE_EQ(native->opd_shape(params.data(), int(params.size()), 1, &descriptor), RAD_OK);
    CHECK_EQ(descriptor.dtype, logical_head.dtype);
    CHECK(native->unrelayout == head->unrelayout);
    Queue first(0), second(0);
    REQUIRE(first.stream != second.stream);
    for (int memory : {int(RAD_MEM_DEVICE), -1}) {
        Storage weight(fixture.joined.size(), memory);
        std::memcpy(weight.expected.data() + 64, fixture.joined.data(), fixture.joined.size());
        weight.push(first.stream);
        REQUIRE_EQ(rad_stream_sync(first.stream), RAD_OK);
        const std::vector<int> live_rows = band == 32 ? std::vector<int>{1, 3, 8, 16, 31, 32} : std::vector<int>{1, 3, 8};
        for (int rows : live_rows) for (int logical : {0, 1}) for (int inplace : {0, 1}) {
            Buffer activation(RAD_BF16, {rows, 2560}, RAD_MEM_DEVICE, 2576);
            Buffer input(RAD_I32, {rows, 8}, RAD_MEM_DEVICE, 12);
            Buffer output(RAD_I32, {rows, 8}, RAD_MEM_DEVICE, 12);
            Buffer values_buffer(RAD_F32, {rows, 8}, RAD_MEM_DEVICE, 16);
            Buffer pairs_buffer(RAD_F32, {rows, 16}, RAD_MEM_DEVICE, 20);
            std::array<Buffer*, 5> buffers{&activation, &input, &output, &values_buffer, &pairs_buffer};
            for (int step = 0; step < 2; ++step) for (int pattern = 0; pattern < 4; ++pattern) {
                auto stream = step ? second.stream : first.stream;
                for (auto* buffer : buffers) std::fill(buffer->expected.begin(), buffer->expected.end(), 0x5a);
                for (int row = 0; row < rows; ++row) {
                    for (int position = 0; position < 2560; ++position)
                        activation.data<uint16_t>()[row * 2576 + position] = rad_f32_to_bf16(pattern == 3 ? 0.0f
                            : step == 0 || edge ? float(position == 0) : float(int((position * 71 + row * 37 + 19) % 521) - 260) / 257.0f);
                    const int32_t local[8]{7, 0, 3, 1, 3, 124159, 2, 4};
                    for (int candidate = 0; candidate < 8; ++candidate) {
                        int32_t id = offset + local[(candidate + row) % 8];
                        if (pattern == 1 && candidate < 4) {
                            const int32_t invalid[]{-1, offset - 1, offset + 124160, INT32_MAX};
                            id = invalid[candidate];
                        }
                        if (pattern == 2) id = candidate & 1 ? -1 : offset + 124160;
                        input.data<int32_t>()[row * 12 + candidate] = id;
                    }
                }
                for (auto* buffer : buffers) buffer->push(stream);
                REQUIRE_EQ(rad_stream_sync(stream), RAD_OK);
                RadTensor device[]{activation.device, matrix(weight.device(), r3d_head::dtype, 124160, 2560), input.device,
                    inplace ? input.device : output.device, values_buffer.device, pairs_buffer.device};
                if (!logical) { device[1].rank = 1; device[1].shape[0] = fixture.joined.size(); device[1].stride[0] = 1; }
                RadTensor host[]{activation.host, matrix(fixture.joined.data(), r3d_head::dtype, 124160, 2560), input.host,
                    inplace ? input.host : output.host, values_buffer.host, pairs_buffer.host};
                const bool present_pairs = pattern != 2;
                if (!present_pairs) { device[5].data = nullptr; host[5].data = nullptr; }
                if (present_pairs && step) {
                    device[5].rank = 3;
                    device[5].shape[1] = 8;
                    device[5].shape[2] = 2;
                    device[5].stride[1] = 2;
                    device[5].stride[2] = 1;
                }
                std::array<std::vector<uint8_t>, 5> copies;
                for (int index = 0; index < 5; ++index) copies[index] = buffers[index]->expected;
                oracle(fixture, host, rows, offset, counts);
                REQUIRE(values_buffer.expected != copies[3]);
                if (step == 0 || edge || pattern == 3) {
                    RadTensor ref[6]; std::copy(std::begin(host), std::end(host), ref);
                    ref[0].data = copies[0].data() + 64; ref[2].data = copies[1].data() + 64;
                    ref[3].data = copies[inplace ? 1 : 2].data() + 64;
                    ref[4].data = copies[3].data() + 64; ref[5].data = copies[4].data() + 64;
                    std::vector<uint8_t> reference_pairs(size_t(rows) * 64, 0x5a);
                    ref[5] = matrix(reference_pairs.data(), RAD_F32, rows, 16);
                    if (!present_pairs) ref[5].data = nullptr;
                    RadArgs args{}; args.t = ref; args.n_t = 6; args.p = params.data(); args.n_p = int(params.size());
                    REQUIRE_EQ(reference->launch(&args, nullptr), RAD_OK);
                    for (int row = 0; present_pairs && row < rows; ++row)
                        std::memcpy(copies[4].data() + 64 + row * 80, reference_pairs.data() + row * 64, 64);
                    for (int index = 0; index < 5; ++index) equal(buffers[index]->expected, copies[index], "libref");
                    ++counts.references;
                }
                RadArgs args{}; args.t = device; args.n_t = 6; args.p = params.data(); args.n_p = int(params.size());
                if (!present_pairs && step == 0) args.n_t = 5;
                REQUIRE_EQ(native->launch(&args, stream), RAD_OK);
                REQUIRE_EQ(rad_stream_sync(stream), RAD_OK);
                for (auto* buffer : buffers) {
                    equal(buffer->expected, buffer->read(stream), "device");
                    counts.bytes += buffer->expected.size();
                }
                ++counts.calls;
                if (!edge && rows == 3 && logical && inplace && step == 1 && pattern == 0) {
                    Fixture prefix(api, 8, 2560, RAD_I8, RAD_BF16, 1, 128);
                    std::copy_n(fixture.codes.begin(), prefix.codes.size(), prefix.codes.begin());
                    std::copy_n(fixture.scale.begin(), prefix.scale.size(), prefix.scale.begin());
                    prefix.pack(api);
                    Storage prefix_weight(prefix.joined.size(), memory);
                    std::memcpy(prefix_weight.expected.data() + 64, prefix.joined.data(), prefix.joined.size());
                    prefix_weight.push(stream);
                    Buffer full_scores(RAD_F32, {rows, 8}, RAD_MEM_DEVICE, 16);
                    full_scores.push(stream);
                    REQUIRE_EQ(rad_stream_sync(stream), RAD_OK);
                    RadTensor head_tensors[]{activation.device, matrix(prefix_weight.device(), r3d_head::dtype, 8, 2560), full_scores.device};
                    const RadParam head_params[]{RAD_INT("M", 8), RAD_INT("n_vocab", 8), RAD_INT("n_embd", 2560), RAD_STR("dtype", "bf16")};
                    RadArgs head_args{}; head_args.t = head_tensors; head_args.n_t = 3; head_args.p = head_params; head_args.n_p = 4;
                    REQUIRE_EQ(api.launch(&head_args, stream), RAD_OK);
                    const auto scored = full_scores.read(stream);
                    for (int row = 0; row < rows; ++row) for (int candidate = 0; candidate < 8; ++candidate) {
                        const int32_t id = static_cast<const int32_t*>(host[3].data)[row * host[3].stride[0] + candidate];
                        const int local = id - offset;
                        if (local < 0 || local >= 8) continue;
                        uint32_t full_bits = 0, rerank_bits = 0;
                        std::memcpy(&full_bits, scored.data() + 64 + (row * 16 + local) * 4, 4);
                        std::memcpy(&rerank_bits, static_cast<const float*>(host[4].data) + row * host[4].stride[0] + candidate, 4);
                        REQUIRE_EQ(full_bits, rerank_bits);
                        ++counts.head_witnesses;
                    }
                    for (int row = 0; row < rows; ++row) for (int column = 0; column < 8; ++column)
                        std::memcpy(full_scores.expected.data() + 64 + (row * 16 + column) * 4, scored.data() + 64 + (row * 16 + column) * 4, 4);
                    equal(full_scores.expected, scored, "strict-head-prefix-guards");
                    equal(prefix_weight.expected, prefix_weight.read(stream), "strict-head-prefix-weight");
                }
                if (step == 0 && pattern == 0 && !inplace) {
                    for (int invalid = 0; invalid < 19; ++invalid) {
                        RadTensor bad[6]; std::copy(std::begin(device), std::end(device), bad);
                        auto broken = params;
                        int status = RAD_E_SHAPE;
                        switch (invalid) {
                        case 0: broken[0] = RAD_INT("M", 0); break;
                        case 1: broken[0] = RAD_INT("M", 33); break;
                        case 2: broken[1] = RAD_INT("R", 7); break;
                        case 3: broken[2] = RAD_INT("n_vocab", 124161); break;
                        case 4: broken[3] = RAD_INT("n_embd", 2559); break;
                        case 5: broken[4] = RAD_INT("vocab_off", 1); break;
                        case 6: bad[0].dtype = RAD_F16; status = RAD_E_DTYPE; break;
                        case 7: bad[1].dtype = RAD_BF16; status = RAD_E_DTYPE; break;
                        case 8: bad[2].dtype = RAD_U32; status = RAD_E_DTYPE; break;
                        case 9: bad[4].dtype = RAD_BF16; status = RAD_E_DTYPE; break;
                        case 10: bad[0].stride[0] = 2559; status = RAD_E_STRIDE; break;
                        case 11: bad[3].shape[0]++; break;
                        case 12: bad[5].shape[1] = 15; break;
                        case 13: bad[3].data = static_cast<uint8_t*>(bad[2].data) + 4; status = RAD_E_INVAL; break;
                        case 14: bad[4].data = bad[0].data; status = RAD_E_INVAL; break;
                        case 15: bad[5].data = bad[4].data; status = RAD_E_INVAL; break;
                        case 18: broken[0] = RAD_INT("M", rows - 1); break;
                        case 16: bad[3].data = bad[1].data; status = RAD_E_INVAL; break;
                        case 17: bad[4].stride[0] = INT64_MAX; status = RAD_E_STRIDE; break;
                        }
                        args.t = bad; args.p = broken.data();
                        REQUIRE_EQ(native->launch(&args, stream), status);
                        ++counts.rejected;
                    }
                    args.t = device; args.p = params.data();
                    REQUIRE_EQ(native->launch(&args, reinterpret_cast<RadStream>(0x12345)), RAD_E_DEVICE);
                    ++counts.rejected;
                    for (auto* buffer : buffers) equal(buffer->expected, buffer->read(stream), "refusal-no-write");
                }
            }
        }
        equal(weight.expected, weight.read(second.stream), "immutable-private-head");
        Buffer activation(RAD_BF16, {1, 2560}, RAD_MEM_DEVICE), input(RAD_I32, {1, 8}, RAD_MEM_DEVICE),
            output(RAD_I32, {1, 8}, RAD_MEM_DEVICE), score(RAD_F32, {1, 8}, RAD_MEM_DEVICE);
        for (auto* buffer : {&activation, &input, &output, &score}) buffer->push(second.stream);
        REQUIRE_EQ(rad_stream_sync(second.stream), RAD_OK);
        for (int column = 0; column < 2560; ++column) activation.data<uint16_t>()[column] = rad_f32_to_bf16(1.0f);
        for (int candidate = 0; candidate < 8; ++candidate) input.data<int32_t>()[candidate] = offset + candidate;
        RadTensor tensors[]{activation.device, matrix(weight.device(), r3d_head::dtype, 124160, 2560),
            input.device, output.device, score.device};
        RadArgs args{}; args.t = tensors; args.n_t = 5; args.p = params.data(); args.n_p = int(params.size());
        activation.push(second.stream); input.push(second.stream);
        REQUIRE_EQ(native->launch(&args, second.stream), RAD_OK);
        REQUIRE(output.read(second.stream) != output.expected);
        REQUIRE(score.read(second.stream) != score.expected);
        for (int invalid = 0; invalid < 14; ++invalid) {
            std::memcpy(weight.expected.data() + 64, fixture.joined.data(), fixture.joined.size());
            auto* header = reinterpret_cast<RadPlanesHeader*>(weight.expected.data() + 64);
            auto* entry = reinterpret_cast<RadPlanesEntry*>(header + 1);
            switch (invalid) {
            case 0: header->magic = 0; break;
            case 1: header->n_planes = 3; break;
            case 2: header->transform[0] = 'x'; break;
            case 3: --header->bytes; break;
            case 4: entry[0].role[0] = 'x'; break;
            case 5: entry[0].dtype = RAD_BF16; break;
            case 6: entry[0].block[1] = 2; break;
            case 7: entry[1].block[0] = -1; break;
            case 8: entry[1].offset += 16; break;
            case 9: ++entry[1].row_bytes; break;
            case 10: --entry[0].rows; break;
            case 11: --entry[1].cols; break;
            case 12: entry[1].dtype = RAD_I8; break;
            case 13: entry[1].kind = RAD_PLANE_TABLE; break;
            }
            weight.push(second.stream);
            output.push(second.stream); score.push(second.stream);
            REQUIRE_EQ(rad_stream_sync(second.stream), RAD_OK);
            REQUIRE_EQ(native->launch(&args, second.stream), RAD_OK);
            for (auto* buffer : {&activation, &input, &output, &score}) equal(buffer->expected, buffer->read(second.stream), "device-header-no-write");
            equal(weight.expected, weight.read(second.stream), "corrupt-private-head-immutable");
            ++counts.no_write;
        }
        counts.bytes += weight.expected.size();
    }
}
}

TEST(r3d_rerank_actual_tp2_aql) {
    using namespace rerank_aql;
    if (!device_available()) SKIP("explicit gfx1100 card run required");
    Api api; Counts counts;
    Model model("/home/zihaomu/bigssd2/radiance_workspace/models/qwen3.8-next-flash-iqr-v4.rad");
    REQUIRE_EQ(model.header.file_bytes, 121678372864ull);
    bool found = false;
    for (const auto& entry : model.entries) {
        if (model.name(entry) != "output.weight") continue;
        found = true;
        REQUIRE_EQ(entry.rank, 2u); REQUIRE_EQ(entry.shape[0], 248320); REQUIRE_EQ(entry.shape[1], 2560);
        REQUIRE(entry.enc < model.encodings.size()); REQUIRE_EQ(entry.n_planes, 2u);
        const auto encoding = model.encodings[entry.enc];
        REQUIRE_EQ(encoding.n_planes, 2); REQUIRE_EQ(encoding.plane[0].dtype, uint32_t(RAD_I8));
        REQUIRE_EQ(encoding.plane[1].dtype, uint32_t(RAD_BF16));
        REQUIRE_EQ(encoding.plane[1].block[0], 1); REQUIRE_EQ(encoding.plane[1].block[1], 128);
        for (int rank = 0; rank < 2; ++rank) {
            Fixture fixture(api, 124160, 2560, RAD_I8, RAD_BF16, 1, 128);
            fixture.encoding = encoding;
            for (unsigned index = 0; index < 2; ++index) {
                REQUIRE(entry.plane_first + index < model.planes.size());
                const auto& location = model.planes[entry.plane_first + index];
                auto& destination = index ? fixture.scale : fixture.codes;
                REQUIRE_EQ(location.bytes, destination.size() * 2);
                REQUIRE(location.offset >= entry.offset && location.offset - entry.offset <= entry.bytes);
                REQUIRE(location.bytes <= entry.bytes - (location.offset - entry.offset));
                model.read(location.offset + uint64_t(rank) * destination.size(), destination.data(), destination.size());
            }
            fixture.pack(api);
            exercise(api, fixture, rank * 124160, false, counts);
            std::printf("RERANK_ACTUAL_SHARD_PASS rank=%d vocab_offset=%d calls=%llu\n", rank, rank * 124160, (unsigned long long)counts.calls);
            exercise(api, fixture, rank * 124160, false, counts, 32);
            std::printf("RERANK_CAPACITY rank=%d band=32 live=1,3,8,16,31,32 exhaustive_scores=1 pairs=2D,3D inplace=0,1 memory=DEVICE,VMEM\n", rank);
        }
    }
    REQUIRE(found);
    std::printf("RERANK_EXTRA_GUARDS device_header_no_write=%llu strict_head_score_witnesses=%llu\n",
        (unsigned long long)counts.no_write, (unsigned long long)counts.head_witnesses);
    std::printf("RERANK_ACTUAL_TP2_PASS calls=%llu libref=%llu candidate_scores=%llu rejects=%llu bytes=%llu scratch=0 queues=2\n",
        (unsigned long long)counts.calls, (unsigned long long)counts.references, (unsigned long long)counts.scored,
        (unsigned long long)counts.rejected, (unsigned long long)counts.bytes);
}

TEST(r3d_rerank_nonfinite_aql) {
    using namespace rerank_aql;
    if (!device_available()) SKIP("explicit gfx1100 card run required");
    Api api; Counts counts;
    Fixture fixture(api, 124160, 2560, RAD_I8, RAD_BF16, 1, 128);
    std::fill(fixture.codes.begin(), fixture.codes.end(), uint8_t(1));
    const uint16_t scales[]{0x7fc0, 0x7f80, 0xff80, 0x3f80, 0x3f80, 0, 0x8000, 0x4000};
    for (int row = 0; row < 8; ++row) std::memcpy(fixture.scale.data() + row * 40, scales + row, 2);
    fixture.pack(api);
    for (int offset : {0, 124160}) exercise(api, fixture, offset, true, counts);
    std::printf("RERANK_EXTRA_GUARDS device_header_no_write=%llu strict_head_score_witnesses=%llu\n",
        (unsigned long long)counts.no_write, (unsigned long long)counts.head_witnesses);
    std::printf("RERANK_NONFINITE_PASS calls=%llu libref=%llu rejects=%llu bytes=%llu duplicates_retained=1\n",
        (unsigned long long)counts.calls, (unsigned long long)counts.references,
        (unsigned long long)counts.rejected, (unsigned long long)counts.bytes);
}

TEST(r3d_rerank_normal_mtp_declare) {
    using namespace rad;
    if (!joined_head_test::device_available()) SKIP("explicit gfx1100 card run required");
    RadFile file;
    REQUIRE_EQ(file.open("/home/zihaomu/bigssd2/radiance_workspace/models/qwen3.8-next-flash-iqr-v4.rad"), RAD_OK);
    LoadedPlugins plugins;
    REQUIRE_EQ(rad_tools_load_plugins(home(), {"libr3d", "libr4d", "r3dh_ple_host"},
        file.str(file.header().arch_id), file.str(file.header().quant), &plugins), RAD_OK);
    RadModelMeta metadata{};
    metadata.arch_id = file.str(file.header().arch_id);
    metadata.name = file.str(file.header().model_name);
    metadata.quant = file.str(file.header().quant);
    auto integer = [&](const char* key) { int64_t value = 0; file.get_i(key, &value); return value; };
    metadata.n_layers = integer("n_layers"); metadata.n_embd = integer("n_embd");
    metadata.n_head = integer("n_head"); metadata.n_head_kv = integer("n_head_kv");
    metadata.head_dim = integer("head_dim"); metadata.n_ff = integer("n_ff");
    metadata.n_vocab = integer("n_vocab"); metadata.n_ctx_train = integer("n_ctx_train");
    metadata.n_expert = integer("n_expert"); metadata.n_expert_used = integer("n_expert_used");
    metadata.n_expert_shared = integer("n_expert_shared");
    double epsilon = 1e-6, theta = 10000;
    file.get_f("rms_eps", &epsilon); file.get_f("rope_theta", &theta);
    metadata.rms_eps = float(epsilon); metadata.rope_theta = float(theta); metadata.rope_scale = 1;
    std::vector<std::string> keys, values;
    std::vector<const char*> key_pointers, value_pointers;
    for (int64_t index = 0; index < file.meta_count(); ++index) {
        const auto& item = file.meta_at(index);
        if (item.type != RAD_P_STR) continue;
        keys.push_back(file.str(item.key)); values.push_back(file.str(item.v.s));
    }
    for (auto& key : keys) key_pointers.push_back(key.c_str());
    for (auto& value : values) value_pointers.push_back(value.c_str());
    metadata.n_kv = int(keys.size()); metadata.kv_key = key_pointers.data(); metadata.kv_val = value_pointers.data();
    Registry& registry = rad_tools_registry();
    registry.set_device_fact(RAD_FACT_P2P, 1);
    const Plugin* architecture = registry.plugin(plugins.arch_plugin);
    REQUIRE(architecture && architecture->arch_declare);
    for (int rank = 0; rank < 2; ++rank) {
        RadBuildCtx context{};
        context.rank = rank; context.world_size = 2; context.max_tok = 64;
        context.max_seqs = 1; context.max_ctx = 4096; context.max_spec = 1;
        context.max_enc_patches = 0; context.kv_dtype = RAD_I8; context.scope = "";
        Builder builder(registry, metadata, context);
        builder.set_instances(false);
        builder.set_sources(rad_container_sources(file));
        const int status = architecture->arch_declare(&builder, &metadata, &context);
        const auto& program = builder.program();
        bool rerank = false, shared_head = false;
        for (size_t index = 1; index < program.ops.size(); ++index) {
            const auto& operation = program.ops[index];
            if (operation.op == "moe_gemm_q" && operation.base.str().find("w4nl64a8i8h") != std::string::npos
                && operation.base.str().find("parts=2") != std::string::npos) {
                for (const auto& band : operation.bands) {
                    const auto* row = band.dom[RAD_DOMAIN_DEVICE].row; REQUIRE(row);
                    REQUIRE_EQ(std::strcmp(row->info->name, rank ? "model_moe_w4nl_gate_512_768_strict"
                        : "model_moe_w4nl_gate_768_512_strict"), 0);
                    std::printf("W4NL_GATE_SELECTED rank=%d op=%zu geometry=%s kernel=%s\n", rank, index,
                        operation.base.str().c_str(), row->info->name);
                }
            }
            if (operation.op == "logit_rerank") {
                REQUIRE_EQ(operation.weights.size(), 1u);
                REQUIRE_EQ(builder.op_read_set(rad_op(index)).size(), 2u);
                REQUIRE_EQ(builder.op_write_set(rad_op(index)).size(), 3u);
                for (const auto& band : operation.bands) {
                    const auto* row = band.dom[RAD_DOMAIN_DEVICE].row;
                    REQUIRE(row);
                    REQUIRE_EQ(std::strcmp(row->info->name, "r3d_logit_rerank_joined_strict"), 0);
                    rerank = true;
                }
                for (const auto& earlier : program.ops) if (earlier.op == "logits_gemm" &&
                    earlier.weights == operation.weights) shared_head = true;
            }
            if (index + 16 >= program.ops.size()) for (const auto& band : operation.bands) {
                const auto* row = band.dom[RAD_DOMAIN_DEVICE].row;
                std::printf("RERANK_DECLARE_TAIL rank=%d op=%zu name=%s geometry=%s kernel=%s miss=%s\n", rank, index,
                    operation.op.c_str(), operation.base.str().c_str(), row ? row->info->name : "NONE",
                    band.miss[RAD_DOMAIN_DEVICE].c_str());
            }
        }
        for (const auto& error : builder.errors()) std::printf("RERANK_DECLARE_ERROR rank=%d %s\n", rank, error.c_str());
        std::printf("RERANK_NORMAL_DECLARE rank=%d status=%d rerank=%d shared_head=%d audit=0 instances=0\n", rank, status, rerank, shared_head);
        REQUIRE(rerank); REQUIRE(shared_head);
        REQUIRE_EQ(status, RAD_OK);
        const int finalized = builder.finish();
        std::printf("RERANK_NORMAL_FINISH rank=%d status=%d errors=%zu\n", rank, finalized, builder.errors().size());
        REQUIRE_EQ(finalized, RAD_OK);
        REQUIRE(builder.errors().empty());
        if (!builder.errors().empty()) std::printf("RERANK_FINISH_FIRST_ERROR rank=%d %s\n", rank, builder.errors().front().c_str());
        for (const auto& error : builder.errors()) CHECK(error.find("output.weight") == std::string::npos);
    }
}
