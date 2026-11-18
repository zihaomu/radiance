#include "../core/rad_internal.h"
namespace {
struct NgramStats {
    uint64_t fixtures = 0, steps = 0, ref_bytes = 0, native = 0, registered = 0, graphs = 0, rejects = 0;
};
void ngram_independent(const RadArgs& args) {
    auto i32 = [&](int operand) { return static_cast<int32_t*>(args.t[operand].data); };
    auto i64 = [&](int operand) { return static_cast<const int64_t*>(args.t[operand].data); };
    const int64_t eos = rad_args_geti_or(&args, "eos", -1);
    const int64_t window = args.t[1].shape[2], sequences = args.t[2].shape[0] - 1;
    const int64_t ahead = args.t[10].data ? args.t[10].shape[0] : 0;
    for (int64_t sequence = 0; sequence < sequences; ++sequence) {
        const int64_t begin = i32(2)[sequence], end = i32(2)[sequence + 1];
        const int64_t slot = i32(6) ? i32(6)[sequence] : sequence;
        if (begin == end || slot < 0) continue;
        const int64_t cursor = i32(8) ? i32(8)[sequence] - 1 : 0;
        const bool warm = !i32(7) || i32(7)[sequence] != 0;
        int32_t* state = i32(1) + slot * window;
        std::vector<int64_t> timeline{warm ? state[cursor] : eos, warm ? state[cursor + 1] : eos};
        const int64_t reach = end - begin + (sequence + 1 == sequences ? ahead : 0);
        for (int64_t token = 0; token < reach; ++token) timeline.push_back(i32(0)[begin + token]);
        for (int64_t token = 0; token < reach; ++token) {
            int32_t* output = token < end - begin ? i32(9) + (begin + token) * args.t[9].stride[0] :
                i32(10) + (token - (end - begin)) * args.t[10].stride[0];
            for (int64_t head = 0; head < 16; ++head) {
                const int64_t terms = head < 8 ? 2 : 3;
                uint64_t mixed = 0;
                for (int64_t shift = 0; shift < terms; ++shift) {
                    bool crossed = false;
                    for (int64_t scan = 0; scan < shift; ++scan)
                        crossed = crossed || timeline[size_t(token + 2 - scan)] == eos;
                    const int64_t value = crossed ? eos : timeline[size_t(token + 2 - shift)];
                    mixed ^= uint64_t(value) * uint64_t(i64(3)[shift]);
                }
                const uint64_t modulus = uint64_t(i64(4)[head]);
                const bool negative = (mixed >> 63) != 0;
                const uint64_t magnitude = negative ? uint64_t(0) - mixed : mixed;
                const uint64_t remainder = magnitude % modulus;
                output[head] = int32_t(i64(5)[head] + int64_t(negative && remainder ? modulus - remainder : remainder));
            }
        }
        if (i32(8)) {
            state[0] = int32_t(timeline[1]);
            for (int64_t token = begin; token < end; ++token) state[1 + token - begin] = i32(0)[token];
        } else {
            state[0] = int32_t(timeline[size_t(end - begin)]);
            state[1] = int32_t(timeline[size_t(end - begin + 1)]);
        }
    }
}
void ngram_focused_case(int64_t count, int64_t eos, int mask, bool real, bool registered, bool single,
                        Blob& real_mult, Blob& real_sizes, Blob& real_offsets, NgramStats& stats) {
    const int64_t window = mask & 4 ? count + 8 : 2, pitch = 21, ahead_count = mask & 8 ? 3 : 0;
    const int64_t sequences = single ? 1 : 4;
    Blob tokens(count + ahead_count, RAD_I32), state(5 * window, RAD_I32), cu(5, RAD_I32);
    Blob multipliers(3, RAD_I64), sizes(16, RAD_I64), offsets(16, RAD_I64);
    Blob indices(4, RAD_I32), history(4, RAD_I32), accepted(4, RAD_I32);
    Blob ids(count * pitch, RAD_I32), ahead(3 * pitch, RAD_I32);
    std::vector<Blob*> blobs{&tokens, &state, &cu, &multipliers, &sizes, &offsets, &indices, &history, &accepted, &ids, &ahead};
    if (real) {
        std::memcpy(multipliers.data<void>(), real_mult.data<void>(), 24);
        std::memcpy(sizes.data<void>(), real_sizes.data<void>(), 128);
        std::memcpy(offsets.data<void>(), real_offsets.data<void>(), 128);
    } else {
        multipliers.data<int64_t>()[0] = 24000000000001ll;
        multipliers.data<int64_t>()[1] = -51000000000007ll;
        multipliers.data<int64_t>()[2] = 0x7ffffffffffffffdll;
        for (int head = 0; head < 16; ++head) {
            sizes.data<int64_t>()[head] = 101 + 2 * head;
            offsets.data<int64_t>()[head] = 200 * head;
        }
    }
    cu.data<int32_t>()[0] = 0; cu.data<int32_t>()[1] = int32_t(count / 3);
    cu.data<int32_t>()[2] = int32_t(count / 3);
    cu.data<int32_t>()[3] = int32_t(std::min(count - 1, count / 3 + count / 2));
    cu.data<int32_t>()[4] = int32_t(count);
    const int32_t slots[]{mask & 4 ? -1 : 3, 4, 0, 2};
    std::memcpy(indices.data<void>(), slots, sizeof(slots));
    if (single) { cu.data<int32_t>()[1] = int32_t(count); indices.data<int32_t>()[0] = 2; }
    for (size_t index = 0; index < state.size(); ++index)
        state.data<int32_t>()[index] = index % 7 == 0 ? int32_t(eos) : int32_t((index * 7919 + 17) % 248000);
    std::vector<RadParam> params{RAD_INT("M", count), RAD_INT("heads", 16), RAD_INT("ngram", 3), RAD_INT("eos", eos)};
    params[0] = RAD_INT("M", registered ? 64 : 2048);
    std::vector<RadTensor> operands{tensor(tokens, {count + ahead_count}), tensor(state, {5, 1, window}), tensor(cu, {sequences + 1}),
        tensor(multipliers, {3}), tensor(sizes, {16}), tensor(offsets, {16}),
        mask & 1 ? tensor(indices, {sequences}) : RadTensor{}, mask & 2 ? tensor(history, {sequences}) : RadTensor{},
        mask & 4 ? tensor(accepted, {sequences}) : RadTensor{}, tensor(ids, {count, 16}, pitch),
        ahead_count ? tensor(ahead, {ahead_count, 16}, pitch) : RadTensor{}};
    std::vector<RadTensor> remote = operands;
    for (size_t operand = 0; operand < remote.size(); ++operand)
        if (remote[operand].data && gpu) remote[operand].data = blobs[operand]->device + 64;
    Function launch = registered ? registry_row("r3d_ngram_ids_i32")->launch : r3dh_mtp_ple_rad_ngram_ids;
    RadArgs args{}; args.p = params.data(); args.n_p = int(params.size()); args.n_t = int(operands.size());
    hipGraph_t graph = nullptr; hipGraphExec_t executable = nullptr;
    for (int step = 0; step < 4; ++step) {
        for (size_t index = 0; index < tokens.size(); ++index) {
            const uint32_t mixed = uint32_t(index * 2654435761u) ^ uint32_t((step + 1) * 2246822519u);
            tokens.data<int32_t>()[index] = index % 7 == size_t(step) ? int32_t(eos) : int32_t(mixed % 248000);
            if ((index + size_t(step)) % 11 == 0) tokens.data<int32_t>()[index] = 0;
        }
        for (int sequence = 0; sequence < 4; ++sequence) {
            history.data<int32_t>()[sequence] = step == 0 ? sequence % 2 : 1;
            accepted.data<int32_t>()[sequence] = 1 + (step + sequence) % 3;
        }
        std::fill(ids.bytes.begin() + 64, ids.bytes.end() - 64, uint8_t(0xa5));
        std::fill(ahead.bytes.begin() + 64, ahead.bytes.end() - 64, uint8_t(0xa5));
        for (Blob* blob : blobs) if (step == 0 || blob != &state) blob->push();
        if (gpu) hipcheck(hipStreamSynchronize(queue));
        std::vector<std::vector<uint8_t>> golden;
        for (Blob* blob : blobs) golden.push_back(blob->bytes);
        auto independent = operands;
        for (size_t operand = 0; operand < independent.size(); ++operand)
            if (independent[operand].data) independent[operand].data = golden[operand].data() + 64;
        args.t = independent.data(); ngram_independent(args);
        args.t = operands.data(); expect(ref_ngram_ids(&args, nullptr) == RAD_OK, "focused libref launch");
        for (size_t operand = 0; operand < blobs.size(); ++operand) {
            if (blobs[operand]->bytes != golden[operand]) {
                std::fprintf(stderr, "NGRAM_ORACLE_FAIL M=%lld eos=%lld mask=%d step=%d operand=%zu\n",
                    static_cast<long long>(count), static_cast<long long>(eos), mask, step, operand);
                throw std::runtime_error("independent CPU/libref exact bytes including canaries");
            }
            stats.ref_bytes += golden[operand].size();
        }
        if (gpu) {
            if (!step) {
                args.t = operands.data(); expect(launch(&args, reinterpret_cast<RadStream>(queue)) == RAD_E_DEVICE, "no host fallback");
                args.t = remote.data(); const auto saved = remote[0]; remote[0] = operands[0];
                expect(launch(&args, reinterpret_cast<RadStream>(queue)) == RAD_E_DEVICE, "no mixed host fallback");
                remote[0] = saved; stats.rejects += 2;
            }
            args.t = remote.data();
            if (!step) expect(launch(&args, reinterpret_cast<RadStream>(queue)) == RAD_OK, "focused direct device launch");
            else {
                if (step == 1) {
                    hipcheck(hipStreamBeginCapture(queue, hipStreamCaptureModeThreadLocal));
                    expect(launch(&args, reinterpret_cast<RadStream>(queue)) == RAD_OK, "registered/native graph capture");
                    hipcheck(hipStreamEndCapture(queue, &graph));
                    hipcheck(hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
                }
                hipcheck(hipGraphLaunch(executable, queue)); ++stats.graphs;
            }
            hipcheck(hipStreamSynchronize(queue));
            for (Blob* blob : blobs) blob->compare("focused ngram device full buffer");
            if (registered) ++stats.registered; else ++stats.native;
        }
        ++stats.steps;
    }
    if (executable) hipcheck(hipGraphExecDestroy(executable));
    if (graph) hipcheck(hipGraphDestroy(graph));
    ++stats.fixtures;
    std::printf("NGRAM_CASE_PASS M=%lld eos=%lld mask=%d weights=%s path=%s steps=4 graph_replays=%d state_rank=3 W=%lld pitch=21 layout=%s band=%lld exhaustive=1\n",
        static_cast<long long>(count), static_cast<long long>(eos), mask, real ? "model" : "overflow",
        gpu ? (registered ? "registered" : "native") : "cpu_libref", gpu ? 3 : 0,
        static_cast<long long>(window), single ? "single" : "mixed", static_cast<long long>(params[0].ival));
}
bool ngram_row_matches(const RadKernelInfo& row, int64_t count, int64_t heads = 16, int64_t order = 3) {
    rad::Geometry geometry;
    geometry.set_i("M", count); geometry.set_i("heads", heads); geometry.set_i("ngram", order);
    for (int index = 0; index < row.n_constraints; ++index)
        if (!rad::constraint_holds(row.constraints[index], geometry)) return false;
    return true;
}
void ngram_focused_suite(const char* path, bool device) {
    gpu = device;
    if (gpu) {
        int visible = 0; hipcheck(hipGetDeviceCount(&visible)); expect(visible == 1, "one permitted visible GPU");
        hipDeviceProp_t properties{}; hipcheck(hipGetDeviceProperties(&properties, 0));
        expect(!std::strncmp(properties.gcnArchName, "gfx1100", 7) && (properties.pciBusID == 0xc3 || properties.pciBusID == 0xe3), "GPU6 or GPU7 gfx1100");
        hipcheck(hipStreamCreateWithFlags(&queue, hipStreamNonBlocking));
        std::printf("NGRAM_DEVICE name=%s arch=%s pci=%x nondefault_stream=1 strict_fp=1\n", properties.name, properties.gcnArchName, properties.pciBusID);
    }
    Model model(path);
    int64_t model_eos = -1;
    for (const auto& entry : model.section<RadFileKV>(model.header.meta_off, model.header.meta_count)) {
        expect(entry.key < model.strings.size() && std::memchr(model.strings.data() + entry.key, 0, model.strings.size() - entry.key), "bounded model metadata key");
        const char* key = model.strings.data() + entry.key;
        if (!std::strcmp(key, "generation.eos_token_id")) continue;
        if (!std::strstr(key, "eos_token_id") && !std::strstr(key, "ngram_size") && !std::strstr(key, "heads_per_ngram")) continue;
        int64_t value = 0;
        if (entry.type == RAD_P_INT) value = entry.v.i;
        else {
            expect(entry.type == RAD_P_STR && entry.v.s < model.strings.size() &&
                std::memchr(model.strings.data() + entry.v.s, 0, model.strings.size() - entry.v.s), "bounded numeric model metadata string");
            const std::string text(model.strings.data() + entry.v.s);
            size_t consumed = 0; value = std::stoll(text, &consumed);
            expect(consumed == text.size(), "complete numeric model metadata string");
        }
        std::printf("NGRAM_MODEL_META %s=%lld stored_type=%u\n", key, static_cast<long long>(value), entry.type);
        if (!std::strcmp(key, "qwen4exp.eos_token_id")) model_eos = value;
        else if (model_eos < 0 && !std::strcmp(key, "eos_token_id")) model_eos = value;
    }
    expect(model_eos >= 0, "model architecture EOS metadata exists");
    RadVocabHeader vocab{}; model.read(model.header.vocab_off, &vocab, sizeof(vocab));
    std::printf("NGRAM_MODEL_TOKENIZER eos=%d eot=%d n_tokens=%u architecture_eos=%lld\n", vocab.eos, vocab.eot, vocab.n_tokens, static_cast<long long>(model_eos));
    std::vector<int64_t> eos_values{248068};
    if (model_eos != 248068) eos_values.push_back(model_eos);
    if (std::find(eos_values.begin(), eos_values.end(), 248044) == eos_values.end()) eos_values.push_back(248044);
    Blob mult(3, RAD_I64), sizes(16, RAD_I64), offsets(16, RAD_I64);
    const char* names[]{"blk.1.ple_mult", "blk.1.ple_vocab_sizes", "blk.1.ple_offsets"};
    Blob* tables[]{&mult, &sizes, &offsets};
    for (int table = 0; table < 3; ++table) {
        const auto* entry = model.find(names[table]); expect(entry, "actual model ngram hash weight exists");
        expect(rad_enc_is(&model.encodings[entry->enc], "plain"), "plain actual ngram hash table");
        model.plane(*entry, "codes", *tables[table]);
    }
    const auto* row = registry_row("r3d_ngram_ids_i32");
    expect(row->launch == r3dh_mtp_ple_rad_ngram_ids && row->domain == RAD_DOMAIN_DEVICE, "actual registered device candidate");
    constraint_is(*row, "M", RAD_C_GE, 1); constraint_is(*row, "M", RAD_C_LE, 64);
    constraint_is(*row, "heads", RAD_C_EQ, 16); constraint_is(*row, "ngram", RAD_C_EQ, 3);
    expect(!ngram_row_matches(*row, 0) && !ngram_row_matches(*row, 65) && !ngram_row_matches(*row, 2049), "engine matcher preserves registry M cap");
    expect(!ngram_row_matches(*row, 1, 8) && !ngram_row_matches(*row, 1, 32) && !ngram_row_matches(*row, 1, 16, 2) && !ngram_row_matches(*row, 1, 16, 4), "engine matcher rejects other geometry");
    std::printf("NGRAM_REGISTRY name=%s M_min=1 M_max=64 heads=16 ngram=3 actual_launch_pointer=1 capabilities=%llu engine_matcher=1\n", row->name, static_cast<unsigned long long>(row->capabilities));
    std::printf("NGRAM_EOS requested=248068 saved_audit=248044 model=%lld tested_values=%zu\n", static_cast<long long>(model_eos), eos_values.size());
    NgramStats stats;
    for (int64_t count : {1, 3, 32, 63, 64, 128, 512, 2048}) {
        expect(ngram_row_matches(*row, count) == (count <= 64), "actual engine registry match per requested batch");
        for (int64_t eos : eos_values) for (bool real : {false, true}) for (int mask = 0; mask < 16; ++mask) for (bool single : {false, true}) {
            ngram_focused_case(count, eos, mask, real, false, single, mult, sizes, offsets, stats);
            if (gpu && count <= 64) ngram_focused_case(count, eos, mask, real, true, single, mult, sizes, offsets, stats);
        }
        std::printf("NGRAM_BATCH_PASS M=%lld registered_supported=%d\n", static_cast<long long>(count), int(count <= 64));
    }
    if (gpu) { hipcheck(hipStreamDestroy(queue)); queue = nullptr; }
    std::printf("NGRAM_FOCUSED_PASS device=%d fixtures=%llu steps=%llu cpu_libref_bytes=%llu native_launches=%llu registered_launches=%llu graph_replays=%llu no_cpu_fallback_rejects=%llu gpu_compared_bytes=%llu\n",
        int(gpu), static_cast<unsigned long long>(stats.fixtures), static_cast<unsigned long long>(stats.steps),
        static_cast<unsigned long long>(stats.ref_bytes), static_cast<unsigned long long>(stats.native),
        static_cast<unsigned long long>(stats.registered), static_cast<unsigned long long>(stats.graphs),
        static_cast<unsigned long long>(stats.rejects), static_cast<unsigned long long>(comparisons));
}
}
