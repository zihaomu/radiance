#include "place/planner.h"
#include "runtime/ctx.h"
#include <sys/mman.h>

TEST(r3d_ple_domain_normal_declaration) {
    using namespace rad;
    if (!joined_head_test::device_available()) SKIP("explicit gfx1100 card required");
    RadFile file;
    REQUIRE_EQ(file.open("/home/zihaomu/bigssd2/radiance_workspace/models/qwen3.8-next-flash-iqr-v4.rad"), RAD_OK);
    LoadedPlugins plugins;
    REQUIRE_EQ(rad_tools_load_plugins(home(), {"libr3d", "libr4d", "r3dh_ple_host"},
        file.str(file.header().arch_id), file.str(file.header().quant), &plugins), RAD_OK);
    RadModelMeta metadata{};
    metadata.arch_id = file.str(file.header().arch_id);
    metadata.name = file.str(file.header().model_name); metadata.quant = file.str(file.header().quant);
    auto value = [&](const char* key) { int64_t result = 0; file.get_i(key, &result); return result; };
    metadata.n_layers = value("n_layers"); metadata.n_embd = value("n_embd");
    metadata.n_head = value("n_head"); metadata.n_head_kv = value("n_head_kv");
    metadata.head_dim = value("head_dim"); metadata.n_ff = value("n_ff");
    metadata.n_vocab = value("n_vocab"); metadata.n_ctx_train = value("n_ctx_train");
    metadata.n_expert = value("n_expert"); metadata.n_expert_used = value("n_expert_used");
    metadata.n_expert_shared = value("n_expert_shared");
    double epsilon = 1e-6, theta = 10000;
    file.get_f("rms_eps", &epsilon); file.get_f("rope_theta", &theta);
    metadata.rms_eps = epsilon; metadata.rope_theta = theta; metadata.rope_scale = 1;
    std::vector<std::string> keys, values;
    std::vector<const char*> key_ptrs, value_ptrs;
    for (int64_t index = 0; index < file.meta_count(); ++index) {
        const auto& item = file.meta_at(index);
        if (item.type == RAD_P_STR) { keys.push_back(file.str(item.key)); values.push_back(file.str(item.v.s)); }
    }
    for (auto& key : keys) key_ptrs.push_back(key.c_str());
    for (auto& text : values) value_ptrs.push_back(text.c_str());
    metadata.n_kv = keys.size(); metadata.kv_key = key_ptrs.data(); metadata.kv_val = value_ptrs.data();
    auto& registry = rad_tools_registry();
    registry.set_device_fact(RAD_FACT_P2P, 1);
    const auto* architecture = registry.plugin(plugins.arch_plugin);
    REQUIRE(architecture && architecture->arch_declare);
    for (const char* operation : {"ngram_ids", "ngram_hint", "embed_lookup_q"}) {
        int host = 0, device = 0;
        for (const auto& row : registry.rows()) if (!std::strcmp(row.info->op, operation)) {
            REQUIRE(!row.reference);
            host += row.info->domain == RAD_DOMAIN_HOST;
            device += row.info->domain == RAD_DOMAIN_DEVICE;
        }
        if (std::strcmp(operation, "embed_lookup_q")) { REQUIRE_EQ(host, 0); REQUIRE_EQ(device, 1); }
        else { REQUIRE_EQ(host, 1); }
    const auto* hint_schema = registry.schema("ngram_hint");
    const auto* gather_schema = registry.schema("embed_lookup_q");
    REQUIRE(registry.schema("ngram_ids") && hint_schema && gather_schema);
    REQUIRE_EQ(hint_schema->n_params, 5); REQUIRE_EQ(hint_schema->n_operands, 9);
    REQUIRE_EQ(gather_schema->n_operands, 5);
    for (const auto& plugin : registry.plugins()) if (plugin.name == "r3dh_ple_host") {
        REQUIRE_EQ(plugin.n_kernels, 1); REQUIRE_EQ(plugin.n_schemas, 1);
    }
    Geometry hint_geometry;
    hint_geometry.set_i("heads", 16); hint_geometry.set_i("ngram", 3); hint_geometry.set_i("eos", 248044);
    for (int rows : {1, 8}) for (int history : {0, 1}) {
        hint_geometry.set_i("M", rows); hint_geometry.set_i("hist", history);
        const auto* row = registry.select("ngram_hint", hint_geometry, RAD_DOMAIN_DEVICE);
        REQUIRE(row && row->info->opd_shape); REQUIRE(!registry.select("ngram_hint", hint_geometry, RAD_DOMAIN_HOST));
        RadParam params[]{RAD_INT("M", rows), RAD_INT("heads", 16), RAD_INT("ngram", 3), RAD_INT("eos", 248044), RAD_INT("hist", history)};
        RadOpdDesc output{};
        REQUIRE_EQ(row->info->opd_shape(params, 5, 8, &output), RAD_OK);
        REQUIRE_EQ(output.dtype, RAD_I32); REQUIRE_EQ(output.rank, 2u);
        REQUIRE_EQ(output.shape[0], rows); REQUIRE_EQ(output.shape[1], 16);
    }
    const char* invalid_keys[]{"M", "M", "heads", "ngram", "hist"};
    const int64_t invalid_values[]{0, 9, 8, 4, 2};
    for (int index = 0; index < 5; ++index) {
        Geometry invalid = hint_geometry;
        invalid.set_i(invalid_keys[index], invalid_values[index]);
        REQUIRE(!registry.select("ngram_hint", invalid, RAD_DOMAIN_DEVICE));
    }
    std::puts("PLE_DOMAIN_CONTRACT_PASS canonical_schemas=consistent public_HOST=1 hint_positive=4 hint_out_of_scope=5");
        std::printf("PLE_DOMAIN_ROWS op=%s HOST=%d DEVICE=%d\n", operation, host, device);
    }
    for (int mtp : {0, 1}) for (int sequences : {1, 8}) for (int rank : {0, 1}) {
        if (mtp && sequences != 1) continue;
        RadBuildCtx context{};
        context.rank = rank; context.world_size = 2; context.max_tok = 64;
        context.max_seqs = sequences; context.max_ctx = 4096; context.max_spec = mtp;
        context.kv_dtype = RAD_I8; context.scope = "";
        Builder builder(registry, metadata, context);
        builder.set_instances(false); builder.set_sources(rad_container_sources(file));
        REQUIRE_EQ(architecture->arch_declare(&builder, &metadata, &context), RAD_OK);
        REQUIRE_EQ(builder.finish(), RAD_OK); REQUIRE(builder.errors().empty());
        const auto& program = builder.program();
        REQUIRE_EQ(rad_gate_reference_kernels(program, false, "PLE domain regression"), RAD_OK);
        uint64_t ids = 0, hints = 0, gathers = 0;
        std::vector<rad_weight> hashes;
        for (size_t index = 1; index < program.ops.size(); ++index) {
            const auto& operation = program.ops[index];
            if (operation.optional_retired) continue;
            const bool ngram = operation.op == "ngram_ids" || operation.op == "ngram_hint";
            const bool gather = operation.op == "embed_lookup_q";
            REQUIRE(!operation.bands.empty());
            for (const auto& band : operation.bands) {
                const auto* host = band.dom[RAD_DOMAIN_HOST].row;
                const auto* device = band.dom[RAD_DOMAIN_DEVICE].row;
                if (ngram) {
                    REQUIRE(!host); REQUIRE(device && !device->reference);
                    REQUIRE_EQ(device->info->domain, RAD_DOMAIN_DEVICE);
                    if (operation.op == "ngram_ids") ++ids; else ++hints;
                } else if (gather) { REQUIRE(host && !host->reference); ++gathers; }
                else { REQUIRE(device && !device->reference); }
            }
            if (ngram) for (auto weight : operation.weights) if (weight) hashes.push_back(weight);
        }
        Config config{};
        PlannerInput input{}; input.program = &program; input.config = &config;
        input.rank = rank; input.world_size = 2;
        Plan plan;
        REQUIRE_EQ(Planner(input).plan(&plan), RAD_OK);
        uint64_t checked = 0;
        for (const auto& unit : plan.units) for (auto weight : unit.weights)
            if (std::find(hashes.begin(), hashes.end(), weight) != hashes.end()) {
                REQUIRE_EQ(unit.site, Site::Device); REQUIRE_EQ(unit.tier, Tier::VRAM); ++checked;
            }
        REQUIRE(ids && gathers && checked); REQUIRE_EQ(bool(hints), bool(mtp));
        std::printf("PLE_DOMAIN_NORMAL mtp=%d seq=%d rank=%d ids=%llu hints=%llu gathers=%llu hash_weights_DEVICE=%llu missing=0 HOST_only=embed_lookup_q placement=verified\n",
            mtp, sequences, rank, (unsigned long long)ids, (unsigned long long)hints,
            (unsigned long long)gathers, (unsigned long long)checked);
    }
}

TEST(r3d_ple_domain_actual_aql_pipeline) {
    if (!joined_head_test::device_available()) SKIP("explicit gfx1100 card required");
    REQUIRE_EQ(rad_dev_set(0), RAD_OK);
    Library native("libr3d"), reference("libref"), host_plugin("r3dh_ple_host");
    const auto open = reinterpret_cast<int(*)()>(dlsym(host_plugin.handle, "rad_plugin_open"));
    const auto close = reinterpret_cast<void(*)()>(dlsym(host_plugin.handle, "rad_plugin_close"));
    REQUIRE(open && close); REQUIRE_EQ(open(), RAD_OK);
    struct Close { void (*function)(); ~Close() { function(); } } closing{close};
    const auto* ids_kernel = native.ngram("ngram_ids");
    const auto* hint_kernel = native.ngram("ngram_hint");
    const auto* ids_oracle = reference.ngram("ngram_ids");
    const auto hint_oracle = reinterpret_cast<int(*)(const RadArgs*, RadStream)>(
        dlsym(host_plugin.handle, "r3dh_ple_host_ngram_hint")); REQUIRE(hint_oracle);
    const auto* gather = host_plugin.ngram("embed_lookup_q");
    REQUIRE_EQ(ids_kernel->domain, RAD_DOMAIN_DEVICE); REQUIRE_EQ(hint_kernel->domain, RAD_DOMAIN_DEVICE);
    REQUIRE_EQ(gather->domain, RAD_DOMAIN_HOST);
    const char* path = "/home/zihaomu/bigssd2/radiance_workspace/models/qwen3.8-next-flash-iqr-v4.rad";
    Model model(path); rad::RadFile file; REQUIRE_EQ(file.open(path), RAD_OK);
    std::array<std::vector<int64_t>, 3> hash;
    const char* needles[]{"mult", "vocab_sizes", "offsets"};
    for (int which = 0; which < 3; ++which) {
        for (const auto& entry : model.entries) {
            const auto name = model.name(entry);
            if (name.find("ple") == std::string::npos || name.find(needles[which]) == std::string::npos) continue;
            const auto* actual = file.find(name.c_str()); REQUIRE(actual);
            const size_t count = which ? 16 : 3;
            REQUIRE_EQ(actual->shape[0], int64_t(count));
            const auto* data = reinterpret_cast<const int64_t*>(file.data(file.plane(*actual, 0)));
            hash[which].assign(data, data + count);
            std::printf("PLE_DOMAIN_HASH source=%s count=%zu\n", name.c_str(), count); break;
        }
        REQUIRE(!hash[which].empty());
    }
    const int64_t vocabulary = hash[2].back() + hash[1].back();
    const size_t table_bytes = size_t(vocabulary) * 160 * sizeof(uint16_t);
    void* table = mmap(nullptr, table_bytes, PROT_READ | PROT_WRITE,
        MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    REQUIRE(table != MAP_FAILED);
    struct Mapping { void* pointer; size_t bytes; ~Mapping() { munmap(pointer, bytes); } } mapping{table, table_bytes};
    Queue queue(0);
    const auto* backend = rad::active_backend();
    REQUIRE(backend->tape_begin && backend->tape_end && backend->tape_play);
    uint64_t fixtures = 0, launches = 0, replays = 0, gathered = 0;
    rad::Registry registry;
    REQUIRE_EQ(registry.load_plugin(home() + "/kernels/libr3d.so", 0), RAD_OK);
    REQUIRE_EQ(registry.load_plugin(home() + "/kernels/r3dh_ple_host.so", 0), RAD_OK);
    uint64_t ctx_issues = 0;
    for (int memory : {RAD_MEM_HOST_MAPPED, RAD_MEM_HOST_PINNED})
    for (int tokens : {1, 3, 8}) for (int sequences : {1, 2, 8}) {
        if (sequences > tokens) continue;
        for (bool indexed : {false, true}) {
            std::vector<std::unique_ptr<Buffer>> buffers;
            auto add = [&](int type, std::initializer_list<int64_t> shape, int kind, int pitch = 0) {
                buffers.push_back(std::make_unique<Buffer>(type, shape, kind, pitch));
            };
            add(RAD_I32, {tokens}, memory); add(RAD_I32, {sequences, 1, 3}, RAD_MEM_DEVICE);
            add(RAD_I32, {sequences + 1}, memory);
            add(RAD_I64, {3}, RAD_MEM_DEVICE); add(RAD_I64, {16}, RAD_MEM_DEVICE); add(RAD_I64, {16}, RAD_MEM_DEVICE);
            add(RAD_I32, {sequences}, memory); add(RAD_I32, {sequences}, memory);
            add(RAD_I32, {tokens}, memory); add(RAD_I32, {tokens, 16}, memory);
            add(RAD_I32, {sequences, 4}, RAD_MEM_DEVICE); add(RAD_I32, {sequences, 16}, memory);
            add(RAD_BF16, {tokens, 2560}, memory);
            for (int which = 0; which < 3; ++which)
                std::memcpy(buffers[which + 3]->host.data, hash[which].data(), hash[which].size() * 8);
            auto ints = [&](int index) { return buffers[index]->data<int32_t>(); };
            for (int index = 0; index < sequences * 3; ++index) ints(1)[index] = index * 7919;
            for (int index = 0; index < sequences * 4; ++index) ints(10)[index] = index * 3571;
            for (int sequence = 0; sequence <= sequences; ++sequence) ints(2)[sequence] = sequence * tokens / sequences;
            for (int sequence = 0; sequence < sequences; ++sequence) {
                ints(6)[sequence] = sequences - 1 - sequence; ints(7)[sequence] = 0;
            }
            std::vector<RadTensor> cpu, device;
            for (auto& buffer : buffers) { cpu.push_back(buffer->host); device.push_back(buffer->device); }
            for (auto& buffer : buffers) buffer->push(queue.stream);
            std::vector<RadParam> params{integer("M", 8), integer("heads", 16), integer("ngram", 3), integer("eos", 248044), integer("hist", 1)};
            RadArgs args{}; args.p = params.data(); args.n_p = params.size(); args.world_size = 1;
            RadModelMeta metadata{}; metadata.n_vocab = vocabulary;
            RadBuildCtx build_context{}; build_context.max_tok = 8; build_context.max_seqs = sequences;
            build_context.max_ctx = 64; build_context.world_size = 1; build_context.scope = "";
            rad::Builder builder(registry, metadata, build_context);
            std::array<rad_weight, 3> weights{};
            for (int which = 0; which < 3; ++which) {
                RadWeightDecl declaration{}; declaration.dtype = RAD_I64; declaration.rank = 1;
                declaration.shape[0] = which ? 16 : 3;
                weights[which] = builder.decl_weight(needles[which], &declaration); REQUIRE(weights[which]);
            }
            RadWeightDecl table_declaration{}; table_declaration.dtype = RAD_BF16; table_declaration.rank = 2;
            table_declaration.shape[0] = vocabulary; table_declaration.shape[1] = 160;
            const auto table_weight = builder.decl_weight("table", &table_declaration); REQUIRE(table_weight);
            RadParam ids_params[]{RAD_RANGE("M", 1, 8), RAD_INT("heads", 16), RAD_INT("ngram", 3), RAD_INT("eos", 248044)};
            const auto ids_op = builder.decl_op("ngram_ids", ids_params, 4, weights.data(), 3); REQUIRE(ids_op);
            std::array<rad_op, 2> hint_ops{};
            for (int history : {0, 1}) {
                RadParam hint_params[]{RAD_RANGE("M", 1, 8), RAD_INT("heads", 16), RAD_INT("ngram", 3), RAD_INT("eos", 248044), RAD_INT("hist", history)};
                hint_ops[history] = builder.decl_op("ngram_hint", hint_params, 5, weights.data(), 3); REQUIRE(hint_ops[history]);
            }
            RadParam gather_geometry[]{RAD_RANGE("M", 1, 128), RAD_INT("n_embd", 160), RAD_INT("n_vocab", vocabulary), RAD_STR("dtype", "bf16")};
            const auto gather_op = builder.decl_op("embed_lookup_q", gather_geometry, 4, &table_weight, 1); REQUIRE(gather_op);
            REQUIRE_EQ(builder.finish(), RAD_OK);
            auto& program = builder.program();
            for (int which = 0; which < 3; ++which) {
                program.weights[weights[which]].ptr = device[which + 3].data;
                program.weights[weights[which]].site = rad::Site::Device;
            }
            program.weights[table_weight].ptr = table; program.weights[table_weight].host_ptr = table;
            program.weights[table_weight].site = rad::Site::Host; program.weights[table_weight].tier = rad::Tier::Mapped;
            rad::Ctx context; rad::CtxDesc description{};
            description.program = &program; description.stream = queue.stream; description.registry = &registry;
            description.record_passes = false;
            REQUIRE_EQ(context.init(description), RAD_OK);
            REQUIRE_EQ(context.op_domain(ids_op), RAD_DOMAIN_DEVICE);
            REQUIRE_EQ(context.op_domain(hint_ops[0]), RAD_DOMAIN_DEVICE);
            REQUIRE_EQ(context.op_domain(hint_ops[1]), RAD_DOMAIN_DEVICE);
            REQUIRE_EQ(context.op_domain(gather_op), RAD_DOMAIN_HOST);
            for (int which = 0; which < 3; ++which)
                REQUIRE(context.weight_ptr(weights[which]) == device[which + 3].data);
            std::array<rad::RadTape, 2> tapes{};
            for (int step = 0; step < 4; ++step) {
                for (int index = 0; index < tokens; ++index) {
                    ints(0)[index] = (index + step) % 5 == 0 ? 248044 : (index * 7919 + step * 104729) % 248000;
                    ints(8)[index] = index / 2 + step * 3;
                }
                buffers[0]->push(queue.stream); buffers[8]->push(queue.stream);
                for (int sequence = 0; sequence < sequences; ++sequence) ints(7)[sequence] = step != 0;
                buffers[7]->push(queue.stream);
                std::vector<RadTensor> ids_cpu(cpu.begin(), cpu.begin() + 10);
                ids_cpu.push_back({}); ids_cpu[8] = {};
                if (!indexed) ids_cpu[6] = {};
                args.n_t = ids_cpu.size(); args.t = ids_cpu.data();
                REQUIRE_EQ(ids_oracle->launch(&args, nullptr), RAD_OK);
                RadOperand ids_operands[]{RAD_P_T2(device[0].data, RAD_I32, tokens, 1),
                    RAD_P_T2(device[1].data, RAD_I32, sequences, 3), RAD_P_T2(device[2].data, RAD_I32, sequences + 1, 1),
                    RAD_W(weights[0]), RAD_W(weights[1]), RAD_W(weights[2]),
                    indexed ? RAD_P_T2(device[6].data, RAD_I32, sequences, 1) : RAD_NONE,
                    RAD_P_T2(device[7].data, RAD_I32, sequences, 1), RAD_NONE,
                    RAD_P_T2(device[9].data, RAD_I32, tokens, 16), RAD_NONE};
                REQUIRE_EQ(context.issue(ids_op, ids_operands, 11, tokens), RAD_OK); ++ctx_issues; ++launches;
                REQUIRE_EQ(rad_stream_sync(queue.stream), RAD_OK);
                REQUIRE(buffers[1]->read(queue.stream) == buffers[1]->expected);
                REQUIRE(buffers[9]->read(queue.stream) == buffers[9]->expected);
                auto* table_values = static_cast<uint16_t*>(table);
                for (int row = 0; row < tokens; ++row) for (int head = 0; head < 16; ++head) {
                    const int32_t id = ints(9)[row * 16 + head]; REQUIRE(id >= 0 && id < vocabulary);
                    for (int column = 0; column < 160; ++column)
                        table_values[int64_t(id) * 160 + column] = uint16_t(0x3e00 + (id + column) % 256);
                }
                RadOperand gather_operands[]{RAD_P_T2(device[9].data, RAD_I32, tokens, 16), RAD_W(table_weight), RAD_NONE,
                    RAD_P_T2(device[12].data, RAD_BF16, tokens, 2560), RAD_NONE};
                REQUIRE_EQ(context.issue(gather_op, gather_operands, 5, tokens * 16), RAD_OK); ++ctx_issues;
                for (int row = 0; row < tokens; ++row) for (int head = 0; head < 16; ++head)
                    for (int column = 0; column < 160; ++column) {
                        const int32_t id = ints(9)[row * 16 + head];
                        buffers[12]->data<uint16_t>()[row * 2560 + head * 160 + column] = table_values[int64_t(id) * 160 + column];
                    }
                REQUIRE(buffers[12]->read(queue.stream) == buffers[12]->expected); ++gathered;
                std::vector<RadTensor> hint_cpu{cpu[0], cpu[8], cpu[2], cpu[3], cpu[4], cpu[5], indexed ? cpu[6] : RadTensor{}, cpu[10], cpu[11]};
                std::vector<RadTensor> hint_device{device[0], device[8], device[2], device[3], device[4], device[5], indexed ? device[6] : RadTensor{}, device[10], device[11]};
                const int history = step == 0 ? 1 : 0; params[4].ival = history;
                args.n_t = 9;
                RadOperand hint_operands[]{RAD_P_T2(device[0].data, RAD_I32, tokens, 1),
                    RAD_P_T2(device[8].data, RAD_I32, tokens, 1), RAD_P_T2(device[2].data, RAD_I32, sequences + 1, 1),
                    RAD_W(weights[0]), RAD_W(weights[1]), RAD_W(weights[2]),
                    indexed ? RAD_P_T2(device[6].data, RAD_I32, sequences, 1) : RAD_NONE,
                    RAD_P_T2(device[10].data, RAD_I32, sequences, 4), RAD_P_T2(device[11].data, RAD_I32, sequences, 16)};
                for (int repeat = 0; repeat < 4; ++repeat) {
                    args.t = hint_cpu.data(); REQUIRE_EQ(hint_oracle(&args, nullptr), RAD_OK);
                    if (!tapes[history]) {
                        REQUIRE_EQ(backend->tape_begin(), RAD_OK);
                        REQUIRE_EQ(context.issue(hint_ops[history], hint_operands, 9, sequences), RAD_OK); ++ctx_issues;
                        tapes[history] = backend->tape_end(1); REQUIRE(tapes[history]); ++launches;
                    } else { REQUIRE_EQ(backend->tape_play(tapes[history], 0, backend->tape_len(tapes[history])), RAD_OK); ++replays; }
                    REQUIRE(buffers[10]->read(queue.stream) == buffers[10]->expected);
                    REQUIRE(buffers[11]->read(queue.stream) == buffers[11]->expected);
                }
                args.t = hint_device.data();
                for (int operand : {0, 3, 7, 8}) {
                    void* saved = hint_device[operand].data;
                    hint_device[operand].data = hint_cpu[operand].data;
                    REQUIRE_EQ(hint_kernel->launch(&args, queue.stream), RAD_E_DEVICE);
                    hint_device[operand].data = saved;
                }
                REQUIRE(buffers[10]->read(queue.stream) == buffers[10]->expected);
                REQUIRE(buffers[11]->read(queue.stream) == buffers[11]->expected);
                for (int index : {0, 2, 3, 4, 5, 6, 7, 8}) REQUIRE(buffers[index]->read(queue.stream) == buffers[index]->expected);
            }
            for (auto tape : tapes) if (tape) backend->tape_destroy(tape);
            ++fixtures;
        }
    }
    std::printf("PLE_DOMAIN_AQL_PASS fixtures=%llu launches=%llu tape_replays=%llu gathers=%llu pointer_rejects=384 real_hash=1 hash_DEVICE=1 tokens_mapped=1 ids_HOST_mapped=1 tail_DEVICE=1 guards=1 oracle=libref_ids+standalone_cpu_hint tolerance=0\n",
        (unsigned long long)fixtures, (unsigned long long)launches, (unsigned long long)replays, (unsigned long long)gathered);
    std::printf("PLE_DOMAIN_CTX_PASS issues=%llu weights_DEVICE=3 HOST_gather=1 normal_MTP1_max_seqs=1\n", (unsigned long long)ctx_issues);
}
