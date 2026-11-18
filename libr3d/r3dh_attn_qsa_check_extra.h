struct ChainStep {
    Function device, reference;
    std::vector<RadParam> params;
    std::vector<RadTensor> tensors, remote;
};
void resident_chain_case(int rows) {
    const int pages = (rows + 3) / 4;
    Blob qg(rows * 6144, RAD_BF16), key(rows * 256, RAD_BF16), value(rows * 256, RAD_BF16);
    Blob query(rows * 3072, RAD_BF16), normalized(rows * 256, RAD_BF16), gate(rows * 3072, RAD_BF16);
    Blob dense(rows * 3072, RAD_BF16), paged(rows * 3072, RAD_BF16), kv(pages * 2064, RAD_I8);
    Blob positions(rows, RAD_I32), weights(256, RAD_F32), cs(64 * 64, RAD_F32);
    Blob slots(rows, RAD_I32), table(pages, RAD_I32), used(1, RAD_I32), cu(2, RAD_I32);
    for (int column = 0; column < 256; ++column) weights.data<float>()[column] = float(column % 7 - 3) / 32;
    for (int row = 0; row < rows; ++row) {
        positions.data<int>()[row] = row;
        slots.data<int>()[row] = (pages - 1 - row / 4) * 4 + row % 4;
    }
    for (int page = 0; page < pages; ++page) table.data<int>()[page] = pages - 1 - page;
    used.data<int>()[0] = rows; cu.data<int>()[0] = 0; cu.data<int>()[1] = rows;
    RadTensor absent{};
    std::vector<ChainStep> steps{
        {r3dh_attn_qsa_rope_table, ref_rope_table,
         {RAD_INT("rot", 64), RAD_F64("theta", 1e7), RAD_STR("dtype", "f32")},
         {tensor(positions, {rows}), tensor(cs, {64, 64})}, {}},
        {r3dh_attn_qsa_prepare, cpu_prepare,
         {RAD_INT("head_dim", 256), RAD_INT("n_head", 12), RAD_INT("n_head_kv", 1),
          RAD_INT("rot", 64), RAD_INT("cs_f32", 1), RAD_F64("eps", 1e-6), RAD_F64("wadd", 1)},
         {tensor(qg, {rows, 6144}), tensor(key, {rows, 256}), tensor(cs, {64, 64}), tensor(positions, {rows}),
          tensor(weights, {256}), tensor(weights, {256}), tensor(query, {rows, 3072}),
          tensor(normalized, {rows, 256}), tensor(gate, {rows, 3072})}, {}},
        {r3dh_attn_qsa_kv_store, ref_kv_store,
         {RAD_INT("head_dim", 256), RAD_INT("n_head_kv", 1), RAD_INT("block_size", 4)},
         {tensor(normalized, {rows, 256}), tensor(value, {rows, 256}), tensor(slots, {rows}),
          tensor(kv, {pages, 1, 4, 516})}, {}},
        {r3dh_attn_qsa_dense, ref_attn_dense,
         {RAD_INT("head_dim", 256), RAD_INT("gqa", 12), RAD_INT("causal", 1)},
         {tensor(query, {rows, 3072}), tensor(normalized, {rows, 256}), tensor(value, {rows, 256}),
          tensor(cu, {2}), tensor(dense, {rows, 3072})}, {}},
        {r3dh_attn_qsa_paged, ref_attn_paged,
         {RAD_INT("head_dim", 256), RAD_INT("gqa", 12), RAD_INT("causal", 1), RAD_INT("block_size", 4)},
         {tensor(query, {rows, 3072}), tensor(kv, {pages, 1, 4, 516}), tensor(table, {1, pages}),
          tensor(used, {1}), absent, absent, tensor(cu, {2}), tensor(paged, {rows, 3072})}, {}}
    };
    for (ChainStep& step : steps) {
        step.remote = step.tensors;
        for (RadTensor& operand : step.remote) if (operand.data) {
            bool found = false;
            for (Blob* blob : allocations) if (operand.data == blob->data<void>()) {
                operand.data = blob->device ? blob->device + 64 : nullptr; found = true; break;
            }
            require(found, "resident chain unmapped operand");
        }
    }
    hipGraph_t graph = nullptr;
    hipGraphExec_t executable = nullptr;
    for (int repeat = 0; repeat < 2; ++repeat) {
        fill(qg, 13 + repeat * 29); fill(key, 17 + repeat * 31); fill(value, 19 + repeat * 37);
        for (Blob* blob : {&query, &normalized, &gate, &dense, &paged, &kv, &cs})
            std::fill(blob->bytes.begin(), blob->bytes.end(), 0xa5);
        for (Blob* blob : allocations) blob->push();
        if (gpu) hipcheck(hipStreamSynchronize(stream));
        for (ChainStep& step : steps) {
            RadArgs args{}; args.t = step.tensors.data(); args.n_t = int(step.tensors.size());
            args.p = step.params.data(); args.n_p = int(step.params.size());
            require(step.reference(&args, nullptr) == RAD_OK, "resident chain CPU oracle");
        }
        if (gpu) {
            if (!repeat) {
                hipcheck(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal));
                for (ChainStep& step : steps) {
                    RadArgs args{}; args.t = step.remote.data(); args.n_t = int(step.remote.size());
                    args.p = step.params.data(); args.n_p = int(step.params.size());
                    require(step.device(&args, reinterpret_cast<RadStream>(stream)) == RAD_OK,
                        "resident chain capture");
                }
                hipcheck(hipStreamEndCapture(stream, &graph));
                hipcheck(hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
            }
            hipcheck(hipGraphLaunch(executable, stream));
            hipcheck(hipStreamSynchronize(stream));
            device_calls += int(steps.size()); ++graph_replays;
            cs.compare("resident_chain_table", true);
            query.compare("resident_chain_query", true);
            normalized.compare("resident_chain_key", true);
            gate.compare("resident_chain_gate", false);
            kv.compare("resident_chain_i8_cache", false);
            dense.compare("resident_chain_dense", true);
            paged.compare("resident_chain_paged", true);
        }
        ++cases;
    }
    if (executable) hipcheck(hipGraphExecDestroy(executable));
    if (graph) hipcheck(hipGraphDestroy(graph));
}
extern "C" int ref_qsa_work(const RadArgs*, RadStream);
extern "C" int ref_qsa_tail_store(const RadArgs*, RadStream);
void state_rejections() {
    int checks = 0;
    Blob key(16 * 128, RAD_BF16), positions(16, RAD_I32), cu(2, RAD_I32), state(3 * 7 * 128, RAD_BF16);
    Blob indices(1, RAD_I32), table(8, RAD_I32), stage(521 * 4 * 128, RAD_BF16), page(521, RAD_I32);
    Blob bpos(521, RAD_I32), complete(1, RAD_I32);
    for (int ring : {4, 5, 6, 7}) {
        std::vector<RadParam> params{RAD_INT("n", 128), RAD_INT("ratio", 4), RAD_INT("ring", ring), RAD_STR("dtype", "bf16")};
        std::vector<RadTensor> operands{tensor(key, {16, 128}), tensor(positions, {16}), tensor(cu, {2}),
            tensor(state, {3, 1, ring, 128}), tensor(indices, {1}), tensor(table, {1, 8}), RadTensor{},
            tensor(stage, {20, 128}), tensor(page, {5}), tensor(bpos, {5}), tensor(complete, {1})};
        RadArgs args{}; args.t = operands.data(); args.n_t = 11; args.p = params.data(); args.n_p = int(params.size());
        auto reject = [&](int expected) { require(r3dh_attn_qsa_work(&args, nullptr) == expected, "QSA work rejection"); ++checks; };
        for (int invalid : {0, 3, 8}) { params[2].ival = invalid; reject(RAD_E_UNSUPPORTED); }
        params[2].ival = ring;
        params[0].ival = 256; reject(RAD_E_UNSUPPORTED); params[0].ival = 128;
        params[1].ival = 3; reject(RAD_E_UNSUPPORTED); params[1].ival = 4;
        const RadParam dtype = params[3]; params[3] = RAD_STR("dtype", "f32"); reject(RAD_E_UNSUPPORTED); params[3] = dtype;
        for (int capacity : {0, 1, 4, 521}) {
            operands[7] = tensor(stage, {capacity * 4, 128}); operands[8] = tensor(page, {capacity}); operands[9] = tensor(bpos, {capacity});
            reject(RAD_E_SHAPE);
        }
        operands[7] = tensor(stage, {20, 128}); operands[8] = tensor(page, {5}); operands[9] = tensor(bpos, {5});
        const RadTensor valid = operands[3];
        operands[3].shape[2] = ring == 4 ? 7 : 4; reject(RAD_E_SHAPE); operands[3] = valid;
        operands[3].shape[1] = 2; reject(RAD_E_SHAPE); operands[3] = valid;
        operands[3].dtype = RAD_F32; reject(RAD_E_SHAPE); operands[3] = valid;
        operands[3].stride[0] = ring * 128 - 1; reject(RAD_E_SHAPE); operands[3] = valid;
        operands[0].shape[0] = 2049; reject(RAD_E_SHAPE); operands[0].shape[0] = 16;
        operands[2].shape[0] = 10; reject(RAD_E_SHAPE); operands[2].shape[0] = 2;
        for (int defect = 0; defect < 3; ++defect) {
            args.n_t = 5;
            operands[3] = valid;
            if (defect == 0) operands[3].shape[2] = ring == 4 ? 7 : 4;
            if (defect == 1) operands[3].dtype = RAD_F32;
            if (defect == 2) operands[3].shape[1] = 2;
            require(r3dh_attn_qsa_tail_store(&args, nullptr) == RAD_E_SHAPE, "QSA tail rejection"); ++checks;
        }
    }
    cases += checks; std::printf("QSA_STATE_REJECTIONS_PASS checks=%d\n", checks);
}
#if defined(R3DH_QSA_REGISTRY_CHECK)
extern "C" int r3d_model_extra_kernel_count();
extern "C" const RadKernelInfo* r3d_model_extra_kernel_at(int);
const RadKernelInfo* state_kernel(const char* operation) {
    for (int index = 0; index < r3d_model_extra_kernel_count(); ++index) {
        const RadKernelInfo* entry = r3d_model_extra_kernel_at(index);
        if (entry && !std::strcmp(entry->op, operation)) return entry;
    }
    throw std::runtime_error("QSA registry entry missing");
}
void state_registry() {
    int checks = 0;
    for (const char* operation : {"qsa_work", "qsa_tail_store"}) {
        const RadKernelInfo* entry = state_kernel(operation);
        require(entry->launch && entry->opd_shape && entry->domain == RAD_DOMAIN_DEVICE, "QSA source API hooks"); ++checks;
        for (int ring : {3, 4, 5, 6, 7, 8}) for (int sequences : {1, 2, 8}) for (int rows : {1, 3, 16, 32, 2048}) {
            const int minimum = (rows + 3) / 4 + sequences;
            for (int capacity : {minimum - 1, minimum, 519, 520, 521}) {
                const std::vector<RadParam> params{RAD_INT("M", rows), RAD_INT("n", 128), RAD_INT("ratio", 4),
                    RAD_INT("ring", ring), RAD_INT("seqs", sequences), RAD_INT("work", capacity), RAD_STR("dtype", "bf16")};
                const bool valid = (ring >= 4 && ring <= 7)
                    && (std::strcmp(operation, "qsa_work") || (capacity >= minimum && capacity <= 520));
                for (int operand = 0; operand < (!std::strcmp(operation, "qsa_work") ? 11 : 5); ++operand) {
                    RadOpdDesc descriptor{};
                    const int status = entry->opd_shape(params.data(), int(params.size()), operand, &descriptor);
                    require(status == (valid ? RAD_OK : RAD_E_UNSUPPORTED), "QSA source API shape status"); ++checks;
                    if (valid && operand == 3) {
                        require(descriptor.dtype == RAD_BF16 && descriptor.rank == 4 && descriptor.shape[0] == sequences
                            && descriptor.shape[1] == 1 && descriptor.shape[2] == ring && descriptor.shape[3] == 128,
                            "QSA source API tail ring shape"); ++checks;
                    }
                    if (valid && operand == 7) {
                        require(descriptor.shape[0] == capacity * 4 && descriptor.shape[1] == 128, "QSA source API work shape"); ++checks;
                    }
                }
            }
        }
    }
    std::printf("QSA_REGISTRY_API_PASS checks=%d rings=4,5,6,7 rejects=3,8 source=r3d_model_extra_kernel_count/at\n", checks);
}
#endif
void state_case() {
    Blob key(4 * 137, RAD_BF16), positions(4, RAD_I32), cu(2, RAD_I32), state(3 * 7 * 139, RAD_BF16);
    Blob indices(3, RAD_I32), table(8, RAD_I32), rope(12, RAD_I32), stage(5 * 4 * 133, RAD_BF16);
    Blob page(5, RAD_I32), bpos(5 * 4, RAD_I32), complete(1, RAD_I32);
    fill(key, 9); fill(state, 1);
    cu.data<int>()[0] = 0; cu.data<int>()[1] = 4; indices.data<int>()[0] = 2;
    for (int position = 0; position <= 8; ++position) for (int column = 0; column < 128; ++column)
        state.data<uint16_t>()[2 * 7 * 139 + (position % 7) * 139 + column]
            = rad_f32_to_bf16(float(position * 128 + column) / 1024);
    for (int block = 0; block < 8; ++block) table.data<int>()[block] = 13 - block;
    RadTensor tail = tensor(state, {3, 1, 7, 128});
    tail.stride[0] = 7 * 139; tail.stride[1] = 7 * 139; tail.stride[2] = 139;
    const std::vector<RadParam> params{RAD_INT("n", 128), RAD_INT("ratio", 4), RAD_INT("ring", 7), RAD_STR("dtype", "bf16")};
    for (int restart : {9, 10}) {
        fill(key, restart);
        for (int row = 0; row < 4; ++row) {
            positions.data<int>()[row] = restart + row;
            for (int component = 0; component < 3; ++component)
                rope.data<int>()[component * 4 + row] = restart + row + 100 + component * 11;
        }
        run("qsa_work_ring7_spec_restart", r3dh_attn_qsa_work, ref_qsa_work, params,
            {tensor(key, {4, 128}, 137), tensor(positions, {4}), tensor(cu, {2}), tail,
             tensor(indices, {1, 2}, 3), tensor(table, {1, 8}), tensor(rope, {3, 4}),
             tensor(stage, {20, 128}, 133), tensor(page, {5}), tensor(bpos, {5, 3}, 4), tensor(complete, {1})},
            {&stage, &page, &bpos, &complete});
        require(stage.data<uint16_t>()[0] == rad_f32_to_bf16(1.0f), "CPU rollback must retain position8");
        run("qsa_tail_ring7_spec_restart", r3dh_attn_qsa_tail_store, ref_qsa_tail_store, params,
            {tensor(key, {4, 128}, 137), tensor(positions, {4}), tensor(cu, {2}), tail, tensor(indices, {1, 2}, 3)}, {&state});
        require(state.data<uint16_t>()[2 * 7 * 139 + 139] == rad_f32_to_bf16(1.0f), "CPU ring7 preserves position8");
    }
    indices.data<int>()[0] = -1;
    RadTensor absent{};
    run("qsa_work_absent_state_text_positions", r3dh_attn_qsa_work, ref_qsa_work, params,
        {tensor(key, {4, 128}, 137), tensor(positions, {4}), tensor(cu, {2}), tail,
         tensor(indices, {1, 2}, 3), tensor(table, {1, 8}), absent, tensor(stage, {20, 128}, 133),
         tensor(page, {5}), tensor(bpos, {5, 1}, 4), tensor(complete, {1})}, {&stage, &page, &bpos, &complete});
}
void state_matrix() {
    const int first_case = cases;
    const size_t first_comparison = comparisons;
    state_rejections();
    Function work_launch = r3dh_attn_qsa_work, tail_launch = r3dh_attn_qsa_tail_store;
#if defined(R3DH_QSA_REGISTRY_CHECK)
    state_registry();
    work_launch = state_kernel("qsa_work")->launch;
    tail_launch = state_kernel("qsa_tail_store")->launch;
#endif
    for (int ring : {4, 5, 6, 7}) for (int sequences : {1, 2, 8}) for (int rows : {1, 3, 16, 32, 2048}) {
        const int slots = sequences + 2, pitch = ring * 139 + 17;
        const int minimum = (rows + 3) / 4 + sequences;
        for (int capacity : {minimum, 519, 520}) {
            Blob key(rows * 137, RAD_BF16), positions(rows, RAD_I32), cu(sequences + 1, RAD_I32);
            Blob state(slots * pitch, RAD_BF16), indices(sequences * 3, RAD_I32);
            Blob table(sequences * 1200, RAD_I32), rope(rows * 3, RAD_I32);
            Blob stage(capacity * 4 * 133, RAD_BF16), page(capacity, RAD_I32);
            Blob bpos(capacity * 4, RAD_I32), complete(sequences, RAD_I32);
            if (capacity < minimum) continue;
            RadTensor tail = tensor(state, {slots, 1, ring, 128});
            tail.stride[0] = pitch; tail.stride[1] = ring * 139; tail.stride[2] = 139;
            const std::vector<RadParam> params{RAD_INT("n", 128), RAD_INT("ratio", 4), RAD_INT("ring", ring),
                RAD_INT("seqs", sequences), RAD_INT("work", capacity), RAD_STR("dtype", "bf16")};
            for (int repeat = 0; repeat < 3; ++repeat) {
                fill(key, 13 + repeat * 29); fill(stage, 51 + repeat);
                if (repeat < 2) fill(state, 71 + repeat * 17);
                for (int sequence = 0; sequence <= sequences; ++sequence)
                    cu.data<int>()[sequence] = rows * sequence / sequences;
                for (int sequence = 0; sequence < sequences; ++sequence) {
                    const int slot = (sequence + (repeat ? 2 : 1)) % slots;
                    indices.data<int>()[sequence * 3] = repeat == 0 && sequence == sequences - 1 ? -1 : slot;
                    for (int block = 0; block < 1200; ++block)
                        table.data<int>()[sequence * 1200 + block] = (block * 7 + sequence * 13 + repeat) % 1200;
                    const int begin = cu.data<int>()[sequence], end = cu.data<int>()[sequence + 1];
                    const int start = 9 + (repeat ? 1 : 0) + sequence * 3
                        + (repeat == 2 ? end - begin - std::min(ring - 4, end - begin) : 0);
                    if (repeat < 2) for (int position = std::max(0, start - ring); position < start; ++position)
                        for (int column = 0; column < 128; ++column)
                            state.data<uint16_t>()[slot * pitch + (position % ring) * 139 + column]
                                = rad_f32_to_bf16(float((position * 41 + column * 7 + repeat * 19) % 503 - 251) / 128);
                    for (int row = begin; row < end; ++row) {
                        positions.data<int>()[row] = start + row - begin;
                        for (int component = 0; component < 3; ++component)
                            rope.data<int>()[component * rows + row] = start + row - begin + 100 + component * 11;
                    }
                }
                const bool multidimensional = repeat != 1;
                RadTensor absent{};
                char label[160];
                std::snprintf(label, sizeof label, "qsa_work_ring%d_seq%d_M%d_work%d_repeat%d", ring, sequences, rows, capacity, repeat);
                run(label, work_launch, ref_qsa_work, params,
                    {tensor(key, {rows, 128}, 137), tensor(positions, {rows}), tensor(cu, {sequences + 1}), tail,
                     tensor(indices, {sequences, 2}, 3), tensor(table, {sequences, 1200}),
                     multidimensional ? tensor(rope, {3, rows}) : absent, tensor(stage, {capacity * 4, 128}, 133),
                     tensor(page, {capacity}), tensor(bpos, {capacity, multidimensional ? 3 : 1}, 4), tensor(complete, {sequences})},
                    {&key, &positions, &cu, &state, &indices, &table, &rope, &stage, &page, &bpos, &complete});
                std::snprintf(label, sizeof label, "qsa_tail_ring%d_seq%d_M%d_work%d_repeat%d", ring, sequences, rows, capacity, repeat);
                run(label, tail_launch, ref_qsa_tail_store, params,
                    {tensor(key, {rows, 128}, 137), tensor(positions, {rows}), tensor(cu, {sequences + 1}), tail,
                     tensor(indices, {sequences, 2}, 3)}, {&key, &positions, &cu, &state, &indices});
            }
        }
    }
    std::printf("QSA_STATE_MATRIX rings=4,5,6,7 seqs=1,2,8 M=1,3,16,32,2048 work=ceilM/4+seqs,519,520 repeats=3 cases=%d comparisons=%zu\n",
        cases - first_case, comparisons - first_comparison);
}
void decode_case(int context, int rows, int seed) {
    const int pages = (context + 3) / 4;
    Blob key(context * 256, RAD_BF16), value(context * 256, RAD_BF16), slots(context, RAD_I32);
    Blob kv(pages * 4 * 516, RAD_I8), query(rows * 3072, RAD_BF16), output(rows * 3072, RAD_BF16);
    Blob table(pages, RAD_I32), used(1, RAD_I32), cu(2, RAD_I32);
    fill(key, seed); fill(value, seed + 1); fill(query, seed + 2);
    for (int page = 0; page < pages; ++page) table.data<int>()[page] = pages - 1 - page;
    for (int token = 0; token < context; ++token) slots.data<int>()[token] = (pages - 1 - token / 4) * 4 + token % 4;
    used.data<int>()[0] = context; cu.data<int>()[0] = 0; cu.data<int>()[1] = rows;
    run("kv_store_history_i8", r3dh_attn_qsa_kv_store, ref_kv_store,
        {RAD_INT("head_dim", 256), RAD_INT("block_size", 4)},
        {tensor(key, {context, 256}), tensor(value, {context, 256}), tensor(slots, {context}), tensor(kv, {pages, 1, 4, 516})}, {&kv});
    for (int repeat = 0; repeat < 2; ++repeat) {
        fill(query, seed + repeat * 13);
        RadTensor absent{};
        run("attn_paged_decode_changed_q_history", r3dh_attn_qsa_paged, ref_attn_paged,
            {RAD_INT("head_dim", 256), RAD_INT("gqa", 12), RAD_INT("causal", 1), RAD_INT("block_size", 4)},
            {tensor(query, {rows, 3072}), tensor(kv, {pages, 1, 4, 516}), tensor(table, {1, pages}), tensor(used, {1}),
             absent, absent, repeat ? absent : tensor(cu, {2}), tensor(output, {rows, 3072})}, {&output}, true);
    }
}
void rejection_cases() {
    RadTensor tensors[11]{};
    RadArgs args{}; args.t = tensors;
    std::vector<RadParam> params{RAD_INT("head_dim", 256), RAD_INT("gqa", 12), RAD_INT("block_size", 4),
        RAD_INT("causal", 1), RAD_INT("window", 0)};
    args.p = params.data(); args.n_p = int(params.size()); args.n_t = 8;
    for (size_t field = 0; field < params.size(); ++field) {
        const long long saved = params[field].ival;
        params[field].ival = field == 4 ? 1 : 0;
        require(r3dh_attn_qsa_paged(&args, nullptr) == RAD_E_UNSUPPORTED, "paged must reject unsupported geometry");
        params[field].ival = saved; ++cases;
    }
    require(r3dh_attn_qsa_paged(&args, nullptr) == RAD_E_SHAPE, "paged must reject absent tensors"); ++cases;
    params = {RAD_INT("n", 128), RAD_INT("ratio", 4), RAD_INT("ring", 3)};
    args.p = params.data(); args.n_p = int(params.size()); args.n_t = 5;
    require(r3dh_attn_qsa_tail_store(&args, nullptr) == RAD_E_UNSUPPORTED, "ring below ratio must be rejected"); ++cases;
    params = {RAD_INT("rot", 64), RAD_STR("ladder", "head")};
    args.p = params.data(); args.n_p = int(params.size()); args.n_t = 2;
    require(r3dh_attn_qsa_rope_table(&args, nullptr) == RAD_E_UNSUPPORTED, "table must reject proportional ladder"); ++cases;
    params = {RAD_INT("head_dim", 128), RAD_INT("n_head", 12), RAD_INT("n_head_kv", 1), RAD_INT("rot", 64), RAD_INT("cs_f32", 1)};
    args.p = params.data(); args.n_p = int(params.size()); args.n_t = 9;
    require(r3dh_attn_qsa_prepare(&args, nullptr) == RAD_E_UNSUPPORTED, "prepare must reject unsupported heads"); ++cases;
    std::printf("PASS strict_host_rejections\n");
}
