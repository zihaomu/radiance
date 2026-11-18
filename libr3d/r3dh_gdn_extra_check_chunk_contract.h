void chunk_rejections(std::vector<RadTensor> solve, std::vector<RadTensor> scan,
        std::vector<RadParam> params, hipStream_t stream) {
    const auto original_solve = solve, original_scan = scan;
    const auto original_params = params;
    auto solve_args = arguments(solve, params), scan_args = arguments(scan, params);
    auto check = [&](bool kkt, int expected, const char* label) {
        const int actual = kkt ? r3dh_gdn_extra_kkt_strict(&solve_args, reinterpret_cast<RadStream>(stream))
            : r3dh_gdn_extra_chunk_strict(&scan_args, reinterpret_cast<RadStream>(stream));
        if (actual != expected) std::fprintf(stderr, "chunk reject %s expected=%d actual=%d\n", label, expected, actual);
        expect(actual == expected, label); ++chunk_checks.rejects;
        expect(hipPeekAtLastError() == hipSuccess, "rejected chunk has no launch error");
    };
    hipGraph_t graph = nullptr;
    hip_ok(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal));
    for (bool kkt : {true, false}) {
        auto& tensors = kkt ? solve : scan;
        auto& args = kkt ? solve_args : scan_args;
        const auto& original = kkt ? original_solve : original_scan;
        const int inverse = kkt ? 4 : 3, cumulative = kkt ? 3 : 7;
        expect((kkt ? r3dh_gdn_extra_kkt_strict(nullptr, reinterpret_cast<RadStream>(stream))
            : r3dh_gdn_extra_chunk_strict(nullptr, reinterpret_cast<RadStream>(stream))) == RAD_E_INVAL, "null chunk arguments");
        ++chunk_checks.rejects;
        args.t = nullptr; check(kkt, RAD_E_INVAL, "null tensor descriptors"); args.t = tensors.data();
        args.p = nullptr; check(kkt, RAD_E_INVAL, "null parameters"); args.p = params.data();
        --args.n_t; check(kkt, RAD_E_INVAL, "missing operand"); args.n_t += 2;
        check(kkt, RAD_E_INVAL, "extra operand"); --args.n_t;
        for (int64_t rows : {int64_t{-1}, int64_t{0}, int64_t{2049}, int64_t{4294967297}}) {
            tensors[inverse].shape[0] = rows; check(kkt, RAD_E_SHAPE, "invalid total rows including empty batch");
        }
        tensors[inverse] = original[inverse];
        tensors[inverse].shape[1] = 4294967320LL; check(kkt, RAD_E_SHAPE, "oversized head count before narrowing");
        tensors[inverse] = original[inverse];
        tensors[inverse].shape[2] = 63; check(kkt, RAD_E_SHAPE, "inverse width"); tensors[inverse] = original[inverse];
        tensors[inverse].dtype = RAD_F32; check(kkt, RAD_E_SHAPE, "inverse dtype"); tensors[inverse] = original[inverse];
        for (int64_t count : {int64_t{0}, int64_t{1}, int64_t{66}}) {
            tensors[cumulative].shape[0] = count; check(kkt, RAD_E_SHAPE, "invalid sequence count no zero grid");
        }
        tensors[cumulative] = original[cumulative];
        tensors[cumulative].stride[0] = 2; check(kkt, RAD_E_SHAPE, "cumulative stride"); tensors[cumulative] = original[cumulative];
        tensors[cumulative].dtype = RAD_F32; check(kkt, RAD_E_SHAPE, "cumulative dtype"); tensors[cumulative] = original[cumulative];
        tensors[cumulative].data = nullptr; check(kkt, RAD_E_SHAPE, "cumulative missing data"); tensors[cumulative] = original[cumulative];
        tensors[0].shape[1] = 7; check(kkt, RAD_E_SHAPE, "key query grouped heads"); tensors[0] = original[0];
        tensors[1].stride[1] = 0; check(kkt, RAD_E_SHAPE, "operand inner stride"); tensors[1] = original[1];
        for (const char* key : {"head_k", "head_v", "chunk"}) {
            for (auto& param : params) if (!std::strcmp(param.key, key)) param.ival = 7;
            check(kkt, RAD_E_UNSUPPORTED, "unsupported head or chunk geometry"); params = original_params;
        }
        for (auto& param : params) if (!std::strcmp(param.key, "scale")) param.dval = INFINITY;
        check(kkt, RAD_E_INVAL, "nonfinite scale"); params = original_params;
    }
    scan[9].dtype = RAD_BF16; check(false, RAD_E_SHAPE, "final state dtype"); scan[9] = original_scan[9];
    scan[6].shape[0] += 1; check(false, RAD_E_SHAPE, "initial final pool mismatch"); scan[6] = original_scan[6];
    scan[10].dtype = RAD_F32; check(false, RAD_E_SHAPE, "index dtype"); scan[10] = original_scan[10];
    scan[10].shape[0] += 1; check(false, RAD_E_SHAPE, "index sequence mismatch"); scan[10] = original_scan[10];
    scan[10].stride[0] = 1; check(false, RAD_E_SHAPE, "overlapping index rows"); scan[10] = original_scan[10];
    scan[10].stride[1] = 2; check(false, RAD_E_SHAPE, "index column stride"); scan[10] = original_scan[10];
    scan[10].shape[1] = 0; check(false, RAD_E_SHAPE, "empty index row"); scan[10] = original_scan[10];
    hip_ok(hipStreamEndCapture(stream, &graph));
    if (chunk_registry) {
        for (bool kkt : {true, false}) {
            auto& tensors = kkt ? solve : scan;
            auto& args = kkt ? solve_args : scan_args;
            const int cumulative = kkt ? 3 : 7;
            const RadTensor saved = tensors[cumulative];
            for (int64_t count : {int64_t{0}, int64_t{1}, int64_t{10}, int64_t{66}}) {
                tensors[cumulative].shape[0] = count;
                expect(chunk_dispatch(kkt, &args, reinterpret_cast<RadStream>(stream)) == RAD_E_UNSUPPORTED, "registry sequence bounds");
                ++chunk_checks.rejects;
            }
            tensors[cumulative] = saved; tensors[cumulative].rank = 2;
            expect(chunk_dispatch(kkt, &args, reinterpret_cast<RadStream>(stream)) == RAD_E_UNSUPPORTED, "registry cumulative rank");
            ++chunk_checks.rejects; tensors[cumulative] = saved;
        }
    }
    size_t nodes = 999;
    hip_ok(hipGraphGetNodes(graph, nullptr, &nodes)); expect(nodes == 0, "all rejected calls launch zero nodes");
    hip_ok(hipGraphDestroy(graph)); hip_ok(hipStreamSynchronize(stream));
    std::printf("PASS chunk_rejections count=%llu graph_nodes=0 empty_batch_rejected no_invalid_grid\n",
        static_cast<unsigned long long>(chunk_checks.rejects));
}
