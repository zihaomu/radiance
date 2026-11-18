namespace {
void contract_checks() {
    int checks = 0;
    const std::vector<RadParam> params{RAD_INT("M", 32), RAD_INT("n", 2560), RAD_INT("hc", 4),
        RAD_INT("side", 64), RAD_INT("n_embd", 2560), RAD_INT("n_vocab", 19)};
    const char* names[]{"mtp_enter", "ple_gate", "grid_embed", "embed_lookup_q"};
    const int counts[]{7, 8, 3, 4};
    for (int operation = 0; operation < 4; ++operation) {
        RadOpdDesc description{};
        for (int operand = 0; operand < counts[operation]; ++operand) {
            expect(r3dh_mtp_ple_shape(names[operation], params.data(), int(params.size()), operand, &description) == RAD_OK, "shape success");
            expect(description.rank && description.shape[0] > 0, "shape nonempty"); ++checks;
        }
        expect(r3dh_mtp_ple_shape(names[operation], params.data(), int(params.size()), counts[operation], &description) == RAD_E_SHAPE, "shape rejects operand"); ++checks;
    }
    RadOpdDesc description{};
    expect(r3dh_mtp_ple_shape("unknown", params.data(), int(params.size()), 0, &description) == RAD_E_UNSUPPORTED, "shape rejects unknown"); ++checks;
    auto invalid = params; invalid[1] = RAD_INT("n", 2559);
    expect(r3dh_mtp_ple_shape("mtp_enter", invalid.data(), int(invalid.size()), 0, &description) == RAD_E_SHAPE, "shape rejects E4M3 partial group"); ++checks;
    RadArgs empty{};
    const Function functions[]{r3dh_mtp_ple_mtp_enter, r3dh_mtp_ple_ple_gate, r3dh_mtp_ple_grid_embed,
        r3dh_mtp_ple_embed_lookup_q, r3dh_mtp_ple_rad_ngram_ids, r3dh_mtp_ple_ngram_hint,
        r3dh_mtp_ple_logit_rerank, r3dh_mtp_ple_rowtopkmerge};
    for (auto function : functions) { expect(function(&empty, nullptr) != RAD_OK, "empty native request rejected"); ++checks; }
    expect(rad_f32_to_bf16(1.00390625f) == 0x3f80 && rad_f32_to_bf16(1.01171875f) == 0x3f82, "BF16 ties to even"); ++checks;
    for (unsigned code = 0; code < 256; ++code) {
        const uint8_t bits = uint8_t(code);
        const int magnitude = code & 127, exponent = magnitude / 8, fraction = magnitude % 8;
        const double expected = exponent ? std::ldexp(double(8 + fraction), exponent - 10) : double(fraction) / 512;
        const float actual = rad_load_f32(&bits, RAD_F8E4M3, 0);
        expect(magnitude == 127 ? std::isnan(actual) : double(actual) == (code & 128 ? -expected : expected), "independent E4M3 truth table"); ++checks;
    }
    alignas(16) uint8_t opaque[16]{};
    RadTensor operands[7]{};
    for (int index = 0; index < 7; ++index) {
        auto& item = operands[index]; item.data = opaque;
        item.dtype = index == 2 || index == 3 ? RAD_F32 : RAD_BF16;
        item.rank = index == 2 || index == 3 ? 1 : 2;
        item.shape[0] = item.rank == 1 ? (index == 2 ? 10240 : 2560) : 1;
        item.stride[0] = item.rank == 1 ? 1 : (index == 0 || index == 6 ? 10240 : 2560);
        if (item.rank == 2) { item.shape[1] = item.stride[0]; item.stride[1] = 1; }
        if (index == 4 || index == 5) { item.dtype = r3dh_mtp_fc_e4m3_dtype; item.shape[0] = 2560; }
    }
    auto private_params = std::vector<RadParam>{RAD_INT("n", 2560), RAD_INT("hc", 4), RAD_STR("fc", "e4m3")};
    RadArgs request{}; request.t = operands; request.n_t = 7; request.p = private_params.data();
    request.n_p = int(private_params.size()); request.scratch = opaque; request.scratch_bytes = 2293760;
    const RadStream invalid_stream = reinterpret_cast<RadStream>(uintptr_t(0x12345));
    for (int index : {4, 5}) for (int malformed = 0; malformed < 10; ++malformed) {
        const auto saved = operands[index]; auto& item = operands[index];
        if (malformed == 0) item.dtype = RAD_F8E4M3;
        if (malformed == 1) item.dtype = RAD_F32;
        if (malformed == 2) item.dtype = RAD_BF16;
        if (malformed == 3) item.rank = 1;
        if (malformed == 4) item.rank = 3;
        if (malformed == 5) --item.shape[0];
        if (malformed == 6) --item.shape[1];
        if (malformed == 7) ++item.stride[0];
        if (malformed == 8) item.stride[1] = 2;
        if (malformed == 9) item.data = opaque + 1;
        expect(r3dh_mtp_ple_mtp_enter(&request, invalid_stream) == RAD_E_DTYPE, "private FC metadata rejects before device access");
        operands[index] = saved; ++checks;
    }
    private_params[2] = RAD_STR("missing_fc", "e4m3");
    expect(r3dh_mtp_ple_mtp_enter(&request, invalid_stream) == RAD_E_UNSUPPORTED, "missing FC contract rejected"); ++checks;
    private_params[2] = RAD_STR("fc", "unknown");
    expect(r3dh_mtp_ple_mtp_enter(&request, invalid_stream) == RAD_E_UNSUPPORTED, "unknown FC contract rejected"); ++checks;
    expect(std::all_of(std::begin(opaque), std::end(opaque), [](uint8_t value) { return value == 0; }), "host private negative guard");
    std::printf("CONTRACT_PASS host_only=1 checks=%d device_residency_tested=%d\n", checks, int(gpu));
}
void analytic_checks() {
    constexpr int64_t width = 128, streams = 4, wide = width * streams, pitch = 144;
    Blob hidden(wide, RAD_BF16), embedding(width, RAD_BF16), wh(wide, RAD_F32), we(width, RAD_F32),
        packed(width * pitch, RAD_F8E4M3), identity(width * width, RAD_F32), output(wide, RAD_BF16);
    std::fill_n(hidden.data<uint16_t>(), wide, uint16_t(0x4000));
    std::fill_n(embedding.data<uint16_t>(), width, uint16_t(0x4080));
    std::fill_n(wh.data<float>(), wide, 1.0f); std::fill_n(we.data<float>(), width, 1.0f);
    std::fill_n(identity.data<float>(), width * width, 0.0f);
    std::fill_n(packed.data<uint8_t>(), width * pitch, uint8_t(0));
    for (int64_t channel = 0; channel < width; ++channel) {
        identity.data<float>()[channel * width + channel] = 1.0f;
        packed.data<uint8_t>()[channel * pitch + channel] = 0x38;
        const float scale = 1.0f; std::memcpy(packed.data<uint8_t>() + channel * pitch + width, &scale, sizeof(scale));
    }
    std::vector<RadTensor> operands{tensor(hidden, {1, wide}), tensor(embedding, {1, width}), tensor(wh, {wide}), tensor(we, {width}),
        tensor(packed, {width, pitch}), tensor(packed, {width, pitch}), tensor(output, {1, wide})};
    auto oracle = operands; oracle[4] = tensor(identity, {width, width}); oracle[5] = oracle[4];
    run("analytic_mtp_identity_expected_2", r3dh_mtp_ple_mtp_enter, ref_mtp_enter,
        {RAD_INT("M", 1), RAD_INT("n", width), RAD_INT("hc", streams), RAD_INT("ngroup", streams), RAD_STR("fc", "e4m3"), RAD_F64("eps", 0.0)}, operands, oracle);
    for (int64_t channel = 0; channel < wide; ++channel) expect(output.data<uint16_t>()[channel] == 0x4000, "MTP analytic value");
    Blob key(wide, RAD_BF16), query(wide, RAD_BF16), gv(wide, RAD_BF16), gvn(wide, RAD_BF16);
    std::fill_n(key.data<uint16_t>(), wide, uint16_t(0)); std::fill_n(query.data<uint16_t>(), wide, uint16_t(0));
    std::fill_n(embedding.data<uint16_t>(), width, uint16_t(0x4000));
    run("analytic_ple_zero_score_expected_half_value", r3dh_mtp_ple_ple_gate, ref_ple_gate,
        {RAD_INT("M", 1), RAD_INT("n", width), RAD_INT("hc", streams), RAD_F64("eps", 1e-6)},
        {tensor(key, {1, wide}), tensor(query, {1, wide}), tensor(embedding, {1, width}), tensor(wh, {wide}), tensor(wh, {wide}),
         tensor(wh, {wide}), tensor(gv, {1, wide}), tensor(gvn, {1, wide})});
    for (int64_t channel = 0; channel < wide; ++channel) expect(gv.data<uint16_t>()[channel] == 0x3f80 && gvn.data<uint16_t>()[channel] == 0x3f80, "PLE analytic value");
    Blob table(width, RAD_BF16), coordinates(4, RAD_I32);
    std::fill_n(table.data<uint16_t>(), width, uint16_t(0x3f00));
    std::fill_n(output.data<uint16_t>(), wide, uint16_t(0x3f80)); std::fill_n(coordinates.data<int32_t>(), 4, 0);
    run("analytic_grid_single_cell_expected_1p5", r3dh_mtp_ple_grid_embed, ref_grid_embed,
        {RAD_INT("M", 1), RAD_INT("n", width), RAD_INT("side", 1)},
        {tensor(output, {1, width}), tensor(table, {1, width}), tensor(coordinates, {4, 1})});
    for (int64_t channel = 0; channel < width; ++channel) expect(output.data<uint16_t>()[channel] == 0x3fc0, "grid analytic value");
    Blob tokens(3, RAD_I32), lookup(width, RAD_F8E4M3), gathered(3 * width, RAD_BF16);
    tokens.data<int32_t>()[0] = 0; tokens.data<int32_t>()[1] = 1; tokens.data<int32_t>()[2] = 2;
    std::fill_n(lookup.data<uint8_t>(), width, uint8_t(0x38));
    run("analytic_lookup_oov_zero_expected_0_1_0", r3dh_mtp_ple_embed_lookup_q, ref_embed_lookup_q,
        {RAD_INT("M", 3), RAD_INT("n_embd", width), RAD_INT("n_vocab", 1), RAD_INT("vocab_offset", 1)},
        {tensor(tokens, {3}), tensor(lookup, {1, width}), RadTensor{}, tensor(gathered, {3, width})});
    for (int64_t index = 0; index < 3 * width; ++index) expect(gathered.data<uint16_t>()[index] == (index / width == 1 ? 0x3f80 : 0), "lookup analytic value");
    std::printf("ANALYTIC_PASS independent_expected_values=1\n");
}
}
