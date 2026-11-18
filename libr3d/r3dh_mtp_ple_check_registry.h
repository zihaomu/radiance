extern "C" int r3d_mtp_kernel_count();
extern "C" const RadKernelInfo* r3d_mtp_kernel_at(int);
const RadKernelInfo* registry_row(const char* name) {
    for (int index = 0; index < r3d_mtp_kernel_count(); ++index) {
        const auto* row = r3d_mtp_kernel_at(index);
        if (!std::strcmp(row->name, name)) return row;
    }
    throw std::runtime_error("required registered candidate missing");
}
void constraint_is(const RadKernelInfo& row, const char* key, int op, int64_t value, const char* text = nullptr) {
    for (int index = 0; index < row.n_constraints; ++index) {
        const auto& constraint = row.constraints[index];
        if (!std::strcmp(constraint.key, key) && constraint.op == op && constraint.ival == value &&
            (text ? constraint.sval && !std::strcmp(constraint.sval, text) : !constraint.sval)) return;
    }
    throw std::runtime_error("registered constraint differs from tested qualification");
}
Function registered_launch(Function native, const std::vector<RadParam>& params) {
    const int64_t width = rad_param_geti(params.data(), int(params.size()), "n", 0);
    const char* name = nullptr;
    if (native == r3dh_mtp_ple_mtp_enter && width == 2560) name = "r3d_mtp_enter_e4m3";
    if (native == r3dh_mtp_ple_ple_gate && width == 2560) name = "r3d_ple_gate_bf16";
    if (native == r3dh_mtp_ple_grid_embed && width == 1152) name = "r3d_grid_embed_1152_s48_bf16";
    if (!name) return native;
    const auto* row = registry_row(name);
    expect(row->launch == native && row->opd_shape, "registered native function and shape");
    std::printf("REGISTERED_LAUNCH %s M=%lld\n", name, rad_param_geti(params.data(), int(params.size()), "M", 0));
    return row->launch;
}
void batch_registry_checks() {
    expect(r3d_mtp_kernel_count() == 9 && !r3d_mtp_kernel_at(-1) && !r3d_mtp_kernel_at(9), "registry count and bounds");
    const char* all_names[]{"r3d_gemm_nt_q_gated_i8a8h_strict", "r3d_mtp_enter_e4m3", "r3d_ple_gate_bf16", "r3d_grid_embed_bf16",
        "r3d_row_topk_merge_f32", "r3d_grid_embed_1152_s48_bf16", "r3d_ngram_ids_i32", "r3d_ngram_hint_i32",
        "r3d_logit_rerank_joined_strict"};
    const char* all_ops[]{"gemm_nt_q_gated", "mtp_enter", "ple_gate", "grid_embed", "row_topk_merge", "grid_embed",
        "ngram_ids", "ngram_hint", "logit_rerank"};
    for (int index = 0; index < 9; ++index) {
        const auto* row = r3d_mtp_kernel_at(index);
        expect(row && !std::strcmp(row->name, all_names[index]) && !std::strcmp(row->op, all_ops[index]) &&
            row->launch && (row->opd_shape || !std::strcmp(row->op, "ngram_ids")) &&
            row->domain == RAD_DOMAIN_DEVICE && row->priority == 100,
            "enumerate every registered MTP candidate");
        std::printf("BATCH_ENUM_PASS index=%d name=%s op=%s constraints=%d\n",
            index, row->name, row->op, row->n_constraints);
    }
    const char* names[]{"r3d_mtp_enter_e4m3", "r3d_ple_gate_bf16", "r3d_grid_embed_1152_s48_bf16"};
    const int operand_counts[]{7, 8, 3};
    for (int operation = 0; operation < 3; ++operation) {
        const auto* row = registry_row(names[operation]);
        constraint_is(*row, "M", RAD_C_GE, 1); constraint_is(*row, "M", RAD_C_LE, 2048);
        constraint_is(*row, "n", RAD_C_EQ, operation == 2 ? 1152 : 2560);
        constraint_is(*row, "dtype", RAD_C_IN, 0, "bf16");
        if (operation == 2) constraint_is(*row, "side", RAD_C_EQ, 48);
        else constraint_is(*row, "hc", RAD_C_EQ, 4);
        if (!operation) {
            constraint_is(*row, "ngroup", RAD_C_EQ, 4); constraint_is(*row, "fc", RAD_C_IN, 0, "e4m3");
            expect(row->scratch == r3dh_mtp_ple_mtp_enter_scratch, "registered scratch callback unchanged");
        }
        for (int64_t count : {1, 3, 16, 32, 33, 64, 128, 256, 512, 2048}) {
            std::vector<RadParam> params{RAD_INT("M", count), RAD_INT("n", operation == 2 ? 1152 : 2560),
                RAD_INT("hc", 4), RAD_INT("ngroup", 4), RAD_INT("side", 48)};
            for (int operand = 0; operand < operand_counts[operation]; ++operand) {
                RadOpdDesc description{};
                expect(row->opd_shape(params.data(), int(params.size()), operand, &description) == RAD_OK, "registered shape at tested batch");
            }
            params[0].kind = RAD_P_RANGE; params[0].ival = 1; params[0].ihi = count;
            RadOpdDesc description{};
            expect(row->opd_shape(params.data(), int(params.size()), 0, &description) == RAD_OK &&
                description.shape[0] == count, "registered range resolves upper endpoint");
            for (int64_t invalid : {0, 2049}) {
                params[0] = RAD_INT("M", invalid);
                expect(row->opd_shape(params.data(), int(params.size()), 0, &description) == RAD_E_UNSUPPORTED, "registry rejects unsupported batches");
            }
            params[0].kind = RAD_P_RANGE; params[0].ival = 2; params[0].ihi = 1;
            expect(row->opd_shape(params.data(), int(params.size()), 0, &description) == RAD_E_SHAPE, "registry rejects reversed range");
        }
    }
    constraint_is(*registry_row("r3d_grid_embed_bf16"), "M", RAD_C_LE, 32);
    std::vector<RadParam> legacy_params{RAD_INT("M", 33), RAD_INT("n", 128), RAD_INT("side", 4)};
    RadOpdDesc legacy_description{};
    expect(registry_row("r3d_grid_embed_bf16")->opd_shape(legacy_params.data(), int(legacy_params.size()), 0,
        &legacy_description) == RAD_E_UNSUPPORTED, "legacy grid shape retains M32 cap");
    constraint_is(*registry_row("r3d_row_topk_merge_f32"), "M", RAD_C_LE, 32);
    std::printf("BATCH_REGISTRY_PASS rows=5 qualified_rows=3 batches=10 invalid_M=0,2049 ranged_M=1 preserved_legacy_caps=1\n");
}
