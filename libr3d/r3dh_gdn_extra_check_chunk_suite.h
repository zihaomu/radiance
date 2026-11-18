void chunk_suite() {
    for (int rows : {1, 63, 64, 65, 127, 128, 129, 2048}) chunk_case({rows}, true, false, rows == 64);
    chunk_case({65, 66}, true, false);
    chunk_case({63, 64, 65, 191}, true, false);
    chunk_case({1, 63, 64, 65, 127, 128, 129, 257}, true, false);
    chunk_case({63, 1985}, true, false);
    chunk_case({1, 63, 65, 1919}, true, false);
    chunk_case({0, 1, 63, 64, 65, 127, 129, 1599}, true, false);
    chunk_case({0, 65}, true, false);
    chunk_case({65, 0}, true, false);
    chunk_case({0, 1, 0, 64}, true, false);
    chunk_case({0, 63, 0, 64, 0, 65, 0, 1}, true, false);
    chunk_case({63, 64, 65, 1}, true, true);
    chunk_case({1, 63, 64, 65, 127, 128, 129, 1}, true, true);
    chunk_case({65, 66}, false, false, true);
    std::printf("PASS chunk_dispatch path=%s graph_capture=21 changed_input_pairs=21\n", chunk_registry ? "actual_registry_chunk_launch" : "direct_strict_API");
    std::printf("PASS GDN_CHUNK cases=%llu runs=%llu rejects=%llu inverse_BF16_elements=%llu output_BF16_elements=%llu pool_F32_elements=%llu result_bytes=%llu immutable_bytes=%llu guard_bytes=%llu tolerance=0 synthetic_only heads24 keys8 K128 V128 chunk64 seq1_2_4_8 total_M1_to2048_tested_points_not_exhaustive\n",
        static_cast<unsigned long long>(chunk_checks.cases), static_cast<unsigned long long>(chunk_checks.runs),
        static_cast<unsigned long long>(chunk_checks.rejects), static_cast<unsigned long long>(chunk_checks.inverse_bf16),
        static_cast<unsigned long long>(chunk_checks.output_bf16), static_cast<unsigned long long>(chunk_checks.pool_f32),
        static_cast<unsigned long long>(chunk_checks.result_bytes), static_cast<unsigned long long>(chunk_checks.immutable_bytes),
        static_cast<unsigned long long>(chunk_checks.guard_bytes));
}
