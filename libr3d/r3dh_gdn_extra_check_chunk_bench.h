void chunk_bench_case(const std::vector<int>& lengths) {
    const int sequences = lengths.size(), heads = 24, keys = 8, slots = sequences + 2;
    std::vector<int> cumulative_values{0};
    for (int length : lengths) cumulative_values.push_back(cumulative_values.back() + length);
    const int rows = cumulative_values.back();
    Fixture fixture;
    auto params = parameters(rows, heads, rows, false, 128, "sigmoid");
    Buffer& query = fixture.add(RAD_BF16, {rows, keys, 128});
    Buffer& key = fixture.add(RAD_BF16, {rows, keys, 128});
    Buffer& value = fixture.add(RAD_BF16, {rows, heads, 128});
    Buffer& inverse = fixture.add(RAD_BF16, {rows, heads, 64});
    Buffer& decay = fixture.add(RAD_F32, {rows, heads});
    Buffer& beta = fixture.add(RAD_F32, {rows, heads});
    Buffer& initial = fixture.add(RAD_F32, {slots, heads, 128, 128});
    Buffer& cumulative = fixture.add(RAD_I32, {sequences + 1});
    Buffer& output = fixture.add(RAD_BF16, {rows, heads, 128});
    Buffer& final = fixture.add(RAD_F32, {slots, heads, 128, 128});
    Buffer& index = fixture.add(RAD_I32, {sequences});
    cumulative.integers(cumulative_values);
    std::vector<int> slot_values(sequences);
    for (int sequence = 0; sequence < sequences; ++sequence) slot_values[sequence] = sequence + 1;
    index.integers(slot_values);
    query.fill(11, 0.06f); key.fill(17, 0.02f); value.fill(23, 0.2f); initial.fill(31, 0.001f);
    decay.fill(0); beta.fill(0);
    for (int sequence = 0; sequence < sequences; ++sequence) {
        for (int local = 0; local < lengths[sequence]; ++local) {
            const int token = cumulative_values[sequence] + local;
            for (int head = 0; head < heads; ++head) {
                reinterpret_cast<float*>(decay.reference.data())[token * decay.descriptor.stride[0] + head]
                    = -0.001f * (local % 64 + 1) * (1 + head % 3);
                reinterpret_cast<float*>(beta.reference.data())[token * beta.descriptor.stride[0] + head]
                    = 0.1f + static_cast<float>((token + head) % 17) / 128;
            }
        }
    }
    for (Buffer* destination : {&inverse, &output, &final}) std::fill(destination->reference.begin(), destination->reference.end(), 0xa5);
    fixture.upload();
    std::vector<Buffer*> solve{&key, &beta, &decay, &cumulative, &inverse};
    std::vector<Buffer*> scan{&query, &key, &value, &inverse, &decay, &beta, &initial, &cumulative, &output, &final, &index};
    auto solve_host = descriptors(solve, false), solve_device = descriptors(solve, true);
    auto scan_host = descriptors(scan, false), scan_device = descriptors(scan, true);
    auto solve_ref = arguments(solve_host, params), solve_gpu = arguments(solve_device, params);
    auto scan_ref = arguments(scan_host, params), scan_gpu = arguments(scan_device, params);
    rad_ok(ref_gdn_kkt_solve(&solve_ref, nullptr));
    rad_ok(ref_gdn_chunk_scan(&scan_ref, nullptr));
    hipEvent_t start = nullptr, stop = nullptr;
    hip_ok(hipEventCreate(&start)); hip_ok(hipEventCreate(&stop));
    const RadStream stream = reinterpret_cast<RadStream>(fixture.stream);
    for (int phase = 0; phase < 2; ++phase) {
        std::vector<float> times;
        for (int iteration = 0; iteration < 36; ++iteration) {
            hip_ok(hipEventRecord(start, fixture.stream));
            rad_ok(phase ? r3dh_gdn_extra_chunk_strict(&scan_gpu, stream) : r3dh_gdn_extra_kkt_strict(&solve_gpu, stream));
            hip_ok(hipEventRecord(stop, fixture.stream));
            hip_ok(hipEventSynchronize(stop));
            float elapsed = 0; hip_ok(hipEventElapsedTime(&elapsed, start, stop));
            if (iteration >= 5) times.push_back(elapsed);
        }
        std::sort(times.begin(), times.end());
        std::printf("BENCH gdn_%s M=%d seq=%d heads=24 keys=8 warmup=5 samples=%zu median_ms=%.4f min_ms=%.4f max_ms=%.4f lengths=",
            phase ? "chunk_scan" : "kkt_solve", rows, sequences, times.size(), times[times.size() / 2], times.front(), times.back());
        for (int length : lengths) std::printf("%d,", length);
        std::puts(""); std::fflush(stdout);
    }
    inverse.compare("bench_inverse_BF16"); output.compare("bench_output_BF16"); final.compare("bench_final_F32");
    for (Buffer* source : {&query, &key, &value, &decay, &beta, &initial, &cumulative, &index}) source->compare("bench_input_immutable");
    fixture.guards();
    hip_ok(hipEventDestroy(start)); hip_ok(hipEventDestroy(stop));
    std::printf("PASS chunk_bench M=%d seq=%d exact_inverse_output_state_vs_libref_after_timed_launches guards\n", rows, sequences);
}
void chunk_bench() {
    chunk_bench_case({585});
    chunk_bench_case({129, 199, 257});
}
