bool chunk_registry = false;
#if defined(R3DH_GDN_CHUNK_CHECK_REGISTRY)
extern "C" int r3dh_gdn_extra_check_kkt_launch(const RadArgs*, RadStream);
extern "C" int r3dh_gdn_extra_check_scan_launch(const RadArgs*, RadStream);
#endif
int chunk_dispatch(bool solve, const RadArgs* args, RadStream stream) {
#if defined(R3DH_GDN_CHUNK_CHECK_REGISTRY)
    if (chunk_registry) return solve ? r3dh_gdn_extra_check_kkt_launch(args, stream)
        : r3dh_gdn_extra_check_scan_launch(args, stream);
#endif
    expect(!chunk_registry, "registry bridge not linked in this build");
    return solve ? r3dh_gdn_extra_kkt_strict(args, stream) : r3dh_gdn_extra_chunk_strict(args, stream);
}
struct ChunkChecks {
    uint64_t cases = 0, runs = 0, rejects = 0, inverse_bf16 = 0, output_bf16 = 0, pool_f32 = 0;
    uint64_t result_bytes = 0, immutable_bytes = 0, guard_bytes = 0;
} chunk_checks;
uint64_t chunk_hash(const Buffer& buffer) {
    uint64_t hash = 1469598103934665603ull;
    for (uint8_t byte : buffer.reference) hash = (hash ^ byte) * 1099511628211ull;
    return hash;
}
void chunk_identity() {
    expect(std::getenv("ROCR_VISIBLE_DEVICES") && !std::strcmp(std::getenv("ROCR_VISIBLE_DEVICES"), "7"), "chunk physical GPU7 required");
    expect(std::getenv("HIP_VISIBLE_DEVICES") && !std::strcmp(std::getenv("HIP_VISIBLE_DEVICES"), "0"), "chunk logical GPU0 required");
    int count = 0;
    hip_ok(hipGetDeviceCount(&count)); expect(count == 1, "chunk one visible GPU");
    hip_ok(hipSetDevice(0)); hipDeviceProp_t property{};
    hip_ok(hipGetDeviceProperties(&property, 0));
    expect(!std::strncmp(property.gcnArchName, "gfx1100", 7), "chunk native gfx1100 required");
    char pci[64]{}; hip_ok(hipDeviceGetPCIBusId(pci, sizeof(pci), 0));
    expect(std::strstr(pci, "e3:00.0"), "chunk GPU7 PCI identity");
    std::printf("GDN_CHUNK gpu=7 logical=0 arch=%s pci=%s contract=off tolerance=0\n", property.gcnArchName, pci);
}
#include "r3dh_gdn_extra_check_chunk_contract.h"
void chunk_case(const std::vector<int>& lengths, bool warm, bool padded, bool flat_index = false) {
    const int sequences = lengths.size(), heads = 24, keys = 8, slots = sequences + 2;
    std::vector<int> cumulative_values{0};
    for (int length : lengths) cumulative_values.push_back(cumulative_values.back() + length);
    const int rows = cumulative_values.back();
    Fixture fixture;
    auto params = parameters(rows, heads, rows, false, 128, "sigmoid");
    Buffer& query = fixture.add(RAD_BF16, {rows, keys, 128}, 13);
    Buffer& key = fixture.add(RAD_BF16, {rows, keys, 128}, 17);
    Buffer& value = fixture.add(RAD_BF16, {rows, heads, 128}, 19);
    Buffer& inverse = fixture.add(RAD_BF16, {rows, heads, 64}, 23);
    Buffer& decay = fixture.add(RAD_F32, {rows, heads}, 7);
    Buffer& beta = fixture.add(RAD_F32, {rows, heads}, 11);
    Buffer& initial = fixture.add(RAD_F32, {slots, heads, 128, 128}, 29);
    Buffer& cumulative = fixture.add(RAD_I32, {sequences + 1});
    Buffer& output = fixture.add(RAD_BF16, {rows, heads, 128}, 31);
    Buffer& final = fixture.add(RAD_F32, {slots, heads, 128, 128}, 37);
    Buffer& index = flat_index ? fixture.add(RAD_I32, {sequences}) : fixture.add(RAD_I32, {sequences, 3});
    cumulative.integers(cumulative_values);
    const int width = flat_index ? 1 : 3;
    std::vector<int> slot_values(sequences * width, slots + 77);
    for (int sequence = 0; sequence < sequences; ++sequence) slot_values[sequence * width] = sequences - sequence;
    if (padded) slot_values[(sequences - 1) * width] = -1;
    index.integers(slot_values);
    std::vector<Buffer*> solve{&key, &beta, &decay, &cumulative, &inverse};
    std::vector<Buffer*> scan{&query, &key, &value, &inverse, &decay, &beta, warm ? &initial : nullptr,
        &cumulative, &output, &final, &index};
    uint64_t previous_output = 0, previous_state = 0;
    ++chunk_checks.cases;
    for (int repeat = 0; repeat < 2; ++repeat) {
        query.fill(11 + repeat, 0.06f); key.fill(17 + repeat, 0.02f);
        value.fill(23 + repeat, 0.2f); initial.fill(31 + repeat, 0.001f);
        decay.fill(0); beta.fill(0);
        for (int sequence = 0; sequence < sequences; ++sequence) {
            for (int local = 0; local < lengths[sequence]; ++local) {
                const int token = cumulative_values[sequence] + local;
                for (int head = 0; head < heads; ++head) {
                    reinterpret_cast<float*>(decay.reference.data())[token * decay.descriptor.stride[0] + head]
                        = -0.001f * (local % 64 + 1) * (1 + head % 3 + repeat);
                    reinterpret_cast<float*>(beta.reference.data())[token * beta.descriptor.stride[0] + head]
                        = 0.1f + static_cast<float>((token + head + repeat) % 17) / 128;
                }
            }
        }
        for (Buffer* destination : {&inverse, &output, &final}) std::fill(destination->reference.begin(), destination->reference.end(), 0xa5);
        fixture.upload();
        auto solve_host = descriptors(solve, false), solve_device = descriptors(solve, true);
        auto scan_host = descriptors(scan, false), scan_device = descriptors(scan, true);
        auto solve_ref = arguments(solve_host, params), solve_gpu = arguments(solve_device, params);
        auto scan_ref = arguments(scan_host, params), scan_gpu = arguments(scan_device, params);
        rad_ok(ref_gdn_kkt_solve(&solve_ref, nullptr));
        if (chunk_checks.cases == 1 && repeat == 0) chunk_rejections(solve_device, scan_device, params, fixture.stream);
        rad_ok(ref_gdn_chunk_scan(&scan_ref, nullptr));
        for (int sequence = 0; sequence < sequences; ++sequence) {
            for (int local = 0; local < lengths[sequence]; ++local) {
                const int token = cumulative_values[sequence] + local;
                for (int head = 0; head < heads; ++head) {
                    const auto* row = reinterpret_cast<const uint16_t*>(inverse.reference.data())
                        + token * inverse.descriptor.stride[0] + head * inverse.descriptor.stride[1];
                    expect(row[local % 64] == bf16(1), "per-sequence inverse unit diagonal");
                    for (int column = local % 64 + 1; column < 64; ++column)
                        expect(row[column] == 0, "per-sequence inverse upper/tail padding zero");
                }
            }
        }
        const uint64_t current_output = chunk_hash(output), current_state = chunk_hash(final);
        if (repeat) expect(current_output != previous_output && current_state != previous_state, "changed input changes output and final state");
        previous_output = current_output; previous_state = current_state;
        hipGraph_t graph = nullptr; hipGraphExec_t executable = nullptr;
        if (repeat) hip_ok(hipStreamBeginCapture(fixture.stream, hipStreamCaptureModeThreadLocal));
        rad_ok(chunk_dispatch(true, &solve_gpu, reinterpret_cast<RadStream>(fixture.stream)));
        rad_ok(chunk_dispatch(false, &scan_gpu, reinterpret_cast<RadStream>(fixture.stream)));
        if (repeat) {
            hip_ok(hipStreamEndCapture(fixture.stream, &graph));
            hip_ok(hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
            hip_ok(hipGraphLaunch(executable, fixture.stream));
        }
        hip_ok(hipStreamSynchronize(fixture.stream));
        if (repeat) { hip_ok(hipGraphExecDestroy(executable)); hip_ok(hipGraphDestroy(graph)); }
        inverse.compare("chunk_inverse_BF16_all_rows_padding"); output.compare("chunk_output_BF16_all_rows_padding");
        final.compare("chunk_final_F32_all_slots_padding");
        for (Buffer* source : {&query, &key, &value, &decay, &beta, &initial, &cumulative, &index}) {
            source->compare("chunk_input_immutable"); chunk_checks.immutable_bytes += source->bytes;
        }
        fixture.guards();
        ++chunk_checks.runs;
        chunk_checks.inverse_bf16 += static_cast<uint64_t>(rows) * heads * 64;
        chunk_checks.output_bf16 += static_cast<uint64_t>(rows) * heads * 128;
        chunk_checks.pool_f32 += static_cast<uint64_t>(slots) * heads * 128 * 128;
        chunk_checks.result_bytes += inverse.bytes + output.bytes + final.bytes;
        chunk_checks.guard_bytes += fixture.buffers.size() * 512;
        std::printf("PASS chunk_multi seq=%d M=%d heads=24 keys=8 K=128 V=128 chunk=64 initial=%d padded=%d index_rank=%d repeat=%d graph=%d exact_inverse_output_state immutable_inputs guards nondefault_stream lengths=",
            sequences, rows, warm, padded, index.descriptor.rank, repeat, repeat);
        for (int length : lengths) std::printf("%d,", length);
        std::puts(""); std::fflush(stdout);
    }
}
