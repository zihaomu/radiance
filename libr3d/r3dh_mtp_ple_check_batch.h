#include "r3dh_mtp_ple_check_registry.h"
std::vector<int64_t> sample_rows(int64_t count) {
    std::vector<int64_t> result{0, 1, 31, count / 4, count / 2, count - 33, count - 2, count - 1};
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    expect(result.front() >= 0 && result.back() < count, "sampled row bounds");
    return result;
}
void sampled_reference(RadArgs args, Function reference, std::vector<RadParam> params,
                       const std::vector<int64_t>& selected) {
    std::vector<RadTensor> original(args.t, args.t + args.n_t);
    params[0] = RAD_INT("M", 1);
    args.p = params.data();
    for (const int64_t row : selected) {
        auto sliced = original;
        for (const int index : {0, 1, 6}) {
            sliced[index].data = static_cast<uint8_t*>(original[index].data) + row * original[index].stride[0] * 2;
            sliced[index].shape[0] = 1;
        }
        args.t = sliced.data();
        expect(reference(&args, nullptr) == RAD_OK, "independent libref sampled row");
    }
}
void compare_sampled(Blob& output, const RadTensor& descriptor,
                     const std::vector<int64_t>& selected, const char* label) {
    std::vector<uint8_t> observed(output.bytes.size());
    hipcheck(hipMemcpy(observed.data(), output.device, observed.size(), hipMemcpyDeviceToHost));
    expect(std::equal(observed.begin(), observed.begin() + 64, output.bytes.begin()) &&
        std::equal(observed.end() - 64, observed.end(), output.bytes.end() - 64), "sampled output allocation guards");
    const auto* actual = reinterpret_cast<const uint16_t*>(observed.data() + 64);
    const auto* expected = output.data<uint16_t>();
    uint64_t matched = 0;
    for (int64_t row = 0; row < descriptor.shape[0]; ++row) {
        const bool checked = std::binary_search(selected.begin(), selected.end(), row);
        for (int64_t column = 0; column < descriptor.stride[0]; ++column) {
            const int64_t offset = row * descriptor.stride[0] + column;
            if (column < descriptor.shape[1]) expect((actual[offset] & 0x7f80u) != 0x7f80u, "all native outputs finite");
            if (!checked && column < descriptor.shape[1]) continue;
            if (actual[offset] != expected[offset]) {
                std::fprintf(stderr, "FAIL %s row=%lld column=%lld expected=%04x actual=%04x\n", label,
                    static_cast<long long>(row), static_cast<long long>(column), expected[offset], actual[offset]);
                throw std::runtime_error("sampled output/padding mismatch");
            }
            comparisons += 2;
            if (column < descriptor.shape[1]) ++matched;
        }
    }
    std::printf("SAMPLED_CRITERION rows=");
    for (size_t index = 0; index < selected.size(); ++index)
        std::printf("%s%lld", index ? "," : "", static_cast<long long>(selected[index]));
    std::printf(" matched_bf16=%llu total_bf16=%lld full_native_launch=1 full_finite_scan=1 nan_sentinel_replaced=1 all_padding_guards=1 full_output_certified=0\n",
        static_cast<unsigned long long>(matched), static_cast<long long>(descriptor.shape[0] * descriptor.shape[1]));
}
