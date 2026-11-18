uint64_t n1_cases = 0, n1_launches = 0, n1_values = 0;
void router_n1_case(int rows, bool padded, hipStream_t first, hipStream_t second,
                    const std::vector<uint16_t>* actual = nullptr, const char* label = "synthetic") {
    constexpr int width = 2560;
    Buffer input(RAD_BF16, rows, width, padded ? 13 : 0);
    Buffer weight(RAD_BF16, 1, width, padded ? 9 : 0);
    Buffer output(RAD_BF16, rows, 1, padded ? 7 : 0);
    std::vector<RadParam> params{RAD_INT("M", rows), RAD_INT("N", 1), RAD_INT("K", width), RAD_STR("dtype", "bf16")};
    std::vector<uint16_t> previous;
    for (int repeat = 0; repeat < 3; ++repeat) {
        const hipStream_t stream = repeat == 1 ? second : first;
        for (int row = 0; row < rows; ++row)
            for (int position = 0; position < width; ++position)
                input.data<uint16_t>()[row * input.pitch + position] = repeat == 2 ? 0
                    : bf16(((row * 47 + position * 13 + repeat * 29) % 197 - 98) / 128.0f);
        for (int position = 0; position < width; ++position)
            weight.data<uint16_t>()[position] = actual ? (*actual)[position] : bf16(((position * 23 + 17) % 251 - 125) / 256.0f);
        std::fill(output.host.begin(), output.host.end(), uint8_t{0xa5});
        const auto input_before = input.host, weight_before = weight.host;
        std::vector<uint8_t> expected = output.host, reference = output.host;
        std::vector<uint16_t> values(static_cast<size_t>(rows));
        for (int row = 0; row < rows; ++row) {
            float sum = 0.0f;
            for (int position = 0; position < width; ++position) {
                const float product = fp32(input.data<uint16_t>()[row * input.pitch + position])
                    * fp32(weight.data<uint16_t>()[position]);
                sum = sum + product;
            }
            const uint16_t value = bf16(sum);
            values[row] = value;
            std::memcpy(expected.data() + 64 + static_cast<size_t>(row) * output.pitch * 2, &value, 2);
            if (repeat == 2) require(value == 0, "N1 zero-input oracle");
        }
        if (repeat == 1) require(values != previous, "N1 changed input must change expected output");
        previous = values;
        std::vector<RadTensor> host_tensors{input.cpu(), weight.cpu(), output.cpu()};
        host_tensors[2].data = reference.data() + 64;
        RadArgs args = arguments(host_tensors, params);
        require(ref_gemm_nt(&args, nullptr) == RAD_OK, "N1 libref GEMM");
        require(reference == expected, "N1 independent CPU/libref bit-exact");
        input.upload(stream); weight.upload(stream); output.upload(stream);
        std::vector<RadTensor> tensors{input.tensor, weight.tensor, output.tensor};
        args = arguments(tensors, params);
        require(r3dh_qwen_ops_router_gemm(&args, reinterpret_cast<RadStream>(stream)) == RAD_OK, "N1 native GEMM launch");
        input.download(stream); weight.download(stream); output.download(stream);
        gpu(hipStreamSynchronize(stream));
        input.guards(); weight.guards(); output.guards();
        require(input.host == input_before && weight.host == weight_before, "N1 inputs unchanged");
        compare(output, expected, "N1 CPU/native exact");
        compare(output, reference, "N1 libref/native exact");
        n1_values += rows;
        ++n1_launches;
    }
    ++n1_cases;
    std::printf("QWEN_N1_CASE source=%s M=%d N=1 K=2560 padded=%d launches=3 full_outputs=%d zero_exact=1 guards=1 streams=2\n", label, rows, int(padded), rows * 3);
}
