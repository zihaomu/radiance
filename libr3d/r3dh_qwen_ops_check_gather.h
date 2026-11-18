uint64_t gather_cases = 0;
void gather_check(int rows, int repeat, Buffer& routing, Buffer& sorted, hipStream_t stream) {
    constexpr int width = 2560, top = 10;
    Buffer experts(RAD_BF16, rows * top, width, 11), shared(RAD_BF16, rows, width, 13);
    Buffer shared_gate(RAD_BF16, rows, 1, 3), result(RAD_BF16, rows, width, 17);
    Buffer float_weights(RAD_F32, rows, top, 5);
    for (int row = 0; row < rows * top; ++row)
        for (int column = 0; column < width; ++column)
            experts.data<uint16_t>()[row * experts.pitch + column] = bf16(((row * 37 + column * 13 + repeat * 47) % 521 - 260) / 64.0f);
    for (int row = 0; row < rows; ++row) {
        shared_gate.data<uint16_t>()[row * shared_gate.pitch] = 0;
        for (int column = 0; column < width; ++column)
            shared.data<uint16_t>()[row * shared.pitch + column] = bf16(((row * 11 + column * 23) % 257 - 128) / 32.0f);
        for (int slot = 0; slot < top; ++slot)
            float_weights.data<float>()[row * float_weights.pitch + slot] = fp32(routing.data<uint16_t>()[row * top + slot]) + (slot + 1) / 16384.0f;
    }
    if (repeat) sorted.data<int32_t>()[rows * top - 1] = -1;
    Buffer& weights = repeat ? float_weights : routing;
    std::vector<RadParam> params{RAD_INT("M", rows), RAD_INT("n", width), RAD_INT("top_k", top),
        RAD_STR("dtype", "bf16"), RAD_STR("act", "sigmoid")};
    std::vector<uint8_t> expected = result.host;
    std::vector<RadTensor> reference{experts.cpu(), weights.cpu(), sorted.cpu(),
        repeat ? RadTensor{} : shared.cpu(), repeat ? RadTensor{} : shared_gate.cpu(), result.cpu()};
    reference.back().data = expected.data() + 64;
    RadArgs args = arguments(reference, params);
    require(ref_moe_gather(&args, nullptr) == RAD_OK, "libref gather");
    std::vector<int> inverse(rows * top, -1);
    for (int position = 0; position < rows * top; ++position) {
        const int slot = sorted.data<int32_t>()[position];
        if (slot >= 0) inverse[slot] = position;
    }
    for (int row = 0; row < rows; ++row)
        for (int column = 0; column < width; ++column) {
            float sum = 0;
            for (int slot = 0; slot < top; ++slot) {
                const int sorted_row = inverse[row * top + slot];
                if (sorted_row < 0) continue;
                const float weight = repeat ? float_weights.data<float>()[row * float_weights.pitch + slot]
                    : fp32(routing.data<uint16_t>()[row * top + slot]);
                sum = sum + weight * fp32(experts.data<uint16_t>()[sorted_row * experts.pitch + column]);
            }
            uint16_t value = bf16(sum);
            if (!repeat) value = bf16(fp32(value) + fp32(bf16(fp32(shared.data<uint16_t>()[row * shared.pitch + column]) * 0.5f)));
            uint16_t ref_value;
            std::memcpy(&ref_value, expected.data() + 64 + (static_cast<size_t>(row) * result.pitch + column) * 2, 2);
            require(value == ref_value, "independent/libref gather roundings");
        }
    const auto experts_before = experts.host, shared_before = shared.host, weights_before = weights.host;
    for (Buffer* buffer : {&experts, &shared, &shared_gate, &result, &weights, &sorted}) buffer->upload(stream);
    std::vector<RadTensor> tensors{experts.tensor, weights.tensor, sorted.tensor,
        repeat ? RadTensor{} : shared.tensor, repeat ? RadTensor{} : shared_gate.tensor, result.tensor};
    args = arguments(tensors, params);
    require(r3d_rad_moe_gather(&args, reinterpret_cast<RadStream>(stream)) == RAD_OK, "existing native MoE gather");
    for (Buffer* buffer : {&experts, &shared, &shared_gate, &result, &weights, &sorted}) buffer->download(stream);
    gpu(hipStreamSynchronize(stream));
    for (Buffer* buffer : {&experts, &shared, &shared_gate, &result, &weights, &sorted}) buffer->guards();
    require(experts.host == experts_before && shared.host == shared_before && weights.host == weights_before, "gather input mutated");
    compare(result, expected, "existing native gather");
    ++gather_cases;
}
