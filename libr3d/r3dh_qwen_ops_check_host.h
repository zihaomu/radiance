void extra_host_checks() {
    alignas(16) std::array<uint8_t, 16> placeholder{};
    auto* address = placeholder.data();
    const std::vector<RadTensor> valid{matrix(address, RAD_BF16, 1, 768, 781),
        matrix(address, RAD_I8, 1, 384, 401), matrix(address, RAD_F32, 1, 3, 8),
        matrix(address, RAD_BF16, 1, 384, 403)};
    const std::vector<RadParam> good_params{RAD_INT("n", 384), RAD_INT("group", 128), RAD_STR("dtype", "bf16")};
    uint64_t rejected = 0;
    auto check = [&](std::vector<RadTensor> tensors, std::vector<RadParam> params, int expected) {
        RadArgs args = arguments(tensors, params);
        require(r3dh_qwen_ops_gated_had_quant_i8g(&args, nullptr) == expected, "gate malformed ABI status");
        ++rejected;
    };
    for (int operand = 0; operand < 4; ++operand) {
        auto tensors = valid; tensors[operand].dtype = RAD_F16;
        check(tensors, good_params, RAD_E_DTYPE);
        if (operand != 3) {
            tensors = valid; tensors[operand].data = nullptr;
            check(tensors, good_params, RAD_E_INVAL);
        }
        tensors = valid; tensors[operand].rank = 0;
        check(tensors, good_params, RAD_E_SHAPE);
        tensors = valid; tensors[operand].shape[0] = -1;
        check(tensors, good_params, RAD_E_SHAPE);
        tensors = valid; tensors[operand].stride[1] = 2;
        check(tensors, good_params, RAD_E_STRIDE);
        tensors = valid; tensors[operand].stride[0] = -1;
        check(tensors, good_params, RAD_E_STRIDE);
        tensors = valid; tensors[operand].shape[0] = 2;
        check(tensors, good_params, RAD_E_SHAPE);
    }
    for (int width : {0, -128, 127, 129, 383, 1048704}) {
        auto params = good_params; params[0] = RAD_INT("n", width);
        check(valid, params, RAD_E_SHAPE);
    }
    for (int group : {0, 64, 256}) {
        auto params = good_params; params[1] = RAD_INT("group", group);
        check(valid, params, RAD_E_SHAPE);
    }
    auto params = good_params; params.push_back(RAD_INT("chain", 1));
    check(valid, params, RAD_E_UNSUPPORTED);
    const std::vector<RadTensor> router_valid{matrix(address, RAD_BF16, 1, 2560, 2573),
        matrix(address, RAD_BF16, 512, 2560, 2569), matrix(address, RAD_BF16, 1, 512, 519)};
    const std::vector<RadParam> router_params{RAD_INT("M", 1), RAD_INT("N", 512), RAD_INT("K", 2560), RAD_STR("dtype", "bf16")};
    for (int operand = 0; operand < 3; ++operand) {
        auto tensors = router_valid; tensors[operand].dtype = RAD_F16;
        RadArgs args = arguments(tensors, router_params);
        require(r3dh_qwen_ops_router_gemm(&args, nullptr) == RAD_E_DTYPE, "router F16 rejected"); ++rejected;
        tensors[operand] = router_valid[operand]; tensors[operand].data = nullptr;
        require(r3dh_qwen_ops_router_gemm(&args, nullptr) == RAD_E_INVAL, "router null operand rejected"); ++rejected;
        tensors[operand] = router_valid[operand]; tensors[operand].stride[1] = 2;
        require(r3dh_qwen_ops_router_gemm(&args, nullptr) == RAD_E_STRIDE, "router inner stride rejected"); ++rejected;
        tensors[operand] = router_valid[operand]; tensors[operand].shape[1] -= 1;
        require(r3dh_qwen_ops_router_gemm(&args, nullptr) == RAD_E_SHAPE, "router wrong width rejected"); ++rejected;
    }
    std::printf("QWEN_ABI_PASS extra_rejections=%llu\n", (unsigned long long)rejected);
}
