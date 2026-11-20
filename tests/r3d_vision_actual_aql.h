TEST(r3d_vision_attention_actual_aql) {
    if (!joined_head_test::device_available()) SKIP("explicit gfx1100 card run required");
    REQUIRE_EQ(rad_dev_set(0), RAD_OK);
    Library native("libr3d"), reference("libref");
    const auto* kernel = attn_prepare_boundary::registered(native, "r3dh_vision_strict_attn72");
    const auto* oracle = reference.ngram("attn_dense");
    Queue compute(0), other(1);
    uint64_t compared = 0, calls = 0;
    for (int rows : {1, 49, 196}) for (int step = 0; step < 2; ++step) {
        const auto stream = step ? other.stream : compute.stream;
        Buffer qkv(RAD_BF16, {rows, 3456}, RAD_MEM_DEVICE);
        Buffer cu(RAD_I32, {rows == 1 ? 2 : 4}, RAD_MEM_HOST_MAPPED);
        Buffer output(RAD_BF16, {rows, 1152}, RAD_MEM_DEVICE, 1160);
        for (int index = 0; index < rows * 3456; ++index)
            qkv.data<uint16_t>()[index] = rad_f32_to_bf16(float((index * 7 + step * 3) % 31 - 15) / 16);
        cu.data<int32_t>()[0] = 0;
        if (rows == 1) cu.data<int32_t>()[1] = rows;
        else {
            cu.data<int32_t>()[1] = rows / 2;
            cu.data<int32_t>()[2] = rows / 2;
            cu.data<int32_t>()[3] = rows;
        }
        for (Buffer* buffer : {&qkv, &cu, &output}) buffer->push(stream);
        REQUIRE_EQ(rad_stream_sync(stream), RAD_OK);
        RadTensor host[]{qkv.host, qkv.host, qkv.host, cu.host, output.host};
        RadTensor device[]{qkv.device, qkv.device, qkv.device, cu.device, output.device};
        for (int operand = 0; operand < 3; ++operand) {
            host[operand].shape[1] = device[operand].shape[1] = 1152;
            host[operand].data = static_cast<uint16_t*>(qkv.host.data) + operand * 1152;
            device[operand].data = static_cast<uint16_t*>(qkv.device.data) + operand * 1152;
        }
        const RadParam params[]{RAD_INT("M", 256), RAD_INT("head_dim", 72), RAD_INT("gqa", 1),
            RAD_INT("causal", 0), RAD_STR("dtype", "bf16"), RAD_F64("scale", 0.11785113019775793),
            RAD_INT("max_seqlen", 256)};
        RadArgs expected{}; expected.t = host; expected.n_t = 5; expected.p = params; expected.n_p = 7;
        REQUIRE_EQ(oracle->launch(&expected, nullptr), RAD_OK);
        RadArgs actual = expected; actual.t = device;
        REQUIRE_EQ(kernel->launch(&actual, stream), RAD_OK);
        REQUIRE(output.read(stream) == output.expected);
        REQUIRE(qkv.read(stream) == qkv.expected);
        REQUIRE(cu.read(stream) == cu.expected);
        device[4].shape[0] = 257;
        CHECK_EQ(kernel->launch(&actual, stream), RAD_E_SHAPE);
        compared += output.expected.size(); ++calls;
    }
    std::printf("VISION_ATTENTION_ACTUAL_AQL_PASS calls=%llu exact_bytes=%llu rows=1,49,196 band=256 queues=2\n",
        (unsigned long long)calls, (unsigned long long)compared);
}

TEST(r3d_vision_rope_actual_aql) {
    if (!joined_head_test::device_available()) SKIP("explicit gfx1100 card run required");
    REQUIRE_EQ(rad_dev_set(0), RAD_OK);
    Library native("libr3d"), reference("libref");
    const auto* kernel = attn_prepare_boundary::registered(native, "r3dh_vision_axial_rope72");
    const auto* oracle = reference.ngram("rope");
    Queue compute(0), other(1);
    uint64_t compared = 0, calls = 0;
    for (int rows : {1, 49, 196}) for (int step = 0; step < 2; ++step) {
        const auto stream = step ? other.stream : compute.stream;
        Buffer qkv(RAD_BF16, {rows, 3456}, RAD_MEM_DEVICE);
        Buffer coordinates(RAD_I32, {2, rows}, RAD_MEM_HOST_MAPPED);
        for (int index = 0; index < rows * 3456; ++index)
            qkv.data<uint16_t>()[index] = rad_f32_to_bf16(float((index * 7 + step * 3) % 31 - 15) / 16);
        for (int row = 0; row < rows; ++row) {
            coordinates.data<int32_t>()[row] = row / 14 + step;
            coordinates.data<int32_t>()[rows + row] = row % 14 + step;
        }
        qkv.push(stream); coordinates.push(stream);
        REQUIRE_EQ(rad_stream_sync(stream), RAD_OK);
        const RadParam params[]{RAD_INT("M", 256), RAD_INT("head_dim", 72), RAD_INT("n_head", 16),
            RAD_INT("n_head_kv", 16), RAD_F64("theta", 10000), RAD_F64("scale", 1),
            RAD_STR("mode", "axial"), RAD_INT("rotary_dim", 72), RAD_STR("sections", "18 18")};
        RadTensor host[]{qkv.host, coordinates.host}, device[]{qkv.device, coordinates.device};
        RadArgs expected{}; expected.t = host; expected.n_t = 2; expected.p = params; expected.n_p = 9;
        REQUIRE_EQ(oracle->launch(&expected, nullptr), RAD_OK);
        RadArgs actual = expected; actual.t = device;
        REQUIRE_EQ(kernel->launch(&actual, stream), RAD_OK);
        REQUIRE(qkv.read(stream) == qkv.expected);
        REQUIRE(coordinates.read(stream) == coordinates.expected);
        device[0].shape[0] = 257;
        CHECK_EQ(kernel->launch(&actual, stream), RAD_E_SHAPE);
        compared += qkv.expected.size(); ++calls;
    }
    std::printf("VISION_ROPE_ACTUAL_AQL_PASS calls=%llu exact_bytes=%llu rows=1,49,196 band=256 queues=2\n",
        (unsigned long long)calls, (unsigned long long)compared);
}

TEST(r3d_vision_patch_actual_aql) {
    if (!joined_head_test::device_available()) SKIP("explicit gfx1100 card run required");
    REQUIRE_EQ(rad_dev_set(0), RAD_OK);
    Library native("libr3d"), reference("libref");
    const auto* kernel = attn_prepare_boundary::registered(native, "r3dh_vision_strict_bias_patch");
    const auto* oracle = reference.ngram("gemm_nt_bias");
    Queue compute(0), other(1);
    uint64_t compared = 0, calls = 0;
    for (int memory : {RAD_MEM_DEVICE, RAD_MEM_HOST_PINNED, RAD_MEM_HOST_MAPPED}) for (int rows : {1, 49, 196}) {
        Buffer input(RAD_BF16, {rows, 1536}, memory, 1544);
        Buffer weight(RAD_BF16, {1152, 1536}, RAD_MEM_DEVICE);
        Buffer bias(RAD_BF16, {1152}, RAD_MEM_DEVICE);
        Buffer output(RAD_BF16, {rows, 1152}, RAD_MEM_DEVICE, 1160);
        for (int column = 0; column < 1152; ++column) {
            bias.data<uint16_t>()[column] = rad_f32_to_bf16(float(column % 11 - 5) / 64);
            for (int depth = 0; depth < 1536; ++depth)
                weight.data<uint16_t>()[column * 1536 + depth] =
                    rad_f32_to_bf16(float((column * 3 + depth) % 17 - 8) / 32);
        }
        const RadParam params[]{RAD_INT("M", 256), RAD_INT("N", 1152), RAD_INT("K", 1536),
            RAD_STR("dtype", "bf16"), RAD_STR("act", "none")};
        for (int step = 0; step < 2; ++step) {
            const auto stream = step ? other.stream : compute.stream;
            for (int row = 0; row < rows; ++row) for (int depth = 0; depth < 1536; ++depth)
                input.data<uint16_t>()[row * 1544 + depth] =
                    rad_f32_to_bf16(float((row + depth * 7 + step * 3) % 19 - 9) / 16);
            std::fill(output.expected.begin(), output.expected.end(), uint8_t(0x5a));
            for (Buffer* buffer : {&input, &weight, &bias, &output}) buffer->push(stream);
            REQUIRE_EQ(rad_stream_sync(stream), RAD_OK);
            RadTensor host[]{input.host, weight.host, bias.host, {}, output.host};
            RadTensor device[]{input.device, weight.device, bias.device, {}, output.device};
            RadArgs expected{}; expected.t = host; expected.n_t = 5;
            expected.p = params; expected.n_p = 5;
            REQUIRE_EQ(oracle->launch(&expected, nullptr), RAD_OK);
            RadArgs actual = expected; actual.t = device;
            REQUIRE_EQ(kernel->launch(&actual, stream), RAD_OK);
            REQUIRE(output.read(stream) == output.expected);
            REQUIRE(input.read(stream) == input.expected);
            REQUIRE(weight.read(stream) == weight.expected);
            REQUIRE(bias.read(stream) == bias.expected);
            compared += output.expected.size(); ++calls;
            auto invalid = device[4]; device[4].shape[0] = 257;
            CHECK_EQ(kernel->launch(&actual, stream), RAD_E_SHAPE);
            device[4] = invalid;
            REQUIRE_EQ(kernel->launch(&actual, stream), RAD_OK);
            REQUIRE(output.read(stream) == output.expected);
            ++calls;
            joined_head_test::Storage virtual_weight(weight.expected.size() - 128, -1);
            virtual_weight.expected = weight.expected;
            virtual_weight.push(stream);
            std::vector<uint8_t> poison(output.expected.size(), 0x5a);
            REQUIRE_EQ(rad_memcpy_async(output.allocation, poison.data(), poison.size(), stream), RAD_OK);
            REQUIRE_EQ(rad_stream_sync(stream), RAD_OK);
            device[1].data = virtual_weight.device();
            REQUIRE_EQ(kernel->launch(&actual, stream), RAD_OK);
            REQUIRE(output.read(stream) == output.expected);
            REQUIRE(virtual_weight.read(stream) == weight.expected);
            device[1] = weight.device;
            ++calls;
        }
    }
    std::printf("VISION_PATCH_ACTUAL_AQL_PASS calls=%llu exact_bytes=%llu rows=1,49,196 band=256 queues=2\n",
        (unsigned long long)calls, (unsigned long long)compared);
}
