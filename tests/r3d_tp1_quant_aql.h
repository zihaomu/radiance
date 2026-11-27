TEST(r3d_tp1_hc_actual_aql) {
    if (!std::getenv("RAD_TP1_HC_CHECK")) SKIP("optional actual HC weights check");
    const bool compare_matrix = std::getenv("RAD_TP1_HC_MATRIX_AB") != nullptr;
    const bool compare_packed = std::getenv("RAD_TP1_HC_AB") != nullptr || compare_matrix;
    const char* override_switch = std::getenv("RAD_TP1_HC_SWITCH");
    const char* switch_name = override_switch ? override_switch
        : compare_matrix ? "RAD_HC_MATRIX_DOWN" : "RAD_HC_DOWN_PACKED";
    const int tolerant_rows = override_switch ? 16 : 1;
    const char* original_packed = std::getenv(switch_name);
    const bool packed_present = original_packed != nullptr;
    const std::string saved_packed = original_packed ? original_packed : "";
    using namespace iqr_output_boundary;
    RadDeviceProps props{};
    REQUIRE_EQ(rad_dev_props(0, &props), RAD_OK);
    REQUIRE(!props.is_host_backend);
    REQUIRE_EQ(std::strcmp(props.arch, "gfx1100"), 0);
    Library native("libr3d");
    using TraceLaunch = int (*)(const RadArgs*, RadStream, const R3dhHcTrace*);
    using Profile = int (*)(const RadArgs*, RadStream, float*);
    using Relayout = int (*)(const RadParam*, int, int, const RadEncoding*, const int*, const RadTensor*, int, void*, int64_t);
    auto launch = reinterpret_cast<TraceLaunch>(dlsym(native.handle, "r3dh_hc_rad_hc_read_strict_trace"));
    auto profile = reinterpret_cast<Profile>(dlsym(native.handle, "r3dh_hc_rad_hc_read_profile"));
    auto scratch_size = reinterpret_cast<int64_t (*)(const RadArgs*)>(dlsym(native.handle, "r3dh_hc_rad_hc_read_strict_scratch"));
    auto relayout = reinterpret_cast<Relayout>(dlsym(native.handle, "r3dh_hc_rad_relayout_hc_e4m3"));
    REQUIRE(launch && profile && scratch_size && relayout);
    Model model("/home/zihaomu/bigssd2/radiance_workspace/models/qwen3.8-next-flash-iqr-v4.rad");
    Queue compute(0);
    Buffer down(RAD_F8E4M3, {320, 10560}, RAD_MEM_DEVICE);
    Buffer up(RAD_F8E4M3, {10240, 336}, RAD_MEM_DEVICE);
    Buffer gain(RAD_F32, {10240}, RAD_MEM_DEVICE);
    Buffer inject_weights(RAD_BF16, {4, 10240}, RAD_MEM_DEVICE);
    RadParam params[]{RAD_INT("M", 16), RAD_INT("n", 2560), RAD_INT("hc", 4),
        RAD_INT("lowrank", 320), RAD_STR("dtype", "bf16"), RAD_STR("mix", "e4m3"),
        RAD_INT("inject", 1), RAD_INT("group", 128), RAD_INT("rotate", 1)};
    for (const char* role : {"down", "up", "norm", "inject"}) {
        const std::string name = "blk.0.attn_hc_" + std::string(role) + ".weight";
        bool found = false;
        for (const auto& entry : model.entries) {
            if (model.name(entry) != name) continue;
            REQUIRE(entry.enc < model.encodings.size());
            const auto& encoding = model.encodings[entry.enc];
            REQUIRE(rad_enc_valid(&encoding));
            std::vector<std::vector<uint8_t>> raw(entry.n_planes);
            for (unsigned index = 0; index < entry.n_planes; ++index) {
                REQUIRE(entry.plane_first + index < model.planes.size());
                const auto& location = model.planes[entry.plane_first + index];
                raw[index].resize(location.bytes);
                model.read(location.offset, raw[index].data(), raw[index].size());
            }
            if (!std::strcmp(role, "down") || !std::strcmp(role, "up")) {
                const bool is_down = !std::strcmp(role, "down");
                const int outputs = is_down ? 320 : 10240, depth = is_down ? 10240 : 320;
                const int group = is_down ? 128 : 80;
                REQUIRE_EQ(entry.shape[0], outputs); REQUIRE_EQ(entry.shape[1], depth);
                REQUIRE_EQ(entry.n_planes, 2u);
                int selected[2]{-1, -1};
                for (unsigned index = 0; index < entry.n_planes; ++index) {
                    if (!std::strcmp(encoding.plane[index].role, "codes")) selected[0] = int(index);
                    if (!std::strcmp(encoding.plane[index].role, "scale")) selected[1] = int(index);
                }
                REQUIRE(selected[0] >= 0 && selected[1] >= 0);
                REQUIRE_EQ(raw[selected[0]].size(), size_t(outputs) * depth);
                REQUIRE_EQ(raw[selected[1]].size(), size_t(outputs) * (depth / group) * 4);
                RadTensor planes[]{matrix(raw[selected[0]].data(), RAD_F8E4M3, outputs, depth),
                    matrix(raw[selected[1]].data(), RAD_F32, outputs, depth / group)};
                auto& destination = is_down ? down : up;
                REQUIRE_EQ(relayout(params, 9, is_down ? 2 : 3, &encoding, selected, planes, 2,
                    destination.host.data, int64_t(outputs) * destination.host.stride[0]), RAD_OK);
            } else {
                REQUIRE_EQ(entry.n_planes, 1u);
                if (!std::strcmp(role, "norm")) {
                    REQUIRE_EQ(raw[0].size(), size_t(10240 * 2));
                    for (int position = 0; position < 10240; ++position) {
                        uint16_t bits = 0;
                        std::memcpy(&bits, raw[0].data() + position * 2, 2);
                        gain.data<float>()[position] = rad_bf16_to_f32(bits);
                    }
                } else {
                    REQUIRE_EQ(raw[0].size(), size_t(40960 * 2));
                    std::memcpy(inject_weights.host.data, raw[0].data(), raw[0].size());
                }
            }
            found = true; break;
        }
        REQUIRE(found);
    }
    for (Buffer* buffer : {&down, &up, &gain, &inject_weights}) buffer->push(compute.stream);
    for (int rows : {1, 2, 3, 4, 5}) {
        Buffer hidden(RAD_BF16, {rows, 10240}, RAD_MEM_DEVICE);
        Buffer output(RAD_BF16, {rows, 2560}, RAD_MEM_DEVICE);
        Buffer inject_output(RAD_BF16, {rows, 4}, RAD_MEM_DEVICE);
        Buffer quantized(RAD_I8, {rows, 2560}, RAD_MEM_DEVICE);
        Buffer quant_scales(RAD_F32, {rows, 20}, RAD_MEM_DEVICE);
        Buffer rotated(RAD_I8, {rows, 2560}, RAD_MEM_DEVICE);
        Buffer rotated_scales(RAD_F32, {rows, 20}, RAD_MEM_DEVICE);
        Buffer traced_down(RAD_BF16, {rows, 320}, RAD_MEM_DEVICE);
        Buffer traced_logits(RAD_F32, {rows, 10240}, RAD_MEM_DEVICE);
        Buffer traced_gates(RAD_BF16, {rows, 10240}, RAD_MEM_DEVICE);
        for (int position = 0; position < rows * 10240; ++position)
            hidden.data<uint16_t>()[position] = rad_f32_to_bf16(float((position * 47) % 257 - 128) / 64.0f);
        for (Buffer* buffer : {&hidden, &output, &inject_output, &quantized, &quant_scales,
                &rotated, &rotated_scales, &traced_down, &traced_logits, &traced_gates}) buffer->push(compute.stream);
        RadTensor tensors[]{hidden.device, gain.device, down.device, up.device, inject_weights.device,
            output.device, inject_output.device, quantized.device, quant_scales.device, rotated.device, rotated_scales.device};
        tensors[2].shape[1] = tensors[2].stride[0] = 10240;
        tensors[3].shape[1] = tensors[3].stride[0] = 320;
        RadArgs args{}; args.t = tensors; args.n_t = 11; args.p = params; args.n_p = 9;
        const int64_t bytes = scratch_size(&args);
        REQUIRE(bytes > 0);
        Buffer scratch(RAD_I8, {bytes}, RAD_MEM_DEVICE);
        args.scratch = scratch.allocation; args.scratch_bytes = bytes;
        R3dhHcTrace trace{static_cast<uint16_t*>(traced_down.device.data), static_cast<float*>(traced_logits.device.data),
            static_cast<uint16_t*>(traced_gates.device.data), rows, 2560, 4, 320};
        if (compare_packed) REQUIRE_EQ(setenv(switch_name, "0", 1), 0);
        REQUIRE_EQ(launch(&args, compute.stream, &trace), RAD_OK);
        const auto normalized = scratch.read(compute.stream);
        const auto actual_down = traced_down.read(compute.stream);
        for (int row = 0; row < rows; ++row) for (int column = 0; column < 320; ++column) {
            const uint8_t* line = down.data<uint8_t>() + column * 10560;
            float sum = 0;
            for (int position = 0; position < 10240; ++position) {
                uint16_t bits = 0; float scale = 0;
                std::memcpy(&bits, normalized.data() + (size_t(row) * 10240 + position) * 2, 2);
                std::memcpy(&scale, line + 10240 + (position / 128) * 4, 4);
                const unsigned code = line[position], exponent = (code >> 3) & 15, mantissa = code & 7;
                const float magnitude = exponent ? std::ldexp(float(8 + mantissa), int(exponent) - 10)
                    : float(mantissa) * 0x1p-9f;
                volatile float weight = (code & 128 ? -magnitude : magnitude) * scale;
                volatile float product = rad_bf16_to_f32(bits) * weight;
                sum = sum + product;
            }
            const float divided = sum / 4.0f;
            const float exponential = float(std::exp(double(-divided)));
            volatile float denominator = 1.0f + exponential;
            const float gate = 1.0f / denominator;
            const uint16_t expected = moe_boundary::pack(divided * gate);
            REQUIRE_EQ(std::memcmp(actual_down.data() + 64 + (size_t(row) * 320 + column) * 2, &expected, 2), 0);
        }
        std::vector<std::vector<uint8_t>> baseline;
        for (Buffer* buffer : {&output, &inject_output, &quantized, &quant_scales, &rotated, &rotated_scales,
                &traced_down, &traced_logits, &traced_gates}) {
            const auto actual = buffer->read(compute.stream);
            const int element_bytes = buffer->host.dtype == RAD_I8 ? 1 : buffer->host.dtype == RAD_F32 ? 4 : 2;
            const size_t end = 64 + size_t(buffer->host.shape[0] * buffer->host.shape[1]) * element_bytes;
            REQUIRE(std::equal(actual.begin(), actual.begin() + 64, buffer->expected.begin()));
            REQUIRE(std::equal(actual.begin() + end, actual.end(), buffer->expected.begin() + end));
            baseline.push_back(actual);
        }
        float stages[6]{};
        REQUIRE_EQ(profile(&args, compute.stream, stages), RAD_OK);
        int stage = 0;
        for (Buffer* buffer : {&output, &inject_output, &quantized, &quant_scales, &rotated, &rotated_scales,
                &traced_down, &traced_logits, &traced_gates}) REQUIRE(buffer->read(compute.stream) == baseline[stage++]);
        REQUIRE(hidden.read(compute.stream) == hidden.expected);
        for (Buffer* buffer : {&down, &up, &gain, &inject_weights}) REQUIRE(buffer->read(compute.stream) == buffer->expected);
        std::printf("TP1_HC_ACTUAL M=%d down_exact=%d norm_us=%.3f down_us=%.3f up_us=%.3f mix_us=%.3f inject_us=%.3f quant_us=%.3f\n",
            rows, rows * 320, stages[0], stages[1], stages[2], stages[3], stages[4], stages[5]);
        if (compare_packed) {
            auto plain = reinterpret_cast<RadLaunchFn>(dlsym(native.handle, "r3dh_hc_rad_hc_read_strict"));
            REQUIRE(plain);
            float timings[2]{};
            for (int variant : {0, 1, 1, 0}) {
                REQUIRE_EQ(setenv(switch_name, variant ? "1" : "0", 1), 0);
                REQUIRE_EQ(launch(&args, compute.stream, &trace), RAD_OK);
                stage = 0;
                for (Buffer* buffer : {&output, &inject_output, &quantized, &quant_scales, &rotated, &rotated_scales,
                        &traced_down, &traced_logits, &traced_gates}) {
                    const auto actual = buffer->read(compute.stream);
                    const auto& expected = baseline[stage++];
                    if (!compare_matrix || !variant || rows > tolerant_rows) REQUIRE(actual == expected);
                    else {
                        const int count = int(buffer->host.shape[0] * buffer->host.shape[1]);
                        const int element_bytes = buffer->host.dtype == RAD_I8 ? 1 : buffer->host.dtype == RAD_F32 ? 4 : 2;
                        const size_t end = 64 + size_t(count) * element_bytes;
                        REQUIRE(std::equal(actual.begin(), actual.begin() + 64, expected.begin()));
                        REQUIRE(std::equal(actual.begin() + end, actual.end(), expected.begin() + end));
                        double error_square = 0, target_square = 0;
                        float maximum = 0, peak = 0;
                        for (int index = 0; index < count; ++index) {
                            const float value = rad_load_f32(actual.data() + 64, buffer->host.dtype, index);
                            const float target = rad_load_f32(expected.data() + 64, buffer->host.dtype, index);
                            REQUIRE(std::isfinite(value));
                            const double error = double(value) - target;
                            maximum = std::max(maximum, float(std::fabs(error))); peak = std::max(peak, std::fabs(target));
                            error_square += error * error; target_square += double(target) * target;
                        }
                        const double relative = std::sqrt(error_square / std::max(target_square, 1e-30));
                        std::printf("HC_MATRIX_ERROR stage=%d rel_l2=%g max_abs=%g peak=%g\n", stage - 1, relative, maximum, peak);
                        if (buffer->host.dtype == RAD_I8) REQUIRE(maximum <= 2);
                        else { REQUIRE(relative <= 5e-3); REQUIRE(maximum <= 1e-2f * std::max(1.0f, peak)); }
                    }
                }
                const auto scratch_actual = scratch.read(compute.stream);
                REQUIRE(std::equal(scratch_actual.begin(), scratch_actual.begin() + size_t(rows) * 10240 * 2, normalized.begin()));
                RadEvent start{}, finish{};
                REQUIRE_EQ(rad_event_create(&start), RAD_OK); REQUIRE_EQ(rad_event_create(&finish), RAD_OK);
                for (int repeat = 0; repeat < 3; ++repeat) REQUIRE_EQ(plain(&args, compute.stream), RAD_OK);
                REQUIRE_EQ(rad_event_record(start, compute.stream), RAD_OK);
                for (int repeat = 0; repeat < 20; ++repeat) REQUIRE_EQ(plain(&args, compute.stream), RAD_OK);
                REQUIRE_EQ(rad_event_record(finish, compute.stream), RAD_OK); REQUIRE_EQ(rad_event_sync(finish), RAD_OK);
                float elapsed = 0;
                REQUIRE_EQ(rad_event_elapsed_ms(start, finish, &elapsed), RAD_OK);
                timings[variant] += elapsed * 25.0f;
                rad_event_destroy(start); rad_event_destroy(finish);
                if (!compare_matrix || !variant || rows > tolerant_rows) REQUIRE(output.read(compute.stream) == baseline[0]);
            }
            if (std::getenv("RAD_TP1_HC_STAGES")) {
                REQUIRE_EQ(setenv(switch_name, "1", 1), 0);
                float candidate[6]{};
                REQUIRE_EQ(profile(&args, compute.stream, candidate), RAD_OK);
                REQUIRE_EQ(setenv(switch_name, "0", 1), 0);
                std::printf("TP1_HC_STAGES switch=%s M=%d norm_us=%.3f down_us=%.3f up_us=%.3f mix_us=%.3f "
                    "inject_us=%.3f quant_us=%.3f\n", switch_name, rows, candidate[0], candidate[1], candidate[2],
                    candidate[3], candidate[4], candidate[5]);
            }
            std::printf("TP1_HC_AB switch=%s M=%d baseline_us=%.3f candidate_us=%.3f speedup=%.3f check=%s\n",
                switch_name, rows, timings[0], timings[1], timings[0] / timings[1],
                compare_matrix && rows <= tolerant_rows ? "preset-tolerance" : "bit-exact");
        }
    }
    if (compare_packed) {
        if (packed_present) setenv(switch_name, saved_packed.c_str(), 1); else unsetenv(switch_name);
    }
}

TEST(r3d_tp1_u5_actual_aql) {
    if (!std::getenv("RAD_TP1_U5_CHECK")) SKIP("optional actual full U5 decode check");
    using namespace iqr_output_boundary;
    RadDeviceProps props{};
    REQUIRE_EQ(rad_dev_props(0, &props), RAD_OK);
    REQUIRE(!props.is_host_backend);
    REQUIRE_EQ(std::strcmp(props.arch, "gfx1100"), 0);
    Library native("libr3d");
    const auto* kernel = moe_boundary::registered(native, "gemm_nt_q_iqr_decode");
    const char* selected = std::getenv("RAD_TP1_U5_N");
    const int columns = selected ? std::atoi(selected) : 16384, pitch = columns + 17;
    REQUIRE(columns == 16384 || columns == 12288);
    Fixture fixture(kernel, columns, 2560, true, columns == 16384);
    Queue compute(0);
    Buffer weight(RAD_I32, {int64_t((fixture.weight.size() + 3) / 4)}, RAD_MEM_DEVICE);
    Buffer weight_scale(RAD_I32, {int64_t((fixture.scale.size() + 3) / 4)}, RAD_MEM_DEVICE);
    std::memcpy(weight.host.data, fixture.weight.data(), fixture.weight.size());
    std::memcpy(weight_scale.host.data, fixture.scale.data(), fixture.scale.size());
    weight.push(compute.stream); weight_scale.push(compute.stream);
    const char* original = std::getenv("RAD_IQR_U5_WAVE");
    const bool present = original != nullptr;
    const std::string saved = original ? original : "";
    for (int rows : {1, 2, 3}) for (int pattern = 0; pattern < (rows == 3 ? 1 : 7); ++pattern) {
        auto header = *reinterpret_cast<const r3d_iqr::Header*>(fixture.weight.data());
        if (pattern == 4) header.state_bits = 15;
        if (pattern == 5) header.run_rows = 128;
        if (pattern == 6) { header.bits = 4; header.run_rows = 128; }
        std::memcpy(weight.host.data, &header, sizeof(header)); weight.push(compute.stream);
        Buffer activation(RAD_I8, {rows, 2560}, RAD_MEM_DEVICE, 2573);
        Buffer scales(RAD_F32, {rows, 20}, RAD_MEM_DEVICE, 25);
        Buffer output(RAD_BF16, {rows, columns}, RAD_MEM_DEVICE, pitch);
        for (int row = 0; row < rows; ++row) {
            for (int position = 0; position < 2560; ++position)
                activation.data<int8_t>()[row * 2573 + position] = pattern
                    ? (position & 1 ? -128 : 127) : int8_t((position * 47 + row * 19) % 256 - 128);
            for (int group = 0; group < 20; ++group) {
                float factor = float(group % 7 + row + 1) * 0x1p-13f;
                if (pattern == 2) factor = group & 1 ? -factor : 0.0f;
                if (pattern == 3) {
                    const uint32_t bits[]{0u, 0x80000000u, 1u, 0x7f7fffffu, 0x7f800000u, 0x7fc01234u};
                    factor = __builtin_bit_cast(float, bits[(row + group) % 6]);
                }
                scales.data<float>()[row * 25 + group] = factor;
            }
        }
        activation.push(compute.stream); scales.push(compute.stream); output.push(compute.stream);
        RadTensor tensors[]{activation.device, scales.device,
            matrix(weight.device.data, r3d_iqr::weight_dtype, columns, 2560),
            matrix(weight_scale.device.data, r3d_iqr::scale_dtype, columns, 20), output.device};
        RadParam params[]{RAD_INT("M", std::max(rows, 2)), RAD_INT("N", columns), RAD_INT("K", 2560),
            RAD_INT("group", 128), RAD_INT("w_rb", 1), RAD_INT("impl", 2), RAD_STR("dtype", "i8a8h")};
        RadArgs args{}; args.t = tensors; args.n_t = 5; args.p = params; args.n_p = 7; args.world_size = 1;
        REQUIRE_EQ(setenv("RAD_IQR_U5_WAVE", "0", 1), 0);
        REQUIRE_EQ(kernel->launch(&args, compute.stream), RAD_OK);
        const auto reference = output.read(compute.stream);
        for (int row = 0; row < rows; ++row) for (int column = 0; column < columns; ++column)
            if (pattern < 3 && (column % 256 < 4 || column % 256 >= 252)) {
                const uint16_t expected = oracle(fixture, activation, scales, row, column);
                REQUIRE_EQ(std::memcmp(reference.data() + 64 + (size_t(row) * pitch + column) * 2, &expected, 2), 0);
            }
        for (int row = 0; row < rows; ++row)
            std::memcpy(output.expected.data() + 64 + size_t(row) * pitch * 2,
                reference.data() + 64 + size_t(row) * pitch * 2, columns * 2);
        REQUIRE(reference == output.expected);
        float timings[2]{};
        for (int variant : {0, 1, 1, 0}) {
            REQUIRE_EQ(setenv("RAD_IQR_U5_WAVE", variant ? "1" : "0", 1), 0);
            output.push(compute.stream);
            REQUIRE_EQ(kernel->launch(&args, compute.stream), RAD_OK);
            REQUIRE(output.read(compute.stream) == reference);
            if (pattern < 2 && rows <= 2 && std::getenv("RAD_TP1_U5_BENCH")) {
                RadEvent start{}, finish{};
                REQUIRE_EQ(rad_event_create(&start), RAD_OK);
                REQUIRE_EQ(rad_event_create(&finish), RAD_OK);
                for (int warmup = 0; warmup < 3; ++warmup) REQUIRE_EQ(kernel->launch(&args, compute.stream), RAD_OK);
                REQUIRE_EQ(rad_event_record(start, compute.stream), RAD_OK);
                for (int repeat = 0; repeat < 20; ++repeat) REQUIRE_EQ(kernel->launch(&args, compute.stream), RAD_OK);
                REQUIRE_EQ(rad_event_record(finish, compute.stream), RAD_OK);
                REQUIRE_EQ(rad_event_sync(finish), RAD_OK);
                float elapsed = 0;
                REQUIRE_EQ(rad_event_elapsed_ms(start, finish, &elapsed), RAD_OK);
                timings[variant] += elapsed * 25.0f;
                rad_event_destroy(start); rad_event_destroy(finish);
                REQUIRE(output.read(compute.stream) == reference);
            }
        }
        REQUIRE(activation.read(compute.stream) == activation.expected);
        REQUIRE(scales.read(compute.stream) == scales.expected);
        REQUIRE(weight.read(compute.stream) == weight.expected);
        REQUIRE(weight_scale.read(compute.stream) == weight_scale.expected);
        std::printf("TP1_U5_AB M=%d pattern=%d exact_values=%d baseline_us=%.3f candidate_us=%.3f speedup=%.3f\n",
            rows, pattern, rows * columns, timings[0], timings[1], timings[1] ? timings[0] / timings[1] : 0);
    }
    if (present) setenv("RAD_IQR_U5_WAVE", saved.c_str(), 1); else unsetenv("RAD_IQR_U5_WAVE");
}

TEST(r3d_tp1_head_wave_edges) {
    using namespace joined_head_test;
    if (!device_available()) SKIP("gfx1100 required");
    Api api;
    Queue compute(0);
    const char* original = std::getenv("RAD_LOGITS_WAVE");
    const bool present = original != nullptr;
    const std::string saved = original ? original : "";
    for (int kind = 0; kind < 4; ++kind) {
        Fixture fixture(api, 7, 2560, kind == 1 ? RAD_F8E4M3 : RAD_I8,
            kind == 2 ? RAD_F32 : RAD_BF16, 1, 128);
        Storage weight(fixture.joined.size(), RAD_MEM_DEVICE);
        std::memcpy(weight.expected.data() + 64, fixture.joined.data(), fixture.joined.size());
        weight.push(compute.stream);
        const int rows = kind == 3 ? 2 : 1;
        Buffer input(RAD_BF16, {rows, 2560}, RAD_MEM_DEVICE, 2576);
        Buffer output(RAD_F32, {rows, 7}, RAD_MEM_DEVICE, 16);
        for (int pattern = 0; pattern < 5; ++pattern) {
            for (int row = 0; row < rows; ++row) for (int position = 0; position < 2560; ++position) {
                uint16_t bits = rad_f32_to_bf16(float((position * 17) % 31 - 15) / 32.0f);
                if (position == 0 && pattern) bits = pattern == 1 ? 0x7f80 : pattern == 2 ? 0x7fc1
                    : pattern == 3 ? 0x0001 : 0x8000;
                input.data<uint16_t>()[row * 2576 + position] = bits;
            }
            input.push(compute.stream);
            auto descriptor = matrix(weight.device(), fixture.stored.dtype, 7, 2560);
            descriptor.rank = 1; descriptor.shape[0] = fixture.stored.bytes; descriptor.stride[0] = 1;
            RadTensor tensors[]{input.device, descriptor, output.device};
            RadParam params[]{RAD_INT("M", 2), RAD_INT("n_vocab", 7), RAD_INT("n_embd", 2560)};
            RadArgs args{}; args.t = tensors; args.n_t = 3; args.p = params; args.n_p = 3;
            std::vector<uint8_t> reference;
            for (int variant = 0; variant < 2; ++variant) {
                REQUIRE_EQ(setenv("RAD_LOGITS_WAVE", variant ? "1" : "0", 1), 0);
                output.push(compute.stream);
                REQUIRE_EQ(api.launch(&args, compute.stream), RAD_OK);
                const auto actual = output.read(compute.stream);
                auto expected = output.expected;
                for (int row = 0; row < rows; ++row) {
                    std::copy_n(actual.data() + 64 + row * 64, 28, expected.data() + 64 + row * 64);
                    for (int column = 0; column < 7; ++column) if (variant) {
                        float value = 0, target = 0;
                        std::memcpy(&value, actual.data() + 64 + row * 64 + column * 4, 4);
                        std::memcpy(&target, reference.data() + 64 + row * 64 + column * 4, 4);
                        if (!std::isfinite(target)) REQUIRE((std::isnan(target) && std::isnan(value)) || target == value);
                        else if (kind) REQUIRE_EQ(std::memcmp(&value, &target, 4), 0);
                        else REQUIRE(std::fabs(value - target) <= 1e-4f * std::max(1.0f, std::fabs(target)));
                    }
                }
                REQUIRE(actual == expected);
                if (!variant) reference = actual;
            }
            REQUIRE(input.read(compute.stream) == input.expected);
            REQUIRE(weight.read(compute.stream) == weight.expected);
        }
    }
    if (present) setenv("RAD_LOGITS_WAVE", saved.c_str(), 1); else unsetenv("RAD_LOGITS_WAVE");
}

TEST(r3d_tp1_head_actual_aql) {
    using namespace joined_head_test;
    RadDeviceProps props{};
    REQUIRE_EQ(rad_dev_props(0, &props), RAD_OK);
    if (props.is_host_backend) {
        REQUIRE(!std::getenv("RAD_R3D_REQUIRE_DEVICE"));
        SKIP("explicit gfx1100 card run required");
    }
    REQUIRE_EQ(std::strcmp(props.arch, "gfx1100"), 0);
    REQUIRE_EQ(rad_dev_set(0), RAD_OK);
    Api api;
    Model model("/home/zihaomu/bigssd2/radiance_workspace/models/qwen3.8-next-flash-iqr-v4.rad");
    Fixture fixture(api, 248320, 2560, RAD_I8, RAD_BF16, 1, 128);
    bool found = false;
    for (const auto& entry : model.entries) {
        if (model.name(entry) != "output.weight") continue;
        REQUIRE_EQ(entry.shape[0], fixture.columns);
        REQUIRE_EQ(entry.shape[1], fixture.reduction);
        REQUIRE_EQ(entry.n_planes, 2u);
        for (unsigned index = 0; index < 2; ++index) {
            const auto& location = model.planes[entry.plane_first + index];
            auto& destination = index ? fixture.scale : fixture.codes;
            REQUIRE_EQ(location.bytes, destination.size());
            model.read(location.offset, destination.data(), destination.size());
        }
        found = true;
    }
    REQUIRE(found);
    fixture.pack(api);
    Queue compute(0);
    Storage weight(fixture.joined.size(), RAD_MEM_DEVICE);
    std::memcpy(weight.expected.data() + 64, fixture.joined.data(), fixture.joined.size());
    Buffer input(RAD_BF16, {1, 2560}, RAD_MEM_DEVICE, 2576);
    Buffer output(RAD_F32, {1, 248320}, RAD_MEM_DEVICE, 248336);
    for (int position = 0; position < 2560; ++position)
        input.data<uint16_t>()[position] = rad_f32_to_bf16(float((position * 47) % 257 - 128) / 64.0f);
    weight.push(compute.stream); input.push(compute.stream); output.push(compute.stream);
    auto descriptor = matrix(weight.device(), fixture.stored.dtype, 248320, 2560);
    descriptor.rank = 1; descriptor.shape[0] = fixture.stored.bytes; descriptor.stride[0] = 1;
    RadTensor tensors[]{input.device, descriptor, output.device};
    RadParam params[]{RAD_INT("M", 64), RAD_INT("n_vocab", 248320), RAD_INT("n_embd", 2560), RAD_INT("impl", 0)};
    RadArgs args{}; args.t = tensors; args.n_t = 3; args.p = params; args.n_p = 4; args.world_size = 1;
    const bool compare_staged = std::getenv("RAD_TP1_HEAD_AB") != nullptr;
    const bool compare_wave = std::getenv("RAD_TP1_HEAD_WAVE_AB") != nullptr;
    const char* original_wave = std::getenv("RAD_LOGITS_WAVE");
    const bool wave_present = original_wave != nullptr;
    const std::string saved_wave = original_wave ? original_wave : "";
    const char* original_staged = std::getenv("RAD_LOGITS_STAGED");
    const bool staged_present = original_staged != nullptr;
    const std::string saved_staged = original_staged ? original_staged : "";
    if (compare_staged) REQUIRE_EQ(setenv("RAD_LOGITS_STAGED", "0", 1), 0);
    if (compare_wave) REQUIRE_EQ(setenv("RAD_LOGITS_WAVE", "0", 1), 0);
    REQUIRE_EQ(api.launch(&args, compute.stream), RAD_OK);
    const auto actual = output.read(compute.stream);
    for (int column = 0; column < 248320; ++column) {
        float value; std::memcpy(&value, actual.data() + 64 + size_t(column) * 4, 4);
        REQUIRE(std::isfinite(value));
    }
    for (int column : {0, 1, 127, 128, 124158, 124159, 124160, 124161, 248191, 248192, 248318, 248319}) {
        float expected = 0;
        for (int position = 0; position < 2560; ++position) {
            const float activation = rad_bf16_to_f32(input.data<uint16_t>()[position]);
            const float decoded = multiply(decode(fixture.codes[size_t(column) * 2560 + position], RAD_I8), fixture.factor(column, position));
            expected = add(expected, multiply(activation, decoded));
        }
        REQUIRE_EQ(std::memcmp(actual.data() + 64 + size_t(column) * 4, &expected, 4), 0);
    }
    for (size_t index = 0; index < actual.size(); ++index)
        if (index < 64 || index >= 64 + size_t(248320) * 4) REQUIRE_EQ(actual[index], output.expected[index]);
    REQUIRE(weight.read(compute.stream) == weight.expected);
    REQUIRE(input.read(compute.stream) == input.expected);
    if (compare_staged) {
        REQUIRE_EQ(setenv("RAD_LOGITS_STAGED", "1", 1), 0);
        output.push(compute.stream);
        REQUIRE_EQ(api.launch(&args, compute.stream), RAD_OK);
        REQUIRE(output.read(compute.stream) == actual);
        REQUIRE(weight.read(compute.stream) == weight.expected);
        REQUIRE(input.read(compute.stream) == input.expected);
        if (staged_present) setenv("RAD_LOGITS_STAGED", saved_staged.c_str(), 1);
        else unsetenv("RAD_LOGITS_STAGED");
        std::printf("PASS TP1_HEAD_STAGED_AB exact_logits=248320 guards=all\n");
    }
    if (compare_wave) {
        REQUIRE_EQ(setenv("RAD_LOGITS_STAGED", "1", 1), 0);
        for (int pattern = 0; pattern < 4; ++pattern) {
            for (int position = 0; position < 2560; ++position) {
                const float value = pattern == 0 ? float((position * 47) % 257 - 128) / 64.0f
                    : pattern == 1 ? (position & 1 ? -1.0f : 1.0f)
                    : pattern == 2 ? (position == 127 || position == 2559 ? 4.0f : 0.0f) : 0.0f;
                input.data<uint16_t>()[position] = rad_f32_to_bf16(value);
            }
            input.push(compute.stream);
            std::vector<std::vector<uint8_t>> results;
            float timings[2]{};
            for (int variant : {0, 1, 1, 0}) {
                REQUIRE_EQ(setenv("RAD_LOGITS_WAVE", variant ? "1" : "0", 1), 0);
                output.push(compute.stream);
                REQUIRE_EQ(api.launch(&args, compute.stream), RAD_OK);
                auto result = output.read(compute.stream);
                for (size_t index = 0; index < result.size(); ++index)
                    if (index < 64 || index >= 64 + size_t(248320) * 4)
                        REQUIRE_EQ(result[index], output.expected[index]);
                if (results.size() < 2) results.push_back(result);
                else REQUIRE(result == results[variant]);
                if (pattern == 0) {
                    RadEvent start{}, finish{};
                    REQUIRE_EQ(rad_event_create(&start), RAD_OK); REQUIRE_EQ(rad_event_create(&finish), RAD_OK);
                    for (int repeat = 0; repeat < 3; ++repeat) REQUIRE_EQ(api.launch(&args, compute.stream), RAD_OK);
                    REQUIRE_EQ(rad_event_record(start, compute.stream), RAD_OK);
                    for (int repeat = 0; repeat < 10; ++repeat) REQUIRE_EQ(api.launch(&args, compute.stream), RAD_OK);
                    REQUIRE_EQ(rad_event_record(finish, compute.stream), RAD_OK); REQUIRE_EQ(rad_event_sync(finish), RAD_OK);
                    float elapsed = 0;
                    REQUIRE_EQ(rad_event_elapsed_ms(start, finish, &elapsed), RAD_OK);
                    timings[variant] += elapsed * 50.0f;
                    rad_event_destroy(start); rad_event_destroy(finish);
                    REQUIRE(output.read(compute.stream) == result);
                }
            }
            std::vector<float> reference(248320), candidate(248320);
            std::memcpy(reference.data(), results[0].data() + 64, reference.size() * 4);
            std::memcpy(candidate.data(), results[1].data() + 64, candidate.size() * 4);
            double error_square = 0, reference_square = 0;
            float maximum_error = 0, peak = 0;
            for (int column = 0; column < 248320; ++column) {
                REQUIRE(std::isfinite(candidate[column]));
                const double error = double(candidate[column]) - reference[column];
                error_square += error * error;
                reference_square += double(reference[column]) * reference[column];
                maximum_error = std::max(maximum_error, float(std::fabs(error)));
                peak = std::max(peak, std::fabs(reference[column]));
            }
            const double relative_l2 = std::sqrt(error_square / std::max(reference_square, 1e-30));
            REQUIRE(relative_l2 <= 1e-5);
            REQUIRE(maximum_error <= 1e-4f * std::max(1.0f, peak));
            std::vector<int> strict_order(248320), wave_order(248320);
            for (int column = 0; column < 248320; ++column) strict_order[column] = wave_order[column] = column;
            auto rank = [&](std::vector<int>& order, const std::vector<float>& values) {
                std::partial_sort(order.begin(), order.begin() + 10, order.end(), [&](int left, int right) {
                    return values[left] == values[right] ? left < right : values[left] > values[right];
                });
            };
            rank(strict_order, reference); rank(wave_order, candidate);
            const float margin = reference[strict_order[0]] - reference[strict_order[1]];
            if (margin > 2 * maximum_error) REQUIRE_EQ(strict_order[0], wave_order[0]);
            int overlap = 0;
            for (int index = 0; index < 10; ++index) {
                REQUIRE(reference[wave_order[index]] + 2 * maximum_error >= reference[strict_order[9]]);
                overlap += std::find(strict_order.begin(), strict_order.begin() + 10, wave_order[index]) != strict_order.begin() + 10;
            }
            for (int column : {0, 127, 128, 124159, 124160, 248319}) {
                double expected = 0, absolute_sum = 0;
                for (int position = 0; position < 2560; ++position) {
                    const double product = double(rad_bf16_to_f32(input.data<uint16_t>()[position]))
                        * decode(fixture.codes[size_t(column) * 2560 + position], RAD_I8) * fixture.factor(column, position);
                    expected += product; absolute_sum += std::fabs(product);
                }
                REQUIRE(std::fabs(double(candidate[column]) - expected) <= 1e-5 * std::max(1.0, absolute_sum));
            }
            REQUIRE(weight.read(compute.stream) == weight.expected);
            REQUIRE(input.read(compute.stream) == input.expected);
            std::printf("TP1_HEAD_WAVE_AB pattern=%d relative_l2=%.9g max_abs=%.9g peak=%.9g top10_overlap=%d top1_margin=%.9g strict_us=%.3f wave_us=%.3f\n",
                pattern, relative_l2, maximum_error, peak, overlap, margin, timings[0], timings[1]);
        }
        if (wave_present) setenv("RAD_LOGITS_WAVE", saved_wave.c_str(), 1); else unsetenv("RAD_LOGITS_WAVE");
        if (staged_present) setenv("RAD_LOGITS_STAGED", saved_staged.c_str(), 1); else unsetenv("RAD_LOGITS_STAGED");
    }
    if (std::getenv("RAD_TP1_HEAD_STREAM_AB")) {
        const char* original_stream = std::getenv("RAD_LOGITS_STREAM");
        const bool stream_present = original_stream != nullptr;
        const std::string saved_stream = original_stream ? original_stream : "";
        REQUIRE_EQ(setenv("RAD_LOGITS_WAVE", "0", 1), 0);
        for (int pattern = 0; pattern < 4; ++pattern) {
            for (int position = 0; position < 2560; ++position) {
                const float value = pattern == 0 ? float((position * 47) % 257 - 128) / 64.0f
                    : pattern == 1 ? (position & 1 ? -1.0f : 1.0f)
                    : pattern == 2 ? (position == 127 || position == 2559 ? 4.0f : 0.0f)
                    : float(int(uint32_t(position) * 2654435761u >> 20) - 2048) * 0x1p-5f;
                input.data<uint16_t>()[position] = rad_f32_to_bf16(value);
            }
            input.push(compute.stream);
            std::vector<uint8_t> results[3];
            float timings[3]{};
            const char* staged_values[3]{"0", "1", "1"};
            const char* stream_values[3]{"0", "0", "1"};
            for (int variant = 0; variant < 3; ++variant) {
                REQUIRE_EQ(setenv("RAD_LOGITS_STAGED", staged_values[variant], 1), 0);
                REQUIRE_EQ(setenv("RAD_LOGITS_STREAM", stream_values[variant], 1), 0);
                output.push(compute.stream);
                REQUIRE_EQ(api.launch(&args, compute.stream), RAD_OK);
                results[variant] = output.read(compute.stream);
                RadEvent start{}, finish{};
                REQUIRE_EQ(rad_event_create(&start), RAD_OK); REQUIRE_EQ(rad_event_create(&finish), RAD_OK);
                for (int repeat = 0; repeat < 3; ++repeat) REQUIRE_EQ(api.launch(&args, compute.stream), RAD_OK);
                REQUIRE_EQ(rad_event_record(start, compute.stream), RAD_OK);
                for (int repeat = 0; repeat < 10; ++repeat) REQUIRE_EQ(api.launch(&args, compute.stream), RAD_OK);
                REQUIRE_EQ(rad_event_record(finish, compute.stream), RAD_OK); REQUIRE_EQ(rad_event_sync(finish), RAD_OK);
                REQUIRE_EQ(rad_event_elapsed_ms(start, finish, &timings[variant]), RAD_OK);
                timings[variant] *= 100.0f;
                rad_event_destroy(start); rad_event_destroy(finish);
                REQUIRE(output.read(compute.stream) == results[variant]);
            }
            REQUIRE(results[1] == results[0]);
            REQUIRE(results[2] == results[0]);
            REQUIRE(weight.read(compute.stream) == weight.expected);
            REQUIRE(input.read(compute.stream) == input.expected);
            std::printf("TP1_HEAD_STREAM_AB pattern=%d exact=1 strict_us=%.1f staged_us=%.1f stream_us=%.1f\n",
                pattern, timings[0], timings[1], timings[2]);
        }
        if (stream_present) setenv("RAD_LOGITS_STREAM", saved_stream.c_str(), 1); else unsetenv("RAD_LOGITS_STREAM");
        if (wave_present) setenv("RAD_LOGITS_WAVE", saved_wave.c_str(), 1); else unsetenv("RAD_LOGITS_WAVE");
        if (staged_present) setenv("RAD_LOGITS_STAGED", saved_staged.c_str(), 1); else unsetenv("RAD_LOGITS_STAGED");
    }
    std::printf("PASS TP1_HEAD_AQL N=248320 K=2560 finite=all sampled_exact=12 shard_boundary=124160 guards=all\n");
}

TEST(r3d_tp1_attention_actual_aql) {
    RadDeviceProps props{};
    REQUIRE_EQ(rad_dev_props(0, &props), RAD_OK);
    if (props.is_host_backend) {
        REQUIRE(!std::getenv("RAD_R3D_REQUIRE_DEVICE"));
        SKIP("explicit gfx1100 card run required");
    }
    REQUIRE_EQ(std::strcmp(props.arch, "gfx1100"), 0);
    REQUIRE_EQ(rad_dev_set(0), RAD_OK);
    Library native("libr3d"), reference("libref");
    const auto store = reinterpret_cast<RadLaunchFn>(dlsym(native.handle, "r3dh_attn_qsa_kv_store"));
    const auto attend = reinterpret_cast<RadLaunchFn>(dlsym(native.handle, "r3dh_attn_qsa_paged"));
    REQUIRE(store && attend);
    Queue compute(0);
    for (int heads : {12, 24}) {
        constexpr int rows = 3, pages = 3;
        const int keys = heads / 12;
        Buffer key(RAD_BF16, {rows, keys * 256}, RAD_MEM_DEVICE, keys * 256 + 7);
        Buffer value(RAD_BF16, {rows, keys * 256}, RAD_MEM_DEVICE, keys * 256 + 9);
        Buffer slots(RAD_I32, {rows}, RAD_MEM_DEVICE);
        Buffer cache(RAD_I8, {pages, keys, 4, 516}, RAD_MEM_DEVICE);
        Buffer query(RAD_BF16, {rows, heads * 256}, RAD_MEM_DEVICE, heads * 256 + 5);
        Buffer output(RAD_BF16, {rows, heads * 256}, RAD_MEM_DEVICE, heads * 256 + 11);
        Buffer table(RAD_I32, {1, pages}, RAD_MEM_DEVICE), used(RAD_I32, {1}, RAD_MEM_DEVICE);
        Buffer cu(RAD_I32, {2}, RAD_MEM_DEVICE);
        for (int row = 0; row < rows; ++row) {
            slots.data<int32_t>()[row] = 4 + row;
            for (int column = 0; column < keys * 256; ++column) {
                key.data<uint16_t>()[row * key.host.stride[0] + column] = rad_f32_to_bf16(float((column * 13 + row * 7) % 97 - 48) / 128.0f);
                value.data<uint16_t>()[row * value.host.stride[0] + column] = rad_f32_to_bf16(float((column * 17 + row * 11) % 101 - 50) / 64.0f + float(column / 256));
            }
            for (int column = 0; column < heads * 256; ++column)
                query.data<uint16_t>()[row * query.host.stride[0] + column] = rad_f32_to_bf16(float((column * 19 + row * 23) % 89 - 44) / 128.0f);
        }
        table.data<int32_t>()[0] = 1; table.data<int32_t>()[1] = 0; table.data<int32_t>()[2] = 2;
        used.data<int32_t>()[0] = rows;
        cu.data<int32_t>()[0] = 0; cu.data<int32_t>()[1] = rows;
        for (Buffer* buffer : {&key, &value, &slots, &cache, &query, &output, &table, &used, &cu}) buffer->push(compute.stream);
        RadParam store_params[]{RAD_INT("M", rows), RAD_INT("head_dim", 256), RAD_INT("n_head_kv", keys), RAD_INT("block_size", 4), RAD_STR("kv_dtype", "i8")};
        RadTensor host_store[]{key.host, value.host, slots.host, cache.host};
        RadTensor device_store[]{key.device, value.device, slots.device, cache.device};
        RadArgs args{}; args.t = host_store; args.n_t = 4; args.p = store_params; args.n_p = 5;
        REQUIRE_EQ(reference.ngram("kv_store")->launch(&args, nullptr), RAD_OK);
        args.t = device_store;
        REQUIRE_EQ(store(&args, compute.stream), RAD_OK);
        REQUIRE(cache.read(compute.stream) == cache.expected);
        RadParam params[]{RAD_INT("head_dim", 256), RAD_INT("n_head", heads), RAD_INT("gqa", 12), RAD_INT("block_size", 4),
            RAD_INT("causal", 1), RAD_INT("window", 0), RAD_STR("q_dtype", "bf16"), RAD_STR("kv_dtype", "i8"), RAD_F64("scale", 0.0625)};
        RadTensor host[]{query.host, cache.host, table.host, used.host, {}, {}, cu.host, output.host};
        RadTensor device[]{query.device, cache.device, table.device, used.device, {}, {}, cu.device, output.device};
        args.t = host; args.n_t = 8; args.p = params; args.n_p = int(sizeof(params) / sizeof(*params));
        REQUIRE_EQ(reference.ngram("attn_paged")->launch(&args, nullptr), RAD_OK);
        args.t = device;
        REQUIRE_EQ(attend(&args, compute.stream), RAD_OK);
        const auto saved_query = device[0];
        device[0].rank = 4;
        REQUIRE_EQ(attend(&args, compute.stream), RAD_E_SHAPE);
        device[0] = saved_query;
        device[0].rank = 3; device[0].shape[1] = INT64_MAX; device[0].shape[2] = INT64_MAX;
        REQUIRE_EQ(attend(&args, compute.stream), RAD_E_SHAPE);
        device[0] = saved_query;
        for (Buffer* buffer : {&key, &value, &slots, &cache, &query, &output, &table, &used, &cu})
            REQUIRE(buffer->read(compute.stream) == buffer->expected);
    }
    std::printf("PASS TP1_ATTENTION_AQL qheads=12,24 kvheads=1,2 store=paged guards=all exact=1\n");
}

TEST(r3d_tp1_moe_actual_aql) {
    RadDeviceProps props{};
    REQUIRE_EQ(rad_dev_props(0, &props), RAD_OK);
    if (props.is_host_backend) {
        REQUIRE(!std::getenv("RAD_R3D_REQUIRE_DEVICE"));
        SKIP("explicit gfx1100 card run required");
    }
    REQUIRE_EQ(std::strcmp(props.arch, "gfx1100"), 0);
    REQUIRE_EQ(rad_dev_set(0), RAD_OK);
    Library library("libr3d");
    Queue compute(0);
    Model model("/home/zihaomu/bigssd2/radiance_workspace/models/qwen3.8-next-flash-iqr-v4.rad");
    uint64_t launches = 0, rejects = 0, compared = 0;
    for (int layer : {0, 34, 44, 47})
        moe_boundary::chain(model, library, layer, 0, compute.stream, launches, rejects, compared, true);
    for (int layer : {34, 44, 47})
        moe_boundary::protected_pair(model, library, layer, 0, compute.stream, launches, rejects, compared, true);
    std::printf("PASS TP1_MOE_AQL launches=%llu rejects=%llu exact_values=%llu\n",
        (unsigned long long)launches, (unsigned long long)rejects, (unsigned long long)compared);
}

TEST(r3d_tp1_gated_quant_actual_aql) {
    RadDeviceProps props{};
    REQUIRE_EQ(rad_dev_props(0, &props), RAD_OK);
    if (props.is_host_backend) {
        REQUIRE(!std::getenv("RAD_R3D_REQUIRE_DEVICE"));
        SKIP("explicit gfx1100 card run required");
    }
    REQUIRE_EQ(std::strcmp(props.arch, "gfx1100"), 0);
    REQUIRE_EQ(rad_dev_set(0), RAD_OK);
    Library native("libr3d"), reference("libref");
    const auto count = reinterpret_cast<int (*)()>(dlsym(native.handle, "rad_kernel_count"));
    const auto at = reinterpret_cast<const RadKernelInfo* (*)(int)>(dlsym(native.handle, "rad_kernel_at"));
    REQUIRE(count && at);
    const auto oracle = reinterpret_cast<RadLaunchFn>(dlsym(reference.handle, "ref_had_quant_act_i8g"));
    REQUIRE(oracle);
    Queue compute(0);
    uint64_t fixtures = 0, bytes = 0;
    for (int width : {256, 384, 640}) {
        const auto name = std::string("model_gated_had_") + std::to_string(width) + "_strict";
        const RadKernelInfo* selected = nullptr;
        for (int index = 0; index < count(); ++index)
            if (!std::strcmp(at(index)->name, name.c_str())) selected = at(index);
        REQUIRE(selected && selected->launch);
        for (int rows : {1, 3, 64})
            for (int padding : {0, 7})
                for (bool product : {false, true})
                    for (int pattern : {0, 1, 2}) {
                        Buffer input(RAD_BF16, {rows + 2, width * 2}, RAD_MEM_DEVICE, width * 2 + padding);
                        Buffer codes(RAD_I8, {rows + 2, width}, RAD_MEM_DEVICE, width + padding);
                        Buffer scales(RAD_F32, {rows + 2, width / 128}, RAD_MEM_DEVICE, width / 128 + padding);
                        Buffer output(RAD_BF16, {rows + 2, width}, RAD_MEM_DEVICE, width + padding);
                        for (Buffer* buffer : {&input, &codes, &scales, &output})
                            buffer->host.shape[0] = buffer->device.shape[0] = rows;
                        for (int row = 0; row < rows; ++row)
                            for (int column = 0; column < width * 2; ++column) {
                                const float value = pattern == 1 ? 0.0f :
                                    float((row * 19 + column * 23) % 131 - 65) * (pattern == 2 ? 2.0f : 0.03125f);
                                input.data<uint16_t>()[row * input.host.stride[0] + column] = rad_f32_to_bf16(value);
                            }
                        for (Buffer* buffer : {&input, &codes, &scales, &output}) buffer->push(compute.stream);
                        RadParam params[]{RAD_INT("M", rows), RAD_INT("n", width), RAD_INT("group", 128),
                            RAD_STR("dtype", "bf16"), RAD_STR("act", "silu")};
                        std::vector<uint16_t> products(size_t(rows) * width);
                        for (int row = 0; row < rows; ++row)
                            for (int column = 0; column < width; ++column) {
                                const auto* source = input.data<uint16_t>() + row * input.host.stride[0];
                                const float gate = rad_bf16_to_f32(source[column]);
                                const float up = rad_bf16_to_f32(source[width + column]);
                                const float sigmoid = 1.0f / (1.0f + std::exp(-gate));
                                const float activation = joined_head_test::multiply(gate, sigmoid);
                                const auto value = rad_f32_to_bf16(joined_head_test::multiply(activation, up));
                                products[size_t(row) * width + column] = value;
                                if (product) output.data<uint16_t>()[row * output.host.stride[0] + column] = value;
                            }
                        RadTensor host[]{joined_head_test::matrix(products.data(), RAD_BF16, rows, width),
                            codes.host, scales.host};
                        RadTensor device[]{input.device, codes.device, scales.device, output.device};
                        RadArgs args{};
                        args.t = host; args.n_t = 3;
                        args.p = params; args.n_p = int(sizeof(params) / sizeof(*params));
                        REQUIRE_EQ(oracle(&args, nullptr), RAD_OK);
                        args.t = device; args.n_t = product ? 4 : 3;
                        REQUIRE_EQ(selected->launch(&args, compute.stream), RAD_OK);
                        REQUIRE_EQ(rad_stream_sync(compute.stream), RAD_OK);
                        for (Buffer* buffer : {&input, &codes, &scales, &output}) {
                            const auto actual = buffer->read(compute.stream);
                            REQUIRE_EQ(std::memcmp(actual.data(), buffer->expected.data(), actual.size()), 0);
                            bytes += actual.size();
                        }
                        ++fixtures;
                    }
    }
    std::printf("PASS TP1_GATED_QUANT_AQL widths=256,384,640 fixtures=%llu exact_bytes=%llu guards=all\n",
        (unsigned long long)fixtures, (unsigned long long)bytes);
}

TEST(r3d_tp1_router_wave) {
    using namespace iqr_output_boundary;
    RadDeviceProps props{};
    REQUIRE_EQ(rad_dev_props(0, &props), RAD_OK);
    if (props.is_host_backend) SKIP("gfx1100 required");
    Library native("libr3d");
    auto launch = reinterpret_cast<RadLaunchFn>(dlsym(native.handle, "r3dh_qwen_ops_router_gemm"));
    REQUIRE(launch);
    Queue compute(0);
    const char* original = std::getenv("RAD_ROUTER_WAVE");
    const bool present = original != nullptr;
    const std::string saved = original ? original : "";
    uint32_t state = 20261119u;
    auto random = [&] { state = state * 1664525u + 1013904223u; return state >> 8; };
    int cases = 0;
    for (int columns : {1, 96, 512}) for (int rows : {1, 2, 17}) for (int pattern = 0; pattern < 3; ++pattern) {
        Buffer input(RAD_BF16, {rows, 2560}, RAD_MEM_DEVICE);
        Buffer weight(RAD_BF16, {columns, 2560}, RAD_MEM_DEVICE);
        Buffer output(RAD_BF16, {rows, columns}, RAD_MEM_DEVICE);
        auto value = [&](bool zero) -> uint16_t {
            if (zero) return 0;
            const uint32_t bits = random();
            const uint32_t exponent = pattern == 2 ? 1 + bits % 253 : 118 + bits % 16;
            return uint16_t(((bits >> 9) & 0x8000u) | (exponent << 7) | ((bits >> 12) & 0x7fu));
        };
        for (int index = 0; index < rows * 2560; ++index) input.data<uint16_t>()[index] = value(pattern == 1);
        for (int index = 0; index < columns * 2560; ++index) weight.data<uint16_t>()[index] = value(false);
        for (Buffer* buffer : {&input, &weight, &output}) buffer->push(compute.stream);
        RadTensor tensors[]{input.device, weight.device, output.device};
        RadParam params[]{RAD_INT("M", rows), RAD_INT("N", columns), RAD_INT("K", 2560), RAD_STR("dtype", "bf16")};
        RadArgs args{}; args.t = tensors; args.n_t = 3; args.p = params; args.n_p = 4; args.world_size = 1;
        REQUIRE_EQ(setenv("RAD_ROUTER_WAVE", "0", 1), 0);
        REQUIRE_EQ(launch(&args, compute.stream), RAD_OK);
        const auto reference = output.read(compute.stream);
        REQUIRE_EQ(setenv("RAD_ROUTER_WAVE", "1", 1), 0);
        output.push(compute.stream);
        REQUIRE_EQ(launch(&args, compute.stream), RAD_OK);
        REQUIRE(output.read(compute.stream) == reference);
        REQUIRE(input.read(compute.stream) == input.expected);
        REQUIRE(weight.read(compute.stream) == weight.expected);
        ++cases;
    }
    if (present) setenv("RAD_ROUTER_WAVE", saved.c_str(), 1); else unsetenv("RAD_ROUTER_WAVE");
    std::printf("PASS TP1_ROUTER_WAVE cases=%d shapes=N1,96,512 rows=1,2,17 patterns=normal,zero,wide exact=all\n", cases);
}

TEST(r3d_tp2_router_reduce_policy) {
    RadDeviceProps props{};
    REQUIRE_EQ(rad_dev_props(0, &props), RAD_OK);
    if (props.is_host_backend) SKIP("gfx1100 required");
    REQUIRE_EQ(rad_dev_set(0), RAD_OK);
    Library native("libr3d");
    auto launch = reinterpret_cast<RadLaunchFn>(dlsym(native.handle, "r3dh_qwen_ops_router_gemm"));
    REQUIRE(launch);
    Queue compute(0);
    struct SavedEnv {
        const char* name;
        bool present;
        std::string value;
        explicit SavedEnv(const char* key) : name(key), present(std::getenv(key) != nullptr),
            value(present ? std::getenv(key) : "") {}
        ~SavedEnv() { if (present) setenv(name, value.c_str(), 1); else unsetenv(name); }
    } precision("RAD_PRECISION"), reduce("RAD_ROUTER_REDUCE"), wave("RAD_ROUTER_WAVE");
    REQUIRE_EQ(setenv(wave.name, "1", 1), 0);
    auto check = [&](int world, int rows, int columns, const char* mode, const char* override_value, bool fast) {
        REQUIRE_EQ(mode ? setenv(precision.name, mode, 1) : unsetenv(precision.name), 0);
        REQUIRE_EQ(override_value ? setenv(reduce.name, override_value, 1) : unsetenv(reduce.name), 0);
        Buffer input(RAD_BF16, {rows, 2560}, RAD_MEM_DEVICE);
        Buffer weights(RAD_BF16, {columns, 2560}, RAD_MEM_DEVICE);
        Buffer output(RAD_BF16, {rows, columns}, RAD_MEM_DEVICE);
        auto fill = [](uint16_t* values, uint32_t state) {
            for (int position = 0; position < 2560; ++position) {
                state = state * 1664525u + 1013904223u;
                values[position] = uint16_t(((state >> 16) & 0x8000u) |
                    ((119u + ((state >> 8) & 15u)) << 7) | (state & 127u));
            }
        };
        for (int row = 0; row < rows; ++row) fill(input.data<uint16_t>() + row * 2560, 74129u);
        for (int column = 0; column < columns; ++column)
            fill(weights.data<uint16_t>() + column * 2560, (column & 1) ? 2691052478u : 3814312059u);
        for (Buffer* buffer : {&input, &weights, &output}) buffer->push(compute.stream);
        RadTensor tensors[]{input.device, weights.device, output.device};
        RadParam params[]{RAD_INT("M", rows), RAD_INT("N", columns), RAD_INT("K", 2560), RAD_STR("dtype", "bf16")};
        RadArgs args{}; args.t = tensors; args.n_t = 3; args.p = params; args.n_p = 4; args.world_size = world;
        REQUIRE_EQ(launch(&args, compute.stream), RAD_OK);
        for (int row = 0; row < rows; ++row) for (int column = 0; column < columns; ++column) {
            // Odd columns round to 0x489f without midpoint replay; even columns distinguish dispatch modes.
            output.data<uint16_t>()[row * columns + column] = (column & 1) ? 0x48a0 : fast ? 0xc4ff : 0xc4fe;
        }
        const auto actual = output.read(compute.stream);
        if (actual != output.expected) {
            const auto mismatch = std::mismatch(actual.begin(), actual.end(), output.expected.begin());
            const size_t offset = size_t(mismatch.first - actual.begin()) & ~size_t(1);
            uint16_t got{}, want{};
            std::memcpy(&got, actual.data() + offset, sizeof(got));
            std::memcpy(&want, output.expected.data() + offset, sizeof(want));
            std::fprintf(stderr, "router policy TP=%d M=%d N=%d precision=%s reduce=%s offset=%zu got=%04x want=%04x\n",
                         world, rows, columns, mode ? mode : "unset", override_value ? override_value : "unset", offset, got, want);
        }
        REQUIRE(actual == output.expected);
        REQUIRE(input.read(compute.stream) == input.expected);
        REQUIRE(weights.read(compute.stream) == weights.expected);
    };
    check(2, 1, 512, nullptr, nullptr, true);
    check(2, 1, 512, "tolerance", "0", false);
    check(2, 1, 512, "exact", "1", false);
    check(1, 1, 512, "tolerance", "1", false);
    check(2, 2, 512, "tolerance", "1", false);
    check(2, 1, 1, "tolerance", "1", false);
}

TEST(r3d_tp1_ple_gate_parallel) {
    using namespace iqr_output_boundary;
    RadDeviceProps props{};
    REQUIRE_EQ(rad_dev_props(0, &props), RAD_OK);
    if (props.is_host_backend) SKIP("gfx1100 required");
    Library native("libr3d");
    auto launch = reinterpret_cast<RadLaunchFn>(dlsym(native.handle, "r3dh_mtp_ple_ple_gate"));
    REQUIRE(launch);
    Queue compute(0);
    const char* original = std::getenv("RAD_PLE_GATE_PARALLEL");
    const bool present = original != nullptr;
    const std::string saved = original ? original : "";
    uint32_t state = 20261120u;
    auto random = [&] { state = state * 1664525u + 1013904223u; return state >> 8; };
    int cases = 0;
    for (int rows : {1, 2}) for (int pattern = 0; pattern < 4; ++pattern) {
        Buffer key(RAD_BF16, {rows, 10240}, RAD_MEM_DEVICE), query(RAD_BF16, {rows, 10240}, RAD_MEM_DEVICE);
        Buffer value(RAD_BF16, {rows, 2560}, RAD_MEM_DEVICE);
        Buffer wk(RAD_F32, {10240}, RAD_MEM_DEVICE), wq(RAD_F32, {10240}, RAD_MEM_DEVICE), wc(RAD_F32, {10240}, RAD_MEM_DEVICE);
        Buffer gv(RAD_BF16, {rows, 10240}, RAD_MEM_DEVICE), gvn(RAD_BF16, {rows, 10240}, RAD_MEM_DEVICE);
        auto bf16 = [&](bool zero) -> uint16_t {
            if (zero) return 0;
            const uint32_t bits = random();
            const uint32_t exponent = pattern == 2 ? 1 + bits % 253 : 118 + bits % 16;
            return uint16_t(((bits >> 9) & 0x8000u) | (exponent << 7) | ((bits >> 12) & 0x7fu));
        };
        for (int index = 0; index < rows * 10240; ++index) {
            key.data<uint16_t>()[index] = bf16(pattern == 1);
            query.data<uint16_t>()[index] = bf16(false);
        }
        for (int index = 0; index < rows * 2560; ++index) value.data<uint16_t>()[index] = bf16(pattern == 3);
        for (Buffer* weight : {&wk, &wq, &wc})
            for (int index = 0; index < 10240; ++index) weight->data<float>()[index] = float(int(random() % 2001) - 1000) / 1024.0f;
        for (Buffer* buffer : {&key, &query, &value, &wk, &wq, &wc, &gv, &gvn}) buffer->push(compute.stream);
        RadTensor tensors[]{key.device, query.device, value.device, wk.device, wq.device, wc.device, gv.device, gvn.device};
        RadParam params[]{RAD_INT("n", 2560), RAD_INT("hc", 4), RAD_F64("eps", 1e-6), RAD_F64("wadd", 1.0), RAD_F64("gate_eps", 1e-6)};
        RadArgs args{}; args.t = tensors; args.n_t = 8; args.p = params; args.n_p = 5; args.world_size = 1;
        REQUIRE_EQ(setenv("RAD_PLE_GATE_PARALLEL", "0", 1), 0);
        REQUIRE_EQ(launch(&args, compute.stream), RAD_OK);
        const auto reference_gv = gv.read(compute.stream), reference_gvn = gvn.read(compute.stream);
        REQUIRE_EQ(setenv("RAD_PLE_GATE_PARALLEL", "1", 1), 0);
        gv.push(compute.stream); gvn.push(compute.stream);
        REQUIRE_EQ(launch(&args, compute.stream), RAD_OK);
        REQUIRE(gv.read(compute.stream) == reference_gv);
        REQUIRE(gvn.read(compute.stream) == reference_gvn);
        for (Buffer* buffer : {&key, &query, &value, &wk, &wq, &wc}) REQUIRE(buffer->read(compute.stream) == buffer->expected);
        ++cases;
    }
    if (present) setenv("RAD_PLE_GATE_PARALLEL", saved.c_str(), 1); else unsetenv("RAD_PLE_GATE_PARALLEL");
    std::printf("PASS TP1_PLE_GATE_PARALLEL cases=%d rows=1,2 patterns=normal,zero-key,wide-fallback,zero-value exact=all\n", cases);
}

TEST(r3d_tp1_qsa_parallel) {
    using namespace iqr_output_boundary;
    RadDeviceProps props{};
    REQUIRE_EQ(rad_dev_props(0, &props), RAD_OK);
    if (props.is_host_backend) SKIP("gfx1100 required");
    Library native("libr3d");
    auto key = reinterpret_cast<RadLaunchFn>(dlsym(native.handle, "r3dh_sparse_qsa_block_key"));
    auto select = reinterpret_cast<RadLaunchFn>(dlsym(native.handle, "r3dh_sparse_qsa_select"));
    REQUIRE(key && select);
    Queue compute(0);
    uint32_t state = 20261121u;
    auto random = [&] { state = state * 1664525u + 1013904223u; return state >> 8; };
    int key_cases = 0, select_cases = 0;
    for (int list = 0; list < 2; ++list) for (int repeat = 0; repeat < 2; ++repeat) {
        const int blocks = 5, pages = 11;
        Buffer raw(RAD_BF16, {blocks * 4, 128}, RAD_MEM_DEVICE, 132), weights(RAD_F32, {128}, RAD_MEM_DEVICE);
        Buffer page(RAD_I32, {blocks}, RAD_MEM_DEVICE), positions(RAD_I32, {blocks, 3}, RAD_MEM_DEVICE);
        Buffer flat(RAD_BF16, {blocks, 128}, RAD_MEM_DEVICE);
        Buffer paged(RAD_BF16, {pages, 1, 4, 32}, RAD_MEM_DEVICE);
        Buffer& output = list ? paged : flat;
        for (int index = 0; index < blocks * 4 * 132; ++index)
            raw.data<uint16_t>()[index] = uint16_t(((random() & 1) << 15) | ((118 + random() % 16) << 7) | (random() & 127));
        for (int lane = 0; lane < 128; ++lane) weights.data<float>()[lane] = float(int(lane % 7) - 3) / 16;
        for (int block = 0; block < blocks; ++block) {
            page.data<int>()[block] = block == 2 ? -1 : (block * 3) % pages;
            for (int component = 0; component < 3; ++component)
                positions.data<int>()[block * 3 + component] = block * 4 + component * 101 + 199600 + repeat * 7;
        }
        for (Buffer* buffer : {&raw, &weights, &page, &positions, &output}) buffer->push(compute.stream);
        RadParam params[]{RAD_INT("n", 128), RAD_INT("ratio", 4), RAD_INT("rotary_dim", 64), RAD_INT("pos0", 199600 + repeat),
            RAD_F64("theta", 1e7), RAD_F64("eps", 1e-6), RAD_F64("wadd", 1), RAD_STR("mode", "mrope"), RAD_STR("sections", "11 11 10")};
        RadTensor tensors[]{raw.device, weights.device, list ? page.device : RadTensor{}, list ? positions.device : RadTensor{}, output.device};
        RadArgs args{}; args.t = tensors; args.n_t = 5; args.p = params; args.n_p = 9; args.world_size = 1;
        REQUIRE_EQ(setenv("RAD_QSA_KEY_PARALLEL", "0", 1), 0);
        REQUIRE_EQ(key(&args, compute.stream), RAD_OK);
        const auto reference = output.read(compute.stream);
        REQUIRE(reference != output.expected);
        REQUIRE_EQ(setenv("RAD_QSA_KEY_PARALLEL", "1", 1), 0);
        output.push(compute.stream);
        REQUIRE_EQ(key(&args, compute.stream), RAD_OK);
        REQUIRE(output.read(compute.stream) == reference);
        ++key_cases;
    }
    for (int width : {20, 512, 600}) for (int signed_score = 0; signed_score < 2; ++signed_score) {
        const int rows = 2, topk = 512;
        Buffer scores(RAD_F32, {rows, width}, RAD_MEM_DEVICE, width + 3), table(RAD_I32, {1, width}, RAD_MEM_DEVICE, width + 2);
        Buffer count(RAD_I32, {1}, RAD_MEM_DEVICE), position(RAD_I32, {rows}, RAD_MEM_DEVICE), cu(RAD_I32, {2}, RAD_MEM_DEVICE);
        Buffer selected(RAD_I32, {rows, topk + 1}, RAD_MEM_DEVICE), nselected(RAD_I32, {rows}, RAD_MEM_DEVICE), used(RAD_I32, {rows}, RAD_MEM_DEVICE);
        for (int row = 0; row < rows; ++row) for (int column = 0; column < width + 3; ++column) {
            float value = float(int(random() % 9) - 4) * 0.25f;
            if (column % 17 == 5) value = -INFINITY;
            scores.data<float>()[row * (width + 3) + column] = value;
        }
        for (int column = 0; column < width + 2; ++column) table.data<int>()[column] = column % 13 == 4 ? -1 : (column * 7 + 3) % (width + 5);
        count.data<int>()[0] = width;
        position.data<int>()[0] = 4 * width - 1;
        position.data<int>()[1] = 4 * (width / 2) + 1;
        cu.data<int>()[0] = 0; cu.data<int>()[1] = rows;
        for (int index = 0; index < rows * (topk + 1); ++index) selected.data<int>()[index] = 12345;
        for (Buffer* buffer : {&scores, &table, &count, &position, &cu, &selected, &nselected, &used}) buffer->push(compute.stream);
        RadParam params[]{RAD_INT("ratio", 4), RAD_INT("topk", topk), RAD_INT("signed", signed_score)};
        RadTensor tensors[]{scores.device, table.device, count.device, position.device, cu.device, selected.device, nselected.device, used.device};
        RadArgs args{}; args.t = tensors; args.n_t = 8; args.p = params; args.n_p = 3; args.world_size = 1;
        REQUIRE_EQ(setenv("RAD_QSA_SELECT_PARALLEL", "0", 1), 0);
        REQUIRE_EQ(select(&args, compute.stream), RAD_OK);
        const auto expected_selected = selected.read(compute.stream), expected_count = nselected.read(compute.stream);
        const auto expected_used = used.read(compute.stream);
        REQUIRE_EQ(setenv("RAD_QSA_SELECT_PARALLEL", "1", 1), 0);
        for (Buffer* buffer : {&selected, &nselected, &used}) buffer->push(compute.stream);
        REQUIRE_EQ(select(&args, compute.stream), RAD_OK);
        REQUIRE(selected.read(compute.stream) == expected_selected);
        REQUIRE(nselected.read(compute.stream) == expected_count);
        REQUIRE(used.read(compute.stream) == expected_used);
        int counts[2];
        std::memcpy(counts, expected_count.data() + 64, sizeof(counts));
        std::printf("TP1_QSA_SELECT width=%d signed=%d nselected=%d,%d\n", width, signed_score, counts[0], counts[1]);
        ++select_cases;
    }
    setenv("RAD_QSA_KEY_PARALLEL", "0", 1); setenv("RAD_QSA_SELECT_PARALLEL", "0", 1);
    std::printf("PASS TP1_QSA_PARALLEL key_cases=%d select_cases=%d exact=all\n", key_cases, select_cases);
}

namespace moe_prefill {
// One device copy per expert, so the timed kernels stream distinct weights as in serving.
struct DistinctTable {
    size_t stride;
    int resident;
    Buffer blob, pooled, pointers;
    int entries;
    // Entries from `resident` on live in pinned host memory, read in place like the host pool.
    DistinctTable(const std::vector<uint8_t>& data, int count, int resident_count, RadStream stream)
        : stride((data.size() + 255) / 256 * 256), resident(resident_count),
          blob(RAD_I32, {std::max<int64_t>(1, int64_t(resident) * int64_t(stride / 4))}, RAD_MEM_DEVICE),
          pooled(RAD_I32, {std::max<int64_t>(1, int64_t(count - resident) * int64_t(stride / 4))},
              RAD_MEM_HOST_PINNED),
          pointers(RAD_I64, {count}, RAD_MEM_DEVICE), entries(count) {
        for (int index = 0; index < count; ++index) {
            Buffer& target = index < resident ? blob : pooled;
            const size_t offset = size_t(index < resident ? index : index - resident) * stride;
            std::memcpy(target.data<uint8_t>() + offset, data.data(), data.size());
            pointers.data<uint64_t>()[index] = reinterpret_cast<uintptr_t>(target.device.data) + offset;
        }
        blob.push(stream); pooled.push(stream); pointers.push(stream);
    }
    RadTensor view(uint32_t dtype, int rows, int columns) {
        auto tensor = pointers.device; tensor.dtype = dtype; tensor.rank = 3;
        tensor.shape[0] = entries; tensor.shape[1] = rows; tensor.shape[2] = columns;
        tensor.stride[0] = 0; tensor.stride[1] = columns; tensor.stride[2] = 1;
        return tensor;
    }
};
}

TEST(r3d_tp1_moe_prefill_bench) {
    if (!std::getenv("RAD_TP1_MOE_PREFILL_BENCH")) SKIP("optional device-only MoE prefill bench");
    RadDeviceProps props{};
    REQUIRE_EQ(rad_dev_props(0, &props), RAD_OK);
    REQUIRE(!props.is_host_backend);
    REQUIRE_EQ(rad_dev_set(0), RAD_OK);
    Library library("libr3d");
    Queue compute(0);
    const RadStream stream = compute.stream;
    Model model("/home/zihaomu/bigssd2/radiance_workspace/models/qwen3.8-next-flash-iqr-v4.rad");
    const int experts = 512, width = 1280, inner = 640;
    std::array<moe_boundary::Expert, 2> weights{moe_boundary::load(model, 0, 0, 0, 0, true),
        moe_boundary::load(model, 0, 1, 0, 1, true)};
    const auto* gate = moe_boundary::registered(library, "model_moe_iqr_1280_1280_strict");
    const auto* down = moe_boundary::registered(library, "model_moe_w4nl_640_640_strict");
    const auto* quant = moe_boundary::registered(library, "model_gated_had_640_strict");
    const char* pooled_percent = std::getenv("RAD_TP1_MOE_HOST_PERCENT");
    const int resident = experts / 2 - (pooled_percent ? std::atoi(pooled_percent) : 0) * (experts / 2) / 100;
    std::printf("MOE_PREFILL_BENCH resident_per_parity=%d of %d (rest pinned host, zero-copy)\n", resident, experts / 2);
    std::array<std::unique_ptr<moe_prefill::DistinctTable>, 8> tables;
    for (int parity = 0; parity < 2; ++parity) {
        tables[parity * 4] = std::make_unique<moe_prefill::DistinctTable>(weights[parity].gate, experts / 2, resident, stream);
        tables[parity * 4 + 1] = std::make_unique<moe_prefill::DistinctTable>(weights[parity].scale, experts / 2, resident, stream);
        tables[parity * 4 + 2] = std::make_unique<moe_prefill::DistinctTable>(weights[parity].down, experts / 2, resident, stream);
        tables[parity * 4 + 3] = std::make_unique<moe_prefill::DistinctTable>(weights[parity].down_scale, experts / 2, resident, stream);
    }
    setenv("RAD_IQR_DOT4", "1", 1); setenv("RAD_W4NL_DOT4", "1", 1);
    RadEvent start{}, finish{};
    REQUIRE_EQ(rad_event_create(&start), RAD_OK);
    REQUIRE_EQ(rad_event_create(&finish), RAD_OK);
    for (int tokens : {1, 4, 16, 63, 64, 128, 256, 512}) {
        const int rows = tokens * 10, band = std::max(tokens, 64);
        Buffer input(RAD_I8, {tokens, 2560}, RAD_MEM_DEVICE);
        Buffer scale(RAD_F32, {tokens, 20}, RAD_MEM_DEVICE);
        Buffer sorted(RAD_I32, {rows}, RAD_MEM_DEVICE), offsets(RAD_I32, {experts + 1}, RAD_MEM_DEVICE);
        Buffer output(RAD_BF16, {rows, width}, RAD_MEM_DEVICE);
        Buffer codes(RAD_I8, {rows, inner}, RAD_MEM_DEVICE);
        Buffer factors(RAD_F32, {rows, inner / 128}, RAD_MEM_DEVICE);
        Buffer result(RAD_BF16, {rows, 2560}, RAD_MEM_DEVICE);
        for (int index = 0; index < tokens * 2560; ++index)
            input.data<int8_t>()[index] = int8_t(int((index * 19 + 7) % 256) - 128);
        for (int index = 0; index < tokens * 20; ++index) scale.data<float>()[index] = float(index % 7 + 1) / 8192.0f;
        std::vector<std::vector<int>> owners(experts);
        uint64_t state = 0x9e3779b97f4a7c15ull ^ uint64_t(tokens);
        for (int token = 0; token < tokens; ++token) {
            std::vector<int> chosen;
            while (chosen.size() < 10) {
                state = state * 6364136223846793005ull + 1442695040888963407ull;
                const int expert = int((state >> 33) % experts);
                if (std::find(chosen.begin(), chosen.end(), expert) != chosen.end()) continue;
                owners[expert].push_back(token * 10 + int(chosen.size()));
                chosen.push_back(expert);
            }
        }
        int filled = 0, distinct = 0, longest = 0;
        for (int expert = 0; expert < experts; ++expert) {
            offsets.data<int32_t>()[expert] = filled;
            for (int slot : owners[expert]) sorted.data<int32_t>()[filled++] = slot;
            distinct += !owners[expert].empty();
            longest = std::max(longest, int(owners[expert].size()));
        }
        offsets.data<int32_t>()[experts] = filled;
        for (Buffer* buffer : {&input, &scale, &sorted, &offsets}) buffer->push(stream);
        std::vector<RadParam> gate_params{RAD_INT("M", band), RAD_INT("N", width), RAD_INT("N_odd", width),
            RAD_INT("K", 2560), RAD_INT("K_odd", 2560), RAD_INT("n_expert", experts), RAD_INT("top_k", 10),
            RAD_INT("group", 128), RAD_INT("a_group", 128), RAD_INT("parts", 2), RAD_STR("dtype", "i8a8h"),
            RAD_STR("a_order", "token")};
        std::array<RadTensor, 9> tensors{input.device, scale.device, {}, {}, sorted.device, offsets.device, {}, {},
            output.device};
        for (int parity = 0; parity < 2; ++parity) {
            tensors[2 + parity * 4] = tables[parity * 4]->view(r3dh_moe_iqr::weight_dtype, width, 2560);
            tensors[3 + parity * 4] = tables[parity * 4 + 1]->view(r3dh_moe_iqr::scale_dtype, width, 20);
        }
        RadArgs args{}; args.t = tensors.data(); args.n_t = tensors.size();
        args.p = gate_params.data(); args.n_p = gate_params.size(); args.world_size = 1;
        const int64_t scratch_bytes = gate->scratch(&args);
        Buffer scratch(RAD_I32, {(scratch_bytes + 3) / 4}, RAD_MEM_DEVICE);
        args.scratch = scratch.device.data; args.scratch_bytes = scratch_bytes;
        std::array<RadTensor, 3> quant_tensors{output.device, codes.device, factors.device};
        std::vector<RadParam> quant_params{RAD_INT("M", rows), RAD_INT("n", inner), RAD_INT("group", 128),
            RAD_STR("dtype", "bf16"), RAD_STR("act", "silu")};
        RadArgs quant_args{}; quant_args.t = quant_tensors.data(); quant_args.n_t = 3;
        quant_args.p = quant_params.data(); quant_args.n_p = quant_params.size();
        std::vector<RadParam> down_params{RAD_INT("M", band), RAD_INT("N", 2560), RAD_INT("N_odd", 2560),
            RAD_INT("K", inner), RAD_INT("K_odd", inner), RAD_INT("n_expert", experts), RAD_INT("top_k", 10),
            RAD_INT("group", 128), RAD_INT("a_group", 128), RAD_INT("parts", 1), RAD_STR("dtype", "w4nl64a8i8h"),
            RAD_STR("a_order", "sorted")};
        auto down_tensors = tensors; down_tensors[0] = codes.device; down_tensors[1] = factors.device;
        down_tensors[8] = result.device;
        for (int parity = 0; parity < 2; ++parity) {
            down_tensors[2 + parity * 4] = tables[parity * 4 + 2]->view(r3dh_moe_w4nl::codes_type, 2560, inner);
            down_tensors[3 + parity * 4] = tables[parity * 4 + 3]->view(r3dh_moe_w4nl::scales_type, 2560, inner);
        }
        RadArgs down_args{}; down_args.t = down_tensors.data(); down_args.n_t = down_tensors.size();
        down_args.p = down_params.data(); down_args.n_p = down_params.size(); down_args.world_size = 1;
        const std::array<std::pair<const RadKernelInfo*, RadArgs*>, 3> stages{{{gate, &args}, {quant, &quant_args},
            {down, &down_args}}};
        float micros[3]{};
        for (int stage = 0; stage < 3; ++stage) {
            for (int warmup = 0; warmup < 2; ++warmup)
                REQUIRE_EQ(stages[stage].first->launch(stages[stage].second, stream), RAD_OK);
            REQUIRE_EQ(rad_event_record(start, stream), RAD_OK);
            for (int repeat = 0; repeat < 10; ++repeat)
                REQUIRE_EQ(stages[stage].first->launch(stages[stage].second, stream), RAD_OK);
            REQUIRE_EQ(rad_event_record(finish, stream), RAD_OK);
            REQUIRE_EQ(rad_event_sync(finish), RAD_OK);
            float elapsed = 0;
            REQUIRE_EQ(rad_event_elapsed_ms(start, finish, &elapsed), RAD_OK);
            micros[stage] = elapsed * 100.0f;
        }
        if (std::getenv("RAD_TP1_MOE_WINDOW_AB")) {
            float window_us[2]{};
            std::vector<uint8_t> outputs[2];
            for (int variant : {0, 1, 1, 0}) {
                setenv("RAD_IQR_DOT4_WINDOW", variant ? "1" : "0", 1);
                std::fill(output.expected.begin(), output.expected.end(), 0x5a);
                output.push(stream);
                REQUIRE_EQ(gate->launch(&args, stream), RAD_OK);
                auto got = output.read(stream);
                if (outputs[variant].empty()) outputs[variant] = std::move(got);
                else REQUIRE(got == outputs[variant]);
                REQUIRE_EQ(rad_event_record(start, stream), RAD_OK);
                for (int repeat = 0; repeat < 10; ++repeat) REQUIRE_EQ(gate->launch(&args, stream), RAD_OK);
                REQUIRE_EQ(rad_event_record(finish, stream), RAD_OK);
                REQUIRE_EQ(rad_event_sync(finish), RAD_OK);
                float elapsed = 0;
                REQUIRE_EQ(rad_event_elapsed_ms(start, finish, &elapsed), RAD_OK);
                window_us[variant] += elapsed * 50.0f;
            }
            setenv("RAD_IQR_DOT4_WINDOW", "0", 1);
            REQUIRE(outputs[0] == outputs[1]);
            std::printf("MOE_WINDOW_AB tokens=%d gate_up_us=%.1f window_us=%.1f speedup=%.3f identical=1\n", tokens,
                window_us[0], window_us[1], window_us[0] / window_us[1]);
        }
        if (std::getenv("RAD_TP1_MOE_GATE_AB")) {
            const char* ab_flag = std::getenv("RAD_TP1_MOE_AB_FLAG") ? std::getenv("RAD_TP1_MOE_AB_FLAG") : "RAD_IQR_SPLIT";
            setenv("RAD_IQR_DOT4_WINDOW", "1", 1);
            for (int pattern = 0; pattern < 3; ++pattern) {
                uint64_t seed = 0x243f6a8885a308d3ull + uint64_t(pattern);
                for (int index = 0; index < tokens * 2560; ++index) {
                    seed = seed * 6364136223846793005ull + 1442695040888963407ull;
                    input.data<int8_t>()[index] = pattern == 0 ? int8_t(int((index * 19 + 7) % 256) - 128)
                        : pattern == 1 ? int8_t(seed >> 56) : int8_t(index % 2 ? -128 : 127);
                }
                input.push(stream);
                float gate_us[2]{};
                std::vector<uint8_t> outputs[2];
                for (int variant : {0, 1, 1, 0}) {
                    setenv(ab_flag, variant ? "1" : "0", 1);
                    std::fill(output.expected.begin(), output.expected.end(), 0x5a);
                    output.push(stream);
                    REQUIRE_EQ(gate->launch(&args, stream), RAD_OK);
                    auto got = output.read(stream);
                    if (outputs[variant].empty()) outputs[variant] = std::move(got);
                    else REQUIRE(got == outputs[variant]);
                    REQUIRE_EQ(rad_event_record(start, stream), RAD_OK);
                    for (int repeat = 0; repeat < 10; ++repeat) REQUIRE_EQ(gate->launch(&args, stream), RAD_OK);
                    REQUIRE_EQ(rad_event_record(finish, stream), RAD_OK);
                    REQUIRE_EQ(rad_event_sync(finish), RAD_OK);
                    float elapsed = 0;
                    REQUIRE_EQ(rad_event_elapsed_ms(start, finish, &elapsed), RAD_OK);
                    gate_us[variant] += elapsed * 50.0f;
                }
                setenv(ab_flag, "0", 1);
                REQUIRE(outputs[0] == outputs[1]);
                std::printf("MOE_GATE_AB flag=%s tokens=%d pattern=%d off_us=%.1f on_us=%.1f speedup=%.3f identical=1\n",
                    ab_flag, tokens, pattern, gate_us[0], gate_us[1], gate_us[0] / gate_us[1]);
            }
            setenv("RAD_IQR_DOT4_WINDOW", "0", 1);
        }
        if (std::getenv("RAD_TP1_MOE_RUN_AB")) {
            const char* ab_flag = std::getenv("RAD_TP1_MOE_AB_FLAG") ? std::getenv("RAD_TP1_MOE_AB_FLAG") : "RAD_W4NL_RUN";
            REQUIRE_EQ(quant->launch(&quant_args, stream), RAD_OK);
            float run_us[2]{};
            std::vector<uint8_t> outputs[2];
            for (int variant : {0, 1, 1, 0}) {
                setenv(ab_flag, variant ? "1" : "0", 1);
                std::fill(result.expected.begin(), result.expected.end(), 0x5a);
                result.push(stream);
                REQUIRE_EQ(down->launch(&down_args, stream), RAD_OK);
                auto got = result.read(stream);
                if (outputs[variant].empty()) outputs[variant] = std::move(got);
                else REQUIRE(got == outputs[variant]);
                REQUIRE_EQ(rad_event_record(start, stream), RAD_OK);
                for (int repeat = 0; repeat < 10; ++repeat) REQUIRE_EQ(down->launch(&down_args, stream), RAD_OK);
                REQUIRE_EQ(rad_event_record(finish, stream), RAD_OK);
                REQUIRE_EQ(rad_event_sync(finish), RAD_OK);
                float elapsed = 0;
                REQUIRE_EQ(rad_event_elapsed_ms(start, finish, &elapsed), RAD_OK);
                run_us[variant] += elapsed * 50.0f;
            }
            setenv(ab_flag, "0", 1);
            REQUIRE(outputs[0] == outputs[1]);
            std::printf("MOE_RUN_AB flag=%s tokens=%d off_us=%.1f on_us=%.1f speedup=%.3f identical=1\n", ab_flag, tokens,
                run_us[0], run_us[1], run_us[0] / run_us[1]);
        }
        std::printf("MOE_PREFILL_BENCH tokens=%d routes=%d distinct=%d longest=%d gate_up_us=%.1f quant_us=%.1f "
            "down_us=%.1f layer_us=%.1f x48_ms=%.1f\n", tokens, rows, distinct, longest, micros[0], micros[1],
            micros[2], micros[0] + micros[1] + micros[2], (micros[0] + micros[1] + micros[2]) * 48 / 1000);
        std::fflush(stdout);
    }
    rad_event_destroy(start); rad_event_destroy(finish);
    setenv("RAD_IQR_DOT4", "0", 1); setenv("RAD_W4NL_DOT4", "0", 1);
}

TEST(r3d_tp1_attention_fast_ab) {
    if (!std::getenv("RAD_TP1_ATTN_AB")) SKIP("optional paged attention / kv store A/B");
    RadDeviceProps props{};
    REQUIRE_EQ(rad_dev_props(0, &props), RAD_OK);
    REQUIRE(!props.is_host_backend);
    REQUIRE_EQ(rad_dev_set(0), RAD_OK);
    Library native("libr3d");
    const auto store = reinterpret_cast<RadLaunchFn>(dlsym(native.handle, "r3dh_attn_qsa_kv_store"));
    const auto attend = reinterpret_cast<RadLaunchFn>(dlsym(native.handle, "r3dh_attn_qsa_paged"));
    REQUIRE(store && attend);
    Queue compute(0);
    RadEvent start{}, finish{};
    REQUIRE_EQ(rad_event_create(&start), RAD_OK); REQUIRE_EQ(rad_event_create(&finish), RAD_OK);
    uint32_t state = 20261119u;
    auto random = [&] { state = state * 1664525u + 1013904223u; return state >> 8; };
    int cases = 0;
    for (int heads : {12, 24}) {
        const int keyheads = heads / 12, pages = 1100;
        for (int rows : {1, 3})
        for (int used_tokens : {1, 3, 4, 5, 63, 200, 257, 600, 1000, 4000}) {
            if (used_tokens < rows) continue;
            Buffer key(RAD_BF16, {rows, keyheads * 256}, RAD_MEM_DEVICE);
            Buffer value(RAD_BF16, {rows, keyheads * 256}, RAD_MEM_DEVICE);
            Buffer slots(RAD_I32, {rows}, RAD_MEM_DEVICE);
            Buffer cache(RAD_I8, {pages, keyheads, 4, 516}, RAD_MEM_DEVICE);
            Buffer query(RAD_BF16, {rows, heads * 256}, RAD_MEM_DEVICE);
            Buffer output(RAD_BF16, {rows, heads * 256}, RAD_MEM_DEVICE);
            Buffer table(RAD_I32, {1, 1024}, RAD_MEM_DEVICE), used(RAD_I32, {1}, RAD_MEM_DEVICE);
            Buffer cu(RAD_I32, {2}, RAD_MEM_DEVICE);
            for (size_t index = 0; index + 64 < cache.expected.size(); ++index)
                cache.expected[64 + index] = uint8_t(random());
            for (int page = 0; page < pages; ++page)
                for (int head = 0; head < keyheads; ++head)
                    for (int slot = 0; slot < 4; ++slot)
                        for (int half = 0; half < 2; ++half) {
                            const uint16_t scale = rad_f32_to_f16(float(random() % 1000 + 1) / 65536.0f);
                            std::memcpy(cache.data<uint8_t>() + (((size_t(page) * keyheads + head) * 4 + slot) * 516
                                + 512 + half * 2), &scale, 2);
                        }
            for (int entry = 0; entry < 1024; ++entry)
                table.data<int32_t>()[entry] = random() % 50 == 0 ? -1 : int(random() % pages);
            for (int row = 0; row < rows; ++row) {
                slots.data<int32_t>()[row] = int(random() % (pages * 4));
                for (int column = 0; column < keyheads * 256; ++column) {
                    key.data<uint16_t>()[row * keyheads * 256 + column] = rad_f32_to_bf16(float(int(random() % 2001) - 1000) / 256.0f);
                    value.data<uint16_t>()[row * keyheads * 256 + column] = rad_f32_to_bf16(float(int(random() % 2001) - 1000) / 512.0f);
                }
                for (int column = 0; column < heads * 256; ++column)
                    query.data<uint16_t>()[row * heads * 256 + column] = rad_f32_to_bf16(float(int(random() % 2001) - 1000) / 64.0f);
            }
            used.data<int32_t>()[0] = used_tokens;
            cu.data<int32_t>()[0] = 0; cu.data<int32_t>()[1] = rows;
            for (Buffer* buffer : {&key, &value, &slots, &cache, &query, &output, &table, &used, &cu}) buffer->push(compute.stream);
            RadParam store_params[]{RAD_INT("M", rows), RAD_INT("head_dim", 256), RAD_INT("n_head_kv", keyheads),
                RAD_INT("block_size", 4), RAD_STR("kv_dtype", "i8")};
            RadTensor store_tensors[]{key.device, value.device, slots.device, cache.device};
            RadArgs store_args{}; store_args.t = store_tensors; store_args.n_t = 4; store_args.p = store_params; store_args.n_p = 5;
            std::vector<uint8_t> caches[2];
            for (int variant = 0; variant < 2; ++variant) {
                cache.push(compute.stream);
                REQUIRE_EQ(setenv("RAD_KV_STORE_PARALLEL", variant ? "1" : "0", 1), 0);
                REQUIRE_EQ(store(&store_args, compute.stream), RAD_OK);
                caches[variant] = cache.read(compute.stream);
            }
            REQUIRE(caches[0] == caches[1]);
            RadParam params[]{RAD_INT("head_dim", 256), RAD_INT("n_head", heads), RAD_INT("gqa", 12),
                RAD_INT("block_size", 4), RAD_INT("causal", 1), RAD_INT("window", 0), RAD_STR("q_dtype", "bf16"),
                RAD_STR("kv_dtype", "i8"), RAD_F64("scale", 0.0625)};
            RadTensor tensors[]{query.device, cache.device, table.device, used.device, {}, {}, cu.device, output.device};
            RadArgs args{}; args.t = tensors; args.n_t = 8; args.p = params; args.n_p = int(sizeof(params) / sizeof(*params));
            std::vector<uint8_t> outputs[2];
            float micros[2]{};
            for (int variant : {0, 1, 1, 0}) {
                REQUIRE_EQ(setenv("RAD_ATTN_PAGED_FAST", variant ? "1" : "0", 1), 0);
                output.push(compute.stream);
                REQUIRE_EQ(attend(&args, compute.stream), RAD_OK);
                auto got = output.read(compute.stream);
                if (outputs[variant].empty()) outputs[variant] = std::move(got); else REQUIRE(got == outputs[variant]);
                REQUIRE_EQ(rad_event_record(start, compute.stream), RAD_OK);
                for (int repeat = 0; repeat < 10; ++repeat) REQUIRE_EQ(attend(&args, compute.stream), RAD_OK);
                REQUIRE_EQ(rad_event_record(finish, compute.stream), RAD_OK);
                REQUIRE_EQ(rad_event_sync(finish), RAD_OK);
                float elapsed = 0;
                REQUIRE_EQ(rad_event_elapsed_ms(start, finish, &elapsed), RAD_OK);
                micros[variant] += elapsed * 50.0f;
            }
            REQUIRE(outputs[0] == outputs[1]);
            std::printf("TP1_ATTN_AB heads=%d rows=%d keys=%d strict_us=%.1f fast_us=%.1f speedup=%.2f identical=1\n",
                heads, rows, used_tokens, micros[0], micros[1], micros[0] / micros[1]);
            ++cases;
        }
    }
    setenv("RAD_ATTN_PAGED_FAST", "0", 1); setenv("RAD_KV_STORE_PARALLEL", "0", 1);
    rad_event_destroy(start); rad_event_destroy(finish);
    std::printf("PASS TP1_ATTN_AB cases=%d attention+store bit-identical\n", cases);
}

TEST(r3d_tp1_attention_fused_ab) {
    if (!std::getenv("RAD_TP1_ATTN_FUSED_AB")) SKIP("optional TP1 fused attention ops A/B");
    RadDeviceProps props{};
    REQUIRE_EQ(rad_dev_props(0, &props), RAD_OK);
    REQUIRE(!props.is_host_backend);
    REQUIRE_EQ(rad_dev_set(0), RAD_OK);
    Library native("libr3d");
    auto symbol = [&](const char* name) { return reinterpret_cast<RadLaunchFn>(dlsym(native.handle, name)); };
    const auto prepare = symbol("r3dh_attn_qsa_prepare"), store = symbol("r3dh_attn_qsa_kv_store");
    const auto prepare_store = symbol("r3dh_attn_qsa_prepare_store"), attend = symbol("r3dh_attn_qsa_paged");
    const auto gate_quant = symbol("r3dh_qwen_ops_gate_had_quant_i8g"), paged_gate = symbol("r3dh_attn_qsa_paged_gate");
    REQUIRE(prepare && store && prepare_store && attend && gate_quant && paged_gate);
    Queue compute(0);
    RadEvent start{}, finish{};
    REQUIRE_EQ(rad_event_create(&start), RAD_OK); REQUIRE_EQ(rad_event_create(&finish), RAD_OK);
    uint32_t state = 20261120u;
    auto random = [&] { state = state * 1664525u + 1013904223u; return state >> 8; };
    auto bf16 = [&](float scale) { return rad_f32_to_bf16(float(int(random() % 2001) - 1000) / scale); };
    auto time = [&](auto&& launch, float& total) {
        for (int warm = 0; warm < 3; ++warm) launch();
        REQUIRE_EQ(rad_event_record(start, compute.stream), RAD_OK);
        for (int repeat = 0; repeat < 20; ++repeat) launch();
        REQUIRE_EQ(rad_event_record(finish, compute.stream), RAD_OK);
        REQUIRE_EQ(rad_event_sync(finish), RAD_OK);
        float elapsed = 0;
        REQUIRE_EQ(rad_event_elapsed_ms(start, finish, &elapsed), RAD_OK);
        total += elapsed * 50.0f;
    };
    int cases = 0;
    // qk_norm_rope_gate + kv_store vs qk_norm_rope_gate_kv_store, k in place as the graph issues it.
    setenv("RAD_QK_PREPARE_PARALLEL", "1", 1); setenv("RAD_KV_STORE_PARALLEL", "1", 1);
    for (int rows : {1, 2, 5, 64}) {
        const int pages = 40, positions = 512;
        Buffer qg(RAD_BF16, {rows, 12288}, RAD_MEM_DEVICE), key(RAD_BF16, {rows, 512}, RAD_MEM_DEVICE);
        Buffer value(RAD_BF16, {rows, 512}, RAD_MEM_DEVICE), table(RAD_F32, {positions, 64}, RAD_MEM_DEVICE);
        Buffer position(RAD_I32, {rows}, RAD_MEM_DEVICE), slots(RAD_I32, {rows}, RAD_MEM_DEVICE);
        Buffer qw(RAD_F32, {256}, RAD_MEM_DEVICE), kw(RAD_F32, {256}, RAD_MEM_DEVICE);
        Buffer query(RAD_BF16, {rows, 6144}, RAD_MEM_DEVICE), cache(RAD_I8, {pages, 2, 4, 516}, RAD_MEM_DEVICE);
        for (int index = 0; index < rows * 12288; ++index) qg.data<uint16_t>()[index] = bf16(64.0f);
        for (int index = 0; index < rows * 512; ++index) {
            key.data<uint16_t>()[index] = bf16(32.0f);
            value.data<uint16_t>()[index] = bf16(128.0f);
        }
        for (int index = 0; index < positions * 64; ++index)
            table.data<float>()[index] = float(int(random() % 2001) - 1000) / 1000.0f;
        for (int index = 0; index < 256; ++index) {
            qw.data<float>()[index] = float(int(random() % 2001) - 1000) / 700.0f;
            kw.data<float>()[index] = float(int(random() % 2001) - 1000) / 700.0f;
        }
        for (int row = 0; row < rows; ++row) {
            position.data<int32_t>()[row] = int(random() % positions);
            slots.data<int32_t>()[row] = row % 7 == 3 ? -1 : (row * 13 + 5) % (pages * 4);
        }
        for (size_t index = 0; index + 128 < cache.expected.size(); ++index) cache.expected[64 + index] = uint8_t(random());
        RadParam params[]{RAD_INT("M", rows), RAD_INT("head_dim", 256), RAD_INT("n_head", 24), RAD_INT("n_head_kv", 2),
            RAD_INT("rot", 64), RAD_STR("q_dtype", "bf16"), RAD_F64("eps", 1e-6), RAD_F64("wadd", 1.0),
            RAD_INT("cs_f32", 1), RAD_INT("block_size", 4), RAD_STR("kv_dtype", "i8")};
        RadArgs args{}; args.p = params; args.n_p = int(sizeof(params) / sizeof(*params));
        std::vector<uint8_t> results[2][3];
        float micros[2]{};
        for (int variant : {0, 1, 1, 0}) {
            for (Buffer* buffer : {&qg, &key, &value, &table, &position, &slots, &qw, &kw, &query, &cache})
                buffer->push(compute.stream);
            RadTensor prep[]{qg.device, key.device, table.device, position.device, qw.device, kw.device,
                query.device, key.device, {}};
            RadTensor kv[]{key.device, value.device, slots.device, cache.device};
            RadTensor fused[]{qg.device, key.device, table.device, position.device, qw.device, kw.device,
                query.device, key.device, {}, value.device, slots.device, cache.device};
            auto pair = [&] {
                args.t = prep; args.n_t = 9; REQUIRE_EQ(prepare(&args, compute.stream), RAD_OK);
                args.t = kv; args.n_t = 4; REQUIRE_EQ(store(&args, compute.stream), RAD_OK);
            };
            auto one = [&] { args.t = fused; args.n_t = 12; REQUIRE_EQ(prepare_store(&args, compute.stream), RAD_OK); };
            if (variant) one(); else pair();
            std::vector<uint8_t> got[3]{query.read(compute.stream), key.read(compute.stream), cache.read(compute.stream)};
            for (int index = 0; index < 3; ++index) {
                if (results[variant][index].empty()) results[variant][index] = got[index];
                else REQUIRE(got[index] == results[variant][index]);
            }
            // The timed launches rewrite k in place (prepare of a prepared k), which only changes values.
            if (variant) time(one, micros[1]); else time(pair, micros[0]);
        }
        for (int index = 0; index < 3; ++index) REQUIRE(results[0][index] == results[1][index]);
        std::printf("TP1_PREP_STORE_AB rows=%d pair_us=%.1f fused_us=%.1f identical=1\n", rows, micros[0] / 2, micros[1] / 2);
        ++cases;
    }
    // attn_paged + gate_had_quant_i8g vs attn_paged_gate_had_quant: rows of q_len 1, each its own selection.
    setenv("RAD_ATTN_PAGED_FAST", "1", 1);
    for (int rows : {1, 3})
    for (int used_tokens : {1, 5, 63, 250, 600, 2051}) {
        const int pages = 1100, width = 513;
        Buffer cache(RAD_I8, {pages, 2, 4, 516}, RAD_MEM_DEVICE);
        Buffer query(RAD_BF16, {rows, 6144}, RAD_MEM_DEVICE), qg(RAD_BF16, {rows, 12288}, RAD_MEM_DEVICE);
        Buffer output(RAD_BF16, {rows, 6144}, RAD_MEM_DEVICE), codes(RAD_I8, {rows, 6144}, RAD_MEM_DEVICE);
        Buffer scales(RAD_F32, {rows, 48}, RAD_MEM_DEVICE);
        Buffer table(RAD_I32, {rows, width}, RAD_MEM_DEVICE), used(RAD_I32, {rows}, RAD_MEM_DEVICE);
        for (size_t index = 0; index + 128 < cache.expected.size(); ++index) cache.expected[64 + index] = uint8_t(random());
        for (int page = 0; page < pages; ++page)
            for (int entry = 0; entry < 16; ++entry) {
                const uint16_t scale = rad_f32_to_f16(float(random() % 1000 + 1) / 65536.0f);
                std::memcpy(cache.data<uint8_t>() + (size_t(page) * 8 + entry / 2) * 516 + 512 + (entry % 2) * 2, &scale, 2);
            }
        for (int index = 0; index < rows * width; ++index)
            table.data<int32_t>()[index] = random() % 50 == 0 ? -1 : int(random() % pages);
        for (int row = 0; row < rows; ++row) used.data<int32_t>()[row] = used_tokens;
        for (int index = 0; index < rows * 6144; ++index) query.data<uint16_t>()[index] = bf16(64.0f);
        for (int index = 0; index < rows * 12288; ++index) qg.data<uint16_t>()[index] = bf16(random() % 3 ? 128.0f : 8.0f);
        RadParam params[]{RAD_INT("q_len", 1), RAD_INT("head_dim", 256), RAD_INT("n_head", 24), RAD_INT("gqa", 12),
            RAD_INT("block_size", 4), RAD_INT("causal", 1), RAD_INT("window", 0), RAD_STR("q_dtype", "bf16"),
            RAD_STR("kv_dtype", "i8"), RAD_INT("group", 128), RAD_STR("act", "sigmoid"), RAD_INT("max_seqs", 256),
            RAD_INT("max_ctx", 4096)};
        RadParam quant_params[]{RAD_INT("M", rows * 24), RAD_INT("n", 256), RAD_INT("group", 128),
            RAD_STR("dtype", "bf16"), RAD_STR("act", "sigmoid")};
        RadTensor gate_view = qg.device;
        gate_view.data = static_cast<uint16_t*>(qg.device.data) + 256;
        gate_view.shape[0] = rows; gate_view.shape[1] = 256;
        RadTensor gate_rows = gate_view;
        gate_rows.shape[0] = rows * 24; gate_rows.stride[0] = 512;
        auto flat = [&](const RadTensor& tensor, int64_t width_) {
            RadTensor result = tensor; result.shape[0] = rows * 24; result.shape[1] = width_; result.stride[0] = width_;
            return result;
        };
        RadTensor attention[]{query.device, cache.device, table.device, used.device, {}, {}, {}, output.device};
        RadTensor quant[]{gate_rows, flat(output.device, 256), flat(codes.device, 256), flat(scales.device, 2),
            flat(output.device, 256)};
        RadTensor fused[]{query.device, cache.device, table.device, used.device, {}, {}, {}, output.device, gate_view,
            codes.device, scales.device};
        RadArgs args{}; args.p = params; args.n_p = int(sizeof(params) / sizeof(*params));
        RadArgs quant_args{}; quant_args.t = quant; quant_args.n_t = 5; quant_args.p = quant_params; quant_args.n_p = 5;
        auto pair = [&] {
            args.t = attention; args.n_t = 8; REQUIRE_EQ(attend(&args, compute.stream), RAD_OK);
            REQUIRE_EQ(gate_quant(&quant_args, compute.stream), RAD_OK);
        };
        auto one = [&] { args.t = fused; args.n_t = 11; REQUIRE_EQ(paged_gate(&args, compute.stream), RAD_OK); };
        std::vector<uint8_t> results[2][3];
        float micros[2]{};
        for (int variant : {0, 1, 1, 0}) {
            for (Buffer* buffer : {&cache, &query, &qg, &output, &codes, &scales, &table, &used}) buffer->push(compute.stream);
            if (variant) one(); else pair();
            std::vector<uint8_t> got[3]{output.read(compute.stream), codes.read(compute.stream), scales.read(compute.stream)};
            for (int index = 0; index < 3; ++index) {
                if (results[variant][index].empty()) results[variant][index] = got[index];
                else REQUIRE(got[index] == results[variant][index]);
            }
            if (variant) time(one, micros[1]); else time(pair, micros[0]);
        }
        for (int index = 0; index < 3; ++index) REQUIRE(results[0][index] == results[1][index]);
        std::printf("TP1_ATTN_GATE_AB rows=%d keys=%d pair_us=%.1f fused_us=%.1f identical=1\n", rows, used_tokens,
            micros[0] / 2, micros[1] / 2);
        ++cases;
    }
    setenv("RAD_ATTN_PAGED_FAST", "0", 1); setenv("RAD_QK_PREPARE_PARALLEL", "0", 1); setenv("RAD_KV_STORE_PARALLEL", "0", 1);
    rad_event_destroy(start); rad_event_destroy(finish);
    std::printf("PASS TP1_ATTN_FUSED_AB cases=%d prepare+store and attention+gate bit-identical\n", cases);
}

TEST(r3d_tp1_qsa_score_work_ab) {
    if (!std::getenv("RAD_TP1_QSA_AB")) SKIP("optional QSA score/work A/B");
    RadDeviceProps props{};
    REQUIRE_EQ(rad_dev_props(0, &props), RAD_OK);
    REQUIRE(!props.is_host_backend);
    REQUIRE_EQ(rad_dev_set(0), RAD_OK);
    Library native("libr3d");
    const auto score = reinterpret_cast<RadLaunchFn>(dlsym(native.handle, "r3dh_sparse_qsa_score"));
    const auto work = reinterpret_cast<RadLaunchFn>(dlsym(native.handle, "r3dh_attn_qsa_work"));
    REQUIRE(score && work);
    Queue compute(0);
    RadEvent start{}, finish{};
    REQUIRE_EQ(rad_event_create(&start), RAD_OK); REQUIRE_EQ(rad_event_create(&finish), RAD_OK);
    uint32_t state = 20261122u;
    auto random = [&] { state = state * 1664525u + 1013904223u; return state >> 8; };
    auto time = [&](RadLaunchFn launch, RadArgs* args, float& total) {
        for (int warm = 0; warm < 3; ++warm) REQUIRE_EQ(launch(args, compute.stream), RAD_OK);
        REQUIRE_EQ(rad_event_record(start, compute.stream), RAD_OK);
        for (int repeat = 0; repeat < 20; ++repeat) REQUIRE_EQ(launch(args, compute.stream), RAD_OK);
        REQUIRE_EQ(rad_event_record(finish, compute.stream), RAD_OK);
        REQUIRE_EQ(rad_event_sync(finish), RAD_OK);
        float elapsed = 0;
        REQUIRE_EQ(rad_event_elapsed_ms(start, finish, &elapsed), RAD_OK);
        total += elapsed * 25.0f;
    };
    int cases = 0;
    for (int columns : {62, 520, 3000}) for (int rows : {1, 3}) {
        const int sequences = rows == 3 ? 2 : 1, pages = 4000;
        Buffer query(RAD_BF16, {rows, 512}, RAD_MEM_DEVICE), keys(RAD_BF16, {pages, 1, 4, 32}, RAD_MEM_DEVICE);
        Buffer table(RAD_I32, {sequences, columns}, RAD_MEM_DEVICE), cu(RAD_I32, {sequences + 1}, RAD_MEM_DEVICE);
        Buffer count(RAD_I32, {sequences}, RAD_MEM_DEVICE), output(RAD_F32, {rows, columns}, RAD_MEM_DEVICE);
        for (int index = 0; index < rows * 512; ++index)
            query.data<uint16_t>()[index] = rad_f32_to_bf16(float(int(random() % 2001) - 1000) / 64.0f);
        for (int index = 0; index < pages * 128; ++index)
            keys.data<uint16_t>()[index] = rad_f32_to_bf16(float(int(random() % 2001) - 1000) / 128.0f);
        for (int index = 0; index < sequences * columns; ++index)
            table.data<int32_t>()[index] = random() % 40 == 0 ? -1 : random() % 60 == 0 ? pages + 3 : int(random() % pages);
        cu.data<int32_t>()[0] = 0; cu.data<int32_t>()[sequences] = rows;
        if (sequences == 2) cu.data<int32_t>()[1] = 1;
        for (int sequence = 0; sequence < sequences; ++sequence)
            count.data<int32_t>()[sequence] = sequence ? columns - 7 : columns;
        for (Buffer* buffer : {&query, &keys, &table, &cu, &count, &output}) buffer->push(compute.stream);
        RadParam params[]{RAD_INT("n", 128), RAD_INT("heads", 4), RAD_STR("dtype", "bf16")};
        RadTensor tensors[]{query.device, keys.device, table.device, cu.device, count.device, output.device};
        RadArgs args{}; args.t = tensors; args.n_t = 6; args.p = params; args.n_p = 3;
        std::vector<uint8_t> results[2];
        float micros[2]{};
        for (int variant : {0, 1, 1, 0}) {
            setenv("RAD_QSA_SCORE_PARALLEL", variant ? "1" : "0", 1);
            output.push(compute.stream);
            REQUIRE_EQ(score(&args, compute.stream), RAD_OK);
            auto got = output.read(compute.stream);
            if (results[variant].empty()) results[variant] = std::move(got); else REQUIRE(got == results[variant]);
            time(score, &args, micros[variant]);
        }
        REQUIRE(results[0] == results[1]);
        REQUIRE(results[0] != output.expected);
        std::printf("TP1_QSA_SCORE_AB columns=%d rows=%d strict_us=%.1f parallel_us=%.1f identical=1\n",
            columns, rows, micros[0], micros[1]);
        ++cases;
    }
    for (int rows : {1, 2, 5, 64}) for (int ring : {4, 7}) {
        const int sequences = rows >= 5 ? 2 : 1, width = 700, slots = 6;
        const int items = (rows + 3) / 4 + sequences;
        Buffer key(RAD_BF16, {rows, 128}, RAD_MEM_DEVICE, 640), positions(RAD_I32, {rows}, RAD_MEM_DEVICE);
        Buffer cu(RAD_I32, {sequences + 1}, RAD_MEM_DEVICE), tail(RAD_BF16, {slots, 1, ring, 128}, RAD_MEM_DEVICE);
        Buffer sidx(RAD_I32, {sequences}, RAD_MEM_DEVICE), table(RAD_I32, {sequences, width}, RAD_MEM_DEVICE);
        Buffer stage(RAD_BF16, {items * 4, 128}, RAD_MEM_DEVICE), page(RAD_I32, {items}, RAD_MEM_DEVICE);
        Buffer bpos(RAD_I32, {items}, RAD_MEM_DEVICE), complete(RAD_I32, {sequences}, RAD_MEM_DEVICE);
        for (int index = 0; index < rows * 640; ++index) key.data<uint16_t>()[index] = uint16_t(random());
        for (int index = 0; index < slots * ring * 128; ++index) tail.data<uint16_t>()[index] = uint16_t(random());
        for (int index = 0; index < sequences * width; ++index) table.data<int32_t>()[index] = int(random() % 5000);
        const int split = sequences == 2 ? rows / 2 : rows;
        cu.data<int32_t>()[0] = 0; cu.data<int32_t>()[sequences] = rows;
        if (sequences == 2) cu.data<int32_t>()[1] = split;
        for (int sequence = 0; sequence < sequences; ++sequence) {
            sidx.data<int32_t>()[sequence] = sequence ? 4 : 1;
            const int begin = sequence ? split : 0, end = sequence ? rows : split, first = sequence ? 44 - (end - begin) : 252 - (end - begin);
            for (int row = begin; row < end; ++row) positions.data<int32_t>()[row] = first + row - begin;
        }
        for (Buffer* buffer : {&key, &positions, &cu, &tail, &sidx, &table, &stage, &page, &bpos, &complete})
            buffer->push(compute.stream);
        RadParam params[]{RAD_INT("M", rows), RAD_INT("n", 128), RAD_INT("ratio", 4), RAD_INT("ring", ring),
            RAD_INT("seqs", sequences), RAD_INT("work", items), RAD_STR("dtype", "bf16")};
        RadTensor tensors[]{key.device, positions.device, cu.device, tail.device, sidx.device, table.device, {},
            stage.device, page.device, bpos.device, complete.device};
        tensors[0].shape[1] = 128;
        RadArgs args{}; args.t = tensors; args.n_t = 11; args.p = params; args.n_p = 7;
        std::vector<uint8_t> results[2][4];
        float micros[2]{};
        for (int variant : {0, 1, 1, 0}) {
            setenv("RAD_QSA_WORK_PARALLEL", variant ? "1" : "0", 1);
            for (Buffer* buffer : {&stage, &page, &bpos, &complete}) buffer->push(compute.stream);
            REQUIRE_EQ(work(&args, compute.stream), RAD_OK);
            std::vector<uint8_t> got[4]{stage.read(compute.stream), page.read(compute.stream),
                bpos.read(compute.stream), complete.read(compute.stream)};
            for (int index = 0; index < 4; ++index) {
                if (results[variant][index].empty()) results[variant][index] = got[index];
                else REQUIRE(got[index] == results[variant][index]);
            }
            time(work, &args, micros[variant]);
        }
        for (int index = 0; index < 4; ++index) REQUIRE(results[0][index] == results[1][index]);
        REQUIRE(results[0][0] != stage.expected);
        std::printf("TP1_QSA_WORK_AB rows=%d ring=%d items=%d strict_us=%.1f parallel_us=%.1f identical=1\n",
            rows, ring, items, micros[0], micros[1]);
        ++cases;
    }
    setenv("RAD_QSA_SCORE_PARALLEL", "0", 1); setenv("RAD_QSA_WORK_PARALLEL", "0", 1);
    rad_event_destroy(start); rad_event_destroy(finish);
    std::printf("PASS TP1_QSA_AB cases=%d score+work bit-identical\n", cases);
}

// Split-KV attention (RAD_ATTN_SPLIT, tolerance) against the exact gated kernel. Gate, fixed before any run:
// per (row, head) relative L2 of the gated BF16 output <= 1e-3; every element within 2^-6 |exact| (two BF16 ulps)
// plus 1e-6 of the head's peak; every int8 code within 1; every group scale within 1e-3 relative.
TEST(r3d_tp1_attention_split_gate) {
    if (!std::getenv("RAD_TP1_ATTN_SPLIT_AB")) SKIP("optional split-KV attention gate");
    RadDeviceProps props{};
    REQUIRE_EQ(rad_dev_props(0, &props), RAD_OK);
    REQUIRE(!props.is_host_backend);
    REQUIRE_EQ(rad_dev_set(0), RAD_OK);
    Library native("libr3d");
    const auto paged_gate = reinterpret_cast<RadLaunchFn>(dlsym(native.handle, "r3dh_attn_qsa_paged_gate"));
    REQUIRE(paged_gate);
    Queue compute(0);
    RadEvent start{}, finish{};
    REQUIRE_EQ(rad_event_create(&start), RAD_OK); REQUIRE_EQ(rad_event_create(&finish), RAD_OK);
    uint32_t state = 20261124u;
    auto random = [&] { state = state * 1664525u + 1013904223u; return state >> 8; };
    const int64_t scratch_bytes = 512LL * 2 * 12 * 258 * 4;
    Buffer scratch(RAD_I32, {scratch_bytes / 4}, RAD_MEM_DEVICE);
    double worst_l2 = 0, worst_element = 0, worst_violation = 0;
    int worst_code = 0, cases = 0;
    long l2_violations = 0, element_violations = 0, code_violations = 0, scale_violations = 0;
    for (float query_scale : {64.0f, 1000.0f}) for (int rows : {1, 3}) for (int used_tokens : {1, 5, 63, 250, 600, 2051}) {
        const int pages = 1100, width = 513;
        Buffer cache(RAD_I8, {pages, 2, 4, 516}, RAD_MEM_DEVICE);
        Buffer query(RAD_BF16, {rows, 6144}, RAD_MEM_DEVICE), qg(RAD_BF16, {rows, 12288}, RAD_MEM_DEVICE);
        Buffer output(RAD_BF16, {rows, 6144}, RAD_MEM_DEVICE), codes(RAD_I8, {rows, 6144}, RAD_MEM_DEVICE);
        Buffer scales(RAD_F32, {rows, 48}, RAD_MEM_DEVICE);
        Buffer table(RAD_I32, {rows, width}, RAD_MEM_DEVICE), used(RAD_I32, {rows}, RAD_MEM_DEVICE);
        for (size_t index = 0; index + 128 < cache.expected.size(); ++index) cache.expected[64 + index] = uint8_t(random());
        for (int page = 0; page < pages; ++page)
            for (int entry = 0; entry < 16; ++entry) {
                const uint16_t scale = rad_f32_to_f16(float(random() % 1000 + 1) / 65536.0f);
                std::memcpy(cache.data<uint8_t>() + (size_t(page) * 8 + entry / 2) * 516 + 512 + (entry % 2) * 2, &scale, 2);
            }
        for (int index = 0; index < rows * width; ++index)
            table.data<int32_t>()[index] = random() % 50 == 0 ? -1 : int(random() % pages);
        for (int row = 0; row < rows; ++row) used.data<int32_t>()[row] = row == 1 ? std::max(1, used_tokens / 3) : used_tokens;
        for (int index = 0; index < rows * 6144; ++index)
            query.data<uint16_t>()[index] = rad_f32_to_bf16(float(int(random() % 2001) - 1000) / query_scale);
        for (int index = 0; index < rows * 12288; ++index)
            qg.data<uint16_t>()[index] = rad_f32_to_bf16(float(int(random() % 2001) - 1000) / 128.0f);
        RadParam params[]{RAD_INT("q_len", 1), RAD_INT("head_dim", 256), RAD_INT("n_head", 24), RAD_INT("gqa", 12),
            RAD_INT("block_size", 4), RAD_INT("causal", 1), RAD_INT("window", 0), RAD_STR("q_dtype", "bf16"),
            RAD_STR("kv_dtype", "i8"), RAD_INT("group", 128), RAD_STR("act", "sigmoid"), RAD_INT("max_seqs", 256),
            RAD_INT("max_ctx", 4096)};
        RadTensor gate_view = qg.device;
        gate_view.data = static_cast<uint16_t*>(qg.device.data) + 256;
        gate_view.shape[0] = rows; gate_view.shape[1] = 256;
        RadTensor tensors[]{query.device, cache.device, table.device, used.device, {}, {}, {}, output.device, gate_view,
            codes.device, scales.device};
        RadArgs args{}; args.t = tensors; args.n_t = 11; args.p = params; args.n_p = int(sizeof(params) / sizeof(*params));
        args.scratch = scratch.device.data; args.scratch_bytes = scratch_bytes;
        std::vector<uint8_t> results[2][3];
        float micros[2]{};
        setenv("RAD_ATTN_PAGED_FAST", "1", 1);
        for (int variant : {0, 1, 1, 0}) {
            setenv("RAD_ATTN_SPLIT", variant ? "1" : "0", 1);
            for (Buffer* buffer : {&cache, &query, &qg, &output, &codes, &scales, &table, &used}) buffer->push(compute.stream);
            REQUIRE_EQ(paged_gate(&args, compute.stream), RAD_OK);
            std::vector<uint8_t> got[3]{output.read(compute.stream), codes.read(compute.stream), scales.read(compute.stream)};
            for (int index = 0; index < 3; ++index) {
                if (results[variant][index].empty()) results[variant][index] = got[index];
                else REQUIRE(got[index] == results[variant][index]);
            }
            for (int warm = 0; warm < 3; ++warm) REQUIRE_EQ(paged_gate(&args, compute.stream), RAD_OK);
            REQUIRE_EQ(rad_event_record(start, compute.stream), RAD_OK);
            for (int repeat = 0; repeat < 20; ++repeat) REQUIRE_EQ(paged_gate(&args, compute.stream), RAD_OK);
            REQUIRE_EQ(rad_event_record(finish, compute.stream), RAD_OK);
            REQUIRE_EQ(rad_event_sync(finish), RAD_OK);
            float elapsed = 0;
            REQUIRE_EQ(rad_event_elapsed_ms(start, finish, &elapsed), RAD_OK);
            micros[variant] += elapsed * 25.0f;
        }
        unsetenv("RAD_ATTN_SPLIT");
        const uint16_t* exact = reinterpret_cast<const uint16_t*>(results[0][0].data() + 64);
        const uint16_t* split = reinterpret_cast<const uint16_t*>(results[1][0].data() + 64);
        const int8_t* exact_codes = reinterpret_cast<const int8_t*>(results[0][1].data() + 64);
        const int8_t* split_codes = reinterpret_cast<const int8_t*>(results[1][1].data() + 64);
        const float* exact_scales = reinterpret_cast<const float*>(results[0][2].data() + 64);
        const float* split_scales = reinterpret_cast<const float*>(results[1][2].data() + 64);
        auto widen = [](uint16_t bits) { uint32_t word = uint32_t(bits) << 16; float value; std::memcpy(&value, &word, 4); return double(value); };
        for (int row = 0; row < rows; ++row) for (int head = 0; head < 24; ++head) {
            double error = 0, norm = 0, peak = 0;
            for (int dim = 0; dim < 256; ++dim) peak = std::max(peak, std::fabs(widen(exact[row * 6144 + head * 256 + dim])));
            for (int dim = 0; dim < 256; ++dim) {
                const int index = row * 6144 + head * 256 + dim;
                const double reference = widen(exact[index]), difference = std::fabs(widen(split[index]) - reference);
                error += difference * difference; norm += reference * reference;
                if (!(difference <= std::ldexp(std::fabs(reference), -6) + 1e-6 * peak)) {
                    ++element_violations;
                    worst_violation = std::max(worst_violation, peak > 0 ? difference / peak : difference);
                }
                worst_element = std::max(worst_element, peak > 0 ? difference / peak : 0.0);
                const int code_difference = std::abs(int(split_codes[index]) - int(exact_codes[index]));
                worst_code = std::max(worst_code, code_difference);
                code_violations += code_difference > 1;
            }
            const double l2 = norm > 0 ? std::sqrt(error / norm) : std::sqrt(error);
            l2_violations += l2 > 1e-3;
            if (l2 > 5e-4) std::printf("TP1_ATTN_SPLIT_L2 query_scale=%g keys=%d row=%d head=%d rel_l2=%.3g norm=%.3g peak=%.3g\n",
                query_scale, used_tokens, row, head, l2, std::sqrt(norm), peak);
            worst_l2 = std::max(worst_l2, l2);
            for (int group = 0; group < 2; ++group) {
                const int index = row * 48 + head * 2 + group;
                scale_violations += std::fabs(split_scales[index] - exact_scales[index]) > 1e-3 * std::fabs(exact_scales[index]);
            }
        }
        std::printf("TP1_ATTN_SPLIT query_scale=%g rows=%d keys=%d exact_us=%.1f split_us=%.1f\n", query_scale, rows,
            used_tokens, micros[0], micros[1]);
        ++cases;
    }
    setenv("RAD_ATTN_PAGED_FAST", "0", 1);
    rad_event_destroy(start); rad_event_destroy(finish);
    std::printf("TP1_ATTN_SPLIT_GATE cases=%d worst_rel_l2=%.3g (violations %ld) element_violations=%ld "
        "(worst %.3g of peak) worst_element_over_peak=%.3g worst_code=%d (violations %ld) scale_violations=%ld\n",
        cases, worst_l2, l2_violations, element_violations, worst_violation, worst_element, worst_code, code_violations,
        scale_violations);
    REQUIRE(l2_violations == 0 && element_violations == 0 && code_violations == 0 && scale_violations == 0);
}

TEST(r3d_tp1_i8_split_ab) {
    if (!std::getenv("RAD_TP1_I8_AB")) SKIP("optional I8 split GEMV A/B");
    RadDeviceProps props{};
    REQUIRE_EQ(rad_dev_props(0, &props), RAD_OK);
    REQUIRE(!props.is_host_backend);
    REQUIRE_EQ(rad_dev_set(0), RAD_OK);
    Library native("libr3d");
    const auto gemm = reinterpret_cast<RadLaunchFn>(dlsym(native.handle, "r3dh_i8_rad_gemm_nt_q"));
    REQUIRE(gemm);
    Queue compute(0);
    RadEvent start{}, finish{};
    REQUIRE_EQ(rad_event_create(&start), RAD_OK); REQUIRE_EQ(rad_event_create(&finish), RAD_OK);
    uint32_t state = 7u;
    auto random = [&] { state = state * 1664525u + 1013904223u; return state >> 8; };
    const std::pair<int, int> shapes[]{{2560, 6144}, {1280, 2560}, {2560, 640}, {512, 2560}, {10240, 2560}, {2560, 16384}};
    for (auto [columns, reduction] : shapes)
        for (int rows : {1, 2}) {
            Buffer input(RAD_I8, {rows, reduction}, RAD_MEM_DEVICE);
            Buffer input_scale(RAD_F32, {rows, reduction / 128}, RAD_MEM_DEVICE);
            Buffer weight(r3dh_i8::weight_dtype, {columns, reduction}, RAD_MEM_DEVICE);
            Buffer weight_scale(r3dh_i8::scale_dtype, {columns, reduction / 128}, RAD_MEM_DEVICE);
            Buffer output(RAD_BF16, {rows, columns}, RAD_MEM_DEVICE);
            for (size_t index = 64; index + 64 < input.expected.size(); ++index) input.expected[index] = uint8_t(random());
            for (size_t index = 64; index + 64 < weight.expected.size(); ++index) weight.expected[index] = uint8_t(random());
            for (int index = 0; index < rows * reduction / 128; ++index)
                input_scale.data<float>()[index] = float(random() % 1000 + 1) / 8192.0f;
            for (int index = 0; index < columns * reduction / 128; ++index)
                weight_scale.data<float>()[index] = float(int(random() % 2001) - 1000) / 65536.0f;
            for (Buffer* buffer : {&input, &input_scale, &weight, &weight_scale, &output}) buffer->push(compute.stream);
            RadTensor tensors[]{input.device, input_scale.device, weight.device, weight_scale.device, output.device};
            RadParam params[]{RAD_INT("M", rows), RAD_INT("N", columns), RAD_INT("K", reduction), RAD_INT("group", 128),
                RAD_STR("dtype", "i8a8h")};
            RadArgs args{}; args.t = tensors; args.n_t = 5; args.p = params; args.n_p = 5;
            std::vector<uint8_t> outputs[2];
            float micros[2]{};
            for (int variant : {0, 1, 1, 0}) {
                REQUIRE_EQ(setenv("RAD_I8_SPLIT", variant ? "1" : "0", 1), 0);
                output.push(compute.stream);
                REQUIRE_EQ(gemm(&args, compute.stream), RAD_OK);
                auto got = output.read(compute.stream);
                if (outputs[variant].empty()) outputs[variant] = std::move(got); else REQUIRE(got == outputs[variant]);
                REQUIRE_EQ(rad_event_record(start, compute.stream), RAD_OK);
                for (int repeat = 0; repeat < 20; ++repeat) REQUIRE_EQ(gemm(&args, compute.stream), RAD_OK);
                REQUIRE_EQ(rad_event_record(finish, compute.stream), RAD_OK);
                REQUIRE_EQ(rad_event_sync(finish), RAD_OK);
                float elapsed = 0;
                REQUIRE_EQ(rad_event_elapsed_ms(start, finish, &elapsed), RAD_OK);
                micros[variant] += elapsed * 25.0f;
            }
            REQUIRE(outputs[0] == outputs[1]);
            std::printf("TP1_I8_AB N=%d K=%d M=%d wave_us=%.1f split_us=%.1f speedup=%.2f identical=1\n",
                columns, reduction, rows, micros[0], micros[1], micros[0] / micros[1]);
        }
    setenv("RAD_I8_SPLIT", "0", 1);
    rad_event_destroy(start); rad_event_destroy(finish);
}

TEST(r3d_mtp_i8_four_exact_tp2_aql) {
    using joined_head_test::Storage;
    using joined_head_test::matrix;
    if (!joined_head_test::device_available()) SKIP("explicit gfx1100 card run required");
    struct SavedFlag {
        const char* name;
        bool present = std::getenv(name) != nullptr;
        std::string saved = present ? std::getenv(name) : "";
        ~SavedFlag() { if (present) setenv(name, saved.c_str(), 1); else unsetenv(name); }
    } four{"RAD_MTP_I8_FOUR"}, precision{"RAD_PRECISION"};
    REQUIRE_EQ(setenv(precision.name, "exact", 1), 0);
    Library native("libr3d");
    const auto gemm = reinterpret_cast<RadLaunchFn>(dlsym(native.handle, "r3dh_i8_rad_gemm_nt_q"));
    const auto info = reinterpret_cast<const RadPluginInfo* (*)()>(dlsym(native.handle, "rad_plugin_info"));
    REQUIRE(info && info() && info()->build_target);
    const bool single_target = !std::strcmp(info()->build_target, "gfx1100");
    const auto* backend = rad::active_backend();
    REQUIRE(gemm && backend && backend->tape_begin && backend->tape_end && backend->tape_len
        && backend->tape_diff && backend->tape_destroy);
    struct Tape {
        const rad::DeviceBackend* backend;
        rad::RadTape value = nullptr;
        ~Tape() { if (value) backend->tape_destroy(value); }
    };
    Queue compute(0);
    auto trial = [&](int columns, int reduction, int rows, int world, int rank, const char* dtype,
                     int weight_offset, bool edge, bool eligible) {
        const int groups = reduction / 128, pad = rank == 1 || edge || weight_offset ? 1 : 0;
        const int input_offset = 4 * pad, input_pitch = reduction + 4 * pad;
        const int scale_pitch = groups + 3 * pad, output_pitch = columns + 7 * pad;
        Storage input(input_offset + rows * input_pitch, RAD_MEM_DEVICE);
        Storage weight(weight_offset + size_t(columns) * reduction, RAD_MEM_DEVICE);
        Buffer input_scale(RAD_F32, {rows, groups}, RAD_MEM_DEVICE, scale_pitch);
        Buffer weight_scale(r3dh_i8::scale_dtype, {columns, groups}, RAD_MEM_DEVICE);
        Buffer output(RAD_BF16, {rows + 1, columns}, RAD_MEM_DEVICE, output_pitch);
        auto* codes = reinterpret_cast<int8_t*>(input.expected.data() + 64 + input_offset);
        auto* weights = reinterpret_cast<int8_t*>(weight.expected.data() + 64 + weight_offset);
        uint32_t state = 19u + rank;
        auto next = [&] { state = state * 1664525u + 1013904223u; return state; };
        auto factor = [&] { return std::ldexp(float(0x800000u | (next() >> 9)), -30); };
        for (int row = 0; row < rows; ++row) for (int group = 0; group < groups; ++group) {
            auto* chunk = codes + row * input_pitch + group * 128;
            for (int pos = 0; pos < 128; ++pos) chunk[pos] = int(next() >> 24) - 128;
            chunk[0] = 1; chunk[1] = row & 1 ? -128 : 127; chunk[127] = row + group;
            input_scale.data<float>()[row * scale_pitch + group] = edge && row == 0
                ? (group == 1 ? 0x1.000002p0f : 1.0f) : factor();
        }
        for (size_t index = 0; index < size_t(columns) * reduction; ++index)
            weights[index] = int(next() >> 24) - 128;
        for (int col = 0; col < columns; ++col) for (int group = 0; group < groups; ++group) {
            float value = ((col + group) & 1 ? -1.0f : 1.0f) * factor();
            if (edge && col < 9) {
                for (int pos = 0; pos < 128; ++pos)
                    weights[r3dh_i8::packed_index(col, group * 128 + pos, reduction)] = pos == 0;
                const float first[]{-0.0f, INFINITY, -INFINITY, std::numeric_limits<float>::quiet_NaN(),
                    0x1.01p0f, 0x1.03p0f, 0x1p24f, 0x1p-40f, -1.0f};
                value = group == 0 ? first[col] : 0.0f;
                if (col == 6 && group == 2) value = 1.0f;
                if (col == 6 && group == 3) value = -0x1p24f;
                if (col == 8 && group == 1) value = 0x1.fffffcp-1f;
            }
            weight_scale.data<float>()[col * groups + group] = value;
        }
        input.push(compute.stream); weight.push(compute.stream);
        input_scale.push(compute.stream); weight_scale.push(compute.stream);
        auto* activation = static_cast<uint8_t*>(input.device()) + input_offset;
        auto* packed = static_cast<uint8_t*>(weight.device()) + weight_offset;
        REQUIRE_EQ(reinterpret_cast<uintptr_t>(activation) % 16, uintptr_t(input_offset));
        REQUIRE_EQ(reinterpret_cast<uintptr_t>(packed) % 16, uintptr_t(weight_offset));
        RadTensor tensors[]{matrix(activation, RAD_I8, rows, reduction, input_pitch), input_scale.device,
            matrix(packed, r3dh_i8::weight_dtype, columns, reduction), weight_scale.device, output.device};
        tensors[4].shape[0] = rows;
        RadParam params[]{RAD_INT("M", 64), RAD_INT("N", columns), RAD_INT("K", reduction),
            RAD_INT("group", 128), RAD_STR("dtype", dtype)};
        RadArgs args{}; args.t = tensors; args.n_t = 5; args.p = params; args.n_p = 5;
        args.world_size = world; args.rank = rank;
        Tape reference{backend};
        auto golden = output.expected;
        const char* settings[]{nullptr, "0", "01", "1"};
        for (const char* setting : settings) {
            REQUIRE_EQ(setting ? setenv(four.name, setting, 1) : unsetenv(four.name), 0);
            output.push(compute.stream);
            REQUIRE_EQ(rad_stream_sync(compute.stream), RAD_OK);
            REQUIRE_EQ(backend->tape_begin(), RAD_OK);
            const int status = gemm(&args, compute.stream);
            Tape actual{backend, backend->tape_end(status == RAD_OK)};
            REQUIRE_EQ(status, RAD_OK); REQUIRE(actual.value); REQUIRE_EQ(backend->tape_len(actual.value), 1);
            const auto got = output.read(compute.stream);
            if (!setting) {
                std::swap(reference.value, actual.value);
                for (int row = 0; row < rows; ++row) {
                    const size_t begin = 64 + size_t(row) * output_pitch * 2;
                    std::copy_n(got.data() + begin, columns * 2, golden.data() + begin);
                    for (int byte = 0; byte < columns * 2; ++byte) output.expected[begin + byte] = ~got[begin + byte];
                    for (int col : {0, 1, 2, 3, 4, 5, 6, 7, 8, 15, 16, columns / 2, columns - 1}) {
                        float sum = 0.0f;
                        for (int group = 0; group < groups; ++group) {
                            int dot = 0;
                            for (int pos = group * 128; pos < (group + 1) * 128; ++pos)
                                dot += int(codes[row * input_pitch + pos]) * weights[r3dh_i8::packed_index(col, pos, reduction)];
                            const float scaled = joined_head_test::multiply(float(dot), input_scale.data<float>()[row * scale_pitch + group]);
                            sum = joined_head_test::add(sum, joined_head_test::multiply(scaled, weight_scale.data<float>()[col * groups + group]));
                        }
                        uint16_t bits;
                        std::memcpy(&bits, got.data() + begin + col * 2, sizeof(bits));
                        if (std::isnan(sum)) REQUIRE((bits & 0x7fffu) > 0x7f80u);
                        else REQUIRE_EQ(bits, rad_f32_to_bf16(sum));
                    }
                }
            } else {
                char why[512]{};
                const bool changed = single_target && eligible && !std::strcmp(setting, "1");
                const int difference = backend->tape_diff(reference.value, actual.value, why, sizeof(why));
                if (difference != (changed ? 0 : -1))
                    std::printf("I8_FOUR M=%d N=%d K=%d world=%d rank=%d flag=%s diff=%d %s\n",
                        rows, columns, reduction, world, rank, setting, difference, why);
                REQUIRE_EQ(difference, changed ? 0 : -1);
                if (changed) {
                    const auto grids = "grid " + std::to_string(columns * 2) + "x1x1 against "
                        + std::to_string(columns * 16) + "x1x1";
                    REQUIRE(std::strstr(why, grids.c_str()));
                }
            }
            REQUIRE(got == golden);
        }
    };
    const std::pair<int, int> shapes[]{{256, 2560}, {512, 2560}, {768, 2560},
        {2560, 256}, {2560, 384}, {2560, 3072}};
    for (int rank : {0, 1}) for (auto [columns, reduction] : shapes)
        trial(columns, reduction, 4, 2, rank, "i8a8h", 0, false, true);
    trial(2560, 3072, 4, 2, 1, "i8a8h", 0, true, true);
    for (int rows : {1, 2, 3, 5}) trial(256, 2560, rows, 2, 0, "i8a8h", 0, false, false);
    trial(256, 2560, 4, 1, 0, "i8a8h", 0, false, false);
    trial(256, 2560, 4, 2, 1, "i8a8", 0, false, false);
    trial(256, 2560, 4, 2, 1, "i8a8h", 4, false, false);
    trial(128, 2560, 4, 2, 0, "i8a8h", 0, false, false);
    trial(2560, 512, 4, 2, 0, "i8a8h", 0, false, false);
    trial(2560, 3200, 4, 2, 1, "i8a8h", 0, false, false);
}
