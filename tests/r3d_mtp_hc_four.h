TEST(r3d_tp2_mtp_hc_four_rows) {
    using namespace joined_head_test;
    if (!device_available()) SKIP("gfx1100 required");
    struct Flag {
        const char* key;
        bool present;
        std::string saved;
        Flag(const char* name, const char* value) : key(name), present(std::getenv(key) != nullptr),
            saved(present ? std::getenv(key) : "") { set(value); }
        void set(const char* value) { if (value) setenv(key, value, 1); else unsetenv(key); }
        ~Flag() { set(present ? saved.c_str() : nullptr); }
    } four("RAD_MTP_HC_FOUR", nullptr), precision("RAD_PRECISION", "tolerance"),
        split("RAD_HC_UP_SPLIT", "1"), upmix("RAD_HC_UPMIX_EXACT", "1"), down_split("RAD_HC_DOWN_SPLIT", "1"),
        norm("RAD_HC_NORM_PARALLEL", "1"), fused("RAD_HC_FUSED", "0"), tile("RAD_HC_EXACT_TILE", "0"),
        wmma("RAD_HC_WMMA", "0"), fused_mix("RAD_HC_UPMIX_FUSED", "0"), matrix_down("RAD_HC_MATRIX_DOWN", "0"),
        fused_inject("RAD_HC_NORM_INJECT_FUSED", "0"), chain("RAD_HC_DOWN_CHAIN", "1");
    constexpr int width = 2560, streams = 4, wide = streams * width, rank = 320, band = 16;
    Library native("libr3d");
    auto plain = reinterpret_cast<int (*)(const RadArgs*, RadStream)>(dlsym(native.handle, "r3dh_hc_rad_hc_read_strict"));
    auto traced = reinterpret_cast<int (*)(const RadArgs*, RadStream, const R3dhHcTrace*)>(
        dlsym(native.handle, "r3dh_hc_rad_hc_read_strict_trace"));
    auto scratch_size = reinterpret_cast<int64_t (*)(const RadArgs*)>(dlsym(native.handle, "r3dh_hc_rad_hc_read_strict_scratch"));
    auto relayout = reinterpret_cast<RadRelayoutFn>(dlsym(native.handle, "r3dh_hc_rad_relayout_hc_e4m3"));
    REQUIRE(plain && traced && scratch_size && relayout);
    Model model("/home/zihaomu/bigssd2/radiance_workspace/models/qwen3.8-next-flash-iqr-v4.rad");
    Queue queue(0);
    Storage down(size_t(rank) * 10560, RAD_MEM_DEVICE), up(size_t(wide) * 336, RAD_MEM_DEVICE);
    Buffer gains(RAD_F32, {wide}, RAD_MEM_DEVICE), inject_weights(RAD_BF16, {streams, wide}, RAD_MEM_DEVICE);
    RadParam params[]{RAD_INT("M", band), RAD_INT("n", width), RAD_INT("hc", streams), RAD_INT("lowrank", rank),
        RAD_STR("dtype", "bf16"), RAD_STR("mix", "e4m3"), RAD_INT("inject", 1), RAD_INT("group", 128), RAD_INT("rotate", 1)};
    for (bool actual : {false, true}) {
        auto entry_for = [&](const char* role) -> decltype(auto) {
            const auto name = std::string("blk.0.attn_hc_") + role + ".weight";
            auto it = std::find_if(model.entries.begin(), model.entries.end(), [&](const auto& entry) { return model.name(entry) == name; });
            if (it == model.entries.end()) throw std::runtime_error("missing HC fixture: " + name);
            return *it;
        };
        for (bool is_down : {true, false}) {
            const int depth = is_down ? wide : rank, outputs = is_down ? rank : wide, group = is_down ? 128 : 80;
            const int groups = depth / group, pitch = depth + groups * 4;
            auto& packed = is_down ? down : up;
            auto* dst = packed.expected.data() + 64;
            if (actual) {
                const auto& entry = entry_for(is_down ? "down" : "up");
                REQUIRE_EQ(entry.shape[0], outputs); REQUIRE_EQ(entry.shape[1], depth); REQUIRE_EQ(entry.n_planes, 2u);
                const auto& enc = model.encodings.at(entry.enc);
                int selected[]{-1, -1};
                for (int plane = 0; plane < 2; ++plane) {
                    if (!std::strcmp(enc.plane[plane].role, "codes")) selected[0] = plane;
                    if (!std::strcmp(enc.plane[plane].role, "scale")) selected[1] = plane;
                }
                REQUIRE(selected[0] >= 0 && selected[1] >= 0);
                std::vector<uint8_t> codes(size_t(outputs) * depth);
                std::vector<float> scales(size_t(outputs) * groups);
                RadTensor planes[]{joined_head_test::matrix(codes.data(), RAD_F8E4M3, outputs, depth),
                    joined_head_test::matrix(scales.data(), RAD_F32, outputs, groups)};
                for (int idx = 0; idx < 2; ++idx) {
                    const auto& loc = model.planes.at(entry.plane_first + selected[idx]);
                    REQUIRE_EQ(loc.bytes, idx ? scales.size() * sizeof(float) : codes.size());
                    model.read(loc.offset, planes[idx].data, loc.bytes);
                }
                REQUIRE_EQ(relayout(params, 9, is_down ? 2 : 3, &enc, selected, planes, 2, dst, int64_t(outputs) * pitch), RAD_OK);
            } else {
                for (int column = 0; column < outputs; ++column) {
                    auto* line = dst + size_t(column) * pitch;
                    for (int pos = 0; pos < depth; ++pos) {
                        const uint32_t hash = uint32_t(column * 3571 + pos) * 2654435761u;
                        line[pos] = uint8_t(hash % 112) | uint8_t((hash >> 17) & 128);
                    }
                    for (int part = 0; part < groups; ++part) {
                        const float scale = float(32 + (column * 13 + part) % 17) * 0x1p-15f;
                        std::memcpy(line + depth + part * 4, &scale, 4);
                    }
                }
            }
            packed.push(queue.stream);
        }
        for (bool is_gain : {true, false}) {
            const int count = is_gain ? wide : wide * streams;
            std::vector<uint16_t> values(count);
            if (actual) {
                const auto& entry = entry_for(is_gain ? "norm" : "inject");
                REQUIRE_EQ(entry.n_planes, 1u);
                const auto& loc = model.planes.at(entry.plane_first);
                REQUIRE_EQ(loc.bytes, values.size() * sizeof(uint16_t));
                model.read(loc.offset, values.data(), loc.bytes);
            } else for (int pos = 0; pos < count; ++pos)
                values[pos] = rad_f32_to_bf16(is_gain ? float(24 + pos % 17) / 32 : float(pos % 73 - 36) / 8192);
            if (is_gain) std::transform(values.begin(), values.end(), gains.data<float>(), [](uint16_t bits) { return rad_bf16_to_f32(bits); });
            else std::memcpy(inject_weights.host.data, values.data(), values.size() * sizeof(uint16_t));
            (is_gain ? gains : inject_weights).push(queue.stream);
        }

        auto check = [&](int rows, int world, int mask, bool trace, const char* flag) {
            Buffer hidden(RAD_BF16, {rows, wide}, RAD_MEM_DEVICE), output(RAD_BF16, {rows, width}, RAD_MEM_DEVICE);
            Buffer inject(RAD_BF16, {rows, streams}, RAD_MEM_DEVICE), codes(RAD_I8, {rows, width}, RAD_MEM_DEVICE);
            Buffer rotated(actual ? RAD_I8 : RAD_F8E4M3, {rows, width}, RAD_MEM_DEVICE);
            Buffer scales(RAD_F32, {rows, width / 128}, RAD_MEM_DEVICE), rotated_scales(RAD_F32, {rows, width / 128}, RAD_MEM_DEVICE);
            Buffer intermediate(RAD_BF16, {rows, rank}, RAD_MEM_DEVICE), logits(RAD_F32, {rows, wide}, RAD_MEM_DEVICE), gates(RAD_BF16, {rows, wide}, RAD_MEM_DEVICE);
            for (int row = 0; row < rows; ++row) for (int pos = 0; pos < wide; ++pos)
                hidden.data<uint16_t>()[row * wide + pos] = rad_f32_to_bf16(float((pos * (47 + row * 2) + row * 71) % 257 - 128) / 64);
            hidden.push(queue.stream);
            RadTensor tensors[]{hidden.device, gains.device, joined_head_test::matrix(down.device(), RAD_F8E4M3, rank, wide),
                joined_head_test::matrix(up.device(), RAD_F8E4M3, wide, rank), inject_weights.device, output.device,
                inject.device, codes.device, scales.device, rotated.device, rotated_scales.device};
            if (!(mask & 1)) tensors[4] = tensors[6] = {};
            if (!(mask & 2)) tensors[7] = tensors[8] = {};
            if (!(mask & 4)) tensors[9] = tensors[10] = {};
            params[6].ival = bool(mask & 1); params[8].ival = bool(mask & 4);
            RadArgs args{}; args.t = tensors; args.n_t = 11; args.p = params; args.n_p = 9; args.world_size = world;
            const size_t norm_band = size_t(band) * wide * 2, down_band = size_t(band) * rank * 2;
            const size_t bytes = norm_band * 2 + down_band;
            REQUIRE_EQ(scratch_size(&args), int64_t(bytes));
            Storage scratch(bytes + 256, RAD_MEM_DEVICE);
            const size_t offset = 64 + ((256 - (uintptr_t(scratch.device()) & 255)) & 255);
            args.scratch = static_cast<uint8_t*>(scratch.allocation) + offset; args.scratch_bytes = bytes;
            R3dhHcTrace state{static_cast<uint16_t*>(intermediate.device.data), static_cast<float*>(logits.device.data),
                static_cast<uint16_t*>(gates.device.data), rows, width, streams, rank};
            const auto untouched = [](const auto& seen, const auto& initial, size_t first, size_t last) {
                REQUIRE(std::equal(seen.begin() + first, seen.begin() + last, initial.begin() + first));
            };
            struct Target { const char* name; Buffer* buffer; bool enabled; };
            const Target outputs[]{{"output", &output, true}, {"inject", &inject, bool(mask & 1)},
                {"codes", &codes, bool(mask & 2)}, {"scales", &scales, bool(mask & 2)},
                {"rotated", &rotated, bool(mask & 4)}, {"rotated_scales", &rotated_scales, bool(mask & 4)},
                {"down_trace", &intermediate, trace}, {"up_logits", &logits, trace}, {"up_gates", &gates, trace}};
            std::vector<std::vector<uint8_t>> baseline;
            for (int variant : {0, 1}) {
                four.set(variant ? flag : "0");
                for (const auto& target : outputs) target.buffer->push(queue.stream);
                scratch.push(queue.stream);
                REQUIRE_EQ(trace ? traced(&args, queue.stream, &state) : plain(&args, queue.stream), RAD_OK);
                auto work = scratch.read(queue.stream);
                const size_t active_wide = size_t(rows) * wide * 2, active_rank = size_t(rows) * rank * 2;
                untouched(work, scratch.expected, 0, offset);
                untouched(work, scratch.expected, offset + active_wide, offset + norm_band);
                untouched(work, scratch.expected, offset + norm_band + active_rank, offset + norm_band + down_band);
                untouched(work, scratch.expected, offset + norm_band + down_band + active_wide, work.size());
                size_t index = 0;
                auto compare = [&](const char* name, std::vector<uint8_t> seen) {
                    if (!variant) baseline.push_back(std::move(seen));
                    else if (seen != baseline.at(index))
                        throw std::runtime_error(std::string("HC four mismatch: ") + name + " rows=" + std::to_string(rows) +
                            " world=" + std::to_string(world) + " mask=" + std::to_string(mask) + (actual ? " actual" : " synthetic") +
                            (trace ? " traced" : " plain"));
                    ++index;
                };
                for (const auto& target : outputs) {
                    auto* buffer = target.buffer;
                    auto seen = buffer->read(queue.stream);
                    const int item = buffer->host.dtype == RAD_F32 ? 4 : buffer->host.dtype == RAD_BF16 ? 2 : 1;
                    const size_t payload = size_t(rows) * buffer->host.shape[1] * item;
                    untouched(seen, buffer->expected, 0, 64);
                    untouched(seen, buffer->expected, 64 + (target.enabled ? payload : 0), seen.size());
                    if (trace && (buffer == &intermediate || buffer == &gates))
                        REQUIRE(std::equal(seen.begin() + 64, seen.begin() + 64 + payload,
                            work.begin() + offset + norm_band + (buffer == &gates ? down_band : 0)));
                    if (trace && buffer == &logits) for (int row = 1; row < rows; ++row)
                        REQUIRE(std::memcmp(seen.data() + 64, seen.data() + 64 + size_t(row) * wide * 4, wide * 4) != 0);
                    compare(target.name, std::move(seen));
                }
                REQUIRE(hidden.read(queue.stream) == hidden.expected);
                compare("scratch", std::move(work));
            }
        };
        for (bool trace : {false, true}) {
            check(4, 2, 7, trace, "1");
            check(4, 2, 0, trace, "1");
        }
        if (!actual) {
            check(4, 2, 2, true, "1"); check(4, 2, 5, true, "1");
            for (int rows : {1, 3, 5}) check(rows, 2, 7, true, "1");
            check(4, 1, 7, true, "1"); check(4, 2, 7, true, nullptr); check(4, 2, 7, true, "01");
            precision.set("exact"); check(4, 2, 7, true, "1"); precision.set("tolerance");
            split.set("0"); check(4, 2, 7, true, "1"); split.set("1");
            upmix.set("0"); check(4, 2, 7, true, "1"); upmix.set("1");
        }
        REQUIRE(down.read(queue.stream) == down.expected); REQUIRE(up.read(queue.stream) == up.expected);
        REQUIRE(gains.read(queue.stream) == gains.expected); REQUIRE(inject_weights.read(queue.stream) == inject_weights.expected);
    }
}
