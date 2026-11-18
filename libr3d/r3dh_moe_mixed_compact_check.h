struct CompactWeight {
    std::vector<uint8_t> codes;
    std::vector<int8_t> decoded, table;
    std::vector<float> row, column;
    std::array<uint32_t, 2> coefficients{};
    RadEncoding encoding{};
};
CompactWeight compact_weight(Model& model, int layer, int expert) {
    const std::string wanted = "blk." + std::to_string(layer) + ".ffn_gate_up_exps."
        + std::to_string(expert) + ".weight";
    CompactWeight result;
    for (const auto& entry : model.entries) {
        if (model.name(entry) != wanted) continue;
        require(entry.enc < model.encodings.size() && entry.rank == 2
            && entry.shape[0] == 1280 && entry.shape[1] == 2560 && entry.n_planes == 5,
            "actual regular schema");
        result.encoding = model.encodings[entry.enc];
        require(rad_enc_valid(&result.encoding) && result.encoding.n_planes == 5
            && !std::strcmp(result.encoding.scheme, "trellis")
            && !std::strcmp(result.encoding.transform, "fwht128"), "actual regular encoding");
        result.codes.resize(1280 * 1280); result.table.resize(65536);
        result.row.resize(1280); result.column.resize(20);
        for (unsigned index = 0; index < entry.n_planes; ++index) {
            require(entry.plane_first + index < model.planes.size(), "regular plane bounds");
            const auto& plane = model.planes[entry.plane_first + index];
            const char* role = result.encoding.plane[index].role;
            void* destination = nullptr; uint64_t bytes = 0;
            if (!std::strcmp(role, "codes")) { destination = result.codes.data(); bytes = result.codes.size(); }
            if (!std::strcmp(role, "trellis")) { destination = result.table.data(); bytes = result.table.size(); }
            if (!std::strcmp(role, "trellis.mad")) { destination = result.coefficients.data(); bytes = 8; }
            if (!std::strcmp(role, "scale")) { destination = result.row.data(); bytes = result.row.size() * 4; }
            if (!std::strcmp(role, "scale.col")) { destination = result.column.data(); bytes = result.column.size() * 4; }
            require(destination && bytes == plane.bytes && plane.offset >= entry.offset
                && plane.offset - entry.offset <= entry.bytes
                && bytes <= entry.bytes - (plane.offset - entry.offset), "regular plane extent");
            model.read(plane.offset, destination, bytes);
        }
        for (uint32_t state = 0; state < 65536; ++state) {
            const uint32_t word = uint32_t(uint64_t(state) * (result.coefficients[0] & 0xffffffu)
                + result.coefficients[1]);
            int value = -126;
            for (unsigned byte = 0; byte < 4; ++byte) value += (word >> (byte * 8)) & 63u;
            require(result.table[state] == value, "actual regular independent MAD");
        }
        result.decoded.resize(1280 * 2560);
        for (int column = 0; column < 2560; ++column) {
            auto nibble = [&](int row) { return (result.codes[size_t(row) * 1280 + column / 2]
                >> ((column % 2) * 4)) & 15u; };
            for (int first = 0; first < 1280; first += 128) {
                uint32_t state = 0;
                for (int back = 0; back < 4; ++back) state |= nibble(first + 127 - back) << (back * 4);
                for (int offset = 0; offset < 128; ++offset) {
                    state = ((state << 4) | nibble(first + offset)) & 65535u;
                    result.decoded[size_t(first + offset) * 2560 + column] = result.table[state];
                }
            }
        }
        return result;
    }
    throw std::runtime_error("missing actual regular weight");
}
RadTensor typed_matrix(void* data, uint32_t dtype, int rows, int columns) {
    auto result = matrix(data, rows, columns); result.dtype = dtype; return result;
}
void compact_run(Model& model, int layer, int rank, int repeat, bool gpu, int tokens) {
    const std::vector<int> protected_ids = layer == 34 ? std::vector<int>{407, 496}
        : layer == 44 ? std::vector<int>{292, 350} : std::vector<int>{122, 143, 399, 445};
    const std::vector<int> original = layer == 34 ? std::vector<int>{408, 409, 510, 511}
        : layer == 44 ? std::vector<int>{348, 349, 510, 511} : std::vector<int>{508, 509, 510, 511};
    const int experts = 512 - int(protected_ids.size()), rows = tokens * 10, output_width = 768;
    std::array<int, 2> widths{rank ? 512 : 768, rank ? 768 : 512};
    std::array<std::vector<std::vector<uint8_t>>, 2> blobs;
    std::array<std::vector<std::vector<float>>, 2> scales;
    std::array<std::vector<const void*>, 4> pointers;
    std::vector<int> slots;
    std::vector<CompactWeight> source;
    std::vector<RadParam> params{RAD_INT("M", 64), RAD_INT("N", widths[0]),
        RAD_INT("N_odd", widths[1]), RAD_INT("K", 2560), RAD_INT("K_odd", 2560),
        RAD_INT("n_expert", experts), RAD_INT("top_k", 10), RAD_INT("group", 128),
        RAD_INT("a_group", 128), RAD_INT("parts", 2), RAD_STR("dtype", "i8a8h"), RAD_STR("a_order", "token")};
    for (int expert : original) {
        const int slot = expert - int(std::count_if(protected_ids.begin(), protected_ids.end(),
            [&](int protected_id) { return protected_id < expert; }));
        slots.push_back(slot); source.push_back(compact_weight(model, layer, expert));
        auto& full = source.back(); const int parity = slot % 2, width = widths[parity];
        const int low = (rank ^ parity) ? 384 : 0, half = width / 2;
        std::vector<uint8_t> codes(size_t(width) * 1280);
        std::vector<float> row(width);
        for (int index = 0; index < width; ++index) {
            const int from = (index / half) * 640 + low + index % half;
            std::copy_n(full.codes.data() + size_t(from) * 1280, 1280, codes.data() + size_t(index) * 1280);
            row[index] = full.row[from];
        }
        const int selected_weight[]{0, 2, 3}, selected_scale[]{1, 4};
        std::array<RadTensor, 3> wp{}; std::array<RadTensor, 2> sp{};
        for (int index = 0; index < 5; ++index) {
            const char* role = full.encoding.plane[index].role;
            RadTensor plane{};
            if (!std::strcmp(role, "codes")) plane = typed_matrix(codes.data(), RAD_U4, width, 2560);
            else if (!std::strcmp(role, "trellis")) plane = typed_matrix(full.table.data(), RAD_I8, 65536, 1);
            else if (!std::strcmp(role, "trellis.mad")) plane = typed_matrix(full.coefficients.data(), RAD_U32, 2, 1);
            else if (!std::strcmp(role, "scale")) plane = typed_matrix(row.data(), RAD_F32, width, 1);
            else plane = typed_matrix(full.column.data(), RAD_F32, 1, 20);
            for (int target = 0; target < 3; ++target) if (selected_weight[target] == index) wp[target] = plane;
            for (int target = 0; target < 2; ++target) if (selected_scale[target] == index) sp[target] = plane;
        }
        RadLayout wl{}, sl{}; const int operand = 2 + parity * 4;
        require(r3dh_moe_mixed_iqr_layout(params.data(), params.size(), operand,
            &full.encoding, selected_weight, wp.data(), 3, &wl) == RAD_OK, "compact weight layout");
        require(r3dh_moe_mixed_iqr_layout(params.data(), params.size(), operand + 1,
            &full.encoding, selected_scale, sp.data(), 2, &sl) == RAD_OK, "compact scale layout");
        blobs[parity].emplace_back(wl.bytes); scales[parity].emplace_back(sl.bytes / 4);
        require(r3dh_moe_mixed_iqr_relayout(params.data(), params.size(), operand,
            &full.encoding, selected_weight, wp.data(), 3, blobs[parity].back().data(), wl.bytes) == RAD_OK,
            "compact weight relayout");
        require(r3dh_moe_mixed_iqr_relayout(params.data(), params.size(), operand + 1,
            &full.encoding, selected_scale, sp.data(), 2, scales[parity].back().data(), sl.bytes) == RAD_OK,
            "compact scale relayout");
    }
    require(slots.back() == experts - 1 && slots[2] == experts - 2, "last compact table entries");
    if (layer == 34) require(slots[1] == 408 && original[1] == 409, "expert409 compact408 parity");
    std::vector<int8_t> activation(tokens * 2560);
    std::vector<float> activation_scale(tokens * 20);
    for (size_t index = 0; index < activation.size(); ++index) activation[index] = int8_t(int((index * 37 + repeat * 29) % 255) - 128);
    for (size_t index = 0; index < activation_scale.size(); ++index) activation_scale[index] = float(index % 7 + 1) / 8192.0f;
    std::vector<int32_t> sorted(rows, -1), offsets(experts + 1);
    for (int row = 0; row < 8; ++row) sorted[row] = (row * 7 + repeat * 3) % rows;
    sorted[1] = -1;
    for (int expert = 0; expert <= experts; ++expert)
        offsets[expert] = int(std::count_if(slots.begin(), slots.end(), [&](int slot) { return slot < expert; })) * 2;
    std::vector<uint16_t> expected(rows * output_width), output(expected.size(), 0x42c6);
    uint64_t nonzero = 0, padded = 0;
    for (int row = 0; row < 8; ++row) for (int column = 0; column < output_width; ++column) {
        const int selected = row / 2, parity = slots[selected] % 2, width = widths[parity];
        const int local = column % 384, low = (rank ^ parity) ? 384 : 0;
        if (sorted[row] < 0 || local >= width / 2) { ++padded; continue; }
        const int from = (column / 384) * 640 + low + local, token = sorted[row] / 10;
        const auto& full = source[selected]; float sum = 0;
        for (int group = 0; group < 20; ++group) {
            int32_t dot = 0;
            for (int inner = group * 128; inner < (group + 1) * 128; ++inner)
                dot += int32_t(activation[token * 2560 + inner]) * int32_t(full.decoded[size_t(from) * 2560 + inner]);
            const float scaled = float(dot) * activation_scale[token * 20 + group];
            sum = sum + scaled * full.column[group];
        }
        expected[row * output_width + column] = pack(sum * full.row[from]);
        nonzero += expected[row * output_width + column] != 0;
    }
    require(nonzero && padded, "compact contribution and padding controls");
    if (gpu) {
        hipStream_t stream{}; hip_ok(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
        std::vector<std::unique_ptr<Device>> buffers;
        for (int parity = 0; parity < 2; ++parity) {
            for (size_t index = 0; index < blobs[parity].size(); ++index) {
                buffers.push_back(std::make_unique<Device>(blobs[parity][index].size()));
                buffers.back()->upload(blobs[parity][index].data(), stream);
                pointers[parity * 2].push_back(buffers.back()->data());
                buffers.push_back(std::make_unique<Device>(scales[parity][index].size() * 4));
                buffers.back()->upload(scales[parity][index].data(), stream);
                pointers[parity * 2 + 1].push_back(buffers.back()->data());
            }
        }
        std::array<std::vector<const void*>, 4> tables;
        for (int parity = 0; parity < 2; ++parity) for (int kind = 0; kind < 2; ++kind) {
            auto& table = tables[parity * 2 + kind]; table.assign(experts / 2, pointers[parity * 2 + kind][0]);
            int fixture = 0;
            for (int slot : slots) if (slot % 2 == parity) table[slot / 2] = pointers[parity * 2 + kind][fixture++];
        }
        Device input(activation.size()), asc(activation_scale.size() * 4), order(sorted.size() * 4), off(offsets.size() * 4), result(output.size() * 2);
        input.upload(activation.data(), stream); asc.upload(activation_scale.data(), stream);
        order.upload(sorted.data(), stream); off.upload(offsets.data(), stream); result.upload(output.data(), stream);
        std::array<RadTensor, 9> tensors{typed_matrix(input.data(), RAD_I8, tokens, 2560),
            typed_matrix(asc.data(), RAD_F32, tokens, 20), {}, {}, vector(sorted), vector(offsets), {}, {}, matrix(result.data(), rows, output_width)};
        tensors[4].data = order.data(); tensors[5].data = off.data();
        std::vector<std::unique_ptr<Device>> device_tables;
        for (int parity = 0; parity < 2; ++parity) for (int kind = 0; kind < 2; ++kind) {
            auto& table = tables[parity * 2 + kind];
            device_tables.push_back(std::make_unique<Device>(table.size() * sizeof(void*)));
            device_tables.back()->upload(table.data(), stream);
            auto& tensor = tensors[2 + parity * 4 + kind];
            tensor = typed_matrix(device_tables.back()->data(), kind ? r3dh_moe_iqr::scale_dtype : r3dh_moe_iqr::weight_dtype, widths[parity], kind ? 20 : 2560);
            tensor.rank = 3; tensor.shape[0] = experts / 2; tensor.shape[1] = widths[parity]; tensor.shape[2] = kind ? 20 : 2560;
            tensor.stride[0] = 0; tensor.stride[1] = kind ? 20 : 2560; tensor.stride[2] = 1;
            if (kind && repeat) {
                tensor.shape[2] = 2560;
                tensor.stride[1] = 2560;
            }
        }
        RadArgs args{}; args.t = tensors.data(); args.n_t = tensors.size(); args.p = params.data(); args.n_p = params.size(); args.rank = rank; args.world_size = 2;
        require(r3dh_moe_mixed_iqr_launch(&args, reinterpret_cast<RadStream>(stream)) == RAD_OK, "native compact launch");
        result.read(output.data(), stream);
        for (size_t index = 0; index < output.size(); ++index) {
            if (output[index] != expected[index]) std::fprintf(stderr, "compact mismatch layer=%d rank=%d repeat=%d index=%zu got=%04x want=%04x\n", layer, rank, repeat, index, output[index], expected[index]);
            require(output[index] == expected[index], "native compact independent oracle/padding");
        }
        comparisons += output.size();
        result.upload(std::vector<uint16_t>(output.size(), 0x42c6).data(), stream);
        require(r3dh_moe_iqr_launch(&args, reinterpret_cast<RadStream>(stream)) == RAD_OK, "base native compact launch");
        result.read(output.data(), stream);
        require(output == expected, "base native compact independent oracle/padding");
        comparisons += output.size();
        hip_ok(hipStreamDestroy(stream));
    }
    ++cases; negatives += 2;
    std::printf("MIXED_COMPACT_CASE layer=%d rank=%d repeat=%d regular=%d protected=%zu last_table=%d original409_compact408=%d M=%d band=64 logical=%d nonzero=%llu pad=%llu gpu=%d\n",
        layer, rank, repeat, experts, protected_ids.size(), experts / 2 - 1, layer == 34, tokens, repeat,
        (unsigned long long)nonzero, (unsigned long long)padded, int(gpu));
}
