float independent_e4m3(uint8_t bits) {
    const int exponent = (bits >> 3) & 15, fraction = bits & 7;
    require(exponent != 15 || fraction != 7, "finite model E4M3");
    const float magnitude = exponent ? std::ldexp(float(8 + fraction), exponent - 10)
        : std::ldexp(float(fraction), -9);
    return bits & 128 ? -magnitude : magnitude;
}
void compact_down_run(Model& model, int layer, int rank, int repeat, bool gpu, int tokens) {
    using namespace r3dh_moe_w4nl;
    const int experts = layer == 47 ? 508 : 510, rows = tokens * 10, columns = 2560;
    const std::array<int, 2> inner{rank ? 256 : 384, rank ? 384 : 256};
    std::array<std::vector<uint8_t>, 2> codes, scale;
    std::array<std::vector<float>, 2> decoded;
    std::array<std::vector<float>, 2> decoded_scale;
    std::vector<RadParam> params{RAD_INT("M", 64), RAD_INT("N", columns), RAD_INT("N_odd", columns),
        RAD_INT("K", inner[0]), RAD_INT("K_odd", inner[1]), RAD_INT("n_expert", experts),
        RAD_INT("top_k", 10), RAD_INT("group", 128), RAD_INT("a_group", 128),
        RAD_INT("parts", 1), RAD_STR("dtype", "w4nl64a8i8h"), RAD_STR("a_order", "sorted")};
    for (int parity = 0; parity < 2; ++parity) {
        const int expert = 510 + parity;
        const std::string wanted = "blk." + std::to_string(layer) + ".ffn_down_exps." + std::to_string(expert) + ".weight";
        bool seen = false;
        for (const auto& entry : model.entries) {
            if (model.name(entry) != wanted) continue;
            require(entry.enc < model.encodings.size() && entry.rank == 2 && entry.shape[0] == 2560
                && entry.shape[1] == 640 && entry.n_planes == 4, "compact down actual schema");
            const auto& encoding = model.encodings[entry.enc];
            std::vector<uint8_t> raw_codes(columns * 320), raw_scale(columns * 10);
            std::array<float, 16> table{}; float level = 0;
            for (int index = 0; index < 4; ++index) {
                require(entry.plane_first + index < model.planes.size(), "down plane index");
                const auto& plane = model.planes[entry.plane_first + index];
                const char* role = encoding.plane[index].role; void* destination = nullptr; uint64_t bytes = 0;
                if (!std::strcmp(role, "codes")) { destination = raw_codes.data(); bytes = raw_codes.size(); }
                if (!std::strcmp(role, "table")) { destination = table.data(); bytes = sizeof(table); }
                if (!std::strcmp(role, "scale")) { destination = raw_scale.data(); bytes = raw_scale.size(); }
                if (!std::strcmp(role, "scale.1")) { destination = &level; bytes = sizeof(level); }
                require(destination && bytes == plane.bytes && plane.offset >= entry.offset
                    && plane.offset - entry.offset <= entry.bytes
                    && bytes <= entry.bytes - (plane.offset - entry.offset), "down plane extent");
                model.read(plane.offset, destination, bytes);
            }
            require(level == 0x1p-13f, "actual down level");
            const int width = inner[parity], low = (rank ^ parity) ? 384 : 0;
            codes[parity].resize(sizeof(CodesHeader) + columns * width / 2);
            scale[parity].resize(sizeof(ScalesHeader) + columns * width / 64);
            CodesHeader ch{codes_magic, 1, uint32_t(columns), uint32_t(width), {}};
            std::copy(table.begin(), table.end(), ch.table);
            ScalesHeader sh{scales_magic, 1, uint32_t(columns), uint32_t(width), level, {}};
            std::memcpy(codes[parity].data(), &ch, sizeof(ch)); std::memcpy(scale[parity].data(), &sh, sizeof(sh));
            decoded[parity].resize(columns * width); decoded_scale[parity].resize(columns * width / 64);
            for (int column = 0; column < columns; ++column) {
                std::copy_n(raw_codes.data() + column * 320 + low / 2, width / 2,
                    codes[parity].data() + sizeof(ch) + column * width / 2);
                std::copy_n(raw_scale.data() + column * 10 + low / 64, width / 64,
                    scale[parity].data() + sizeof(sh) + column * width / 64);
                for (int position = 0; position < width; ++position) {
                    const int source = low + position;
                    const unsigned nibble = (raw_codes[column * 320 + source / 2] >> ((source % 2) * 4)) & 15u;
                    decoded[parity][column * width + position] = table[nibble];
                }
                for (int group = 0; group < width / 64; ++group)
                    decoded_scale[parity][column * (width / 64) + group] = independent_e4m3(raw_scale[column * 10 + low / 64 + group]) * level;
            }
            seen = true;
        }
        require(seen, "actual compact down expert");
    }
    std::vector<int8_t> activation(rows * 384); std::vector<float> activation_scale(rows * 3);
    for (size_t index = 0; index < activation.size(); ++index) activation[index] = int8_t(int((index * 23 + repeat * 31) % 255) - 128);
    for (size_t index = 0; index < activation_scale.size(); ++index) activation_scale[index] = float(index % 5 + 1) / 4096.0f;
    std::vector<int32_t> sorted(rows, -1), offsets(experts + 1, 0);
    for (int row = 0; row < 4; ++row) sorted[row] = (row * 7 + repeat * 3) % rows;
    sorted[1] = -1; offsets[experts - 1] = 2; offsets[experts] = 4;
    std::vector<uint16_t> expected(rows * columns), output(expected.size(), 0x42c6);
    for (int row = 0; row < 4; ++row) for (int column = 0; column < columns; ++column) {
        if (sorted[row] < 0) continue;
        const int parity = row / 2, width = inner[parity]; float sum = 0;
        for (int group = 0; group < width / 64; ++group) {
            float subtotal = 0;
            for (int position = group * 64; position < (group + 1) * 64; ++position)
                subtotal = subtotal + float(activation[row * 384 + position]) * decoded[parity][column * width + position];
            const float first = subtotal * activation_scale[row * 3 + group / 2];
            sum = sum + first * decoded_scale[parity][column * (width / 64) + group];
        }
        expected[row * columns + column] = pack(sum);
    }
    require(std::any_of(expected.begin(), expected.begin() + 4 * columns, [](uint16_t value) { return value != 0; }), "down nonzero contribution");
    if (gpu) {
        hipStream_t stream{}; hip_ok(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
        std::vector<std::unique_ptr<Device>> buffers;
        std::array<std::vector<const void*>, 4> tables;
        for (int parity = 0; parity < 2; ++parity) for (int kind = 0; kind < 2; ++kind) {
            auto& data = kind ? scale[parity] : codes[parity];
            buffers.push_back(std::make_unique<Device>(data.size())); buffers.back()->upload(data.data(), stream);
            tables[parity * 2 + kind].assign(experts / 2, buffers.back()->data());
        }
        Device input(activation.size()), asc(activation_scale.size() * 4), order(sorted.size() * 4), off(offsets.size() * 4), result(output.size() * 2);
        input.upload(activation.data(), stream); asc.upload(activation_scale.data(), stream);
        order.upload(sorted.data(), stream); off.upload(offsets.data(), stream); result.upload(output.data(), stream);
        std::array<RadTensor, 9> tensors{typed_matrix(input.data(), RAD_I8, rows, 384), typed_matrix(asc.data(), RAD_F32, rows, 3),
            {}, {}, vector(sorted), vector(offsets), {}, {}, matrix(result.data(), rows, columns)};
        tensors[4].data = order.data(); tensors[5].data = off.data();
        for (int parity = 0; parity < 2; ++parity) for (int kind = 0; kind < 2; ++kind) {
            auto& table = tables[parity * 2 + kind];
            buffers.push_back(std::make_unique<Device>(table.size() * sizeof(void*))); buffers.back()->upload(table.data(), stream);
            const int bytes = int(kind ? scale[parity].size() : codes[parity].size());
            auto& tensor = tensors[2 + parity * 4 + kind]; tensor = typed_matrix(buffers.back()->data(), kind ? scales_type : codes_type, experts / 2, bytes);
            tensor.stride[0] = 0;
            if (repeat) {
                tensor.rank = 3; tensor.shape[1] = columns;
                tensor.shape[2] = inner[parity];
                tensor.stride[1] = inner[parity]; tensor.stride[2] = 1;
            }
        }
        RadArgs args{}; args.t = tensors.data(); args.n_t = tensors.size(); args.p = params.data(); args.n_p = params.size(); args.rank = rank; args.world_size = 2;
        require(r3dh_moe_w4nl_launch(&args, reinterpret_cast<RadStream>(stream)) == RAD_OK, "compact native down launch");
        result.read(output.data(), stream);
        require(output == expected, "actual compact native down oracle/zero protected rows"); comparisons += output.size();
        hip_ok(hipStreamDestroy(stream));
    }
    ++cases;
    std::printf("MIXED_DOWN_CASE layer=%d rank=%d repeat=%d regular=%d last_table=%d M=%d band=64 logical=%d zero_protected_rows=1 gpu=%d\n", layer, rank, repeat, experts, experts / 2 - 1, tokens, repeat, int(gpu));
}
