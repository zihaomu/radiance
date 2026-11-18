struct Model {
    static constexpr uint64_t pinned_bytes = 121678372864ull;
    std::ifstream input;
    RadFileHeader header{};
    explicit Model(const char* path) : input(path, std::ios::binary) {
        expect(input.is_open(), "open actual model");
        input.seekg(0, std::ios::end);
        expect(static_cast<uint64_t>(input.tellg()) == pinned_bytes, "pinned model size");
        read(0, &header, sizeof(header));
        expect(header.magic == RAD_MAGIC && header.version == RAD_FORMAT_VER
            && header.file_bytes == pinned_bytes, "pinned RAD header");
        expect(header.dir_count < 1000000 && header.enc_count < 100000
            && header.str_bytes < 100000000, "bounded RAD tables");
        std::printf("model_identity=pinned_size_prior_verified_sha256 bytes=%llu sha256="
            "25aff8c950f4b823bc8b77c8184ed48827cb47058ffce08ae2c83b12e021fea0\n",
            static_cast<unsigned long long>(pinned_bytes));
    }
    void read(uint64_t offset, void* output, uint64_t bytes) {
        expect(offset <= pinned_bytes && bytes <= pinned_bytes - offset, "bounded model read");
        input.clear();
        input.seekg(offset);
        input.read(static_cast<char*>(output), bytes);
        expect(static_cast<uint64_t>(input.gcount()) == bytes, "complete model read");
    }
    template<class Value> Value item(uint64_t base, uint64_t index) {
        expect(base <= pinned_bytes && index <= (pinned_bytes - base) / sizeof(Value), "bounded index");
        Value result{};
        read(base + index * sizeof(Value), &result, sizeof(result));
        return result;
    }
    std::string name(rad_stroff offset) {
        expect(offset < header.str_bytes, "bounded model name");
        std::array<char, 1024> text{};
        const uint64_t count = std::min<uint64_t>(text.size(), header.str_bytes - offset);
        read(header.str_off + offset, text.data(), count);
        expect(std::memchr(text.data(), 0, count) != nullptr, "terminated model name");
        return text.data();
    }
};

std::array<Fixture, 4> load4experts(const char* path) {
    Model model(path);
    std::array<Fixture, 4> fixtures{Fixture(1280), Fixture(1280), Fixture(1280), Fixture(1280)};
    std::array<bool, 4> seen{};
    unsigned entries = 0;
    for (uint64_t index = 0; index < model.header.dir_count; ++index) {
        const auto entry = model.item<RadFileEntry>(model.header.dir_off, index);
        if (entry.layer != 0 || (entry.expert != 0 && entry.expert != 1
            && entry.expert != 510 && entry.expert != 511)) continue;
        const std::string name = model.name(entry.name);
        if (name != "blk.0.ffn_gate_up_exps." + std::to_string(entry.expert) + ".weight") continue;
        expect(entry.rank == 2 && entry.shape[0] == 1280 && entry.shape[1] == 2560,
               "actual fused gate/up geometry");
        const unsigned fixture_index = (entry.expert & 1) + (entry.expert >= 510 ? 2 : 0);
        expect(!seen[fixture_index], "unique actual expert");
        seen[fixture_index] = true;
        Fixture& fixture = fixtures[fixture_index];
        expect(entry.enc < model.header.enc_count, "bounded encoding index");
        const auto encoding = model.item<RadEncoding>(model.header.enc_off, entry.enc);
        expect(entry.n_planes == static_cast<unsigned>(encoding.n_planes)
            && entry.plane_first <= model.header.plane_count
            && entry.n_planes <= model.header.plane_count - entry.plane_first, "bounded entry planes");
        expect(encoding.n_planes == 5 && rad_enc_valid(&encoding), "actual encoding schema");
        fixture.encoding = encoding;
        std::printf("manifest expert=%d slot=%d source=%s rows=%lld cols=%lld scheme=%s transform=%s\n",
            entry.expert, entry.slot, name.c_str(), static_cast<long long>(entry.shape[0]),
            static_cast<long long>(entry.shape[1]), encoding.scheme, encoding.transform);
        for (unsigned plane = 0; plane < entry.n_planes; ++plane) {
            const auto location = model.item<RadFilePlane>(model.header.plane_off, entry.plane_first + plane);
            const char* role = encoding.plane[plane].role;
            void* destination = nullptr;
            uint64_t bytes = 0;
            if (!std::strcmp(role, "codes")) { destination = fixture.codes.data(); bytes = fixture.codes.size(); }
            else if (!std::strcmp(role, "trellis")) { destination = fixture.table.data(); bytes = fixture.table.size(); }
            else if (!std::strcmp(role, "trellis.mad")) { destination = fixture.coefficients.data(); bytes = sizeof(fixture.coefficients); }
            else if (!std::strcmp(role, "scale")) { destination = fixture.row.data(); bytes = fixture.row.size() * sizeof(float); }
            else if (!std::strcmp(role, "scale.col")) { destination = fixture.column.data(); bytes = fixture.column.size() * sizeof(float); }
            expect(destination && bytes == location.bytes, "actual plane role and exact bytes");
            expect(location.offset >= entry.offset && location.offset - entry.offset <= entry.bytes
                && bytes <= entry.bytes - (location.offset - entry.offset), "plane inside weight entry");
            model.read(location.offset, destination, bytes);
            std::printf("plane role=%s dtype=%u block=%lld,%lld offset=%llu bytes=%llu\n",
                encoding.plane[plane].role, encoding.plane[plane].dtype,
                static_cast<long long>(encoding.plane[plane].block[0]),
                static_cast<long long>(encoding.plane[plane].block[1]),
                static_cast<unsigned long long>(location.offset), static_cast<unsigned long long>(location.bytes));
        }
        ++entries;
    }
    expect(entries == 4, "four actual fused gate/up experts");
    for (const auto& fixture : fixtures) {
        for (unsigned index = 0; index < 65536; ++index)
            expect(fixture.table[index] == reference_mad(index, fixture.coefficients[0], fixture.coefficients[1]),
                   "actual full table versus independent mad");
        const auto decoded = fixture.decode();
        for (int row = 0; row < 1280; ++row)
            for (int column = 0; column < 2560; ++column)
                expect(decoded[static_cast<size_t>(row) * 2560 + column]
                    == mad(state(fixture.codes.data(), 2560, row, column), fixture.coefficients[0], fixture.coefficients[1]),
                    "actual full sequential table versus computed decoder");
    }
    std::printf("PASS actual_full_decode experts=4 elements=13107200 table_entries=262144\n");
    return fixtures;
}

void actual_tests(const char* path, bool gpu) {
    auto full = load4experts(path);
    if (gpu) gpu_identity();
    for (int rank = 0; rank < 2; ++rank) {
        const int even = rank == 0 ? 768 : 512;
        const int odd = rank == 0 ? 512 : 768;
        std::array<Fixture, 4> shards{Fixture(even), Fixture(odd), Fixture(even), Fixture(odd)};
        auto params = parameters(3, even, odd);
        for (unsigned index = 0; index < 4; ++index) {
            auto& shard = shards[index];
            const auto& original = full[index];
            const int low = (rank ^ (index & 1u)) ? 384 : 0;
            const int part = shard.rows / 2;
            shard.encoding = original.encoding;
            shard.table = original.table;
            shard.coefficients = original.coefficients;
            shard.column = original.column;
            for (int row = 0; row < shard.rows; ++row) {
                const int source_row = (row / part) * 640 + low + row % part;
                std::memcpy(shard.codes.data() + static_cast<size_t>(row) * 1280,
                    original.codes.data() + static_cast<size_t>(source_row) * 1280, 1280);
                shard.row[row] = original.row[source_row];
            }
            const auto source = original.decode();
            const auto decoded = shard.decode();
            for (int row = 0; row < shard.rows; ++row) {
                const int source_row = (row / part) * 640 + low + row % part;
                for (int column = 0; column < 2560; ++column)
                    expect(decoded[static_cast<size_t>(row) * 2560 + column]
                        == source[static_cast<size_t>(source_row) * 2560 + column], "actual TP2 shard decoder");
            }
            shard.encode(params, (index & 1u) ? 6 : 2);
            shard.encode(params, (index & 1u) ? 7 : 3);
            if (gpu) gpu_decode_case(shard, params, (index & 1u) ? 6 : 2);
            std::printf("shard rank=%d expert=%u gate=[%d,%d) up=[%d,%d) N=%d\n",
                rank, index < 2 ? index : index + 508, low, low + part, 640 + low, 640 + low + part, shard.rows);
        }
        if (gpu) {
            const char* label = rank ? "actual_rank1" : "actual_rank0";
            gpu_case(label, shards, 1, 0, false);
            gpu_case(label, shards, 3, 1, false);
            gpu_case(label, shards, 8, 4, false);
            gpu_case(label, shards, 17, 0, false);
            gpu_case(label, shards, 3, 3, false);
            gpu_case(label, shards, 3, 1, true);
            for (int tokens : {1, 3, 63, 64})
                for (bool logical : {false, true})
                    gpu_case(label, shards, tokens, 4, false, 64, logical);
        }
        std::printf("PASS actual_TP2 rank=%d N=%d N_odd=%d shard_elements=6553600\n", rank, even, odd);
    }
}
