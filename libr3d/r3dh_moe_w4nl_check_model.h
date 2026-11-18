uint64_t fingerprint(const void* data, size_t bytes, uint64_t hash = 14695981039346656037ull) {
    const auto* values = static_cast<const uint8_t*>(data);
    for (size_t index = 0; index < bytes; ++index) hash = (hash ^ values[index]) * 1099511628211ull;
    return hash;
}
struct Model {
    std::ifstream input;
    std::string path;
    uint64_t size = 0, metadata_hash = 0;
    RadFileHeader header{};
    std::vector<char> strings;
    std::vector<RadFileEntry> directory;
    std::vector<RadFilePlane> planes;
    std::vector<RadEncoding> encodings;
    explicit Model(const std::string& model_path) : input(model_path, std::ios::binary), path(model_path) {
        expect(bool(input), "open RAD model");
        input.seekg(0, std::ios::end);
        const auto length = input.tellg();
        expect(length > 0, "model size");
        size = uint64_t(length);
        read(0, &header, sizeof(header));
        expect(header.magic == RAD_MAGIC && header.version == RAD_FORMAT_VER && header.file_bytes == size,
               "RAD container header");
        strings = section<char>(header.str_off, header.str_bytes);
        directory = section<RadFileEntry>(header.dir_off, header.dir_count);
        planes = section<RadFilePlane>(header.plane_off, header.plane_count);
        encodings = section<RadEncoding>(header.enc_off, header.enc_count);
        metadata_hash = fingerprint(&header, sizeof(header));
        metadata_hash = fingerprint(strings.data(), strings.size(), metadata_hash);
        metadata_hash = fingerprint(directory.data(), directory.size() * sizeof(RadFileEntry), metadata_hash);
        metadata_hash = fingerprint(planes.data(), planes.size() * sizeof(RadFilePlane), metadata_hash);
        metadata_hash = fingerprint(encodings.data(), encodings.size() * sizeof(RadEncoding), metadata_hash);
    }
    void read(uint64_t offset, void* target, size_t bytes) {
        expect(offset <= size && bytes <= size - offset, "bounded model read");
        input.clear(); input.seekg(std::streamoff(offset));
        input.read(static_cast<char*>(target), std::streamsize(bytes));
        expect(bool(input), "read model plane");
    }
    template<class Element> std::vector<Element> section(uint64_t offset, uint64_t count) {
        expect(count <= 100000000 && count <= size / sizeof(Element), "bounded metadata count");
        std::vector<Element> result(static_cast<size_t>(count));
        if (count) read(offset, result.data(), result.size() * sizeof(Element));
        return result;
    }
    std::string name(const RadFileEntry& entry) const {
        expect(entry.name < strings.size(), "name offset");
        const char* text = strings.data() + entry.name;
        expect(std::memchr(text, 0, strings.size() - entry.name), "terminated name");
        return text;
    }
    bool candidate(const RadFileEntry& entry) const {
        if (entry.enc >= encodings.size() || entry.rank != 2 || entry.shape[0] != 2560
            || entry.shape[1] != 640 || entry.n_planes != 4) return false;
        const auto& encoding = encodings[entry.enc];
        if (encoding.n_planes != 4 || std::strcmp(encoding.transform, "fwht128")) return false;
        const auto* codes = rad_enc_plane(&encoding, "codes");
        const auto* table = rad_enc_plane(&encoding, "table");
        const auto* scale = rad_enc_plane(&encoding, "scale");
        const auto* level = rad_enc_plane(&encoding, "scale.1");
        return codes && table && scale && level && codes->dtype == RAD_U4
            && table->dtype == RAD_F32 && table->extent[0] == 1 && table->extent[1] == 16
            && scale->dtype == RAD_F8E4M3 && scale->block[0] == 1 && scale->block[1] == 64
            && level->dtype == RAD_F32 && level->block[0] == 0 && level->block[1] == 0;
    }
    std::vector<uint8_t> plane(size_t entry_index, const char* role) {
        expect(entry_index < directory.size(), "entry index");
        const auto& entry = directory[entry_index];
        expect(candidate(entry), "W4NL64 down entry schema");
        const auto& encoding = encodings[entry.enc];
        const auto* selected = rad_enc_plane(&encoding, role);
        expect(selected, "required plane role");
        const uint64_t plane_index = entry.plane_first + uint64_t(selected - encoding.plane);
        expect(plane_index >= entry.plane_first && plane_index < planes.size(), "plane index");
        const auto& descriptor = planes[size_t(plane_index)];
        expect(descriptor.offset >= header.data_off && descriptor.bytes <= 1024 * 1024 * 16,
               "plane data extent");
        std::vector<uint8_t> result(size_t(descriptor.bytes));
        read(descriptor.offset, result.data(), result.size());
        return result;
    }
    std::string record(size_t index) {
        expect(index < directory.size(), "manifest entry index");
        const auto& entry = directory[index];
        expect(candidate(entry), "manifest selected format");
        const auto& encoding = encodings[entry.enc];
        std::ostringstream result;
        result << index << ' ' << std::quoted(name(entry)) << ' ' << entry.layer << ' ' << entry.expert;
        for (const char* role : {"codes", "table", "scale", "scale.1"}) {
            const auto* selected = rad_enc_plane(&encoding, role);
            const auto data = plane(index, role);
            const auto& descriptor = planes[size_t(entry.plane_first + uint64_t(selected - encoding.plane))];
            result << ' ' << descriptor.offset << ' ' << descriptor.bytes << ' '
                   << fingerprint(data.data(), data.size());
        }
        return result.str();
    }
    std::vector<size_t> selection() const {
        std::vector<size_t> parity[2];
        int layer = -1;
        for (size_t index = 0; index < directory.size(); ++index) {
            const auto& entry = directory[index];
            if (!candidate(entry) || entry.expert < 0) continue;
            if (layer < 0) layer = entry.layer;
            if (entry.layer != layer) continue;
            auto& target = parity[entry.expert % 2];
            if (target.size() < 2) target.push_back(index);
        }
        expect(parity[0].size() == 2 && parity[1].size() == 2, "two model experts per parity");
        return {parity[0][0], parity[1][0], parity[0][1], parity[1][1]};
    }
    void write_manifest(const std::string& destination) {
        std::ofstream output(destination, std::ios::out | std::ios::trunc);
        expect(bool(output), "open manifest output");
        output << "R3DH_W4NL64_MANIFEST_V1\n" << std::quoted(path) << ' ' << size << ' '
               << metadata_hash << '\n' << "4\n";
        for (size_t index : selection()) output << record(index) << '\n';
        output.close();
        expect(bool(output), "write manifest");
        std::printf("W4NL_MANIFEST_WRITTEN %s file_bytes=%llu metadata_fnv64=%llu\n",
            destination.c_str(), static_cast<unsigned long long>(size),
            static_cast<unsigned long long>(metadata_hash));
    }
    std::vector<Weight> parse_manifest(std::istream& source) {
        std::string line;
        expect(bool(std::getline(source, line)) && line == "R3DH_W4NL64_MANIFEST_V1", "manifest version");
        std::ostringstream identity;
        identity << std::quoted(path) << ' ' << size << ' ' << metadata_hash;
        expect(bool(std::getline(source, line)) && line == identity.str(), "manifest model identity");
        expect(bool(std::getline(source, line)) && line == "4", "manifest count");
        const size_t count = 4;
        std::vector<Weight> weights;
        const auto selected = selection();
        for (size_t ordinal = 0; ordinal < count; ++ordinal) {
            expect(bool(std::getline(source, line)), "manifest record");
            std::istringstream fields(line);
            size_t index = size_t(-1);
            fields >> index;
            expect(bool(fields) && index == selected[ordinal] && line == record(index), "strict manifest plane fingerprints");
            Weight weight(2560, 640);
            weight.codes = plane(index, "codes");
            weight.scales = plane(index, "scale");
            const auto levels = plane(index, "table"), level = plane(index, "scale.1");
            expect(weight.codes.size() == 2560 * 320 && weight.scales.size() == 2560 * 10
                && levels.size() == 64 && level.size() == 4, "exact model plane sizes");
            std::memcpy(weight.table.data(), levels.data(), 64);
            std::memcpy(&weight.level, level.data(), 4);
            const auto& encoding = encodings[directory[index].enc];
            auto tensors = weight.planes();
            auto params = parameters(1, 2560, 640, 640);
            for (int operand : {2, 3}) {
                const int selected_planes[2] = {
                    int(rad_enc_plane(&encoding, operand == 2 ? "codes" : "scale") - encoding.plane),
                    int(rad_enc_plane(&encoding, operand == 2 ? "table" : "scale.1") - encoding.plane)};
                RadLayout layout{};
                expect(r3dh_moe_w4nl_layout(params.data(), int(params.size()), operand, &encoding,
                    selected_planes, tensors.data() + (operand == 2 ? 0 : 2), 2, &layout) == RAD_OK,
                    "actual container encoding accepted strictly");
            }
            bool grid = true;
            for (float value : weight.table)
                grid = grid && std::isfinite(value) && value >= -128 && value <= 127 && value == std::trunc(value);
            std::printf("W4NL_MODEL_EXPERT name=%s scheme=%s int8_grid=%d level=%.9g table=",
                name(directory[index]).c_str(), encoding.scheme, grid, weight.level);
            for (float value : weight.table) std::printf(" %.9g", value);
            std::printf("\n");
            weights.push_back(std::move(weight));
        }
        expect(!std::getline(source, line), "no trailing manifest records");
        return weights;
    }
    std::vector<Weight> load_manifest(const std::string& manifest) {
        std::ifstream source(manifest);
        return parse_manifest(source);
    }
    void manifest_tests(const std::string& manifest) {
        static_cast<void>(load_manifest(manifest));
        std::vector<std::string> records;
        for (size_t index : selection()) records.push_back(record(index));
        auto prefix = [&](uint64_t bytes, uint64_t hash, const char* count) {
            std::ostringstream output;
            output << "R3DH_W4NL64_MANIFEST_V1\n" << std::quoted(path) << ' ' << bytes << ' '
                   << hash << '\n' << count << '\n';
            return output.str();
        };
        std::string all;
        for (const auto& record : records) all += record + '\n';
        const std::string correct_prefix = prefix(size, metadata_hash, "4");
        const std::vector<std::string> corruptions{
            "BAD_VERSION\n", prefix(size + 1, metadata_hash, "4") + all,
            prefix(size, metadata_hash ^ 1, "4") + all,
            prefix(size, metadata_hash, "4 extra") + all,
            correct_prefix + records[0] + " changed_plane_record\n" + records[1] + '\n'
                + records[2] + '\n' + records[3] + '\n',
            correct_prefix + records[0] + '\n', correct_prefix + all + "extra\n"};
        for (const auto& text : corruptions) {
            std::istringstream input(text);
            bool rejected = false;
            try { static_cast<void>(parse_manifest(input)); }
            catch (const std::runtime_error&) { rejected = true; }
            expect(rejected, "corrupt manifest accepted");
        }
        std::printf("W4NL_MANIFEST_PASS rejected=%zu checks=%llu\n", corruptions.size(),
            static_cast<unsigned long long>(checks));
    }
};
std::vector<Weight> model_slice(const std::vector<Weight>& full, int rank, int rows) {
    std::vector<Weight> result;
    for (int index = 0; index < 4; ++index) {
        const int inner = (rank ^ (index % 2)) ? 256 : 384;
        const int begin = rank ? 640 - inner : 0;
        Weight slice(rows, inner);
        slice.table = full[index].table;
        slice.level = full[index].level;
        for (int row = 0; row < rows; ++row) {
            std::memcpy(slice.codes.data() + size_t(row) * (inner / 2),
                full[index].codes.data() + size_t(row) * 320 + begin / 2, size_t(inner / 2));
            std::memcpy(slice.scales.data() + size_t(row) * (inner / 64),
                full[index].scales.data() + size_t(row) * 10 + begin / 64, size_t(inner / 64));
        }
        result.push_back(std::move(slice));
    }
    return result;
}
void model_tests(const std::string& path, const std::string& manifest) {
    Model model(path);
    const auto full = model.load_manifest(manifest);
    for (int rank = 0; rank < 2; ++rank) {
        const auto slice = model_slice(full, rank, 2560);
        for (int tokens : {1, 3, 17}) gpu_case(tokens, tokens == 1 ? 2 : tokens == 3 ? 10 : 64,
            bool(rank), false, 2560, &slice);
        gpu_case(2048, 512, bool(rank), false, 2560, &slice, false, 0, 0, true);
        const auto small = model_slice(full, rank, 16);
        gpu_case(2048, 512, bool(rank), false, 16, &small);
    }
}
