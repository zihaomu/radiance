struct Model {
    std::ifstream input;
    uint64_t size = 0;
    RadFileHeader header{};
    std::vector<char> strings;
    std::vector<RadFileEntry> directory;
    std::vector<RadFilePlane> planes;
    std::vector<RadEncoding> encodings;
    explicit Model(const char* path) : input(path, std::ios::binary) {
        expect(bool(input), "open current model");
        input.seekg(0, std::ios::end); size = uint64_t(input.tellg());
        read(0, &header, sizeof(header));
        expect(header.magic == RAD_MAGIC && header.version == RAD_FORMAT_VER && header.file_bytes == size && size == 121678372864ull, "target model identity/size");
        strings = section<char>(header.str_off, header.str_bytes);
        directory = section<RadFileEntry>(header.dir_off, header.dir_count);
        planes = section<RadFilePlane>(header.plane_off, header.plane_count);
        encodings = section<RadEncoding>(header.enc_off, header.enc_count);
        std::printf("MODEL_MANIFEST file_bytes=%llu metadata_fingerprint=%llu\n", static_cast<unsigned long long>(size), static_cast<unsigned long long>(hash(directory.data(), directory.size() * sizeof(RadFileEntry))));
        for (const auto& entry : directory) {
            const char* text = name(entry);
            if (std::strstr(text, "ple") || !std::strcmp(text, "output.weight")) {
                const auto& encoding = encodings[entry.enc];
                std::printf("MODEL_SCHEMA %s shape=%lld,%lld scheme=%s transform=%s\n", text, static_cast<long long>(entry.shape[0]), static_cast<long long>(entry.shape[1]), encoding.scheme, encoding.transform);
            }
        }
    }
    static uint64_t hash(const void* memory, size_t count) {
        const auto* data = static_cast<const uint8_t*>(memory); uint64_t value = 14695981039346656037ull;
        for (size_t index = 0; index < count; ++index) value = (value ^ data[index]) * 1099511628211ull;
        return value;
    }
    void read(uint64_t offset, void* destination, size_t bytes) {
        expect(offset <= size && bytes <= size - offset, "bounded model read");
        input.clear(); input.seekg(std::streamoff(offset)); input.read(static_cast<char*>(destination), std::streamsize(bytes));
        expect(bool(input), "read current model plane");
    }
    template<class Element> std::vector<Element> section(uint64_t offset, uint64_t count) {
        expect(count < 16000000 && count <= size / sizeof(Element), "bounded metadata count");
        std::vector<Element> result(static_cast<size_t>(count));
        if (count) read(offset, result.data(), result.size() * sizeof(Element));
        return result;
    }
    const char* name(const RadFileEntry& entry) const {
        expect(entry.name < strings.size() && entry.enc < encodings.size() &&
            std::memchr(strings.data() + entry.name, 0, strings.size() - entry.name), "bounded model weight name/encoding");
        return strings.data() + entry.name;
    }
    const RadFileEntry* find(const char* text, bool suffix = false) const {
        for (const auto& entry : directory) {
            const char* candidate = name(entry);
            if ((!suffix && !std::strcmp(candidate, text)) || (suffix && std::strstr(candidate, text))) return &entry;
        }
        return nullptr;
    }
    void plane(const RadFileEntry& entry, const char* role, Blob& destination) {
        const auto& encoding = encodings[entry.enc]; const int index = rad_enc_find(&encoding, role);
        expect(index >= 0 && entry.n_planes == uint32_t(encoding.n_planes) && entry.plane_first <= planes.size() &&
            uint64_t(index) < planes.size() - entry.plane_first, "model plane role/index");
        const auto& descriptor = planes[size_t(entry.plane_first + uint64_t(index))];
        expect(descriptor.offset >= header.data_off && descriptor.bytes == destination.size() * size_t(rad_dtype_bits(destination.dtype) / 8) &&
            encoding.plane[index].dtype == destination.dtype, "model plane dtype/byte extent");
        read(descriptor.offset, destination.data<void>(), size_t(descriptor.bytes));
        std::printf("MODEL_PLANE %s role=%s dtype=%u offset=%llu bytes=%llu fnv64=%llu\n", name(entry), role, destination.dtype,
            static_cast<unsigned long long>(descriptor.offset), static_cast<unsigned long long>(descriptor.bytes), static_cast<unsigned long long>(hash(destination.data<void>(), size_t(descriptor.bytes))));
    }
    void gain(const char* text, Blob& destination, bool suffix = false) {
        const auto* entry = find(text, suffix); expect(entry, "required real norm gain exists");
        expect(rad_enc_is(&encodings[entry->enc], "plain") && entry->rank == 1 && uint64_t(entry->shape[0]) == destination.size(), "real norm gain schema");
        const auto* stored = rad_enc_plane(&encodings[entry->enc], "codes");
        expect(stored && destination.dtype == RAD_F32, "gain runtime dtype");
        if (stored->dtype == RAD_F32) { plane(*entry, "codes", destination); return; }
        expect(stored->dtype == RAD_BF16, "gain storage must be F32 or BF16");
        Blob raw(destination.size(), RAD_BF16);
        plane(*entry, "codes", raw);
        for (size_t index = 0; index < destination.size(); ++index) {
            const uint32_t bits = uint32_t(raw.data<uint16_t>()[index]) << 16;
            std::memcpy(destination.data<float>() + index, &bits, sizeof(bits));
        }
        std::printf("MODEL_GAIN_CONVERT %s bf16_to_f32_exact=1\n", text);
    }
    void matrix(const char* text, Blob& codes, Blob& scales, int64_t width) {
        const auto* entry = find(text); expect(entry, "required real MTP projection exists");
        const auto& encoding = encodings[entry->enc];
        const auto* code = rad_enc_plane(&encoding, "codes"); const auto* scale = rad_enc_plane(&encoding, "scale");
        expect(entry->rank == 2 && entry->shape[0] == width && entry->shape[1] == width && rad_enc_is(&encoding, "affine") && encoding.n_planes == 2 && !encoding.transform[0] &&
            code && scale && code->dtype == RAD_F8E4M3 && code->block[0] == 1 && code->block[1] == 1 && scale->dtype == RAD_F32 && scale->block[0] == 1 && scale->block[1] == 128, "real MTP E4M3 group128 schema");
        plane(*entry, "codes", codes); plane(*entry, "scale", scales);
    }
};
Model* current_model = nullptr;
