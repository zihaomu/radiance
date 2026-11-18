struct RouterN1Model {
    std::ifstream input;
    uint64_t size = 0;
    RadFileHeader header{};
    std::vector<char> strings;
    std::vector<RadFileEntry> entries;
    std::vector<RadFilePlane> planes;
    std::vector<RadEncoding> encodings;
    explicit RouterN1Model(const char* path) : input(path, std::ios::binary) {
        require(bool(input), "N1 open actual model");
        input.seekg(0, std::ios::end);
        const auto length = input.tellg();
        require(length > 0, "N1 model size");
        size = static_cast<uint64_t>(length);
        read(0, &header, sizeof(header));
        require(header.magic == RAD_MAGIC && header.version == RAD_FORMAT_VER
            && header.file_bytes == size && size == 121678372864ull, "N1 target model format/size");
        strings = section<char>(header.str_off, header.str_bytes);
        entries = section<RadFileEntry>(header.dir_off, header.dir_count);
        planes = section<RadFilePlane>(header.plane_off, header.plane_count);
        encodings = section<RadEncoding>(header.enc_off, header.enc_count);
    }
    void read(uint64_t offset, void* target, size_t count) {
        require(offset <= size && count <= size - offset, "N1 bounded model read");
        input.clear(); input.seekg(static_cast<std::streamoff>(offset));
        input.read(static_cast<char*>(target), static_cast<std::streamsize>(count));
        require(bool(input), "N1 complete model read");
    }
    template<class Element> std::vector<Element> section(uint64_t offset, uint64_t count) {
        require(count < 16000000 && count <= size / sizeof(Element), "N1 bounded model metadata");
        std::vector<Element> result(static_cast<size_t>(count));
        if (count) read(offset, result.data(), result.size() * sizeof(Element));
        return result;
    }
    std::vector<uint16_t> weight(int layer, bool log) {
        const std::string name = "blk." + std::to_string(layer) + ".ffn_gate_shexp.weight";
        const RadFileEntry* selected = nullptr;
        for (const auto& entry : entries) {
            require(entry.name < strings.size()
                && std::memchr(strings.data() + entry.name, 0, strings.size() - entry.name), "N1 bounded tensor name");
            if (name != strings.data() + entry.name) continue;
            require(!selected, "N1 unique actual tensor");
            selected = &entry;
        }
        require(selected, "N1 actual shared-expert gate tensor present");
        const auto& entry = *selected;
        require(entry.rank == 2 && entry.shape[0] == 1 && entry.shape[1] == 2560, "N1 actual weight shape [1,2560]");
        require(entry.enc < encodings.size() && entry.n_planes == 1
            && entry.plane_first < planes.size(), "N1 actual weight plane");
        const auto& encoding = encodings[entry.enc];
        require(encoding.n_planes == 1 && encoding.plane[0].dtype == RAD_BF16
            && rad_enc_is(&encoding, "plain") && !encoding.transform[0], "N1 actual weight plain BF16");
        const auto& plane = planes[static_cast<size_t>(entry.plane_first)];
        require(plane.bytes == 5120 && plane.offset >= header.data_off, "N1 actual weight 5120 bytes");
        std::vector<uint16_t> result(2560);
        read(plane.offset, result.data(), plane.bytes);
        uint64_t hash = 14695981039346656037ull;
        bool nonzero = false;
        for (const uint16_t value : result) {
            require(std::isfinite(fp32(value)), "N1 finite actual weights");
            nonzero = nonzero || (value & 0x7fff) != 0;
            hash = (hash ^ (value & 255)) * 1099511628211ull;
            hash = (hash ^ (value >> 8)) * 1099511628211ull;
        }
        require(nonzero, "N1 nonzero actual weight");
        if (log) std::printf("QWEN_N1_WEIGHT name=%s shape=1x2560 dtype=bf16 offset=%llu bytes=5120 fnv1a64=%016llx\n",
            name.c_str(), (unsigned long long)plane.offset, (unsigned long long)hash);
        return result;
    }
};
void router_n1_model(const char* path, hipStream_t first, hipStream_t second) {
    RouterN1Model model(path);
    for (int layer = 0; layer < 48; ++layer) model.weight(layer, true);
    std::printf("QWEN_N1_MANIFEST_PASS actual_layers=48 shape=1x2560 dtype=bf16 bytes_each=5120\n");
    const uint64_t cases_before = n1_cases, launches_before = n1_launches, values_before = n1_values;
    for (int layer : {0, 23, 47}) {
        const auto weight = model.weight(layer, false);
        const std::string label = "blk." + std::to_string(layer) + ".ffn_gate_shexp.weight";
        for (int rows : {1, 3, 8, 16, 32, 64, 128, 512, 2048})
            for (bool padded : {false, true}) router_n1_case(rows, padded, first, second, &weight, label.c_str());
    }
    std::printf("QWEN_N1_ACTUAL_PASS layers=0,23,47 cases=%llu launches=%llu cpu_gpu=%llu cpu_libref=%llu libref_gpu=%llu mismatches=0\n",
        (unsigned long long)(n1_cases - cases_before), (unsigned long long)(n1_launches - launches_before),
        (unsigned long long)(n1_values - values_before), (unsigned long long)(n1_values - values_before),
        (unsigned long long)(n1_values - values_before));
}
