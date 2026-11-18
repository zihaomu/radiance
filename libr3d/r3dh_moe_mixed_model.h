#include "rad_format.h"
#include "rad_encoding.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
struct Model {
    std::ifstream input;
    RadFileHeader header{};
    std::vector<char> names;
    std::vector<RadFileEntry> entries;
    std::vector<RadEncoding> encodings;
    std::vector<RadFilePlane> planes;
    explicit Model(const char* path) : input(path, std::ios::binary) {
        if (!input) throw std::runtime_error("open model");
        input.seekg(0, std::ios::end);
        const auto bytes = input.tellg();
        read(0, &header, sizeof(header), uint64_t(bytes));
        if (header.magic != RAD_MAGIC || header.version != RAD_FORMAT_VER
            || header.file_bytes != uint64_t(bytes) || header.file_bytes != 121678372864ull
            || header.str_bytes > 100000000 || header.dir_count > 1000000
            || header.enc_count > 100000 || header.plane_count > 10000000)
            throw std::runtime_error("model identity/header bounds");
        section(names, header.str_off, header.str_bytes);
        section(entries, header.dir_off, header.dir_count);
        section(encodings, header.enc_off, header.enc_count);
        section(planes, header.plane_off, header.plane_count);
    }
    void read(uint64_t offset, void* output, uint64_t bytes, uint64_t limit = 0) {
        if (!limit) limit = header.file_bytes;
        if (offset > limit || bytes > limit - offset) throw std::runtime_error("read bounds");
        input.clear(); input.seekg(offset); input.read(static_cast<char*>(output), bytes);
        if (uint64_t(input.gcount()) != bytes) throw std::runtime_error("short model read");
    }
    template<class Value> void section(std::vector<Value>& output, uint64_t offset, uint64_t count) {
        if (count > header.file_bytes / sizeof(Value)) throw std::runtime_error("section bounds");
        output.resize(count); read(offset, output.data(), count * sizeof(Value));
    }
    std::string name(const RadFileEntry& entry) const {
        if (entry.name >= names.size()
            || !std::memchr(names.data() + entry.name, 0, names.size() - entry.name))
            throw std::runtime_error("entry name");
        return names.data() + entry.name;
    }
};
}
