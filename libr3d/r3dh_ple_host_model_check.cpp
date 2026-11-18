#include "r3dh_ple_host.h"
#include "rad_format.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace {
uint64_t checks = 0;
void require(bool condition, const char* message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}
struct Model {
    int descriptor = -1;
    uint64_t bytes = 0;
    struct stat identity{};
    explicit Model(const char* path) {
        descriptor = open(path, O_RDONLY | O_CLOEXEC);
        require(descriptor >= 0, "open model");
        struct stat status{};
        if (fstat(descriptor, &status) || status.st_size < 0) {
            close(descriptor); descriptor = -1; throw std::runtime_error("stat model");
        }
        bytes = uint64_t(status.st_size);
        identity = status;
    }
    ~Model() { if (descriptor >= 0) close(descriptor); }
    void read(uint64_t offset, void* output, uint64_t size) const {
        require(offset <= bytes && size <= bytes - offset && size <= 128 * 1024 * 1024, "bounded model read");
        require(pread(descriptor, output, size_t(size), off_t(offset)) == ssize_t(size), "complete model read");
    }
    template<class Element> std::vector<Element> section(uint64_t offset, uint64_t count) const {
        require(count <= (128 * 1024 * 1024) / sizeof(Element), "bounded metadata section");
        std::vector<Element> result(static_cast<size_t>(count));
        read(offset, result.data(), count * sizeof(Element)); return result;
    }
};
RadTensor tensor(void* data, uint32_t type, int64_t rows, int64_t columns = 0) {
    RadTensor result{}; result.data = data; result.dtype = type; result.rank = columns ? 2 : 1;
    result.shape[0] = rows; result.shape[1] = columns;
    result.stride[0] = columns ? columns : 1; result.stride[1] = 1; return result;
}
RadParam integer(const char* key, int64_t value) {
    RadParam result{}; result.key = key; result.kind = RAD_P_INT; result.ival = value; return result;
}
struct Mapping {
    void* data;
    uint64_t bytes;
    Mapping(int descriptor, uint64_t count) : data(mmap(nullptr, size_t(count), PROT_READ, MAP_SHARED, descriptor, 0)), bytes(count) {
        require(data != MAP_FAILED, "read-only file mapping");
    }
    ~Mapping() { if (data != MAP_FAILED) munmap(data, size_t(bytes)); }
};
}
int main(int argc, char** argv) {
    try {
        require(argc == 3, "usage: model_check libavx.so model.rad");
        Model model(argv[2]);
        RadFileHeader header{}; model.read(0, &header, sizeof(header));
        require(header.magic == RAD_MAGIC && header.version == RAD_FORMAT_VER
            && header.file_bytes == model.bytes && model.bytes == 121678372864ull, "target model identity/size");
        const auto strings = model.section<char>(header.str_off, header.str_bytes);
        const auto directory = model.section<RadFileEntry>(header.dir_off, header.dir_count);
        const auto planes = model.section<RadFilePlane>(header.plane_off, header.plane_count);
        const auto encodings = model.section<RadEncoding>(header.enc_off, header.enc_count);
        const RadFileEntry* target = nullptr;
        uint64_t fingerprint = 14695981039346656037ull;
        for (const auto& entry : directory) {
            require(entry.name < strings.size() && std::memchr(strings.data() + entry.name, 0, strings.size() - entry.name), "bounded weight name");
            require(entry.enc < encodings.size() && entry.plane_first <= planes.size()
                && entry.n_planes <= planes.size() - entry.plane_first, "bounded encoding planes");
            if (!std::strcmp(strings.data() + entry.name, "blk.1.ple_ngram.weight")) target = &entry;
        }
        const auto* raw = reinterpret_cast<const unsigned char*>(directory.data());
        for (size_t index = 0; index < directory.size() * sizeof(RadFileEntry); ++index)
            fingerprint = (fingerprint ^ raw[index]) * 1099511628211ull;
        require(fingerprint == 2682156130085983292ull, "pinned model directory identity");
        require(target && target->rank == 2 && target->shape[0] == 320001536 && target->shape[1] == 160, "huge PLE dimensions");
        const auto& encoding = encodings[target->enc];
        require(std::memchr(encoding.scheme, 0, RAD_ENC_STR) && std::memchr(encoding.transform, 0, RAD_ENC_STR)
            && !encoding.transform[0] && encoding.n_planes > 0
            && uint32_t(encoding.n_planes) == target->n_planes && target->n_planes >= 1, "stored PLE encoding");
        const bool bf16 = !std::strcmp(encoding.scheme, "plain") && encoding.plane[0].dtype == RAD_BF16 && target->n_planes == 1;
        const bool e4m3 = !std::strcmp(encoding.scheme, "affine") && encoding.plane[0].dtype == RAD_F8E4M3 && target->n_planes == 2;
        require(bf16 || e4m3, "supported stored PLE dtype");
        const auto& stored = planes[target->plane_first];
        const uint64_t row_bytes = bf16 ? 320 : 160;
        require(stored.bytes == uint64_t(target->shape[0]) * row_bytes && stored.offset <= model.bytes
            && stored.bytes <= model.bytes - stored.offset, "bounded huge table plane");
        require(stored.offset == 75370496 && stored.bytes == 51200245760ull && e4m3,
            "pinned complete current-model PLE plane");
        uint16_t scale = 0x3f80;
        if (e4m3) {
            const auto& scalar = planes[target->plane_first + 1];
            require(encoding.plane[1].dtype == RAD_BF16 && scalar.bytes == 2, "scalar BF16 scale");
            model.read(scalar.offset, &scale, sizeof(scale));
        }
        require(r3dh_ple_host_open(argv[1]) == RAD_OK, "private model provider");
        const auto* row = r3dh_ple_host_registry_at(1);
        RadParam params[]{integer("M", 5), integer("n_embd", 160), integer("n_vocab", target->shape[0]), {}};
        params[3].key = "dtype"; params[3].kind = RAD_P_STR; params[3].sval = "bf16";
        void* instance = nullptr;
        const int initialized = row->init(params, 4, 0, 1, &instance);
        if (initialized == RAD_E_UNSUPPORTED) { std::puts("PLE_HOST_MODEL_SKIP io_uring_unavailable=1"); return 77; }
        require(initialized == RAD_OK, "huge table reader init");
        Mapping mapping(model.descriptor, model.bytes);
        auto* table = static_cast<unsigned char*>(mapping.data) + stored.offset;
        int32_t tokens[]{0, 1, 1234567, 160000768, 320001535}, ahead[]{7, 320001534};
        std::vector<uint16_t> output(802, 0xa5a5);
        RadTensor values[]{tensor(tokens, RAD_I32, 5), tensor(table, bf16 ? RAD_BF16 : RAD_F8E4M3, target->shape[0], 160),
            bf16 ? RadTensor{} : tensor(&scale, RAD_BF16, 1, 1), tensor(output.data() + 1, RAD_BF16, 1, 800), tensor(ahead, RAD_I32, 2)};
        RadArgs args{}; args.t = values; args.n_t = 5; args.p = params; args.n_p = 4; args.instance = instance; args.world_size = 1;
        auto compare = [&](const int32_t* ids, int count) {
            for (int index = 0; index < count; ++index) {
                std::vector<unsigned char> bytes(static_cast<size_t>(row_bytes));
                model.read(stored.offset + uint64_t(ids[index]) * row_bytes, bytes.data(), row_bytes);
                for (int column = 0; column < 160; ++column) {
                    uint16_t expected;
                    if (bf16) std::memcpy(&expected, bytes.data() + column * 2, 2);
                    else expected = rad_f32_to_bf16(rad_fp8e4m3_to_f32(bytes[column]) * rad_bf16_to_f32(scale));
                    require(output[1 + index * 160 + column] == expected, "actual model row oracle");
                }
            }
        };
        require(row->launch(&args, nullptr) == RAD_OK, "huge table sparse HOST gather"); compare(tokens, 5);
        require(output.front() == 0xa5a5 && output.back() == 0xa5a5, "huge gather guards");
        values[0] = tensor(ahead, RAD_I32, 2); values[3] = tensor(output.data() + 1, RAD_BF16, 1, 320); values[4] = {};
        std::fill(output.begin(), output.end(), 0xa5a5);
        require(row->launch(&args, nullptr) == RAD_OK, "huge table ahead gather"); compare(ahead, 2);
        require(output.front() == 0xa5a5 && output[321] == 0xa5a5, "huge ahead guards");
        if (row->fini) row->fini(instance);
        struct stat final_identity{};
        require(fstat(model.descriptor, &final_identity) == 0
            && final_identity.st_dev == model.identity.st_dev && final_identity.st_ino == model.identity.st_ino
            && final_identity.st_size == model.identity.st_size
            && final_identity.st_mtim.tv_sec == model.identity.st_mtim.tv_sec
            && final_identity.st_mtim.tv_nsec == model.identity.st_mtim.tv_nsec
            && final_identity.st_ctim.tv_sec == model.identity.st_ctim.tv_sec
            && final_identity.st_ctim.tv_nsec == model.identity.st_ctim.tv_nsec, "stable model fstat identity");
        std::printf("PLE_HOST_MODEL_PASS checks=%llu rows=7 values=1120 file_bytes=%llu metadata_fnv1a64=%llu table_offset=%llu table_bytes=%llu storage=%s expanded=0 fstat_identity=1 full_sha256_recomputed=0 GPU_used=0\n",
            static_cast<unsigned long long>(checks), static_cast<unsigned long long>(model.bytes), static_cast<unsigned long long>(fingerprint),
            static_cast<unsigned long long>(stored.offset), static_cast<unsigned long long>(stored.bytes), bf16 ? "bf16" : "e4m3");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "PLE_HOST_MODEL_FAIL %s checks=%llu\n", error.what(), static_cast<unsigned long long>(checks)); return 1;
    }
}
