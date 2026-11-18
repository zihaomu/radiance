#include "r3dh_ple_host.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

namespace {
uint64_t checks = 0;
void require(bool condition, const char* message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}
RadTensor tensor(void* data, uint32_t type, int64_t rows, int64_t columns = 0) {
    RadTensor result{}; result.data = data; result.dtype = type; result.rank = columns ? 2 : 1;
    result.shape[0] = rows; result.shape[1] = columns;
    result.stride[0] = columns ? columns : 1; result.stride[1] = 1; return result;
}
RadParam integer(const char* key, int64_t value) {
    RadParam result{}; result.key = key; result.kind = RAD_P_INT; result.ival = value; return result;
}
struct File {
    std::string path;
    int descriptor = -1;
    void* mapping = MAP_FAILED;
    size_t bytes = 131072;
    explicit File(const char* directory) {
        path = std::string(directory) + "/r3dh_ple_host_table_XXXXXX";
        std::vector<char> name(path.begin(), path.end()); name.push_back(0);
        descriptor = mkstemp(name.data()); path = name.data();
        if (descriptor < 0) throw std::runtime_error("mkstemp");
        if (ftruncate(descriptor, off_t(bytes))) { close(descriptor); unlink(path.c_str()); throw std::runtime_error("ftruncate"); }
        mapping = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, descriptor, 0);
        if (mapping == MAP_FAILED) { close(descriptor); unlink(path.c_str()); throw std::runtime_error("mmap"); }
    }
    ~File() { if (mapping != MAP_FAILED) munmap(mapping, bytes); if (descriptor >= 0) close(descriptor); unlink(path.c_str()); }
};
float e4m3(uint8_t code) {
    const unsigned exponent = (code >> 3) & 15, mantissa = code & 7;
    const float value = exponent ? float(8 + mantissa) * float(uint32_t(1) << exponent) / 1024.0f
                                : float(mantissa) / 512.0f;
    return code & 128 ? -value : value;
}
void run(File& file, bool bf16, uint64_t (*reads)()) {
    constexpr int width = 160, rows = 128;
    auto* bytes = static_cast<uint8_t*>(file.mapping);
    const size_t table_offset = bf16 ? 32768 : 4096;
    void* table = bytes + table_offset;
    for (int index = 0; index < rows * width; ++index) {
        if (bf16) {
            const uint16_t value = uint16_t(0x3c00 + index % 1024);
            std::memcpy(bytes + table_offset + index * 2, &value, 2);
        } else bytes[table_offset + index] = uint8_t(0x20 + index % 64);
    }
    require(msync(file.mapping, file.bytes, MS_SYNC) == 0, "sync stored table");
    int32_t tokens[]{3, 7, 9, 17, 3, 127}, ahead[]{25, 39};
    uint16_t scale = 0x3f00;
    std::vector<uint16_t> output(6 * width + 2, 0xa5a5);
    RadParam params[]{integer("M", 6), integer("n_embd", width), integer("n_vocab", rows), {}};
    params[3].key = "dtype"; params[3].kind = RAD_P_STR; params[3].sval = "bf16";
    RadTensor values[]{tensor(tokens, RAD_I32, 6), tensor(table, bf16 ? RAD_BF16 : RAD_F8E4M3, rows, width),
        bf16 ? RadTensor{} : tensor(&scale, RAD_BF16, 1, 1), tensor(output.data() + 1, RAD_BF16, 1, 6 * width), tensor(ahead, RAD_I32, 2)};
    RadArgs args{}; args.t = values; args.n_t = 5; args.p = params; args.n_p = 4; args.world_size = 1;
    const auto* row = r3dh_ple_host_registry_at(1);
    require(row->init(params, 4, 0, 1, &args.instance) == RAD_OK, "reader init");
    const uint64_t before = reads();
    require(row->launch(&args, nullptr) == RAD_OK, "file-backed launch");
    const uint64_t after = reads();
    std::printf("PLE_HOST_FILE_READS bf16=%d before=%llu after=%llu\n", int(bf16), static_cast<unsigned long long>(before), static_cast<unsigned long long>(after));
    require(after - before >= 7 && after - before <= 8, "demand and ahead rows read from file");
    for (int index = 0; index < 6 * width; ++index) {
        uint16_t expected;
        if (bf16) std::memcpy(&expected, bytes + table_offset + (tokens[index / width] * width + index % width) * 2, 2);
        else expected = rad_f32_to_bf16(e4m3(bytes[table_offset + tokens[index / width] * width + index % width]) * 0.5f);
        require(output[index + 1] == expected, "independent mapped table oracle");
    }
    require(output.front() == 0xa5a5 && output.back() == 0xa5a5, "demand guards");
    values[0] = tensor(ahead, RAD_I32, 2); values[4] = {};
    values[3] = tensor(output.data() + 1, RAD_BF16, 1, 2 * width);
    std::fill(output.begin(), output.end(), 0xa5a5);
    require(row->launch(&args, nullptr) == RAD_OK && reads() == after, "ahead cache hit without disk read");
    for (int index = 0; index < 2 * width; ++index) {
        uint16_t expected;
        if (bf16) std::memcpy(&expected, bytes + table_offset + (ahead[index / width] * width + index % width) * 2, 2);
        else expected = rad_f32_to_bf16(e4m3(bytes[table_offset + ahead[index / width] * width + index % width]) * 0.5f);
        require(output[index + 1] == expected, "ahead cache data oracle");
    }
    require(output.front() == 0xa5a5 && output[321] == 0xa5a5, "ahead guards");
    if (row->fini) row->fini(args.instance);
    std::printf("PLE_HOST_FILE_CASE_PASS storage=%s read_rows=%llu compared=1280 ahead_cache_hit=1 offset=%zu\n",
        bf16 ? "bf16" : "e4m3", static_cast<unsigned long long>(after - before), table_offset);
}
}
int main(int argc, char** argv) {
    try {
        require(argc == 3, "usage: file_check libavx.so fixture_directory");
        require(r3dh_ple_host_open(argv[1]) == RAD_OK, "open private provider");
        const auto* row = r3dh_ple_host_registry_at(1);
        RadParam dtype{}; dtype.key = "dtype"; dtype.kind = RAD_P_STR; dtype.sval = "bf16";
        void* instance = nullptr;
        const int initialized = row->init(&dtype, 1, 0, 1, &instance);
        if (initialized == RAD_E_UNSUPPORTED) {
            std::puts("PLE_HOST_FILE_SKIP io_uring_unavailable=1 refusal_preserved=1 GPU_used=0"); return 77;
        }
        require(initialized == RAD_OK, "reader availability");
        if (row->fini) row->fini(instance);
        void* library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
        require(library != nullptr, "counter library");
        const auto reads = reinterpret_cast<uint64_t(*)()>(dlsym(library, "avx_ngram_reads"));
        require(reads != nullptr, "real disk counter");
        File first(argv[2]), second(argv[2]);
        run(first, false, reads); run(second, true, reads);
        dlclose(library);
        std::printf("PLE_HOST_FILE_PASS checks=%llu cases=2 values=2560 GPU_used=0\n", static_cast<unsigned long long>(checks));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "PLE_HOST_FILE_FAIL %s checks=%llu\n", error.what(), static_cast<unsigned long long>(checks)); return 1;
    }
}
