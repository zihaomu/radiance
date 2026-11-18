#include "r3dh_ple_host.h"
#include <array>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
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
RadTensor tensor(void* data, uint32_t type, int64_t rows, int64_t columns = 0) {
    RadTensor result{}; result.data = data; result.dtype = type;
    result.rank = columns ? 2 : 1; result.shape[0] = rows; result.shape[1] = columns;
    result.stride[0] = columns ? columns : 1; result.stride[1] = 1; return result;
}
RadParam integer(const char* key, int64_t value) {
    RadParam result{}; result.key = key; result.kind = RAD_P_INT; result.ival = value; return result;
}
struct File {
    static constexpr uint64_t rows = 320001536, row_bytes = 320, offset = 4096;
    static constexpr uint64_t bytes = offset + rows * row_bytes;
    int descriptor = -1;
    void* mapping = MAP_FAILED;
    std::string path;
    struct stat identity{};
    explicit File(const char* directory) {
        path = std::string(directory) + "/r3dh_ple_host_sparse_XXXXXX";
        std::vector<char> name(path.begin(), path.end()); name.push_back(0);
        descriptor = mkstemp(name.data());
        if (descriptor < 0) throw std::runtime_error("sparse mkstemp");
        path = name.data();
        if (ftruncate(descriptor, off_t(bytes))) { cleanup(); throw std::runtime_error("sparse ftruncate"); }
        mapping = mmap(nullptr, size_t(bytes), PROT_READ, MAP_SHARED, descriptor, 0);
        if (mapping == MAP_FAILED) { cleanup(); throw std::runtime_error("sparse mmap"); }
        if (fstat(descriptor, &identity) || uint64_t(identity.st_size) != bytes) {
            cleanup(); throw std::runtime_error("sparse fstat size");
        }
    }
    bool cleanup() {
        bool success = true;
        if (mapping != MAP_FAILED) { success = munmap(mapping, size_t(bytes)) == 0; mapping = MAP_FAILED; }
        if (descriptor >= 0) { success = close(descriptor) == 0 && success; descriptor = -1; }
        if (!path.empty()) {
            success = unlink(path.c_str()) == 0 && success;
            path.clear();
        }
        return success;
    }
    ~File() { cleanup(); }
};
struct Instance {
    const RadKernelInfo* row;
    void* value = nullptr;
    ~Instance() { if (value && row->fini) row->fini(value); }
};
uint16_t oracle(int32_t row, int column) {
    return uint16_t(0x3c00 + (uint64_t(uint32_t(row)) * 37 + uint64_t(column) * 11) % 2048);
}
}
int main(int argc, char** argv) {
    try {
        require(argc == 3, "usage: sparse_check provider_or_HOME fixture_directory");
        require(r3dh_ple_host_open(argv[1]) == RAD_OK, "real provider open");
        const auto* row = r3dh_ple_host_registry_at(1);
        require(row && row->domain == RAD_DOMAIN_HOST, "HOST gather only");
        RadParam params[]{integer("M", 5), integer("n_embd", 160), integer("n_vocab", File::rows), {}};
        params[3].key = "dtype"; params[3].kind = RAD_P_STR; params[3].sval = "bf16";
        File file(argv[2]);
        int32_t tokens[]{0, 12, 1234567, 160000768, 320001535}, ahead[]{7, 320001534};
        for (int32_t token : {0, 12, 1234567, 160000768, 320001535, 7, 320001534}) {
            std::array<uint16_t, 160> stored{};
            for (int column = 0; column < 160; ++column) stored[column] = oracle(token, column);
            require(pwrite(file.descriptor, stored.data(), sizeof(stored),
                off_t(File::offset + uint64_t(token) * File::row_bytes)) == ssize_t(sizeof(stored)), "sparse pwrite");
        }
        require(fsync(file.descriptor) == 0, "sparse fsync");
        struct stat opened{}, named{};
        require(fstat(file.descriptor, &opened) == 0 && stat(file.path.c_str(), &named) == 0
            && opened.st_dev == file.identity.st_dev && opened.st_ino == file.identity.st_ino
            && opened.st_dev == named.st_dev && opened.st_ino == named.st_ino
            && uint64_t(opened.st_size) == File::bytes, "fstat identity and exact size");
        require(uint64_t(opened.st_blocks) * 512 < 1024 * 1024, "sparse allocation below one MiB");
        const uint64_t allocated = uint64_t(opened.st_blocks) * 512;
        const std::string path = file.path;
        {
            Instance instance{row};
            require(row->init(params, 4, 0, 1, &instance.value) == RAD_OK, "real reader init");
            std::vector<uint16_t> output(802, 0xa5a5);
            auto* table = static_cast<unsigned char*>(file.mapping) + File::offset;
            RadTensor values[]{tensor(tokens, RAD_I32, 5), tensor(table, RAD_BF16, File::rows, 160), {},
                tensor(output.data() + 1, RAD_BF16, 1, 800), tensor(ahead, RAD_I32, 2)};
            RadArgs args{}; args.t = values; args.n_t = 5; args.p = params; args.n_p = 4;
            args.world_size = 1; args.instance = instance.value;
            require(row->launch(&args, nullptr) == RAD_OK, "huge sparse BF16 demand launch");
            for (int index = 0; index < 800; ++index)
                require(output[index + 1] == oracle(tokens[index / 160], index % 160), "sparse BF16 oracle");
            require(output.front() == 0xa5a5 && output.back() == 0xa5a5, "demand guards");
            std::fill(output.begin(), output.end(), 0xa5a5);
            values[0] = tensor(ahead, RAD_I32, 2); values[3] = tensor(output.data() + 1, RAD_BF16, 1, 320); values[4] = {};
            require(row->launch(&args, nullptr) == RAD_OK, "huge sparse BF16 ahead launch");
            for (int index = 0; index < 320; ++index)
                require(output[index + 1] == oracle(ahead[index / 160], index % 160), "sparse ahead oracle");
            require(output.front() == 0xa5a5 && output[321] == 0xa5a5, "ahead guards");
        }
        struct stat closing{};
        require(fstat(file.descriptor, &closing) == 0
            && closing.st_dev == opened.st_dev && closing.st_ino == opened.st_ino,
            "caller retains original file until cleanup");
        require(file.cleanup() && file.descriptor == -1, "owned descriptor close and unmap succeed");
        errno = 0;
        require(access(path.c_str(), F_OK) == -1 && errno == ENOENT && file.mapping == MAP_FAILED, "fixture unlinked and unmapped");
        std::printf("PLE_HOST_SPARSE_PASS checks=%llu rows=7 values=1120 logical_table_bytes=%llu file_bytes=%llu allocated_bytes=%llu last_row=320001535 cross_4K_row=12 fstat_identity=1 owned_cleanup=1 GPU_used=0\n",
            static_cast<unsigned long long>(checks), static_cast<unsigned long long>(File::rows * File::row_bytes),
            static_cast<unsigned long long>(File::bytes), static_cast<unsigned long long>(allocated));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "PLE_HOST_SPARSE_FAIL %s checks=%llu\n", error.what(), static_cast<unsigned long long>(checks)); return 1;
    }
}
