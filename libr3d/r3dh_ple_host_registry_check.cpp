#include "../core/plugin/registry.h"
#include <array>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <stdexcept>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
RadTensor tensor(void* data, uint32_t type, int64_t rows, int64_t columns = 0) {
    RadTensor result{};
    result.data = data;
    result.dtype = type;
    result.rank = columns ? 2 : 1;
    result.shape[0] = rows;
    result.shape[1] = columns;
    result.stride[0] = columns ? columns : 1;
    result.stride[1] = 1;
    return result;
}
}
int main(int argc, char** argv) {
    try {
        require(argc == 3, "usage: registry_check plugin.so provider.so");
        void* plugin = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
        require(plugin, "plugin metadata handle");
        const auto at_plugin = reinterpret_cast<const RadKernelInfo*(*)(int)>(dlsym(plugin, "rad_kernel_at"));
        require(at_plugin, "plugin metadata export");
        const auto* stable = at_plugin(0);
        for (int cycle = 0; cycle < 2; ++cycle) {
            if (cycle == 0) {
                void* existing = dlopen(argv[2], RTLD_NOW | RTLD_NOLOAD);
                if (existing) dlclose(existing);
                require(!existing, "provider must not be preopened");
            }
            require(stable && at_plugin(0) == stable && stable->launch(nullptr, nullptr) == RAD_E_STATE,
                "Registry opens from a logically closed provider without preopen");
            rad::Registry registry;
            const int status = registry.load_plugin(argv[1], 0);
            for (const auto& error : registry.load_errors())
                std::fprintf(stderr, "Registry: %s\n", error.c_str());
            require(status == RAD_OK, "real Registry loads plugin without preopen");
            require(registry.plugins().size() == 1 && registry.rows().size() == 1,
                "real Registry must register exactly one gather row");
            require(registry.plugins()[0].n_kernels == 1 && registry.plugins()[0].n_schemas == 1,
                "real Registry records one HOST schema and kernel");
            for (const auto& row : registry.rows())
                require(row.info && row.info->domain == RAD_DOMAIN_HOST && row.info->launch,
                    "Registry HOST descriptors are complete");
            const auto* ids = registry.select("ngram_ids", {}, RAD_DOMAIN_HOST);
            const auto* gather = registry.select("embed_lookup_q", {}, RAD_DOMAIN_HOST);
            require(!ids && !registry.select("ngram_hint", {}, RAD_DOMAIN_HOST) && gather,
                "public HOST plugin exposes only gather, never device-state ngram operations");
            require(!at_plugin(-1) && !at_plugin(1), "public gather-only bounds");
            require(!registry.select("ngram_ids", {}, RAD_DOMAIN_DEVICE)
                && !registry.select("embed_lookup_q", {}, RAD_DOMAIN_DEVICE), "no DEVICE fallback");
            std::array<uint16_t, 160> table{}, output{};
            for (int index = 0; index < 160; ++index) table[index] = uint16_t(0x3f00 + index + cycle);
            int32_t token = 0;
            RadTensor values[]{tensor(&token, RAD_I32, 1), tensor(table.data(), RAD_BF16, 1, 160), {},
                tensor(output.data(), RAD_BF16, 1, 160), {}};
            RadParam dtype{};
            dtype.key = "dtype";
            dtype.kind = RAD_P_STR;
            dtype.sval = "bf16";
            RadArgs args{};
            args.t = values;
            args.n_t = 5;
            args.p = &dtype;
            args.n_p = 1;
            args.world_size = 1;
            require(gather->info->launch(&args, nullptr) == RAD_OK, "Registry real borrowed gather launch");
            require(output == table, "Registry BF16 gather oracle");
            RadParam geometry[]{dtype, {}, {}, {}};
            const char* keys[]{"M", "n_embd", "n_vocab"};
            const int64_t extents[]{1, 160, 1};
            for (int index = 0; index < 3; ++index) {
                geometry[index + 1].key = keys[index];
                geometry[index + 1].kind = RAD_P_INT;
                geometry[index + 1].ival = extents[index];
            }
            RadOpdDesc shape{};
            void* borrowed = dlopen(argv[2], RTLD_NOW | RTLD_NOLOAD);
            require(borrowed, "Registry opened real provider");
            const auto count = reinterpret_cast<int(*)()>(dlsym(borrowed, "rad_kernel_count"));
            const auto at = reinterpret_cast<const RadKernelInfo*(*)(int)>(dlsym(borrowed, "rad_kernel_at"));
            require(count && at, "real provider enumeration");
            const RadKernelInfo* actual = nullptr;
            for (int index = 0; index < count(); ++index) {
                const auto* row = at(index);
                if (row && row->name && !std::strcmp(row->name, "avx_embed_lookup_q")) actual = row;
            }
            require(actual, "real provider gather row");
            RadOpdDesc expected{};
            const int shape_status = actual->opd_shape
                ? actual->opd_shape(geometry, 4, 0, &expected) : RAD_E_UNSUPPORTED;
            require(gather->info->opd_shape(geometry, 4, 0, &shape) == shape_status
                && !std::memcmp(&shape, &expected, sizeof(shape)), "dynamic provider shape status preserved");
            require(dlclose(borrowed) == 0, "release borrowed test handle");
            std::printf("PLE_HOST_REAL_REGISTRY cycle=%d HOST=%zu ids=0 hint=0 gather=1 preopen=0\n",
                cycle, registry.rows().size());
        }
        require(stable->launch(nullptr, nullptr) == RAD_E_STATE, "Registry close releases provider");
        require(dlclose(plugin) == 0, "release plugin metadata handle");
        std::puts("PLE_HOST_REAL_REGISTRY_PASS HOST=1 DEVICE=0 preopen=0 close_reopen=2 GPU_used=0");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "PLE_HOST_REAL_REGISTRY_FAIL %s\n", error.what());
        return 1;
    }
}
