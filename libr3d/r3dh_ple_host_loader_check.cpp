#include "r3dh_ple_host.h"
#include "rad_abi_layout.h"
#include <array>
#include <atomic>
#include <chrono>
#include <future>
#include <thread>
#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <dlfcn.h>
#include <stdexcept>
#include <string>

namespace {
uint64_t checks = 0;
void require(bool condition, const char* message) {
    ++checks; if (!condition) throw std::runtime_error(message);
}
struct Library {
    void* handle;
    explicit Library(const char* path) : handle(dlopen(path, RTLD_NOW | RTLD_LOCAL)) {
        if (!handle) throw std::runtime_error(dlerror());
    }
    ~Library() { dlclose(handle); }
    template<class Function> Function symbol(const char* name) {
        auto result = reinterpret_cast<Function>(dlsym(handle, name));
        require(result != nullptr, name); return result;
    }
};
int descriptors() {
    DIR* directory = opendir("/proc/self/fd"); require(directory != nullptr, "fd directory");
    int count = 0;
    while (const auto* entry = readdir(directory)) if (entry->d_name[0] != 46) ++count;
    require(closedir(directory) == 0, "close fd directory"); return count;
}
RadTensor tensor(void* data, uint32_t type, int64_t rows, int64_t columns = 0) {
    RadTensor result{}; result.data = data; result.dtype = type;
    result.rank = columns ? 2 : 1; result.shape[0] = rows; result.shape[1] = columns;
    result.stride[0] = columns ? columns : 1; result.stride[1] = 1; return result;
}
}
int main(int argc, char** argv) {
    try {
        require(argc == 5, "usage: loader_check plugin.so provider.so HOME fixture_directory");
        Library plugin(argv[1]);
        const auto open = plugin.symbol<int(*)(const char*)>("r3dh_ple_host_open");
        const auto close = plugin.symbol<int(*)()>("r3dh_ple_host_close");
        const auto count = plugin.symbol<int(*)()>("r3dh_ple_host_registry_count");
        const auto at = plugin.symbol<const RadKernelInfo*(*)(int)>("r3dh_ple_host_registry_at");
        require(plugin.symbol<uint32_t(*)()>("rad_plugin_abi_version")() == RAD_ABI_VERSION, "plugin ABI");
        require(plugin.symbol<uint64_t(*)()>("rad_plugin_abi_layout")() == rad_abi_layout(), "plugin layout");
        const auto* info = plugin.symbol<const RadPluginInfo*(*)()>("rad_plugin_info")();
        require(info && !std::strcmp(info->build_target, "host"), "truthful HOST metadata");
        const auto plugin_at = plugin.symbol<const RadKernelInfo*(*)(int)>("rad_kernel_at");
        std::array<const RadKernelInfo*, 3> saved{at(0), at(1), at(2)};
        const int enumeration_fds = descriptors();
        require(count() == 3 && plugin.symbol<int(*)()>("rad_kernel_count")() == 1,
            "enumeration before open exposes only HOST gather publicly");
        require(plugin_at(0) == saved[1] && !plugin_at(-1) && !plugin_at(1),
            "stable public gather descriptor before open");
        for (int index = 0; index < 3; ++index)
            require(saved[index] && at(index) == saved[index] && saved[index]->launch
                && saved[index]->domain == RAD_DOMAIN_HOST, "preopen complete stable descriptors");
        require(saved[1]->init && saved[1]->fini && saved[1]->layout
            && saved[0]->opd_shape && saved[1]->opd_shape, "preopen forwarding hooks");
        require(!dlopen(argv[2], RTLD_NOW | RTLD_NOLOAD), "enumeration does not load provider");
        require(descriptors() == enumeration_fds, "enumeration opens no descriptors");
        require(saved[0]->launch(nullptr, nullptr) == RAD_E_STATE
            && saved[1]->launch(nullptr, nullptr) == RAD_E_STATE, "preopen launch refuses provider access");
        require(open("/nonexistent/r3dh_ple_host_provider.so") == RAD_E_NOTFOUND, "missing provider");
        require(open(argv[1]) == RAD_E_UNSUPPORTED, "self provider refused without recursive loading");
        for (int fixture = 1; fixture <= 7; ++fixture) {
            const std::string path = std::string(argv[4]) + "/r3dh_ple_host_refusal" + std::to_string(fixture) + ".so";
            const int before = descriptors();
            if (fixture == 5) {
                Library refused(path.c_str());
                const auto opened = refused.symbol<int(*)()>("r3dh_ple_host_fixture_opened");
                const auto closed = refused.symbol<int(*)()>("r3dh_ple_host_fixture_closed");
                require(opened() == 0 && closed() == 0, "rollback fixture initially closed");
                require(open(path.c_str()) == RAD_E_UNSUPPORTED, "opened provider validation refused");
                require(opened() == 1 && closed() == 1, "successful provider open rolled back exactly once");
            } else if (fixture == 6) {
                Library recursive(path.c_str());
                recursive.symbol<void(*)(int(*)(const char*))>("r3dh_ple_host_fixture_reenter")(open);
                require(open(path.c_str()) == RAD_E_STATE, "recursive initializer returns state error");
            } else require(open(path.c_str()) == RAD_E_UNSUPPORTED, "invalid provider refused");
            require(count() == 3 && at(0) == saved[0] && at(0)->launch(nullptr, nullptr) == RAD_E_STATE,
                "failed provider not published; descriptors remain stable");
            require(descriptors() == before, "failed provider descriptors closed");
        }
        const int before = descriptors();
        {
            const std::string path = std::string(argv[4]) + "/r3dh_ple_host_refusal8.so";
            Library fixture(path.c_str());
            const auto closed = fixture.symbol<int(*)()>("r3dh_ple_host_fixture_closed");
            const auto finalized = fixture.symbol<int(*)()>("r3dh_ple_host_fixture_finalized");
            require(open(path.c_str()) == RAD_OK, "instrumented provider open");
            void* instance = nullptr;
            require(saved[1]->init(nullptr, 0, 0, 1, &instance) == RAD_OK && instance,
                "stable init delegates actual instance");
            saved[1]->fini(instance);
            require(finalized() == 1, "stable fini delegates actual instance");
            RadLayout layout{};
            require(saved[1]->layout(nullptr, 0, 1, nullptr, nullptr, nullptr, 0, &layout) == RAD_OK
                && layout.bytes == 4321, "stable layout delegates actual output");
            require(saved[0]->opd_shape(nullptr, 0, 0, nullptr) == RAD_E_SHAPE
                && saved[1]->opd_shape(nullptr, 0, 0, nullptr) == RAD_E_SHAPE,
                "stable shape delegates nonnull actual hooks");
            std::atomic<int> gate{0};
            fixture.symbol<void(*)(std::atomic<int>*)>("r3dh_ple_host_fixture_gate")(&gate);
            auto launched = std::async(std::launch::async, [&] { return saved[0]->launch(nullptr, nullptr); });
            while (gate.load() != 1) std::this_thread::yield();
            std::promise<void> started;
            auto ready = started.get_future();
            auto closing = std::async(std::launch::async, [&] { started.set_value(); return close(); });
            ready.get();
            const bool waited = closing.wait_for(std::chrono::milliseconds(25)) == std::future_status::timeout;
            gate.store(2);
            const int launch_status = launched.get();
            const int close_status = closing.get();
            require(waited && launch_status == RAD_OK && close_status == RAD_OK && closed() == 1,
                "close waits for borrowed callback then closes provider exactly once");
            require(saved[0]->launch(nullptr, nullptr) == RAD_E_STATE, "postrace provider closed");
            std::puts("PLE_HOST_PROXY_HOOKS_PASS init=1 fini=1 layout=1 shape=2 concurrent_close=1");
        }
        require(open(argv[3]) == RAD_OK, "HOME resolves kernels/libavx.so");
        require(descriptors() == before, "provider verification descriptor closed");
        require(open(argv[2]) == RAD_OK && open(nullptr) == RAD_OK, "canonical and default reopening");
        require(plugin.symbol<int(*)()>("rad_plugin_open")() == RAD_OK, "main-style plugin open is nonrecursive");
        require(count() == 3 && plugin.symbol<int(*)()>("rad_kernel_count")() == 1, "three internal rows, one public HOST gather");
        const char* operations[]{"ngram_ids", "embed_lookup_q", "ngram_hint"};
        for (int index = 0; index < 3; ++index) {
            saved[index] = at(index);
            require(saved[index] && at(index) == saved[index] && saved[index]->domain == RAD_DOMAIN_HOST
                && !std::strcmp(saved[index]->op, operations[index]), "stable HOST metadata");
        }
        require(plugin_at(0) == saved[1] && !plugin_at(-1) && !plugin_at(1), "plugin gather-only kernel bounds");
        const auto schema_count = plugin.symbol<int(*)()>("rad_kernel_schema_count");
        const auto schema_at = plugin.symbol<const RadOpSchema*(*)(int)>("rad_kernel_schema_at");
        require(schema_count() == 1 && !schema_at(-1) && !schema_at(1), "engine gather-only HOST schema exports");
        require(schema_at(0) && !std::strcmp(schema_at(0)->op, "embed_lookup_q")
            && schema_at(0)->n_operands == 5, "canonical public gather schema");
        {
            Library provider(argv[2]);
            const auto source_count = provider.symbol<int(*)()>("rad_kernel_count");
            const auto source_at = provider.symbol<const RadKernelInfo*(*)(int)>("rad_kernel_at");
            int matched = 0;
            for (int index = 0; index < source_count(); ++index) {
                const auto* source = source_at(index);
                if (!source || !source->op) continue;
                for (int which = 0; which < 2; ++which) if (!std::strcmp(source->op, operations[which])) {
                    const auto* bound = saved[which];
                    require(bound->launch != source->launch && bound->launch,
                        "provider launch behind stable forwarding wrapper");
                    require(bound->priority == source->priority && bound->n_constraints == source->n_constraints
                        && bound->n_tunables == source->n_tunables && bound->capabilities == source->capabilities
                        && bound->relayout == source->relayout && bound->unrelayout == source->unrelayout
                        && bound->scratch == source->scratch, "provider static metadata preserved");
                    if (source->opd_shape) {
                        for (int operand = 0; operand < (which ? 5 : 14); ++operand) {
                            RadOpdDesc expected{}, actual{};
                            const int reference = source->opd_shape(nullptr, 0, operand, &expected);
                            require(bound->opd_shape(nullptr, 0, operand, &actual) == reference,
                                "shape wrapper delegates provider status");
                            require(!std::memcmp(&actual, &expected, sizeof(actual)),
                                "shape wrapper delegates provider output");
                        }
                    }
                    ++matched;
                }
            }
            require(matched == 2, "real provider rows matched");
        }
        require(open("/different/provider.so") == RAD_E_STATE, "cannot replace live provider");
        for (int index = 0; index < 3; ++index) require(at(index) == saved[index], "rows stable after external dlclose");
        std::array<uint16_t, 160> table{}, output{};
        for (int index = 0; index < 160; ++index) table[index] = uint16_t(0x3f00 + index);
        int32_t token = 0;
        RadTensor values[]{tensor(&token, RAD_I32, 1), tensor(table.data(), RAD_BF16, 1, 160), {},
            tensor(output.data(), RAD_BF16, 1, 160), {}};
        RadParam dtype{}; dtype.key = "dtype"; dtype.kind = RAD_P_STR; dtype.sval = "bf16";
        RadArgs args{}; args.t = values; args.n_t = 5; args.p = &dtype; args.n_p = 1; args.world_size = 1;
        require(saved[1]->launch(&args, nullptr) == RAD_OK, "real HOST launch after borrowed handle closes");
        for (int index = 0; index < 160; ++index) require(output[index] == table[index], "real BF16 lifetime oracle");
        const auto plugin_close = plugin.symbol<void(*)()>("rad_plugin_close");
        plugin_close();
        require(count() == 3 && at(0) == saved[0] && at(1) == saved[1], "close preserves static descriptors");
        require(saved[0]->launch(nullptr, nullptr) == RAD_E_STATE, "closed provider cannot launch");
        require(plugin.symbol<int(*)()>("rad_kernel_count")() == 1 && plugin_at(0) == saved[1],
            "closed plugin enumeration has no loading side effects");
        require(close() == RAD_OK, "close is idempotent");
        require(open(argv[2]) == RAD_OK && count() == 3, "explicit reopen after plugin close");
        require(close() == RAD_OK && count() == 3 && at(0) == saved[0], "explicit close preserves descriptors");
        require(plugin.symbol<int(*)()>("rad_plugin_open")() == RAD_OK, "default reopen after explicit close");
        plugin_close();
        require(count() == 3 && descriptors() == before, "final close leaves no provider descriptors");
        std::printf("PLE_HOST_LOADER_PASS checks=%llu refusal_fixtures=7 self_refusal=1 reentry_refusal=1 failed_fd_cleanup=1 internal_rows=3 stable_public_rows=1 lifetime_values=160 close_reopen=2 HOST=1 DEVICE=0 GPU_used=0\n", static_cast<unsigned long long>(checks));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "PLE_HOST_LOADER_FAIL %s checks=%llu\n", error.what(), static_cast<unsigned long long>(checks)); return 1;
    }
}
