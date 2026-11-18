#include "r3dh_ple_host.h"
#include "rad_abi_layout.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <mutex>
#include <string>
#include <sys/stat.h>
#include <type_traits>
#include <unistd.h>

namespace {
struct Provider {
    std::mutex mutex;
    void* library = nullptr;
    RadPluginCloseFn close = nullptr;
    R3dhPleHost host{};
    std::string path;
    struct stat identity{};
    bool loading = false;
    ~Provider() {
        if (library && close) close();
        if (library) dlclose(library);
    }
};
Provider& provider() { static Provider result; return result; }
template<int Index, auto Member, class Function> struct Forward;
template<int Index, auto Member, class Result, class... Arguments>
struct Forward<Index, Member, Result(*)(Arguments...)> {
    static Result call(Arguments... arguments) {
        auto& instance = provider();
        std::lock_guard<std::mutex> guard(instance.mutex);
        const auto function = instance.library ? instance.host.rows[Index].*Member : nullptr;
        if constexpr (std::is_void_v<Result>) {
            if (function) function(arguments...);
        } else {
            if (function) return function(arguments...);
            return instance.library ? RAD_E_UNSUPPORTED : RAD_E_STATE;
        }
    }
};
template<int Index> const RadKernelInfo* proxy() {
    static constexpr RadKernelInfo result = [] {
        RadKernelInfo row{};
        row.name = Index == 0 ? "r3dh_ple_host_ngram_ids" : "r3dh_ple_host_embed_lookup_q";
        row.op = Index == 0 ? "ngram_ids" : "embed_lookup_q";
        row.family = "vocab";
        row.computes = Index == 0 ? "hashed n-gram row ids for PLE"
            : "embedding gather, one scale for the table: an E4M3 table into bf16 through a shared row cache "
              "and direct reads of the container, a later call's rows read ahead";
        row.shape = "any: every extent is read off the operands, strides included";
        row.dtypes = "any dtype rad_types.h names, per operand";
        row.domain = RAD_DOMAIN_HOST;
        row.launch = Forward<Index, &RadKernelInfo::launch, RadLaunchFn>::call;
        row.opd_shape = Forward<Index, &RadKernelInfo::opd_shape, RadShapeFn>::call;
        if constexpr (Index == 1) {
            row.init = Forward<Index, &RadKernelInfo::init, RadInitFn>::call;
            row.fini = Forward<Index, &RadKernelInfo::fini, RadFiniFn>::call;
            row.layout = Forward<Index, &RadKernelInfo::layout, RadLayoutFn>::call;
        }
        return row;
    }();
    return &result;
}
struct Library {
    void* handle = nullptr;
    RadPluginCloseFn close = nullptr;
    void reset() {
        if (handle && close) close();
        if (handle) dlclose(handle);
        handle = nullptr;
        close = nullptr;
    }
    ~Library() { reset(); }
};
struct Descriptor {
    int value = -1;
    ~Descriptor() { if (value >= 0) close(value); }
};
bool same_file(const struct stat& first, const struct stat& second) {
    return first.st_dev == second.st_dev && first.st_ino == second.st_ino
        && first.st_size == second.st_size
        && first.st_mtim.tv_sec == second.st_mtim.tv_sec
        && first.st_mtim.tv_nsec == second.st_mtim.tv_nsec
        && first.st_ctim.tv_sec == second.st_ctim.tv_sec
        && first.st_ctim.tv_nsec == second.st_ctim.tv_nsec;
}
template<class Function> Function symbol(void* library, const char* name, void* owner) {
    dlerror();
    void* address = dlsym(library, name);
    const char* error = dlerror();
    Dl_info location{};
    if (error || !address || !dladdr(address, &location) || (owner && location.dli_fbase != owner)) {
        std::fprintf(stderr, "PLE HOST provider symbol %s: %s\n", name,
            error ? error : "missing or supplied by another library");
        return nullptr;
    }
    return reinterpret_cast<Function>(address);
}
int load(const std::string& path, R3dhPleHost* output, Library& library) {
    library.handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL | RTLD_DEEPBIND);
    if (!library.handle) {
        std::fprintf(stderr, "PLE HOST provider %s: %s\n", path.c_str(), dlerror());
        return RAD_E_NOTFOUND;
    }
    const auto version = symbol<uint32_t(*)()>(library.handle, "rad_plugin_abi_version", nullptr);
    Dl_info location{};
    if (!version || !dladdr(reinterpret_cast<void*>(version), &location)) return RAD_E_UNSUPPORTED;
    struct stat requested{}, exported{};
    if (!location.dli_fname || stat(path.c_str(), &requested)
        || stat(location.dli_fname, &exported) || !same_file(requested, exported)) {
        std::fprintf(stderr, "PLE HOST provider exports belong to a dependency, not %s\n", path.c_str());
        return RAD_E_UNSUPPORTED;
    }
    void* owner = location.dli_fbase;
    const auto layout = symbol<uint64_t(*)()>(library.handle, "rad_plugin_abi_layout", owner);
    const auto info = symbol<const RadPluginInfo*(*)()>(library.handle, "rad_plugin_info", owner);
    const auto count = symbol<int(*)()>(library.handle, "rad_kernel_count", owner);
    const auto at = symbol<const RadKernelInfo*(*)(int)>(library.handle, "rad_kernel_at", owner);
    if (!layout || !info || !count || !at || version() != RAD_ABI_VERSION
        || layout() != rad_abi_layout()) return RAD_E_UNSUPPORTED;
    const auto* identity = info();
    if (!identity || identity->kind != RAD_PLUGIN_KERNEL || !identity->name
        || std::strcmp(identity->name, "libavx")) return RAD_E_UNSUPPORTED;
    dlerror();
    const auto initialize = reinterpret_cast<int(*)()>(dlsym(library.handle, "rad_plugin_open"));
    if (initialize) {
        Dl_info entry{};
        if (!dladdr(reinterpret_cast<void*>(initialize), &entry) || entry.dli_fbase != owner)
            return RAD_E_UNSUPPORTED;
        const int initialized = initialize();
        if (initialized != RAD_OK) return initialized;
        library.close = symbol<RadPluginCloseFn>(library.handle, "rad_plugin_close", owner);
    }
    const int total = count();
    if (total < 1 || total > 65536) return RAD_E_UNSUPPORTED;
    const RadKernelInfo *ids = nullptr, *gather = nullptr;
    for (int index = 0; index < total; ++index) {
        const auto* row = at(index);
        if (!row || !row->name || !row->op || row->domain != RAD_DOMAIN_HOST) continue;
        if (!std::strcmp(row->name, "avx_ngram_ids") && !std::strcmp(row->op, "ngram_ids")) {
            if (ids) return RAD_E_UNSUPPORTED;
            ids = row;
        }
        if (!std::strcmp(row->name, "avx_embed_lookup_q") && !std::strcmp(row->op, "embed_lookup_q")) {
            if (gather) return RAD_E_UNSUPPORTED;
            gather = row;
        }
    }
    for (const auto* row : {ids, gather})
        if (!row || row->priority || row->n_constraints || row->n_tunables || row->tune_valid
            || row->scratch || row->relayout || row->unrelayout || row->replaces || row->capabilities)
            return RAD_E_UNSUPPORTED;
    if (ids->init || ids->fini) return RAD_E_UNSUPPORTED;
    return r3dh_ple_host_bind(output, int(version()), ids, gather);
}
}
extern "C" int r3dh_ple_host_open(const char* requested) {
    auto& instance = provider();
    {
        std::lock_guard<std::mutex> guard(instance.mutex);
        if (instance.loading) return RAD_E_STATE;
        if ((!requested || !requested[0]) && instance.library) {
            struct stat current{};
            return stat(instance.path.c_str(), &current) == 0
                && same_file(current, instance.identity) ? RAD_OK : RAD_E_STATE;
        }
    }
    std::string path;
    if (requested && requested[0]) path = requested;
    else {
        Dl_info location{};
        if (!dladdr(reinterpret_cast<void*>(r3dh_ple_host_bind), &location) || !location.dli_fname)
            return RAD_E_NOTFOUND;
        path = location.dli_fname;
        const auto slash = path.find_last_of('/');
        path = (slash == std::string::npos ? std::string("./") : path.substr(0, slash + 1)) + "libavx.so";
    }
    struct stat before{};
    if (stat(path.c_str(), &before) == 0 && S_ISDIR(before.st_mode)) path += "/kernels/libavx.so";
    char* resolved = realpath(path.c_str(), nullptr);
    {
        std::lock_guard<std::mutex> guard(instance.mutex);
        if (instance.loading) { std::free(resolved); return RAD_E_STATE; }
        if (instance.library) {
            const bool same = resolved && instance.path == resolved
                && stat(resolved, &before) == 0 && same_file(before, instance.identity);
            std::free(resolved);
            return same ? RAD_OK : RAD_E_STATE;
        }
        if (!resolved) return RAD_E_NOTFOUND;
        path = resolved;
        std::free(resolved);
        if (stat(path.c_str(), &before) || !S_ISREG(before.st_mode)) return RAD_E_NOTFOUND;
        instance.loading = true;
    }
    Descriptor descriptor;
    descriptor.value = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    struct stat pinned{};
    int status = descriptor.value >= 0 && !fstat(descriptor.value, &pinned)
        && same_file(before, pinned) ? RAD_OK : RAD_E_STATE;
    Library library;
    R3dhPleHost bound{};
    try { if (status == RAD_OK) status = load(path, &bound, library); }
    catch (...) { status = RAD_E_IO; }
    struct stat after{};
    if (status == RAD_OK && (fstat(descriptor.value, &after) || !same_file(before, after)
        || stat(path.c_str(), &after) || !same_file(before, after))) status = RAD_E_STATE;
    if (status != RAD_OK) std::fprintf(stderr, "PLE HOST provider %s refused status=%d\n", path.c_str(), status);
    if (status != RAD_OK) library.reset();
    std::lock_guard<std::mutex> guard(instance.mutex);
    instance.loading = false;
    if (status == RAD_OK) {
        instance.path = std::move(path);
        instance.identity = before;
        instance.host = bound;
        instance.library = library.handle;
        instance.close = library.close;
        library.handle = nullptr;
    }
    return status;
}
extern "C" int r3dh_ple_host_close() {
    auto& instance = provider();
    void* library = nullptr;
    RadPluginCloseFn close = nullptr;
    {
        std::lock_guard<std::mutex> guard(instance.mutex);
        if (instance.loading) return RAD_E_STATE;
        if (!instance.library) return RAD_OK;
        instance.loading = true;
        library = instance.library;
        close = instance.close;
        instance.close = nullptr;
        instance.library = nullptr;
        instance.host = {};
        instance.path.clear();
        instance.identity = {};
    }
    if (close) close();
    const int status = dlclose(library) == 0 ? RAD_OK : RAD_E_IO;
    std::lock_guard<std::mutex> guard(instance.mutex);
    instance.loading = false;
    return status;
}

extern "C" int r3dh_ple_host_registry_count() {
    return 3;
}
extern "C" const RadKernelInfo* r3dh_ple_host_registry_at(int index) {
    if (index == 0) return proxy<0>();
    if (index == 1) return proxy<1>();
    return index == 2 ? r3dh_ple_host_hint_kernel() : nullptr;
}
