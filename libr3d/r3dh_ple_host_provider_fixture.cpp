#include "rad_abi_layout.h"
#include <array>
#include <atomic>
#include <thread>
#ifndef R3DH_PLE_HOST_FIXTURE
#error R3DH_PLE_HOST_FIXTURE is required for refusal-only tests
#endif
namespace {
int (*reenter)(const char*) = nullptr;
int opened = 0;
int closed = 0;
const RadPluginInfo info{RAD_PLUGIN_KERNEL, R3DH_PLE_HOST_FIXTURE == 3 ? "not_libavx" : "libavx", "test", "refusal-only fixture; no numerical kernels", "host"};
}
extern "C" uint32_t rad_plugin_abi_version() { return RAD_ABI_VERSION + (R3DH_PLE_HOST_FIXTURE == 1); }
extern "C" uint64_t rad_plugin_abi_layout() { return rad_abi_layout() ^ uint64_t(R3DH_PLE_HOST_FIXTURE == 2); }
extern "C" const RadPluginInfo* rad_plugin_info() { return &info; }
extern "C" int rad_kernel_count() {
    return R3DH_PLE_HOST_FIXTURE == 4 ? 65537 : R3DH_PLE_HOST_FIXTURE == 8 ? 2 : 1;
}
#if R3DH_PLE_HOST_FIXTURE == 8
namespace {
std::atomic<int>* gate = nullptr;
int finalized = 0;
int state = 0;
int launch(const RadArgs*, RadStream) {
    if (gate) {
        gate->store(1);
        while (gate->load() != 2) std::this_thread::yield();
    }
    return RAD_OK;
}
int init(const RadParam*, int, int, int, void** output) { *output = &state; return RAD_OK; }
void fini(void* instance) { if (instance == &state) ++finalized; }
int layout(const RadParam*, int, int, const RadEncoding*, const int*, const RadTensor*, int, RadLayout* output) {
    output->bytes = 4321;
    return RAD_OK;
}
int shape(const RadParam*, int, int, RadOpdDesc*) { return RAD_E_SHAPE; }
const auto rows = [] {
    std::array<RadKernelInfo, 2> result{};
    result[0].name = "avx_ngram_ids";
    result[0].op = "ngram_ids";
    result[1].name = "avx_embed_lookup_q";
    result[1].op = "embed_lookup_q";
    for (auto& row : result) {
        row.domain = RAD_DOMAIN_HOST;
        row.launch = launch;
        row.opd_shape = shape;
    }
    result[1].init = init;
    result[1].fini = fini;
    result[1].layout = layout;
    return result;
}();
}
extern "C" const RadKernelInfo* rad_kernel_at(int index) {
    return index >= 0 && index < 2 ? rows.data() + index : nullptr;
}
extern "C" void r3dh_ple_host_fixture_gate(std::atomic<int>* value) { gate = value; }
extern "C" int r3dh_ple_host_fixture_finalized() { return finalized; }
#else
#if R3DH_PLE_HOST_FIXTURE != 7
extern "C" const RadKernelInfo* rad_kernel_at(int) { return nullptr; }
#endif
#endif
extern "C" void r3dh_ple_host_fixture_reenter(int (*callback)(const char*)) { reenter = callback; }
extern "C" int rad_plugin_open() {
    if (R3DH_PLE_HOST_FIXTURE == 6) return reenter ? reenter(nullptr) : RAD_E_INVAL;
    ++opened;
    return RAD_OK;
}
extern "C" void rad_plugin_close() { ++closed; }
extern "C" int r3dh_ple_host_fixture_opened() { return opened; }
extern "C" int r3dh_ple_host_fixture_closed() { return closed; }
