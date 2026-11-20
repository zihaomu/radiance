#include "r3dh_ple_host.h"
#include "rad_abi_layout.h"

namespace {
const RadPluginInfo info{RAD_PLUGIN_KERNEL, "r3dh_ple_host", "0.1",
    "HOST PLE mapped stored-table gather with ahead", "host"};
}

extern "C" uint32_t rad_plugin_abi_version() { return RAD_ABI_VERSION; }
RAD_PLUGIN_ABI_LAYOUT_EXPORT
extern "C" const RadPluginInfo* rad_plugin_info() { return &info; }
extern "C" int rad_plugin_open() { return r3dh_ple_host_open(nullptr); }
extern "C" void rad_plugin_close() { r3dh_ple_host_close(); }
extern "C" int rad_kernel_schema_count() { return 1; }
extern "C" const RadOpSchema* rad_kernel_schema_at(int index) {
    return index == 0 ? r3dh_ple_host_schema_at(1) : nullptr;
}
extern "C" int rad_kernel_count() {
    return 1;
}
extern "C" const RadKernelInfo* rad_kernel_at(int index) {
    return index == 0 ? r3dh_ple_host_registry_at(1) : nullptr;
}
