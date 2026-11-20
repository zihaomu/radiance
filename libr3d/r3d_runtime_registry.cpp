#include "r3dh_allgather.h"
#include "r3dh_sample_merge.h"
#include "r3dh_attn_fused.h"
#include "r3d_flags.h"

namespace {
// A fused attention op set to 0 leaves the vocabulary, so the architecture declares the unfused chain.
bool fused_on(int index) { return r3d_exact_path(index ? "RAD_ATTN_FUSE_GATE" : "RAD_ATTN_FUSE_STORE"); }
int fused_count() { return int(fused_on(0)) + int(fused_on(1)); }
int fused_index(int index) {
    for (int fused = 0; fused < 2; ++fused)
        if (fused_on(fused) && index-- == 0) return fused;
    return -1;
}
}

extern "C" int r3d_runtime_schema_count() {
    return r3dh_allgather_schema_count() + r3dh_sample_merge_schema_count() + fused_count();
}

extern "C" const RadOpSchema* r3d_runtime_schema_at(int index) {
    if (index < 0) return nullptr;
    if (index < r3dh_allgather_schema_count()) return r3dh_allgather_schema_at(index);
    index -= r3dh_allgather_schema_count();
    if (index < r3dh_sample_merge_schema_count()) return r3dh_sample_merge_schema_at(index);
    return r3dh_attn_fused_schema_at(fused_index(index - r3dh_sample_merge_schema_count()));
}

extern "C" int r3d_runtime_kernel_count() {
    return r3dh_allgather_kernel_count() + r3dh_sample_merge_kernel_count() + fused_count();
}

extern "C" const RadKernelInfo* r3d_runtime_kernel_at(int index) {
    if (index < 0) return nullptr;
    if (index < r3dh_allgather_kernel_count()) return r3dh_allgather_kernel_at(index);
    index -= r3dh_allgather_kernel_count();
    if (index < r3dh_sample_merge_kernel_count()) return r3dh_sample_merge_kernel_at(index);
    return r3dh_attn_fused_kernel_at(fused_index(index - r3dh_sample_merge_kernel_count()));
}
