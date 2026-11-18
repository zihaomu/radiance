#include "r3dh_allgather.h"
#include "r3dh_sample_merge.h"
#include "r3dh_attn_fused.h"

extern "C" int r3d_runtime_schema_count() {
    return r3dh_allgather_schema_count() + r3dh_sample_merge_schema_count()
         + r3dh_attn_fused_schema_count();
}

extern "C" const RadOpSchema* r3d_runtime_schema_at(int index) {
    if (index < 0) return nullptr;
    if (index < r3dh_allgather_schema_count()) return r3dh_allgather_schema_at(index);
    index -= r3dh_allgather_schema_count();
    if (index < r3dh_sample_merge_schema_count()) return r3dh_sample_merge_schema_at(index);
    return r3dh_attn_fused_schema_at(index - r3dh_sample_merge_schema_count());
}

extern "C" int r3d_runtime_kernel_count() {
    return r3dh_allgather_kernel_count() + r3dh_sample_merge_kernel_count()
         + r3dh_attn_fused_kernel_count();
}

extern "C" const RadKernelInfo* r3d_runtime_kernel_at(int index) {
    if (index < 0) return nullptr;
    if (index < r3dh_allgather_kernel_count()) return r3dh_allgather_kernel_at(index);
    index -= r3dh_allgather_kernel_count();
    if (index < r3dh_sample_merge_kernel_count()) return r3dh_sample_merge_kernel_at(index);
    return r3dh_attn_fused_kernel_at(index - r3dh_sample_merge_kernel_count());
}
