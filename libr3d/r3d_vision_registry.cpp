#include "r3dh_vision_strict.h"
#include "r3dh_vision_rope.h"

extern "C" int r3d_vision_schema_count() {
    int count = 0;
    r3dh_vision_strict_schemas(&count);
    return count + 1;
}
extern "C" const RadOpSchema* r3d_vision_schema_at(int index) {
    int count = 0;
    const auto* schemas = r3dh_vision_strict_schemas(&count);
    if (index >= 0 && index < count) return schemas + index;
    return index == count ? r3dh_vision_rope_schema() : nullptr;
}
extern "C" int r3d_vision_kernel_count() {
    int count = 0;
    r3dh_vision_strict_kernels(&count);
    return count + 1;
}
extern "C" const RadKernelInfo* r3d_vision_kernel_at(int index) {
    int count = 0;
    const auto* rows = r3dh_vision_strict_kernels(&count);
    if (index >= 0 && index < count) return rows + index;
    return index == count ? r3dh_vision_rope_kernel() : nullptr;
}
