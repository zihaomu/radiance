#include "r3dh_moe_mixed.h"
#include <cstdio>
#include <cstdlib>
int main() {
    RadParam params[] = {RAD_INT("M", 16), RAD_INT("N", 768), RAD_INT("N_odd", 512),
        RAD_INT("K", 2560), RAD_INT("K_odd", 2560), RAD_INT("n_expert", 510),
        RAD_INT("top_k", 10), RAD_INT("group", 128), RAD_INT("a_group", 128),
        RAD_INT("parts", 2), RAD_STR("a_order", "token"), RAD_STR("dtype", "i8a8h")};
    const int count = sizeof(params) / sizeof(*params);
    r3dh_moe_iqr::Config geometry{};
    if (r3dh_moe_iqr::config(params, count, &geometry) != RAD_OK || geometry.experts != 510) return 1;
    if (r3dh_moe_mixed::config(params, count, &geometry) != RAD_OK
        || geometry.experts != 510) return 2;
    RadOpdDesc shape{};
    if (r3dh_moe_mixed_iqr_shape(params, count, 5, &shape) != RAD_OK
        || shape.shape[0] != 511) return 3;
    for (int experts : {508, 510}) {
        params[5] = RadParam RAD_INT("n_expert", experts);
        if (r3dh_moe_mixed::config(params, count, &geometry) != RAD_OK
            || geometry.experts != experts) return 5;
        if (r3dh_moe_mixed_iqr_shape(params, count, 5, &shape) != RAD_OK
            || shape.shape[0] != experts + 1) return 6;
    }
    params[5] = RadParam RAD_INT("n_expert", 512);
    if (r3dh_moe_mixed::config(params, count, &geometry) != RAD_E_SHAPE) return 4;
    if (r3dh_moe_mixed_kernel_count() != 10 || r3dh_moe_mixed_kernel_at(-1)
        || r3dh_moe_mixed_kernel_at(10)) return 7;
    for (int index = 0; index < 10; ++index) {
        const auto* kernel = r3dh_moe_mixed_kernel_at(index);
        if (!kernel || !kernel->launch || !kernel->opd_shape) return 8;
        if (index < 8 && (!kernel->layout || !kernel->relayout || !kernel->unrelayout)) return 14;
        const bool protected_row = index >= 8, down = index >= 4 && index < 8 || index == 9;
        const bool reverse = index % 2;
        RadParam candidate[]{RAD_INT("M", 3), RAD_INT("N", protected_row ? (down ? 2560 : 640) : (down ? 2560 : (reverse ? 512 : 768))),
            RAD_INT("N_odd", down ? 2560 : (reverse ? 768 : 512)),
            RAD_INT("K", protected_row ? (down ? 320 : 2560) : (down ? (reverse ? 256 : 384) : 2560)),
            RAD_INT("K_odd", down ? (reverse ? 384 : 256) : 2560),
            RAD_INT("n_expert", protected_row ? 2 : (index % 4 < 2 ? 510 : 508)),
            RAD_INT("top_k", 10), RAD_INT("group", 128), RAD_INT("a_group", 128), RAD_INT("parts", down ? 1 : 2),
            RAD_STR("dtype", protected_row ? "bf16" : down ? "w4nl64a8i8h" : "i8a8h"), RAD_STR("a_order", down ? "sorted" : "token")};
        for (int operand = 0; operand < (protected_row ? 5 : 9); ++operand)
            if (kernel->opd_shape(candidate, 12, operand, &shape) != RAD_OK) return 15;
        if (kernel->opd_shape(candidate, 12, protected_row ? 3 : 5, &shape) != RAD_OK
            || shape.shape[0] != candidate[5].ival + 1) return 16;
        candidate[9].ival = down ? 2 : 1;
        if (kernel->opd_shape(candidate, 12, 0, &shape) != RAD_E_SHAPE) return 17;
    }
    for (bool down : {false, true}) for (int experts : {2, 4}) {
        RadParam protected_params[]{RAD_INT("M", 3), RAD_INT("N", down ? 2560 : 640),
            RAD_INT("K", down ? 320 : 2560), RAD_INT("n_expert", experts), RAD_INT("top_k", 10),
            RAD_STR("dtype", "bf16"), RAD_STR("a_order", down ? "sorted" : "token"), RAD_INT("parts", down ? 1 : 2)};
        for (int operand = 0; operand < 5; ++operand)
            if (r3dh_moe_mixed_bf16_shape(protected_params, 8, operand, &shape) != RAD_OK) return 9;
        if (r3dh_moe_mixed_bf16_shape(protected_params, 8, 3, &shape) != RAD_OK || shape.shape[0] != experts + 1) return 10;
        protected_params[7].ival = down ? 2 : 1;
        if (r3dh_moe_mixed_bf16_shape(protected_params, 8, 0, &shape) != RAD_E_SHAPE) return 11;
        protected_params[7].ival = down ? 1 : 2;
        protected_params[3].ival = 3;
        if (r3dh_moe_mixed_bf16_shape(protected_params, 8, 0, &shape) != RAD_E_SHAPE) return 12;
    }
    for (int experts : {0, 3, 507, 509, 511, 513}) {
        params[5].ival = experts;
        if (r3dh_moe_mixed_iqr_shape(params, count, 0, &shape) != RAD_E_SHAPE) return 13;
    }
    std::puts("MIXED_SHAPE_PASS native_compact=1 experts=508,510 tables=254,255 offsets=509,511 kernels=10 parts=2,1 protected=2,4 rejects=14");
}
