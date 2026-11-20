#pragma once
#include <cstdlib>
#include <cstring>

// RDNA3 fast paths whose output is bit-identical to the strict kernels are on by default;
// setting the variable to 0 selects the strict kernel. Paths that reorder floating-point
// work (RAD_LOGITS_WAVE, RAD_HC_DOWN_SPLIT, RAD_HC_UP_SPLIT, RAD_HC_NORM_INJECT_FUSED, RAD_HC_WMMA,
// RAD_HC_MATRIX_DOWN, RAD_ATTN_SPLIT) and exact paths without a measured serving gain stay
// opt-in and are read where they are used.
inline constexpr const char* r3d_exact_flags[]{
    "RAD_IQR_FUSED_WMMA", "RAD_IQR_FUSED_DECODE", "RAD_IQR_U5_WAVE", "RAD_IQR_U5_QG",
    "RAD_IQR_DOT4", "RAD_IQR_DOT4_WINDOW", "RAD_IQR_SPLIT", "RAD_W4NL_DOT4", "RAD_W4NL_RUN", "RAD_W4NL_VEC",
    "RAD_I8_SPLIT", "RAD_LOGITS_STAGED", "RAD_LOGITS_STREAM", "RAD_ROUTER_WAVE", "RAD_PLE_GATE_PARALLEL",
    "RAD_GDN_PAIRED", "RAD_GDN_QUANT_PARALLEL", "RAD_HC_NORM_PARALLEL", "RAD_HC_INJECT_PARALLEL",
    "RAD_HC_DOWN_PACKED", "RAD_HC_DOWN_CHAIN", "RAD_HC_UPMIX_EXACT", "RAD_QK_PREPARE_PARALLEL",
    "RAD_KV_STORE_PARALLEL", "RAD_ATTN_PAGED_FAST", "RAD_ATTN_FUSE_STORE", "RAD_ATTN_FUSE_GATE",
    "RAD_QSA_KEY_PARALLEL", "RAD_QSA_SCORE_PARALLEL", "RAD_QSA_SELECT_PARALLEL",
    "RAD_QSA_WORK_PARALLEL"};

inline bool r3d_exact_path(const char* name) {
    const char* value = std::getenv(name);
    return !value || std::strcmp(value, "0") != 0;
}
