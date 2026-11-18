#include "r3dh_moe_mixed.h"
#include <array>

namespace {
#define EQ(key, value) {key, RAD_C_EQ, value, nullptr}
#define IN(key, value) {key, RAD_C_IN, 0, value}
#define TOKENS {"M", RAD_C_GE, 1, nullptr}, {"M", RAD_C_LE, 2048, nullptr}
#define COMMON TOKENS, EQ("top_k", 10), EQ("group", 128), EQ("a_group", 128)
#define IQR(name, experts, even, odd) const RadConstraint name[]{COMMON, \
    EQ("n_expert", experts), EQ("N", even), EQ("N_odd", odd), \
    EQ("K", 2560), EQ("K_odd", 2560), EQ("parts", 2), \
    IN("dtype", "i8a8h"), IN("a_order", "token")}
#define DOWN(name, experts, even, odd) const RadConstraint name[]{COMMON, \
    EQ("n_expert", experts), EQ("N", 2560), EQ("N_odd", 2560), \
    EQ("K", even), EQ("K_odd", odd), EQ("parts", 1), \
    IN("dtype", "w4nl64a8i8h"), IN("a_order", "sorted")}
IQR(iqr510a, 510, 768, 512);
IQR(iqr510b, 510, 512, 768);
IQR(iqr508a, 508, 768, 512);
IQR(iqr508b, 508, 512, 768);
DOWN(down510a, 510, 384, 256);
DOWN(down510b, 510, 256, 384);
DOWN(down508a, 508, 384, 256);
DOWN(down508b, 508, 256, 384);
const RadConstraint bf16gu[]{TOKENS, IN("n_expert", "2 4"), EQ("top_k", 10), EQ("N", 640),
    EQ("K", 2560), IN("dtype", "bf16"), IN("a_order", "token")};
const RadConstraint bf16dn[]{TOKENS, IN("n_expert", "2 4"), EQ("top_k", 10), EQ("N", 2560),
    EQ("K", 320), IN("dtype", "bf16"), IN("a_order", "sorted")};
template<size_t count>
RadKernelInfo row(const char* name, const RadConstraint (&constraints)[count], int kind) {
    RadKernelInfo result{};
    result.name = name;
    result.op = kind == 2 ? "moe_gemm" : "moe_gemm_q";
    result.family = "moe";
    result.computes = "Strict compact regular experts or resident protected BF16 experts";
    result.shape = "Qualified TP2 mixed-expert shapes";
    result.dtypes = kind == 2 ? "bf16" : kind == 1 ? "w4nl64a8i8h" : "i8a8h";
    result.domain = RAD_DOMAIN_DEVICE;
    result.priority = 1100;
    result.constraints = constraints;
    result.n_constraints = count;
    result.launch = kind == 2 ? r3dh_moe_mixed_bf16_launch
        : kind == 1 ? r3dh_moe_w4nl_launch : r3dh_moe_mixed_iqr_launch;
    result.opd_shape = kind == 2 ? r3dh_moe_mixed_bf16_shape
        : kind == 1 ? r3dh_moe_w4nl_shape : r3dh_moe_mixed_iqr_shape;
    if (kind != 2) {
        result.layout = kind == 1 ? r3dh_moe_w4nl_layout : r3dh_moe_mixed_iqr_layout;
        result.relayout = kind == 1 ? r3dh_moe_w4nl_transform : r3dh_moe_mixed_iqr_relayout;
        result.unrelayout = kind == 1 ? r3dh_moe_w4nl_inverse : r3dh_moe_mixed_iqr_unrelayout;
    }
    return result;
}
const std::array<RadKernelInfo, 10> rows{{
    row("mixed_iqr_510_768_512_strict", iqr510a, 0),
    row("mixed_iqr_510_512_768_strict", iqr510b, 0),
    row("mixed_iqr_508_768_512_strict", iqr508a, 0),
    row("mixed_iqr_508_512_768_strict", iqr508b, 0),
    row("mixed_w4nl_510_384_256_strict", down510a, 1),
    row("mixed_w4nl_510_256_384_strict", down510b, 1),
    row("mixed_w4nl_508_384_256_strict", down508a, 1),
    row("mixed_w4nl_508_256_384_strict", down508b, 1),
    row("mixed_bf16_gate_up_strict", bf16gu, 2),
    row("mixed_bf16_down_strict", bf16dn, 2)}};
}
extern "C" int r3dh_moe_mixed_kernel_count() { return rows.size(); }
extern "C" const RadKernelInfo* r3dh_moe_mixed_kernel_at(int index) {
    return index >= 0 && index < int(rows.size()) ? &rows[index] : nullptr;
}
