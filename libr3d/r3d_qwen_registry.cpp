#include "rad_plugin.h"
#include "r3dh_hc_api.h"
#include "r3dh_i8.h"

#include <cstring>

extern "C" int r3dh_rad_ar_init(const RadParam*, int, int, int, void**);
extern "C" void r3dh_rad_ar_fini(void*);
extern "C" int r3dh_rad_ar_oneshot_2rank_exact(const RadArgs*, RadStream);
extern "C" int r3d_runtime_schema_count();
extern "C" const RadOpSchema* r3d_runtime_schema_at(int);
extern "C" int r3d_runtime_kernel_count();
extern "C" const RadKernelInfo* r3d_runtime_kernel_at(int);
extern "C" int r3d_mtp_schema_count();
extern "C" const RadOpSchema* r3d_mtp_schema_at(int);
extern "C" int r3d_mtp_kernel_count();
extern "C" const RadKernelInfo* r3d_mtp_kernel_at(int);
extern "C" int r3d_sparse_schema_count();
extern "C" const RadOpSchema* r3d_sparse_schema_at(int);
extern "C" int r3d_sparse_kernel_count();
extern "C" const RadKernelInfo* r3d_sparse_kernel_at(int);
extern "C" const RadOpSchema* r3d_vision_schema_at(int);
extern "C" int r3d_vision_kernel_count();
extern "C" const RadKernelInfo* r3d_vision_kernel_at(int);

namespace {

#define COUNT(values) static_cast<int>(sizeof(values) / sizeof(values[0]))
#define RANGE(values) values, COUNT(values)
#define REQUIRED(key, type) {key, type, RAD_REQUIRED}
#define OPTIONAL(key, type) {key, type, RAD_OPTIONAL}
#define EQ(key, value) {key, RAD_C_EQ, value, nullptr}
#define GE(key, value) {key, RAD_C_GE, value, nullptr}
#define LE(key, value) {key, RAD_C_LE, value, nullptr}
#define DIV(key, value) {key, RAD_C_DIV, value, nullptr}
#define IN(key, value) {key, RAD_C_IN, 0, value}

const RadParamSpec hc_plain[]{REQUIRED("M", RAD_P_INT), REQUIRED("n", RAD_P_INT),
    REQUIRED("hc", RAD_P_INT), REQUIRED("dtype", RAD_P_STR)};
const RadOperandSpec hc_enter[]{{"x", RAD_OPD_IN, 0}, {"h", RAD_OPD_OUT, 0}};
const RadOperandSpec hc_write[]{{"y", RAD_OPD_IN, 0}, {"inj", RAD_OPD_IN, 0}, {"h", RAD_OPD_INOUT, 0}};
const RadParamSpec hc_read[]{REQUIRED("M", RAD_P_INT), REQUIRED("n", RAD_P_INT),
    REQUIRED("hc", RAD_P_INT), REQUIRED("lowrank", RAD_P_INT), REQUIRED("eps", RAD_P_F64),
    REQUIRED("dtype", RAD_P_STR), OPTIONAL("wadd", RAD_P_F64), OPTIONAL("inject", RAD_P_INT),
    OPTIONAL("group", RAD_P_INT), OPTIONAL("rotate", RAD_P_INT), OPTIONAL("mix", RAD_P_STR)};
const RadOperandSpec hc_read_operands[]{
    {"h", RAD_OPD_IN, 0}, {"w", RAD_OPD_WEIGHT, 0}, {"mix_down", RAD_OPD_WEIGHT, 0},
    {"mix_up", RAD_OPD_WEIGHT, 0}, {"inject_w", RAD_OPD_WEIGHT, 1}, {"x", RAD_OPD_OUT, 0},
    {"inj", RAD_OPD_OUT, 1}, {"q", RAD_OPD_OUT, 1}, {"scale", RAD_OPD_OUT, 1},
    {"rq", RAD_OPD_OUT, 1}, {"rscale", RAD_OPD_OUT, 1}};
const RadParamSpec all_reduce[]{REQUIRED("world_size", RAD_P_INT), REQUIRED("numel", RAD_P_INT),
    REQUIRED("dtype", RAD_P_STR), REQUIRED("exact", RAD_P_INT), OPTIONAL("min_bytes", RAD_P_INT),
    OPTIONAL("hops", RAD_P_INT), OPTIONAL("overlap", RAD_P_INT)};
const RadOperandSpec all_reduce_operands[]{{"x", RAD_OPD_INOUT, 0}, {"y", RAD_OPD_OUT, 1}};
const RadOpSchema schemas[]{
    {"hc_enter", RANGE(hc_plain), RANGE(hc_enter), "Copy into independent residual streams"},
    {"hc_write", RANGE(hc_plain), RANGE(hc_write), "Residual stream write"},
    {"hc_read", RANGE(hc_read), RANGE(hc_read_operands), "Strict gated residual mixer"},
    {"all_reduce", RANGE(all_reduce), RANGE(all_reduce_operands), "Exact tensor-parallel sum"}
};
const RadConstraint hc_plain_constraints[]{GE("M", 1), LE("M", 2048), EQ("n", 2560), EQ("hc", 4), IN("dtype", "bf16")};
const RadConstraint hc_read_constraints[]{GE("M", 1), LE("M", 2048), EQ("n", 2560), EQ("hc", 4),
    EQ("lowrank", 320), IN("dtype", "bf16"), IN("mix", "e4m3"), LE("group", 128), LE("inject", 1)};
const RadConstraint ar_constraints[]{EQ("world_size", 2), EQ("exact", 1), IN("dtype", "bf16 fp16 fp32"),
    {RAD_FACT_P2P, RAD_C_EQ, 1, nullptr}};
const RadConstraint i8_constraints[]{GE("M", 1), LE("M", 2048), DIV("N", 128), DIV("K", 128),
    EQ("group", 128), IN("dtype", "i8a8 i8a8h")};
const RadConstraint quant_constraints[]{GE("M", 1), LE("M", 2048), DIV("n", 128),
    EQ("group", 128), IN("dtype", "bf16")};

int ar_shape(const RadParam* params, int count, int operand, RadOpdDesc* output) {
    if (!output || operand < 0 || operand > 1) return RAD_E_INVAL;
    const int64_t elements = rad_param_geti(params, count, "numel", 0);
    const uint32_t dtype = rad_dtype_parse(rad_param_gets(params, count, "dtype", ""));
    if (elements < 1 || (dtype != RAD_BF16 && dtype != RAD_F16 && dtype != RAD_F32)) return RAD_E_SHAPE;
    *output = RadOpdDesc{};
    output->dtype = dtype; output->rank = 1; output->shape[0] = elements; output->idx_const = -1;
    return RAD_OK;
}

const RadKernelInfo kernels[]{
    {"qwen_hc_enter_strict", "hc_enter", "norm", "Residual entry", "model geometry", "bf16",
      RAD_DOMAIN_DEVICE, 0, RANGE(hc_plain_constraints), nullptr, 0, nullptr,
      nullptr, nullptr, r3dh_hc_rad_hc_enter, nullptr, nullptr, nullptr, r3dh_hc_shape_hc_enter},
    {"qwen_hc_write_strict", "hc_write", "norm", "Residual write", "model geometry", "bf16 RNE",
      RAD_DOMAIN_DEVICE, 0, RANGE(hc_plain_constraints), nullptr, 0, nullptr,
      nullptr, nullptr, r3dh_hc_rad_hc_write, nullptr, nullptr, nullptr, r3dh_hc_shape_hc_write},
    {"qwen_hc_read_strict", "hc_read", "norm", "Exact staged residual mixer", "model geometry", "e4m3 rows and bf16 state",
      RAD_DOMAIN_DEVICE, 0, RANGE(hc_read_constraints), nullptr, 0, nullptr,
      nullptr, nullptr, r3dh_hc_rad_hc_read_strict, r3dh_hc_rad_hc_read_strict_scratch,
      r3dh_hc_rad_layout_hc_e4m3, r3dh_hc_rad_relayout_hc_e4m3, r3dh_hc_shape_hc_read, r3dh_hc_rad_unrelayout_hc_e4m3},
    {"qwen_ar_tp2_exact", "all_reduce", "ar", "Native peer-pull sum", "two gfx1100 ranks", "exact bf16/fp16/fp32",
      RAD_DOMAIN_DEVICE, 0, RANGE(ar_constraints), nullptr, 0, nullptr,
      r3dh_rad_ar_init, r3dh_rad_ar_fini, r3dh_rad_ar_oneshot_2rank_exact, nullptr, nullptr, nullptr, ar_shape},
    {"qwen_i8_rowgroup", "gemm_nt_q", "gemm", "Integer group product", "whole 128 groups", "bf16 group weight scale",
      RAD_DOMAIN_DEVICE, 0, RANGE(i8_constraints), nullptr, 0, nullptr,
      nullptr, nullptr, r3dh_i8_rad_gemm_nt_q, nullptr, r3dh_i8_rad_layout, r3dh_i8_rad_relayout,
      r3dh_i8_shape_gemm_nt_q, r3dh_i8_rad_unrelayout},
    {"qwen_quant_i8_rowgroup", "quant_act_i8g", "quant", "Row-major int8 groups", "whole 128 groups", "RNE",
      RAD_DOMAIN_DEVICE, 0, RANGE(quant_constraints), nullptr, 0, nullptr,
      nullptr, nullptr, r3dh_i8_rad_quant_act_i8g, nullptr, nullptr, nullptr, r3dh_i8_shape_quant_act_i8g},
    {"qwen_had_quant_i8_rowgroup", "had_quant_act_i8g", "quant", "FWHT128 int8 groups", "whole 128 groups", "RNE",
      RAD_DOMAIN_DEVICE, 0, RANGE(quant_constraints), nullptr, 0, nullptr,
      nullptr, nullptr, r3dh_i8_rad_had_quant_act_i8g, nullptr, nullptr, nullptr, r3dh_i8_shape_quant_act_i8g}
};
}

extern "C" int r3d_qwen_schema_count() { return COUNT(schemas) + r3d_mtp_schema_count() + r3d_sparse_schema_count() + r3d_runtime_schema_count() + 1; }
extern "C" const RadOpSchema* r3d_qwen_schema_at(int index) {
    if (index < 0) return nullptr;
    if (index < COUNT(schemas)) return &schemas[index];
    index -= COUNT(schemas);
    if (index < r3d_mtp_schema_count()) return r3d_mtp_schema_at(index);
    index -= r3d_mtp_schema_count();
    if (index < r3d_sparse_schema_count()) return r3d_sparse_schema_at(index);
    index -= r3d_sparse_schema_count();
    if (index < r3d_runtime_schema_count()) return r3d_runtime_schema_at(index);
    return index == r3d_runtime_schema_count() ? r3d_vision_schema_at(1) : nullptr;
}
extern "C" int r3d_qwen_kernel_count() { return COUNT(kernels) + r3d_mtp_kernel_count() + r3d_sparse_kernel_count() + r3d_runtime_kernel_count() + r3d_vision_kernel_count(); }
extern "C" const RadKernelInfo* r3d_qwen_kernel_at(int index) {
    if (index < 0) return nullptr;
    if (index < COUNT(kernels)) return &kernels[index];
    index -= COUNT(kernels);
    if (index < r3d_mtp_kernel_count()) return r3d_mtp_kernel_at(index);
    index -= r3d_mtp_kernel_count();
    if (index < r3d_sparse_kernel_count()) return r3d_sparse_kernel_at(index);
    index -= r3d_sparse_kernel_count();
    if (index < r3d_runtime_kernel_count()) return r3d_runtime_kernel_at(index);
    return r3d_vision_kernel_at(index - r3d_runtime_kernel_count());
}
