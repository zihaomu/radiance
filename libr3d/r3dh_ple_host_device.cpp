#include "r3dh_ple_host.h"
#include "r3dh_mtp_ple.h"
#include "r3dh_i8.h"

#include <vector>

namespace {
#define EQ(key, value) {key, RAD_C_EQ, value, nullptr}
#define GE(key, value) {key, RAD_C_GE, value, nullptr}
#define LE(key, value) {key, RAD_C_LE, value, nullptr}
#define IN(key, value) {key, RAD_C_IN, 0, value}
#define RANGE(values) values, int(sizeof(values) / sizeof(values[0]))
const RadConstraint gate_limits[]{GE("M", 1), LE("M", 32), EQ("n", 2560), EQ("hc", 4), IN("dtype", "bf16")};
const RadConstraint ids_limits[]{GE("M", 1), LE("M", 32), EQ("heads", 16), EQ("ngram", 3)};
const RadConstraint gemm_limits[]{GE("M", 1), LE("M", 32), IN("N", "2560,10240"), EQ("K", 2560),
    EQ("group", 128), IN("dtype", "i8a8,i8a8h")};
const RadConstraint quant_limits[]{GE("M", 1), LE("M", 32), EQ("n", 2560), EQ("group", 128), IN("dtype", "bf16")};

int resolved(const RadParam* params, int count, std::vector<RadParam>& result) {
    if (count < 0 || (count && !params)) return RAD_E_INVAL;
    result.clear();
    if (count) result.assign(params, params + count);
    for (auto& param : result) {
        if (param.kind != RAD_P_RANGE) continue;
        if (param.ival < 1 || param.ihi < param.ival) return RAD_E_SHAPE;
        param.kind = RAD_P_INT;
        param.ival = param.ihi;
    }
    return RAD_OK;
}
int gate_shape(const RadParam* params, int count, int operand, RadOpdDesc* output) {
    std::vector<RadParam> result;
    const int status = resolved(params, count, result);
    return status == RAD_OK ? r3dh_mtp_ple_shape("ple_gate", result.data(), count, operand, output) : status;
}
int gemm_shape(const RadParam* params, int count, int operand, RadOpdDesc* output) {
    std::vector<RadParam> result;
    const int status = resolved(params, count, result);
    return status == RAD_OK ? r3dh_i8_shape_gemm_nt_q(result.data(), count, operand, output) : status;
}
int quant_shape(const RadParam* params, int count, int operand, RadOpdDesc* output) {
    std::vector<RadParam> result;
    const int status = resolved(params, count, result);
    return status == RAD_OK ? r3dh_i8_shape_quant_act_i8g(result.data(), count, operand, output) : status;
}
RadKernelInfo row(const char* name, const char* operation, RadLaunchFn launch,
    const RadConstraint* limits, int limit_count, RadShapeFn shape) {
    RadKernelInfo result{};
    result.name = name;
    result.op = operation;
    result.family = "ple";
    result.computes = "Native gfx1100 PLE with explicit HOST gather boundary";
    result.shape = "M=1..32; n=2560; hc=4; embedding=16*160";
    result.dtypes = "bf16 activations, F32 gains; affine i8 group128 projections";
    result.domain = RAD_DOMAIN_DEVICE;
    result.priority = 100;
    result.launch = launch;
    result.constraints = limits;
    result.n_constraints = limit_count;
    result.opd_shape = shape;
    return result;
}
const RadKernelInfo* rows() {
    static const RadKernelInfo result[]{
        row("r3dh_ple_ngram_ids", "ngram_ids", r3dh_mtp_ple_rad_ngram_ids, RANGE(ids_limits), nullptr),
        [] {
            auto item = row("r3dh_ple_gate", "ple_gate", r3dh_mtp_ple_ple_gate, RANGE(gate_limits), gate_shape);
            item.layout = r3dh_ple_host_gain_layout;
            item.relayout = r3dh_ple_host_gain_relayout;
            item.unrelayout = r3dh_ple_host_gain_unrelayout;
            return item;
        }(),
        [] {
            auto item = row("r3dh_ple_gemm_i8", "gemm_nt_q", r3dh_i8_rad_gemm_nt_q, RANGE(gemm_limits), gemm_shape);
            item.layout = r3dh_i8_rad_layout;
            item.relayout = r3dh_i8_rad_relayout;
            item.unrelayout = r3dh_i8_rad_unrelayout;
            return item;
        }(),
        row("r3dh_ple_quant_i8g", "quant_act_i8g", r3dh_i8_rad_quant_act_i8g, RANGE(quant_limits), quant_shape),
        row("r3dh_ple_had_quant_i8g", "had_quant_act_i8g", r3dh_i8_rad_had_quant_act_i8g, RANGE(quant_limits), quant_shape)
    };
    return result;
}
}

extern "C" int r3dh_ple_host_device_kernel_count() { return 5; }
extern "C" const RadKernelInfo* r3dh_ple_host_device_kernel_at(int index) {
    return index >= 0 && index < r3dh_ple_host_device_kernel_count() ? rows() + index : nullptr;
}
