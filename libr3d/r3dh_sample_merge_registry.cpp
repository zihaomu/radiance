#include "r3dh_sample_merge.h"
namespace {
const RadParamSpec parameters[]{{"M",RAD_P_INT,1,0},{"n_cand",RAD_P_INT,1,0}};
const RadOperandSpec operands[]{{"gathered",RAD_OPD_IN,0},{"params",RAD_OPD_IN,0},
    {"cand_idx",RAD_OPD_OUT,0},{"cand_val",RAD_OPD_OUT,0}};
const RadOpSchema schema{"sample_merge_topk",parameters,2,operands,4,
    "Vocab-parallel descending candidates, global ids and per-row top_k"};
const RadConstraint constraints[]{RAD_CGE("M",1),RAD_CLE("M",32),
    RAD_CGE("n_cand",1),RAD_CLE("n_cand",2048)};
const RadKernelInfo kernel = [] {
    RadKernelInfo result{};
    result.name = "r3dh_sample_merge_topk_f32"; result.op = "sample_merge_topk";
    result.family = "sample"; result.computes = "Deterministic GPU sort, duplicate suppression and per-row top_k";
    result.shape = "M=1..32, n_cand=1..2048, gathered pairs=1..4096; TP2 width1024 startup";
    result.dtypes = "u32/i32 gathered words, u32/i32/f32 params, i32 ids, f32 logits";
    result.domain = RAD_DOMAIN_DEVICE; result.priority = 200;
    result.constraints = constraints; result.n_constraints = 4;
    result.launch = r3dh_sample_merge_topk; result.opd_shape = r3dh_sample_merge_shape;
    return result;
}();
}
extern "C" int r3dh_sample_merge_shape(const RadParam* params, int count, int operand, RadOpdDesc* output) {
    if (!output || count < 0 || (count && !params)) return RAD_E_INVAL;
    const int64_t rows = rad_param_geti(params,count,"M",0), width = rad_param_geti(params,count,"n_cand",0);
    if (rows < 1 || rows > 32 || width < 1 || width > 2048 || operand < 0 || operand > 3) return RAD_E_SHAPE;
    const int64_t world = rad_param_geti(params,count,"world_size",0);
    if (operand == 0 && !world) return RAD_E_UNSUPPORTED;
    if (world && (world < 1 || world > 16 || world * width > 4096)) return RAD_E_SHAPE;
    RadOpdDesc result{}; result.rank = 2; result.shape[0] = rows;
    result.shape[1] = operand == 0 ? 2 * world * width : operand == 1 ? 24 : width;
    result.dtype = operand < 2 ? RAD_U32 : operand == 2 ? RAD_I32 : RAD_F32;
    result.fill = operand == 1 ? RAD_FILL_SAMPLE_PARAMS : RAD_FILL_NORMAL;
    result.idx_const = -1; *output = result; return RAD_OK;
}
extern "C" const RadKernelInfo* r3dh_sample_merge_kernels(int* count) {
    if (count) *count = 1; return &kernel;
}
extern "C" int r3dh_sample_merge_kernel_count() { return 1; }
extern "C" const RadKernelInfo* r3dh_sample_merge_kernel_at(int index) { return index == 0 ? &kernel : nullptr; }
extern "C" const RadOpSchema* r3dh_sample_merge_schemas(int* count) {
    if (count) *count = 1; return &schema;
}
extern "C" int r3dh_sample_merge_schema_count() { return 1; }
extern "C" const RadOpSchema* r3dh_sample_merge_schema_at(int index) { return index == 0 ? &schema : nullptr; }
