#include "r3dh_vision_strict.h"
namespace {
int plain_shape(const RadParam* params, int count, int operand, RadOpdDesc* output) {
    return r3dh_vision_strict_shape("gemm_nt", params, count, operand, output);
}
int bias_shape(const RadParam* params, int count, int operand, RadOpdDesc* output) {
    return r3dh_vision_strict_shape("gemm_nt_bias", params, count, operand, output);
}
int attention_shape(const RadParam* params, int count, int operand, RadOpdDesc* output) {
    return r3dh_vision_strict_shape("attn_dense", params, count, operand, output);
}
const RadParamSpec gemm_params[]{
    {"M",RAD_P_INT,1,0},{"N",RAD_P_INT,1,0},{"K",RAD_P_INT,1,0},{"dtype",RAD_P_STR,1,0}};
const RadParamSpec bias_params[]{
    {"M",RAD_P_INT,1,0},{"N",RAD_P_INT,1,0},{"K",RAD_P_INT,1,0},{"dtype",RAD_P_STR,1,0},{"act",RAD_P_STR,0,0}};
const RadParamSpec attention_params[]{
    {"M",RAD_P_INT,1,0},{"head_dim",RAD_P_INT,1,0},{"gqa",RAD_P_INT,1,0},{"causal",RAD_P_INT,1,0},
    {"dtype",RAD_P_STR,1,0},{"scale",RAD_P_F64,0,0},{"max_seqlen",RAD_P_INT,0,RAD_PROLE_CAPACITY},
    {"window",RAD_P_INT,0,0},{"cu_div",RAD_P_INT,0,0},{"block",RAD_P_INT,0,0},{"window_right",RAD_P_INT,0,0}};
const RadOperandSpec gemm_operands[]{{"a",RAD_OPD_IN,0},{"b",RAD_OPD_WEIGHT,0},{"y",RAD_OPD_OUT,0}};
const RadOperandSpec bias_operands[]{{"a",RAD_OPD_IN,0},{"b",RAD_OPD_WEIGHT,0},{"bias",RAD_OPD_WEIGHT,0},
    {"res",RAD_OPD_IN,1},{"y",RAD_OPD_OUT,0}};
const RadOperandSpec attention_operands[]{{"q",RAD_OPD_IN,0},{"k",RAD_OPD_IN,0},{"v",RAD_OPD_IN,0},
    {"cu_seqlens",RAD_OPD_IN,0},{"out",RAD_OPD_OUT,0}};
const RadOpSchema schemas[]{
    {"gemm_nt",gemm_params,4,gemm_operands,3,"Strict ascending-K BF16 vision projection"},
    {"gemm_nt_bias",bias_params,5,bias_operands,5,"Strict BF16 projection, bias, rounded GELU and residual"},
    {"attn_dense",attention_params,11,attention_operands,5,"Noncausal BF16 16-head dimension72, F32 softmax"}};
const char* plain_names[]{"r3dh_vision_strict_nt_patch", "r3dh_vision_strict_nt_qkv", "r3dh_vision_strict_nt_proj",
    "r3dh_vision_strict_nt_up", "r3dh_vision_strict_nt_down", "r3dh_vision_strict_nt_merge1", "r3dh_vision_strict_nt_merge2"};
const char* bias_names[]{"r3dh_vision_strict_bias_patch", "r3dh_vision_strict_bias_qkv", "r3dh_vision_strict_bias_proj",
    "r3dh_vision_strict_bias_up", "r3dh_vision_strict_bias_down", "r3dh_vision_strict_bias_merge1", "r3dh_vision_strict_bias_merge2"};
struct Tables {
    RadConstraint constraints[15][6]{};
    RadKernelInfo rows[15]{};
    Tables() {
        const int64_t shapes[][2]{{1152,1536},{3456,1152},{1152,1152},{4304,1152},{1152,4304},{4608,4608},{2560,4608}};
        for (int index = 0; index < 15; ++index) {
            auto& row = rows[index];
            row.domain = RAD_DOMAIN_DEVICE; row.priority = 200;
            row.dtypes = "bf16"; row.constraints = constraints[index]; row.n_constraints = index == 14 ? 6 : 5;
            constraints[index][0] = RadConstraint RAD_CGE("M",1);
            constraints[index][1] = RadConstraint RAD_CLE("M",256);
            constraints[index][2] = RadConstraint RAD_CIN("dtype","bf16");
            if (index < 14) {
                const int shape = index % 7;
                row.name = index < 7 ? plain_names[shape] : bias_names[shape];
                row.op = index < 7 ? "gemm_nt" : "gemm_nt_bias";
                row.family = "gemm"; row.computes = "ascending K, separate F32 multiply and add, BF16 RNE";
                row.shape = "M1..256; seven exact vision N/K geometries";
                row.launch = index < 7 ? r3dh_vision_strict_gemm_nt : r3dh_vision_strict_gemm_nt_bias;
                row.opd_shape = index < 7 ? plain_shape : bias_shape;
                constraints[index][3] = RadConstraint RAD_CEQ("N",shapes[shape][0]);
                constraints[index][4] = RadConstraint RAD_CEQ("K",shapes[shape][1]);
            } else {
                row.name = "r3dh_vision_strict_attn72"; row.op = "attn_dense"; row.family = "attn";
                row.computes = "noncausal segmented BF16, strict F32 scores/softmax/value reduction";
                row.shape = "M1..256, heads16, head_dim72, gqa1, cu_div1, no window/block";
                row.launch = r3dh_vision_strict_attn_dense; row.opd_shape = attention_shape;
                constraints[index][3] = RadConstraint RAD_CEQ("head_dim",72);
                constraints[index][4] = RadConstraint RAD_CEQ("gqa",1);
                constraints[index][5] = RadConstraint RAD_CEQ("causal",0);
            }
        }
    }
};
const Tables tables;
}
extern "C" const RadKernelInfo* r3dh_vision_strict_kernels(int* count) {
    if (count) *count = 15; return tables.rows;
}
extern "C" const RadOpSchema* r3dh_vision_strict_schemas(int* count) {
    if (count) *count = 3; return schemas;
}
