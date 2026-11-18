#include "rad_plugin.h"
#include "r3dh_sparse_qsa.h"
#include "r3dh_attn_qsa.h"

#include <algorithm>
#include <cstring>
#include <initializer_list>

namespace {

#define COUNT(values) static_cast<int>(sizeof(values) / sizeof(values[0]))
#define RANGE(values) values, COUNT(values)
#define PI(key) {key, RAD_P_INT, RAD_REQUIRED, RAD_PROLE_NONE}
#define OI(key) {key, RAD_P_INT, RAD_OPTIONAL, RAD_PROLE_NONE}
#define PF(key) {key, RAD_P_F64, RAD_REQUIRED, RAD_PROLE_NONE}
#define OF(key) {key, RAD_P_F64, RAD_OPTIONAL, RAD_PROLE_NONE}
#define PS(key) {key, RAD_P_STR, RAD_REQUIRED, RAD_PROLE_NONE}
#define OS(key) {key, RAD_P_STR, RAD_OPTIONAL, RAD_PROLE_NONE}
#define EQ(key, value) {key, RAD_C_EQ, value, nullptr}
#define GE(key, value) {key, RAD_C_GE, value, nullptr}
#define LE(key, value) {key, RAD_C_LE, value, nullptr}
#define IN(key, value) {key, RAD_C_IN, 0, value}

constexpr int64_t ratio = 4;
constexpr int64_t topk = 512;
constexpr int64_t selected_width = topk + 1;
constexpr int64_t selected_capacity = 256;
constexpr int64_t max_context = 200000;

const RadParamSpec key_params[]{
    PI("M"), PI("n"), PI("ratio"), PI("rotary_dim"), OI("pos0"),
    OI("block_size"), PF("eps"), PF("theta"), OF("scale"), OF("wadd"),
    PS("dtype"), OS("mode"), OS("sections"), OF("yarn_factor"),
    OF("yarn_low"), OF("yarn_high"), OF("yarn_magnitude")};
const RadOperandSpec key_operands[]{
    {"k", RAD_OPD_IN, 0}, {"w", RAD_OPD_WEIGHT, 0}, {"page", RAD_OPD_IN, 1},
    {"pos", RAD_OPD_IN, 1}, {"out", RAD_OPD_OUT, 0}};
const RadParamSpec score_params[]{
    PI("M"), PI("n"), PI("heads"), PI("blocks"), PI("seqs"), PS("dtype")};
const RadOperandSpec score_operands[]{
    {"q", RAD_OPD_IN, 0}, {"bk", RAD_OPD_IN, 0}, {"bt", RAD_OPD_IN, 0},
    {"cu", RAD_OPD_IN, 0}, {"nc", RAD_OPD_IN, 0}, {"score", RAD_OPD_OUT, 0}};
const RadParamSpec select_params[]{
    PI("M"), PI("blocks"), PI("topk"), PI("ratio"), PI("seqs"), OI("signed")};
const RadOperandSpec select_operands[]{
    {"score", RAD_OPD_IN, 0}, {"bt", RAD_OPD_IN, 0}, {"nc", RAD_OPD_IN, 0},
    {"pos", RAD_OPD_IN, 0}, {"cu", RAD_OPD_IN, 0}, {"sel", RAD_OPD_OUT, 0},
    {"nsel", RAD_OPD_OUT, 0}, {"seqused", RAD_OPD_OUT, 0}};
const RadOpSchema schemas[]{
    {"qsa_block_key", RANGE(key_params), RANGE(key_operands),
     "BF16 four-key pooling, F32 RMSNorm weight[128], rotary64; ratio4."},
    {"qsa_score", RANGE(score_params), RANGE(score_operands),
     "BF16 index query[4,128] against block keys[128], F32 ReLU head sum."},
    {"qsa_select", RANGE(select_params), RANGE(select_operands),
     "Stable unsigned or signed top512 physical pages plus partial page; ratio4."}
};

const RadConstraint key_constraints[]{
    GE("M", 1), LE("M", 520), EQ("n", 128), EQ("ratio", 4),
    EQ("rotary_dim", 64), EQ("block_size", 4), IN("dtype", "bf16"),
    IN("mode", "neox mrope imrope")};
const RadConstraint score_constraints[]{
    GE("M", 1), LE("M", 2048), EQ("n", 128), EQ("heads", 4),
    GE("blocks", 1), LE("blocks", 50000), GE("seqs", 1), LE("seqs", 8),
    IN("dtype", "bf16")};
const RadConstraint select_constraints[]{
    GE("M", 1), LE("M", 2048), GE("blocks", 1), LE("blocks", 50000),
    EQ("topk", 512), EQ("ratio", 4), GE("seqs", 1), LE("seqs", 8)};
#define PAGED_CONSTRAINTS(name, dtype) const RadConstraint name[]{ \
    GE("q_len", 1), LE("q_len", 2048), EQ("head_dim", 256), EQ("gqa", 12), \
    EQ("n_head", 12), EQ("block_size", 4), EQ("causal", 1), EQ("window", 0), \
    EQ("max_seqs", 256), GE("max_ctx", 1), LE("max_ctx", 200000), \
    IN("q_dtype", "bf16"), IN("kv_dtype", dtype)}
PAGED_CONSTRAINTS(paged_bf16_constraints, "bf16");
PAGED_CONSTRAINTS(paged_i8_constraints, "i8");

RadOpdDesc tensor(uint32_t dtype, std::initializer_list<int64_t> shape) {
    RadOpdDesc result{};
    result.dtype = dtype;
    result.rank = static_cast<uint32_t>(shape.size());
    result.idx_const = -1;
    std::copy(shape.begin(), shape.end(), result.shape);
    return result;
}

RadOpdDesc indices(std::initializer_list<int64_t> shape, int64_t limit,
                   int64_t constant = -1, uint32_t flags = 0) {
    RadOpdDesc result = tensor(RAD_I32, shape);
    result.fill = RAD_FILL_INDEX;
    result.idx_max = limit;
    result.idx_const = constant;
    result.flags = flags;
    return result;
}

bool paged_geometry(const RadParam* params, int count) {
    if (count < 0 || (count && !params)) return false;
    const int64_t rows = rad_param_getdim(params, count, "q_len", 0);
    const int64_t context = rad_param_getdim(params, count, "max_ctx", 0);
    const char* dtype = rad_param_gets(params, count, "kv_dtype", "");
    return rows >= 1 && rows <= 2048 && context >= 1 && context <= max_context
        && rad_param_geti(params, count, "head_dim", 0) == 256
        && rad_param_geti(params, count, "gqa", 0) == 12
        && rad_param_geti(params, count, "n_head", 0) == 12
        && rad_param_geti(params, count, "block_size", 0) == ratio
        && rad_param_geti(params, count, "causal", 0) == 1
        && rad_param_geti(params, count, "window", -1) == 0
        && rad_param_geti(params, count, "max_seqs", 0) == selected_capacity
        && !std::strcmp(rad_param_gets(params, count, "q_dtype", ""), "bf16")
        && (!std::strcmp(dtype, "bf16") || !std::strcmp(dtype, "i8"));
}

int paged_shape(const RadParam* params, int count, int operand, RadOpdDesc* out) {
    if (!out || operand < 0 || operand >= 8 || count < 0 || (count && !params))
        return RAD_E_INVAL;
    if (!paged_geometry(params, count)) return RAD_E_UNSUPPORTED;
    const int64_t rows = rad_param_getdim(params, count, "q_len", 0);
    const int64_t context = rad_param_getdim(params, count, "max_ctx", 0);
    const int64_t pages = (context + ratio - 1) / ratio;
    const int64_t used = std::min(context / ratio, topk) * ratio + context % ratio;
    const bool i8 = !std::strcmp(rad_param_gets(params, count, "kv_dtype", ""), "i8");
    const RadOpdDesc descriptors[]{
        tensor(RAD_BF16, {rows, 3072}),
        tensor(i8 ? RAD_I8 : RAD_BF16, {pages, 1, ratio, 512 + (i8 ? 4 : 0)}),
        indices({rows, selected_width}, pages),
        indices({rows}, used + 1, used),
        {}, {}, {}, tensor(RAD_BF16, {rows, 3072})};
    *out = descriptors[operand];
    return RAD_OK;
}

int paged_launch(const RadArgs* args, RadStream stream) {
    if (!args || !args->t || args->n_t != 8 || args->n_p < 0
        || (args->n_p && !args->p)) return RAD_E_INVAL;
    if (!paged_geometry(args->p, args->n_p)) return RAD_E_UNSUPPORTED;
    const int status = r3dh_sparse_qsa_shape("attn_paged", args);
    if (status != RAD_OK) return status;
    const RadTensor& query = args->t[0];
    const RadTensor& cache = args->t[1];
    const RadTensor& table = args->t[2];
    const RadTensor& lengths = args->t[3];
    const RadTensor& prefix = args->t[6];
    if (prefix.data && table.shape[1] > topk
        && cache.dtype == RAD_I8)
        return r3dh_attn_qsa_paged(args, stream);
    if (cache.shape[1] != 1 || table.shape[1] > selected_width)
        return RAD_E_UNSUPPORTED;
    if (prefix.data) {
        if (lengths.shape[0] > 8 || table.shape[0] != lengths.shape[0]
            || table.shape[1] > topk)
            return RAD_E_UNSUPPORTED;
    } else if (lengths.shape[0] != query.shape[0] || table.shape[0] != query.shape[0]) {
        return RAD_E_UNSUPPORTED;
    }
    return r3dh_sparse_qsa_paged(args, stream);
}

RadKernelInfo kernel(const char* name, const char* operation,
                     const RadConstraint* constraints, int count,
                     RadLaunchFn launch, RadShapeFn shape) {
    RadKernelInfo result{};
    result.name = name;
    result.op = operation;
    result.family = "attn";
    result.computes = "Strict native sparse QSA; no dense replacement above top512";
    result.shape = "n128 ratio4 index4x128; attention D256 GQA12 KV1 page4";
    result.dtypes = "BF16 query/output/block keys; F32 norm weight/scores; BF16 or inline-scale I8 KV";
    result.domain = RAD_DOMAIN_DEVICE;
    result.priority = 1200;
    result.constraints = constraints;
    result.n_constraints = count;
    result.launch = launch;
    result.opd_shape = shape;
    return result;
}

const RadKernelInfo kernels[]{
    kernel("sparse_qsa_block_key_bf16_strict", "qsa_block_key", RANGE(key_constraints),
           r3dh_sparse_qsa_block_key, r3dh_sparse_qsa_block_key_shape),
    kernel("sparse_qsa_score_bf16_strict", "qsa_score", RANGE(score_constraints),
           r3dh_sparse_qsa_score, r3dh_sparse_qsa_score_shape),
    kernel("sparse_qsa_select_signed_or_relu_strict", "qsa_select", RANGE(select_constraints),
           r3dh_sparse_qsa_select, r3dh_sparse_qsa_select_shape),
    kernel("sparse_qsa_selected_union_bf16_strict", "attn_paged", RANGE(paged_bf16_constraints),
           paged_launch, paged_shape),
    kernel("sparse_qsa_selected_union_i8_strict", "attn_paged", RANGE(paged_i8_constraints),
           paged_launch, paged_shape)
};

const char* const sources[]{
    "libr3d/r3dh_sparse_qsa_host.cpp",
    "libr3d/r3dh_sparse_qsa_device.hip",
    "libr3d/r3dh_sparse_qsa_index.hip"
};

}

extern "C" int r3d_sparse_schema_count() { return COUNT(schemas); }
extern "C" const RadOpSchema* r3d_sparse_schema_at(int index) {
    return index >= 0 && index < COUNT(schemas) ? &schemas[index] : nullptr;
}
extern "C" int r3d_sparse_kernel_count() { return COUNT(kernels); }
extern "C" const RadKernelInfo* r3d_sparse_kernel_at(int index) {
    return index >= 0 && index < COUNT(kernels) ? &kernels[index] : nullptr;
}
extern "C" int r3d_sparse_source_count() { return COUNT(sources); }
extern "C" const char* r3d_sparse_source_at(int index) {
    return index >= 0 && index < COUNT(sources) ? sources[index] : nullptr;
}
