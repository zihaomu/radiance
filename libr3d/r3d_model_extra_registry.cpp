#include "rad_plugin.h"
#include "r3dh_moe_iqr.h"
#include "r3dh_moe_w4nl.h"
#include "r3dh_gdn_extra.h"
#include "r3dh_attn_qsa.h"
#include "r3dh_qwen_ops.h"
#include "r3dh_moe_mixed.h"

#include <algorithm>
#include <cstring>
#include <initializer_list>

extern "C" int r3d_shape_gated_had_quant_i8g(const RadParam*, int, int, RadOpdDesc*);
extern "C" int r3d_shape_gemm_nt(const RadParam*, int, int, RadOpdDesc*);

namespace {

#define COUNT(values) static_cast<int>(sizeof(values) / sizeof(values[0]))
#define RANGE(values) values, COUNT(values)
#define REQUIRED(key, type) {key, type, RAD_REQUIRED, RAD_PROLE_NONE}
#define OPTIONAL(key, type) {key, type, RAD_OPTIONAL, RAD_PROLE_NONE}
#define EQ(key, value) {key, RAD_C_EQ, value, nullptr}
#define GE(key, value) {key, RAD_C_GE, value, nullptr}
#define LE(key, value) {key, RAD_C_LE, value, nullptr}
#define IN(key, value) {key, RAD_C_IN, 0, value}
#define PI(key) REQUIRED(key, RAD_P_INT)
#define PS(key) REQUIRED(key, RAD_P_STR)
#define PF(key) REQUIRED(key, RAD_P_F64)
#define OI(key) OPTIONAL(key, RAD_P_INT)
#define OS(key) OPTIONAL(key, RAD_P_STR)
#define OF(key) OPTIONAL(key, RAD_P_F64)
#define INPUT(key) {key, RAD_OPD_IN, 0}
#define OUTPUT(key) {key, RAD_OPD_OUT, 0}
#define INOUT(key) {key, RAD_OPD_INOUT, 0}
#define WEIGHT(key) {key, RAD_OPD_WEIGHT, 0}
#define INPUT_OPTIONAL(key) {key, RAD_OPD_IN, 1}
#define OUTPUT_OPTIONAL(key) {key, RAD_OPD_OUT, 1}
#define WEIGHT_OPTIONAL(key) {key, RAD_OPD_WEIGHT, 1}
#define TOKENS GE("M", 1), LE("M", 2048)
#define GDN EQ("head_k", 128), EQ("head_v", 128)
#define MOE EQ("n_expert", 512), EQ("top_k", 10), EQ("group", 128), EQ("a_group", 128)

const RadParamSpec rope_table_params[]{
    PI("M"), PI("rot"), PF("theta"), PF("scale"), PS("dtype"),
    OF("yarn_factor"), OF("yarn_low"), OF("yarn_high"), OF("yarn_magnitude"),
    OI("head_dim"), OS("ladder")};
const RadOperandSpec rope_table_operands[]{INPUT("positions"), OUTPUT("cos_sin")};
const RadParamSpec prepare_params[]{
    PI("M"), PI("head_dim"), PI("n_head"), PI("n_head_kv"), PI("rot"),
    PS("q_dtype"), OF("eps"), OF("wadd"), OI("cs_f32"), OI("pos_i64")};
const RadOperandSpec prepare_operands[]{
    INPUT_OPTIONAL("q_gate"), INPUT("k"), INPUT("cos_sin"), INPUT("positions"),
    WEIGHT_OPTIONAL("q_w"), WEIGHT("k_w"), OUTPUT_OPTIONAL("q_out"),
    OUTPUT("k_out"), OUTPUT_OPTIONAL("gate_out")};
const RadParamSpec dense_params[]{
    PI("M"), PI("head_dim"), PI("gqa"), PI("causal"), PS("dtype"), OF("scale"),
    OI("max_seqlen"), OI("window"), OI("cu_div"), OI("block"), OI("window_right")};
const RadOperandSpec dense_operands[]{
    INPUT("q"), INPUT("k"), INPUT("v"), INPUT("cu_seqlens"), OUTPUT("out")};
const RadParamSpec work_params[]{
    PI("M"), PI("n"), PI("ratio"), PI("ring"), PI("seqs"),
    {"work", RAD_P_INT, RAD_REQUIRED, RAD_PROLE_CAPACITY}, PS("dtype")};
const RadOperandSpec work_operands[]{
    INPUT("k"), INPUT("pos"), INPUT("cu"), INPUT("state"), INPUT("sidx"),
    INPUT("bt"), INPUT_OPTIONAL("rope"), OUTPUT("stage"), OUTPUT("page"),
    OUTPUT("bpos"), OUTPUT("nc")};
const RadParamSpec fused_gdn_params[]{
    PI("q_len"), PI("head_k"), PI("head_v"), PI("conv_width"), OI("n_head_v"),
    OF("scale"), OF("eps"), OF("l2_eps"), OF("softplus_thr"), OS("act"),
    OI("state_fp16"), OS("state_form"), OI("group")};
const RadOperandSpec fused_gdn_operands[]{
    INPUT("x"), WEIGHT("conv_w"), WEIGHT_OPTIONAL("conv_b"), INOUT("conv_state"),
    INPUT("conv_idx"), INPUT("a"), INPUT("b"), WEIGHT("A_log"), WEIGHT("dt_bias"),
    INOUT("state"), INPUT("state_idx"), INPUT("num_accepted"), INPUT("cu"),
    INPUT("z"), WEIGHT("norm_w"), OUTPUT("o"), OUTPUT("o_q"), OUTPUT("o_scale")};
const RadParamSpec gate_params[]{PI("M"), PI("n"), PI("group"), PS("dtype"), OS("act")};
const RadOperandSpec gate_operands[]{
    INPUT("gate"), INPUT("x"), OUTPUT("q"), OUTPUT("scale"), OUTPUT_OPTIONAL("out_bf16")};
const RadOpSchema schemas[]{
    {"rope_table", RANGE(rope_table_params), RANGE(rope_table_operands), "Position-indexed rotary table"},
    {"qk_norm_rope_gate", RANGE(prepare_params), RANGE(prepare_operands), "Normalize, rotate, and split Q/gate"},
    {"attn_dense", RANGE(dense_params), RANGE(dense_operands), "Ragged dense attention"},
    {"qsa_work", RANGE(work_params), RANGE(work_operands), "Collect completing QSA blocks"},
    {"gdn_conv_recurrent_update_had", RANGE(fused_gdn_params), RANGE(fused_gdn_operands), "Strict GDN update and grouped Hadamard quantization"},
    {"gate_had_quant_i8g", RANGE(gate_params), RANGE(gate_operands), "Separate gate and grouped Hadamard quantization"}
};

int64_t dimension(const RadParam* params, int count, const char* key, int64_t fallback = 0) {
    return rad_param_getdim(params, count, key, fallback);
}
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
template <size_t length>
int pick(const RadOpdDesc (&descriptors)[length], int operand, RadOpdDesc* output) {
    if (!output || operand < 0 || operand >= static_cast<int>(length)) return RAD_E_INVAL;
    for (uint32_t axis = 0; axis < descriptors[operand].rank; ++axis)
        if (descriptors[operand].shape[axis] < 1) return RAD_E_UNSUPPORTED;
    *output = descriptors[operand];
    return RAD_OK;
}
int64_t table_rows(const RadParam* params, int count, int64_t rows) {
    return dimension(params, count, "max_ctx", std::max<int64_t>(rows, 4));
}
int rope_table_shape(const RadParam* params, int count, int operand, RadOpdDesc* output) {
    const int64_t rows = dimension(params, count, "M");
    const int64_t capacity = table_rows(params, count, rows);
    if (rows < 1 || rows > 2048 || capacity < 1 || capacity > 200000) return RAD_E_UNSUPPORTED;
    const RadOpdDesc descriptors[]{indices({rows}, capacity), tensor(RAD_F32, {capacity, 64})};
    return pick(descriptors, operand, output);
}
int prepare_shape(const RadParam* params, int count, int operand, RadOpdDesc* output) {
    const int64_t rows = dimension(params, count, "M");
    const int64_t width = dimension(params, count, "head_dim");
    const int64_t heads = dimension(params, count, "n_head");
    const int64_t keys = dimension(params, count, "n_head_kv");
    if (rows < 1 || rows > 2048 || !((width == 256 && heads == 12 && keys == 1)
        || (width == 128 && heads == 0 && keys == 4))) return RAD_E_UNSUPPORTED;
    const int64_t capacity = table_rows(params, count, rows);
    if (capacity < 1 || capacity > 200000) return RAD_E_UNSUPPORTED;
    if (heads == 0 && (operand == 0 || operand == 4 || operand == 6 || operand == 8))
        return RAD_E_UNSUPPORTED;
    const RadOpdDesc descriptors[]{
        tensor(RAD_BF16, {rows, 2 * heads * width}), tensor(RAD_BF16, {rows, keys * width}),
        tensor(RAD_F32, {capacity, 64}), indices({rows}, capacity),
        tensor(RAD_F32, {width}), tensor(RAD_F32, {width}),
        tensor(RAD_BF16, {rows, heads * width}), tensor(RAD_BF16, {rows, keys * width}),
        tensor(RAD_BF16, {rows, heads, width})};
    return pick(descriptors, operand, output);
}
int rope_shape(const RadParam* params, int count, int operand, RadOpdDesc* output) {
    const int64_t rows = dimension(params, count, "M");
    const int64_t width = dimension(params, count, "head_dim");
    const int64_t heads = dimension(params, count, "n_head");
    const bool multicomponent = !std::strcmp(rad_param_gets(params, count, "mode", ""), "imrope");
    if (rows < 1 || rows > 2048 || !((width == 256 && (heads == 1 || heads == 12))
        || (width == 128 && heads == 4))) return RAD_E_UNSUPPORTED;
    const RadOpdDesc descriptors[]{tensor(RAD_BF16, {rows, heads * width}),
        multicomponent ? indices({3, rows}, 200000) : indices({rows}, 200000)};
    return pick(descriptors, operand, output);
}
int kv_shape(const RadParam* params, int count, int operand, RadOpdDesc* output) {
    const int64_t rows = dimension(params, count, "M");
    if (rows < 1 || rows > 2048) return RAD_E_UNSUPPORTED;
    const int64_t pages = (rows + 3) / 4 + 2;
    const RadOpdDesc descriptors[]{tensor(RAD_BF16, {rows, 256}), tensor(RAD_BF16, {rows, 256}),
        indices({rows}, pages * 4, -1, RAD_OPD_F_IDX_UNIQUE), tensor(RAD_I8, {pages, 1, 4, 516})};
    return pick(descriptors, operand, output);
}
int dense_shape(const RadParam* params, int count, int operand, RadOpdDesc* output) {
    const int64_t rows = dimension(params, count, "M");
    if (rows < 1 || rows > 2048) return RAD_E_UNSUPPORTED;
    const RadOpdDesc descriptors[]{tensor(RAD_BF16, {rows, 3072}), tensor(RAD_BF16, {rows, 256}),
        tensor(RAD_BF16, {rows, 256}), indices({2}, rows + 1, -1, RAD_OPD_F_IDX_CU),
        tensor(RAD_BF16, {rows, 3072})};
    return pick(descriptors, operand, output);
}
int paged_shape(const RadParam* params, int count, int operand, RadOpdDesc* output) {
    const int64_t rows = dimension(params, count, "q_len");
    const int64_t capacity = dimension(params, count, "max_ctx", rows);
    if (rows < 1 || rows > 2048 || capacity < rows || capacity > 200000) return RAD_E_UNSUPPORTED;
    if (operand == 4 || operand == 5) return RAD_E_UNSUPPORTED;
    const int64_t pages = (capacity + 3) / 4;
    const RadOpdDesc descriptors[]{tensor(RAD_BF16, {rows, 3072}), tensor(RAD_I8, {pages, 1, 4, 516}),
        indices({1, pages}, pages, -1, RAD_OPD_F_IDX_UNIQUE), indices({1}, capacity + 1, capacity),
        {}, {}, indices({2}, rows + 1, -1, RAD_OPD_F_IDX_CU), tensor(RAD_BF16, {rows, 3072})};
    return pick(descriptors, operand, output);
}
template <bool work>
int qsa_shape(const RadParam* params, int count, int operand, RadOpdDesc* output) {
    const int64_t rows = dimension(params, count, "M");
    const int64_t sequences = dimension(params, count, "seqs");
    const int64_t ring = dimension(params, count, "ring");
    const int64_t capacity = dimension(params, count, "work", 520);
    if (rows < 1 || rows > 2048 || sequences < 1 || sequences > 8
        || ring < 4 || ring > 7
        || (work && (capacity < (rows + 3) / 4 + sequences || capacity > 520))) return RAD_E_UNSUPPORTED;
    const RadOpdDesc common[]{tensor(RAD_BF16, {rows, 128}), indices({rows}, 200000),
        indices({sequences + 1}, rows + 1, -1, RAD_OPD_F_IDX_CU),
        tensor(RAD_BF16, {sequences, 1, ring, 128}),
        indices({sequences}, sequences, -1, RAD_OPD_F_IDX_UNIQUE)};
    if constexpr (!work) return pick(common, operand, output);
    if (operand < 5) return pick(common, operand, output);
    const int64_t pages = std::min<int64_t>(50000, (rows + 3) / 4 + 2);
    const RadOpdDesc extra[]{indices({sequences, pages}, pages), indices({3, rows}, 200000),
        tensor(RAD_BF16, {4 * capacity, 128}), indices({capacity}, pages),
        indices({capacity, 3}, 200000), indices({sequences}, capacity + 1)};
    return pick(extra, operand - 5, output);
}
template <int operation>
int gdn_shape(const RadParam* params, int count, int operand, RadOpdDesc* output) {
    const int64_t bound = dimension(params, count, operation == 1 || operation >= 4 ? "M" : "q_len");
    const int64_t rows = dimension(params, count, "M", bound);
    const int64_t heads = dimension(params, count, "n_head_v", 24);
    const int64_t keys = heads / 3;
    const int64_t channels = (2 * keys + heads) * 128;
    const bool anchor = !std::strcmp(rad_param_gets(params, count, "state_form", "per_candidate"), "anchor");
    const int64_t columns = anchor ? 6 : bound;
    const int64_t slots = std::max<int64_t>(6, bound);
    const int64_t group = dimension(params, count, "group", 128);
    if (rows < 1 || rows > 2048 || (heads != 24 && heads != 48)
        || (group != 128 && group != 256) || bound < 1
        || ((operation == 0 || operation == 3) && bound > 8)
        || (operation == 2 && bound > 16)) return RAD_E_UNSUPPORTED;
    const RadOpdDesc query = tensor(RAD_BF16, {rows, keys, 128});
    const RadOpdDesc value = tensor(RAD_BF16, {rows, heads, 128});
    const RadOpdDesc gate = tensor(RAD_BF16, {rows, heads});
    const RadOpdDesc weights = tensor(RAD_F32, {heads});
    const RadOpdDesc state = tensor(RAD_F32, {slots, heads, 128, 128});
    const RadOpdDesc state_index = indices({1, columns}, slots, -1, RAD_OPD_F_IDX_UNIQUE);
    const RadOpdDesc cumulative = indices({2}, rows + 1, -1, RAD_OPD_F_IDX_CU);
    const RadOpdDesc accepted = indices({1}, 2, 1);
    const RadOpdDesc convolution[]{tensor(RAD_BF16, {rows, channels}), tensor(RAD_BF16, {channels, 4}),
        tensor(RAD_BF16, {channels}), tensor(RAD_BF16, {1, channels, operation == 1 ? 3 : bound + 2})};
    if constexpr (operation == 0) {
        const RadOpdDesc descriptors[]{convolution[0], convolution[1], convolution[2], convolution[3],
            indices({1}, 1, 0), accepted, cumulative, query, query, value};
        return pick(descriptors, operand, output);
    } else if constexpr (operation == 1) {
        const RadOpdDesc descriptors[]{convolution[0], convolution[1], convolution[2], weights, weights,
            convolution[3], cumulative, gate, gate, indices({1}, 1, 0), indices({1}, 2, 1),
            query, query, value, tensor(RAD_F32, {rows, heads}), tensor(RAD_F32, {rows, heads})};
        return pick(descriptors, operand, output);
    } else if constexpr (operation == 2) {
        if (operand == 14 || operand == 15) return RAD_E_UNSUPPORTED;
        const RadOpdDesc descriptors[]{query, query, value, gate, gate, weights, weights,
            state, state_index, accepted, cumulative, value, tensor(RAD_F32, {128}), value};
        return pick(descriptors, operand, output);
    } else if constexpr (operation == 3) {
        const RadOpdDesc descriptors[]{convolution[0], convolution[1], convolution[2], convolution[3],
            indices({1}, 1, 0), gate, gate, weights, weights, state, state_index, accepted, cumulative,
            value, tensor(RAD_F32, {128}), value, tensor(RAD_I8, {rows, heads * 128}),
            tensor(RAD_F32, {rows, heads * 128 / group})};
        return pick(descriptors, operand, output);
    } else {
        RadOpdDesc decay = tensor(RAD_F32, {rows, heads});
        decay.fill = RAD_FILL_GATE_CUMSUM;
        decay.fill_chunk = 64;
        RadOpdDesc beta = tensor(RAD_F32, {rows, heads});
        beta.fill = RAD_FILL_SIGMOID;
        RadOpdDesc inverse = tensor(RAD_BF16, {rows, heads, 64});
        inverse.fill = RAD_FILL_TRI_LOWER;
        inverse.fill_chunk = 64;
        if constexpr (operation == 4) {
            const RadOpdDesc descriptors[]{query, beta, decay, cumulative, inverse};
            return pick(descriptors, operand, output);
        } else {
            const RadOpdDesc descriptors[]{query, query, value, inverse, decay, beta, state,
                cumulative, value, state, indices({1}, slots, 0)};
            return pick(descriptors, operand, output);
        }
    }
}
int recurrent_launch(const RadArgs* arguments, RadStream stream) {
    if (!arguments || !arguments->t || arguments->n_t != 16) return RAD_E_INVAL;
    if (arguments->t[14].data || arguments->t[15].data) return RAD_E_UNSUPPORTED;
    RadArgs forwarded = *arguments;
    forwarded.n_t = 14;
    return r3dh_gdn_extra_recurrent_strict(&forwarded, stream);
}
template <bool solve>
int chunk_launch(const RadArgs* arguments, RadStream stream) {
    const RadTensor* cumulative = rad_arg_in(arguments, solve ? 3 : 7);
    if (!cumulative || cumulative->rank != 1 || cumulative->shape[0] < 2
        || cumulative->shape[0] > 9) return RAD_E_UNSUPPORTED;
    return solve ? r3dh_gdn_extra_kkt_strict(arguments, stream)
                 : r3dh_gdn_extra_chunk_strict(arguments, stream);
}
int gate_shape(const RadParam* params, int count, int operand, RadOpdDesc* output) {
    const int64_t rows = dimension(params, count, "M");
    if (rows < 1 || rows > 24576) return RAD_E_UNSUPPORTED;
    const RadOpdDesc descriptors[]{tensor(RAD_BF16, {rows, 256}), tensor(RAD_BF16, {rows, 256}),
        tensor(RAD_I8, {rows, 256}), tensor(RAD_F32, {rows, 2}), tensor(RAD_BF16, {rows, 256})};
    return pick(descriptors, operand, output);
}

#define IQR_CONSTRAINTS(name, even, odd) const RadConstraint name[]{TOKENS, MOE, \
    EQ("N", even), EQ("N_odd", odd), EQ("K", 2560), EQ("K_odd", 2560), \
    EQ("parts", 2), IN("dtype", "i8a8h"), IN("a_order", "token")}
#define W4NL_CONSTRAINTS(name, even, odd) const RadConstraint name[]{TOKENS, MOE, \
    EQ("N", 2560), EQ("N_odd", 2560), EQ("K", even), EQ("K_odd", odd), \
    EQ("parts", 1), IN("dtype", "w4nl64a8i8h"), IN("a_order", "sorted")}
IQR_CONSTRAINTS(iqr_768_512, 768, 512);
IQR_CONSTRAINTS(iqr_512_768, 512, 768);
W4NL_CONSTRAINTS(w4nl_384_256, 384, 256);
W4NL_CONSTRAINTS(w4nl_256_384, 256, 384);
#define W4NL_GATE_CONSTRAINTS(name, even, odd) const RadConstraint name[]{TOKENS, MOE, \
    EQ("N", even), EQ("N_odd", odd), EQ("K", 2560), EQ("K_odd", 2560), \
    EQ("parts", 2), IN("dtype", "w4nl64a8i8h"), IN("a_order", "token")}
W4NL_GATE_CONSTRAINTS(w4nl_gate_768_512, 768, 512);
W4NL_GATE_CONSTRAINTS(w4nl_gate_512_768, 512, 768);
const RadConstraint conv_prep_constraints[]{TOKENS, GDN, EQ("conv_width", 4), EQ("chunk", 64)};
const RadConstraint conv_update_constraints[]{GDN, GE("q_len", 1), LE("q_len", 8), EQ("conv_width", 4)};
const RadConstraint recurrent_constraints[]{GDN, GE("q_len", 1), LE("q_len", 16), EQ("n_head_v", 24), IN("act", "sigmoid silu"), IN("state_form", "anchor per_candidate")};
const RadConstraint fused_constraints[]{GDN, GE("q_len", 1), LE("q_len", 8), EQ("conv_width", 4), EQ("n_head_v", 24), IN("act", "sigmoid silu"), IN("state_form", "anchor per_candidate")};
const RadConstraint kkt_constraints[]{TOKENS, EQ("head_k", 128), EQ("chunk", 64)};
const RadConstraint chunk_constraints[]{TOKENS, GDN, EQ("chunk", 64)};
const RadConstraint table_constraints[]{TOKENS, EQ("rot", 64), IN("dtype", "f32")};
const RadConstraint prepare_constraints[]{TOKENS, EQ("head_dim", 256), EQ("n_head", 12), EQ("n_head_kv", 1), EQ("rot", 64), EQ("cs_f32", 1), IN("q_dtype", "bf16")};
const RadConstraint index_prepare_constraints[]{TOKENS, EQ("head_dim", 128), EQ("n_head", 0), EQ("n_head_kv", 4), EQ("rot", 64), EQ("cs_f32", 1), IN("q_dtype", "bf16")};
#define ROPE_CONSTRAINTS(name, width, heads) const RadConstraint name[]{TOKENS, \
    EQ("head_dim", width), EQ("n_head", heads), EQ("n_head_kv", 0), \
    EQ("rotary_dim", 64), IN("mode", "neox imrope")}
ROPE_CONSTRAINTS(rope_query_constraints, 256, 12);
ROPE_CONSTRAINTS(rope_key_constraints, 256, 1);
ROPE_CONSTRAINTS(rope_index_constraints, 128, 4);
const RadConstraint kv_constraints[]{TOKENS, EQ("head_dim", 256), EQ("n_head_kv", 1), EQ("block_size", 4), IN("kv_dtype", "i8")};
const RadConstraint dense_constraints[]{TOKENS, EQ("head_dim", 256), EQ("gqa", 12), EQ("causal", 1), IN("dtype", "bf16")};
const RadConstraint paged_constraints[]{GE("q_len", 1), LE("q_len", 2048), EQ("head_dim", 256), EQ("gqa", 12), EQ("n_head", 12), EQ("block_size", 4), EQ("causal", 1), EQ("window", 0), IN("q_dtype", "bf16"), IN("kv_dtype", "i8")};
#define QSA TOKENS, EQ("n", 128), EQ("ratio", 4), IN("ring", "4 5 6 7"), GE("seqs", 1), LE("seqs", 8), IN("dtype", "bf16")
const RadConstraint work_constraints[]{QSA, GE("work", 1), LE("work", 520)};
const RadConstraint tail_constraints[]{QSA};
#define PACKED_CONSTRAINTS(name, width) const RadConstraint name[]{GE("M", 1), LE("M", 20480), EQ("n", width), EQ("group", 128), IN("dtype", "bf16"), IN("act", "silu")}
PACKED_CONSTRAINTS(packed_256_constraints, 256);
PACKED_CONSTRAINTS(packed_384_constraints, 384);
const RadConstraint gate_constraints[]{GE("M", 1), LE("M", 24576), EQ("n", 256), EQ("group", 128), IN("dtype", "bf16"), IN("act", "sigmoid")};
const RadConstraint router_constraints[]{TOKENS, IN("N", "1 512"), EQ("K", 2560), IN("dtype", "bf16")};

RadKernelInfo kernel(const char* name, const char* operation, const char* family,
                     const RadConstraint* constraints, int count, RadLaunchFn launch,
                     RadShapeFn shape, RadScratchFn scratch = nullptr) {
    RadKernelInfo result{};
    result.name = name;
    result.op = operation;
    result.family = family;
    result.computes = "Strict native model operation";
    result.shape = "Contract-qualified geometry only";
    result.dtypes = "Strict port contract";
    result.domain = RAD_DOMAIN_DEVICE;
    result.priority = 1000;
    result.constraints = constraints;
    result.n_constraints = count;
    result.launch = launch;
    result.scratch = scratch;
    result.opd_shape = shape;
    return result;
}
RadKernelInfo moe_kernel(const char* name, const RadConstraint* constraints, int count, bool nonlinear) {
    RadKernelInfo result = kernel(name, "moe_gemm_q", "moe", constraints, count,
        nonlinear ? r3dh_moe_w4nl_launch : r3dh_moe_iqr_launch,
        nonlinear ? r3dh_moe_w4nl_shape : r3dh_moe_iqr_shape);
    result.layout = nonlinear ? r3dh_moe_w4nl_layout : r3dh_moe_iqr_layout;
    result.relayout = nonlinear ? r3dh_moe_w4nl_transform : r3dh_moe_iqr_relayout;
    result.unrelayout = nonlinear ? r3dh_moe_w4nl_inverse : r3dh_moe_iqr_unrelayout;
    return result;
}
const RadKernelInfo kernels[]{
    moe_kernel("model_moe_iqr_768_512_strict", RANGE(iqr_768_512), false),
    moe_kernel("model_moe_iqr_512_768_strict", RANGE(iqr_512_768), false),
    moe_kernel("model_moe_w4nl_384_256_strict", RANGE(w4nl_384_256), true),
    moe_kernel("model_moe_w4nl_256_384_strict", RANGE(w4nl_256_384), true),
    moe_kernel("model_moe_w4nl_gate_768_512_strict", RANGE(w4nl_gate_768_512), true),
    moe_kernel("model_moe_w4nl_gate_512_768_strict", RANGE(w4nl_gate_512_768), true),
    kernel("model_gdn_conv_prep_strict", "gdn_conv_prep", "gdn", RANGE(conv_prep_constraints), r3dh_gdn_extra_conv_prep_strict, gdn_shape<1>),
    kernel("model_gdn_conv_update_strict", "gdn_conv_update", "gdn", RANGE(conv_update_constraints), r3dh_gdn_extra_conv_update_strict, gdn_shape<0>),
    kernel("model_gdn_recurrent_strict", "gdn_recurrent_update", "gdn", RANGE(recurrent_constraints), recurrent_launch, gdn_shape<2>),
    kernel("model_gdn_conv_recurrent_had_strict", "gdn_conv_recurrent_update_had", "gdn", RANGE(fused_constraints), r3dh_gdn_extra_conv_recurrent_update_had_strict, gdn_shape<3>, r3dh_gdn_extra_scratch),
    kernel("model_gdn_kkt_strict", "gdn_kkt_solve", "gdn", RANGE(kkt_constraints), chunk_launch<true>, gdn_shape<4>),
    kernel("model_gdn_chunk_strict", "gdn_chunk_scan", "gdn", RANGE(chunk_constraints), chunk_launch<false>, gdn_shape<5>),
    kernel("model_rope_table_strict", "rope_table", "attn", RANGE(table_constraints), r3dh_attn_qsa_rope_table, rope_table_shape),
    kernel("model_qk_prepare_strict", "qk_norm_rope_gate", "attn", RANGE(prepare_constraints), r3dh_attn_qsa_prepare, prepare_shape),
    kernel("model_qsa_prepare_strict", "qk_norm_rope_gate", "attn", RANGE(index_prepare_constraints), r3dh_attn_qsa_prepare, prepare_shape),
    kernel("model_rope_query_strict", "rope", "attn", RANGE(rope_query_constraints), r3dh_attn_qsa_rope, rope_shape),
    kernel("model_rope_key_strict", "rope", "attn", RANGE(rope_key_constraints), r3dh_attn_qsa_rope, rope_shape),
    kernel("model_rope_index_strict", "rope", "attn", RANGE(rope_index_constraints), r3dh_attn_qsa_rope, rope_shape),
    kernel("model_kv_i8_strict", "kv_store", "kv", RANGE(kv_constraints), r3dh_attn_qsa_kv_store, kv_shape),
    kernel("model_attn_dense_strict", "attn_dense", "attn", RANGE(dense_constraints), r3dh_attn_qsa_dense, dense_shape),
    kernel("model_attn_paged_i8_strict", "attn_paged", "attn", RANGE(paged_constraints), r3dh_attn_qsa_paged, paged_shape),
    kernel("model_qsa_work_strict", "qsa_work", "kv", RANGE(work_constraints), r3dh_attn_qsa_work, qsa_shape<true>),
    kernel("model_qsa_tail_strict", "qsa_tail_store", "kv", RANGE(tail_constraints), r3dh_attn_qsa_tail_store, qsa_shape<false>),
    kernel("model_gated_had_256_strict", "gated_had_quant_i8g", "quant", RANGE(packed_256_constraints), r3dh_qwen_ops_gated_had_quant_i8g, r3d_shape_gated_had_quant_i8g),
    kernel("model_gated_had_384_strict", "gated_had_quant_i8g", "quant", RANGE(packed_384_constraints), r3dh_qwen_ops_gated_had_quant_i8g, r3d_shape_gated_had_quant_i8g),
    kernel("model_gate_had_strict", "gate_had_quant_i8g", "quant", RANGE(gate_constraints), r3dh_qwen_ops_gate_had_quant_i8g, gate_shape),
    kernel("model_router_gemm_strict", "gemm_nt", "gemm", RANGE(router_constraints), r3dh_qwen_ops_router_gemm, r3d_shape_gemm_nt)
};
}

extern "C" int r3d_model_extra_schema_count() { return COUNT(schemas); }
extern "C" const RadOpSchema* r3d_model_extra_schema_at(int index) {
    return index >= 0 && index < COUNT(schemas) ? &schemas[index] : nullptr;
}
extern "C" int r3d_model_extra_kernel_count() { return COUNT(kernels) + r3dh_moe_mixed_kernel_count(); }
extern "C" const RadKernelInfo* r3d_model_extra_kernel_at(int index) {
    if (index < 0) return nullptr;
    return index < COUNT(kernels) ? &kernels[index] : r3dh_moe_mixed_kernel_at(index - COUNT(kernels));
}
