#include "r3dh_i8.h"
extern "C" int r3d_i8_gated_launch(const RadArgs*, RadStream);
extern "C" int64_t r3d_i8_gated_scratch(const RadArgs*);
extern "C" int r3d_rad_layout_iqr(const RadParam*, int, int, const RadEncoding*,
    const int*, const RadTensor*, int, RadLayout*);
extern "C" int r3d_rad_relayout_iqr(const RadParam*, int, int, const RadEncoding*,
    const int*, const RadTensor*, int, void*, int64_t);
extern "C" int r3d_rad_unrelayout_iqr(const RadParam*, int, int, const RadEncoding*,
    const int*, const void*, int64_t, const RadTensor*, int);
#include "r3dh_mtp_ple.h"
extern "C" int r3d_rad_logit_rerank_strict(const RadArgs*, RadStream);
extern "C" int r3d_layout_logits_head(const RadParam*, int, int, const RadEncoding*,
    const int*, const RadTensor*, int, RadLayout*);
extern "C" int r3d_relayout_logits_head(const RadParam*, int, int, const RadEncoding*,
    const int*, const RadTensor*, int, void*, int64_t);

#include <cstring>
#include <vector>

namespace {

#define COUNT(values) static_cast<int>(sizeof(values) / sizeof(values[0]))
#define RANGE(values) values, COUNT(values)
#define PI(key) {key, RAD_P_INT, RAD_REQUIRED, RAD_PROLE_NONE}
#define PF(key) {key, RAD_P_F64, RAD_REQUIRED, RAD_PROLE_NONE}
#define PS(key) {key, RAD_P_STR, RAD_REQUIRED, RAD_PROLE_NONE}
#define OI(key) {key, RAD_P_INT, RAD_OPTIONAL, RAD_PROLE_NONE}
#define OF(key) {key, RAD_P_F64, RAD_OPTIONAL, RAD_PROLE_NONE}
#define OS(key) {key, RAD_P_STR, RAD_OPTIONAL, RAD_PROLE_NONE}
#define INPUT(key) {key, RAD_OPD_IN, 0}
#define OUTPUT(key) {key, RAD_OPD_OUT, 0}
#define WEIGHT(key) {key, RAD_OPD_WEIGHT, 0}
#define INOUT(key) {key, RAD_OPD_INOUT, 0}
#define INPUT_OPTIONAL(key) {key, RAD_OPD_IN, 1}
#define OUTPUT_OPTIONAL(key) {key, RAD_OPD_OUT, 1}
#define WEIGHT_OPTIONAL(key) {key, RAD_OPD_WEIGHT, 1}
#define EQ(key, value) {key, RAD_C_EQ, value, nullptr}
#define GE(key, value) {key, RAD_C_GE, value, nullptr}
#define LE(key, value) {key, RAD_C_LE, value, nullptr}
#define IN(key, value) {key, RAD_C_IN, 0, value}
#define TOKENS GE("M", 1), LE("M", 32)
#define PREFILL_TOKENS GE("M", 1), LE("M", 2048)

const RadParamSpec mtp_params[]{PI("M"), PI("n"), PI("hc"), PF("eps"),
    PS("dtype"), OF("wadd"), OI("ngroup"), OS("fc")};
const RadOperandSpec mtp_operands[]{INPUT("h"), INPUT("e"), WEIGHT("w_h"),
    WEIGHT("w_e"), WEIGHT("fc_h"), WEIGHT("fc_e"), OUTPUT("x")};
const RadParamSpec gate_params[]{PI("M"), PI("n"), PI("hc"), PF("eps"),
    PS("dtype"), OF("wadd"), OF("gate_eps")};
const RadOperandSpec gate_operands[]{INPUT("k"), INPUT("q"), INPUT("v"),
    WEIGHT("w_key"), WEIGHT("w_query"), WEIGHT("w_conv"), OUTPUT("gv"), OUTPUT("gvn")};
const RadParamSpec grid_params[]{PI("M"), PI("n"), PI("side"), PS("dtype")};
const RadOperandSpec grid_operands[]{INOUT("x"), WEIGHT("table"), INPUT("coord")};
const RadParamSpec embed_params[]{PI("M"), PI("n_embd"), PI("n_vocab"), PS("dtype"),
    OI("vocab_offset"), OF("wscale")};
const RadOperandSpec embed_operands[]{INPUT("tokens"), WEIGHT("wte"),
    WEIGHT_OPTIONAL("scale"), OUTPUT("x"), INPUT_OPTIONAL("ahead")};
const RadParamSpec hint_params[]{PI("M"), PI("heads"), PI("ngram"), PI("eos"), PI("hist")};
const RadOperandSpec hint_operands[]{INPUT("tok"), INPUT("pos"), INPUT("cu"),
    WEIGHT("mult"), WEIGHT("vocab_sizes"), WEIGHT("offsets"), INPUT_OPTIONAL("cache_idx"),
    INOUT("tail"), OUTPUT("ids")};
const RadParamSpec rerank_params[]{PI("M"), PI("R"), PI("n_vocab"), PI("n_embd"),
    OI("vocab_off"), PS("dtype")};
const RadOperandSpec rerank_operands[]{INPUT("x"), WEIGHT("head"), INPUT("idx_in"),
    OUTPUT("idx"), OUTPUT("val"), OUTPUT_OPTIONAL("pairs")};

const RadOpSchema schemas[]{
    {"mtp_enter", RANGE(mtp_params), RANGE(mtp_operands), "Wide-stream MTP entry"},
    {"ple_gate", RANGE(gate_params), RANGE(gate_operands), "Grouped PLE signed-square-root gate"},
    {"grid_embed", RANGE(grid_params), RANGE(grid_operands), "Align-corners grid embedding"},
    {"embed_lookup_q", RANGE(embed_params), RANGE(embed_operands), "Quantized embedding gather"},
    {"ngram_hint", RANGE(hint_params), RANGE(hint_operands), "Draft-tail n-gram hint"},
    {"logit_rerank", RANGE(rerank_params), RANGE(rerank_operands), "Exact candidate logit reranking"}
};

int unsupported_shape(const RadParam*, int, int, RadOpdDesc*) {
    return RAD_E_UNSUPPORTED;
}

int rerank_shape(const RadParam* params, int count, int operand, RadOpdDesc* output) {
    if (!params || !output || operand < 0 || operand > 5) return RAD_E_INVAL;
    const int64_t rows = rad_param_getdim(params, count, "M", 0);
    const int64_t offset = rad_param_geti(params, count, "vocab_off", 0);
    if (rows < 1 || rows > 32 || rad_param_geti(params, count, "R", 0) != 8 ||
        rad_param_geti(params, count, "n_embd", 0) != 2560 ||
        rad_param_geti(params, count, "n_vocab", 0) != 124160 || (offset != 0 && offset != 124160)) return RAD_E_SHAPE;
    *output = {};
    output->rank = 2;
    output->dtype = operand <= 1 ? RAD_BF16 : operand == 2 || operand == 3 ? RAD_I32 : RAD_F32;
    output->shape[0] = operand == 1 ? 124160 : rows;
    output->shape[1] = operand == 0 || operand == 1 ? 2560 : operand == 5 ? 16 : 8;
    output->fill = RAD_FILL_NORMAL;
    output->idx_const = -1;
    if (operand == 2) { output->fill = RAD_FILL_INDEX; output->idx_max = offset + 124160; }
    return RAD_OK;
}

template <int operation, int maximum_rows = 2048>
int shape(const RadParam* params, int count, int operand, RadOpdDesc* output) {
    if (!params || count <= 0 || !output) return RAD_E_INVAL;
    const int64_t rows = rad_param_getdim(params, count, "M", 0);
    if (rows < 1 || rows > maximum_rows) return RAD_E_UNSUPPORTED;
    std::vector<RadParam> resolved(params, params + count);
    for (auto& param : resolved) {
        if (param.kind == RAD_P_RANGE) {
            if (param.ival < 1 || param.ihi < param.ival) return RAD_E_SHAPE;
            param.kind = RAD_P_INT;
            param.ival = param.ihi;
            param.ihi = 0;
        }
    }
    return r3dh_mtp_ple_shape(schemas[operation].op, resolved.data(), count, operand, output);
}

int gain_layout(const RadParam* params, int count, int operand, const RadEncoding* encoding,
                const int* selection, const RadTensor* planes, int plane_count, RadLayout* output) {
    if (!encoding || !selection || !planes || !output || plane_count != 1)
        return RAD_E_UNSUPPORTED;
    if (!rad_enc_is(encoding, "plain") || encoding->n_planes != 1 || encoding->transform[0]
        || encoding->plane[0].dtype != planes[0].dtype
        || encoding->plane[0].block[0] != 1 || encoding->plane[0].block[1] != 1
        || selection[0] != 0 || planes[0].rank != 1 || !rad_tensor_is_contiguous(planes))
        return RAD_E_UNSUPPORTED;
    if (planes[0].dtype != RAD_F32 && planes[0].dtype != RAD_BF16)
        return RAD_E_DTYPE;
    const int64_t width = rad_param_geti(params, count, "n", 0);
    const int64_t streams = rad_param_geti(params, count, "hc", 0);
    if (width != 2560 || streams != 4 || planes[0].shape[0] != (operand == 3 ? width : width * streams))
        return RAD_E_SHAPE;
    *output = {};
    output->tag = "r3d.mtp.gain.f32";
    output->dtype = RAD_F32;
    output->rank = 1;
    output->shape[0] = planes[0].shape[0];
    output->align = 4;
    output->bytes = 4 * output->shape[0];
    return RAD_OK;
}

int gain_relayout(const RadParam* params, int count, int operand, const RadEncoding* encoding,
                  const int* selection, const RadTensor* planes, int plane_count,
                  void* destination, int64_t bytes) {
    RadLayout layout{};
    const int status = gain_layout(params, count, operand, encoding, selection, planes, plane_count, &layout);
    if (status != RAD_OK) return status;
    if (bytes != layout.bytes) return RAD_E_SHAPE;
    if (!destination || !planes[0].data) return RAD_E_INVAL;
    if (planes[0].dtype == RAD_F32) {
        std::memcpy(destination, planes[0].data, static_cast<size_t>(bytes));
    } else {
        auto* output = static_cast<float*>(destination);
        const auto* input = static_cast<const uint16_t*>(planes[0].data);
        for (int64_t index = 0; index < layout.shape[0]; ++index) {
            const uint32_t bits = uint32_t(input[index]) << 16;
            std::memcpy(output + index, &bits, sizeof(bits));
        }
    }
    return RAD_OK;
}

int mtp_layout(const RadParam* params, int count, int operand, const RadEncoding* encoding,
               const int* selection, const RadTensor* planes, int plane_count, RadLayout* output) {
    if (operand == 2 || operand == 3)
        return gain_layout(params, count, operand, encoding, selection, planes, plane_count, output);
    return r3dh_mtp_ple_layout_mtp_e4m3(params, count, operand, encoding, selection, planes, plane_count, output);
}

int mtp_relayout(const RadParam* params, int count, int operand, const RadEncoding* encoding,
                 const int* selection, const RadTensor* planes, int plane_count,
                 void* destination, int64_t bytes) {
    if (operand == 2 || operand == 3)
        return gain_relayout(params, count, operand, encoding, selection, planes, plane_count, destination, bytes);
    return r3dh_mtp_ple_relayout_mtp_e4m3(params, count, operand, encoding, selection, planes, plane_count, destination, bytes);
}

int gate_layout(const RadParam* params, int count, int operand, const RadEncoding* encoding,
                const int* selection, const RadTensor* planes, int plane_count, RadLayout* output) {
    return operand >= 3 && operand <= 5
        ? gain_layout(params, count, 2, encoding, selection, planes, plane_count, output)
        : RAD_E_UNSUPPORTED;
}

int gate_relayout(const RadParam* params, int count, int operand, const RadEncoding* encoding,
                  const int* selection, const RadTensor* planes, int plane_count,
                  void* destination, int64_t bytes) {
    return operand >= 3 && operand <= 5
        ? gain_relayout(params, count, 2, encoding, selection, planes, plane_count, destination, bytes)
        : RAD_E_UNSUPPORTED;
}

int gain_unrelayout(const RadParam* params, int count, int operand, const RadEncoding* encoding,
                    const int* selection, const void* source, int64_t bytes,
                    const RadTensor* planes, int plane_count) {
    RadLayout layout{};
    const int status = gain_layout(params, count, operand, encoding, selection, planes, plane_count, &layout);
    if (status != RAD_OK) return status;
    if (bytes != layout.bytes) return RAD_E_SHAPE;
    if (!source || !planes[0].data) return RAD_E_INVAL;
    if (planes[0].dtype == RAD_F32) {
        std::memcpy(planes[0].data, source, static_cast<size_t>(bytes));
    } else {
        auto* output = static_cast<uint16_t*>(planes[0].data);
        const auto* input = static_cast<const float*>(source);
        for (int64_t index = 0; index < layout.shape[0]; ++index) {
            output[index] = rad_f32_to_bf16(input[index]);
        }
    }
    return RAD_OK;
}

int mtp_unrelayout(const RadParam* params, int count, int operand, const RadEncoding* encoding,
                   const int* selection, const void* source, int64_t bytes,
                   const RadTensor* planes, int plane_count) {
    if (operand == 2 || operand == 3)
        return gain_unrelayout(params, count, operand, encoding, selection, source, bytes, planes, plane_count);
    return r3dh_mtp_ple_unrelayout_mtp_e4m3(params, count, operand, encoding, selection, source, bytes, planes, plane_count);
}

int gate_unrelayout(const RadParam* params, int count, int operand, const RadEncoding* encoding,
                    const int* selection, const void* source, int64_t bytes,
                    const RadTensor* planes, int plane_count) {
    return operand >= 3 && operand <= 5
        ? gain_unrelayout(params, count, 2, encoding, selection, source, bytes, planes, plane_count)
        : RAD_E_UNSUPPORTED;
}

int i8_gated_shape(const RadParam* params, int count, int operand, RadOpdDesc* output) {
    if (!params || !output || operand < 0 || operand > 6) return RAD_E_INVAL;
    const int64_t rows = rad_param_getdim(params, count, "M", 0);
    const int64_t columns = rad_param_geti(params, count, "N", 0);
    if (std::strcmp(rad_param_gets(params, count, "dtype", ""), "i8a8h")) return RAD_E_DTYPE;
    if (std::strcmp(rad_param_gets(params, count, "act", ""), "silu")) return RAD_E_UNSUPPORTED;
    if (rows < 1 || rows > 16 || (columns != 512 && columns != 768)
        || rad_param_geti(params, count, "K", 0) != 2560
        || rad_param_geti(params, count, "group", 128) != 128) return RAD_E_SHAPE;
    *output = {};
    output->rank = 2;
    output->dtype = operand == 1 || operand == 5 ? RAD_F32 : operand == 3 || operand == 6 ? RAD_BF16 : RAD_I8;
    output->shape[0] = operand == 2 || operand == 3 ? columns : rows;
    output->shape[1] = operand == 0 || operand == 2 ? 2560 : operand == 1 || operand == 3 ? 20
        : operand == 5 ? columns / 256 : columns / 2;
    output->fill = RAD_FILL_NORMAL;
    output->idx_const = -1;
    return RAD_OK;
}

int i8_gated_layout(const RadParam* params, int count, int operand, const RadEncoding* encoding,
                    const int* selection, const RadTensor* planes, int plane_count, RadLayout* output) {
    const int status = r3d_rad_layout_iqr(params, count, operand, encoding, selection, planes, plane_count, output);
    return status == RAD_E_DTYPE || status == RAD_E_UNSUPPORTED
        ? r3dh_i8_rad_layout(params, count, operand, encoding, selection, planes, plane_count, output) : status;
}

int i8_gated_relayout(const RadParam* params, int count, int operand, const RadEncoding* encoding,
                      const int* selection, const RadTensor* planes, int plane_count,
                      void* destination, int64_t bytes) {
    RadLayout layout{};
    const int status = r3d_rad_layout_iqr(params, count, operand, encoding, selection, planes, plane_count, &layout);
    if (status == RAD_OK)
        return r3d_rad_relayout_iqr(params, count, operand, encoding, selection, planes, plane_count, destination, bytes);
    if (status != RAD_E_DTYPE && status != RAD_E_UNSUPPORTED) return status;
    return r3dh_i8_rad_relayout(params, count, operand, encoding, selection, planes, plane_count, destination, bytes);
}

int i8_gated_unrelayout(const RadParam* params, int count, int operand, const RadEncoding* encoding,
                        const int* selection, const void* source, int64_t bytes,
                        const RadTensor* planes, int plane_count) {
    RadLayout layout{};
    const int status = r3d_rad_layout_iqr(params, count, operand, encoding, selection, planes, plane_count, &layout);
    if (status == RAD_OK)
        return r3d_rad_unrelayout_iqr(params, count, operand, encoding, selection, source, bytes, planes, plane_count);
    if (status != RAD_E_DTYPE && status != RAD_E_UNSUPPORTED) return status;
    return r3dh_i8_rad_unrelayout(params, count, operand, encoding, selection, source, bytes, planes, plane_count);
}

const RadConstraint i8_gated_constraints[]{GE("M", 1), LE("M", 16), IN("N", "512 768"),
    EQ("K", 2560), EQ("group", 128), IN("dtype", "i8a8h"), IN("act", "silu")};

const RadConstraint mtp_constraints[]{PREFILL_TOKENS, EQ("n", 2560), EQ("hc", 4), EQ("ngroup", 4),
    IN("dtype", "bf16"), IN("fc", "e4m3")};
const RadConstraint gate_constraints[]{PREFILL_TOKENS, EQ("n", 2560), EQ("hc", 4), IN("dtype", "bf16")};
const RadConstraint grid_constraints[]{TOKENS, EQ("n", 128), EQ("side", 4), IN("dtype", "bf16")};
const RadConstraint grid_prefill_constraints[]{PREFILL_TOKENS, EQ("n", 1152), EQ("side", 48), IN("dtype", "bf16")};
const RadConstraint merge_constraints[]{TOKENS, EQ("R", 8), EQ("world_size", 2), IN("dtype", "bf16")};
const RadConstraint ids_constraints[]{GE("M", 1), LE("M", 2048), EQ("heads", 16), EQ("ngram", 3)};
const RadConstraint hint_constraints[]{GE("M", 1), LE("M", 8), EQ("heads", 16),
    EQ("ngram", 3), IN("hist", "0 1")};
const RadConstraint rerank_constraints[]{GE("M", 1), LE("M", 32), EQ("R", 8),
    EQ("n_embd", 2560), EQ("n_vocab", 124160), IN("vocab_off", "0 124160"), IN("dtype", "bf16")};

RadKernelInfo kernel(const char* name, const char* operation, const char* family,
                     const char* dtypes, const RadConstraint* constraints, int count,
                     RadLaunchFn launch, RadShapeFn operand_shape) {
    RadKernelInfo result{};
    result.name = name;
    result.op = operation;
    result.family = family;
    result.computes = "Strict gfx1100 native MTP/PLE port";
    result.shape = "Qualified geometry and batch limits are explicit in constraints";
    result.dtypes = dtypes;
    result.domain = RAD_DOMAIN_DEVICE;
    result.priority = 100;
    result.constraints = constraints;
    result.n_constraints = count;
    result.launch = launch;
    result.opd_shape = operand_shape;
    return result;
}

const auto& kernels() {
    static const RadKernelInfo rows[]{
        [] {
            auto result = kernel("r3d_gemm_nt_q_gated_i8a8h_strict", "gemm_nt_q_gated", "gemm",
                "private IQR or packed I8 weights; I8/F32 activations; BF16 SiLU/FWHT128", RANGE(i8_gated_constraints),
                r3d_i8_gated_launch, i8_gated_shape);
            result.scratch = r3d_i8_gated_scratch;
            result.layout = i8_gated_layout;
            result.relayout = i8_gated_relayout;
            result.unrelayout = i8_gated_unrelayout;
            return result;
        }(),
        [] {
            auto result = kernel("r3d_mtp_enter_e4m3", "mtp_enter", "norm",
                "bf16; F32 gains; r4d.mtp.e4m3.g128 FC", RANGE(mtp_constraints),
                r3dh_mtp_ple_mtp_enter, shape<0>);
            result.scratch = r3dh_mtp_ple_mtp_enter_scratch;
            result.layout = mtp_layout;
            result.relayout = mtp_relayout;
            result.unrelayout = mtp_unrelayout;
            return result;
        }(),
        [] {
            auto result = kernel("r3d_ple_gate_bf16", "ple_gate", "norm",
                "bf16 activations; F32 gains", RANGE(gate_constraints), r3dh_mtp_ple_ple_gate, shape<1>);
            result.layout = gate_layout;
            result.relayout = gate_relayout;
            result.unrelayout = gate_unrelayout;
            return result;
        }(),
        kernel("r3d_grid_embed_bf16", "grid_embed", "elem", "bf16; i32 coordinates",
            RANGE(grid_constraints), r3dh_mtp_ple_grid_embed, shape<2, 32>),
        kernel("r3d_row_topk_merge_f32", "row_topk_merge", "moe", "f32 pairs/values; i32 indices",
            RANGE(merge_constraints), r3dh_mtp_ple_rowtopkmerge, unsupported_shape),
        kernel("r3d_grid_embed_1152_s48_bf16", "grid_embed", "elem", "bf16; i32 coordinates",
            RANGE(grid_prefill_constraints), r3dh_mtp_ple_grid_embed, shape<2>),
        kernel("r3d_ngram_ids_i32", "ngram_ids", "ple", "i32 tokens, device state and ids; i64 hash tables",
            RANGE(ids_constraints), r3dh_mtp_ple_rad_ngram_ids, nullptr),
        kernel("r3d_ngram_hint_i32", "ngram_hint", "ple", "i32 tokens and draft tail; i64 hash tables",
            RANGE(hint_constraints), r3dh_mtp_ple_ngram_hint, shape<4, 8>),
        [] {
            auto result = kernel("r3d_logit_rerank_joined_strict", "logit_rerank", "moe",
                "bf16; certified joined head; i32 candidates; f32 scores and pairs",
                RANGE(rerank_constraints), r3d_rad_logit_rerank_strict, rerank_shape);
            result.layout = r3d_layout_logits_head;
            result.relayout = r3d_relayout_logits_head;
            return result;
        }()
    };
    return rows;
}

}

extern "C" int r3d_mtp_schema_count() { return COUNT(schemas); }
extern "C" const RadOpSchema* r3d_mtp_schema_at(int index) {
    return index >= 0 && index < COUNT(schemas) ? schemas + index : nullptr;
}
extern "C" int r3d_mtp_kernel_count() { return COUNT(kernels()); }
extern "C" const RadKernelInfo* r3d_mtp_kernel_at(int index) {
    return index >= 0 && index < r3d_mtp_kernel_count() ? kernels() + index : nullptr;
}
