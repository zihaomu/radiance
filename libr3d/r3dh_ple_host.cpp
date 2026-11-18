#include "r3dh_ple_host.h"

#include <cstring>

namespace {
#define PARAM(key, type, needed) {key, type, needed, RAD_PROLE_NONE}
#define PI(key) PARAM(key, RAD_P_INT, RAD_REQUIRED)
#define PF(key) PARAM(key, RAD_P_F64, RAD_REQUIRED)
#define PS(key) PARAM(key, RAD_P_STR, RAD_REQUIRED)
#define OI(key) PARAM(key, RAD_P_INT, RAD_OPTIONAL)
#define OF(key) PARAM(key, RAD_P_F64, RAD_OPTIONAL)
#define INPUT(key) {key, RAD_OPD_IN, 0}
#define OPTIONAL(key) {key, RAD_OPD_IN, 1}
#define OUTPUT(key) {key, RAD_OPD_OUT, 0}
#define WEIGHT(key) {key, RAD_OPD_WEIGHT, 0}
#define RANGE(values) values, int(sizeof(values) / sizeof(values[0]))

const RadParamSpec ids_params[]{PI("M"), PI("heads"), PI("ngram"), PI("eos")};
const RadOperandSpec ids_operands[]{INPUT("tok"), {"state", RAD_OPD_INOUT, 0},
    INPUT("cu"), WEIGHT("mult"), WEIGHT("vocab_sizes"), WEIGHT("offsets"),
    OPTIONAL("cache_idx"), OPTIONAL("has_init"), OPTIONAL("num_accepted"),
    OUTPUT("ids"), {"ahead_ids", RAD_OPD_OUT, 1}};
const RadParamSpec embed_params[]{PI("M"), PI("n_embd"), PI("n_vocab"), PS("dtype"),
    OI("vocab_offset"), OF("wscale")};
const RadOperandSpec embed_operands[]{INPUT("tokens"), WEIGHT("wte"),
    {"scale", RAD_OPD_WEIGHT, 1}, OUTPUT("x"), OPTIONAL("ahead")};
const RadParamSpec gate_params[]{PI("M"), PI("n"), PI("hc"), PF("eps"), PS("dtype"),
    OF("wadd"), OF("gate_eps")};
const RadOperandSpec gate_operands[]{INPUT("k"), INPUT("q"), INPUT("v"), WEIGHT("w_key"),
    WEIGHT("w_query"), WEIGHT("w_conv"), OUTPUT("gv"), OUTPUT("gvn")};
const RadParamSpec gemm_params[]{PI("M"), PI("N"), PI("K"), PI("group"), PS("dtype"),
    OI("pb1"), OI("pb2"), OI("split_n"), OI("w_rb")};
const RadOperandSpec gemm_operands[]{INPUT("a"), OPTIONAL("a_scale"), WEIGHT("b"),
    WEIGHT("b_scale"), OUTPUT("y"), OPTIONAL("a_sum"), {"b_ref", RAD_OPD_WEIGHT, 1}};
const RadParamSpec quant_params[]{PI("M"), PI("n"), PI("group"), PS("dtype")};
const RadOperandSpec quant_operands[]{INPUT("x"), OUTPUT("q"), OUTPUT("scale"),
    {"asum", RAD_OPD_OUT, 1}};
const RadParamSpec hint_params[]{PI("M"), PI("heads"), PI("ngram"), PI("eos"), PI("hist")};
const RadOperandSpec hint_operands[]{INPUT("tok"), INPUT("pos"), INPUT("cu"),
    WEIGHT("mult"), WEIGHT("vocab_sizes"), WEIGHT("offsets"), OPTIONAL("cache_idx"),
    {"tail", RAD_OPD_INOUT, 0}, OUTPUT("ids")};
const RadOpSchema schemas[]{
    {"ngram_ids", RANGE(ids_params), RANGE(ids_operands), "HOST ngram ids with prompt read-ahead"},
    {"embed_lookup_q", RANGE(embed_params), RANGE(embed_operands), "HOST stored-table gather with read-ahead"},
    {"ple_gate", RANGE(gate_params), RANGE(gate_operands), "DEVICE PLE gate after HOST embedding handoff"},
    {"gemm_nt_q", RANGE(gemm_params), RANGE(gemm_operands), "Native group128 i8 PLE projections"},
    {"quant_act_i8g", RANGE(quant_params), RANGE(quant_operands), "Native group128 activation quantization"},
    {"had_quant_act_i8g", RANGE(quant_params), RANGE(quant_operands), "Native FWHT128 activation quantization"},
    {"ngram_hint", RANGE(hint_params), RANGE(hint_operands), "HOST draft ngram prefetch ids"}
};

bool host_row(const RadKernelInfo* row, const char* operation) {
    return row && row->name && row->op && row->launch
        && row->domain == RAD_DOMAIN_HOST && !std::strcmp(row->op, operation)
        && !std::strncmp(row->name, "avx_", 4);
}
}

extern "C" int r3dh_ple_host_bind(R3dhPleHost* output, int abi,
    const RadKernelInfo* ids, const RadKernelInfo* gather) {
    if (!output) return RAD_E_INVAL;
    *output = {};
    if (abi != RAD_ABI_VERSION) return RAD_E_UNSUPPORTED;
    if (!host_row(ids, "ngram_ids") || !host_row(gather, "embed_lookup_q")
        || !gather->init || !gather->layout) return RAD_E_UNSUPPORTED;
    output->rows[0] = *ids;
    output->rows[1] = *gather;
    output->rows[0].name = "r3dh_ple_host_ngram_ids";
    output->rows[1].name = "r3dh_ple_host_embed_lookup_q";
    return RAD_OK;
}

extern "C" int r3dh_ple_host_schema_count() { return int(sizeof(schemas) / sizeof(schemas[0])); }
extern "C" const RadOpSchema* r3dh_ple_host_schema_at(int index) {
    return index >= 0 && index < r3dh_ple_host_schema_count() ? schemas + index : nullptr;
}
extern "C" const RadKernelInfo* r3dh_ple_host_kernel_at(const R3dhPleHost* provider, int index) {
    if (provider && provider->rows[0].launch && provider->rows[1].launch && index == 2)
        return r3dh_ple_host_hint_kernel();
    return provider && index >= 0 && index < 2 && provider->rows[index].launch
        ? provider->rows + index : nullptr;
}

extern "C" int r3dh_ple_host_gain_layout(const RadParam* params, int count, int operand,
    const RadEncoding* encoding, const int* selection, const RadTensor* planes,
    int plane_count, RadLayout* output) {
    if (!output || count < 0 || (count && !params) || !encoding || !selection || !planes)
        return RAD_E_INVAL;
    if (operand < 3 || operand > 5 || plane_count != 1) return RAD_E_UNSUPPORTED;
    if (!std::memchr(encoding->scheme, 0, RAD_ENC_STR)
        || !std::memchr(encoding->transform, 0, RAD_ENC_STR)
        || std::strcmp(encoding->scheme, "plain") || encoding->transform[0]
        || encoding->reserved || encoding->n_planes != 1 || selection[0] != 0)
        return RAD_E_UNSUPPORTED;
    const RadEncPlane& encoded = encoding->plane[0];
    const RadTensor& plane = planes[0];
    if (encoded.kind != RAD_PLANE_TILED || encoded.block[0] != 1 || encoded.block[1] != 1
        || encoded.extent[0] || encoded.extent[1] || encoded.dtype != plane.dtype
        || (plane.dtype != RAD_F32 && plane.dtype != RAD_BF16)) return RAD_E_DTYPE;
    const int64_t width = rad_param_geti(params, count, "n", 0);
    const int64_t streams = rad_param_geti(params, count, "hc", 0);
    if (width < 1 || width > 4096 || streams < 2 || streams > 8
        || plane.rank != 1 || plane.shape[0] != width * streams) return RAD_E_SHAPE;
    if (!rad_tensor_is_contiguous(&plane)) return RAD_E_STRIDE;
    *output = {};
    output->tag = "r3dh.ple.gain.f32";
    output->dtype = RAD_F32;
    output->rank = 1;
    output->shape[0] = width * streams;
    output->align = 4;
    output->bytes = 4 * output->shape[0];
    return RAD_OK;
}

extern "C" int r3dh_ple_host_gain_relayout(const RadParam* params, int count, int operand,
    const RadEncoding* encoding, const int* selection, const RadTensor* planes,
    int plane_count, void* destination, int64_t bytes) {
    RadLayout layout{};
    const int status = r3dh_ple_host_gain_layout(params, count, operand, encoding, selection,
        planes, plane_count, &layout);
    if (status != RAD_OK) return status;
    if (bytes != layout.bytes) return RAD_E_SHAPE;
    if (!destination || !planes[0].data) return RAD_E_INVAL;
    if (planes[0].dtype == RAD_F32) std::memcpy(destination, planes[0].data, size_t(bytes));
    else {
        auto* output = static_cast<unsigned char*>(destination);
        const auto* input = static_cast<const unsigned char*>(planes[0].data);
        for (int64_t index = 0; index < layout.shape[0]; ++index) {
            uint16_t value;
            std::memcpy(&value, input + index * 2, 2);
            const uint32_t bits = uint32_t(value) << 16;
            std::memcpy(output + index * 4, &bits, 4);
        }
    }
    return RAD_OK;
}

extern "C" int r3dh_ple_host_gain_unrelayout(const RadParam* params, int count, int operand,
    const RadEncoding* encoding, const int* selection, const void* source, int64_t bytes,
    const RadTensor* planes, int plane_count) {
    RadLayout layout{};
    const int status = r3dh_ple_host_gain_layout(params, count, operand, encoding, selection,
        planes, plane_count, &layout);
    if (status != RAD_OK) return status;
    if (bytes != layout.bytes) return RAD_E_SHAPE;
    if (!source || !planes[0].data) return RAD_E_INVAL;
    if (planes[0].dtype == RAD_F32) std::memcpy(planes[0].data, source, size_t(bytes));
    else {
        auto* output = static_cast<unsigned char*>(planes[0].data);
        const auto* input = static_cast<const unsigned char*>(source);
        for (int64_t index = 0; index < layout.shape[0]; ++index) {
            float value;
            std::memcpy(&value, input + index * 4, 4);
            const uint16_t bits = rad_f32_to_bf16(value);
            std::memcpy(output + index * 2, &bits, 2);
        }
    }
    return RAD_OK;
}
