#include <hip/hip_runtime.h>
#include "r3dh_vision_rope.h"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <limits>

namespace {
struct Config { int64_t rows; float scale; float frequency[18]; };
int config(const RadParam* params, int count, Config* output, bool ranged) {
    if (!params || count < 1 || !output) return RAD_E_INVAL;
    int64_t rows = 0, head = 0, heads = 0, kv = 16, rotary = 72;
    double theta = 10000, scale = 1;
    const char* mode = nullptr;
    const char* sections = nullptr;
    for (int index = 0; index < count; ++index) {
        const auto& param = params[index];
        if (!param.key) return RAD_E_INVAL;
        for (int previous = 0; previous < index; ++previous)
            if (!std::strcmp(param.key, params[previous].key)) return RAD_E_INVAL;
        auto is = [&](const char* key) { return !std::strcmp(param.key, key); };
        if (is("M") || is("head_dim") || is("n_head") || is("n_head_kv") || is("rotary_dim")) {
            int64_t value = param.ival;
            if (is("M") && ranged && param.kind == RAD_P_RANGE) {
                if (param.ival < 1 || param.ihi < param.ival) return RAD_E_SHAPE;
                value = param.ihi;
            } else if (param.kind != RAD_P_INT) return RAD_E_INVAL;
            if (is("M")) rows = value;
            if (is("head_dim")) head = value;
            if (is("n_head")) heads = value;
            if (is("n_head_kv")) kv = value;
            if (is("rotary_dim")) rotary = value;
        } else if (is("theta") || is("scale") || is("yarn_factor") || is("llama3_factor")) {
            if (param.kind != RAD_P_F64 || !std::isfinite(param.dval)) return RAD_E_INVAL;
            if (is("theta")) theta = param.dval;
            else if (is("scale")) scale = param.dval;
            else if (param.dval != 0) return RAD_E_UNSUPPORTED;
        } else if (is("mode") || is("sections") || is("dtype") || is("ladder")) {
            if (param.kind != RAD_P_STR || !param.sval) return RAD_E_INVAL;
            if (is("mode")) mode = param.sval;
            if (is("sections")) sections = param.sval;
            if (is("dtype") && std::strcmp(param.sval, "bf16")) return RAD_E_DTYPE;
            if (is("ladder") && std::strcmp(param.sval, "rotary")) return RAD_E_UNSUPPORTED;
        } else return RAD_E_UNSUPPORTED;
    }
    if (rows < 1 || rows > 256) return RAD_E_SHAPE;
    if (head != 72 || heads != 16 || kv != 16 || rotary != 72 || !mode
        || std::strcmp(mode, "axial") || !sections) return RAD_E_UNSUPPORTED;
    char* end = nullptr;
    const long first = std::strtol(sections, &end, 10);
    if (end == sections || first != 18) return RAD_E_UNSUPPORTED;
    const char* next = end;
    const long second = std::strtol(next, &end, 10);
    if (end == next || second != 18) return RAD_E_UNSUPPORTED;
    while (*end == ' ' || *end == '\t' || *end == '\n' || *end == '\r') ++end;
    if (*end) return RAD_E_UNSUPPORTED;
    const float theta32 = static_cast<float>(theta), scale32 = static_cast<float>(scale);
    if (!std::isfinite(theta32) || theta32 <= 0 || !std::isfinite(scale32) || scale32 <= 0)
        return RAD_E_INVAL;
    output->rows = rows;
    output->scale = scale32;
    for (int pair = 0; pair < 18; ++pair) {
        const float exponent = static_cast<float>(2 * pair) / 36.0f;
        output->frequency[pair] = 1.0f / static_cast<float>(std::pow(double(theta32), double(exponent)));
        if (!std::isfinite(output->frequency[pair])) return RAD_E_INVAL;
    }
    return RAD_OK;
}
bool matrix(const RadTensor& tensor, int64_t rows, int width, bool view) {
    return tensor.data && tensor.dtype == RAD_BF16 && tensor.rank == 2
        && tensor.shape[0] == rows && tensor.shape[1] == width && tensor.stride[1] == 1
        && (tensor.stride[0] == width || (view && tensor.stride[0] == 3456))
        && reinterpret_cast<uintptr_t>(tensor.data) % 2 == 0;
}
bool positions(const RadTensor& tensor, int64_t rows) {
    return tensor.data && tensor.dtype == RAD_I32 && tensor.rank == 2
        && (tensor.shape[0] == 2 || tensor.shape[0] == 3)
        && tensor.shape[1] == rows && tensor.stride[1] == 1
        && tensor.stride[0] == rows && reinterpret_cast<uintptr_t>(tensor.data) % 4 == 0;
}
__device__ float widen(uint16_t value) { return __uint_as_float(uint32_t(value) << 16); }
__device__ uint16_t narrow(float value) {
    const uint32_t bits = __float_as_uint(value);
    if ((bits & 0x7fffffffu) > 0x7f800000u) return uint16_t((bits >> 16) | 0x40);
    return uint16_t((bits + 0x7fffu + ((bits >> 16) & 1)) >> 16);
}
__global__ void rotate(uint16_t* query, uint16_t* key, const int32_t* position,
                       int64_t query_pitch, int64_t key_pitch, Config options) {
    const int pair_index = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (pair_index >= options.rows * 32 * 36) return;
    const int pair = pair_index % 36;
    const int head_index = pair_index / 36 % 32;
    const int token = pair_index / (36 * 32);
    const float coordinate = float(position[(pair / 18) * options.rows + token]) / options.scale;
    const float angle = coordinate * options.frequency[pair % 18];
    const float cosine = float(cos(double(angle))), sine = float(sin(double(angle)));
    uint16_t* base = (head_index < 16 ? query : key)
        + token * (head_index < 16 ? query_pitch : key_pitch) + (head_index % 16) * 72;
    const float first = widen(base[pair]), second = widen(base[pair + 36]);
    const float first_cos = first * cosine, second_sin = second * sine;
    const float first_sin = first * sine, second_cos = second * cosine;
    base[pair] = narrow(first_cos - second_sin);
    base[pair + 36] = narrow(first_sin + second_cos);
}
int launch(const RadArgs* args, RadStream stream, bool fused) {
    if (!args || !args->t || args->n_t != (fused ? 2 : 3)) return RAD_E_INVAL;
    Config options{};
    const int status = config(args->p, args->n_p, &options, false);
    if (status != RAD_OK) return status;
    const auto& query = args->t[0];
    if (query.rank != 2 || query.shape[0] < 1 || query.shape[0] > options.rows) return RAD_E_SHAPE;
    options.rows = query.shape[0];
    const auto& key = args->t[fused ? 0 : 1];
    const auto& coordinate = args->t[fused ? 1 : 2];
    if (query.dtype != RAD_BF16 || key.dtype != RAD_BF16 || coordinate.dtype != RAD_I32)
        return RAD_E_DTYPE;
    if (!matrix(query, options.rows, fused ? 3456 : 1152, !fused)
        || (!fused && !matrix(key, options.rows, 1152, true))
        || !positions(coordinate, options.rows)) return RAD_E_SHAPE;
    if (!fused) {
        const uintptr_t query_address = reinterpret_cast<uintptr_t>(query.data);
        const uintptr_t key_address = reinterpret_cast<uintptr_t>(key.data);
        const uintptr_t query_bytes = ((options.rows - 1) * query.stride[0] + 1152) * 2;
        const uintptr_t key_bytes = ((options.rows - 1) * key.stride[0] + 1152) * 2;
        const bool overlap = query_address < key_address + key_bytes && key_address < query_address + query_bytes;
        const bool qkv_views = query.stride[0] == 3456 && key.stride[0] == 3456
            && key_address == query_address + 2304;
        if (overlap && !qkv_views) return RAD_E_SHAPE;
    }
    auto* query_data = static_cast<uint16_t*>(query.data);
    auto* key_data = fused ? query_data + 1152 : static_cast<uint16_t*>(key.data);
    hipLaunchKernelGGL(rotate, dim3((options.rows * 32 * 36 + 255) / 256), dim3(256), 0,
        reinterpret_cast<hipStream_t>(stream), query_data, key_data,
        static_cast<const int32_t*>(coordinate.data), query.stride[0], key.stride[0], options);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
const RadParamSpec params[]{
    {"M",RAD_P_INT,1,0},{"head_dim",RAD_P_INT,1,0},{"n_head",RAD_P_INT,1,0},
    {"n_head_kv",RAD_P_INT,0,0},{"theta",RAD_P_F64,0,0},{"scale",RAD_P_F64,0,0},
    {"mode",RAD_P_STR,1,0},{"rotary_dim",RAD_P_INT,0,0},{"sections",RAD_P_STR,1,0},
    {"dtype",RAD_P_STR,0,0},{"ladder",RAD_P_STR,0,0},
    {"yarn_factor",RAD_P_F64,0,0},{"llama3_factor",RAD_P_F64,0,0}};
const RadOperandSpec operands[]{{"qkv",RAD_OPD_INOUT,0},{"positions",RAD_OPD_IN,0}};
const RadOpSchema schema{"rope",params,13,operands,2,"Strict BF16 axial 18+18 vision RoPE72"};
const RadConstraint constraints[]{RAD_CGE("M",1),RAD_CLE("M",256),RAD_CEQ("head_dim",72),
    RAD_CEQ("n_head",16),RAD_CEQ("n_head_kv",16),RAD_CEQ("rotary_dim",72),
    RAD_CIN("mode","axial")};
}
extern "C" int r3dh_vision_rope(const RadArgs* args, RadStream stream) { return launch(args, stream, true); }
extern "C" int r3dh_vision_rope_qk(const RadArgs* args, RadStream stream) { return launch(args, stream, false); }
extern "C" int r3dh_vision_rope_shape(const RadParam* params, int count, int operand, RadOpdDesc* output) {
    if (!output) return RAD_E_INVAL;
    Config options{};
    const int status = config(params, count, &options, true);
    if (status != RAD_OK) return status;
    if (operand < 0 || operand > 1) return RAD_E_INVAL;
    *output = {};
    output->rank = 2;
    output->dtype = operand ? RAD_I32 : RAD_BF16;
    output->shape[0] = operand ? 2 : options.rows;
    output->shape[1] = operand ? options.rows : 3456;
    if (operand) { output->fill = RAD_FILL_INDEX; output->idx_max = 48; output->idx_const = -1; }
    return RAD_OK;
}
extern "C" const RadOpSchema* r3dh_vision_rope_schema() { return &schema; }
extern "C" const RadKernelInfo* r3dh_vision_rope_kernel() {
    static const RadKernelInfo row = [] {
        RadKernelInfo result{};
        result.name = "r3dh_vision_axial_rope72"; result.op = "rope"; result.family = "attn";
        result.computes = "F32 separate products, double trig rounded to F32, BF16 RNE; V unchanged";
        result.shape = "M1..256, Q/K16x72, rotary72, axial18+18, fused QKV pitch3456";
        result.dtypes = "bf16"; result.domain = RAD_DOMAIN_DEVICE; result.priority = 200;
        result.constraints = constraints; result.n_constraints = 7;
        result.launch = r3dh_vision_rope; result.opd_shape = r3dh_vision_rope_shape;
        return result;
    }();
    return &row;
}
