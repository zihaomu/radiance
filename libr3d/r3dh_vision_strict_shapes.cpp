#include "r3dh_vision_strict.h"
#include <cstring>
extern "C" int r3dh_vision_strict_geometry(int64_t columns, int64_t depth) {
    return (columns == 1152 && (depth == 1536 || depth == 1152 || depth == 4304)) ||
        (columns == 3456 && depth == 1152) || (columns == 4304 && depth == 1152) ||
        ((columns == 4608 || columns == 2560) && depth == 4608);
}
extern "C" int r3dh_vision_strict_shape(const char* operation, const RadParam* params,
                                      int count, int operand, RadOpdDesc* output) {
    if (!operation || !output || count < 0 || (count && !params)) return RAD_E_INVAL;
    const int64_t rows = rad_param_geti(params, count, "M", 0);
    const char* dtype = rad_param_gets(params, count, "dtype", "bf16");
    if (std::strcmp(dtype, "bf16")) return RAD_E_DTYPE;
    if (rows < 1 || rows > 256 || operand < 0) return RAD_E_SHAPE;
    RadOpdDesc result{};
    result.dtype = RAD_BF16; result.fill = RAD_FILL_NORMAL; result.idx_const = -1;
    if (!std::strcmp(operation, "gemm_nt") || !std::strcmp(operation, "gemm_nt_bias")) {
        const bool bias = !std::strcmp(operation, "gemm_nt_bias");
        const int64_t columns = rad_param_geti(params, count, "N", 0), depth = rad_param_geti(params, count, "K", 0);
        if (!r3dh_vision_strict_geometry(columns, depth)) return RAD_E_UNSUPPORTED;
        if (operand >= (bias ? 5 : 3)) return RAD_E_SHAPE;
        result.rank = 2;
        result.shape[0] = operand == 1 ? columns : rows;
        result.shape[1] = operand < 2 ? depth : columns;
        if (bias && operand == 2) { result.rank = 1; result.shape[0] = columns; }
    } else if (!std::strcmp(operation, "attn_dense")) {
        if (operand > 4) return RAD_E_SHAPE;
        if (rad_param_geti(params, count, "head_dim", 0) != 72 ||
            rad_param_geti(params, count, "gqa", 1) != 1 || rad_param_geti(params, count, "causal", 0) ||
            rad_param_geti(params, count, "window", 0) || rad_param_geti(params, count, "window_right", 0) ||
            rad_param_geti(params, count, "block", 0) || rad_param_geti(params, count, "cu_div", 1) != 1)
            return RAD_E_UNSUPPORTED;
        result.rank = 3; result.shape[0] = rows; result.shape[1] = 16; result.shape[2] = 72;
        if (operand == 3) {
            result.rank = 1; result.dtype = RAD_I32; result.shape[0] = 2;
            result.fill = RAD_FILL_INDEX; result.flags = RAD_OPD_F_IDX_CU;
        }
    } else return RAD_E_UNSUPPORTED;
    *output = result;
    return RAD_OK;
}
