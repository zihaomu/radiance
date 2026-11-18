#include "r3dh_mtp_ple.h"
#include <cstring>

namespace {
RadOpdDesc describe(uint32_t type, int64_t rows, int64_t columns = 0) {
    RadOpdDesc result{};
    result.dtype = type; result.rank = columns ? 2 : 1;
    result.shape[0] = rows; result.shape[1] = columns;
    result.fill = RAD_FILL_NORMAL; result.idx_const = -1;
    return result;
}
}
extern "C" int r3dh_mtp_ple_shape(const char* operation, const RadParam* params, int count,
                                    int operand, RadOpdDesc* output) {
    if (!operation || !output || count < 0 || (count && !params)) return RAD_E_INVAL;
    if (operand < 0) return RAD_E_SHAPE;
    const int64_t rows = rad_param_geti(params, count, "M", 0), width = rad_param_geti(params, count, "n", 0),
        streams = rad_param_geti(params, count, "hc", 0);
    if (rows <= 0 || rows > 1048576) return RAD_E_SHAPE;
    if (!std::strcmp(operation, "mtp_enter") || !std::strcmp(operation, "ple_gate")) {
        if (width <= 0 || width > 4096 || streams < 2 || streams > 8) return RAD_E_SHAPE;
        const int64_t wide = width * streams;
        if (!std::strcmp(operation, "mtp_enter")) {
            if (operand > 6 || width % 128) return RAD_E_SHAPE;
            const int64_t groups = rad_param_geti(params, count, "ngroup", streams);
            if (groups != 1 && groups != streams) return RAD_E_SHAPE;
            const int64_t pitch = (width + 4 * (width / 128) + 15) & ~int64_t(15);
            const RadOpdDesc shapes[]{describe(RAD_BF16, rows, wide), describe(RAD_BF16, rows, width),
                describe(RAD_F32, wide), describe(RAD_F32, width), describe(RAD_F8E4M3, width, pitch),
                describe(RAD_F8E4M3, width, pitch), describe(RAD_BF16, rows, wide)};
            *output = shapes[operand];
        } else {
            if (operand > 7) return RAD_E_SHAPE;
            const RadOpdDesc shapes[]{describe(RAD_BF16, rows, wide), describe(RAD_BF16, rows, wide),
                describe(RAD_BF16, rows, width), describe(RAD_F32, wide), describe(RAD_F32, wide),
                describe(RAD_F32, wide), describe(RAD_BF16, rows, wide), describe(RAD_BF16, rows, wide)};
            *output = shapes[operand];
        }
        return RAD_OK;
    }
    if (!std::strcmp(operation, "grid_embed")) {
        const int64_t side = rad_param_geti(params, count, "side", 0);
        if (width <= 0 || width > 4096 || side < 1 || side > 4096 || operand > 2) return RAD_E_SHAPE;
        const RadOpdDesc shapes[]{describe(RAD_BF16, rows, width), describe(RAD_BF16, side * side, width), describe(RAD_I32, 4, rows)};
        *output = shapes[operand];
        if (operand == 2) { output->fill = RAD_FILL_INDEX; output->idx_max = side; }
        return RAD_OK;
    }
    if (!std::strcmp(operation, "ngram_hint")) {
        const int64_t heads = rad_param_geti(params, count, "heads", 0);
        const int64_t order = rad_param_geti(params, count, "ngram", 0);
        const int64_t history = rad_param_geti(params, count, "hist", -1);
        if (rows > 8 || heads != 16 || order != 3 || operand > 8
            || rad_param_geti(params, count, "eos", -1) < 0
            || (history != 0 && history != 1)) return RAD_E_SHAPE;
        const RadOpdDesc shapes[]{describe(RAD_I32, rows), describe(RAD_I32, rows),
            describe(RAD_I32, rows + 1), describe(RAD_I64, order),
            describe(RAD_I64, heads), describe(RAD_I64, heads), describe(RAD_I32, rows),
            describe(RAD_I32, rows, 2 * (order - 1)), describe(RAD_I32, rows, heads)};
        *output = shapes[operand];
        if (operand == 6) {
            output->fill = RAD_FILL_INDEX;
            output->idx_max = rows;
        }
        return RAD_OK;
    }
    if (!std::strcmp(operation, "embed_lookup_q")) {
        const int64_t embedding = rad_param_geti(params, count, "n_embd", 0), vocabulary = rad_param_geti(params, count, "n_vocab", 0);
        if (embedding < 1 || embedding > 4096 || vocabulary < 1 || operand > 3) return RAD_E_SHAPE;
        const RadOpdDesc shapes[]{describe(RAD_I32, rows), describe(RAD_F8E4M3, vocabulary, embedding),
            describe(RAD_BF16, 1), describe(RAD_BF16, rows, embedding)};
        *output = shapes[operand];
        if (!operand) { output->fill = RAD_FILL_INDEX; output->idx_max = vocabulary; }
        return RAD_OK;
    }
    return RAD_E_UNSUPPORTED;
}
