#include "r3dh_hc_api.h"
#include "r3dh_hc_args.h"
#include <limits>

namespace {
RadOpdDesc describe(uint32_t dtype, int64_t rows, int64_t columns = 0) {
    RadOpdDesc result{};
    result.dtype = dtype; result.fill = RAD_FILL_NORMAL; result.idx_const = -1;
    result.rank = columns ? 2 : 1;
    result.shape[0] = rows; result.shape[1] = columns;
    return result;
}
RadOpdDesc absent() {
    RadOpdDesc result{}; result.flags = RAD_OPD_F_ABSENT; result.idx_const = -1;
    return result;
}
int shape(const RadParam* params, int count, int operand, RadOpdDesc* output, int operation) {
    if (!output) return RAD_E_INVAL;
    const int64_t rows = r3dh_hc_pgeti_or(params, count, "M", 0);
    const int64_t width = r3dh_hc_pgeti_or(params, count, "n", 0);
    const int64_t streams = r3dh_hc_pgeti_or(params, count, "hc", 0);
    const int64_t rank = r3dh_hc_pgeti_or(params, count, "lowrank", 0);
    if (rows <= 0 || width <= 0 || streams <= 1 || streams > 8 ||
        width > std::numeric_limits<int64_t>::max() / streams) return RAD_E_UNSUPPORTED;
    const int64_t channels = width * streams;
    if (operation == 0) {
        if (operand < 0 || operand >= 2) return RAD_E_SHAPE;
        *output = describe(RAD_BF16, rows, operand == 0 ? width : channels);
        return RAD_OK;
    }
    if (operation == 1) {
        if (operand < 0 || operand >= 3) return RAD_E_SHAPE;
        *output = describe(RAD_BF16, rows, operand == 0 ? width : operand == 1 ? streams : channels);
        if (operand == 1) output->fill = RAD_FILL_SIGMOID;
        return RAD_OK;
    }
    if (rank <= 0) return RAD_E_UNSUPPORTED;
    if (operand < 0 || operand >= 11) return RAD_E_SHAPE;
    const int64_t inject = r3dh_hc_pgeti_or(params, count, "inject", 1);
    const int64_t group = r3dh_hc_pgeti_or(params, count, "group", 0);
    const int64_t rotate = r3dh_hc_pgeti_or(params, count, "rotate", 0);
    if (group < 0 || (group && width % group)) return RAD_E_UNSUPPORTED;
    const RadOpdDesc operands[] = {
        describe(RAD_BF16, rows, channels), describe(RAD_F32, channels),
        describe(RAD_BF16, rank, channels), describe(RAD_BF16, channels, rank),
        inject ? describe(RAD_BF16, streams, channels) : absent(),
        describe(RAD_BF16, rows, width), inject ? describe(RAD_BF16, rows, streams) : absent(),
        group ? describe(RAD_F8E4M3, rows, width) : absent(),
        group ? describe(RAD_F32, rows, width / group) : absent(),
        group && rotate ? describe(RAD_F8E4M3, rows, width) : absent(),
        group && rotate ? describe(RAD_F32, rows, width / group) : absent(),
    };
    *output = operands[operand];
    return RAD_OK;
}
}

extern "C" int r3dh_hc_shape_hc_enter(const RadParam* params, int count, int operand,
                                       RadOpdDesc* output) {
    return shape(params, count, operand, output, 0);
}
extern "C" int r3dh_hc_shape_hc_write(const RadParam* params, int count, int operand,
                                       RadOpdDesc* output) {
    return shape(params, count, operand, output, 1);
}
extern "C" int r3dh_hc_shape_hc_read(const RadParam* params, int count, int operand,
                                      RadOpdDesc* output) {
    return shape(params, count, operand, output, 2);
}
