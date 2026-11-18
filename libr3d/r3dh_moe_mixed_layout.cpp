#include "r3dh_moe_mixed.h"

namespace {
int compact(const RadParam* params, int count) {
    if (count < 0 || count > 32 || (count && !params)) return RAD_E_INVAL;
    const auto experts = rad_param_geti(params, count, "n_expert", 0);
    return experts == 510 || experts == 508 ? RAD_OK : RAD_E_SHAPE;
}
}

int r3dh_moe_mixed::config(const RadParam* params, int count,
                          r3dh_moe_iqr::Config* output) {
    const int status = compact(params, count);
    return status == RAD_OK ? r3dh_moe_iqr::config(params, count, output) : status;
}

extern "C" int r3dh_moe_mixed_iqr_shape(const RadParam* params, int count,
                                        int operand, RadOpdDesc* output) {
    const int status = compact(params, count);
    return status == RAD_OK ? r3dh_moe_iqr_shape(params, count, operand, output) : status;
}

extern "C" int r3dh_moe_mixed_iqr_layout(const RadParam* params, int count, int operand,
    const RadEncoding* encoding, const int* selected, const RadTensor* planes,
    int plane_count, RadLayout* output) {
    const int status = compact(params, count);
    return status == RAD_OK ? r3dh_moe_iqr_layout(params, count, operand,
        encoding, selected, planes, plane_count, output) : status;
}

extern "C" int r3dh_moe_mixed_iqr_relayout(const RadParam* params, int count, int operand,
    const RadEncoding* encoding, const int* selected, const RadTensor* planes,
    int plane_count, void* destination, int64_t bytes) {
    const int status = compact(params, count);
    return status == RAD_OK ? r3dh_moe_iqr_relayout(params, count, operand,
        encoding, selected, planes, plane_count, destination, bytes) : status;
}

extern "C" int r3dh_moe_mixed_iqr_unrelayout(const RadParam* params, int count, int operand,
    const RadEncoding* encoding, const int* selected, const void* source,
    int64_t bytes, const RadTensor* planes, int plane_count) {
    const int status = compact(params, count);
    return status == RAD_OK ? r3dh_moe_iqr_unrelayout(params, count, operand,
        encoding, selected, source, bytes, planes, plane_count) : status;
}
