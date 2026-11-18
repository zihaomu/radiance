#include "r3dh_mtp_ple.h"
#include <cstring>

namespace {
int describe(const RadParam* params, int count, int operand, const RadEncoding* encoding,
             const int* selection, const RadTensor* planes, int plane_count, RadLayout* layout) {
    if (operand != 4 && operand != 5) return RAD_E_UNSUPPORTED;
    if (!planes || !layout || !encoding || !selection) return RAD_E_INVAL;
    const int64_t width = rad_param_geti(params, count, "n", 0);
    if (width <= 0 || width > 4096 || width % 128) return RAD_E_SHAPE;
    if (!rad_enc_is(encoding, "affine") || encoding->n_planes != 2 || encoding->transform[0]) return RAD_E_DTYPE;
    const int codes = rad_enc_find(encoding, "codes"), scale = rad_enc_find(encoding, "scale");
    if (codes < 0 || scale < 0 || encoding->plane[codes].dtype != RAD_F8E4M3 ||
        encoding->plane[codes].block[0] != 1 || encoding->plane[codes].block[1] != 1 ||
        encoding->plane[scale].dtype != RAD_F32 || encoding->plane[scale].kind != RAD_PLANE_TILED ||
        encoding->plane[scale].block[0] != 1 || encoding->plane[scale].block[1] != 128) return RAD_E_DTYPE;
    if (plane_count != 2 || selection[0] != codes || selection[1] != scale) return RAD_E_SHAPE;
    for (int index = 0; index < 2; ++index) {
        if (planes[index].rank != 2 || planes[index].shape[0] != width ||
            planes[index].shape[1] != (index ? width / 128 : width)) return RAD_E_SHAPE;
        if (planes[index].dtype != (index ? RAD_F32 : RAD_F8E4M3)) return RAD_E_DTYPE;
        if (!rad_tensor_is_contiguous(planes + index)) return RAD_E_STRIDE;
    }
    *layout = {};
    layout->tag = "r4d.mtp.e4m3.g128";
    layout->dtype = r3dh_mtp_fc_e4m3_dtype; layout->rank = 2;
    layout->shape[0] = width; layout->shape[1] = (width + 4 * (width / 128) + 15) & ~int64_t(15);
    layout->align = 16; layout->bytes = width * layout->shape[1];
    return RAD_OK;
}
}
extern "C" int r3dh_mtp_ple_layout_mtp_e4m3(const RadParam* params, int count, int operand,
    const RadEncoding* encoding, const int* selection, const RadTensor* planes, int plane_count, RadLayout* layout) {
    return describe(params, count, operand, encoding, selection, planes, plane_count, layout);
}
extern "C" int r3dh_mtp_ple_relayout_mtp_e4m3(const RadParam* params, int count, int operand,
    const RadEncoding* encoding, const int* selection, const RadTensor* planes, int plane_count, void* destination, int64_t bytes) {
    RadLayout layout{};
    const int status = describe(params, count, operand, encoding, selection, planes, plane_count, &layout);
    if (status != RAD_OK) return status;
    if (bytes != layout.bytes) return RAD_E_SHAPE;
    if (!destination || !planes[0].data || !planes[1].data) return RAD_E_INVAL;
    const int64_t width = layout.shape[0], row_bytes = layout.shape[1];
    auto* output = static_cast<uint8_t*>(destination);
    std::memset(output, 0, size_t(bytes));
    for (int64_t row = 0; row < width; ++row) {
        std::memcpy(output + row * row_bytes, static_cast<const uint8_t*>(planes[0].data) + row * width, size_t(width));
        std::memcpy(output + row * row_bytes + width, static_cast<const float*>(planes[1].data) + row * (width / 128), size_t(4 * (width / 128)));
    }
    return RAD_OK;
}
extern "C" int r3dh_mtp_ple_unrelayout_mtp_e4m3(const RadParam* params, int count, int operand,
    const RadEncoding* encoding, const int* selection, const void* source, int64_t bytes, const RadTensor* planes, int plane_count) {
    RadLayout layout{};
    const int status = describe(params, count, operand, encoding, selection, planes, plane_count, &layout);
    if (status != RAD_OK) return status;
    if (bytes != layout.bytes) return RAD_E_SHAPE;
    if (!source || !planes[0].data || !planes[1].data) return RAD_E_INVAL;
    const int64_t width = layout.shape[0], row_bytes = layout.shape[1];
    const auto* input = static_cast<const uint8_t*>(source);
    for (int64_t row = 0; row < width; ++row) {
        std::memcpy(static_cast<uint8_t*>(planes[0].data) + row * width, input + row * row_bytes, size_t(width));
        std::memcpy(static_cast<float*>(planes[1].data) + row * (width / 128), input + row * row_bytes + width, size_t(4 * (width / 128)));
    }
    return RAD_OK;
}
