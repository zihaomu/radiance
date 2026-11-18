#include "r3dh_hc_api.h"
#include "r3dh_hc_args.h"
#include "r3dh_hc_enc.h"
#include "r3dh_hc_hc_fp8.h"
#include <cstring>
#include <limits>

namespace {
struct Geometry { int64_t rows, cols, group, pitch, bytes; };

int geometry(const RadParam* params, int count, int operand, Geometry& geom) {
    if (operand != 2 && operand != 3) return RAD_E_UNSUPPORTED;
    const int64_t width = r3dh_hc_pgeti_or(params, count, "n", 0);
    const int64_t streams = r3dh_hc_pgeti_or(params, count, "hc", 0);
    const int64_t rank = r3dh_hc_pgeti_or(params, count, "lowrank", 0);
    if (width <= 0 || streams <= 0 || streams > 8 || rank <= 0 ||
        width > std::numeric_limits<int64_t>::max() / streams) return RAD_E_SHAPE;
    geom.rows = operand == 2 ? rank : streams * width;
    geom.cols = operand == 2 ? streams * width : rank;
    geom.group = operand == 2 ? r3dh_hc_hc8::down_group(geom.cols)
                             : r3dh_hc_hc8::up_group(geom.cols);
    if (!r3dh_hc_hc8::group_ok(geom.cols, geom.group) ||
        geom.cols > (std::numeric_limits<int64_t>::max() - 15) / 5) return RAD_E_SHAPE;
    geom.pitch = r3dh_hc_hc8::row_bytes(geom.cols, geom.group);
    if (geom.rows > std::numeric_limits<int64_t>::max() / geom.pitch) return RAD_E_SHAPE;
    geom.bytes = geom.rows * geom.pitch;
    return RAD_OK;
}

int validate(const RadParam* params, int count, int operand, const RadEncoding* encoding,
             const int* selection, const RadTensor* planes, int plane_count,
             RadLayout* layout, Geometry& geom) {
    if (!layout || !planes) return RAD_E_INVAL;
    const int result = geometry(params, count, operand, geom);
    if (result != RAD_OK) return result;
    if (!r3dh_hc_enc::affine(encoding, RAD_F8E4M3, RAD_F32, 1, geom.group))
        return RAD_E_DTYPE;
    if (!r3dh_hc_enc::selects(encoding, selection, plane_count, "codes", "scale") ||
        !r3dh_hc_enc::dims(planes[0], geom.rows, geom.cols) ||
        !r3dh_hc_enc::dims(planes[1], geom.rows, geom.cols / geom.group)) return RAD_E_SHAPE;
    *layout = {};
    layout->tag = operand == 2 ? "r3dh.hc.e4m3.g128" : "r3dh.hc.e4m3.q4";
    layout->dtype = RAD_F8E4M3;
    layout->rank = 2;
    layout->shape[0] = geom.rows;
    layout->shape[1] = geom.pitch;
    layout->align = 16;
    layout->bytes = geom.bytes;
    return RAD_OK;
}
}

extern "C" int r3dh_hc_rad_layout_hc_e4m3(
    const RadParam* params, int count, int operand, const RadEncoding* encoding,
    const int* selection, const RadTensor* planes, int plane_count, RadLayout* layout) {
    Geometry geom{};
    return validate(params, count, operand, encoding, selection, planes, plane_count, layout, geom);
}

extern "C" int r3dh_hc_rad_relayout_hc_e4m3(
    const RadParam* params, int count, int operand, const RadEncoding* encoding,
    const int* selection, const RadTensor* planes, int plane_count, void* destination,
    int64_t bytes) {
    Geometry geom{};
    RadLayout layout{};
    const int result = validate(params, count, operand, encoding, selection, planes,
                                plane_count, &layout, geom);
    if (result != RAD_OK) return result;
    if (bytes != geom.bytes) return RAD_E_SHAPE;
    if (!destination || !planes[0].data || !planes[1].data) return RAD_E_INVAL;
    auto* output = static_cast<uint8_t*>(destination);
    const auto* codes = static_cast<const uint8_t*>(planes[0].data);
    const auto* scales = static_cast<const float*>(planes[1].data);
    std::memset(output, 0, static_cast<size_t>(bytes));
    for (int64_t row = 0; row < geom.rows; ++row) {
        std::memcpy(output + row * geom.pitch, codes + row * geom.cols, geom.cols);
        std::memcpy(output + row * geom.pitch + geom.cols,
                    scales + row * (geom.cols / geom.group), 4 * (geom.cols / geom.group));
    }
    return RAD_OK;
}

extern "C" int r3dh_hc_rad_unrelayout_hc_e4m3(
    const RadParam* params, int count, int operand, const RadEncoding* encoding,
    const int* selection, const void* source, int64_t bytes, const RadTensor* planes,
    int plane_count) {
    Geometry geom{};
    RadLayout layout{};
    const int result = validate(params, count, operand, encoding, selection, planes,
                                plane_count, &layout, geom);
    if (result != RAD_OK) return result;
    if (bytes != geom.bytes) return RAD_E_SHAPE;
    if (!source || !planes[0].data || !planes[1].data) return RAD_E_INVAL;
    const auto* input = static_cast<const uint8_t*>(source);
    auto* codes = static_cast<uint8_t*>(planes[0].data);
    auto* scales = static_cast<float*>(planes[1].data);
    for (int64_t row = 0; row < geom.rows; ++row) {
        std::memcpy(codes + row * geom.cols, input + row * geom.pitch, geom.cols);
        std::memcpy(scales + row * (geom.cols / geom.group),
                    input + row * geom.pitch + geom.cols, 4 * (geom.cols / geom.group));
    }
    return RAD_OK;
}
