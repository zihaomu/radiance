#ifndef R3D_W2_HEAD_H
#define R3D_W2_HEAD_H
#include "r3d_logits_planes.h"
namespace r3d_w2 {
constexpr uint32_t dtype = 0x33573253u;
#if defined(__HIPCC__)
#define R3D_W2_HD __host__ __device__
#else
#define R3D_W2_HD
#endif
R3D_W2_HD inline int header(const RadPlanesHeader* value, int64_t rows, int64_t cols,
                            int64_t available = 0) {
    if (rows <= 0 || cols <= 0 || cols % 128) return RAD_E_SHAPE;
    if (value->magic != RAD_PLANES_MAGIC || value->n_planes != 2 ||
        (value->transform[0] && !r3d_head::role(value->transform, "givens")) ||
        value->bytes < r3d_head::rounded(sizeof(RadPlanesHeader) + 2 * sizeof(RadPlanesEntry)) ||
        (available && value->bytes > available)) return RAD_E_FORMAT;
    const auto* entries = reinterpret_cast<const RadPlanesEntry*>(value + 1);
    int64_t offset = r3d_head::rounded(sizeof(RadPlanesHeader) + 2 * sizeof(RadPlanesEntry));
    for (int index = 0; index < 2; ++index) {
        const auto& entry = entries[index];
        const int width = index ? 1 : 2;
        if (!r3d_head::role(entry.role, index ? "zero" : "scale") ||
            entry.kind != RAD_PLANE_TILED || entry.dtype != (index ? RAD_U8 : RAD_F16))
            return RAD_E_DTYPE;
        if (entry.block[0] != 1 || entry.block[1] != 128 ||
            entry.rows != rows || entry.cols != cols / 128) return RAD_E_SHAPE;
        if (entry.cols > r3d_head::limit / width || entry.row_bytes != entry.cols * width)
            return RAD_E_STRIDE;
        if (entry.offset != offset || entry.rows > (r3d_head::limit - offset) / entry.row_bytes)
            return RAD_E_FORMAT;
        offset = r3d_head::rounded(offset + entry.rows * entry.row_bytes);
    }
    return offset == value->bytes ? RAD_OK : RAD_E_FORMAT;
}
#undef R3D_W2_HD
}
#endif
