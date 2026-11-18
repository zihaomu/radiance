#ifndef R3D_LOGITS_PLANES_H
#define R3D_LOGITS_PLANES_H
#include "rad_planes.h"
#include <stdint.h>
#if defined(__HIPCC__)
#define R3D_HEAD_HD __host__ __device__
#else
#define R3D_HEAD_HD
#endif
namespace r3d_head {
constexpr uint32_t dtype = 0x33504c4eu;
constexpr int64_t limit = INT64_MAX - (RAD_PLANES_ALIGN - 1);
R3D_HEAD_HD inline bool role(const char* actual, const char* expected) {
    for (int index = 0; index < RAD_ENC_STR; ++index) {
        if (actual[index] != expected[index]) return false;
        if (!expected[index]) return true;
    }
    return false;
}
R3D_HEAD_HD inline int scale_bytes(uint32_t value) {
    return value == RAD_F32 ? 4 : value == RAD_BF16 || value == RAD_F16 ? 2 : 0;
}
R3D_HEAD_HD inline int64_t cover(int64_t extent, int64_t block) {
    return block ? 1 + (extent - 1) / block : 1;
}
R3D_HEAD_HD inline int64_t rounded(int64_t value) {
    return (value + RAD_PLANES_ALIGN - 1) & -int64_t(RAD_PLANES_ALIGN);
}
R3D_HEAD_HD inline int entries(const RadPlanesEntry* plane, int64_t rows, int64_t cols,
                               int64_t* total) {
    if (rows <= 0 || cols <= 0) return RAD_E_SHAPE;
    int64_t offset = rounded(sizeof(RadPlanesHeader) + 2 * sizeof(RadPlanesEntry));
    for (int index = 0; index < 2; ++index) {
        const RadPlanesEntry& entry = plane[index];
        const int width = index ? scale_bytes(entry.dtype) : 1;
        if (!role(entry.role, index ? "scale" : "codes") ||
            entry.kind != RAD_PLANE_TILED || !width ||
            (!index && entry.dtype != RAD_I8 && entry.dtype != RAD_F8E4M3) ||
            (!index && (entry.block[0] != 1 || entry.block[1] != 1))) return RAD_E_DTYPE;
        if (entry.block[0] < 0 || entry.block[1] < 0) return RAD_E_SHAPE;
        if (entry.rows != (index ? cover(rows, entry.block[0]) : rows) ||
            entry.cols != (index ? cover(cols, entry.block[1]) : cols)) return RAD_E_SHAPE;
        if (entry.cols > limit / width || entry.row_bytes != entry.cols * width)
            return RAD_E_STRIDE;
        if (entry.offset != offset || entry.rows > (limit - offset) / entry.row_bytes)
            return RAD_E_FORMAT;
        offset = rounded(offset + entry.rows * entry.row_bytes);
    }
    *total = offset;
    return RAD_OK;
}
R3D_HEAD_HD inline int header(const RadPlanesHeader* value, int64_t rows, int64_t cols,
                              int64_t available = 0) {
    if (value->magic != RAD_PLANES_MAGIC || value->n_planes != 2 || value->transform[0] ||
        value->bytes < int64_t(sizeof(RadPlanesHeader) + 2 * sizeof(RadPlanesEntry)) ||
        (available && value->bytes > available)) return RAD_E_FORMAT;
    int64_t total = 0;
    const int status = entries(reinterpret_cast<const RadPlanesEntry*>(value + 1), rows, cols, &total);
    if (status != RAD_OK) return status;
    return total == value->bytes ? RAD_OK : RAD_E_FORMAT;
}
inline int plan(const RadEncoding* encoding, const int* selected, const RadTensor* planes,
                int count, RadLayout* output) {
    if (!encoding || !selected || !planes || !output || count != 2 ||
        !rad_enc_is(encoding, "affine") || encoding->transform[0] || encoding->n_planes != 2)
        return RAD_E_DTYPE;
    RadPlanesEntry entry[2]{};
    int64_t offset = rounded(sizeof(RadPlanesHeader) + sizeof(entry));
    for (int index = 0; index < 2; ++index) {
        if (selected[index] < 0 || selected[index] >= 2) return RAD_E_DTYPE;
        const RadEncPlane& source = encoding->plane[selected[index]];
        const RadTensor& tensor = planes[index];
        const int width = index ? scale_bytes(tensor.dtype) : 1;
        if (source.dtype != tensor.dtype || tensor.rank != 2 || !width) return RAD_E_DTYPE;
        if (tensor.shape[0] <= 0 || tensor.shape[1] <= 0 || tensor.shape[1] > limit / width)
            return RAD_E_SHAPE;
        if (tensor.stride[1] != 1 || tensor.stride[0] != tensor.shape[1]) return RAD_E_STRIDE;
        for (int character = 0; character < RAD_ENC_STR; ++character)
            entry[index].role[character] = source.role[character];
        entry[index].dtype = tensor.dtype;
        entry[index].kind = source.kind;
        entry[index].block[0] = source.block[0];
        entry[index].block[1] = source.block[1];
        entry[index].rows = tensor.shape[0];
        entry[index].cols = tensor.shape[1];
        entry[index].row_bytes = tensor.shape[1] * width;
        entry[index].offset = offset;
        if (tensor.shape[0] > (limit - offset) / entry[index].row_bytes) return RAD_E_SHAPE;
        offset = rounded(offset + tensor.shape[0] * entry[index].row_bytes);
    }
    int64_t total = 0;
    const int status = entries(entry, planes[0].shape[0], planes[0].shape[1], &total);
    if (status != RAD_OK) return status;
    const int result = rad_planes_layout(encoding, selected, planes, count, output);
    if (result == RAD_OK) output->dtype = dtype;
    return result;
}
}
#undef R3D_HEAD_HD
#endif
