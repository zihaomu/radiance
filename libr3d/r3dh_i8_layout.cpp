#include "r3dh_i8.h"
#include "rad_encoding.h"
#include <cstring>
#include <limits>

namespace {
struct Geometry { int64_t rows, columns, groups; };

int geometry(const RadParam* params, int count, Geometry& geom) {
    if (count < 0 || (count && !params)) return RAD_E_INVAL;
    const char* dtype = rad_param_gets(params, count, "dtype", "");
    if (std::strcmp(dtype, "i8a8") && std::strcmp(dtype, "i8a8h")
        && std::strcmp(dtype, "i8")) return RAD_E_DTYPE;
    geom.rows = rad_param_geti(params, count, "N", 0);
    geom.columns = rad_param_geti(params, count, "K", 0);
    if (geom.rows < 128 || geom.rows > 2147483520 || geom.rows % 128
        || geom.columns < 128 || geom.columns > 2147483520 || geom.columns % 128
        || rad_param_geti(params, count, "group", 128) != 128
        || rad_param_geti(params, count, "w_rb", 1) != 1
        || rad_param_geti(params, count, "pb1", 0)
        || rad_param_geti(params, count, "pb2", 0)) return RAD_E_SHAPE;
    geom.groups = geom.columns / 128;
    return RAD_OK;
}

int inspect(const RadParam* params, int count, int operand, const RadEncoding* encoding,
            const int* selected, const RadTensor* planes, int plane_count, Geometry& geom) {
    if (operand != 2 && operand != 3) return RAD_E_UNSUPPORTED;
    const int status = geometry(params, count, geom);
    if (status != RAD_OK) return status;
    if (!encoding || !selected || !planes) return RAD_E_INVAL;
    if (encoding->n_planes != 2 || encoding->reserved
        || !std::memchr(encoding->scheme, 0, RAD_ENC_STR)
        || !std::memchr(encoding->transform, 0, RAD_ENC_STR)
        || std::strcmp(encoding->scheme, "affine")) return RAD_E_DTYPE;
    const bool rotated = !std::strcmp(rad_param_gets(params, count, "dtype", ""), "i8a8h");
    if (std::strcmp(encoding->transform, rotated ? "fwht128" : "")) return RAD_E_DTYPE;
    int code_index = -1, scale_index = -1;
    for (int index = 0; index < 2; ++index) {
        const RadEncPlane& plane = encoding->plane[index];
        if (!std::memchr(plane.role, 0, RAD_ENC_STR) || plane.kind != RAD_PLANE_TILED
            || plane.extent[0] || plane.extent[1]) return RAD_E_DTYPE;
        if (!std::strcmp(plane.role, "codes") && code_index < 0
            && plane.dtype == RAD_I8 && plane.block[0] == 1 && plane.block[1] == 1)
            code_index = index;
        else if (!std::strcmp(plane.role, "scale") && scale_index < 0
            && plane.dtype == RAD_BF16 && plane.block[0] == 1 && plane.block[1] == 128)
            scale_index = index;
        else return RAD_E_DTYPE;
    }
    if (code_index < 0 || scale_index < 0) return RAD_E_DTYPE;
    if (plane_count != 1 || selected[0] != (operand == 2 ? code_index : scale_index))
        return RAD_E_SHAPE;
    const RadTensor& plane = planes[0];
    const int64_t width = operand == 2 ? geom.columns : geom.groups;
    if (plane.dtype != (operand == 2 ? RAD_I8 : RAD_BF16)) return RAD_E_DTYPE;
    if (plane.rank != 2 || plane.shape[0] != geom.rows || plane.shape[1] != width)
        return RAD_E_SHAPE;
    if (plane.stride[1] != 1 || plane.stride[0] < width
        || (geom.rows - 1) > (std::numeric_limits<int64_t>::max() - width) / plane.stride[0])
        return RAD_E_STRIDE;
    return RAD_OK;
}
int64_t stored_bytes(const Geometry& geom, int operand) {
    return geom.rows * (operand == 2 ? geom.columns : geom.groups * 4);
}
}

extern "C" int r3dh_i8_rad_layout(const RadParam* params, int count, int operand,
    const RadEncoding* encoding, const int* selected, const RadTensor* planes,
    int plane_count, RadLayout* output) {
    if (!output) return RAD_E_INVAL;
    Geometry geom{};
    const int status = inspect(params, count, operand, encoding, selected, planes, plane_count, geom);
    if (status != RAD_OK) return status;
    *output = {};
    output->tag = operand == 2 ? "r3dh.i8.n16g128k16.v1" : "r3dh.i8.scale.f32.v1";
    output->dtype = operand == 2 ? r3dh_i8::weight_dtype : r3dh_i8::scale_dtype;
    output->rank = 2;
    output->shape[0] = geom.rows;
    output->shape[1] = operand == 2 ? geom.columns : geom.groups;
    output->align = 256;
    output->bytes = stored_bytes(geom, operand);
    return RAD_OK;
}

extern "C" int r3dh_i8_rad_relayout(const RadParam* params, int count, int operand,
    const RadEncoding* encoding, const int* selected, const RadTensor* planes,
    int plane_count, void* destination, int64_t bytes) {
    Geometry geom{};
    const int status = inspect(params, count, operand, encoding, selected, planes, plane_count, geom);
    if (status != RAD_OK) return status;
    if (bytes != stored_bytes(geom, operand)) return RAD_E_SHAPE;
    if (!destination || !planes[0].data) return RAD_E_INVAL;
    if (operand == 2) {
        auto* output = static_cast<int8_t*>(destination);
        const auto* input = static_cast<const int8_t*>(planes[0].data);
        for (int64_t row = 0; row < geom.rows; ++row)
            for (int64_t column = 0; column < geom.columns; ++column)
                output[r3dh_i8::packed_index(row, column, geom.columns)]
                    = input[row * planes[0].stride[0] + column];
    } else {
        auto* output = static_cast<uint8_t*>(destination);
        const auto* input = static_cast<const uint16_t*>(planes[0].data);
        for (int64_t row = 0; row < geom.rows; ++row)
            for (int64_t group = 0; group < geom.groups; ++group) {
                const uint32_t bits = static_cast<uint32_t>(input[row * planes[0].stride[0] + group]) << 16;
                std::memcpy(output + (row * geom.groups + group) * 4, &bits, 4);
            }
    }
    return RAD_OK;
}

extern "C" int r3dh_i8_rad_unrelayout(const RadParam* params, int count, int operand,
    const RadEncoding* encoding, const int* selected, const void* source,
    int64_t bytes, const RadTensor* planes, int plane_count) {
    Geometry geom{};
    const int status = inspect(params, count, operand, encoding, selected, planes, plane_count, geom);
    if (status != RAD_OK) return status;
    if (bytes != stored_bytes(geom, operand)) return RAD_E_SHAPE;
    if (!source || !planes[0].data) return RAD_E_INVAL;
    if (operand == 3) {
        const auto* input = static_cast<const uint8_t*>(source);
        for (int64_t index = 0; index < geom.rows * geom.groups; ++index) {
            uint32_t bits;
            std::memcpy(&bits, input + index * 4, 4);
            if (bits & 0xffffu) return RAD_E_FORMAT;
        }
    }
    if (operand == 2) {
        const auto* input = static_cast<const int8_t*>(source);
        auto* output = static_cast<int8_t*>(planes[0].data);
        for (int64_t row = 0; row < geom.rows; ++row)
            for (int64_t column = 0; column < geom.columns; ++column)
                output[row * planes[0].stride[0] + column]
                    = input[r3dh_i8::packed_index(row, column, geom.columns)];
    } else {
        const auto* input = static_cast<const uint8_t*>(source);
        auto* output = static_cast<uint16_t*>(planes[0].data);
        for (int64_t row = 0; row < geom.rows; ++row)
            for (int64_t group = 0; group < geom.groups; ++group) {
                uint32_t bits;
                std::memcpy(&bits, input + (row * geom.groups + group) * 4, 4);
                output[row * planes[0].stride[0] + group] = static_cast<uint16_t>(bits >> 16);
            }
    }
    return RAD_OK;
}

extern "C" int r3dh_i8_shape_gemm_nt_q(const RadParam* params, int count, int operand,
                                      RadOpdDesc* output) {
    if (!output || operand < 0 || operand > 6) return RAD_E_INVAL;
    Geometry geom{};
    const int status = geometry(params, count, geom);
    if (status != RAD_OK) return status;
    const int64_t rows = rad_param_geti(params, count, "M", 0);
    if (rows < 1 || rows > 2048) return RAD_E_SHAPE;
    *output = {};
    output->idx_const = -1;
    if (operand >= 5) { output->flags = RAD_OPD_F_ABSENT; return RAD_OK; }
    output->rank = 2;
    output->dtype = operand == 0 || operand == 2 ? RAD_I8
                  : operand == 3 || operand == 4 ? RAD_BF16 : RAD_F32;
    output->shape[0] = operand == 2 || operand == 3 ? geom.rows : rows;
    output->shape[1] = operand == 4 ? geom.rows
                      : operand == 1 || operand == 3 ? geom.groups : geom.columns;
    output->fill = operand == 1 || operand == 3 ? RAD_FILL_SIGMOID : RAD_FILL_NORMAL;
    return RAD_OK;
}

extern "C" int r3dh_i8_shape_quant_act_i8g(const RadParam* params, int count, int operand,
                                         RadOpdDesc* output) {
    if (!output || operand < 0 || operand > 2 || count < 0 || (count && !params))
        return RAD_E_INVAL;
    const int64_t rows = rad_param_geti(params, count, "M", 0);
    const int64_t width = rad_param_geti(params, count, "n", 0);
    if (rows < 1 || rows > 2048 || width < 128 || width > 2147483520 || width % 128
        || rad_param_geti(params, count, "group", 128) != 128) return RAD_E_SHAPE;
    *output = {};
    output->idx_const = -1;
    output->rank = 2;
    output->dtype = operand == 0 ? RAD_BF16 : operand == 1 ? RAD_I8 : RAD_F32;
    output->shape[0] = rows;
    output->shape[1] = operand == 2 ? width / 128 : width;
    output->fill = RAD_FILL_NORMAL;
    return RAD_OK;
}
