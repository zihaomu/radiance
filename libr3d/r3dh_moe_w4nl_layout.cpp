#include "r3dh_moe_w4nl.h"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace r3dh_moe_w4nl {
int config(const RadParam* params, int count, Config* output) {
    if (!output || count < 0 || (count && !params)) return RAD_E_INVAL;
    if (!std::strcmp(rad_param_gets(params, count, "dtype", ""), "w4nl64a8i8h")
        && !std::strcmp(rad_param_gets(params, count, "a_order", "token"), "token")
        && rad_param_geti(params, count, "parts", 1) == 2) {
        const int64_t tokens = rad_param_geti(params, count, "M", 0);
        const int64_t columns = rad_param_geti(params, count, "N", 0);
        const int64_t odd = rad_param_geti(params, count, "N_odd", columns);
        if (tokens < 1 || tokens > 2048
            || !((columns == 768 && odd == 512) || (columns == 512 && odd == 768))
            || rad_param_geti(params, count, "K", 0) != 2560
            || rad_param_geti(params, count, "K_odd", 2560) != 2560
            || rad_param_geti(params, count, "n_expert", 0) != 512
            || rad_param_geti(params, count, "top_k", 0) != 10
            || rad_param_geti(params, count, "group", 128) != 128
            || rad_param_geti(params, count, "a_group", 128) != 128) return RAD_E_SHAPE;
        *output = Config{int(tokens), 768, {2560, 2560}, 512, 10, 2560,
                         true, {int(columns), int(odd)}};
        return RAD_OK;
    }
    if (std::strcmp(rad_param_gets(params, count, "dtype", ""), "w4nl64a8i8h")
        || std::strcmp(rad_param_gets(params, count, "a_order", "token"), "sorted"))
        return RAD_E_UNSUPPORTED;
    const int64_t tokens = rad_param_geti(params, count, "M", 0);
    const int64_t columns = rad_param_geti(params, count, "N", 0);
    const int64_t inner = rad_param_geti(params, count, "K", 0);
    const int64_t odd = rad_param_geti(params, count, "K_odd", inner);
    const int64_t experts = rad_param_geti(params, count, "n_expert", 0);
    const int64_t top = rad_param_geti(params, count, "top_k", 0);
    if (tokens < 1 || tokens > 2048 || columns < 16 || columns > 65536 || columns % 16
        || inner < 128 || inner > 65536 || inner % 128
        || odd < 128 || odd > 65536 || odd % 128
        || rad_param_geti(params, count, "N_odd", columns) != columns
        || experts < 1 || experts > 512 || top < 1 || top > 10 || top > experts
        || rad_param_geti(params, count, "parts", 1) != 1
        || rad_param_geti(params, count, "group", 128) != 128
        || rad_param_geti(params, count, "a_group", 128) != 128) return RAD_E_SHAPE;
    *output = Config{int(tokens), int(columns), {int(inner), int(odd)}, int(experts),
                     int(top), int(std::max(inner, odd))};
    return RAD_OK;
}
}
namespace {
using namespace r3dh_moe_w4nl;
struct View {
    int rows, columns;
    bool codes;
    const RadTensor *first, *second;
};
bool matrix(const RadTensor& value, uint32_t type, int rows, int columns) {
    return value.dtype == type && value.rank == 2 && value.shape[0] == rows
        && value.shape[1] == columns && value.stride[0] == columns && value.stride[1] == 1;
}
bool scalar(const RadTensor& value) {
    return value.dtype == RAD_F32 && (value.rank == 0 || matrix(value, RAD_F32, 1, 1));
}
int inspect(const RadParam* params, int count, int operand, const RadEncoding* encoding,
            const int* selected, const RadTensor* planes, int plane_count, View* view) {
    Config geometry;
    const int status = config(params, count, &geometry);
    if (status != RAD_OK) return status;
    if (operand != 2 && operand != 3 && operand != 6 && operand != 7)
        return RAD_E_UNSUPPORTED;
    if (!encoding || !selected || !planes) return RAD_E_INVAL;
    if (plane_count != 2 || encoding->n_planes != 4) return RAD_E_SHAPE;
    if (std::strcmp(encoding->scheme, "affine") || std::strcmp(encoding->transform, "fwht128"))
        return RAD_E_DTYPE;
    const RadEncPlane* codes = rad_enc_plane(encoding, "codes");
    const RadEncPlane* table = rad_enc_plane(encoding, "table");
    const RadEncPlane* scale = rad_enc_plane(encoding, "scale");
    const RadEncPlane* level = rad_enc_plane(encoding, "scale.1");
    if (!codes || !table || !scale || !level
        || codes->kind != RAD_PLANE_TILED || codes->dtype != RAD_U4
        || codes->block[0] != 1 || codes->block[1] != 1
        || table->kind != RAD_PLANE_TABLE || table->dtype != RAD_F32
        || table->extent[0] != 1 || table->extent[1] != 16
        || scale->kind != RAD_PLANE_TILED || scale->dtype != RAD_F8E4M3
        || scale->block[0] != 1 || scale->block[1] != 64
        || level->kind != RAD_PLANE_TILED || level->dtype != RAD_F32
        || level->block[0] != 0 || level->block[1] != 0) return RAD_E_DTYPE;
    view->rows = geometry.gate ? geometry.gate_columns[operand >= 6] : geometry.columns;
    view->columns = geometry.inner[operand >= 6];
    view->codes = operand == 2 || operand == 6;
    if (selected[0] < 0 || selected[0] >= 4 || selected[1] < 0 || selected[1] >= 4
        || selected[0] == selected[1]) return RAD_E_SHAPE;
    if (&encoding->plane[selected[0]] != (view->codes ? codes : scale)
        || &encoding->plane[selected[1]] != (view->codes ? table : level)) return RAD_E_DTYPE;
    view->first = planes;
    view->second = planes + 1;
    if (view->codes) return matrix(planes[0], RAD_U4, view->rows, view->columns)
        && matrix(planes[1], RAD_F32, 1, 16) ? RAD_OK : RAD_E_SHAPE;
    return matrix(planes[0], RAD_F8E4M3, view->rows, view->columns / 64)
        && scalar(planes[1]) ? RAD_OK : RAD_E_SHAPE;
}
int64_t bytes(const View& view) {
    return view.codes ? sizeof(CodesHeader) + int64_t(view.rows) * view.columns / 2
                      : sizeof(ScalesHeader) + int64_t(view.rows) * (view.columns / 64);
}
int values(const View& view) {
    if (!view.first->data || !view.second->data) return RAD_E_INVAL;
    if (view.codes) {
        const auto* table = static_cast<const float*>(view.second->data);
        for (int index = 0; index < 16; ++index)
            if (!std::isfinite(table[index])) return RAD_E_FORMAT;
    } else {
        float level;
        std::memcpy(&level, view.second->data, sizeof(level));
        if (level != 0x1p-13f) return RAD_E_FORMAT;
        const auto* scales = static_cast<const uint8_t*>(view.first->data);
        for (int64_t index = 0; index < int64_t(view.rows) * (view.columns / 64); ++index)
            if ((scales[index] & 127) == 127) return RAD_E_FORMAT;
    }
    return RAD_OK;
}
}
extern "C" int r3dh_moe_w4nl_shape(const RadParam* params, int count, int operand,
                                  RadOpdDesc* output) {
    if (!output || operand < 0 || operand > 8) return RAD_E_INVAL;
    Config geometry;
    const int status = config(params, count, &geometry);
    if (status != RAD_OK) return status;
    *output = RadOpdDesc{};
    output->rank = 2;
    if (operand == 2 || operand == 3 || operand == 6 || operand == 7) {
        output->dtype = RAD_F32;
        output->shape[0] = geometry.gate ? geometry.gate_columns[operand >= 6] : geometry.columns;
        output->shape[1] = operand == 2 || operand == 6
            ? geometry.inner[operand >= 6] : geometry.inner[operand >= 6] / 64;
    } else if (operand == 4 || operand == 5) {
        output->dtype = RAD_I32;
        output->rank = 1;
        output->shape[0] = operand == 4 ? geometry.tokens * geometry.top_k : geometry.experts + 1;
    } else {
        output->dtype = operand == 0 ? RAD_I8 : operand == 1 ? RAD_F32 : RAD_BF16;
        output->shape[0] = geometry.gate && operand < 2
            ? geometry.tokens : geometry.tokens * geometry.top_k;
        output->shape[1] = operand == 0 ? geometry.activation_columns
            : operand == 1 ? geometry.activation_columns / 128 : geometry.columns;
    }
    return RAD_OK;
}
extern "C" int r3dh_moe_w4nl_layout(const RadParam* params, int count, int operand,
    const RadEncoding* encoding, const int* selected, const RadTensor* planes,
    int plane_count, RadLayout* output) {
    if (!output) return RAD_E_INVAL;
    View view{};
    const int status = inspect(params, count, operand, encoding, selected, planes, plane_count, &view);
    if (status != RAD_OK) return status;
    *output = RadLayout{};
    output->tag = view.codes ? codes_tag : scales_tag;
    output->dtype = view.codes ? codes_type : scales_type;
    output->rank = 1;
    output->shape[0] = bytes(view);
    output->bytes = bytes(view);
    output->align = 256;
    return RAD_OK;
}
extern "C" int r3dh_moe_w4nl_transform(const RadParam* params, int count, int operand,
    const RadEncoding* encoding, const int* selected, const RadTensor* planes,
    int plane_count, void* destination, int64_t size) {
    View view{};
    int status = inspect(params, count, operand, encoding, selected, planes, plane_count, &view);
    if (status != RAD_OK) return status;
    if (!destination || size != bytes(view)) return RAD_E_SHAPE;
    status = values(view);
    if (status != RAD_OK) return status;
    auto* output = static_cast<uint8_t*>(destination);
    if (view.codes) {
        CodesHeader header{codes_magic, 1, uint32_t(view.rows), uint32_t(view.columns), {}};
        std::memcpy(header.table, view.second->data, sizeof(header.table));
        std::memcpy(output, &header, sizeof(header));
        std::memcpy(output + sizeof(header), view.first->data, size - sizeof(header));
    } else {
        const ScalesHeader header{scales_magic, 1, uint32_t(view.rows), uint32_t(view.columns),
                                  0x1p-13f, {0, 0, 0}};
        std::memcpy(output, &header, sizeof(header));
        std::memcpy(output + sizeof(header), view.first->data, size - sizeof(header));
    }
    return RAD_OK;
}
extern "C" int r3dh_moe_w4nl_inverse(const RadParam* params, int count, int operand,
    const RadEncoding* encoding, const int* selected, const void* source,
    int64_t size, const RadTensor* planes, int plane_count) {
    View view{};
    const int status = inspect(params, count, operand, encoding, selected, planes, plane_count, &view);
    if (status != RAD_OK) return status;
    if (!source || !view.first->data || !view.second->data) return RAD_E_INVAL;
    if (size != bytes(view)) return RAD_E_SHAPE;
    const auto* input = static_cast<const uint8_t*>(source);
    if (view.codes) {
        CodesHeader header;
        std::memcpy(&header, input, sizeof(header));
        if (header.magic != codes_magic || header.version != 1 || header.rows != uint32_t(view.rows)
            || header.columns != uint32_t(view.columns)) return RAD_E_FORMAT;
        for (float value : header.table) if (!std::isfinite(value)) return RAD_E_FORMAT;
        std::memcpy(view.second->data, header.table, sizeof(header.table));
        std::memcpy(view.first->data, input + sizeof(header), size - sizeof(header));
    } else {
        ScalesHeader header;
        std::memcpy(&header, input, sizeof(header));
        if (header.magic != scales_magic || header.version != 1 || header.rows != uint32_t(view.rows)
            || header.columns != uint32_t(view.columns) || header.level != 0x1p-13f
            || header.reserved[0] || header.reserved[1] || header.reserved[2]) return RAD_E_FORMAT;
        for (int64_t index = sizeof(header); index < size; ++index)
            if ((input[index] & 127) == 127) return RAD_E_FORMAT;
        std::memcpy(view.second->data, &header.level, sizeof(header.level));
        std::memcpy(view.first->data, input + sizeof(header), size - sizeof(header));
    }
    return RAD_OK;
}
