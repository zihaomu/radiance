#pragma once
#include "r3d_iqr.h"
#include "r3dh_i8.h"
#include "r3dh_qwen_ops.h"
#include <cstring>
#include <limits>

extern "C" int r3d_rad_gemm_iqr_scalar(const RadArgs*, RadStream);

namespace r3d_i8_gated {
constexpr int64_t scratch_bytes = 16 * 768 * 2;
inline int geometry(const RadArgs* args) {
    if (!args || args->n_p < 0 || (args->n_p && !args->p)) return RAD_E_INVAL;
    if (std::strcmp(rad_args_gets_or(args, "dtype", ""), "i8a8h")) return RAD_E_DTYPE;
    if (std::strcmp(rad_args_gets_or(args, "act", ""), "silu")) return RAD_E_UNSUPPORTED;
    const int64_t band = rad_args_geti_or(args, "M", 0);
    const int64_t columns = rad_args_geti_or(args, "N", 0);
    if (band < 1 || band > 16 || (columns != 512 && columns != 768)
        || rad_args_geti_or(args, "K", 0) != 2560
        || rad_args_geti_or(args, "group", 128) != 128
        || rad_args_geti_or(args, "w_rb", 1) != 1
        || rad_args_geti_or(args, "pb1", 0) || rad_args_geti_or(args, "pb2", 0))
        return RAD_E_SHAPE;
    return RAD_OK;
}
inline bool overlap(uintptr_t left, uint64_t left_bytes, uintptr_t right, uint64_t right_bytes) {
    return left < right + right_bytes && right < left + left_bytes;
}
inline int matrix(const RadTensor& value, uint32_t dtype, int64_t rows, int64_t columns,
                  unsigned bytes, uint64_t& extent) {
    if (!value.data || value.dtype != dtype) return RAD_E_DTYPE;
    if (value.rank != 2 || value.shape[0] != rows || value.shape[1] != columns) return RAD_E_SHAPE;
    if (value.stride[1] != 1 || value.stride[0] < columns
        || value.stride[0] > (std::numeric_limits<int64_t>::max() / bytes - columns) / rows
        || reinterpret_cast<uintptr_t>(value.data) % bytes) return RAD_E_STRIDE;
    extent = uint64_t((rows - 1) * value.stride[0] + columns) * bytes;
    if (reinterpret_cast<uintptr_t>(value.data) > std::numeric_limits<uintptr_t>::max() - extent)
        return RAD_E_STRIDE;
    return RAD_OK;
}
}

extern "C" int64_t r3d_i8_gated_scratch(const RadArgs* args) {
    const int status = r3d_i8_gated::geometry(args);
    return status ? status : r3d_i8_gated::scratch_bytes;
}

extern "C" int r3d_i8_gated_launch(const RadArgs* args, RadStream stream) {
    using namespace r3d_i8_gated;
    int status = geometry(args);
    if (status) return status;
    if (!args->t || args->n_t < 6 || args->n_t > 7) return RAD_E_INVAL;
    if (!args->scratch || args->scratch_bytes < scratch_bytes
        || reinterpret_cast<uintptr_t>(args->scratch) % 256) return RAD_E_INVAL;
    RadTensor normalized_tensors[7]{};
    std::memcpy(normalized_tensors, args->t, args->n_t * sizeof(RadTensor));
    RadArgs normalized = *args; normalized.t = normalized_tensors; args = &normalized;
    auto& scale_view = normalized_tensors[3];
    if (normalized_tensors[2].dtype == r3dh_i8::weight_dtype && scale_view.dtype == r3dh_i8::scale_dtype
        && scale_view.rank == 2 && scale_view.shape[1] == 2560 && scale_view.stride[0] == 2560 && scale_view.stride[1] == 1) {
        scale_view.shape[1] = 20; scale_view.stride[0] = 20;
    }
    const int64_t rows = args->t[0].rank == 2 ? args->t[0].shape[0] : 0;
    const int64_t columns = rad_args_geti_or(args, "N", 0);
    if (rows < 1 || rows > rad_args_geti_or(args, "M", 0)) return RAD_E_SHAPE;
    const bool iqr = args->t[2].dtype == r3d_iqr::weight_dtype;
    if ((!iqr && args->t[2].dtype != r3dh_i8::weight_dtype)
        || (iqr && args->t[3].dtype != r3d_iqr::scale_dtype && args->t[3].dtype != r3d_iqr::group_scale_dtype)
        || (!iqr && args->t[3].dtype != r3dh_i8::scale_dtype)) return RAD_E_DTYPE;
    uint64_t extents[7]{};
    const uint32_t types[]{RAD_I8, RAD_F32, args->t[2].dtype, args->t[3].dtype, RAD_I8, RAD_F32, RAD_BF16};
    const int64_t heights[]{rows, rows, columns, columns, rows, rows, rows};
    const int64_t widths[]{2560, 20, 2560, 20, columns / 2, columns / 256, columns / 2};
    const unsigned sizes[]{1, 4, 1, 4, 1, 4, 2};
    const int count = args->n_t == 7 && (args->t[6].data || args->t[6].rank) ? 7 : 6;
    for (int index = 0; index < count; ++index) {
        int64_t width = widths[index];
        if (iqr && index == 3 && args->t[index].shape[1] == 2560) width = 2560;
        status = matrix(args->t[index], types[index], heights[index], width, sizes[index], extents[index]);
        if (status) return status;
        if (!iqr && (index == 2 || index == 3) && args->t[index].stride[0] != width) return RAD_E_STRIDE;
    }
    if (args->t[0].stride[0] % 4 || reinterpret_cast<uintptr_t>(args->t[0].data) % 4
        || reinterpret_cast<uintptr_t>(args->t[2].data) % 4) return RAD_E_STRIDE;
    if (iqr) {
        extents[2] = uint64_t(columns) * 2560 + sizeof(r3d_iqr::Header);
        extents[3] = args->t[3].dtype == r3d_iqr::group_scale_dtype
            ? uint64_t(columns) * 20 * 2 : uint64_t(columns + 20) * 4;
    }
    const uintptr_t scratch = reinterpret_cast<uintptr_t>(args->scratch);
    if (scratch > std::numeric_limits<uintptr_t>::max() - scratch_bytes) return RAD_E_SHAPE;
    for (int index = 0; index < count; ++index) {
        const uintptr_t begin = reinterpret_cast<uintptr_t>(args->t[index].data);
        if (begin > std::numeric_limits<uintptr_t>::max() - extents[index]
            || overlap(scratch, scratch_bytes, begin, extents[index])) return RAD_E_SHAPE;
        for (int output = 4; index >= 4 && output < index; ++output)
            if (overlap(begin, extents[index],
                reinterpret_cast<uintptr_t>(args->t[output].data), extents[output])) return RAD_E_SHAPE;
    }
    RadTensor intermediate{};
    intermediate.data = args->scratch; intermediate.dtype = RAD_BF16; intermediate.rank = 2;
    intermediate.shape[0] = rows; intermediate.shape[1] = columns;
    intermediate.stride[0] = columns; intermediate.stride[1] = 1;
    RadTensor gemm_tensors[]{args->t[0], args->t[1], args->t[2], args->t[3], intermediate};
    RadParam gemm_params[]{RAD_INT("M", rad_args_geti_or(args, "M", 0)), RAD_INT("N", columns),
        RAD_INT("K", 2560), RAD_INT("group", 128), RAD_STR("dtype", "i8a8h"), RAD_INT("impl", rows > 2 ? 1 : 2)};
    RadArgs gemm = *args;
    gemm.t = gemm_tensors; gemm.n_t = 5; gemm.p = gemm_params; gemm.n_p = 6;
    status = iqr ? r3d_rad_gemm_iqr_scalar(&gemm, stream) : r3dh_i8_rad_gemm_nt_q(&gemm, stream);
    if (status) return status;
    RadTensor gate_tensors[]{intermediate, args->t[4], args->t[5], count == 7 ? args->t[6] : RadTensor{}};
    RadParam gate_params[]{RAD_INT("n", columns / 2), RAD_INT("group", 128),
        RAD_STR("dtype", "bf16"), RAD_STR("act", "silu")};
    RadArgs gate = *args;
    gate.t = gate_tensors; gate.n_t = count == 7 ? 4 : 3; gate.p = gate_params; gate.n_p = 4;
    return r3dh_qwen_ops_gated_had_quant_i8g(&gate, stream);
}
