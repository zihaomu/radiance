#ifndef R3DH_SPARSE_QSA_INTERNAL_H
#define R3DH_SPARSE_QSA_INTERNAL_H
#include "r3dh_sparse_qsa.h"
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <cstdint>
namespace r3dh_sparse {
inline long long integer(const RadArgs* args, const char* key, long long fallback) {
    rad_args_geti(args, key, &fallback);
    return fallback;
}
inline double real(const RadArgs* args, const char* key, double fallback) {
    for (int index = 0; index < args->n_p; ++index)
        if (!std::strcmp(args->p[index].key, key) && args->p[index].kind == RAD_P_F64) return args->p[index].dval;
    return fallback;
}
inline bool string_is(const RadArgs* args, const char* key, const char* value) {
    const char* text = rad_args_gets(args, key);
    return !text || !std::strcmp(text, value);
}
inline bool tensor(const RadTensor& value, unsigned dtype, int rank) {
    if (!value.data || value.dtype != dtype || int(value.rank) != rank) return false;
    int64_t extent = 1;
    for (int dimension = rank - 1; dimension >= 0; --dimension) {
        if (value.shape[dimension] < 1 || value.shape[dimension] > (1LL << 30)
            || value.stride[dimension] < extent || value.stride[dimension] > (1LL << 30)) return false;
        extent = value.shape[dimension] * value.stride[dimension];
        if (extent > (1LL << 30)) return false;
    }
    return value.stride[rank - 1] == 1;
}
inline bool page_tensor(const RadTensor& value) {
    if (value.rank != 2 && value.rank != 4) return false;
    if (!tensor(value, RAD_BF16, value.rank)) return false;
    int64_t width = 1;
    for (int dimension = int(value.rank) - 1; dimension > 0; --dimension) {
        if (value.stride[dimension] != width) return false;
        width *= value.shape[dimension];
    }
    return width == 128;
}
inline bool sections(const RadArgs* args, int* output) {
    const char* cursor = rad_args_gets(args, "sections");
    if (!cursor) { output[0] = 11; output[1] = 11; output[2] = 10; return true; }
    for (int component = 0; component < 3; ++component) {
        char* end = nullptr;
        long value = std::strtol(cursor, &end, 10);
        if (end == cursor || value < 0 || value > 64) return false;
        output[component] = int(value);
        cursor = end;
    }
    while (*cursor == ' ') ++cursor;
    return !*cursor;
}
}
#endif
