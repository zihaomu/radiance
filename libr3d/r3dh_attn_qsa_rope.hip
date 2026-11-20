#include <hip/hip_runtime.h>
#include "r3dh_attn_qsa.h"
#include <cmath>
#include <cstring>
#if defined(__FAST_MATH__)
#error r3dh_attn_qsa requires strict floating point
#endif
namespace {
long integer(const RadArgs* args, const char* name, long fallback) {
    long long value = fallback;
    rad_args_geti(args, name, &value);
    return value;
}
double real(const RadArgs* args, const char* name, double fallback) {
    for (int index = 0; index < args->n_p; ++index)
        if (args->p[index].kind == RAD_P_F64 && !std::strcmp(args->p[index].key, name))
            return args->p[index].dval;
    return fallback;
}
__global__ void table_kernel(const int* positions, float* table, long count,
                             long rows, float theta, float scale) {
    const long token = blockIdx.x;
    const int column = threadIdx.x;
    if (token >= count || column >= 32) return;
    const int position = positions[token];
    if (position < 0 || position >= rows) return;
    const float frequency = 1.0f / float(pow(double(theta), double(float(2 * column) / 64.0f)));
    const float angle = (float(position) / scale) * frequency;
    table[long(position) * 64 + column] = float(cos(double(angle)));
    table[long(position) * 64 + 32 + column] = float(sin(double(angle)));
}
}
extern "C" int r3dh_attn_qsa_rope_table(const RadArgs* args, RadStream stream) {
    if (!args || args->n_t != 2 || !args->t) return RAD_E_INVAL;
    const RadTensor& positions = args->t[0];
    const RadTensor& table = args->t[1];
    const char* mode = rad_args_gets(args, "mode");
    const char* ladder = rad_args_gets(args, "ladder");
    if (integer(args, "rot", 0) != 64 || real(args, "yarn_factor", 0) != 0
        || real(args, "llama3_factor", 0) != 0
        || (rad_args_gets(args, "dtype") && std::strcmp(rad_args_gets(args, "dtype"), "f32"))
        || (mode && std::strcmp(mode, "neox"))
        || (ladder && std::strcmp(ladder, "rotary"))) return RAD_E_UNSUPPORTED;
    if (!positions.data || positions.dtype != RAD_I32 || positions.rank != 1
        || positions.shape[0] < 1 || positions.shape[0] > 2048 || positions.stride[0] != 1
        || !table.data || table.dtype != RAD_F32 || table.rank != 2 || table.shape[0] < 1
        || table.shape[1] != 64 || table.stride[0] != 64 || table.stride[1] != 1)
        return RAD_E_SHAPE;
    const float theta = float(real(args, "theta", 10000));
    const float scale = float(real(args, "scale", 1));
    if (!std::isfinite(theta) || theta <= 0 || !std::isfinite(scale) || scale <= 0)
        return RAD_E_INVAL;
    hipLaunchKernelGGL(table_kernel, dim3(positions.shape[0]), dim3(32), 0,
        reinterpret_cast<hipStream_t>(stream), static_cast<const int*>(positions.data),
        static_cast<float*>(table.data), positions.shape[0], table.shape[0], theta, scale);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
