#ifndef R3DH_QSA_KEY_DEVICE_H
#define R3DH_QSA_KEY_DEVICE_H
// The QSA block key (qsa_block_key) per block.
#include <hip/hip_runtime.h>
#include "r3dh_sparse_qsa_internal.h"
#include <cstring>
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
namespace r3dh_qsa_key {
template<class Tensor> __device__ inline float bf_load(const Tensor& value, int64_t offset) {
    return __uint_as_float(unsigned(static_cast<const uint16_t*>(value.data)[offset]) << 16);
}
__device__ inline uint16_t bf_round(float value) {
    unsigned bits = __float_as_uint(value);
    if ((bits & 0x7fffffffU) > 0x7f800000U) return uint16_t((bits >> 16) | 64);
    return uint16_t((bits + 0x7fffU + ((bits >> 16) & 1)) >> 16);
}
__device__ inline float bf_value(float value) { return __uint_as_float(unsigned(bf_round(value)) << 16); }
struct KeyCall {
    RadTensor keys, weights, pages, positions, output;
    int blocks, rot, mode, sections[3];
    int64_t pos0;
    float eps, wadd, theta, scale;
};
// key_kernel with one thread per lane (128 threads); the double square sum stays one ordered chain.
template<class Call> __device__ inline void block_key(const Call& call, int block, int lane) {
    const int page = call.pages.data ? static_cast<const int*>(call.pages.data)[block * call.pages.stride[0]] : block;
    if (page < 0 || page >= call.output.shape[0]) return;
    __shared__ float pooled[128], normed[128];
    __shared__ float inverse;
    float sum = 0;
    for (int token = 0; token < 4; ++token) sum += bf_load(call.keys, (block * 4 + token) * call.keys.stride[0] + lane);
    pooled[lane] = bf_value(sum / 4.0f);
    __syncthreads();
    if (lane == 0) {
        double square = 0;
        for (int index = 0; index < 128; ++index) square += double(pooled[index]) * double(pooled[index]);
        inverse = 1.0f / sqrtf(float(square / 128.0) + call.eps);
    }
    __syncthreads();
    normed[lane] = pooled[lane] * inverse * (call.wadd + static_cast<const float*>(call.weights.data)[lane]);
    __syncthreads();
    const int half = call.rot / 2;
    float value = normed[lane];
    if (lane < call.rot) {
        const int frequency = lane % half;
        int component = 0;
        if (call.mode == 2) component = frequency < call.sections[0] ? 0 : frequency < call.sections[0] + call.sections[1] ? 1 : 2;
        if (call.mode == 3) {
            if (frequency % 3 == 1 && frequency < 3 * call.sections[1]) component = 1;
            if (frequency % 3 == 2 && frequency < 3 * call.sections[2]) component = 2;
        }
        const int64_t position = call.positions.data
            ? static_cast<const int*>(call.positions.data)[block * call.positions.stride[0] + component]
            : call.pos0 + block * 4;
        const float inverse_frequency = 1.0f / float(pow(double(call.theta), double(float(2 * frequency) / float(call.rot))));
        const float angle = (float(position) / call.scale) * inverse_frequency;
        const float cosine = float(cos(double(angle))), sine = float(sin(double(angle)));
        value = lane < half ? normed[lane] * cosine - normed[lane + half] * sine
                            : normed[lane - half] * sine + normed[lane] * cosine;
    }
    static_cast<uint16_t*>(call.output.data)[page * call.output.stride[0] + lane] = bf_round(value);
}
// A validated qsa_block_key argument list (r3dh_sparse_qsa_shape) as the kernel's call.
inline KeyCall key_call(const RadArgs* args) {
    KeyCall call{};
    call.keys = args->t[0]; call.weights = args->t[1]; call.pages = args->t[2];
    call.positions = args->t[3]; call.output = args->t[4];
    call.blocks = call.pages.data ? call.pages.shape[0] : call.output.shape[0];
    call.rot = r3dh_sparse::integer(args, "rotary_dim", 0);
    call.pos0 = r3dh_sparse::integer(args, "pos0", -1);
    call.eps = r3dh_sparse::real(args, "eps", 1e-6); call.wadd = r3dh_sparse::real(args, "wadd", 1);
    call.theta = r3dh_sparse::real(args, "theta", 10000); call.scale = r3dh_sparse::real(args, "scale", 1);
    r3dh_sparse::sections(args, call.sections);
    const char* mode = rad_args_gets(args, "mode");
    if (call.positions.rank == 2 && call.positions.shape[1] == 3 && mode && rad_args_gets(args, "sections"))
        call.mode = !std::strcmp(mode, "mrope") ? 2 : !std::strcmp(mode, "imrope") ? 3 : 0;
    return call;
}
}
#endif
