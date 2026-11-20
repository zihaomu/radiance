#pragma once
#include <cstdint>
#include <cmath>
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__)
#error "r3dh_vision_strict requires strict floating point"
#endif
#ifdef __HIPCC__
#define R3DH_VISION_HD __host__ __device__
#else
#define R3DH_VISION_HD
#endif
namespace r3dh_vision_strict {
R3DH_VISION_HD inline float widen(uint16_t value) {
    union Bits { uint32_t bits; float value; } result;
    result.bits = uint32_t(value) << 16;
    return result.value;
}
R3DH_VISION_HD inline uint16_t narrow(float value) {
    union Bits { float value; uint32_t bits; } result;
    result.value = value;
    if ((result.bits & 0x7fffffffu) > 0x7f800000u)
        return uint16_t((result.bits >> 16) | 0x40u);
    return uint16_t((result.bits + 0x7fffu + ((result.bits >> 16) & 1u)) >> 16);
}
R3DH_VISION_HD inline float add_product(float sum, float left, float right) {
    volatile float product = left * right;
    return sum + product;
}
R3DH_VISION_HD inline uint16_t product(const uint16_t* left, const uint16_t* right,
                                      int64_t depth, int64_t left_step = 1,
                                      int64_t right_step = 1, float bias = 0.0f) {
    float sum = 0.0f;
    for (int64_t inner = 0; inner < depth; ++inner)
        sum = add_product(sum, widen(left[inner * left_step]), widen(right[inner * right_step]));
    return narrow(sum + bias);
}
}
