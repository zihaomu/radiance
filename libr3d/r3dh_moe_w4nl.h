#pragma once
#include "rad_plugin.h"
#include "rad_encoding.h"
#include <cstddef>
#include <cstdint>
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__)
#error W4NL64 requires -ffp-contract=off -fno-fast-math
#endif
namespace r3dh_moe_w4nl {
constexpr uint32_t codes_type = 0x72346e63u;
constexpr uint32_t scales_type = 0x72346e73u;
constexpr uint32_t codes_magic = 0x344c4e43u;
constexpr uint32_t scales_magic = 0x344c4e53u;
constexpr const char* codes_tag = "r3dh.moe.w4nl64.codes.f32table.v1";
constexpr const char* scales_tag = "r3dh.moe.w4nl64.e4m3.f32level.v1";
struct Config {
    int tokens;
    int columns;
    int inner[2];
    int experts;
    int top_k;
    int activation_columns;
    bool gate = false;
    int gate_columns[2] = {};
};
struct alignas(16) CodesHeader {
    uint32_t magic, version, rows, columns;
    float table[16];
};
struct alignas(16) ScalesHeader {
    uint32_t magic, version, rows, columns;
    float level;
    uint32_t reserved[3];
};
static_assert(sizeof(CodesHeader) == 80);
static_assert(sizeof(ScalesHeader) == 32);
int config(const RadParam*, int, Config*);
}
extern "C" int r3dh_moe_w4nl_layout(const RadParam*, int, int,
    const RadEncoding*, const int*, const RadTensor*, int, RadLayout*);
extern "C" int r3dh_moe_w4nl_transform(const RadParam*, int, int,
    const RadEncoding*, const int*, const RadTensor*, int, void*, int64_t);
extern "C" int r3dh_moe_w4nl_inverse(const RadParam*, int, int,
    const RadEncoding*, const int*, const void*, int64_t, const RadTensor*, int);
extern "C" int r3dh_moe_w4nl_shape(const RadParam*, int, int, RadOpdDesc*);
extern "C" int r3dh_moe_w4nl_launch(const RadArgs*, RadStream);
