#pragma once
#include "rad_abi.h"
#include "rad_plugin.h"
#include <cstdint>
namespace r3dh_i8 {
constexpr uint32_t weight_dtype = 0x33880100u;
constexpr uint32_t scale_dtype = 0x33880200u;
inline int64_t packed_index(int64_t column, int64_t position, int64_t reduction) {
    return (((column / 16 * (reduction / 128) + position / 128) * 8
             + position % 128 / 16) * 16 + column % 16) * 16 + position % 16;
}
}
extern "C" {
int r3dh_i8_rad_gemm_nt_q(const RadArgs*, RadStream);
int r3dh_i8_rad_quant_act_i8g(const RadArgs*, RadStream);
int r3dh_i8_rad_had_quant_act_i8g(const RadArgs*, RadStream);
int r3dh_i8_shape_gemm_nt_q(const RadParam*, int, int, RadOpdDesc*);
int r3dh_i8_shape_quant_act_i8g(const RadParam*, int, int, RadOpdDesc*);
int r3dh_i8_rad_layout(const RadParam*, int, int, const RadEncoding*,
                        const int*, const RadTensor*, int, RadLayout*);
int r3dh_i8_rad_relayout(const RadParam*, int, int, const RadEncoding*,
                          const int*, const RadTensor*, int, void*, int64_t);
int r3dh_i8_rad_unrelayout(const RadParam*, int, int, const RadEncoding*,
                            const int*, const void*, int64_t, const RadTensor*, int);
}
