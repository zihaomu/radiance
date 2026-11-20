#pragma once
#include "rad_abi.h"
#include "rad_plugin.h"
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__)
#error r3dh_vision_rope requires strict floating point
#endif
extern "C" {
int r3dh_vision_rope(const RadArgs*, RadStream);
int r3dh_vision_rope_qk(const RadArgs*, RadStream);
int r3dh_vision_rope_shape(const RadParam*, int, int, RadOpdDesc*);
const RadKernelInfo* r3dh_vision_rope_kernel();
const RadOpSchema* r3dh_vision_rope_schema();
int r3d_vision_schema_count();
const RadOpSchema* r3d_vision_schema_at(int);
int r3d_vision_kernel_count();
const RadKernelInfo* r3d_vision_kernel_at(int);
}
