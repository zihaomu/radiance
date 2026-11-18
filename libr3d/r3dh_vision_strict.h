#pragma once
#include "rad_abi.h"
#include "rad_plugin.h"
#include "r3dh_vision_strict_math.h"
extern "C" {
const RadKernelInfo* r3dh_vision_strict_kernels(int*);
const RadOpSchema* r3dh_vision_strict_schemas(int*);
int r3dh_vision_strict_gemm_nt(const RadArgs*, RadStream);
int r3dh_vision_strict_gemm_nt_bias(const RadArgs*, RadStream);
int r3dh_vision_strict_attn_dense(const RadArgs*, RadStream);
int r3dh_vision_strict_shape(const char*, const RadParam*, int, int, RadOpdDesc*);
int r3dh_vision_strict_geometry(int64_t, int64_t);
}
