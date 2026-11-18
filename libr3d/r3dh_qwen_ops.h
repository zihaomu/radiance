#pragma once
#include "rad_abi.h"
#include "rad_plugin.h"
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__)
#error Qwen ops require strict floating point compilation
#endif
extern "C" {
int r3dh_qwen_ops_gated_had_quant_i8g(const RadArgs*, RadStream);
int r3dh_qwen_ops_gate_had_quant_i8g(const RadArgs*, RadStream);
int r3dh_qwen_ops_router_gemm(const RadArgs*, RadStream);
}
