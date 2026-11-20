#ifndef R3DH_ATTN_FUSED_H
#define R3DH_ATTN_FUSED_H
#include "rad_plugin.h"
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__)
#error Strict floating point required
#endif
extern "C" {
int r3dh_attn_fused_workspace_bytes(const RadArgs*, int, uint64_t*);
int r3dh_attn_fused_prepare_store(const RadArgs*, RadStream);
int r3dh_attn_fused_paged_gate(const RadArgs*, RadStream);
int r3dh_attn_fused_kernel_count();
const RadKernelInfo* r3dh_attn_fused_kernel_at(int);
int r3dh_attn_fused_schema_count();
const RadOpSchema* r3dh_attn_fused_schema_at(int);
}
#endif
