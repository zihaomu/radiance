#pragma once
#include "r3dh_moe_iqr.h"
#include "r3dh_moe_w4nl.h"
extern "C" {
int r3dh_moe_mixed_bf16_shape(const RadParam*, int, int, RadOpdDesc*);
int r3dh_moe_mixed_bf16_launch(const RadArgs*, RadStream);
int r3dh_moe_mixed_iqr_shape(const RadParam*, int, int, RadOpdDesc*);
int r3dh_moe_mixed_iqr_layout(const RadParam*, int, int, const RadEncoding*,
    const int*, const RadTensor*, int, RadLayout*);
int r3dh_moe_mixed_iqr_relayout(const RadParam*, int, int, const RadEncoding*,
    const int*, const RadTensor*, int, void*, int64_t);
int r3dh_moe_mixed_iqr_unrelayout(const RadParam*, int, int, const RadEncoding*,
    const int*, const void*, int64_t, const RadTensor*, int);
int r3dh_moe_mixed_iqr_launch(const RadArgs*, RadStream);
int r3dh_moe_mixed_kernel_count();
const RadKernelInfo* r3dh_moe_mixed_kernel_at(int);
}
namespace r3dh_moe_mixed {
int config(const RadParam*, int, r3dh_moe_iqr::Config*);
}
