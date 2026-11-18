#pragma once
#include "rad_plugin.h"
#include "rad_sample.h"
extern "C" {
int r3dh_sample_merge_topk(const RadArgs*, RadStream);
int r3dh_sample_merge_shape(const RadParam*, int, int, RadOpdDesc*);
const RadKernelInfo* r3dh_sample_merge_kernels(int*);
int r3dh_sample_merge_kernel_count();
const RadKernelInfo* r3dh_sample_merge_kernel_at(int);
const RadOpSchema* r3dh_sample_merge_schemas(int*);
int r3dh_sample_merge_schema_count();
const RadOpSchema* r3dh_sample_merge_schema_at(int);
}
