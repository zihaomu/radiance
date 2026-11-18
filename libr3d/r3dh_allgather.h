#pragma once
#include "rad_plugin.h"

extern "C" int r3dh_allgather_init(const RadParam*, int, int, int, void**);
extern "C" void r3dh_allgather_fini(void*);
extern "C" int r3dh_allgather_launch(const RadArgs*, RadStream);
extern "C" int r3dh_allgather_shape(const RadParam*, int, int, RadOpdDesc*);
extern "C" int r3dh_allgather_kernel_count();
extern "C" int r3dh_allgather_schema_count();
extern "C" const RadKernelInfo* r3dh_allgather_kernels(int*);
extern "C" const RadKernelInfo* r3dh_allgather_kernel_at(int);
extern "C" const RadOpSchema* r3dh_allgather_schemas(int*);
extern "C" const RadOpSchema* r3dh_allgather_schema_at(int);
extern "C" int r3dh_allgather_status(void*, unsigned*);
extern "C" int r3dh_allgather_seed(void*, unsigned);
extern "C" int r3dh_allgather_validate(const RadArgs*);
