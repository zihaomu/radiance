#pragma once
#include <hip/hip_runtime.h>
#include "rad_plugin.h"

extern "C" int r3dh_ar_launch(void* peer_scratch, void* my_scratch,
    unsigned* peer_flags, unsigned* my_flags, unsigned* seq_ctrs,
    long slot_stride16, const void* input, void* output, long numel,
    int dtype, hipStream_t stream, int blocks, int threads, int drain, int acquire);
extern "C" int r3dh_rad_ar_init(const RadParam*, int, int, int, void**);
extern "C" void r3dh_rad_ar_fini(void*);
extern "C" int64_t r3dh_rad_ar_scratch(const RadArgs*);
extern "C" int r3dh_rad_ar_oneshot_2rank_exact(const RadArgs*, RadStream);
