#pragma once
#include <hip/hip_runtime.h>
#include "rad_plugin.h"

extern "C" int r3dh_ar_launch(void* peer_scratch, void* my_scratch,
    unsigned* peer_flags, unsigned* my_flags, unsigned* seq_ctrs,
    long slot_stride16, const void* input, void* output, long numel,
    int dtype, hipStream_t stream, int blocks, int threads, int drain, int acquire);
extern "C" int r3dh_ar_hc_write_launch(void* peer_scratch, void* my_scratch,
    unsigned* peer_flags, unsigned* my_flags, unsigned* seq_ctrs, long slot_stride16,
    void* y, const void* inj, void* h, long rows, long n, long hc, long irow, long hrow,
    hipStream_t stream, const void* gather = nullptr);
// A GatherRow is at most this many bytes.
constexpr int r3dh_gather_row_bytes = 64;
extern "C" int r3dh_ar_gather_row(void* out, const void* ye, long ye_ld, const void* ew, int f32w,
    const int32_t* sorted, int T, int top_k, const void* sh, const void* sg, int sig);
extern "C" int r3dh_rad_ar_hc_init(const RadParam*, int, int, int, void**);
extern "C" int r3dh_rad_ar_hc_write(const RadArgs*, RadStream);
extern "C" int r3dh_rad_ar_gather_hc_write(const RadArgs*, RadStream);
extern "C" int r3dh_rad_ar_init(const RadParam*, int, int, int, void**);
extern "C" void r3dh_rad_ar_fini(void*);
extern "C" int64_t r3dh_rad_ar_scratch(const RadArgs*);
extern "C" int r3dh_rad_ar_oneshot_2rank_exact(const RadArgs*, RadStream);
