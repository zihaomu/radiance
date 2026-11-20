#include "r3dh_ar_exact_body.h"
#include <climits>

extern "C" int r3dh_ar_launch(void* peer_scratch, void* my_scratch,
    unsigned* peer_flags, unsigned* my_flags, unsigned* seq_ctrs,
    long slot_stride16, const void* input, void* output, long numel,
    int dtype, hipStream_t stream, int blocks, int threads, int drain, int acquire) {
    if (dtype < 0 || dtype > 2) return RAD_E_DTYPE;
    const long element_bytes = dtype == 2 ? 4 : 2;
    if (numel <= 0 || numel > LONG_MAX / element_bytes) return RAD_E_SHAPE;
    const long bytes = numel * element_bytes;
    if (bytes % 16 || bytes / 16 > INT_MAX - 1024 || slot_stride16 < bytes / 16 ||
        slot_stride16 > INT_MAX - 1024) return RAD_E_SHAPE;
    if (!peer_scratch || !my_scratch || !peer_flags || !my_flags || !seq_ctrs ||
        !input || !output) return RAD_E_INVAL;
    if (((uintptr_t)input | (uintptr_t)output | (uintptr_t)peer_scratch |
         (uintptr_t)my_scratch) & 15u) return RAD_E_STRIDE;
    if (blocks < 1 || blocks > 512 || threads < 64 || threads > 1024 ||
        threads % 32 || drain < 0 || drain > 4 || acquire < 0 || acquire > 1)
        return RAD_E_INVAL;
    const int words = (int)(bytes / 16);
    if (blocks > words) blocks = words;
#define R3DH_AR_RUN(Type, Lanes) \
    r3dh_ar_oneshot_2rank_exact_kernel<Type, Lanes><<<blocks, threads, 0, stream>>>( \
        (const int4*)input, (int4*)peer_scratch, (const int4*)my_scratch, (int4*)output, \
        peer_flags, my_flags, seq_ctrs, words, (int)slot_stride16, drain, acquire, 512u)
    if (dtype == 0) { R3DH_AR_RUN(__hip_bfloat16, 8); }
    else if (dtype == 1) { R3DH_AR_RUN(__half, 8); }
    else { R3DH_AR_RUN(float, 4); }
#undef R3DH_AR_RUN
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
