/* Derived from libr4d/r4d_ar_oneshot_2rank_exact.hip by the libhalo rename transform (2026-10-26),
 * then ported to gfx1151; see libhalo/README.md. */
#include "r3dh_ar.h"
#include "r3dh_ar_seq.h"
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

#define R3DH_AR_SPIN_MAX 4000000000ULL
#define R3DH_AR_MAX_BLOCKS 512
#define HIPOK(e) do { hipError_t _e = (e); if (_e != hipSuccess) \
    throw std::runtime_error(std::string("hip: ") + hipGetErrorString(_e)); } while (0)

__device__ __forceinline__ void store_sys_rel(unsigned int* p, unsigned int v) {
  __hip_atomic_store(p, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ unsigned int load_sys_acq(const unsigned int* p) {
  return __hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ __forceinline__ float rd_to_f(const float& x) { return x; }
__device__ __forceinline__ float rd_to_f(const __half& x) { return __half2float(x); }
__device__ __forceinline__ float rd_to_f(const __hip_bfloat16& x) { return (float)x; }
__device__ __forceinline__ void rd_from_f(float v, float& o) { o = v; }
__device__ __forceinline__ void rd_from_f(float v, __half& o) { o = __float2half(v); }
__device__ __forceinline__ void rd_from_f(float v, __hip_bfloat16& o) { o = (__hip_bfloat16)v; }

// Vectorised 128-bit add: an int4 packs LANES elements of T, summed in fp32.
template <typename T, int LANES>
__device__ __forceinline__ int4 vadd(const int4& a, const int4& b) {
  int4 r;
  const T* pa = reinterpret_cast<const T*>(&a);
  const T* pb = reinterpret_cast<const T*>(&b);
  T* pr = reinterpret_cast<T*>(&r);
#pragma unroll
  for (int j = 0; j < LANES; ++j) rd_from_f(rd_to_f(pa[j]) + rd_to_f(pb[j]), pr[j]);
  return r;
}

// Multi-block one-shot all-reduce. Each block owns words [start,end) of the 16B-word space and
// drives its own flag handshake; no cross-block sync.
template <typename T, int LANES>
__global__ void r3dh_ar_oneshot_2rank_exact_kernel(
    const int4* in4, int4* peer_base, const int4* my_base, int4* out4,
    unsigned int* peer_flags, unsigned int* my_flags, unsigned int* seq_ctrs,
    int n16, int slot_stride16, int drain, int acq, unsigned int nbw) {
  const int b = blockIdx.x, nb = gridDim.x, tid = threadIdx.x, nt = blockDim.x;
  __shared__ unsigned int s_seq, s_slot;
  r3dh_ar_seq_take(seq_ctrs, nbw, s_seq, s_slot);   // device-resident, replay-safe
  const unsigned int s = s_seq;
  const int slot = (int)s_slot;
  int4* ps = const_cast<int4*>(my_base) + (size_t)slot * slot_stride16;
  const int4* ms = peer_base + (size_t)slot * slot_stride16;
  auto read_peer = [&](int index) {
    const volatile int* words = reinterpret_cast<const volatile int*>(ms + index);
    return make_int4(words[0], words[1], words[2], words[3]);
  };

  const int w = (n16 + nb - 1) / nb;      // 16B words per block
  const int start = b * w;
  int end = start + w; if (end > n16) end = n16;

  if (drain == 4) {
    if (tid == 0) store_sys_rel(&peer_flags[b], s);
    const unsigned long long t0 = wall_clock64();
    while (wall_clock64() - t0 < 2000ull) __builtin_amdgcn_s_sleep(8);   // 100 MHz: 20 us
  }
  {
    int k = start + tid;
    for (; k + 3 * nt < end; k += 4 * nt) {
      int4 v0 = in4[k], v1 = in4[k + nt], v2 = in4[k + 2 * nt], v3 = in4[k + 3 * nt];
      ps[k] = v0; ps[k + nt] = v1; ps[k + 2 * nt] = v2; ps[k + 3 * nt] = v3;
    }
    for (; k < end; k += nt) ps[k] = in4[k];
  }
  __threadfence_system();
  if (drain == 1) __threadfence_system();
  else if (drain == 3) asm volatile("s_waitcnt vmcnt(0)" ::: "memory");  // pre-gfx12 store wait
  __syncthreads();
  if (tid == 0) {
    if (drain == 2) __threadfence_system();
    if (drain != 4) store_sys_rel(&peer_flags[b], s);
    store_sys_rel(&peer_flags[nbw + (unsigned)slot * nbw + b], s);
    const auto start_wait = wall_clock64();
    while (load_sys_acq(&my_flags[nbw + (unsigned)slot * nbw + b]) != s) {
      if (wall_clock64() - start_wait > 200000000ull) __builtin_trap();
      __builtin_amdgcn_s_sleep(1);
    }
  }
  __syncthreads();
  (void)acq;
  __threadfence_system();
  {
    int k = start + tid;
    for (; k + 3 * nt < end; k += 4 * nt) {
      out4[k]          = vadd<T, LANES>(in4[k],          read_peer(k));
      out4[k + nt]     = vadd<T, LANES>(in4[k + nt],     read_peer(k + nt));
      out4[k + 2 * nt] = vadd<T, LANES>(in4[k + 2 * nt], read_peer(k + 2 * nt));
      out4[k + 3 * nt] = vadd<T, LANES>(in4[k + 3 * nt], read_peer(k + 3 * nt));
    }
    for (; k < end; k += nt) out4[k] = vadd<T, LANES>(in4[k], read_peer(k));
  }
}

