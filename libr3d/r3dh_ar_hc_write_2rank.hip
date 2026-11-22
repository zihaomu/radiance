// ar_hc_write at two gfx1100 ranks, exact wire: the native peer-pull all-reduce of y
// (r3dh_ar_exact_body.h's transport, handshake and bf16 rounding) with hc_write as its epilogue,
// h[c][i] = bf16(fmaf(inj[c], y[i], h[c][i])) over the reduced bf16 y. Byte-identical to
// all_reduce + hc_write: each word is summed exactly as the plain kernel sums it, and the write
// is hc_write_kernel's arithmetic on the stored value.
#include "r3dh_ar_exact_body.h"
#include <algorithm>
#include <climits>

namespace {
__device__ __forceinline__ float hcw_bf2f(unsigned short v) {
  return __builtin_bit_cast(float, (unsigned)v << 16);
}
__device__ __forceinline__ unsigned short hcw_f2bf(float f) {
  const unsigned u = __builtin_bit_cast(unsigned, f);
  return (unsigned short)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

// moe_gather of one row (decode): gather_kernel's arithmetic for the eight columns of a word.
struct GatherRow {
  const uint16_t* ye; const void* ew; const int32_t* sorted; const uint16_t* sh; const uint16_t* sg;
  long ye_ld; int T, top_k, f32w, sig;
};
__device__ __forceinline__ uint16_t gather_f2bf(float f) {
  const unsigned u = __builtin_bit_cast(unsigned, f);
  if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x0040u);
  return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
__device__ __forceinline__ int4 gather_word(const GatherRow& g, const int* slots, const float* weights,
                                            float gate, int k) {
  float acc[8] = {};
  for (int j = 0; j < g.top_k; ++j) {
    const int i = slots[j];
    if (i < 0) continue;
    const float wj = weights[j];
    const int4 v = *reinterpret_cast<const int4*>(g.ye + (long)i * g.ye_ld + (long)k * 8);
    const unsigned short* e = reinterpret_cast<const unsigned short*>(&v);
#pragma unroll
    for (int c = 0; c < 8; ++c) acc[c] += wj * hcw_bf2f(e[c]);
  }
  int4 out;
  unsigned short* o = reinterpret_cast<unsigned short*>(&out);
  if (!g.sh) {
#pragma unroll
    for (int c = 0; c < 8; ++c) o[c] = gather_f2bf(acc[c]);
    return out;
  }
  const int4 sv = *reinterpret_cast<const int4*>(g.sh + (long)k * 8);
  const unsigned short* sh = reinterpret_cast<const unsigned short*>(&sv);
#pragma unroll
  for (int c = 0; c < 8; ++c) {
    const uint16_t p = gather_f2bf(hcw_bf2f(sh[c]) * gate);
    o[c] = gather_f2bf(hcw_bf2f(p) + hcw_bf2f(gather_f2bf(acc[c])));
  }
  return out;
}

template <bool Gather>
__global__ void r3dh_ar_hc_write_2rank_kernel(
    int4* y4, int4* peer_base, const int4* my_base,
    unsigned int* peer_flags, unsigned int* my_flags, unsigned int* seq_ctrs,
    const unsigned short* __restrict__ inj, unsigned short* __restrict__ h,
    int n16, int n, int hc, long irow, long hrow, int slot_stride16, unsigned int nbw, GatherRow g) {
  const int b = blockIdx.x, nb = gridDim.x, tid = threadIdx.x, nt = blockDim.x;
  __shared__ unsigned int s_seq, s_slot;
  __shared__ int s_i[64];
  __shared__ float s_w[64];
  float gate = 0.0f;
  if constexpr (Gather) {
    for (int j = tid; j < g.top_k; j += nt) s_i[j] = -1;
    __syncthreads();
    for (int i = tid; i < g.T; i += nt) {
      const int f = g.sorted[i];
      if (f >= 0 && f < g.top_k) s_i[f] = i;
    }
    for (int j = tid; j < g.top_k; j += nt)
      s_w[j] = g.f32w ? static_cast<const float*>(g.ew)[j] : hcw_bf2f(static_cast<const unsigned short*>(g.ew)[j]);
    if (g.sh) {
      const float v0 = hcw_bf2f(g.sg[0]);
      gate = g.sig ? (1.0f / (1.0f + __expf(-v0))) : v0;
    }
    __syncthreads();
  }
  r3dh_ar_seq_take(seq_ctrs, nbw, s_seq, s_slot);
  const unsigned int s = s_seq;
  const int slot = (int)s_slot;
  int4* ps = const_cast<int4*>(my_base) + (size_t)slot * slot_stride16;
  const int4* ms = peer_base + (size_t)slot * slot_stride16;

  const int w = (n16 + nb - 1) / nb;
  const int start = b * w;
  int end = start + w; if (end > n16) end = n16;

  for (int k = start + tid; k < end; k += nt) {
    if constexpr (Gather) {
      const int4 word = gather_word(g, s_i, s_w, gate, k);
      y4[k] = word;
      ps[k] = word;
    } else {
      ps[k] = y4[k];
    }
  }
  __threadfence_system();
  __threadfence_system();
  __syncthreads();
  if (tid == 0) {
    store_sys_rel(&peer_flags[b], s);
    store_sys_rel(&peer_flags[nbw + (unsigned)slot * nbw + b], s);
    const auto start_wait = wall_clock64();
    while (load_sys_acq(&my_flags[nbw + (unsigned)slot * nbw + b]) != s) {
      if (wall_clock64() - start_wait > 200000000ull) __builtin_trap();
      __builtin_amdgcn_s_sleep(1);
    }
  }
  __syncthreads();
  __threadfence_system();
  for (int k = start + tid; k < end; k += nt) {
    const volatile int* pw = reinterpret_cast<const volatile int*>(ms + k);
    const int4 sum = vadd<__hip_bfloat16, 8>(y4[k], make_int4(pw[0], pw[1], pw[2], pw[3]));
    y4[k] = sum;
    const long e = (long)k * 8;
    const long r = e / n, i = e - r * n;
    const unsigned short* sv = reinterpret_cast<const unsigned short*>(&sum);
    float v[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) v[j] = hcw_bf2f(sv[j]);
    const unsigned short* ip = inj + r * irow;
    unsigned short* hp = h + r * hrow + i;
    for (int c = 0; c < hc; ++c) {
      const float g = hcw_bf2f(ip[c]);
      int4* hq = reinterpret_cast<int4*>(hp + (long)c * n);
      int4 hv = *hq;
      unsigned short* hs = reinterpret_cast<unsigned short*>(&hv);
#pragma unroll
      for (int j = 0; j < 8; ++j) hs[j] = hcw_f2bf(fmaf(g, v[j], hcw_bf2f(hs[j])));
      *hq = hv;
    }
  }
}
}  // namespace

extern "C" int r3dh_ar_hc_write_launch(void* peer_scratch, void* my_scratch,
    unsigned* peer_flags, unsigned* my_flags, unsigned* seq_ctrs, long slot_stride16,
    void* y, const void* inj, void* h, long rows, long n, long hc, long irow, long hrow,
    hipStream_t stream, const void* gather) {
  if (rows < 1 || n < 8 || n % 8 || hc < 1 || hc > 8 || irow < hc || hrow < n * hc || hrow % 8)
    return RAD_E_SHAPE;
  if (rows > LONG_MAX / n) return RAD_E_SHAPE;
  const long words = rows * n / 8;
  if (words > INT_MAX - 1024 || slot_stride16 < words || slot_stride16 > INT_MAX - 1024)
    return RAD_E_SHAPE;
  if (!peer_scratch || !my_scratch || !peer_flags || !my_flags || !seq_ctrs || !y || !inj || !h)
    return RAD_E_INVAL;
  if (((uintptr_t)y | (uintptr_t)h | (uintptr_t)peer_scratch | (uintptr_t)my_scratch) & 15u)
    return RAD_E_STRIDE;
  const int blocks = (int)std::min(24L, std::max(1L, (words + 255) / 256));
  if (gather)
    r3dh_ar_hc_write_2rank_kernel<true><<<blocks, 256, 0, stream>>>(
        (int4*)y, (int4*)peer_scratch, (const int4*)my_scratch, peer_flags, my_flags, seq_ctrs,
        (const unsigned short*)inj, (unsigned short*)h, (int)words, (int)n, (int)hc, irow, hrow,
        (int)slot_stride16, 512u, *static_cast<const GatherRow*>(gather));
  else
    r3dh_ar_hc_write_2rank_kernel<false><<<blocks, 256, 0, stream>>>(
        (int4*)y, (int4*)peer_scratch, (const int4*)my_scratch, peer_flags, my_flags, seq_ctrs,
        (const unsigned short*)inj, (unsigned short*)h, (int)words, (int)n, (int)hc, irow, hrow,
        (int)slot_stride16, 512u, GatherRow{});
  return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}

// Fills the GatherRow a decode ar_gather_hc_write passes to r3dh_ar_hc_write_launch.
extern "C" int r3dh_ar_gather_row(void* out, const void* ye, long ye_ld, const void* ew, int f32w,
    const int32_t* sorted, int T, int top_k, const void* sh, const void* sg, int sig) {
  if (!out || !ye || !ew || !sorted || top_k < 1 || top_k > 64 || T < 0 || ye_ld % 8
      || ((uintptr_t)ye & 15) || (sh && ((uintptr_t)sh & 15)) || (sh && !sg)) return RAD_E_UNSUPPORTED;
  *static_cast<GatherRow*>(out) = GatherRow{static_cast<const uint16_t*>(ye), ew, sorted,
      static_cast<const uint16_t*>(sh), static_cast<const uint16_t*>(sg), ye_ld, T, top_k, f32w, sig};
  return RAD_OK;
}
