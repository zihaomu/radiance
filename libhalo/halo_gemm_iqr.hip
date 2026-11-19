/* gemm_nt_q over the IQR weights (halo_iqr.h) on gfx1151: the rows gemm_iqr_nt_m64 and
 * gemm_iqr_tiled, ported from libr4d/r4d_gemm_iqr.hip.
 *
 * This first gfx1151 form decodes the weight once a call into an int8 row-major plane in the op's
 * scratch (halo_iqr_dec.h's decoders, libr4d's own arithmetic), then runs one iu8 WMMA GEMM over
 * it that folds each 128-K group's int32 sum in f32 with the activation's scale and the weight's
 * (fold_f32) and the row's (row_scale) at the end. The product is the exact int8 dot product per
 * group, folded in f32: within f32 rounding of libr4d's kernels, not their bytes.
 *
 * gfx1151's iu8 WMMA (see halo_gemm_fp8a8.hip's int8 twins): A lane l holds row (l & 15)'s 16
 * consecutive K bytes, B lane l column (l & 15)'s, and accumulator slot e is row 2e + (l >> 4),
 * column (l & 15). */

#include "halo.h"
#include "halo_args.h"
#include "halo_iqr.h"
#include "halo_iqr_dec.h"
#include <cstring>

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdint>

namespace {

using namespace halo_iqr;
using namespace halo_iqr_dev;

constexpr uint32_t kWave = 32, kBlock = 128, kFK = 256;
/* Waves a decode block: one 128-K unit of one 16-row tile each. */
constexpr uint32_t kDecWaves = 4;

typedef int v4i __attribute__((ext_vector_type(4)));
typedef int v8i __attribute__((ext_vector_type(8)));

/* The scratch: the int8 plane [N][K], then the fold scales [N][K/128] f32, then the row scales
 * [N] f32, each 256-aligned. */
struct Plane { int64_t fs_off, rs_off, bytes; };
constexpr Plane plane_of(int64_t N, int64_t K) {
    Plane p{};
    p.fs_off = align256(N * K);
    p.rs_off = p.fs_off + align256(N * (K / 128) * 4);
    p.bytes = p.rs_off + align256(N * 4);
    return p;
}

template <class F>
__global__ __launch_bounds__(kDecWaves * kWave) void iqr_decode_kernel(Ctx c, int8_t* __restrict__ P,
                                                                      float* __restrict__ FS,
                                                                      float* __restrict__ RS) {
    using D = Dec<F>;
    __shared__ __attribute__((aligned(16))) int8_t s_tab[D::kTab];
    const uint32_t tid = threadIdx.x, lane = tid & (kWave - 1), wave = tid / kWave;
    const uint32_t rt = blockIdx.x;
    const uint32_t K = c.nkb * 128u;
    c.tab = s_tab;
    D::stage(s_tab, c, tid, kDecWaves * kWave);
    __syncthreads();
    for (uint32_t kb = blockIdx.y * kDecWaves + wave; kb < c.nkb; kb += gridDim.y * kDecWaves) {
        const typename D::Raw raw = D::fetch(c, rt, kb, lane, true);
        uint32_t w[16];
        D::rows(raw, c, rb_ratio<F>(c, rt * 16u + (lane & 15u), kb), w);
        D::unzero(c, rt, kb, w);
#pragma unroll
        for (uint32_t q = 0; q < 16; ++q)
            *(uint32_t*)(P + (size_t)(rt * 16u + q) * K + kb * 128u + 4u * lane) = w[q];
        if (lane < 16u) FS[(size_t)(rt * 16u + lane) * c.nkb + kb] = fold_f32<F>(c, rt * 16u + lane, kb);
    }
    if (blockIdx.y == 0 && tid < 16u) RS[rt * 16u + tid] = row_scale<F>(c, rt * 16u + tid);
}

/* One wave a 16 x 16 output tile: the int8 dot product of each 128-K group by eight iu8 WMMAs,
 * folded in f32 with the activation's and the weight's group scales; the row scale at the end. */
__global__ __launch_bounds__(4 * kWave) void iqr_i8_gemm_kernel(
        const int8_t* __restrict__ A, const float* __restrict__ As, const int8_t* __restrict__ P,
        const float* __restrict__ FS, const float* __restrict__ RS, __hip_bfloat16* __restrict__ Y,
        uint32_t M, uint32_t N, uint32_t K, uint32_t a_ld, uint32_t sa_ld) {
    const uint32_t lane = threadIdx.x & (kWave - 1), wave = threadIdx.x / kWave;
    const uint32_t nt = blockIdx.x * 4u + wave;
    if (nt * 16u >= N) return;
    const uint32_t slot = lane & 15u, half = lane >> 4;
    const uint32_t n = nt * 16u + slot;
    const uint32_t m0 = blockIdx.y * 16u;
    const uint32_t am = m0 + slot < M ? m0 + slot : M - 1u;
    const uint32_t nkb = K / 128u;
    const int8_t* ap = A + (size_t)am * a_ld;
    const int8_t* wp = P + (size_t)n * K;
    v8f acc = (v8f)0.f;
    for (uint32_t kb = 0; kb < nkb; ++kb) {
        v8i tmp = (v8i)0;
#pragma unroll
        for (uint32_t u = 0; u < 8; ++u) {
            const uint4 av = *(const uint4*)(ap + kb * 128u + u * 16u);
            const uint4 bv = *(const uint4*)(wp + kb * 128u + u * 16u);
            v4i af, bf;
            af[0] = (int)av.x; af[1] = (int)av.y; af[2] = (int)av.z; af[3] = (int)av.w;
            bf[0] = (int)bv.x; bf[1] = (int)bv.y; bf[2] = (int)bv.z; bf[3] = (int)bv.w;
            tmp = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, af, true, bf, tmp, false);
        }
        const float ws = FS[(size_t)n * nkb + kb];
#pragma unroll
        for (uint32_t e = 0; e < 8; ++e) {
            uint32_t mr = m0 + 2u * e + half;
            mr = mr < M ? mr : M - 1u;
            acc[e] = fmaf((float)tmp[e], As[(size_t)mr * sa_ld + kb] * ws, acc[e]);
        }
    }
    const float rs = RS[n];
#pragma unroll
    for (uint32_t e = 0; e < 8; ++e) {
        const uint32_t m = m0 + 2u * e + half;
        if (m < M) Y[(size_t)m * N + n] = (__hip_bfloat16)(acc[e] * rs);
    }
}

/* The decode-shape form (M <= 64): one wave a 16-column tile of the weight, every 128-K unit
 * decoded from the codes straight into the wave's LDS (16 rows x 128 bytes, the decoder's lane
 * layout), read back as iu8 B fragments and multiplied against MT 16-row tiles of A. The weight
 * is read once, its codes at their own width; no scratch plane. Same arithmetic as the
 * decode-then-GEMM pair: each group's exact int32 sum folded in f32 by fold_f32, row_scale last. */
constexpr uint32_t kFusedWaves = 4, kPitch = 136;

/* ONE (M == 1): the activation row and its scales are wave-uniform, read as scalar loads, so
 * the next unit's codes and fold scale, issued a unit ahead, are not waited on with them. */
template <class F, uint32_t MT, bool ONE = false, uint32_t KS = 1>
__global__ __launch_bounds__(kFusedWaves * kWave) void iqr_fused_kernel(
        Ctx c, const int8_t* __restrict__ A, const float* __restrict__ As,
        __hip_bfloat16* __restrict__ Y, uint32_t M, uint32_t N, uint32_t a_ld, uint32_t sa_ld) {
    using D = Dec<F>;
    __shared__ __attribute__((aligned(16))) int8_t s_tab[D::kTab];
    __shared__ __attribute__((aligned(16))) uint8_t s_tr[kFusedWaves][16 * kPitch];
    const uint32_t tid = threadIdx.x, lane = tid & (kWave - 1), wave = tid / kWave;
    c.tab = s_tab;
    D::stage(s_tab, c, tid, kFusedWaves * kWave);
    __syncthreads();
    static_assert(KS == 1 || MT == 1, "split K is the one-tile (M <= 16) form's");
    const uint32_t rt = blockIdx.x * (kFusedWaves / KS) + wave / KS, ks = wave % KS;
    const uint32_t kb0 = ks * c.nkb / KS, kb1 = (ks + 1u) * c.nkb / KS;
    if (KS == 1 && rt * 16u >= N) return;
    const uint32_t slot = lane & 15u, half = lane >> 4, n = rt * 16u + slot;
    uint8_t* tr = s_tr[wave];
    const int8_t* ap[MT];
#pragma unroll
    for (uint32_t mt = 0; mt < MT; ++mt) {
        const uint32_t am = mt * 16u + slot;
        ap[mt] = A + (size_t)(am < M ? am : M - 1u) * a_ld;
    }
    v8f acc[MT];
#pragma unroll
    for (uint32_t mt = 0; mt < MT; ++mt) acc[mt] = (v8f)0.f;
    typename D::Raw cur{};
    float ws_cur = 0.f;
    if constexpr (ONE) {
        cur = D::fetch(c, rt, kb0, lane, true);
        ws_cur = fold_f32<F>(c, n, kb0);
    }
    for (uint32_t kb = kb0; kb < kb1; ++kb) {
        typename D::Raw nxt{};
        float ws_nxt = 0.f;
        if constexpr (ONE) {
            const uint32_t kn = kb + 1u < kb1 ? kb + 1u : kb;
            nxt = D::fetch(c, rt, kn, lane, true);
            ws_nxt = fold_f32<F>(c, n, kn);
        }
        {
            const typename D::Raw raw = ONE ? cur : D::fetch(c, rt, kb, lane, true);
            uint32_t w[16];
            D::rows(raw, c, rb_ratio<F>(c, rt * 16u + slot, kb), w);
            D::unzero(c, rt, kb, w);
#pragma unroll
            for (uint32_t q = 0; q < 16; ++q) *(uint32_t*)(tr + q * kPitch + 4u * lane) = w[q];
        }
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront", "local");
        __builtin_amdgcn_wave_barrier();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront", "local");
        v4i bf[8];
#pragma unroll
        for (uint32_t u = 0; u < 8; ++u) {
            const uint4 bv = *(const uint4*)(tr + slot * kPitch + u * 16u);
            bf[u][0] = (int)bv.x; bf[u][1] = (int)bv.y; bf[u][2] = (int)bv.z; bf[u][3] = (int)bv.w;
        }
        const float ws = ONE ? ws_cur : fold_f32<F>(c, n, kb);
#pragma unroll
        for (uint32_t mt = 0; mt < MT; ++mt) {
            v8i tmp = (v8i)0;
#pragma unroll
            for (uint32_t u = 0; u < 8; ++u) {
                const uint4 av = ONE ? *(const uint4*)(A + kb * 128u + u * 16u)
                                     : *(const uint4*)(ap[mt] + kb * 128u + u * 16u);
                v4i af;
                af[0] = (int)av.x; af[1] = (int)av.y; af[2] = (int)av.z; af[3] = (int)av.w;
                tmp = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, af, true, bf[u], tmp, false);
            }
#pragma unroll
            for (uint32_t e = 0; e < 8; ++e) {
                uint32_t mr = mt * 16u + 2u * e + half;
                mr = mr < M ? mr : M - 1u;
                const float as = ONE ? As[kb] : As[(size_t)mr * sa_ld + kb];
                acc[mt][e] = fmaf((float)tmp[e], as * ws, acc[mt][e]);
            }
        }
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront", "local");
        __builtin_amdgcn_wave_barrier();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront", "local");
        if constexpr (ONE) {
            cur = nxt;
            ws_cur = ws_nxt;
        }
    }
    if constexpr (KS > 1) {
        /* The K slices of a tile meet in LDS; the first slice's wave sums them in slice order. */
        __shared__ float s_red[kFusedWaves][8][kWave];
#pragma unroll
        for (uint32_t e = 0; e < 8; ++e) s_red[wave][e][lane] = acc[0][e];
        __syncthreads();
        if (ks) return;
#pragma unroll
        for (uint32_t j = 1; j < KS; ++j)
#pragma unroll
            for (uint32_t e = 0; e < 8; ++e) acc[0][e] += s_red[wave + j][e][lane];
    }
    const float rs = row_scale<F>(c, n);
#pragma unroll
    for (uint32_t mt = 0; mt < MT; ++mt)
#pragma unroll
        for (uint32_t e = 0; e < 8; ++e) {
            const uint32_t m = mt * 16u + 2u * e + half;
            if (m < M) Y[(size_t)m * N + n] = (__hip_bfloat16)(acc[mt][e] * rs);
        }
}

/* The prefill GEMM over the decoded plane (M > 64): libhalo's i8_tiled_block shape over the
 * row-major plane P[N][K] -- a 64 x 128 output tile a block of eight waves (2 x 4, each 2 x 2
 * fragments), K staged 64 at a time through LDS (pitch 80: every row 16-aligned), each 128-K
 * group's int32 sums folded in f32 with the activation's scale (from LDS) and fold_f32's FS,
 * row_scale's RS last. The arithmetic per element is the small kernels' own. */
constexpr int kTBM = 64, kTBN = 128, kTBK = 64, kTPitch = 80, kTTM = 2, kTTN = 2;

__global__ __launch_bounds__(256) void iqr_tiled_gemm_kernel(
        const int8_t* __restrict__ A, const float* __restrict__ As, const int8_t* __restrict__ P,
        const float* __restrict__ FS, const float* __restrict__ RS, __hip_bfloat16* __restrict__ Y,
        int M, int N, int K, int a_ld, int sa_ld) {
    __shared__ __attribute__((aligned(16))) int8_t sA[kTBM * kTPitch];
    __shared__ __attribute__((aligned(16))) int8_t sW[kTBN * kTPitch];
    __shared__ float sS[kTBM];
    const int tid = threadIdx.x, wave = tid >> 5, lane = tid & 31;
    const int wm = wave >> 2, wn = wave & 3;
    const int m0 = blockIdx.y * kTBM, n0 = blockIdx.x * kTBN;
    const int nkb = K / 128;
    /* Staging: A 64 rows x 64 B (one 16-byte chunk a thread), W 128 x 64 (two). */
    const int ar = tid >> 2, ae = (tid & 3) * 16;
    const int8_t* asrc = A + (size_t)(m0 + ar < M ? m0 + ar : M - 1) * a_ld + ae;
    int8_t* adst = sA + ar * kTPitch + ae;
    const int c0 = tid, r0 = c0 >> 2, e0 = (c0 & 3) * 16;
    const int c1 = tid + 256, r1 = c1 >> 2, e1 = (c1 & 3) * 16;
    const int8_t* wsrc0 = P + (size_t)(n0 + r0) * K + e0;
    const int8_t* wsrc1 = P + (size_t)(n0 + r1) * K + e1;
    int8_t* wdst0 = sW + r0 * kTPitch + e0;
    int8_t* wdst1 = sW + r1 * kTPitch + e1;
    const int srow = tid < kTBM ? (m0 + tid < M ? m0 + tid : M - 1) : 0;
    const float* sp = As + (size_t)srow * sa_ld;
    const int8_t* fa = sA + (wm * kTTM * 16 + (lane & 15)) * kTPitch;
    const int8_t* fw = sW + (wn * kTTN * 16 + (lane & 15)) * kTPitch;
    const int ncol0 = n0 + (wn * kTTN) * 16 + (lane & 15);
    const int ncol1 = ncol0 + 16;

    v8f acc[kTTM][kTTN];
    v8i tmp[kTTM][kTTN];
#pragma unroll
    for (int i = 0; i < kTTM; ++i)
#pragma unroll
        for (int j = 0; j < kTTN; ++j) { acc[i][j] = (v8f)0.f; tmp[i][j] = (v8i)0; }

    uint4 ra, rw0, rw1;
    float rs = 0.f;
    auto fetch = [&](int kt) {
        ra = *(const uint4*)(asrc + kt * kTBK);
        rw0 = *(const uint4*)(wsrc0 + kt * kTBK);
        rw1 = *(const uint4*)(wsrc1 + kt * kTBK);
        if (!(kt & 1) && tid < kTBM) rs = sp[kt >> 1];
    };
    fetch(0);
    const int nk = K / kTBK;
    for (int kt = 0; kt < nk; ++kt) {
        __syncthreads();
        *(uint4*)adst = ra;
        *(uint4*)wdst0 = rw0;
        *(uint4*)wdst1 = rw1;
        if (!(kt & 1) && tid < kTBM) sS[tid] = rs;
        __syncthreads();
        if (kt + 1 < nk) fetch(kt + 1);
#pragma unroll
        for (int s = 0; s < kTBK / 16; ++s) {
            const v4i af0 = *(const v4i*)(fa + 0 * 16 * kTPitch + s * 16);
            const v4i af1 = *(const v4i*)(fa + 1 * 16 * kTPitch + s * 16);
            const v4i bf0 = *(const v4i*)(fw + 0 * 16 * kTPitch + s * 16);
            const v4i bf1 = *(const v4i*)(fw + 1 * 16 * kTPitch + s * 16);
            tmp[0][0] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, af0, true, bf0, tmp[0][0], false);
            tmp[0][1] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, af0, true, bf1, tmp[0][1], false);
            tmp[1][0] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, af1, true, bf0, tmp[1][0], false);
            tmp[1][1] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, af1, true, bf1, tmp[1][1], false);
        }
        if (kt & 1) {
            const int g = kt >> 1;
            const float ws0 = FS[(size_t)ncol0 * nkb + g];
            const float ws1 = FS[(size_t)ncol1 * nkb + g];
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const float as0 = sS[(wm * kTTM + 0) * 16 + 2 * e + (lane >> 4)];
                const float as1 = sS[(wm * kTTM + 1) * 16 + 2 * e + (lane >> 4)];
                acc[0][0][e] = fmaf((float)tmp[0][0][e], as0 * ws0, acc[0][0][e]);
                acc[0][1][e] = fmaf((float)tmp[0][1][e], as0 * ws1, acc[0][1][e]);
                acc[1][0][e] = fmaf((float)tmp[1][0][e], as1 * ws0, acc[1][0][e]);
                acc[1][1][e] = fmaf((float)tmp[1][1][e], as1 * ws1, acc[1][1][e]);
                tmp[0][0][e] = 0;
                tmp[0][1][e] = 0;
                tmp[1][0][e] = 0;
                tmp[1][1][e] = 0;
            }
        }
    }
    const float r0s = RS[ncol0];
    const float r1s = RS[ncol1];
    const int mb0 = m0 + (wm * kTTM + 0) * 16;
    const int mb1 = m0 + (wm * kTTM + 1) * 16;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        const int mo0 = mb0 + 2 * e + (lane >> 4);
        const int mo1 = mb1 + 2 * e + (lane >> 4);
        if (mo0 < M) {
            Y[(size_t)mo0 * N + ncol0] = (__hip_bfloat16)(acc[0][0][e] * r0s);
            Y[(size_t)mo0 * N + ncol1] = (__hip_bfloat16)(acc[0][1][e] * r1s);
        }
        if (mo1 < M) {
            Y[(size_t)mo1 * N + ncol0] = (__hip_bfloat16)(acc[1][0][e] * r0s);
            Y[(size_t)mo1 * N + ncol1] = (__hip_bfloat16)(acc[1][1][e] * r1s);
        }
    }
}

#define HALO_IQR_T(k, b, lv, l, lrb, sc) FormT<k, b, lv, l, lrb, sc>

struct Operands {
    const RadTensor *a, *as, *b, *bs, *y;
    long long M, N, K, nkb, a_ld, sa_ld;
    uint32_t word;
};

int operands(const RadArgs* a, Operands* g) {
    g->a = halo_opt(a, 0);
    g->as = halo_opt(a, 1);
    g->b = halo_opt(a, 2);
    g->bs = halo_opt(a, 3);
    g->y = halo_opt(a, 4);
    if (!g->a || !g->as || !g->b || !g->bs || !g->y) return HALO_NO(RAD_E_INVAL);
    if (!is_weight_dtype(g->b->dtype) || !is_scale_dtype(g->bs->dtype)) return HALO_NO(RAD_E_DTYPE);
    g->word = g->b->dtype & kWordMask;
    if ((g->bs->dtype & kWordMask) != g->word || !built(g->word)) return HALO_NO(RAD_E_DTYPE);
    if (g->a->dtype != RAD_I8 || g->as->dtype != RAD_F32 || g->y->dtype != RAD_BF16)
        return HALO_NO(RAD_E_DTYPE);
    if (g->b->rank != 2 || g->y->rank < 2 || g->a->rank < 2 || g->as->rank != 2)
        return HALO_NO(RAD_E_SHAPE);
    g->M = g->y->shape[0];
    if (g->M < 1 || halo_numel(g->y) % g->M) return HALO_NO(RAD_E_SHAPE);
    g->N = halo_numel(g->y) / g->M;
    g->K = g->b->shape[1];
    if (g->b->shape[0] != g->N || g->N % 16 || g->K % kFK) return HALO_NO(RAD_E_SHAPE);
    g->nkb = g->K / kBlock;
    if (g->a->shape[0] != g->M || halo_numel(g->a) != g->M * g->K) return HALO_NO(RAD_E_SHAPE);
    if (g->as->shape[0] != g->M || g->as->shape[1] != g->nkb) return HALO_NO(RAD_E_SHAPE);
    g->a_ld = halo_row_pitch(g->a);
    g->sa_ld = halo_row_pitch(g->as);
    if (g->a_ld < 0 || g->sa_ld < 0 || halo_row_pitch(g->y) != g->N) return HALO_NO(RAD_E_STRIDE);
    if (!halo_aligned(g->a->data, 16) || g->a_ld % 16 || !halo_aligned(g->b->data, 256))
        return HALO_NO(RAD_E_ALIGN);
    const Form f = form(g->word);
    if (is_trellis(f.kind) && g->N % ((long long)1 << f.lrun)) return HALO_NO(RAD_E_SHAPE);
    return RAD_OK;
}

Ctx ctx_of(const Operands& g) {
    Ctx c{};
    c.W = (const uint8_t*)g.b->data;
    c.S = (const uint8_t*)g.bs->data;
    c.N = (uint32_t)g.N;
    c.nkb = (uint32_t)g.nkb;
    const Form f = form(g.word);
    c.run_tiles = is_trellis(f.kind) ? (1u << f.lrun) / 16u : 1u;
    return c;
}

template <class F, uint32_t MT>
int launch_fused(const Operands& g, hipStream_t st) {
    const uint32_t ntile = (uint32_t)(g.N / 16);
    const char* one = std::getenv("HALO_IQR_ONE");
    const char* kse = std::getenv("HALO_IQR_KS");
    const uint32_t ks = kse ? (uint32_t)std::atoi(kse) : 4u;
    const bool m1 = g.M == 1 && !(one && *one == '0');
    const char* kme = std::getenv("HALO_IQR_KS_M");
    const long long ksm = kme ? std::atoll(kme) : 16;
    if (MT == 1 && g.M <= ksm && (ks == 2 || ks == 4) && g.nkb >= ks && ntile % (kFusedWaves / ks) == 0) {
        auto k = m1 ? (ks == 4 ? iqr_fused_kernel<F, 1, true, 4> : iqr_fused_kernel<F, 1, true, 2>)
                    : (ks == 4 ? iqr_fused_kernel<F, 1, false, 4> : iqr_fused_kernel<F, 1, false, 2>);
        hipLaunchKernelGGL(k, dim3(ntile / (kFusedWaves / ks)), dim3(kFusedWaves * kWave), 0, st,
                           ctx_of(g), (const int8_t*)g.a->data, (const float*)g.as->data,
                           (__hip_bfloat16*)g.y->data, (uint32_t)g.M, (uint32_t)g.N,
                           (uint32_t)g.a_ld, (uint32_t)g.sa_ld);
        return hipGetLastError() == hipSuccess ? RAD_OK : HALO_NO(RAD_E_DEVICE);
    }
    if (MT == 1 && g.M == 1 && !(one && *one == '0')) {
        hipLaunchKernelGGL((iqr_fused_kernel<F, 1, true>), dim3((ntile + kFusedWaves - 1) / kFusedWaves),
                           dim3(kFusedWaves * kWave), 0, st, ctx_of(g), (const int8_t*)g.a->data,
                           (const float*)g.as->data, (__hip_bfloat16*)g.y->data, (uint32_t)g.M,
                           (uint32_t)g.N, (uint32_t)g.a_ld, (uint32_t)g.sa_ld);
        return hipGetLastError() == hipSuccess ? RAD_OK : HALO_NO(RAD_E_DEVICE);
    }
    hipLaunchKernelGGL((iqr_fused_kernel<F, MT>), dim3((ntile + kFusedWaves - 1) / kFusedWaves),
                       dim3(kFusedWaves * kWave), 0, st, ctx_of(g), (const int8_t*)g.a->data,
                       (const float*)g.as->data, (__hip_bfloat16*)g.y->data, (uint32_t)g.M,
                       (uint32_t)g.N, (uint32_t)g.a_ld, (uint32_t)g.sa_ld);
    return hipGetLastError() == hipSuccess ? RAD_OK : HALO_NO(RAD_E_DEVICE);
}

template <class F>
int launch(const Operands& g, void* scratch, long long scratch_bytes, hipStream_t st) {
    if (g.M <= 16) return launch_fused<F, 1>(g, st);
    if (g.M <= 32) return launch_fused<F, 2>(g, st);
    if (g.M <= 64) return launch_fused<F, 4>(g, st);
    const Plane pl = plane_of(g.N, g.K);
    if (!scratch || scratch_bytes < pl.bytes || !halo_aligned(scratch, 256))
        return HALO_NO(RAD_E_NOMEM);
    int8_t* P = (int8_t*)scratch;
    float* FS = (float*)((uint8_t*)scratch + pl.fs_off);
    float* RS = (float*)((uint8_t*)scratch + pl.rs_off);
    const uint32_t ntile = (uint32_t)(g.N / 16);
    const uint32_t ky = (uint32_t)((g.nkb + kDecWaves - 1) / kDecWaves);
    hipLaunchKernelGGL(iqr_decode_kernel<F>, dim3(ntile, ky), dim3(kDecWaves * kWave), 0, st,
                       ctx_of(g), P, FS, RS);
    if (g.N % kTBN == 0 && g.K % 128 == 0) {
        hipLaunchKernelGGL(iqr_tiled_gemm_kernel, dim3((unsigned)(g.N / kTBN), (unsigned)((g.M + kTBM - 1) / kTBM)),
                           dim3(256), 0, st, (const int8_t*)g.a->data, (const float*)g.as->data, P, FS, RS,
                           (__hip_bfloat16*)g.y->data, (int)g.M, (int)g.N, (int)g.K, (int)g.a_ld,
                           (int)g.sa_ld);
        return hipGetLastError() == hipSuccess ? RAD_OK : HALO_NO(RAD_E_DEVICE);
    }
    hipLaunchKernelGGL(iqr_i8_gemm_kernel, dim3((ntile + 3) / 4, (uint32_t)((g.M + 15) / 16)),
                       dim3(4 * kWave), 0, st, (const int8_t*)g.a->data, (const float*)g.as->data,
                       P, FS, RS, (__hip_bfloat16*)g.y->data, (uint32_t)g.M, (uint32_t)g.N,
                       (uint32_t)g.K, (uint32_t)g.a_ld, (uint32_t)g.sa_ld);
    return hipGetLastError() == hipSuccess ? RAD_OK : HALO_NO(RAD_E_DEVICE);
}

int run(const RadArgs* a, RadStream s) {
    Operands g;
    const int rc = operands(a, &g);
    if (rc != RAD_OK) return rc;
    const uint32_t kw = kernel_word(g.word);
    hipStream_t st = (hipStream_t)s;
#define HALO_IQR_GO(k, b, lv, l, lrb, sc) \
    if (kw == HALO_IQR_T(k, b, lv, l, lrb, sc)::word()) \
        return launch<HALO_IQR_T(k, b, lv, l, lrb, sc)>(g, a->scratch, a->scratch_bytes, st);
    HALO_IQR_FORMS(HALO_IQR_GO)
#undef HALO_IQR_GO
    return HALO_NO(RAD_E_DTYPE);
}

int64_t scratch_of(const RadArgs* a) {
    const long long N = halo_geti_or(a, "N", 0), K = halo_geti_or(a, "K", 0);
    if (N < 16 || K < (long long)kFK) return 0;
    return plane_of(N, K).bytes;
}

}  // namespace

/* gemm_nt_q_gated over an IQR gate_up weight, composed: the GEMM above writes the [M, 2n] bf16
 * product into the op's scratch (after the GEMM's own decode plane), then gated_had_quant_i8g
 * takes it to the quantiser's outputs. libr4d's gemm_iqr_gated folds the second pass into the
 * epilogue and states it byte-identical to this pair. Operands: a, a_scale, b, b_scale, q, scale,
 * out_bf16? */
extern "C" int halo_rad_gated_had_quant_i8g(const RadArgs*, RadStream);

namespace {

int64_t gated_product_off(int64_t N, int64_t K) { return (plane_of(N, K).bytes + 255) & ~(int64_t)255; }

}  // namespace

extern "C" int64_t halo_rad_gemm_iqr_gated_scratch(const RadArgs* a) {
    const long long M = halo_geti_or(a, "M", 0), N = halo_geti_or(a, "N", 0),
                    K = halo_geti_or(a, "K", 0);
    if (M < 1 || N < 16 || K < (long long)kFK) return 0;
    return gated_product_off(N, K) + M * N * 2;
}

extern "C" int halo_rad_gemm_iqr_gated(const RadArgs* a, RadStream s) {
    if (!a || !a->t || a->n_t < 6) return HALO_NO(RAD_E_INVAL);
    const RadTensor* q  = halo_opt(a, 4);
    const RadTensor* qs = halo_opt(a, 5);
    const RadTensor* b  = halo_opt(a, 2);
    if (!q || !qs || !b) return HALO_NO(RAD_E_INVAL);
    if (q->dtype != RAD_I8 || q->rank != 2 || b->rank != 2) return HALO_NO(RAD_E_SHAPE);
    const char* act = halo_gets(a, "act");
    if (act && std::strcmp(act, "silu")) return HALO_NO(RAD_E_UNSUPPORTED);
    const long long M = q->shape[0], N = b->shape[0], K = b->shape[1];
    if (q->shape[1] * 2 != N) return HALO_NO(RAD_E_SHAPE);
    const int64_t off = gated_product_off(N, K);
    if (!a->scratch || a->scratch_bytes < off + M * N * 2) return HALO_NO(RAD_E_NOMEM);
    RadTensor y;
    std::memset(&y, 0, sizeof(y));
    y.data = (uint8_t*)a->scratch + off;
    y.dtype = RAD_BF16;
    y.rank = 2;
    y.shape[0] = M; y.shape[1] = N;
    y.stride[0] = N; y.stride[1] = 1;
    RadTensor gt[5] = { a->t[0], a->t[1], a->t[2], a->t[3], y };
    RadArgs g = *a;
    g.t = gt; g.n_t = 5;
    g.scratch_bytes = off;
    int rc = run(&g, s);
    if (rc != RAD_OK) return rc;
    RadTensor qt[4];
    std::memset(qt, 0, sizeof(qt));
    qt[0] = y; qt[1] = a->t[4]; qt[2] = a->t[5];
    if (a->n_t > 6) qt[3] = a->t[6];
    const RadParam qp[] = { RAD_STR("act", "silu") };
    RadArgs qa = *a;
    qa.t = qt; qa.n_t = 4; qa.p = qp; qa.n_p = 1;
    qa.scratch = nullptr; qa.scratch_bytes = 0; qa.instance = nullptr;
    return halo_rad_gated_had_quant_i8g(&qa, s);
}

extern "C" int halo_rad_gemm_iqr_m64(const RadArgs* a, RadStream s) { return run(a, s); }
extern "C" int64_t halo_rad_gemm_iqr_m64_scratch(const RadArgs* a) { return scratch_of(a); }
extern "C" int halo_rad_gemm_iqr_tiled(const RadArgs* a, RadStream s) { return run(a, s); }
extern "C" int64_t halo_rad_gemm_iqr_tiled_scratch(const RadArgs* a) { return scratch_of(a); }
