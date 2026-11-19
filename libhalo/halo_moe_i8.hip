/* moe_gemm_q against an int8 activation on gfx1151: the routed experts' grouped GEMM at the two
 * stored forms Qwen3.8-Flash-Next's IQR-V4 container reads with an int8 activation (libr4d's
 * moe_gemm_w4a8i8 and moe_gemm_iqr, r4d_moe.hip / r4d_moe_iqr.hip):
 *
 *   halo_rad_moe_gemm_w4a8i8   dtype w4nl64a8i8h: libquant's w4nl codes (the stored planes of
 *                              moe_gemm_w4a8, halo_layout.cpp's moew4_permute order) read as
 *                              libr4d's int8 levels kW4nlI8 times kW4nlI8Unit, an E4M3 scale per
 *                              (row, 64 of K) under the fixed kW4G64Scale
 *   halo_rad_moe_gemm_iqr      dtype i8a8h over IQR experts (halo_iqr.h), each decoded by
 *                              halo_iqr_dec.h's decoders where it is read
 *
 * One wave a 16-row x 16-column output tile of the sorted rows. The tile's rows may belong to
 * several experts (sorted, so a few at most); the wave takes them in turn, each expert's WMMAs
 * reading only its own rows' activation (the rest are zero), so every row sums its own expert's
 * product. A weight unit of 16 rows x 128 of K is laid out row-major in the wave's LDS tile and
 * read back as gfx1151 iu8 B fragments: lane l takes row (l & 15)'s sixteen K bytes; the
 * accumulator's slot e is output row 2e + (l >> 4), column (l & 15) (halo_gemm_fp8a8.hip).
 * Each scale group's int32 sum is folded in f32: the product within f32 rounding. */

#include "halo.h"
#include "halo_args.h"
#include "halo_iqr.h"
#include "halo_iqr_dec.h"

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstring>
#include <cstdlib>

namespace {

using namespace halo_iqr;
using namespace halo_iqr_dev;

constexpr uint32_t kWave = 32, kWaves = 4, kPitch = 136;

typedef int v4i __attribute__((ext_vector_type(4)));
typedef int v8i __attribute__((ext_vector_type(8)));

/* The unit of libr4d's w4nl int8 levels (r4d_args.h kW4nlI8, which w4nl_i8x4 holds), and the
 * fixed second scale level of w4nl64a8h (kW4G64Scale). */
constexpr float kW4nlI8Unit = 0.18965f;
constexpr float kW4G64Scale = 0x1p-13f;
/* One class of experts: entry `i` of a weight-table operand (a device pointer array when the
 * stacked stride is 0). */
struct Cls {
    const void* const* w;
    const void* const* s;
    uint64_t w_s0, s_s0;
    uint32_t n, k, w_ld, s_ld, run_tiles;
};
/* An expert's class, chosen field by field: indexing a Cls[2] by the expert's parity puts both
 * in scratch, and every field read in the loops then goes to memory. */
__device__ __forceinline__ Cls pick(const Cls (&cl)[2], uint32_t ex) {
    const bool o = (ex & 1u) != 0u;
    Cls c;
    c.w = o ? cl[1].w : cl[0].w;
    c.s = o ? cl[1].s : cl[0].s;
    c.w_s0 = o ? cl[1].w_s0 : cl[0].w_s0;
    c.s_s0 = o ? cl[1].s_s0 : cl[0].s_s0;
    c.n = o ? cl[1].n : cl[0].n;
    c.k = o ? cl[1].k : cl[0].k;
    c.w_ld = o ? cl[1].w_ld : cl[0].w_ld;
    c.s_ld = o ? cl[1].s_ld : cl[0].s_ld;
    c.run_tiles = o ? cl[1].run_tiles : cl[0].run_tiles;
    return c;
}
__device__ __forceinline__ const uint8_t* tab_at(const void* const* t, uint64_t s0, uint32_t i) {
    return s0 == 0 ? (const uint8_t*)t[i] : (const uint8_t*)t + (size_t)i * s0;
}

struct Route {
    const int8_t* A;
    const float* As;
    const int32_t* sorted;
    const int32_t* off;
    uint16_t* Y;
    uint32_t T, N, n_expert, top_k, a_ld, as_ld, y_ld, a_group, a_sorted;
};

__device__ __forceinline__ uint16_t f32_to_bf16(float f) {
    uint32_t u = __float_as_uint(f);
    u += 0x7FFFu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}
__device__ __forceinline__ float e4m3f(uint32_t b) {
    const uint32_t sign = (b & 0x80u) << 24, ex = (b >> 3) & 0x0Fu, mn = b & 0x07u;
    const float nrm = __uint_as_float(((ex + 120u) << 23) | (mn << 20));
    const float sub = (float)mn * (1.0f / 512.0f);
    return __uint_as_float(__float_as_uint(ex ? nrm : sub) | sign);
}

/* kW4nlI8 at four nibbles (one a byte), by a byte permute in registers: a constant-memory
 * table is a vector memory load a byte. Codes 8-15 are 0-7 negated; no level is 0, so each
 * byte's two's complement stays in its byte. */
__device__ __forceinline__ uint32_t w4nl_i8x4(uint32_t nib) {
    constexpr uint32_t lo = 6u | 17u << 8 | 29u << 16 | 42u << 24;
    constexpr uint32_t hi = 58u | 74u << 8 | 95u << 16 | 116u << 24;
    const uint32_t mag = __builtin_amdgcn_perm(hi, lo, nib & 0x07070707u);
    const uint32_t neg = ((nib >> 3) & 0x01010101u) * 0xFFu;
    return (mag ^ neg) + (neg & 0x01010101u);
}

/* The w4nl expert: a unit's 16 rows of lane l's four columns, as the IQR decoders give them. */
struct W4nl {
    static constexpr uint32_t kSub = 64;   /* K per weight scale */
    const uint8_t* w;
    const uint8_t* s;
    uint32_t w_ld, s_ld;
    struct Raw { uint32_t d[16]; };
    __device__ Raw fetch(uint32_t row0, uint32_t kb, uint32_t lane) const {
        Raw r;
#pragma unroll
        for (uint32_t q = 0; q < 16; ++q)
            r.d[q] = *(const uint32_t*)(w + (size_t)(row0 + q) * w_ld + kb * 64u + (lane >> 1) * 4u);
        return r;
    }
    __device__ void make(const Raw& r, uint32_t, uint32_t, uint32_t lane, uint32_t (&o)[16]) const {
#pragma unroll
        for (uint32_t q = 0; q < 16; ++q) {
            const uint32_t d = r.d[q];
            const uint32_t nib = (lane & 1u) ? (d >> 4) & 0x0F0F0F0Fu : d & 0x0F0F0F0Fu;
            o[q] = w4nl_i8x4(nib);
        }
    }
    __device__ void unit(uint32_t row0, uint32_t kb, uint32_t lane, uint32_t (&o)[16]) const {
        make(fetch(row0, kb, lane), row0, kb, lane, o);
    }
    /* The f32 a sub-group's int32 sum of row r takes, and the row's own factor (1). */
    __device__ float fold(uint32_t r, uint32_t sub) const {
        return e4m3f(s[(size_t)r * s_ld + sub]) * (kW4G64Scale * kW4nlI8Unit);
    }
    __device__ float row(uint32_t) const { return 1.0f; }
};

/* An IQR expert: the dense GEMM's decoders over this expert's buffers. */
template <class F>
struct Iqr {
    static constexpr uint32_t kSub = 128;
    Ctx c;
    using Raw = typename Dec<F>::Raw;
    __device__ Raw fetch(uint32_t row0, uint32_t kb, uint32_t lane) const {
        return Dec<F>::fetch(c, row0 / 16u, kb, lane, true);
    }
    __device__ void make(const Raw& raw, uint32_t row0, uint32_t kb, uint32_t lane, uint32_t (&o)[16]) const {
        const uint32_t rt = row0 / 16u;
        Dec<F>::rows(raw, c, rb_ratio<F>(c, rt * 16u + (lane & 15u), kb), o);
        Dec<F>::unzero(c, rt, kb, o);
    }
    __device__ void unit(uint32_t row0, uint32_t kb, uint32_t lane, uint32_t (&o)[16]) const {
        make(fetch(row0, kb, lane), row0, kb, lane, o);
    }
    __device__ float fold(uint32_t r, uint32_t kb) const { return fold_f32<F>(c, r, kb); }
    __device__ float row(uint32_t r) const { return row_scale<F>(c, r); }
};

template <class F>
__device__ __forceinline__ Iqr<F> make_iqr(const Cls& cl, uint32_t idx, int8_t* tab) {
    Iqr<F> x;
    x.c.W = tab_at(cl.w, cl.w_s0, idx);
    x.c.S = tab_at(cl.s, cl.s_s0, idx);
    x.c.tab = tab;
    x.c.N = cl.n;
    x.c.nkb = cl.k / 128u;
    x.c.run_tiles = cl.run_tiles;
    return x;
}

__device__ __forceinline__ uint32_t expert_of(const int32_t* off, uint32_t n_expert, uint32_t i) {
    uint32_t lo = 0, hi = n_expert;
    while (hi - lo > 1u) {
        const uint32_t mid = (lo + hi) / 2u;
        if ((uint32_t)off[mid] <= i) lo = mid; else hi = mid;
    }
    return lo;
}

/* The wave's 16 x 16 tile: rows r0.., columns n0... `make` builds expert e's weight view, whose
 * scales cover SUB of K each. */
/* `zj` of `zn`: the experts of the tile are split over zn waves (grid z), the wave taking the
 * ones whose rank in the tile (ascending) is zj modulo zn, and writing only their rows; wave 0
 * also writes the rows no expert owns (dropped, past the live rows), as zeros. */
template <uint32_t SUB, class MAKE>
__device__ void tile(const Route& R, const Cls (&cl)[2], uint32_t r0, uint32_t n0, uint8_t* tr,
                     int8_t* tab, MAKE make, uint32_t zj, uint32_t zn) {
    const uint32_t lane = threadIdx.x & (kWave - 1u), half = lane >> 4, slot = lane & 15u;
    const uint32_t live = (uint32_t)R.off[R.n_expert];
    /* This lane's activation row (slot) and accumulator rows (2e + half): expert and source. */
    const uint32_t ia = r0 + slot;
    int32_t tok = -1;
    uint32_t ea = 0xFFFFFFFFu;
    if (ia < live && ia < R.T) {
        const int32_t f = R.sorted[ia];
        if (f >= 0) {
            tok = R.a_sorted ? (int32_t)ia : f / (int32_t)R.top_k;
            ea = expert_of(R.off, R.n_expert, ia);
        }
    }
    uint32_t eo[8];
#pragma unroll
    for (uint32_t e = 0; e < 8; ++e) {
        const uint32_t i = r0 + 2u * e + half;
        eo[e] = 0xFFFFFFFFu;
        if (i < live && i < R.T && R.sorted[i] >= 0) eo[e] = expert_of(R.off, R.n_expert, i);
    }
    v8f acc = (v8f)0.f;
    /* The tile's experts in ascending order: the smallest not yet done, over the lanes. */
    uint32_t done = 0u, rank = 0u;
    for (;; ++rank) {
        uint32_t mine = (ea != 0xFFFFFFFFu && (done == 0u || ea >= done)) ? ea : 0xFFFFFFFFu;
        for (uint32_t o = 16; o >= 1; o >>= 1) {
            const uint32_t other = __shfl_xor(mine, o, kWave);
            mine = other < mine ? other : mine;
        }
        const uint32_t ex = mine;
        if (ex == 0xFFFFFFFFu) break;
        done = ex + 1u;
        if (rank % zn != zj) continue;
        const Cls c = pick(cl, ex);
        const uint32_t idx = ex >> 1;
        if (n0 >= c.n) continue;   /* a narrower class: its columns past n are zero */
        const auto W = make(c, idx, tab);
        const uint32_t nkb = c.k / 128u;
        const bool arow = ea == ex;
        const int8_t* ap = arow ? R.A + (size_t)tok * R.a_ld : nullptr;
        v8f part = (v8f)0.f;
        for (uint32_t kb = 0; kb < nkb; ++kb) {
            uint32_t w[16];
            W.unit(n0, kb, lane, w);
#pragma unroll
            for (uint32_t q = 0; q < 16; ++q) *(uint32_t*)(tr + q * kPitch + 4u * lane) = w[q];
            __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront", "local");
            __builtin_amdgcn_wave_barrier();
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront", "local");
            constexpr uint32_t SUBS = 128u / SUB;
#pragma unroll
            for (uint32_t sg = 0; sg < SUBS; ++sg) {
                v8i tmp = (v8i)0;
#pragma unroll
                for (uint32_t u = 0; u < 8u / SUBS; ++u) {
                    const uint32_t ks = sg * (128u / SUBS) + u * 16u;
                    v4i af = (v4i)0, bf;
                    if (arow) {
                        const uint4 av = *(const uint4*)(ap + kb * 128u + ks);
                        af[0] = (int)av.x; af[1] = (int)av.y; af[2] = (int)av.z; af[3] = (int)av.w;
                    }
                    const uint4 bv = *(const uint4*)(tr + slot * kPitch + ks);
                    bf[0] = (int)bv.x; bf[1] = (int)bv.y; bf[2] = (int)bv.z; bf[3] = (int)bv.w;
                    tmp = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, af, true, bf, tmp, false);
                }
                const uint32_t kabs = kb * 128u + sg * (128u / SUBS);
                const float ws = W.fold(n0 + slot, kb * SUBS + sg);
#pragma unroll
                for (uint32_t e = 0; e < 8; ++e) {
                    if (eo[e] != ex) continue;
                    const uint32_t i = r0 + 2u * e + half;
                    const int32_t f = R.sorted[i];
                    const uint32_t at = R.a_sorted ? i : (uint32_t)f / R.top_k;
                    part[e] = fmaf((float)tmp[e], R.As[(size_t)at * R.as_ld + kabs / R.a_group] * ws, part[e]);
                }
            }
            __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront", "local");
            __builtin_amdgcn_wave_barrier();
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront", "local");
        }
        const float rs = W.row(n0 + slot);
#pragma unroll
        for (uint32_t e = 0; e < 8; ++e) acc[e] += part[e] * rs;
    }
    /* A row's owner: the rank of its expert among the tile's, from the slots' experts (the
     * sorted rows are grouped by expert, so a rank counts the distinct experts below it). */
    uint32_t ea_s[16], first = 0u;
#pragma unroll
    for (uint32_t s = 0; s < 16; ++s) ea_s[s] = __shfl(ea, s, kWave);
#pragma unroll
    for (uint32_t s = 0; s < 16; ++s) {
        bool f = ea_s[s] != 0xFFFFFFFFu;
#pragma unroll
        for (uint32_t s2 = 0; s2 < s; ++s2) f = f && ea_s[s2] != ea_s[s];
        first |= f ? 1u << s : 0u;
    }
#pragma unroll
    for (uint32_t e = 0; e < 8; ++e) {
        const uint32_t i = r0 + 2u * e + half;
        if (i >= R.T || n0 + slot >= R.N) continue;
        bool own;
        if (eo[e] == 0xFFFFFFFFu) {
            own = zj == 0u;
        } else {
            uint32_t rk = 0u;
#pragma unroll
            for (uint32_t s = 0; s < 16; ++s)
                rk += ((first >> s) & 1u) && ea_s[s] < eo[e] ? 1u : 0u;
            own = rk % zn == zj;
        }
        if (own) R.Y[(size_t)i * R.y_ld + n0 + slot] = f32_to_bf16(acc[e]);
    }
}

}  // namespace

namespace {

struct W4Make {
    __device__ W4nl operator()(const Cls& c, uint32_t idx, int8_t*) const {
        W4nl x;
        x.w = tab_at(c.w, c.w_s0, idx);
        x.s = tab_at(c.s, c.s_s0, idx);
        x.w_ld = c.w_ld;
        x.s_ld = c.s_ld;
        return x;
    }
};
template <class F>
struct IqrMake {
    __device__ Iqr<F> operator()(const Cls& c, uint32_t idx, int8_t* tab) const {
        return make_iqr<F>(c, idx, tab);
    }
};

__global__ __launch_bounds__(kWaves * kWave) void moe_w4nl_i8_kernel(Route R, Cls c0, Cls c1) {
    __shared__ __attribute__((aligned(16))) uint8_t s_tr[kWaves][16 * kPitch];
    const uint32_t wave = threadIdx.x / kWave;
    const uint32_t nt = blockIdx.y * kWaves + wave;
    if (nt * 16u >= R.N) return;
    const Cls cl[2] = { c0, c1 };
    tile<W4nl::kSub>(R, cl, blockIdx.x * 16u, nt * 16u, s_tr[wave], nullptr, W4Make{}, blockIdx.z,
                     gridDim.z);
}

template <class F>
__global__ __launch_bounds__(kWaves * kWave) void moe_iqr_i8_kernel(Route R, Cls c0, Cls c1) {
    __shared__ __attribute__((aligned(16))) uint8_t s_tr[kWaves][16 * kPitch];
    __shared__ __attribute__((aligned(16))) int8_t s_tab[Dec<F>::kTab];
    const uint32_t wave = threadIdx.x / kWave;
    const uint32_t nt = blockIdx.y * kWaves + wave;
    if (nt * 16u >= R.N) return;
    const Cls cl[2] = { c0, c1 };
    tile<Iqr<F>::kSub>(R, cl, blockIdx.x * 16u, nt * 16u, s_tr[wave], s_tab, IqrMake<F>{}, blockIdx.z,
                       gridDim.z);
}

/* The decode form (T <= 64): a block per (expert, 128 output columns), eight waves of 16
 * columns each over the whole K, the expert's rows as up to MT row tiles. Block column y works
 * only when sorted row y opens its expert's run, so each expert is one block column and no wave
 * searches a tile for its experts; the block at y == live zeroes the rows past the live ones.
 * A wave issues the next 128-K unit's codes before decoding the current one. The arithmetic is
 * tile()'s: the same unit, iu8 WMMA and per-group f32 fold. */
constexpr uint32_t kDecW = 8;

/* The lane-xor exchange in registers (DPP within a row of 16, permlanex16 across the rows)
 * rather than __shfl_xor's LDS-path bpermute. */
template <uint32_t M>
__device__ __forceinline__ int32_t xor_lane(int32_t x) {
    static_assert(M == 1u || M == 2u || M == 4u || M == 8u || M == 16u, "a lane bit of a wave");
    if constexpr (M == 16u)
        return __builtin_amdgcn_permlanex16(x, x, 0x76543210u, 0xFEDCBA98u, false, false);
    else
        return __builtin_amdgcn_update_dpp(x, x, 0x160 | (int)M, 0xF, 0xF, false);
}

/* The lanes' 16 int32 values, halved across lanes by xor mask M: each lane keeps the half its
 * bit M selects and adds its partner's copy of that half. */
template <uint32_t N, uint32_t M>
__device__ __forceinline__ void halve(int32_t (&v)[16], uint32_t lane) {
    const bool hi = (lane & M) != 0u;
#pragma unroll
    for (uint32_t i = 0; i < N / 2u; ++i) {
        /* select on the values, never on the array, or the array goes to scratch */
        const int32_t a = v[i], b = v[i + N / 2u];
        const int32_t send = hi ? a : b;
        v[i] = (hi ? b : a) + xor_lane<M>(send);
    }
}

/* The decode GEMV, an expert's rows <= RMAX (the decode and verify case, about one row an
 * expert): lane l's decoded dword of weight row q is K 4l..4l+3, so each row's partial is one
 * sudot4 against the activation's dword, no WMMA. The block stages the rows' activations and
 * scales in LDS once, so a wave's loop waits only on its prefetched codes; the weight scales are
 * read a unit ahead. A unit's 16 column sums close by halving across the lanes (16 shuffles):
 * the exact int32 sums the WMMA forms, folded in the same order with the same factors, so the
 * output is theirs byte for byte. */
template <uint32_t RMAX, uint32_t SUB, class MAKE>
__device__ void gemv_expert(const Route& R, const Cls& c, uint32_t ex, uint32_t y, uint32_t r1,
                            uint8_t* lds, int8_t* tab, MAKE make) {
    const uint32_t tid = threadIdx.x, lane = tid & (kWave - 1u);
    const uint32_t n0 = blockIdx.x * (16u * kDecW) + (tid / kWave) * 16u;
    constexpr uint32_t SUBS = 128u / SUB;
    const uint32_t cnt = r1 - y, K = c.k, nsc = (K + R.a_group - 1u) / R.a_group;
    int8_t* sa = (int8_t*)lds;
    float* ss = (float*)(lds + RMAX * K);
    for (uint32_t i = tid; i < cnt * (K / 16u); i += kDecW * kWave) {
        const uint32_t r = i / (K / 16u), o = i % (K / 16u);
        const int32_t f = R.sorted[y + r];
        uint4 v = make_uint4(0u, 0u, 0u, 0u);
        if (f >= 0)
            v = *(const uint4*)(R.A + (size_t)(R.a_sorted ? y + r : (uint32_t)f / R.top_k) * R.a_ld + o * 16u);
        *(uint4*)(sa + r * K + o * 16u) = v;
    }
    for (uint32_t i = tid; i < cnt * nsc; i += kDecW * kWave) {
        const uint32_t r = i / nsc, j = i % nsc;
        const int32_t f = R.sorted[y + r];
        ss[i] = f >= 0 ? R.As[(size_t)(R.a_sorted ? y + r : (uint32_t)f / R.top_k) * R.as_ld + j] : 0.f;
    }
    __syncthreads();
    if (n0 >= R.N) return;
    if (n0 >= c.n) {   /* a narrower class: its columns past n are zero */
        for (uint32_t i = y + (lane >> 4); i < r1; i += 2u) R.Y[(size_t)i * R.y_ld + n0 + (lane & 15u)] = 0;
        return;
    }
    bool vr[RMAX];
#pragma unroll
    for (uint32_t r = 0; r < RMAX; ++r) vr[r] = r < cnt && R.sorted[y + r] >= 0;
    /* SUBS 1: the column of lanes 2j, 2j + 1 after five steps; SUBS 2: four steps keep the halves
     * apart (lanes 0-15 hold K 0-63, the first scale's) and lane l, l ^ 16 hold one column. */
    const uint32_t col = SUBS == 1u
        ? ((lane >> 4) & 1u) * 8u + ((lane >> 3) & 1u) * 4u + ((lane >> 2) & 1u) * 2u + ((lane >> 1) & 1u)
        : ((lane >> 3) & 1u) * 8u + ((lane >> 2) & 1u) * 4u + ((lane >> 1) & 1u) * 2u + (lane & 1u);
    const auto W = make(c, ex >> 1, tab);
    const uint32_t nkb = K / 128u;
    float acc[RMAX];
#pragma unroll
    for (uint32_t r = 0; r < RMAX; ++r) acc[r] = 0.f;
    auto cur = W.fetch(n0, 0, lane);
    float ws[SUBS];
#pragma unroll
    for (uint32_t sg = 0; sg < SUBS; ++sg) ws[sg] = W.fold(n0 + col, sg);
    for (uint32_t kb = 0; kb < nkb; ++kb) {
        const uint32_t kn = kb + 1u < nkb ? kb + 1u : kb;
        const auto nxt = W.fetch(n0, kn, lane);
        float wn[SUBS];
#pragma unroll
        for (uint32_t sg = 0; sg < SUBS; ++sg) wn[sg] = W.fold(n0 + col, kn * SUBS + sg);
        uint32_t w[16];
        W.make(cur, n0, kb, lane, w);
#pragma unroll
        for (uint32_t r = 0; r < RMAX; ++r) {
            if (!vr[r]) continue;
            const uint32_t av = *(const uint32_t*)(sa + r * K + kb * 128u + 4u * lane);
            const float* as = ss + r * nsc;
            int32_t v[16];
#pragma unroll
            for (uint32_t q = 0; q < 16; ++q)
                v[q] = __builtin_amdgcn_sudot4(true, (int)w[q], true, (int)av, 0, false);
            if constexpr (SUBS == 1u) {
                halve<16, 16>(v, lane); halve<8, 8>(v, lane); halve<4, 4>(v, lane); halve<2, 2>(v, lane);
                v[0] += xor_lane<1>(v[0]);
                acc[r] = fmaf((float)v[0], as[kb * 128u / R.a_group] * ws[0], acc[r]);
            } else {
                halve<16, 8>(v, lane); halve<8, 4>(v, lane); halve<4, 2>(v, lane); halve<2, 1>(v, lane);
                const int32_t o = xor_lane<16>(v[0]);
                const int32_t s0 = (lane & 16u) ? o : v[0], s1 = (lane & 16u) ? v[0] : o;
                acc[r] = fmaf((float)s0, as[(kb * 128u) / R.a_group] * ws[0], acc[r]);
                acc[r] = fmaf((float)s1, as[(kb * 128u + 64u) / R.a_group] * ws[SUBS - 1u], acc[r]);
            }
        }
        cur = nxt;
#pragma unroll
        for (uint32_t sg = 0; sg < SUBS; ++sg) ws[sg] = wn[sg];
    }
    /* ws[] above is the scale of this lane's own column for each sub-group: under SUBS 2 the
     * lane's half only selects which partial it holds first, both halves fold both. */
    const float rs = W.row(n0 + col);
    const bool writer = SUBS == 1u ? (lane & 1u) == 0u : lane < 16u;
#pragma unroll
    for (uint32_t r = 0; r < RMAX; ++r)
        if (r < cnt && writer) R.Y[(size_t)(y + r) * R.y_ld + n0 + col] = f32_to_bf16(0.f + acc[r] * rs);
}

template <uint32_t MT, uint32_t SUB, class MAKE>
__device__ void dec_expert(const Route& R, const Cls (&cl)[2], uint8_t* lds, int8_t* tab, MAKE make) {
    const uint32_t lane = threadIdx.x & (kWave - 1u), half = lane >> 4, slot = lane & 15u;
    const uint32_t y = blockIdx.y, n0 = blockIdx.x * (16u * kDecW) + (threadIdx.x / kWave) * 16u;
    const uint32_t live = (uint32_t)R.off[R.n_expert];
    /* Block-uniform exits only until the GEMV's barrier. */
    if (y >= live) {
        if (y == live && n0 < R.N)
            for (uint32_t i = live + half; i < R.T; i += 2u) R.Y[(size_t)i * R.y_ld + n0 + slot] = 0;
        return;
    }
    const uint32_t ex = expert_of(R.off, R.n_expert, y);
    if ((uint32_t)R.off[ex] != y) return;
    const uint32_t r1 = (uint32_t)R.off[ex + 1u];
    const Cls c = pick(cl, ex);
    if (r1 - y <= 4u) {
        gemv_expert<4, SUB>(R, c, ex, y, r1, lds, tab, make);
        return;
    }
    uint8_t* tr = lds + (threadIdx.x / kWave) * (16u * kPitch);
    if (n0 >= R.N) return;
    if (n0 >= c.n) {   /* a narrower class: its columns past n are zero */
        for (uint32_t i = y + half; i < r1; i += 2u) R.Y[(size_t)i * R.y_ld + n0 + slot] = 0;
        return;
    }
    const int8_t* ap[MT];
    int32_t at[MT][8];
#pragma unroll
    for (uint32_t mt = 0; mt < MT; ++mt) {
        const uint32_t ia = y + mt * 16u + slot;
        const int32_t fa = ia < r1 ? R.sorted[ia] : -1;
        ap[mt] = fa >= 0 ? R.A + (size_t)(R.a_sorted ? ia : (uint32_t)fa / R.top_k) * R.a_ld : nullptr;
#pragma unroll
        for (uint32_t e = 0; e < 8; ++e) {
            const uint32_t i = y + mt * 16u + 2u * e + half;
            const int32_t f = i < r1 ? R.sorted[i] : -1;
            at[mt][e] = f < 0 ? -1 : (int32_t)(R.a_sorted ? i : (uint32_t)f / R.top_k);
        }
    }
    const auto W = make(c, ex >> 1, tab);
    const uint32_t nkb = c.k / 128u;
    constexpr uint32_t SUBS = 128u / SUB;
    v8f part[MT];
#pragma unroll
    for (uint32_t mt = 0; mt < MT; ++mt) part[mt] = (v8f)0.f;
    auto cur = W.fetch(n0, 0, lane);
    for (uint32_t kb = 0; kb < nkb; ++kb) {
        const auto nxt = W.fetch(n0, kb + 1u < nkb ? kb + 1u : kb, lane);
        {
            uint32_t w[16];
            W.make(cur, n0, kb, lane, w);
#pragma unroll
            for (uint32_t q = 0; q < 16; ++q) *(uint32_t*)(tr + q * kPitch + 4u * lane) = w[q];
        }
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront", "local");
        __builtin_amdgcn_wave_barrier();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront", "local");
#pragma unroll
        for (uint32_t sg = 0; sg < SUBS; ++sg) {
            v4i bf[8u / SUBS];
#pragma unroll
            for (uint32_t u = 0; u < 8u / SUBS; ++u) {
                const uint4 bv = *(const uint4*)(tr + slot * kPitch + sg * (128u / SUBS) + u * 16u);
                bf[u][0] = (int)bv.x; bf[u][1] = (int)bv.y; bf[u][2] = (int)bv.z; bf[u][3] = (int)bv.w;
            }
            const uint32_t kabs = kb * 128u + sg * (128u / SUBS);
            const float ws = W.fold(n0 + slot, kb * SUBS + sg);
#pragma unroll
            for (uint32_t mt = 0; mt < MT; ++mt) {
                v8i tmp = (v8i)0;
#pragma unroll
                for (uint32_t u = 0; u < 8u / SUBS; ++u) {
                    v4i af = (v4i)0;
                    if (ap[mt]) {
                        const uint4 av = *(const uint4*)(ap[mt] + kabs + u * 16u);
                        af[0] = (int)av.x; af[1] = (int)av.y; af[2] = (int)av.z; af[3] = (int)av.w;
                    }
                    tmp = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, af, true, bf[u], tmp, false);
                }
#pragma unroll
                for (uint32_t e = 0; e < 8; ++e)
                    if (at[mt][e] >= 0)
                        part[mt][e] = fmaf((float)tmp[e],
                                           R.As[(size_t)at[mt][e] * R.as_ld + kabs / R.a_group] * ws,
                                           part[mt][e]);
            }
        }
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront", "local");
        __builtin_amdgcn_wave_barrier();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront", "local");
        cur = nxt;
    }
    const float rs = W.row(n0 + slot);
#pragma unroll
    for (uint32_t mt = 0; mt < MT; ++mt)
#pragma unroll
        for (uint32_t e = 0; e < 8; ++e) {
            const uint32_t i = y + mt * 16u + 2u * e + half;
            if (i < r1) R.Y[(size_t)i * R.y_ld + n0 + slot] = f32_to_bf16(0.f + part[mt][e] * rs);
        }
}

template <uint32_t MT>
__global__ __launch_bounds__(kDecW * kWave) void moe_w4nl_dec_kernel(Route R, Cls c0, Cls c1) {
    __shared__ __attribute__((aligned(16))) uint8_t s_tr[kDecW][16 * kPitch];
    const Cls cl[2] = { c0, c1 };
    dec_expert<MT, W4nl::kSub>(R, cl, &s_tr[0][0], nullptr, W4Make{});
}

template <class F, uint32_t MT>
__global__ __launch_bounds__(kDecW * kWave) void moe_iqr_dec_kernel(Route R, Cls c0, Cls c1) {
    __shared__ __attribute__((aligned(16))) uint8_t s_tr[kDecW][16 * kPitch];
    __shared__ __attribute__((aligned(16))) int8_t s_tab[Dec<F>::kTab];
    const Cls cl[2] = { c0, c1 };
    dec_expert<MT, Iqr<F>::kSub>(R, cl, &s_tr[0][0], s_tab, IqrMake<F>{});
}

/* The routing and output operands shared by both rows, checked as halo_moe_gemm_q.hip's launcher
 * checks them. [0] a [1] a_scale [2] w [3] w_scale [4] sorted [5] off [6] w_odd [7] w_odd_scale
 * [8] y. */
struct Ops {
    const RadTensor *A, *As, *sorted, *off, *Y, *Wc[2], *Wsc[2];
    long long Nc[2], Kc[2], N, K, T, M, parts, n_expert, top_k, a_group, a_ld, as_ld, y_ld;
    bool a_sorted;
};

int ops_of(const RadArgs* a, Ops* o) {
    o->A = halo_opt(a, 0);
    o->As = halo_opt(a, 1);
    o->sorted = halo_opt(a, 4);
    o->off = halo_opt(a, 5);
    o->Y = halo_opt(a, 8);
    o->Wc[0] = halo_opt(a, 2);
    o->Wsc[0] = halo_opt(a, 3);
    o->Wc[1] = halo_opt(a, 6);
    o->Wsc[1] = halo_opt(a, 7);
    if (!o->A || !o->As || !o->sorted || !o->off || !o->Y) return HALO_NO(RAD_E_INVAL);
    for (int q = 0; q < 2; ++q)
        if (!o->Wc[q] || !o->Wsc[q]) return HALO_NO(RAD_E_INVAL);
    o->Nc[0] = halo_geti_or(a, "N", 0);
    o->Kc[0] = halo_geti_or(a, "K", 0);
    o->Nc[1] = halo_geti_or(a, "N_odd", o->Nc[0]);
    o->Kc[1] = halo_geti_or(a, "K_odd", o->Kc[0]);
    o->parts = halo_geti_or(a, "parts", 1);
    o->n_expert = halo_geti_or(a, "n_expert", 0);
    o->top_k = halo_geti_or(a, "top_k", 0);
    o->a_group = halo_geti_or(a, "a_group", 128);
    if (o->parts <= 0 || o->n_expert <= 0 || o->top_k <= 0) return HALO_NO(RAD_E_SHAPE);
    if (halo_geti_or(a, "group", 128) != 128) return HALO_NO(RAD_E_SHAPE);
    if (o->a_group != 128 && o->a_group != 256) return HALO_NO(RAD_E_UNSUPPORTED);
    o->N = o->K = 0;
    for (int q = 0; q < 2; ++q) {
        if (o->Nc[q] <= 0 || o->Kc[q] <= 0) return HALO_NO(RAD_E_SHAPE);
        if (o->Kc[q] % o->a_group || o->Nc[q] % (16 * o->parts)) return HALO_NO(RAD_E_SHAPE);
        if (o->Nc[q] > o->N) o->N = o->Nc[q];
        if (o->Kc[q] > o->K) o->K = o->Kc[q];
    }
    if (o->A->dtype != RAD_I8 || o->As->dtype != RAD_F32 || o->Y->dtype != RAD_BF16 ||
        o->sorted->dtype != RAD_I32 || o->off->dtype != RAD_I32)
        return HALO_NO(RAD_E_DTYPE);
    o->y_ld = halo_rows_pitch(o->Y, o->N, &o->T);
    if (o->y_ld < 0 || o->T <= 0) return HALO_NO(RAD_E_STRIDE);
    o->a_ld = halo_rows_pitch(o->A, o->K, &o->M);
    if (o->a_ld < 0 || o->M <= 0 || o->a_ld % 16) return HALO_NO(RAD_E_STRIDE);
    long long Ms = 0;
    o->as_ld = halo_rows_pitch(o->As, o->K / o->a_group, &Ms);
    if (o->as_ld < 0 || Ms < o->M) return HALO_NO(RAD_E_STRIDE);
    if (halo_numel(o->sorted) < o->T || halo_numel(o->off) < o->n_expert + 1) return HALO_NO(RAD_E_SHAPE);
    const char* ord = halo_gets(a, "a_order");
    o->a_sorted = ord && !std::strcmp(ord, "sorted");
    if (ord && !o->a_sorted && std::strcmp(ord, "token")) return HALO_NO(RAD_E_UNSUPPORTED);
    if (o->M < (o->a_sorted ? o->T : (o->T + o->top_k - 1) / o->top_k)) return HALO_NO(RAD_E_SHAPE);
    if (!halo_aligned(o->A->data, 16)) return HALO_NO(RAD_E_ALIGN);
    return RAD_OK;
}

Route route_of(const Ops& o) {
    Route R{};
    R.A = (const int8_t*)o.A->data;
    R.As = (const float*)o.As->data;
    R.sorted = (const int32_t*)o.sorted->data;
    R.off = (const int32_t*)o.off->data;
    R.Y = (uint16_t*)o.Y->data;
    R.T = (uint32_t)o.T;
    R.N = (uint32_t)o.N;
    R.n_expert = (uint32_t)o.n_expert;
    R.top_k = (uint32_t)o.top_k;
    R.a_ld = (uint32_t)o.a_ld;
    R.as_ld = (uint32_t)o.as_ld;
    R.y_ld = (uint32_t)o.y_ld;
    R.a_group = (uint32_t)o.a_group;
    R.a_sorted = o.a_sorted ? 1u : 0u;
    return R;
}

dim3 grid_of(const Ops& o) {
    /* z splits a row tile's experts over waves: a decode tile's rows are each a different
     * expert, a prefill tile's a run of one or two. Measured on gfx1151 (16 experts of layer 0,
     * gate_up / down): T=10 z=8 86 / 110 us against z=1 166 / 203; T=40 z=2 164 / 187 against
     * 210 / 270; T=160 z=2 245 / 246 against 243 / 274; T>=370 z=1. HALO_MOE_Z overrides. */
    const char* ze = std::getenv("HALO_MOE_Z");
    const unsigned z = ze && *ze ? (unsigned)std::atoi(ze)
                     : o.T <= 16 ? 8u : o.T <= 256 ? 2u : 1u;
    return dim3((unsigned)((o.T + 15) / 16), (unsigned)((o.N / 16 + kWaves - 1) / kWaves), z);
}

bool dec_form(const Ops& o) {
    if (o.T > 64) return false;
    /* gemv_expert stages four rows of activations and scales in the transpose LDS */
    const long long kmax = o.Kc[0] > o.Kc[1] ? o.Kc[0] : o.Kc[1];
    if (4 * kmax + 4 * 4 * ((kmax + o.a_group - 1) / o.a_group) > (long long)kDecW * 16 * kPitch) return false;
    if (o.a_ld % 16) return false;
    const char* e = std::getenv("HALO_MOE_DEC");
    return !(e && *e == '0');
}
dim3 dec_grid(const Ops& o) {
    return dim3((unsigned)((o.N + 16 * kDecW - 1) / (16 * kDecW)), (unsigned)o.T);
}

/* A weight-table operand: rank 3 [entries, rows, cols]; `bits` the element width, since a
 * plugin-private dtype has none (halo_moe_gemm_q.hip's wtab_of). */
int tab_of(const RadTensor* t, uint32_t dtype, int bits, long long cnt, long long rows,
           long long cols, const void* const** tp, uint64_t* s0, long long* ld) {
    if (!t || t->rank != 3) return HALO_NO(RAD_E_SHAPE);
    if (cnt > 0 && !t->data) return HALO_NO(RAD_E_SHAPE);
    if (t->dtype != dtype) return HALO_NO(RAD_E_DTYPE);
    if (t->shape[0] < cnt || t->shape[1] < rows || t->shape[2] < cols) return HALO_NO(RAD_E_SHAPE);
    if (t->stride[2] != 1) return HALO_NO(RAD_E_STRIDE);
    *tp = (const void* const*)t->data;
    *s0 = (uint64_t)((t->stride[0] * (int64_t)bits) / 8);
    *ld = t->stride[1];
    return RAD_OK;
}

int w4nl_run(const RadArgs* a, RadStream s) {
    Ops o;
    int rc = ops_of(a, &o);
    if (rc != RAD_OK) return rc;
    const char* wdt = halo_gets(a, "dtype");
    if (!wdt || std::strcmp(wdt, "w4nl64a8i8h")) return HALO_NO(RAD_E_DTYPE);
    Cls cl[2]{};
    for (int q = 0; q < 2; ++q) {
        const long long cnt = (o.n_expert + 1 - q) / 2;
        long long wld = 0, sld = 0;
        rc = tab_of(o.Wc[q], (uint32_t)HALO_DT_MOEW4, 4, cnt, o.Nc[q], o.Kc[q], &cl[q].w, &cl[q].w_s0, &wld);
        if (rc != RAD_OK) return rc;
        rc = tab_of(o.Wsc[q], RAD_F8E4M3, 8, cnt, o.Nc[q], o.Kc[q] / 64, &cl[q].s, &cl[q].s_s0, &sld);
        if (rc != RAD_OK) return rc;
        if (wld % 2 || o.Kc[q] % 128) return HALO_NO(RAD_E_STRIDE);
        cl[q].w_ld = (uint32_t)(wld / 2);
        cl[q].s_ld = (uint32_t)(o.Kc[q] / 64);
        cl[q].n = (uint32_t)o.Nc[q];
        cl[q].k = (uint32_t)o.Kc[q];
    }
    if (dec_form(o)) {
        const dim3 g = dec_grid(o), b(kDecW * kWave);
        hipStream_t st = (hipStream_t)s;
        if (o.T <= 16) hipLaunchKernelGGL(moe_w4nl_dec_kernel<1>, g, b, 0, st, route_of(o), cl[0], cl[1]);
        else if (o.T <= 32) hipLaunchKernelGGL(moe_w4nl_dec_kernel<2>, g, b, 0, st, route_of(o), cl[0], cl[1]);
        else hipLaunchKernelGGL(moe_w4nl_dec_kernel<4>, g, b, 0, st, route_of(o), cl[0], cl[1]);
        return hipGetLastError() == hipSuccess ? RAD_OK : HALO_NO(RAD_E_DEVICE);
    }
    hipLaunchKernelGGL(moe_w4nl_i8_kernel, grid_of(o), dim3(kWaves * kWave), 0, (hipStream_t)s,
                       route_of(o), cl[0], cl[1]);
    return hipGetLastError() == hipSuccess ? RAD_OK : HALO_NO(RAD_E_DEVICE);
}

#define HALO_MOE_IQR_T(b, l) FormT<kTrellisMad, b, 0, l, 0, kRowF32>

template <class F>
int iqr_dec_go(const Ops& o, const Cls (&cl)[2], hipStream_t st) {
    const dim3 g = dec_grid(o), b(kDecW * kWave);
    if (o.T <= 16) hipLaunchKernelGGL((moe_iqr_dec_kernel<F, 1>), g, b, 0, st, route_of(o), cl[0], cl[1]);
    else if (o.T <= 32) hipLaunchKernelGGL((moe_iqr_dec_kernel<F, 2>), g, b, 0, st, route_of(o), cl[0], cl[1]);
    else hipLaunchKernelGGL((moe_iqr_dec_kernel<F, 4>), g, b, 0, st, route_of(o), cl[0], cl[1]);
    return hipGetLastError() == hipSuccess ? RAD_OK : HALO_NO(RAD_E_DEVICE);
}

int iqr_dec_launch(uint32_t kw, const Ops& o, const Cls (&cl)[2], hipStream_t st) {
#define HALO_MOE_IQR_DEC(b, l) \
    if (kw == HALO_MOE_IQR_T(b, l)::word()) return iqr_dec_go<HALO_MOE_IQR_T(b, l)>(o, cl, st);
    HALO_MOE_IQR_DEC(2, 16) HALO_MOE_IQR_DEC(3, 16) HALO_MOE_IQR_DEC(4, 16) HALO_MOE_IQR_DEC(5, 16)
    HALO_MOE_IQR_DEC(6, 16) HALO_MOE_IQR_DEC(7, 16) HALO_MOE_IQR_DEC(2, 15)
#undef HALO_MOE_IQR_DEC
    return HALO_NO(RAD_E_DTYPE);
}

int iqr_run(const RadArgs* a, RadStream s) {
    Ops o;
    int rc = ops_of(a, &o);
    if (rc != RAD_OK) return rc;
    if (o.n_expert > 512) return HALO_NO(RAD_E_UNSUPPORTED);
    const uint32_t wdt = o.Wc[0]->dtype;
    if (!is_weight_dtype(wdt)) return HALO_NO(RAD_E_DTYPE);
    const uint32_t wd = wdt & kWordMask;
    const Form fm = form(wd);
    if (fm.kind != kTrellisMad || fm.scale != kRowF32 || fm.lrb != 0 || fm.lrun < 4)
        return HALO_NO(RAD_E_DTYPE);
    const long long run = (long long)1 << fm.lrun;
    Cls cl[2]{};
    for (int q = 0; q < 2; ++q) {
        const long long cnt = (o.n_expert + 1 - q) / 2;
        const RadTensor *w = o.Wc[q], *ws = o.Wsc[q];
        if (w->dtype != wdt || ws->dtype != (HALO_DT_IQR_S | wd)) return HALO_NO(RAD_E_DTYPE);
        /* Pointer tables, one entry an expert, as libr4d's moe_gemm_iqr requires. */
        if (w->rank != 3 || ws->rank != 3 || w->stride[0] != 0 || ws->stride[0] != 0 ||
            w->shape[0] < cnt || ws->shape[0] < cnt || (cnt > 0 && (!w->data || !ws->data)))
            return HALO_NO(RAD_E_SHAPE);
        if (o.Kc[q] % 256 || (o.Nc[q] / o.parts) % run) return HALO_NO(RAD_E_SHAPE);
        cl[q].w = (const void* const*)w->data;
        cl[q].s = (const void* const*)ws->data;
        cl[q].n = (uint32_t)o.Nc[q];
        cl[q].k = (uint32_t)o.Kc[q];
        cl[q].run_tiles = (uint32_t)(run / 16);
    }
    const uint32_t kw = kernel_word(wd);
    hipStream_t st = (hipStream_t)s;
    if (dec_form(o)) return iqr_dec_launch(kw, o, cl, st);
#define HALO_MOE_IQR_GO(b, l)                                                                  \
    if (kw == HALO_MOE_IQR_T(b, l)::word()) {                                                  \
        hipLaunchKernelGGL(moe_iqr_i8_kernel<HALO_MOE_IQR_T(b, l)>, grid_of(o),                \
                           dim3(kWaves * kWave), 0, st, route_of(o), cl[0], cl[1]);            \
        return hipGetLastError() == hipSuccess ? RAD_OK : HALO_NO(RAD_E_DEVICE);               \
    }
    HALO_MOE_IQR_GO(2, 16) HALO_MOE_IQR_GO(3, 16) HALO_MOE_IQR_GO(4, 16) HALO_MOE_IQR_GO(5, 16)
    HALO_MOE_IQR_GO(6, 16) HALO_MOE_IQR_GO(7, 16) HALO_MOE_IQR_GO(2, 15)
#undef HALO_MOE_IQR_GO
    return HALO_NO(RAD_E_DTYPE);
}

}  // namespace

extern "C" int halo_rad_moe_gemm_w4a8i8(const RadArgs* a, RadStream s) { return w4nl_run(a, s); }
extern "C" int halo_rad_moe_gemm_iqr(const RadArgs* a, RadStream s) { return iqr_run(a, s); }
