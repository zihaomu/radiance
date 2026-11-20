#ifndef R3DH_ATTN_QSA_H
#define R3DH_ATTN_QSA_H
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ != 0)
#error r3dh_attn_qsa requires strict floating point
#endif
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
#include "rad_abi.h"
extern "C" int r3dh_attn_qsa_rope_table(const RadArgs*, RadStream);
extern "C" int r3dh_attn_qsa_prepare(const RadArgs*, RadStream);
inline constexpr int64_t r3dh_attn_qsa_max_user_sequences = 8;
inline constexpr int64_t r3dh_attn_qsa_max_context = 200000;
inline constexpr int64_t r3dh_attn_qsa_max_batch = 2048;
inline constexpr int64_t r3dh_attn_qsa_max_cache_pages =
    r3dh_attn_qsa_max_user_sequences * ((r3dh_attn_qsa_max_context + 3) / 4 + (32 + 3) / 4 + 1)
    + (r3dh_attn_qsa_max_batch + 3) / 4;
extern "C" int r3dh_attn_qsa_rope(const RadArgs*, RadStream);
extern "C" int r3dh_attn_qsa_kv_store(const RadArgs*, RadStream);
extern "C" int r3dh_attn_qsa_dense(const RadArgs*, RadStream);
extern "C" int r3dh_attn_qsa_paged(const RadArgs*, RadStream);
extern "C" int r3dh_attn_qsa_work(const RadArgs*, RadStream);
extern "C" int r3dh_attn_qsa_tail_store(const RadArgs*, RadStream);
extern "C" int r3dh_attn_qsa_paged_gate(const RadArgs*, RadStream);
extern "C" int64_t r3dh_attn_qsa_paged_gate_scratch(const RadArgs*);
extern "C" int r3dh_attn_qsa_prepare_store(const RadArgs*, RadStream);
#endif
