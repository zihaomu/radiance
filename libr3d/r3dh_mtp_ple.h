#pragma once
#include "rad_abi.h"
#include "rad_plugin.h"
inline constexpr uint32_t r3dh_mtp_fc_e4m3_dtype = 0x33000400u;
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__)
#error "r3dh_mtp_ple requires strict floating point"
#endif
extern "C" {
int r3dh_mtp_ple_shape(const char*, const RadParam*, int, int, RadOpdDesc*);
int r3dh_mtp_ple_device(const RadArgs*, RadStream);
int r3dh_mtp_ple_mtp_enter(const RadArgs*, RadStream);
int64_t r3dh_mtp_ple_mtp_enter_scratch(const RadArgs*);
int r3dh_mtp_ple_ple_gate(const RadArgs*, RadStream);
int r3dh_mtp_ple_grid_embed(const RadArgs*, RadStream);
int r3dh_mtp_ple_embed_lookup_q(const RadArgs*, RadStream);
int r3dh_mtp_ple_rad_ngram_ids(const RadArgs*, RadStream);
int r3dh_mtp_ple_ngram_hint(const RadArgs*, RadStream);
int r3dh_mtp_ple_logit_rerank(const RadArgs*, RadStream);
int r3dh_mtp_ple_rowtopkmerge(const RadArgs*, RadStream);
int r3dh_mtp_ple_layout_mtp_e4m3(const RadParam*, int, int, const RadEncoding*, const int*, const RadTensor*, int, RadLayout*);
int r3dh_mtp_ple_relayout_mtp_e4m3(const RadParam*, int, int, const RadEncoding*, const int*, const RadTensor*, int, void*, int64_t);
int r3dh_mtp_ple_unrelayout_mtp_e4m3(const RadParam*, int, int, const RadEncoding*, const int*, const void*, int64_t, const RadTensor*, int);
}
