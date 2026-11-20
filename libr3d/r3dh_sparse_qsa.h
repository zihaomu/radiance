#ifndef R3DH_SPARSE_QSA_H
#define R3DH_SPARSE_QSA_H
#include "rad_abi.h"
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__)
#error Strict floating point required
#endif
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
extern "C" int r3dh_sparse_qsa_block_key(const RadArgs*, RadStream);
extern "C" int r3dh_sparse_qsa_score(const RadArgs*, RadStream);
extern "C" int r3dh_sparse_qsa_select(const RadArgs*, RadStream);
extern "C" int r3dh_sparse_qsa_paged(const RadArgs*, RadStream);
extern "C" int r3dh_sparse_qsa_shape(const char*, const RadArgs*);
extern "C" const RadOpSchema* r3dh_sparse_qsa_schema_at(int);
extern "C" int r3dh_sparse_qsa_schema_count();
extern "C" int r3dh_sparse_qsa_block_key_shape(const RadParam*, int, int, RadOpdDesc*);
extern "C" int r3dh_sparse_qsa_score_shape(const RadParam*, int, int, RadOpdDesc*);
extern "C" int r3dh_sparse_qsa_select_shape(const RadParam*, int, int, RadOpdDesc*);
extern "C" int r3dh_sparse_qsa_paged_shape(const RadParam*, int, int, RadOpdDesc*);
#endif
