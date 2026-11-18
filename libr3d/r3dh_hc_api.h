#pragma once
#include "rad_abi.h"

struct R3dhHcTrace {
    uint16_t* down;
    float* logits;
    uint16_t* gates;
    int64_t rows, n, hc, lowrank;
};

extern "C" {
int r3dh_hc_rad_hc_read_strict(const RadArgs*, RadStream);
int r3dh_hc_rad_hc_read_strict_trace(const RadArgs*, RadStream, const R3dhHcTrace*);
int64_t r3dh_hc_rad_hc_read_strict_scratch(const RadArgs*);
int r3dh_hc_rad_hc_read_fast(const RadArgs*, RadStream);
int r3dh_hc_rad_hc_read_fast_trace(const RadArgs*, RadStream, const R3dhHcTrace*);
int r3dh_hc_rad_hc_read_trace(const RadArgs*, RadStream, const R3dhHcTrace*);
int r3dh_hc_rad_hc_enter(const RadArgs*, RadStream);
int r3dh_hc_rad_hc_write(const RadArgs*, RadStream);
int r3dh_hc_rad_hc_read(const RadArgs*, RadStream);
int r3dh_hc_shape_hc_enter(const RadParam*, int, int, RadOpdDesc*);
int r3dh_hc_shape_hc_write(const RadParam*, int, int, RadOpdDesc*);
int r3dh_hc_shape_hc_read(const RadParam*, int, int, RadOpdDesc*);
int64_t r3dh_hc_rad_hc_read_scratch(const RadArgs*);
int r3dh_hc_rad_layout_hc_e4m3(const RadParam*, int, int, const RadEncoding*,
                               const int*, const RadTensor*, int, RadLayout*);
int r3dh_hc_rad_relayout_hc_e4m3(const RadParam*, int, int, const RadEncoding*,
                                 const int*, const RadTensor*, int, void*, int64_t);
int r3dh_hc_rad_unrelayout_hc_e4m3(const RadParam*, int, int, const RadEncoding*,
                                   const int*, const void*, int64_t, const RadTensor*, int);
}
