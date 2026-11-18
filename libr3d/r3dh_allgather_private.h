#pragma once
#include "r3dh_allgather.h"
#include <cstdint>

struct R3dhAllgatherGeometry {
    int64_t low = 0, high = 0, row = 0;
    uint32_t dtype = 0;
    int bytes = 0;
};
struct R3dhAllgatherView {
    int64_t numel = 0, row = 0, rows = 0;
    int64_t input_pitch = 0, output_pitch = 0;
};
int r3dh_allgather_geometry(const RadParam*, int, R3dhAllgatherGeometry*);
int r3dh_allgather_init_geometry(const RadParam*, int, R3dhAllgatherGeometry*);
int r3dh_allgather_view(const RadArgs*, R3dhAllgatherGeometry*, R3dhAllgatherView*);
constexpr int64_t r3dh_allgather_max_numel = 16 * 1024 * 1024;
