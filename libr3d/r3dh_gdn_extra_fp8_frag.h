#pragma once
#include <cstdint>

namespace r3dh_gdn_extra_fp8 {

enum { kFragRows = 16, kFragK = 16, kSuperK = 64, kTileBytes = 1024, kLaneBytes = 32 };

inline int64_t frag_bytes(int64_t rows, int64_t columns) {
    return rows * columns;
}

inline void frag_src(int64_t row_tile, int64_t key_tile, int64_t lane,
                     int64_t fragment, int64_t byte, int64_t* row, int64_t* column) {
    *row = row_tile * static_cast<int64_t>(kFragRows) + (lane & 15);
    *column = key_tile * static_cast<int64_t>(kSuperK)
        + fragment * static_cast<int64_t>(kFragK) + (lane >> 4) * 8 + byte;
}

}
