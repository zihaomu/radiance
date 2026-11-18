#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>

template <int Offset>
__device__ __forceinline__ uint32_t r3dh_xor_lane_u32(uint32_t value) {
    static_assert(Offset >= 1 && Offset <= 16 && (Offset & (Offset - 1)) == 0);
    if constexpr (Offset < 16)
        return (uint32_t)__builtin_amdgcn_update_dpp(
            0, (int)value, 0x160 | Offset, 0xf, 0xf, true);
    else
        return (uint32_t)__builtin_amdgcn_permlanex16(
            (int)value, (int)value, 0x76543210u, 0xFEDCBA98u, false, false);
}
