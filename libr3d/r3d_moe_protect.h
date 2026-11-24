#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdlib>

namespace r3d_scatter {
constexpr uint32_t kMaxProtect = 8;

struct Protect {
    uint32_t n;
    uint16_t e[kMaxProtect];
};

inline bool read_protect(const char* text, long long experts, Protect& protect) {
    long long previous = -1;
    while (text && *text) {
        char* end = nullptr;
        const long long value = std::strtoll(text, &end, 10);
        if (end == text || value <= previous || value >= experts || protect.n >= kMaxProtect) return false;
        protect.e[protect.n++] = static_cast<uint16_t>(value);
        previous = value;
        text = end;
        if (*text == ',') ++text;
        else if (*text) return false;
    }
    return true;
}

__device__ __forceinline__ int32_t sort_place(int32_t expert, uint32_t local, const Protect& protect) {
    int32_t below = 0;
    for (uint32_t index = 0; index < protect.n; ++index) {
        if (int32_t(protect.e[index]) == expert) return int32_t(local - protect.n + index);
        below += int32_t(protect.e[index]) < expert;
    }
    return expert - below;
}
}