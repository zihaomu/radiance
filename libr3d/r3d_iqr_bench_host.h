#pragma once
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace iqr_bench {
constexpr int columns = 8192;
constexpr int reduction = 2560;
constexpr int groups = 20;
constexpr uint64_t model_bytes = 121678372864ull;
constexpr uint64_t codes_offset = 53946896384ull;
constexpr uint64_t rows_offset = 53973110784ull;
constexpr uint64_t table_offset = 53973176320ull;
constexpr uint64_t mad_offset = 53973241856ull;
constexpr uint64_t cols_offset = 53973242112ull;
constexpr uint32_t input_seed = 20261117u;
constexpr uint32_t order_seed = 0x49515235u;
constexpr size_t rotation_bytes = 256ull * 1024 * 1024;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
struct Random {
    uint32_t state;
    uint32_t next() { state = state * 1664525u + 1013904223u; return state; }
};
uint16_t oracle_bf16(float value) {
    uint32_t word = std::bit_cast<uint32_t>(value);
    if ((word & 0x7fffffffu) > 0x7f800000u)
        return static_cast<uint16_t>((word >> 16) | 64u);
    const uint32_t lower = word & 65535u;
    uint32_t upper = word >> 16;
    if (lower > 32768u || (lower == 32768u && upper % 2u)) ++upper;
    return static_cast<uint16_t>(upper);
}
int oracle_mad(uint32_t state, uint32_t multiplier, uint32_t increment) {
    const uint32_t mixed = static_cast<uint32_t>(uint64_t(state % 16777216u)
        * (multiplier % 16777216u) + increment);
    int total = -126;
    for (unsigned byte = 0; byte != 4; ++byte)
        total += (mixed >> (byte * 8)) % 64u;
    return total;
}
void pread_all(int descriptor, void* target, size_t count, uint64_t offset) {
    auto* bytes = static_cast<uint8_t*>(target);
    while (count) {
        const ssize_t received = pread(descriptor, bytes, count, offset);
        require(received > 0, "short model read");
        bytes += received; count -= received; offset += received;
    }
}
struct Model {
    std::vector<uint8_t> weight;
    std::vector<float> scales;
    std::vector<int8_t> decoded;
    explicit Model(const char* path) : weight(32 + size_t(columns) * 1600),
        scales(columns + groups), decoded(size_t(columns) * reduction) {
        const int descriptor = open(path, O_RDONLY);
        require(descriptor >= 0, "model open failed");
        struct stat information{};
        if (fstat(descriptor, &information) || uint64_t(information.st_size) != model_bytes) {
            close(descriptor); throw std::runtime_error("model filesize identity mismatch");
        }
        uint32_t coefficients[2];
        std::vector<int8_t> table(65536);
        pread_all(descriptor, weight.data() + 32, weight.size() - 32, codes_offset);
        pread_all(descriptor, scales.data(), columns * sizeof(float), rows_offset);
        pread_all(descriptor, scales.data() + columns, groups * sizeof(float), cols_offset);
        pread_all(descriptor, coefficients, sizeof(coefficients), mad_offset);
        pread_all(descriptor, table.data(), table.size(), table_offset);
        close(descriptor);
        for (uint32_t state = 0; state < 65536; ++state)
            require(table[state] == oracle_mad(state, coefficients[0], coefficients[1]),
                "real model table differs from computed mad oracle");
        for (float scale : scales) require(std::isfinite(scale), "nonfinite model scale");
        r3d_iqr::Header header{r3d_iqr::weight_magic, 5, 256, 16,
            coefficients[0], coefficients[1], columns, reduction};
        std::memcpy(weight.data(), &header, sizeof(header));
        for (int start = 0; start < columns; start += 256) {
            for (int position = 0; position < reduction; ++position) {
                auto digit = [&](int row) {
                    unsigned result = 0;
                    for (int bit = 4; bit >= 0; --bit) {
                        const size_t address = size_t(row) * reduction * 5 + position * 5 + bit;
                        result = result * 2 + ((weight[32 + address / 8] >> (address % 8)) % 2);
                    }
                    return result;
                };
                uint32_t state = 0;
                for (int row = start + 252; row < start + 256; ++row)
                    state = (state * 32 + digit(row)) % 65536;
                for (int row = start; row < start + 256; ++row) {
                    state = (state * 32 + digit(row)) % 65536;
                    decoded[size_t(row) * reduction + position] =
                        static_cast<int8_t>(oracle_mad(state, coefficients[0], coefficients[1]));
                }
            }
        }
    }
};
struct Input {
    std::vector<int8_t> activation;
    std::vector<float> scale;
    Input(int rows, unsigned repeat) : activation(size_t(rows) * reduction), scale(rows * groups) {
        Random random{input_seed ^ (uint32_t(rows) * 2654435761u) ^ (repeat * 2246822519u)};
        for (auto& value : activation) value = static_cast<int8_t>(int(random.next() % 256) - 128);
        for (auto& value : scale) value = float(1 + random.next() % 127) / 1003.0f;
    }
};
std::vector<size_t> positions(int rows) {
    std::vector<size_t> selected;
    if (rows <= 32) {
        selected.resize(size_t(rows) * columns);
        for (size_t index = 0; index < selected.size(); ++index) selected[index] = index;
    } else {
        for (int row : {0, 1, 15, 16, 31, rows / 2, rows - 1})
            for (int column : {0, 1, 15, 16, 127, 128, 252, 253, 254, 255, 256, 257, 8191})
                selected.push_back(size_t(row) * columns + column);
        Random random{input_seed ^ uint32_t(rows)};
        while (selected.size() < 4096) selected.push_back(random.next() % (size_t(rows) * columns));
        std::sort(selected.begin(), selected.end());
        selected.erase(std::unique(selected.begin(), selected.end()), selected.end());
    }
    return selected;
}
std::vector<uint16_t> reference(const Model& model, const Input& input,
                              const std::vector<size_t>& selected) {
    std::vector<uint16_t> expected;
    expected.reserve(selected.size());
    for (size_t index : selected) {
        const size_t row = index / columns;
        const size_t column = index % columns;
        float accumulator = 0.0f;
        for (int group = 0; group < groups; ++group) {
            int32_t subtotal = 0;
            for (int offset = 0; offset < 128; ++offset) {
                const int position = group * 128 + offset;
                subtotal += int(input.activation[row * reduction + position])
                    * int(model.decoded[column * reduction + position]);
            }
            const float activation_scaled = float(subtotal) * input.scale[row * groups + group];
            const float weight_scaled = activation_scaled * model.scales[columns + group];
            accumulator = accumulator + weight_scaled;
        }
        expected.push_back(oracle_bf16(accumulator * model.scales[column]));
    }
    return expected;
}
void host_tests() {
    static_assert(sizeof(r3d_iqr::Header) == 32);
    static_assert(r3d_iqr::weight_dtype == 0x33000100u);
    static_assert(r3d_iqr::scale_dtype == 0x33000200u);
    require(oracle_bf16(std::bit_cast<float>(0x3f808000u)) == 0x3f80, "BF16 even tie");
    require(oracle_bf16(std::bit_cast<float>(0x3f818000u)) == 0x3f82, "BF16 odd tie");
    require(oracle_bf16(-1.0f) == 0xbf80, "BF16 negative");
    require(oracle_mad(0, 0, 0) == -126, "mad zero");
    require(oracle_mad(0, 0, 0xffffffffu) == 126, "mad maximum");
    require(r3d_rad_gemm_iqr_scalar(nullptr, nullptr) == RAD_E_INVAL, "null API");
}
}
