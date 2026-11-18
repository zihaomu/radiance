#include <hip/hip_runtime.h>
#include "r3dh_moe_iqr.h"
#include "rad_format.h"
#include <fstream>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#if defined(__FAST_MATH__)
#error strict oracle requires -fno-fast-math -ffp-contract=off
#endif

namespace {
using namespace r3dh_moe_iqr;
uint64_t checks = 0;
uint64_t gpu_cases = 0, gpu_runs = 0, gpu_output_values = 0, gpu_decoded_values = 0;
void expect(bool condition, const char* message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}
void hip_check(hipError_t status) {
    if (status != hipSuccess) throw std::runtime_error(hipGetErrorString(status));
}
int8_t reference_mad(uint32_t state, uint32_t multiplier, uint32_t increment) {
    uint32_t word = static_cast<uint32_t>(static_cast<uint64_t>(state & 0xffffffu)
        * (multiplier & 0xffffffu) + increment);
    int total = -126;
    for (unsigned byte = 0; byte < 4; ++byte) total += (word >> (byte * 8)) & 63u;
    return static_cast<int8_t>(total);
}
uint16_t reference_bf16(float value) {
    uint32_t word = std::bit_cast<uint32_t>(value);
    if ((word & 0x7fffffffu) > 0x7f800000u) return static_cast<uint16_t>((word >> 16) | 64u);
    const uint32_t tail = word & 65535u;
    uint32_t head = word >> 16;
    if (tail > 32768u || (tail == 32768u && (head & 1u))) ++head;
    return static_cast<uint16_t>(head);
}
std::vector<RadParam> parameters(int tokens, int even = 512, int odd = 768) {
    return {RAD_INT("M", tokens), RAD_INT("N", even), RAD_INT("K", 2560),
        RAD_INT("N_odd", odd), RAD_INT("K_odd", 2560), RAD_INT("n_expert", 512),
        RAD_INT("top_k", 10), RAD_INT("group", 128), RAD_INT("a_group", 128),
        RAD_INT("parts", 2), RAD_STR("a_order", "token"), RAD_STR("dtype", "i8a8h")};
}
RadTensor matrix(void* data, uint32_t type, int rows, int columns, int pitch = 0) {
    RadTensor result{};
    result.data = data; result.dtype = type; result.rank = 2;
    result.shape[0] = rows; result.shape[1] = columns;
    result.stride[0] = pitch ? pitch : columns; result.stride[1] = 1;
    return result;
}
struct Fixture {
    int rows;
    std::vector<uint8_t> codes;
    std::array<uint32_t, 2> coefficients{0xfc13a7d5u, 0x76543210u};
    std::vector<int8_t> table;
    std::vector<float> row, column;
    RadEncoding encoding{};
    explicit Fixture(int width, unsigned seed = 7) : rows(width), codes(width * 1280),
        table(65536), row(width), column(20) {
        std::strcpy(encoding.scheme, "trellis");
        std::strcpy(encoding.transform, "fwht128");
        rad_enc_add_plane(&encoding, "codes", RAD_U4, 1, 1);
        const int trellis = rad_enc_add_table(&encoding, "trellis", RAD_I8, 65536, 1);
        encoding.plane[trellis].block[0] = 128;
        encoding.plane[trellis].block[1] = 1;
        rad_enc_add_table(&encoding, "trellis.mad", RAD_U32, 2, 1);
        rad_enc_add_plane(&encoding, "scale", RAD_F32, 1, 0);
        rad_enc_add_plane(&encoding, "scale.col", RAD_F32, 0, 128);
        for (auto& code : codes) {
            seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
            code = static_cast<uint8_t>(seed);
        }
        for (unsigned state = 0; state < 65536; ++state)
            table[state] = reference_mad(state, coefficients[0], coefficients[1]);
        for (int index = 0; index < width; ++index) row[index] = (index % 31 + 1) * 0.000137f;
        for (int index = 0; index < 20; ++index) column[index] = (index % 7 + 1) * 0.01317f;
    }
    std::array<RadTensor, 5> planes() {
        return {matrix(codes.data(), RAD_U4, rows, 2560), matrix(table.data(), RAD_I8, 65536, 1),
            matrix(coefficients.data(), RAD_U32, 2, 1), matrix(row.data(), RAD_F32, rows, 1),
            matrix(column.data(), RAD_F32, 1, 20)};
    }
    std::vector<uint8_t> encode(std::vector<RadParam>& params, int operand) {
        auto source = planes();
        const bool weight = operand == 2 || operand == 6;
        const char* roles[3] = {weight ? "codes" : "scale", weight ? "trellis" : "scale.col", "trellis.mad"};
        int selected[3]{};
        for (int index = 0; index < (weight ? 3 : 2); ++index) {
            const auto* plane = rad_enc_plane(&encoding, roles[index]);
            expect(plane != nullptr, "selected role present");
            selected[index] = static_cast<int>(plane - encoding.plane);
        }
        const RadTensor* input = source.data() + (weight ? 0 : 3);
        RadLayout layout{};
        expect(r3dh_moe_iqr_layout(params.data(), params.size(), operand, &encoding,
            selected, input, weight ? 3 : 2, &layout) == RAD_OK, "layout");
        expect(layout.dtype == (weight ? weight_dtype : scale_dtype), "private dtype");
        std::vector<uint8_t> output(layout.bytes);
        expect(r3dh_moe_iqr_relayout(params.data(), params.size(), operand, &encoding,
            selected, input, weight ? 3 : 2, output.data(), output.size()) == RAD_OK, "relayout");
        return output;
    }
    std::vector<int8_t> decode() const {
        std::vector<int8_t> output(static_cast<size_t>(rows) * 2560);
        for (int column_index = 0; column_index < 2560; ++column_index) {
            const auto nibble = [&](int row_index) {
                const uint8_t packed = codes[static_cast<size_t>(row_index) * 1280 + column_index / 2];
                return static_cast<uint32_t>((column_index & 1) ? packed >> 4 : packed & 15);
            };
            for (int run = 0; run < rows; run += 128) {
                uint32_t history = nibble(run + 127) | (nibble(run + 126) << 4)
                                 | (nibble(run + 125) << 8);
                for (int row_index = run; row_index < run + 128; ++row_index) {
                    history = ((history << 4) | nibble(row_index)) & 65535u;
                    output[static_cast<size_t>(row_index) * 2560 + column_index]
                        = table[history];
                }
            }
        }
        return output;
    }
};
void host_tests() {
    for (uint32_t multiplier : {0u, 1u, 0xffffffffu, 0xfc13a7d5u})
        for (uint32_t state_index = 0; state_index < 65536; ++state_index)
            expect(mad(state_index, multiplier, 0xfedcba98u)
                == reference_mad(state_index, multiplier, 0xfedcba98u), "mad exhaustive");
    for (int width : {512, 768}) {
        auto params = parameters(3, width, width == 512 ? 768 : 512);
        Fixture fixture(width), inverse(width, 31);
        auto weight = fixture.encode(params, 2);
        auto scales = fixture.encode(params, 3);
        auto output = inverse.planes();
        const int codes_selected[] = {0, 1, 2}, scales_selected[] = {3, 4};
        expect(r3dh_moe_iqr_unrelayout(params.data(), params.size(), 2, &fixture.encoding,
            codes_selected, weight.data(), weight.size(), output.data(), 3) == RAD_OK, "inverse weight");
        expect(r3dh_moe_iqr_unrelayout(params.data(), params.size(), 3, &fixture.encoding,
            scales_selected, scales.data(), scales.size(), output.data() + 3, 2) == RAD_OK, "inverse scale");
        expect(inverse.codes == fixture.codes && inverse.table == fixture.table
            && inverse.coefficients == fixture.coefficients && inverse.row == fixture.row
            && inverse.column == fixture.column, "inverse exact");
        const auto decoded = fixture.decode();
        for (int row_index = 0; row_index < width; ++row_index)
            for (int column_index : {0, 1, 127, 128, 2559})
                expect(decoded[static_cast<size_t>(row_index) * 2560 + column_index]
                    == mad(state(fixture.codes.data(), 2560, row_index, column_index),
                        fixture.coefficients[0], fixture.coefficients[1]), "cyclic decoder");
        auto source = fixture.planes();
        auto relayout = [&] {
            return r3dh_moe_iqr_relayout(params.data(), params.size(), 2, &fixture.encoding,
                codes_selected, source.data(), 3, weight.data(), weight.size());
        };
        fixture.table[53] ^= 1;
        expect(relayout() == RAD_E_FORMAT, "reject table mismatch");
        fixture.table[53] ^= 1;
        for (uint32_t bad_type : {RAD_U3, RAD_U5, RAD_I8}) {
            fixture.encoding.plane[0].dtype = bad_type;
            expect(relayout() != RAD_OK, "reject non-u4");
        }
        fixture.encoding.plane[0].dtype = RAD_U4;
        fixture.encoding.plane[1].block[0] = 256;
        expect(relayout() != RAD_OK, "reject run256");
        fixture.encoding.plane[1].block[0] = 128;
        weight[0] ^= 1;
        expect(r3dh_moe_iqr_unrelayout(params.data(), params.size(), 2, &fixture.encoding,
            codes_selected, weight.data(), weight.size(), output.data(), 3) == RAD_E_FORMAT,
            "reject corrupt header");
        weight[0] ^= 1;
        expect(r3dh_moe_iqr_unrelayout(params.data(), params.size(), 2, &fixture.encoding,
            codes_selected, weight.data(), weight.size() - 1, output.data(), 3) != RAD_OK, "reject size");
        for (int operand = 0; operand < 9; ++operand) {
            RadOpdDesc description{};
            expect(r3dh_moe_iqr_shape(params.data(), params.size(), operand, &description) == RAD_OK,
                   "shape mapping");
            if (operand == 6) expect(description.shape[0] == (width == 512 ? 768 : 512), "odd shape");
            if (operand == 8) expect(description.shape[0] == 30 && description.shape[1] == 768,
                                     "padded output shape");
        }
        params[10] = RAD_STR("a_order", "sorted");
        Config geometry;
        expect(config(params.data(), params.size(), &geometry) == RAD_E_UNSUPPORTED, "reject down");
        params[10] = RAD_STR("a_order", "token");
        params[0] = RAD_INT("M", 2049);
        expect(config(params.data(), params.size(), &geometry) != RAD_OK, "reject unbounded M");
    }
    std::printf("PASS host checks=%llu strict_u4_l16_run128 parity512_768\n",
                static_cast<unsigned long long>(checks));
}
#include "r3dh_moe_iqr_gpu_check.h"
#include "r3dh_moe_iqr_actual_check.h"
}

int main(int argc, char** argv) {
    try {
        if (argc == 3 && (!std::strcmp(argv[1], "--actual-host") || !std::strcmp(argv[1], "--actual-gpu"))) {
            host_tests();
            actual_tests(argv[2], !std::strcmp(argv[1], "--actual-gpu"));
            std::printf("PASS total checks=%llu gpu_cases=%llu gpu_runs=%llu output_values=%llu decoded_values=%llu\n",
                static_cast<unsigned long long>(checks), static_cast<unsigned long long>(gpu_cases),
                static_cast<unsigned long long>(gpu_runs), static_cast<unsigned long long>(gpu_output_values),
                static_cast<unsigned long long>(gpu_decoded_values));
            return 0;
        }
        if (argc != 2 || (std::strcmp(argv[1], "--host") && std::strcmp(argv[1], "--gpu")))
            throw std::runtime_error("usage: r3dh_moe_iqr_check --host|--gpu|--actual-host MODEL|--actual-gpu MODEL");
        host_tests();
        if (!std::strcmp(argv[1], "--gpu")) synthetic_gpu_tests();
        std::printf("PASS total checks=%llu gpu_cases=%llu gpu_runs=%llu output_values=%llu decoded_values=%llu\n",
            static_cast<unsigned long long>(checks), static_cast<unsigned long long>(gpu_cases),
            static_cast<unsigned long long>(gpu_runs), static_cast<unsigned long long>(gpu_output_values),
            static_cast<unsigned long long>(gpu_decoded_values));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL %s\n", error.what());
        return 1;
    }
}
