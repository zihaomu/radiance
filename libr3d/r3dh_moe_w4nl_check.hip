#include <hip/hip_runtime.h>
#include "r3dh_moe_w4nl.h"
#include "rad_format.h"
#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__)
#error W4NL64 checker requires -ffp-contract=off -fno-fast-math
#endif
#pragma clang fp contract(off)
namespace {
using namespace r3dh_moe_w4nl;
uint64_t checks = 0, outputs = 0, gpu_cases = 0;
void expect(bool condition, const char* message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}
void hip_check(hipError_t status) {
    if (status != hipSuccess) throw std::runtime_error(hipGetErrorString(status));
}
std::vector<RadParam> parameters(int tokens, int columns = 16, int even = 384, int odd = 256,
                                 int experts = 512, int top = 10) {
    return {RAD_INT("M", tokens), RAD_INT("N", columns), RAD_INT("K", even),
        RAD_INT("N_odd", columns), RAD_INT("K_odd", odd), RAD_INT("n_expert", experts),
        RAD_INT("top_k", top), RAD_INT("group", 128), RAD_INT("a_group", 128),
        RAD_INT("parts", 1), RAD_STR("a_order", "sorted"), RAD_STR("dtype", "w4nl64a8i8h")};
}
RadTensor matrix(void* data, uint32_t type, int rows, int columns, int pitch = 0) {
    RadTensor result{};
    result.data = data; result.dtype = type; result.rank = 2;
    result.shape[0] = rows; result.shape[1] = columns;
    result.stride[0] = pitch ? pitch : columns; result.stride[1] = 1;
    return result;
}
RadTensor vector_tensor(void* data, int length) {
    RadTensor result{};
    result.data = data; result.dtype = RAD_I32; result.rank = 1;
    result.shape[0] = length; result.stride[0] = 1;
    return result;
}
float reference_scale(uint8_t raw) {
    const int exponent = (raw / 8) % 16, fraction = raw % 8;
    const double magnitude = exponent ? std::ldexp(double(8 + fraction), exponent - 10)
                                      : std::ldexp(double(fraction), -9);
    return float(raw >= 128 ? -magnitude : magnitude);
}
float rounded_product(float left, float right) {
    volatile float result = left * right;
    return result;
}
float rounded_sum(float left, float right) {
    volatile float result = left + right;
    return result;
}
uint16_t reference_bf16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7fffffffu) > 0x7f800000u) return uint16_t((bits >> 16) | 64);
    uint32_t upper = bits / 65536, lower = bits % 65536;
    if (lower > 32768 || (lower == 32768 && (upper % 2))) ++upper;
    return uint16_t(upper);
}
struct Weight {
    int rows, inner;
    std::vector<uint8_t> codes, scales;
    std::array<float, 16> table{};
    float level = 0x1p-13f;
    explicit Weight(int columns, int reduction, unsigned seed = 7)
        : rows(columns), inner(reduction), codes(size_t(columns) * reduction / 2),
          scales(size_t(columns) * reduction / 64) {
        for (size_t index = 0; index < codes.size(); ++index)
            codes[index] = uint8_t(index * 73 + index / 7 + seed * 31);
        for (size_t index = 0; index < scales.size(); ++index) {
            unsigned raw = unsigned(index + seed * 11) % 254;
            if (raw >= 127) ++raw;
            scales[index] = uint8_t(raw);
        }
        for (int index = 0; index < 16; ++index)
            table[index] = float(index - 7) * 0.173123f + float(index % 3) * 0.000317f;
    }
    RadEncoding encoding() const {
        RadEncoding result{};
        std::strcpy(result.scheme, "affine");
        std::strcpy(result.transform, "fwht128");
        expect(rad_enc_add_plane(&result, "codes", RAD_U4, 1, 1) == 0, "codes fixture ABI");
        expect(rad_enc_add_table(&result, "table", RAD_F32, 1, 16) == 1, "table fixture ABI");
        expect(rad_enc_add_plane(&result, "scale", RAD_F8E4M3, 1, 64) == 2, "scale fixture ABI");
        expect(rad_enc_add_plane(&result, "scale.1", RAD_F32, 0, 0) == 3, "scale2 fixture ABI");
        return result;
    }
    std::array<RadTensor, 4> planes() {
        return {matrix(codes.data(), RAD_U4, rows, inner), matrix(table.data(), RAD_F32, 1, 16),
            matrix(scales.data(), RAD_F8E4M3, rows, inner / 64), matrix(&level, RAD_F32, 1, 1)};
    }
    std::vector<uint8_t> encode(std::vector<RadParam>& params, int operand) {
        auto enc = encoding();
        auto tensors = planes();
        const bool code_operand = operand == 2 || operand == 6;
        const int selected[2] = {code_operand ? 0 : 2, code_operand ? 1 : 3};
        RadLayout layout{};
        const RadTensor* source = tensors.data() + (code_operand ? 0 : 2);
        expect(r3dh_moe_w4nl_layout(params.data(), int(params.size()), operand, &enc,
            selected, source, 2, &layout) == RAD_OK, "layout");
        expect(layout.dtype == (code_operand ? codes_type : scales_type), "native dtype");
        expect(layout.rank == 1 && layout.shape[0] == layout.bytes && layout.align == 256, "layout ABI");
        std::vector<uint8_t> result(size_t(layout.bytes));
        expect(r3dh_moe_w4nl_transform(params.data(), int(params.size()), operand, &enc,
            selected, source, 2, result.data(), layout.bytes) == RAD_OK, "transform");
        return result;
    }
};
uint16_t oracle(const Weight& weight, const int8_t* activation, const float* scale, int column) {
    float total = 0;
    for (int group = 0; group < weight.inner / 64; ++group) {
        float partial = 0;
        for (int offset = 0; offset < 64; ++offset) {
            const int position = group * 64 + offset;
            const size_t bit = (size_t(column) * weight.inner + position) * 4;
            unsigned index = 0;
            for (unsigned digit = 0; digit < 4; ++digit)
                if (weight.codes[(bit + digit) / 8] & (1u << ((bit + digit) % 8))) index += 1u << digit;
            partial = rounded_sum(partial, rounded_product(float(activation[position]), weight.table[index]));
        }
        const float weight_scale = rounded_product(
            reference_scale(weight.scales[size_t(column) * (weight.inner / 64) + group]), weight.level);
        total = rounded_sum(total, rounded_product(rounded_product(partial, scale[group / 2]), weight_scale));
    }
    return reference_bf16(total);
}
void host_launch_abi_tests() {
    alignas(256) uint8_t placeholder[256]{};
    auto params = parameters(1);
    std::array<RadTensor, 9> tensors{};
    tensors[0] = matrix(placeholder, RAD_I8, 10, 384);
    tensors[1] = matrix(placeholder, RAD_F32, 10, 3);
    tensors[4] = vector_tensor(placeholder, 10);
    tensors[5] = vector_tensor(placeholder, 513);
    tensors[8] = matrix(placeholder, RAD_BF16, 10, 16);
    for (int operand : {2, 3, 6, 7}) {
        const int inner = operand >= 6 ? 256 : 384;
        const bool codes = operand == 2 || operand == 6;
        const int bytes = codes ? int(sizeof(CodesHeader)) + 16 * inner / 2
                                : int(sizeof(ScalesHeader)) + 16 * inner / 64;
        tensors[operand] = matrix(placeholder, codes ? codes_type : scales_type, 256, bytes);
        tensors[operand].stride[0] = 0;
    }
    RadArgs arguments{};
    arguments.t = tensors.data();
    arguments.n_t = int(tensors.size());
    arguments.p = params.data();
    arguments.n_p = int(params.size());
    arguments.world_size = 2;
    arguments.kv_used_max_h = -1;
    auto reject = [&](int expected, const char* message) {
        expect(r3dh_moe_w4nl_launch(&arguments, nullptr) == expected, message);
    };
    expect(r3dh_moe_w4nl_launch(nullptr, nullptr) == RAD_E_INVAL, "null launch ABI");
    arguments.n_t = 8;
    reject(RAD_E_INVAL, "nine operand ABI");
    arguments.n_t = 9;
    for (int operand : {0, 1, 2, 3, 4, 5, 6, 7, 8}) {
        const RadTensor saved = tensors[operand];
        tensors[operand].data = nullptr;
        reject(RAD_E_INVAL, "null operand ABI");
        tensors[operand] = saved;
        tensors[operand].dtype = RAD_U8;
        reject(RAD_E_DTYPE, "operand dtype ABI");
        tensors[operand] = saved;
        ++tensors[operand].shape[0];
        reject(RAD_E_SHAPE, "operand extent ABI");
        tensors[operand] = saved;
        tensors[operand].stride[0] = -1;
        reject(RAD_E_STRIDE, "negative operand stride ABI");
        tensors[operand] = saved;
    }
    for (int operand : {2, 3, 6, 7}) {
        const RadTensor saved = tensors[operand];
        tensors[operand].stride[1] = 2;
        reject(RAD_E_STRIDE, "WTAB byte stride ABI");
        tensors[operand] = saved;
        tensors[operand].stride[0] = saved.shape[1] - 16;
        reject(RAD_E_STRIDE, "WTAB overlapping blobs ABI");
        tensors[operand].stride[0] = saved.shape[1] + 1;
        reject(RAD_E_STRIDE, "WTAB misaligned blobs ABI");
        tensors[operand] = saved;
    }
    std::puts("W4NL_FIXTURE_ABI_PASS gpu_required=0 operands=9 tp=2 errors=INVAL,DTYPE,SHAPE,STRIDE");
}
void host_reference_tests() {
    const uint8_t raw_scales[] = {0, 128, 1, 129, 7, 8, 56, 126, 254};
    const float decoded[] = {0.0f, -0.0f, 0x1p-9f, -0x1p-9f, 7 * 0x1p-9f,
                             0x1p-6f, 1.0f, 448.0f, -448.0f};
    for (size_t index = 0; index < std::size(raw_scales); ++index) {
        const float got = reference_scale(raw_scales[index]);
        expect(got == decoded[index] && std::signbit(got) == std::signbit(decoded[index]),
               "independent E4M3 golden");
    }
    const float halfway[] = {1.00390625f, 1.01171875f, -1.00390625f, -1.01171875f};
    const uint16_t rounded[] = {0x3f80, 0x3f82, 0xbf80, 0xbf82};
    for (size_t index = 0; index < std::size(halfway); ++index)
        expect(reference_bf16(halfway[index]) == rounded[index], "independent BF16 tie golden");
    for (int inner : {128, 256, 384}) {
        Weight weight(16, inner);
        std::fill(weight.codes.begin(), weight.codes.end(), 0);
        std::fill(weight.scales.begin(), weight.scales.end(), 56);
        weight.table.fill(1.0f);
        std::vector<int8_t> activation(inner, 1);
        const float scale[] = {64.0f, 128.0f, 256.0f};
        const uint16_t expected = inner == 128 ? 0x3f80 : inner == 256 ? 0x4040 : 0x40e0;
        for (int column = 0; column < 16; ++column)
            expect(oracle(weight, activation.data(), scale, column) == expected,
                   "independent scale128/group64 golden");
        std::fill(activation.begin(), activation.end(), 0);
        activation[0] = 1;
        const float identity[] = {8192.0f, 8192.0f, 8192.0f};
        for (int code = 0; code < 16; ++code) {
            weight.table[code] = float(code - 8) + 0.125f;
            weight.codes[0] = uint8_t(code);
            expect(oracle(weight, activation.data(), identity, 0)
                       == reference_bf16(float(code - 8) + 0.125f), "low nibble/table16 golden");
            activation[0] = 0;
            activation[1] = -128;
            weight.codes[0] = uint8_t(code * 16);
            expect(oracle(weight, activation.data(), identity, 0)
                       == reference_bf16(-128.0f * (float(code - 8) + 0.125f)),
                   "high nibble/signed int8 golden");
            activation[0] = 1;
            activation[1] = 0;
        }
        weight.table.fill(0);
        weight.table[0] = 0x1p80f;
        weight.table[1] = -0x1p80f;
        weight.table[2] = 1.0f;
        std::fill(weight.codes.begin(), weight.codes.end(), 0x33);
        weight.codes[0] = 0x10;
        weight.codes[1] = 0x32;
        activation[0] = activation[1] = activation[2] = 1;
        expect(oracle(weight, activation.data(), identity, 0) == 0x3f80,
               "ascending noncontracted cancellation golden");
    }
    std::puts("W4NL_REFERENCE_PASS gpu_required=0 golden=e4m3,bf16_rne,scale128,group64,table16,nibbles,int8,cancellation");
}
void host_tests() {
    expect(std::fesetround(FE_TONEAREST) == 0, "CPU rounding mode");
    host_launch_abi_tests();
    host_reference_tests();
    for (int tokens = 1; tokens <= 2048; ++tokens) {
        auto params = parameters(tokens);
        Config geometry;
        expect(config(params.data(), int(params.size()), &geometry) == RAD_OK, "M range");
        for (int operand = 0; operand < 9; ++operand) {
            RadOpdDesc description{};
            expect(r3dh_moe_w4nl_shape(params.data(), int(params.size()), operand, &description) == RAD_OK,
                   "shape hook");
        }
    }
    for (int operand : {2, 3, 6, 7}) {
        auto params = parameters(1);
        Weight weight(16, operand >= 6 ? 256 : 384);
        auto enc = weight.encoding();
        auto tensors = weight.planes();
        const bool codes = operand == 2 || operand == 6;
        const int selected[2] = {codes ? 0 : 2, codes ? 1 : 3};
        RadTensor* source = tensors.data() + (codes ? 0 : 2);
        auto blob = weight.encode(params, operand);
        auto original_codes = weight.codes, original_scales = weight.scales;
        auto original_table = weight.table;
        if (codes) { std::fill(weight.codes.begin(), weight.codes.end(), 0); weight.table.fill(0); }
        else std::fill(weight.scales.begin(), weight.scales.end(), 0);
        expect(r3dh_moe_w4nl_inverse(params.data(), int(params.size()), operand, &enc,
            selected, blob.data(), int64_t(blob.size()), source, 2) == RAD_OK, "inverse");
        expect(weight.codes == original_codes && weight.scales == original_scales
            && weight.table == original_table, "bit-preserving roundtrip");
        blob[0] ^= 1;
        expect(r3dh_moe_w4nl_inverse(params.data(), int(params.size()), operand, &enc,
            selected, blob.data(), int64_t(blob.size()), source, 2) == RAD_E_FORMAT, "bad header");
        blob[0] ^= 1;
        RadLayout layout{};
        auto probe = [&] { return r3dh_moe_w4nl_layout(params.data(), int(params.size()), operand,
            &enc, selected, source, 2, &layout); };
        enc.plane[2].block[1] = 128;
        expect(probe() == RAD_E_DTYPE, "reject wrong weight group");
        enc.plane[2].block[1] = 64;
        source[0].stride[1] = 2;
        expect(probe() == RAD_E_SHAPE, "reject plane stride");
        source[0].stride[1] = 1;
        std::strcpy(enc.transform, "none");
        expect(probe() == RAD_E_DTYPE, "reject missing rotation");
        enc = weight.encoding();
        std::strcpy(enc.scheme, "trellis");
        expect(probe() == RAD_E_DTYPE, "reject encoding scheme");
        enc = weight.encoding();
        if (codes) weight.table[3] = std::numeric_limits<float>::infinity();
        else weight.level = std::nextafter(0x1p-13f, 1.0f);
        expect(r3dh_moe_w4nl_transform(params.data(), int(params.size()), operand, &enc,
            selected, source, 2, blob.data(), int64_t(blob.size())) == RAD_E_FORMAT, "reject invalid values");
        weight.level = 0x1p-13f;
        weight.table = original_table;
        if (!codes) {
            for (int raw = 0; raw < 256; ++raw) {
                weight.scales[0] = uint8_t(raw);
                const int result = r3dh_moe_w4nl_transform(params.data(), int(params.size()), operand,
                    &enc, selected, source, 2, blob.data(), int64_t(blob.size()));
                expect(result == (((raw & 127) == 127) ? RAD_E_FORMAT : RAD_OK), "E4M3 validity");
            }
        }
    }
    auto params = parameters(1);
    Config geometry;
    params[0] = RAD_INT("M", 0);
    expect(config(params.data(), int(params.size()), &geometry) == RAD_E_SHAPE, "reject M0");
    params = parameters(2049);
    expect(config(params.data(), int(params.size()), &geometry) == RAD_E_SHAPE, "reject M2049");
    params = parameters(1, 17);
    expect(config(params.data(), int(params.size()), &geometry) == RAD_E_SHAPE, "reject N17");
    params = parameters(1, 16, 320);
    expect(config(params.data(), int(params.size()), &geometry) == RAD_E_SHAPE, "reject K320");
    params = parameters(1);
    params.back() = RAD_STR("dtype", "w4nl64a8h");
    expect(config(params.data(), int(params.size()), &geometry) == RAD_E_UNSUPPORTED, "no dtype alias");
    std::printf("W4NL_HOST_PASS checks=%llu\n", static_cast<unsigned long long>(checks));
}
#include "r3dh_moe_w4nl_check_gpu.h"
#include "r3dh_moe_w4nl_check_model.h"
}
int main(int argc, char** argv) {
    try {
        host_tests();
        if (argc == 2 && !std::strcmp(argv[1], "--host")) return 0;
        if (argc == 4 && !std::strcmp(argv[1], "--write-manifest")) {
            Model model(argv[2]);
            model.write_manifest(argv[3]);
            return 0;
        }
        if (argc == 4 && !std::strcmp(argv[1], "--manifest-check")) {
            Model model(argv[2]);
            model.manifest_tests(argv[3]);
            return 0;
        }
        if (argc == 2 && !std::strcmp(argv[1], "--gpu")) gpu_tests();
        else if (argc == 4 && !std::strcmp(argv[1], "--model")) model_tests(argv[2], argv[3]);
        else throw std::runtime_error("use --host, --gpu, --write-manifest MODEL MANIFEST, --manifest-check MODEL MANIFEST, or --model MODEL MANIFEST");
        std::printf("W4NL_PASS checks=%llu gpu_cases=%llu compared_outputs_and_padding=%llu\n",
            static_cast<unsigned long long>(checks), static_cast<unsigned long long>(gpu_cases),
            static_cast<unsigned long long>(outputs));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "W4NL_FAIL %s checks=%llu\n", error.what(),
                     static_cast<unsigned long long>(checks));
        return 1;
    }
}
