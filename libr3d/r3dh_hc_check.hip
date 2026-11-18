#include "r3dh_hc_api.h"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include "rad_planes.h"
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <map>
#include <regex>
#include <sstream>
#include <memory>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void check_hip(hipError_t result) {
    if (result != hipSuccess) throw std::runtime_error(hipGetErrorString(result));
}
uint16_t narrow(float value) {
    const uint32_t bits = std::bit_cast<uint32_t>(value);
    return static_cast<uint16_t>((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}
float widen(uint16_t value) { return std::bit_cast<float>(uint32_t(value) << 16); }
float rounded(float value) { return widen(narrow(value)); }
float decode(uint8_t code) {
    const int exponent = (code >> 3) & 15, mantissa = code & 7;
    const float magnitude = exponent == 0 ? std::ldexp(float(mantissa), -9)
                                          : std::ldexp(1.0f + float(mantissa) / 8, exponent - 7);
    return code & 128 ? -magnitude : magnitude;
}
float sigmoid(float value) { return 1.0f / (1.0f + float(std::exp(double(-value)))); }
RadTensor tensor(void* pointer, uint32_t dtype, int64_t rows, int64_t columns) {
    RadTensor result{};
    result.data = pointer; result.dtype = dtype; result.rank = 2;
    result.shape[0] = rows; result.shape[1] = columns;
    result.stride[0] = columns; result.stride[1] = 1;
    return result;
}
std::vector<RadParam> parameters(int rows) {
    return {RAD_INT("M", rows), RAD_INT("n", 2560), RAD_INT("hc", 4),
            RAD_INT("lowrank", 320), RAD_STR("dtype", "bf16"),
            RAD_STR("mix", "e4m3"), RAD_F64("eps", 1e-6), RAD_F64("wadd", 0.125),
            RAD_INT("inject", 1), RAD_INT("group", 128), RAD_INT("rotate", 1)};
}
void check_shapes() {
    auto params = parameters(1);
    RadOpdDesc description{};
    require(r3dh_hc_shape_hc_enter(params.data(), int(params.size()), 1, &description) == RAD_OK &&
            description.shape[1] == 10240, "HC enter shape");
    require(r3dh_hc_shape_hc_write(params.data(), int(params.size()), 1, &description) == RAD_OK &&
            description.fill == RAD_FILL_SIGMOID && description.shape[1] == 4, "HC write gain shape");
    for (int operand = 0; operand < 11; ++operand)
        require(r3dh_hc_shape_hc_read(params.data(), int(params.size()), operand, &description) == RAD_OK &&
                !(description.flags & RAD_OPD_F_ABSENT), "HC trunk shape");
    params[8].ival = 0; params[9].ival = 0; params[10].ival = 0;
    for (int operand : {4, 6, 7, 8, 9, 10})
        require(r3dh_hc_shape_hc_read(params.data(), int(params.size()), operand, &description) == RAD_OK &&
                (description.flags & RAD_OPD_F_ABSENT), "HC final mixer optional shape");
    require(r3dh_hc_shape_hc_read(params.data(), int(params.size()), 11, &description) == RAD_E_SHAPE,
            "HC invalid operand shape");
    std::puts("hc_shapes,trunk_and_final_mixer=PASS");
}
struct Matrix {
    int rows, columns, group;
    std::vector<uint8_t> codes;
    std::vector<float> scales;
    std::vector<uint8_t> packed;
    Matrix(int height, int width, int block, int seed)
        : rows(height), columns(width), group(block), codes(size_t(height) * width),
          scales(size_t(height) * (width / block)) {
        const uint8_t values[] = {0, 1, 7, 8, 24, 37, 45, 54, 64, 81, 102, 118};
        for (size_t index = 0; index < codes.size(); ++index) {
            uint32_t hash = uint32_t(index) + uint32_t(seed) * 0x9e3779b9u;
            hash ^= hash >> 16; hash *= 0x85ebca6bu; hash ^= hash >> 13;
            codes[index] = values[hash % 12] | ((hash >> 11) & 128);
        }
        for (size_t index = 0; index < scales.size(); ++index)
            scales[index] = std::ldexp(1.0f + float((index * 13 + seed) % 17) / 32, -11);
    }
    float at(int row, int column) const {
        return decode(codes[size_t(row) * columns + column]) *
               scales[size_t(row) * (columns / group) + column / group];
    }
    void pack(const std::vector<RadParam>& params, int operand) {
        RadEncoding encoding{};
        rad_enc_copy_str(encoding.scheme, "affine"); encoding.n_planes = 2;
        rad_enc_copy_str(encoding.plane[0].role, "codes");
        encoding.plane[0].dtype = RAD_F8E4M3;
        encoding.plane[0].kind = RAD_PLANE_TILED;
        encoding.plane[0].block[0] = encoding.plane[0].block[1] = 1;
        rad_enc_copy_str(encoding.plane[1].role, "scale");
        encoding.plane[1].dtype = RAD_F32; encoding.plane[1].kind = RAD_PLANE_TILED;
        encoding.plane[1].block[0] = 1; encoding.plane[1].block[1] = group;
        RadTensor planes[] = {tensor(codes.data(), RAD_F8E4M3, rows, columns),
                              tensor(scales.data(), RAD_F32, rows, columns / group)};
        const int selection[] = {0, 1};
        RadLayout layout{};
        auto describe = [&](int selected) {
            return r3dh_hc_rad_layout_hc_e4m3(params.data(), int(params.size()), selected,
                                              &encoding, selection, planes, 2, &layout);
        };
        require(describe(operand) == RAD_OK, "HC layout rejected canonical E4M3");
        const int64_t pitch = (columns + 4 * (columns / group) + 15) / 16 * 16;
        require(layout.bytes == rows * pitch && layout.align == 16, "HC row geometry");
        packed.resize(size_t(layout.bytes));
        auto relayout = [&](int64_t bytes) {
            return r3dh_hc_rad_relayout_hc_e4m3(params.data(), int(params.size()), operand,
                &encoding, selection, planes, 2, packed.data(), bytes);
        };
        require(relayout(layout.bytes) == RAD_OK, "HC packing failed");
        for (int row = 0; row < rows; ++row) {
            require(!std::memcmp(packed.data() + row * pitch, codes.data() + row * columns,
                                 columns), "packed HC code bytes");
            require(!std::memcmp(packed.data() + row * pitch + columns,
                scales.data() + row * (columns / group), 4 * (columns / group)),
                "packed HC scale bytes");
            for (int64_t offset = columns + 4 * (columns / group); offset < pitch; ++offset)
                require(packed[row * pitch + offset] == 0, "HC row padding");
        }
        require(relayout(layout.bytes - 1) == RAD_E_SHAPE, "HC wrong byte count accepted");
        require(describe(1) == RAD_E_UNSUPPORTED, "HC non-matrix operand accepted");
        encoding.plane[1].block[1] = group + 1;
        require(describe(operand) == RAD_E_DTYPE, "HC wrong group accepted");
        encoding.plane[1].block[1] = group;
        encoding.plane[1].dtype = RAD_BF16;
        require(describe(operand) == RAD_E_DTYPE, "HC wrong scale dtype accepted");
        encoding.plane[1].dtype = RAD_F32;
        rad_enc_copy_str(encoding.transform, "fwht128");
        require(describe(operand) == RAD_E_DTYPE, "HC unexpected transform accepted");
        rad_enc_copy_str(encoding.transform, "");
        std::vector<uint8_t> restored_codes(codes.size());
        std::vector<float> restored_scales(scales.size());
        planes[0].data = restored_codes.data(); planes[1].data = restored_scales.data();
        require(r3dh_hc_rad_unrelayout_hc_e4m3(params.data(), int(params.size()), operand,
            &encoding, selection, packed.data(), layout.bytes, planes, 2) == RAD_OK,
            "HC unpacking failed");
        require(restored_codes == codes && restored_scales == scales, "HC roundtrip mismatch");
    }
};
struct DeviceBuffer {
    uint8_t* allocation = nullptr;
    size_t bytes;
    explicit DeviceBuffer(size_t size) : bytes(size) {
        check_hip(hipMalloc(reinterpret_cast<void**>(&allocation), bytes + 512));
        check_hip(hipMemset(allocation, 0xa5, bytes + 512));
    }
    ~DeviceBuffer() { if (allocation) static_cast<void>(hipFree(allocation)); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    void* data() const { return allocation + 256; }
    void upload(const void* source) { check_hip(hipMemcpy(data(), source, bytes, hipMemcpyHostToDevice)); }
    void download(void* destination) const {
        check_hip(hipMemcpy(destination, data(), bytes, hipMemcpyDeviceToHost));
    }
    void guards() const {
        uint8_t before[256], after[256];
        check_hip(hipMemcpy(before, allocation, 256, hipMemcpyDeviceToHost));
        check_hip(hipMemcpy(after, allocation + 256 + bytes, 256, hipMemcpyDeviceToHost));
        for (int index = 0; index < 256; ++index)
            require(before[index] == 0xa5 && after[index] == 0xa5, "HC device guard overwritten");
    }
};
struct Stream {
    hipStream_t value = nullptr;
    Stream() { check_hip(hipStreamCreateWithFlags(&value, hipStreamNonBlocking)); }
    ~Stream() { if (value) static_cast<void>(hipStreamDestroy(value)); }
};
uint8_t encode(float value) {
    const float magnitude = std::min(std::abs(value), 448.0f);
    int closest = 0;
    float distance = magnitude;
    for (int code = 1; code <= 126; ++code) {
        const float candidate = std::abs(magnitude - decode(uint8_t(code)));
        if (candidate < distance || (candidate == distance && !(code & 1))) {
            closest = code; distance = candidate;
        }
    }
    return uint8_t(closest | (std::signbit(value) ? 128 : 0));
}
void check_twins(const std::vector<uint16_t>& output, const DeviceBuffer& codes,
                 const DeviceBuffer& scales, bool rotated, bool int8_codes = false) {
    std::vector<uint8_t> actual_codes(output.size());
    std::vector<float> actual_scales(output.size() / 128);
    codes.download(actual_codes.data()); scales.download(actual_scales.data());
    for (size_t group = 0; group < actual_scales.size(); ++group) {
        float values[128];
        for (int index = 0; index < 128; ++index) values[index] = widen(output[group * 128 + index]);
        if (rotated)
            for (int step = 1; step < 128; step *= 2)
                for (int base = 0; base < 128; base += 2 * step)
                    for (int offset = 0; offset < step; ++offset) {
                        const float first = values[base + offset], second = values[base + step + offset];
                        values[base + offset] = first + second;
                        values[base + step + offset] = first - second;
                    }
        float maximum = 0;
        for (float value : values) maximum = std::max(maximum, std::abs(value));
        const float top = int8_codes ? 127.0f : 448.0f;
        const float expected_scale = maximum > 0 ? maximum * (1.0f / top) : 1;
        require(actual_scales[group] == expected_scale, "HC twin scale mismatch");
        const float inverse = 1.0f / expected_scale;
        for (int index = 0; index < 128; ++index) {
            const float value = std::clamp(values[index] * inverse, -top, top);
            const uint8_t expected = int8_codes ? uint8_t(int8_t(std::nearbyint(value))) : encode(value);
            require(actual_codes[group * 128 + index] == expected, "HC twin code mismatch");
        }
    }
    std::printf("hc_%stwin,dtype=%s,elements=%zu,bit_exact=PASS\n", rotated ? "rotated_" : "",
                int8_codes ? "int8" : "e4m3", output.size());
}
#include "r3dh_hc_verify.h"
void compare(const char* label, const std::vector<uint16_t>& actual,
             const std::vector<uint16_t>& expected, int allowance) {
    int largest = 0, changed = 0;
    for (size_t index = 0; index < actual.size(); ++index) {
        require(std::isfinite(widen(actual[index])), "HC nonfinite output");
        const int distance = std::abs(int(actual[index]) - int(expected[index]));
        if (distance > largest) largest = distance;
        if (distance) ++changed;
    }
    std::printf("%s,elements=%zu,different=%d,max_bf16_ulp=%d\n", label, actual.size(), changed, largest);
    if (largest > allowance) ++gate_failures;
}
void oracle(const std::vector<uint16_t>& hidden, const std::vector<float>& gain,
            const Matrix& down, const Matrix& up, const std::vector<uint16_t>& injection,
            int rows, std::vector<uint16_t>& output, std::vector<uint16_t>& gates, float wadd) {
    constexpr int width = 2560, streams = 4, channels = streams * width, rank = 320;
    oracle_points.scales.resize(size_t(rows) * streams);
    oracle_points.normalized.resize(size_t(rows) * channels);
    oracle_points.down_sum.resize(size_t(rows) * rank);
    oracle_points.intermediate.resize(size_t(rows) * rank);
    for (int row = 0; row < rows; ++row) {
        std::vector<float> normalized(channels), intermediate(rank);
        for (int stream = 0; stream < streams; ++stream) {
            double squares = 0;
            for (int index = 0; index < width; ++index) {
                const double value = widen(hidden[size_t(row) * channels + stream * width + index]);
                squares += value * value;
            }
            const float scale = 1.0f / std::sqrt(float(squares / width) + 1e-6f);
            oracle_points.scales[row * streams + stream] = scale;
            for (int index = 0; index < width; ++index) {
                const int channel = stream * width + index;
                normalized[channel] = rounded(widen(hidden[size_t(row) * channels + channel]) *
                                               scale * (gain[channel] + wadd));
                oracle_points.normalized[size_t(row) * channels + channel] = normalized[channel];
            }
        }
        for (int column = 0; column < rank; ++column) {
            float sum = 0;
            for (int channel = 0; channel < channels; ++channel)
                sum += normalized[channel] * down.at(column, channel);
            sum /= streams;
            oracle_points.down_sum[size_t(row) * rank + column] = sum;
            intermediate[column] = rounded(sum * sigmoid(sum));
            oracle_points.intermediate[size_t(row) * rank + column] = intermediate[column];
        }
        for (int index = 0; index < width; ++index) {
            float mixed = 0;
            for (int stream = 0; stream < streams; ++stream) {
                const int channel = stream * width + index;
                float sum = 0;
                for (int column = 0; column < rank; ++column)
                    sum += intermediate[column] * up.at(channel, column);
                mixed += rounded(sigmoid(sum)) * normalized[channel];
            }
            output[size_t(row) * width + index] = narrow(mixed / streams);
        }
        for (int stream = 0; stream < streams; ++stream) {
            float sum = 0;
            for (int channel = 0; channel < channels; ++channel)
                sum += normalized[channel] * widen(injection[stream * channels + channel]);
            gates[row * streams + stream] = narrow(2 * sigmoid(sum / streams));
        }
    }
}
void gpu_check(Matrix& down, Matrix& up, int rows, const std::vector<float>* model_gain = nullptr,
               const std::vector<uint16_t>* model_weights = nullptr, bool inject = true) {
    constexpr int width = 2560, streams = 4, channels = width * streams;
    auto params = parameters(rows);
    const float wadd = model_gain ? 1.0f : 0.125f;
    params[7].dval = wadd;
    params[8].ival = inject ? 1 : 0;
    Stream stream;
    DeviceBuffer hidden(size_t(rows) * channels * 2), input(size_t(rows) * width * 2),
        gain(channels * 4), weights(streams * channels * 2),
        packed_down(down.packed.size()), packed_up(up.packed.size()),
        output(size_t(rows) * width * 2), gates(size_t(rows) * streams * 2),
        quantized(size_t(rows) * width), scales(size_t(rows) * (width / 128) * 4),
        rotated(size_t(rows) * width), rotated_scales(size_t(rows) * (width / 128) * 4),
        traced_down(size_t(rows) * 320 * 2), traced_logits(size_t(rows) * channels * 4),
        traced_gates(size_t(rows) * channels * 2);
    std::vector<float> host_gain(channels);
    std::vector<uint16_t> host_weights(streams * channels), host_input(size_t(rows) * width),
        host_hidden(size_t(rows) * channels), actual(size_t(rows) * width),
        expected(actual.size()), actual_gates(rows * streams), expected_gates(rows * streams);
    for (int index = 0; index < channels; ++index) host_gain[index] = 0.75f + float(index % 17) / 32;
    for (size_t index = 0; index < host_weights.size(); ++index)
        host_weights[index] = narrow(float(int(index * 19 % 101) - 50) / 4096);
    if (model_gain) host_gain = *model_gain;
    if (model_weights) host_weights = *model_weights;
    gain.upload(host_gain.data()); weights.upload(host_weights.data());
    packed_down.upload(down.packed.data()); packed_up.upload(up.packed.data());
    RadTensor operands[] = {tensor(hidden.data(), RAD_BF16, rows, channels),
        tensor(gain.data(), RAD_F32, 1, channels),
        tensor(packed_down.data(), RAD_F8E4M3, 320, channels),
        tensor(packed_up.data(), RAD_F8E4M3, channels, 320),
        tensor(weights.data(), RAD_BF16, streams, channels),
        tensor(output.data(), RAD_BF16, rows, width), tensor(gates.data(), RAD_BF16, rows, streams),
        tensor(quantized.data(), RAD_F8E4M3, rows, width),
        tensor(scales.data(), RAD_F32, rows, width / 128),
        tensor(rotated.data(), RAD_F8E4M3, rows, width),
        tensor(rotated_scales.data(), RAD_F32, rows, width / 128)};
    if (!inject) { operands[4] = {}; operands[6] = {}; }
    RadArgs args{}; args.t = operands; args.n_t = 11;
    args.p = params.data(); args.n_p = int(params.size());
    const int64_t scratch_bytes = r3dh_hc_rad_hc_read_scratch(&args);
    require(scratch_bytes > 0, "HC scratch size invalid");
    DeviceBuffer scratch{size_t(scratch_bytes)};
    args.scratch = scratch.data(); args.scratch_bytes = scratch_bytes;
    args.scratch_bytes = 0;
    require(r3dh_hc_rad_hc_read(&args, nullptr) == RAD_E_SCRATCH, "HC short scratch accepted");
    args.scratch_bytes = scratch_bytes;
    operands[0].dtype = RAD_F32;
    require(r3dh_hc_rad_hc_read(&args, nullptr) == RAD_E_DTYPE, "HC wrong activation dtype accepted");
    operands[0].dtype = RAD_BF16;
    require(r3dh_hc_rad_hc_read_strict_scratch(&args) <= scratch_bytes, "strict scratch exceeds existing ABI");
    params[1].ival = 1280;
    require(r3dh_hc_rad_hc_read(&args, nullptr) == RAD_E_UNSUPPORTED, "strict accepted unsupported width");
    params[1].ival = width;
    params[9].ival = 64;
    require(r3dh_hc_rad_hc_read(&args, nullptr) == RAD_E_UNSUPPORTED, "strict accepted unsupported group");
    params[9].ival = 128;
    operands[8].data = nullptr;
    require(r3dh_hc_rad_hc_read(&args, nullptr) == RAD_E_INVAL, "strict accepted unpaired quant output");
    operands[8].data = scales.data();
    operands[0].stride[1] = 2;
    require(r3dh_hc_rad_hc_read(&args, nullptr) == RAD_E_DTYPE, "strict accepted noncontiguous input");
    operands[0].stride[1] = 1;
    const uint32_t saved_rank = operands[0].rank;
    operands[0].rank = RAD_MAX_RANK + 1;
    require(r3dh_hc_rad_hc_read(&args, nullptr) == RAD_E_DTYPE, "strict accepted invalid rank");
    operands[0].rank = saved_rank;
    const int64_t saved_rows = operands[0].shape[0];
    operands[0].shape[0] = INT64_MAX;
    require(r3dh_hc_rad_hc_read(&args, nullptr) == RAD_E_DTYPE, "strict accepted overflowing shape");
    operands[0].shape[0] = saved_rows;
    args.scratch_bytes = r3dh_hc_rad_hc_read_strict_scratch(&args) - 1;
    require(r3dh_hc_rad_hc_read(&args, nullptr) == RAD_E_SCRATCH, "strict accepted short exact scratch");
    args.scratch_bytes = scratch_bytes;
    for (int repeat = 0; repeat < 2; ++repeat) {
        for (size_t index = 0; index < host_input.size(); ++index)
            host_input[index] = narrow(float(int((index * 37 + repeat * 29) % 257) - 128) / 64);
        input.upload(host_input.data());
        RadTensor enter_operands[] = {tensor(input.data(), RAD_BF16, rows, width), operands[0]};
        RadArgs enter = args; enter.t = enter_operands; enter.n_t = 2;
        require(r3dh_hc_rad_hc_enter(&enter, reinterpret_cast<RadStream>(stream.value)) == RAD_OK,
                "HC enter launch");
        check_hip(hipStreamSynchronize(stream.value));
        hidden.download(host_hidden.data());
        for (int row = 0; row < rows; ++row)
            for (int channel = 0; channel < channels; ++channel)
                require(host_hidden[size_t(row) * channels + channel] ==
                        host_input[size_t(row) * width + channel % width], "HC enter copy mismatch");
        for (size_t index = 0; index < host_hidden.size(); ++index)
            host_hidden[index] = narrow(widen(host_hidden[index]) + float(int(index % 13) - 6) / 32);
        hidden.upload(host_hidden.data());
        std::printf("hc_case,fixture=%s,M=%d,repeat=%d,inject=%d,wadd=%g\n", fixture_name, rows, repeat, inject, wadd);
        oracle(host_hidden, host_gain, down, up, host_weights, rows, expected, expected_gates, wadd);
        std::vector<uint16_t> ref_output(expected.size()), ref_gates(expected_gates.size());
        libref_check(down, up, host_hidden, host_gain, host_weights, rows, inject, params, ref_output, ref_gates);
        compare("cpu_vs_existing_libref", expected, ref_output, 0);
        if (inject) compare("cpu_inj_vs_existing_libref", expected_gates, ref_gates, 0);
        require(r3dh_hc_rad_hc_read(&args, reinterpret_cast<RadStream>(stream.value)) == RAD_OK,
                "HC read launch");
        check_hip(hipStreamSynchronize(stream.value));
        output.download(actual.data()); gates.download(actual_gates.data());
        R3dhHcTrace trace{static_cast<uint16_t*>(traced_down.data()), static_cast<float*>(traced_logits.data()),
            static_cast<uint16_t*>(traced_gates.data()), rows, width, streams, 320};
        require(r3dh_hc_rad_hc_read_trace(&args, reinterpret_cast<RadStream>(stream.value), &trace) == RAD_OK,
                "HC diagnostic trace launch");
        check_hip(hipStreamSynchronize(stream.value));
        std::vector<uint16_t> traced_output(actual.size()); output.download(traced_output.data());
        compare("hc_trace_vs_normal", traced_output, actual, 0);
        std::printf("M=%d,repeat=%d,", rows, repeat); compare("hc_read", actual, expected, 0);
        if (inject) compare("hc_inj", actual_gates, ref_gates, 0);
        strict_checkpoint_check(scratch, rows);
        trace_check(traced_down, traced_logits, traced_gates, up, actual, expected, rows);
        check_twins(actual, quantized, scales, false);
        check_twins(actual, rotated, rotated_scales, true);
        libref_twins_check(quantized, scales, rotated, rotated_scales);
        operands[7].dtype = RAD_I8; operands[9].dtype = RAD_I8;
        require(r3dh_hc_rad_hc_read_strict(&args, reinterpret_cast<RadStream>(stream.value)) == RAD_OK,
                "HC int8 paired outputs launch");
        check_hip(hipStreamSynchronize(stream.value));
        std::vector<uint16_t> int8_output(actual.size()); output.download(int8_output.data());
        compare("hc_int8_bf16", int8_output, ref_output, 0);
        check_twins(int8_output, quantized, scales, false, true);
        check_twins(int8_output, rotated, rotated_scales, true, true);
        libref_check(down, up, host_hidden, host_gain, host_weights, rows, inject, params,
                     ref_output, ref_gates, RAD_I8);
        libref_twins_check(quantized, scales, rotated, rotated_scales);
        const RadTensor ordinary_codes = operands[7], ordinary_scales = operands[8];
        operands[7] = {}; operands[8] = {};
        require(r3dh_hc_rad_hc_read_strict(&args, reinterpret_cast<RadStream>(stream.value)) == RAD_OK,
                "HC rotated-only int8 launch");
        check_hip(hipStreamSynchronize(stream.value));
        check_twins(int8_output, rotated, rotated_scales, true, true);
        operands[9].dtype = RAD_F8E4M3;
        require(r3dh_hc_rad_hc_read_strict(&args, reinterpret_cast<RadStream>(stream.value)) == RAD_OK,
                "HC rotated-only E4M3 launch");
        check_hip(hipStreamSynchronize(stream.value));
        check_twins(int8_output, rotated, rotated_scales, true);
        operands[7] = ordinary_codes; operands[8] = ordinary_scales;
        operands[7].dtype = RAD_F8E4M3;
        RadTensor final_operands[11];
        std::copy(operands, operands + 11, final_operands);
        for (int operand : {4, 6, 7, 8, 9, 10}) final_operands[operand] = {};
        auto final_params = params;
        final_params[8].ival = 0; final_params[9].ival = 0; final_params[10].ival = 0;
        RadArgs final_args = args;
        final_args.t = final_operands; final_args.p = final_params.data();
        require(r3dh_hc_rad_hc_read(&final_args, reinterpret_cast<RadStream>(stream.value)) == RAD_OK,
                "HC final mixer launch");
        check_hip(hipStreamSynchronize(stream.value));
        std::vector<uint16_t> final_output(actual.size()); output.download(final_output.data());
        compare("hc_final_mixer", final_output, actual, 0);
        if (inject) {
        RadTensor write_operands[] = {operands[5], operands[6], operands[0]};
        RadArgs write = args; write.t = write_operands; write.n_t = 3;
        for (int row = 0; row < rows; ++row)
            for (int channel = 0; channel < channels; ++channel) {
                const size_t index = size_t(row) * channels + channel;
                host_hidden[index] = narrow(std::fma(
                    widen(actual_gates[row * streams + channel / width]),
                    widen(actual[size_t(row) * width + channel % width]), widen(host_hidden[index])));
            }
        require(r3dh_hc_rad_hc_write(&write, reinterpret_cast<RadStream>(stream.value)) == RAD_OK,
                "HC write launch");
        check_hip(hipStreamSynchronize(stream.value));
        std::vector<uint16_t> actual_hidden(host_hidden.size()); hidden.download(actual_hidden.data());
        compare("hc_write", actual_hidden, host_hidden, 0);
        }
        for (const DeviceBuffer* buffer : {&hidden, &input, &gain, &weights, &packed_down,
             &packed_up, &output, &gates, &quantized, &scales, &rotated, &rotated_scales, &scratch})
            buffer->guards();
        traced_down.guards(); traced_logits.guards(); traced_gates.guards();
    }
}
void bench(Matrix& down, Matrix& up, int rows) {
    constexpr int width = 2560, streams = 4, channels = width * streams, launches = 31, warmup = 5;
    auto params = parameters(rows);
    Stream stream;
    DeviceBuffer hidden(size_t(rows) * channels * 2), gain(channels * 4), weights(streams * channels * 2),
        packed_down(down.packed.size()), packed_up(up.packed.size()), output(size_t(rows) * width * 2),
        gates(size_t(rows) * streams * 2), quantized(size_t(rows) * width),
        scales(size_t(rows) * (width / 128) * 4), rotated(size_t(rows) * width),
        rotated_scales(size_t(rows) * (width / 128) * 4);
    std::vector<float> host_gain(channels);
    std::vector<uint16_t> host_weights(streams * channels), host_hidden(size_t(rows) * channels);
    for (int index = 0; index < channels; ++index) host_gain[index] = 0.75f + float(index % 17) / 32;
    for (size_t index = 0; index < host_weights.size(); ++index)
        host_weights[index] = narrow(float(int(index * 19 % 101) - 50) / 4096);
    for (size_t index = 0; index < host_hidden.size(); ++index)
        host_hidden[index] = narrow(float(int((index * 37) % 257) - 128) / 64);
    hidden.upload(host_hidden.data()); gain.upload(host_gain.data()); weights.upload(host_weights.data());
    packed_down.upload(down.packed.data()); packed_up.upload(up.packed.data());
    RadTensor operands[] = {tensor(hidden.data(), RAD_BF16, rows, channels),
        tensor(gain.data(), RAD_F32, 1, channels), tensor(packed_down.data(), RAD_F8E4M3, 320, channels),
        tensor(packed_up.data(), RAD_F8E4M3, channels, 320), tensor(weights.data(), RAD_BF16, streams, channels),
        tensor(output.data(), RAD_BF16, rows, width), tensor(gates.data(), RAD_BF16, rows, streams),
        tensor(quantized.data(), RAD_F8E4M3, rows, width), tensor(scales.data(), RAD_F32, rows, width / 128),
        tensor(rotated.data(), RAD_F8E4M3, rows, width), tensor(rotated_scales.data(), RAD_F32, rows, width / 128)};
    RadArgs args{}; args.t = operands; args.n_t = 11;
    args.p = params.data(); args.n_p = int(params.size());
    const int64_t scratch_bytes = r3dh_hc_rad_hc_read_scratch(&args);
    require(scratch_bytes > 0, "HC bench scratch size invalid");
    DeviceBuffer scratch{size_t(scratch_bytes)};
    args.scratch = scratch.data(); args.scratch_bytes = scratch_bytes;
    const auto device_stream = reinterpret_cast<RadStream>(stream.value);
    for (int index = 0; index < warmup; ++index) require(r3dh_hc_rad_hc_read(&args, device_stream) == RAD_OK, "HC bench warmup");
    check_hip(hipStreamSynchronize(stream.value));
    std::vector<uint16_t> first(size_t(rows) * width), last(first.size());
    output.download(first.data());
    hipEvent_t start, stop;
    check_hip(hipEventCreate(&start)); check_hip(hipEventCreate(&stop));
    std::vector<float> samples;
    for (int index = 0; index < launches; ++index) {
        check_hip(hipEventRecord(start, stream.value));
        require(r3dh_hc_rad_hc_read(&args, device_stream) == RAD_OK, "HC bench launch");
        check_hip(hipEventRecord(stop, stream.value));
        check_hip(hipEventSynchronize(stop));
        float milliseconds = 0; check_hip(hipEventElapsedTime(&milliseconds, start, stop));
        samples.push_back(milliseconds * 1000.0f);
    }
    check_hip(hipEventDestroy(start)); check_hip(hipEventDestroy(stop));
    output.download(last.data());
    require(first == last, "HC bench output changed across launches");
    for (const DeviceBuffer* buffer : {&hidden, &gain, &weights, &packed_down, &packed_up, &output, &gates,
         &quantized, &scales, &rotated, &rotated_scales, &scratch}) buffer->guards();
    std::sort(samples.begin(), samples.end());
    std::printf("hc_bench,M=%d,inject=1,quant_pairs=2,warmup=%d,launches=%d,median_us=%.1f,min_us=%.1f,max_us=%.1f\n",
                rows, warmup, launches, samples[launches / 2], samples.front(), samples.back());
}
}

int main(int argc, char** argv) {
    try {
        require(argc == 2 || argc == 3 || argc == 4 || argc == 5,
                "usage: r3dh_hc_check --host-only | --host-model MODEL MANIFEST | --suite LIBREF [MODEL MANIFEST]");
        check_shapes();
        Matrix down(320, 10240, 128, 3), up(10240, 320, 80, 7);
        const auto params = parameters(1);
        down.pack(params, 2); up.pack(params, 3);
        std::puts("hc_layout,model_shape,roundtrip_and_rejects=PASS");
        if (argc == 2 && !std::strcmp(argv[1], "--host-only")) return 0;
        if (argc == 4 && !std::strcmp(argv[1], "--host-model")) {
            ModelManifest model(argv[2], argv[3]);
            for (const char* prefix : {"blk.0.attn_hc", "blk.0.ffn_hc", "output_hc"}) {
                const std::string base(prefix);
                model.matrix(base + "_down.weight", down);
                model.matrix(base + "_up.weight", up);
                down.pack(params, 2); up.pack(params, 3);
                model.bf16(base + "_norm.weight", "10240", 10240);
                if (base != "output_hc") model.bf16(base + "_inject.weight", "4x10240", 40960);
            }
            std::puts("hc_model_host,manifest_planes_group80_group128_roundtrip=PASS");
            return 0;
        }
        if (argc == 2 && !std::strcmp(argv[1], "--bench")) {
            hipDeviceProp_t properties{}; check_hip(hipGetDeviceProperties(&properties, 0));
            require(!std::strncmp(properties.gcnArchName, "gfx1100", 7), "bench requires gfx1100");
            std::printf("device=%s,arch=%s\n", properties.name, properties.gcnArchName);
            for (int rows : {4, 585}) bench(down, up, rows);
            return 0;
        }
        require((argc == 3 || argc == 5) && !std::strcmp(argv[1], "--suite"), "HC suite requires existing libref");
        void* library = dlopen(argv[2], RTLD_NOW | RTLD_LOCAL);
        if (!library) throw std::runtime_error(dlerror());
        reference_read = reinterpret_cast<ReferenceRead>(dlsym(library, "ref_hc_read"));
        require(reference_read, "libref hc_read symbol missing");
        std::printf("hc_reference,library=%s,read_ulp_tolerance=0,inject_ulp_tolerance=0\n", argv[2]);
        hipDeviceProp_t properties{}; check_hip(hipGetDeviceProperties(&properties, 0));
        require(!std::strncmp(properties.gcnArchName, "gfx1100", 7), "checker requires gfx1100");
        std::printf("device=%s,arch=%s\n", properties.name, properties.gcnArchName);
        for (int rows : {1, 3, 4, 5, 8, 12, 16, 24, 32, 33, 64, 65, 585}) gpu_check(down, up, rows);
        if (argc == 5) {
            ModelManifest model(argv[3], argv[4]);
            for (const char* prefix : {"blk.0.attn_hc", "blk.0.ffn_hc", "output_hc"}) {
                const std::string base(prefix);
                fixture_name = prefix;
                model.matrix(base + "_down.weight", down);
                model.matrix(base + "_up.weight", up);
                down.pack(params, 2); up.pack(params, 3);
                const auto bf_gain = model.bf16(base + "_norm.weight", "10240", 10240);
                std::vector<float> model_gain(10240);
                for (int channel = 0; channel < 10240; ++channel) model_gain[channel] = widen(bf_gain[channel]);
                const bool inject = base != "output_hc";
                std::vector<uint16_t> model_weights(40960);
                if (inject) model_weights = model.bf16(base + "_inject.weight", "4x10240", 40960);
                for (int rows : {1, 3, 4, 8, 16, 32, 64, 585})
                    gpu_check(down, up, rows, &model_gain, &model_weights, inject);
            }
        }
        std::printf("hc_gpu,guards_nondefault_stream_changed_input=PASS,bitexact_gate=%s,failing_comparisons=%d\n",
                    gate_failures ? "FAIL_NOT_CERTIFIED" : "PASS", gate_failures);
        if (gate_failures) return 1;
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "hc_check: %s\n", error.what()); return 1;
    }
}
