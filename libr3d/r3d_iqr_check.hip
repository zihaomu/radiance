#include <hip/hip_runtime.h>
#include "r3d_iqr.h"
#include "rad_plugin.h"

#include <array>
#include <bit>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <fcntl.h>
#include <unistd.h>

extern "C" int r3d_rad_gemm_iqr_scalar(const RadArgs*, RadStream);

namespace {

void hip_check(hipError_t status) {
    if (status != hipSuccess) throw std::runtime_error(hipGetErrorString(status));
}

template<class Value> struct Device {
    Value* pointer = nullptr;
    size_t count;
    explicit Device(size_t elements) : count(elements) {
        hip_check(hipMalloc(reinterpret_cast<void**>(&pointer), count * sizeof(Value)));
    }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    ~Device() { if (pointer) hipFree(pointer); }
    void upload(const std::vector<Value>& source) {
        if (source.size() != count) throw std::runtime_error("upload size");
        hip_check(hipMemcpy(pointer, source.data(), count * sizeof(Value), hipMemcpyHostToDevice));
    }
};

RadTensor tensor(void* data, uint32_t dtype, int64_t rows, int64_t columns) {
    RadTensor result{};
    result.data = data; result.dtype = dtype; result.rank = 2;
    result.shape[0] = rows; result.shape[1] = columns;
    rad_tensor_pack(&result);
    return result;
}

RadParam integer(const char* key, int64_t value) {
    RadParam result{};
    result.key = key; result.kind = RAD_P_INT; result.ival = value;
    return result;
}

uint16_t bf16(float value) {
    const uint32_t word = std::bit_cast<uint32_t>(value);
    return static_cast<uint16_t>((word + 32767u + ((word >> 16) & 1u)) >> 16);
}

int independent_value(uint32_t state, uint32_t multiplier, uint32_t increment) {
    const uint32_t word = static_cast<uint32_t>(uint64_t(state % 16777216u)
                         * (multiplier % 16777216u) + increment);
    int total = -126;
    for (unsigned byte = 0; byte < 4; ++byte) total += (word >> (8 * byte)) & 63u;
    return total;
}

void read_exact(int descriptor, void* output, size_t bytes, uint64_t offset) {
    auto* destination = static_cast<uint8_t*>(output);
    while (bytes) {
        const ssize_t count = pread(descriptor, destination, bytes, static_cast<off_t>(offset));
        if (count <= 0) throw std::runtime_error("model pread failed");
        destination += count; bytes -= count; offset += count;
    }
}

bool run_case(int rows, int columns, int reduction, unsigned bits, unsigned run,
              const char* model, int band_rows = 0, bool logical_scale = false) {
    r3d_iqr::Header header{r3d_iqr::weight_magic, bits, run, 16,
        0xcd123457u, 0xfedcba98u, static_cast<uint32_t>(columns), static_cast<uint32_t>(reduction)};
    const size_t code_stride = size_t(reduction) * bits / 8;
    std::vector<uint8_t> codes(size_t(columns) * code_stride);
    std::vector<float> scales(columns + reduction / 128);
    uint32_t random_state = 20261117u;
    auto random = [&] { random_state = random_state * 1664525u + 1013904223u; return random_state; };
    for (auto& value : codes) value = static_cast<uint8_t>(random() >> 16);
    for (int column = 0; column < columns; ++column)
        scales[column] = static_cast<float>(1 + random() % 127) / 8192.0f;
    for (int group = 0; group < reduction / 128; ++group)
        scales[columns + group] = static_cast<float>(1 + random() % 127) / 128.0f;
    if (model) {
        if (bits != 5 || run != 256 || reduction != 2560 || columns > 16384)
            throw std::runtime_error("real-weight case must match verified blk.0.ssm_inz planes");
        const int descriptor = open(model, O_RDONLY);
        if (descriptor < 0) throw std::runtime_error("model open failed");
        read_exact(descriptor, codes.data(), codes.size(), 53946896384ull);
        read_exact(descriptor, scales.data(), columns * sizeof(float), 53973110784ull);
        read_exact(descriptor, scales.data() + columns, reduction / 128 * sizeof(float), 53973242112ull);
        uint32_t coefficients[2];
        read_exact(descriptor, coefficients, sizeof(coefficients), 53973241856ull);
        header.multiplier = coefficients[0]; header.increment = coefficients[1];
        close(descriptor);
    }
    std::vector<uint8_t> weight(sizeof(header) + codes.size());
    std::memcpy(weight.data(), &header, sizeof(header));
    std::memcpy(weight.data() + sizeof(header), codes.data(), codes.size());
    std::vector<int8_t> decoded(size_t(columns) * reduction);
    for (int start = 0; start < columns; start += run) {
        for (int position = 0; position < reduction; ++position) {
            uint32_t state = 0;
            auto code = [&](int column) {
                const unsigned bit = position * bits;
                uint32_t result = 0;
                for (unsigned digit = 0; digit < bits; ++digit) {
                    const unsigned index = bit + digit;
                    result |= ((codes[size_t(column) * code_stride + index / 8] >> (index % 8)) & 1u) << digit;
                }
                return result;
            };
            for (int column = start + run - (16 + bits - 1) / bits; column < start + int(run); ++column)
                state = ((state << bits) | code(column)) & 65535u;
            for (int column = start; column < start + int(run); ++column) {
                state = ((state << bits) | code(column)) & 65535u;
                decoded[size_t(column) * reduction + position] = static_cast<int8_t>(
                    independent_value(state, header.multiplier, header.increment));
            }
        }
    }
    std::vector<int8_t> activation(size_t(rows) * reduction);
    std::vector<float> activation_scale(size_t(rows) * (reduction / 128));
    for (auto& value : activation) value = static_cast<int8_t>(int(random() % 255) - 127);
    for (auto& value : activation_scale) value = static_cast<float>(1 + random() % 127) / 1024.0f;
    std::vector<uint16_t> reference(size_t(rows) * columns);
    for (int row = 0; row < rows; ++row)
        for (int column = 0; column < columns; ++column) {
            float accumulator = 0.0f;
            for (int group = 0; group < reduction / 128; ++group) {
                int subtotal = 0;
                for (int offset = 0; offset < 128; ++offset) {
                    const int position = group * 128 + offset;
                    subtotal += int(activation[size_t(row) * reduction + position])
                              * int(decoded[size_t(column) * reduction + position]);
                }
                accumulator += (static_cast<float>(subtotal) * activation_scale[row * (reduction / 128) + group])
                             * scales[columns + group];
            }
            reference[size_t(row) * columns + column] = bf16(accumulator * scales[column]);
        }
    Device<int8_t> device_activation(activation.size()); device_activation.upload(activation);
    Device<float> device_activation_scale(activation_scale.size()); device_activation_scale.upload(activation_scale);
    Device<uint8_t> device_weight(weight.size()); device_weight.upload(weight);
    Device<float> device_scale(scales.size()); device_scale.upload(scales);
    Device<uint16_t> device_output(reference.size() + 256);
    std::vector<uint16_t> initial(reference.size() + 256, 0x5a5au); device_output.upload(initial);
    std::array<RadTensor, 5> operands{{
        tensor(device_activation.pointer, RAD_I8, rows, reduction),
        tensor(device_activation_scale.pointer, RAD_F32, rows, reduction / 128),
        tensor(device_weight.pointer, r3d_iqr::weight_dtype, columns, reduction),
        tensor(device_scale.pointer, r3d_iqr::scale_dtype, columns,
               logical_scale ? reduction : reduction / 128),
        tensor(device_output.pointer, RAD_BF16, rows, columns)
    }};
    const int implementation = std::getenv("R3D_IQR_IMPL") ? std::atoi(std::getenv("R3D_IQR_IMPL")) : 0;
    std::array<RadParam, 5> params{{integer("M", band_rows ? band_rows : rows), integer("N", columns),
                                  integer("K", reduction), integer("group", 128), integer("impl", implementation)}};
    RadArgs arguments{}; arguments.t = operands.data(); arguments.n_t = operands.size();
    arguments.p = params.data(); arguments.n_p = params.size();
    if (r3d_rad_gemm_iqr_scalar(&arguments, nullptr) != RAD_OK) throw std::runtime_error("launch refused");
    hip_check(hipDeviceSynchronize());
    std::vector<uint16_t> output(initial.size());
    hip_check(hipMemcpy(output.data(), device_output.pointer, output.size() * sizeof(uint16_t), hipMemcpyDeviceToHost));
    size_t mismatches = 0;
    for (size_t index = 0; index < reference.size(); ++index) mismatches += reference[index] != output[index];
    for (size_t index = reference.size(); index < output.size(); ++index)
        if (output[index] != 0x5a5au) throw std::runtime_error("output guard corrupted");
    hipEvent_t begin, end; hip_check(hipEventCreate(&begin)); hip_check(hipEventCreate(&end));
    hip_check(hipEventRecord(begin));
    for (int iteration = 0; iteration < 10; ++iteration)
        if (r3d_rad_gemm_iqr_scalar(&arguments, nullptr) != RAD_OK) throw std::runtime_error("timed launch refused");
    hip_check(hipEventRecord(end)); hip_check(hipEventSynchronize(end));
    float elapsed = 0; hip_check(hipEventElapsedTime(&elapsed, begin, end));
    hipEventDestroy(begin); hipEventDestroy(end);
    std::printf("%s,%u,%u,%d,%d,%d,%zu,%.3f\n", model ? "real_row_slice" : "synthetic",
                bits, run, rows, columns, reduction, mismatches, elapsed * 100.0f);
    return mismatches == 0;
}

}

int main(int argc, char** argv) {
    try {
        std::puts("source,bits,run,M,N,K,bf16_mismatches,preliminary_hot_us");
        bool passed = true;
        if (argc == 2 && !std::strcmp(argv[1], "--ragged")) {
            for (unsigned bits : {3u, 4u, 5u})
                for (int rows : {1, 3, 17, 33, 63, 65}) {
                    const int band = rows <= 16 ? 16 : rows <= 64 ? 64 : 128;
                    for (bool logical_scale : {false, true})
                        passed &= run_case(rows, 256, 256, bits, bits == 4 ? 128 : 256,
                                           nullptr, band, logical_scale);
                }
            return passed ? 0 : 1;
        }
        for (unsigned bits : {3u, 4u, 5u})
            for (int rows : {1, 2, 4, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 256, 512, 1024, 2048})
                passed &= run_case(rows, 256, 256, bits, bits == 4 ? 128 : 256, nullptr);
        if (argc == 2)
            for (int rows : {1, 4, 16, 32}) passed &= run_case(rows, 8192, 2560, 5, 256, argv[1]);
        return passed ? 0 : 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "r3d_iqr_check: %s\n", error.what()); return 2;
    }
}
