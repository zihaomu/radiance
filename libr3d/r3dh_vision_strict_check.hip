#include <hip/hip_runtime.h>
#include "r3dh_vision_strict.h"
#include "rad_format.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <dlfcn.h>
namespace {
uint64_t checks = 0, comparisons = 0, cases = 0;
using Reference = int (*)(const RadArgs*, RadStream);
Reference reference_gemm = nullptr, reference_bias = nullptr, reference_attention = nullptr;
uint64_t reference_comparisons = 0;
void expect(bool condition, const char* text) {
    ++checks; if (!condition) throw std::runtime_error(text);
}
void hip_check(hipError_t status) { if (status != hipSuccess) throw std::runtime_error(hipGetErrorString(status)); }
float cpu_float(uint16_t bits) {
    uint32_t word = uint32_t(bits) << 16; float value; std::memcpy(&value, &word, 4); return value;
}
uint16_t cpu_bf16(float value) {
    uint32_t word; std::memcpy(&word, &value, 4);
    if ((word & 0x7fffffffu) > 0x7f800000u) return uint16_t((word >> 16) | 64u);
    uint32_t upper = word >> 16, lower = word & 65535u;
    if (lower > 32768 || (lower == 32768 && (upper & 1))) ++upper;
    return uint16_t(upper);
}
float cpu_mac(float sum, float left, float right) { volatile float multiply = left * right; return sum + multiply; }
uint16_t cpu_epilogue(float sum, int activation, uint16_t residual, bool has_residual) {
    uint16_t rounded = cpu_bf16(sum);
    const float value = cpu_float(rounded);
    if (activation == 1) rounded = cpu_bf16(0.5f * value * (1.0f + float(std::erf(double(value * 0.70710678118654752f)))));
    if (activation == 2) rounded = cpu_bf16(0.5f * value * (1.0f + float(std::tanh(double(0.7978845608028654f *
        (value + 0.044715f * value * value * value))))));
    return has_residual ? cpu_bf16(cpu_float(residual) + cpu_float(rounded)) : rounded;
}
RadParam number(const char* key, int64_t value) { RadParam result{}; result.key = key; result.kind = RAD_P_INT; result.ival = value; return result; }
RadParam text(const char* key, const char* value) { RadParam result{}; result.key = key; result.kind = RAD_P_STR; result.sval = value; return result; }
RadTensor tensor(void* memory, int64_t rows, int64_t columns, int64_t pitch = 0) {
    RadTensor result{}; result.data = memory; result.dtype = RAD_BF16; result.rank = 2;
    result.shape[0] = rows; result.shape[1] = columns; result.stride[0] = pitch ? pitch : columns; result.stride[1] = 1; return result;
}
RadTensor vector_tensor(void* memory, int64_t size, uint32_t dtype = RAD_BF16) {
    RadTensor result{}; result.data = memory; result.dtype = dtype; result.rank = 1; result.shape[0] = size; result.stride[0] = 1; return result;
}
RadArgs arguments(const std::vector<RadTensor>& tensors, const std::vector<RadParam>& params) {
    RadArgs result{}; result.t = tensors.data(); result.n_t = int(tensors.size()); result.p = params.data(); result.n_p = int(params.size()); return result;
}
void fill(std::vector<uint16_t>& values, uint32_t seed) {
    for (auto& value : values) { seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5; value = cpu_bf16(float(int(seed % 513) - 256) / 512.0f); }
}
struct Buffer {
    void* allocation = nullptr; uint8_t* data = nullptr; size_t bytes;
    explicit Buffer(size_t count) : bytes(count) {
        hip_check(hipMalloc(&allocation, bytes + 256)); data = static_cast<uint8_t*>(allocation) + 128;
    }
    ~Buffer() { if (allocation) static_cast<void>(hipFree(allocation)); }
    void reset(hipStream_t stream) { hip_check(hipMemsetAsync(allocation, 0xa5, bytes + 256, stream)); }
    void upload(const void* source, hipStream_t stream) { reset(stream); hip_check(hipMemcpyAsync(data, source, bytes, hipMemcpyHostToDevice, stream)); }
    void guards(hipStream_t stream) {
        uint8_t before[128], after[128];
        hip_check(hipMemcpyAsync(before, allocation, 128, hipMemcpyDeviceToHost, stream));
        hip_check(hipMemcpyAsync(after, data + bytes, 128, hipMemcpyDeviceToHost, stream)); hip_check(hipStreamSynchronize(stream));
        for (size_t index = 0; index < 128; ++index) expect(before[index] == 0xa5 && after[index] == 0xa5, "device guard changed");
    }
};
struct Model {
    std::ifstream input; uint64_t size; RadFileHeader header{};
    std::vector<char> strings; std::vector<RadFileEntry> entries; std::vector<RadFilePlane> planes; std::vector<RadEncoding> encodings;
    explicit Model(const char* path) : input(path, std::ios::binary) {
        expect(bool(input), "open current model"); input.seekg(0, std::ios::end); size = uint64_t(input.tellg()); read(0, &header, sizeof(header));
        expect(header.magic == RAD_MAGIC && header.version == RAD_FORMAT_VER && header.file_bytes == size && size == 121678372864ull, "target model format/size");
        strings = section<char>(header.str_off, header.str_bytes); entries = section<RadFileEntry>(header.dir_off, header.dir_count);
        planes = section<RadFilePlane>(header.plane_off, header.plane_count); encodings = section<RadEncoding>(header.enc_off, header.enc_count);
    }
    void read(uint64_t offset, void* target, size_t count) {
        expect(offset <= size && count <= size - offset, "bounded model read"); input.clear(); input.seekg(std::streamoff(offset));
        input.read(static_cast<char*>(target), std::streamsize(count)); expect(bool(input), "model read");
    }
    template<class Element> std::vector<Element> section(uint64_t offset, uint64_t count) {
        expect(count < 16000000 && count <= size / sizeof(Element), "bounded model metadata");
        std::vector<Element> result(static_cast<size_t>(count)); if (count) read(offset, result.data(), result.size() * sizeof(Element)); return result;
    }
    std::vector<uint16_t> weight(const char* name, int64_t rows, int64_t columns = 0) {
        for (const auto& entry : entries) {
            expect(entry.name < strings.size() && std::memchr(strings.data() + entry.name, 0, strings.size() - entry.name), "bounded weight name");
            if (std::strcmp(strings.data() + entry.name, name)) continue;
            int64_t trailing = 1;
            expect(entry.rank >= 1 && entry.rank <= RAD_MAX_RANK, "bounded model rank");
            for (uint32_t axis = 1; axis < entry.rank; ++axis) {
                expect(entry.shape[axis] > 0 && entry.shape[axis] <= 4608 && trailing <= 4608 / entry.shape[axis], "bounded trailing shape");
                trailing *= entry.shape[axis];
            }
            const bool patch = !std::strcmp(name, "v.patch.weight") && entry.rank == 5 && entry.shape[1] == 3 && entry.shape[2] == 2 && entry.shape[3] == 16 && entry.shape[4] == 16;
            expect((entry.rank == (columns ? 2u : 1u) || patch) && entry.shape[0] == rows && trailing == (columns ? columns : 1), "real weight shape");
            expect(entry.enc < encodings.size() && entry.n_planes == 1 && entry.plane_first < planes.size(), "real weight plane");
            const auto& encoding = encodings[entry.enc];
            expect(encoding.n_planes == 1 && encoding.plane[0].dtype == RAD_BF16 && rad_enc_is(&encoding, "plain") && !encoding.transform[0], "plain BF16 model weight");
            std::vector<uint16_t> result(size_t(rows * (columns ? columns : 1)));
            const auto& plane = planes[size_t(entry.plane_first)]; expect(plane.bytes == result.size() * 2 && plane.offset >= header.data_off, "real plane bytes");
            read(plane.offset, result.data(), plane.bytes);
            uint64_t hash = 14695981039346656037ull;
            for (const auto value : result) { hash = (hash ^ (value & 255)) * 1099511628211ull; hash = (hash ^ (value >> 8)) * 1099511628211ull; }
            std::printf("VISION_WEIGHT name=%s rows=%lld columns=%lld offset=%llu bytes=%llu fnv1a64=%016llx\n", name,
                static_cast<long long>(rows), static_cast<long long>(columns), static_cast<unsigned long long>(plane.offset),
                static_cast<unsigned long long>(plane.bytes), static_cast<unsigned long long>(hash)); return result;
        }
        throw std::runtime_error(std::string("missing real weight: ") + name);
    }
};
void host_tests() {
    for (uint32_t bits = 0; bits < 65536; ++bits) {
        const uint16_t value = uint16_t(bits);
        expect(r3dh_vision_strict::narrow(cpu_float(value)) == cpu_bf16(cpu_float(value)), "BF16 independent conversion");
    }
    volatile float factor = 1.00000011920928955078125f, sum = -1.0000002384185791015625f;
    expect(r3dh_vision_strict::add_product(sum, factor, factor) == 0.0f, "non-FMA witness");
    expect(std::fma(float(factor), float(factor), float(sum)) != 0.0f, "FMA witness discriminates");
    expect(r3dh_vision_strict::narrow(1.00390625f) == 0x3f80 && r3dh_vision_strict::narrow(1.01171875f) == 0x3f82, "RNE ties");
    expect(r3dh_vision_strict_gemm_nt(nullptr, nullptr) == RAD_E_INVAL, "null GEMM");
    expect(r3dh_vision_strict_attn_dense(nullptr, nullptr) == RAD_E_INVAL, "null attention");
    for (int64_t rows : {1, 3, 8, 16, 256}) {
        std::vector<RadParam> params{number("M", rows), number("N", 4304), number("K", 1152), text("dtype", "bf16")};
        for (int operand = 0; operand < 5; ++operand) { RadOpdDesc result{}; expect(r3dh_vision_strict_shape("gemm_nt_bias", params.data(), int(params.size()), operand, &result) == RAD_OK, "GEMM shape hook"); }
        params = {number("M", rows), number("head_dim", 72), number("gqa", 1), number("causal", 0)};
        for (int operand = 0; operand < 5; ++operand) { RadOpdDesc result{}; expect(r3dh_vision_strict_shape("attn_dense", params.data(), int(params.size()), operand, &result) == RAD_OK, "attention shape hook"); }
        params[1].ival = 128; RadOpdDesc result{}; expect(r3dh_vision_strict_shape("attn_dense", params.data(), int(params.size()), 0, &result) == RAD_E_UNSUPPORTED, "head128 reject");
    }
    int row_count = 0, schema_count = 0;
    const auto* registry = r3dh_vision_strict_kernels(&row_count);
    expect(registry && row_count == 15 && r3dh_vision_strict_schemas(&schema_count) && schema_count == 3, "registry counts");
    for (int index = 0; index < row_count; ++index)
        expect(registry[index].domain == RAD_DOMAIN_DEVICE && registry[index].launch && registry[index].opd_shape && registry[index].n_constraints >= 5, "strict DEVICE registration");
    std::vector<RadParam> invalid{number("M", 257), number("N", 1152), number("K", 1152)};
    RadOpdDesc description{};
    expect(r3dh_vision_strict_shape("gemm_nt", invalid.data(), int(invalid.size()), 0, &description) == RAD_E_SHAPE, "M257 rejected");
    invalid[0].ival = 0;
    expect(r3dh_vision_strict_shape("gemm_nt", invalid.data(), int(invalid.size()), 0, &description) == RAD_E_SHAPE, "M0 rejected");
    invalid[0].ival = 1; invalid[1].ival = 128;
    expect(r3dh_vision_strict_shape("gemm_nt", invalid.data(), int(invalid.size()), 0, &description) == RAD_E_UNSUPPORTED, "unscoped geometry rejected");
    expect(r3dh_vision_strict_shape("other", invalid.data(), int(invalid.size()), 0, &description) == RAD_E_UNSUPPORTED, "unknown op rejected");
    std::printf("VISION_HOST_PASS checks=%llu\n", static_cast<unsigned long long>(checks));
}
void gemm_case(hipStream_t stream, int64_t rows, int64_t columns, int64_t depth, int activation, bool with_bias,
               bool with_residual, bool inplace, const std::vector<uint16_t>* actual_weight = nullptr, const std::vector<uint16_t>* actual_bias = nullptr) {
    const int64_t left_pitch = depth + 3, output_pitch = columns + 5;
    std::vector<uint16_t> left(size_t(rows * left_pitch)), right(size_t(columns * depth)), bias(size_t(columns), 0), residual(size_t(rows * output_pitch)), output(residual.size());
    fill(right, 7801); fill(bias, 76543); fill(residual, 90561);
    if (actual_weight) right = *actual_weight; if (actual_bias) bias = *actual_bias;
    Buffer device_left(left.size() * 2), device_right(right.size() * 2), device_bias(bias.size() * 2), device_residual(residual.size() * 2), device_output(output.size() * 2);
    device_right.upload(right.data(), stream); device_bias.upload(bias.data(), stream);
    std::vector<uint16_t> previous;
    for (int repeat = 0; repeat < 2; ++repeat) {
        fill(left, uint32_t(34123 + repeat * 971)); device_left.upload(left.data(), stream);
        device_residual.upload(residual.data(), stream); device_output.reset(stream);
        auto out = tensor(inplace ? device_residual.data : device_output.data, rows, columns, output_pitch);
        std::vector<RadTensor> tensors{tensor(device_left.data, rows, depth, left_pitch), tensor(device_right.data, columns, depth)};
        if (with_bias) { tensors.push_back(vector_tensor(device_bias.data, columns)); tensors.push_back(with_residual ? tensor(device_residual.data, rows, columns, output_pitch) : RadTensor{}); }
        tensors.push_back(out);
        std::vector<RadParam> params{number("M", rows), number("N", columns), number("K", depth), text("dtype", "bf16"), text("act", activation == 1 ? "gelu" : activation == 2 ? "gelu_tanh" : "none")};
        auto args = arguments(tensors, params);
        if (with_bias && rows == 1 && columns == 1152 && depth == 1152 && repeat == 0) {
            const auto saved = tensors.back();
            tensors.back().data = static_cast<uint16_t*>(tensors[0].data) + 1;
            expect(r3dh_vision_strict_gemm_nt_bias(&args, reinterpret_cast<RadStream>(stream)) == RAD_E_UNSUPPORTED, "partial input overlap rejected");
            tensors.back() = saved;
            const auto saved_left = tensors[0]; tensors[0].data = left.data();
            expect(r3dh_vision_strict_gemm_nt_bias(&args, reinterpret_cast<RadStream>(stream)) == RAD_E_UNSUPPORTED, "host pointer rejected");
            tensors[0] = saved_left;
            const auto saved_residual = tensors[3]; tensors[3].data = static_cast<uint16_t*>(tensors.back().data) + 1;
            expect(r3dh_vision_strict_gemm_nt_bias(&args, reinterpret_cast<RadStream>(stream)) == RAD_E_UNSUPPORTED, "partial residual overlap rejected");
            tensors[3] = saved_residual;
        }
        expect((with_bias ? r3dh_vision_strict_gemm_nt_bias(&args, reinterpret_cast<RadStream>(stream)) : r3dh_vision_strict_gemm_nt(&args, reinterpret_cast<RadStream>(stream))) == RAD_OK, "native GEMM launch");
        hip_check(hipMemcpyAsync(output.data(), out.data, output.size() * 2, hipMemcpyDeviceToHost, stream)); hip_check(hipStreamSynchronize(stream));
        if (rows == 1 && (with_bias ? reference_bias : reference_gemm)) {
            std::vector<uint16_t> reference(output.size(), 0xa5a5);
            auto host = tensors; host[0].data = left.data(); host[1].data = right.data(); host.back().data = reference.data();
            if (with_bias) { host[2].data = bias.data(); if (with_residual) host[3].data = residual.data(); }
            auto host_args = arguments(host, params);
            expect((with_bias ? reference_bias : reference_gemm)(&host_args, nullptr) == RAD_OK, "libref GEMM call");
            for (int64_t column = 0; column < columns; ++column) {
                ++reference_comparisons; expect(output[size_t(column)] == reference[size_t(column)], "direct libref GEMM comparison");
            }
        }
        const int64_t total = rows * columns, samples = rows <= 3 ? total : std::min<int64_t>(total, 2048);
        for (int64_t sample = 0; sample < samples; ++sample) {
            const int64_t index = samples == total ? sample : sample * (total - 1) / (samples - 1), row = index / columns, column = index % columns;
            float sum = 0.0f;
            for (int64_t inner = 0; inner < depth; ++inner) sum = cpu_mac(sum, cpu_float(left[size_t(row * left_pitch + inner)]), cpu_float(right[size_t(column * depth + inner)]));
            if (with_bias) sum += cpu_float(bias[size_t(column)]);
            const auto expected = cpu_epilogue(sum, activation, residual[size_t(row * output_pitch + column)], with_residual);
            ++comparisons;
            if (output[size_t(row * output_pitch + column)] != expected) {
                std::fprintf(stderr, "GEMM_MISMATCH M=%lld N=%lld K=%lld repeat=%d row=%lld col=%lld got=%04x want=%04x\n", (long long)rows, (long long)columns, (long long)depth, repeat, (long long)row, (long long)column, output[size_t(row * output_pitch + column)], expected);
                throw std::runtime_error("strict GEMM comparison");
            }
        }
        for (int64_t row = 0; row < rows; ++row) for (int64_t column = columns; column < output_pitch; ++column)
            expect(output[size_t(row * output_pitch + column)] == (inplace ? residual[size_t(row * output_pitch + column)] : 0xa5a5), "row padding changed");
        if (repeat) expect(output != previous, "changed input must change GEMM output"); previous = output;
        device_left.guards(stream); device_right.guards(stream); device_bias.guards(stream); device_residual.guards(stream); device_output.guards(stream);
    }
    ++cases; std::printf("VISION_GEMM_PASS source=%s M=%lld N=%lld K=%lld act=%d bias=%d residual=%d inplace=%d repeats=2\n", actual_weight ? "model" : "synthetic", (long long)rows, (long long)columns, (long long)depth, activation, with_bias, with_residual, inplace);
}
void attention_case(hipStream_t stream, int64_t rows, bool segmented) {
    const int64_t fused_pitch = 3456, output_pitch = 1155;
    std::vector<uint16_t> fused(size_t(rows * fused_pitch)), output(size_t(rows * output_pitch));
    std::vector<int32_t> boundaries = segmented && rows > 1 ? std::vector<int32_t>{0, 0, int32_t(rows / 2), int32_t(rows)} : std::vector<int32_t>{0, int32_t(rows)};
    Buffer device_fused(fused.size() * 2), device_output(output.size() * 2), device_boundaries(boundaries.size() * sizeof(int32_t));
    device_boundaries.upload(boundaries.data(), stream); std::vector<uint16_t> previous;
    for (int repeat = 0; repeat < 2; ++repeat) {
        fill(fused, uint32_t(14501 + repeat * 871)); device_fused.upload(fused.data(), stream); device_output.reset(stream);
        auto* base = reinterpret_cast<uint16_t*>(device_fused.data);
        std::vector<RadTensor> tensors{tensor(base, rows, 1152, fused_pitch), tensor(base + 1152, rows, 1152, fused_pitch), tensor(base + 2304, rows, 1152, fused_pitch),
            vector_tensor(device_boundaries.data, int64_t(boundaries.size()), RAD_I32), tensor(device_output.data, rows, 1152, output_pitch)};
        std::vector<RadParam> params{number("M", rows), number("head_dim", 72), number("gqa", 1), number("causal", 0), text("dtype", "bf16")};
        auto args = arguments(tensors, params); expect(r3dh_vision_strict_attn_dense(&args, reinterpret_cast<RadStream>(stream)) == RAD_OK, "native attention launch");
        hip_check(hipMemcpyAsync(output.data(), device_output.data, output.size() * 2, hipMemcpyDeviceToHost, stream)); hip_check(hipStreamSynchronize(stream));
        if (reference_attention) {
            std::vector<uint16_t> reference(output.size(), 0xa5a5);
            auto host = tensors; host[0].data = fused.data(); host[1].data = fused.data() + 1152;
            host[2].data = fused.data() + 2304; host[3].data = boundaries.data(); host[4].data = reference.data();
            auto host_args = arguments(host, params); expect(reference_attention(&host_args, nullptr) == RAD_OK, "libref attention call");
            for (int64_t row = 0; row < rows; ++row) for (int64_t column = 0; column < 1152; ++column) {
                ++reference_comparisons;
                expect(output[size_t(row * output_pitch + column)] == reference[size_t(row * output_pitch + column)], "direct libref attention comparison");
            }
        }
        const float scale = 1.0f / std::sqrt(72.0f);
        for (size_t sequence = 0; sequence + 1 < boundaries.size(); ++sequence) for (int64_t row = boundaries[sequence]; row < boundaries[sequence + 1]; ++row)
            for (int64_t head = 0; head < 16; ++head) {
                std::vector<float> scores(size_t(boundaries[sequence + 1] - boundaries[sequence])); float maximum = -INFINITY;
                for (int64_t column = boundaries[sequence]; column < boundaries[sequence + 1]; ++column) {
                    float dot = 0.0f;
                    for (int64_t dimension = 0; dimension < 72; ++dimension) dot = cpu_mac(dot,
                        cpu_float(fused[size_t(row * fused_pitch + head * 72 + dimension)]), cpu_float(fused[size_t(column * fused_pitch + 1152 + head * 72 + dimension)]));
                    scores[size_t(column - boundaries[sequence])] = dot * scale; maximum = std::max(maximum, dot * scale);
                }
                float denominator = 0.0f;
                for (auto& score : scores) { score = float(std::exp(double(score - maximum))); denominator += score; }
                for (int64_t dimension = 0; dimension < 72; ++dimension) {
                    float sum = 0.0f;
                    for (int64_t column = boundaries[sequence]; column < boundaries[sequence + 1]; ++column) sum = cpu_mac(sum,
                        scores[size_t(column - boundaries[sequence])], cpu_float(fused[size_t(column * fused_pitch + 2304 + head * 72 + dimension)]));
                    const auto expected = cpu_bf16(sum * (1.0f / denominator)); ++comparisons;
                    if (output[size_t(row * output_pitch + head * 72 + dimension)] != expected) throw std::runtime_error("strict attention comparison");
                }
            }
        for (int64_t row = 0; row < rows; ++row) for (int64_t column = 1152; column < output_pitch; ++column) expect(output[size_t(row * output_pitch + column)] == 0xa5a5, "attention row padding");
        if (repeat) expect(previous != output, "changed attention input"); previous = output;
        device_fused.guards(stream); device_output.guards(stream); device_boundaries.guards(stream);
    }
    ++cases; std::printf("VISION_ATTN_PASS M=%lld heads=16 head_dim=72 packed_pitch=3456 segmented=%d repeats=2\n", (long long)rows, segmented);
}
}
int main(int argc, char** argv) {
    try {
        host_tests(); if (argc == 2 && !std::strcmp(argv[1], "--host")) return 0;
        expect(argc == 2 || argc == 3, "usage: --host | --gpu [model]"); expect(!std::strcmp(argv[1], "--gpu"), "GPU mode required");
        int devices = 0; hip_check(hipGetDeviceCount(&devices)); expect(devices == 1, "exactly one GPU");
        hipDeviceProp_t properties{}; hip_check(hipGetDeviceProperties(&properties, 0));
        expect(!std::strncmp(properties.gcnArchName, "gfx1100", 7) && properties.pciBusID == 0xa3, "GPU5 gfx1100 PCI a3");
        std::printf("VISION_DEVICE arch=%s bus=%02x name=%s strict=1\n", properties.gcnArchName, properties.pciBusID, properties.name);
        if (const char* path = std::getenv("VISION_REF_LIBRARY")) {
            void* library = dlopen(path, RTLD_NOW | RTLD_LOCAL); expect(library != nullptr, "load libref");
            reference_gemm = reinterpret_cast<Reference>(dlsym(library, "ref_gemm_nt"));
            reference_bias = reinterpret_cast<Reference>(dlsym(library, "ref_gemm_nt_bias"));
            reference_attention = reinterpret_cast<Reference>(dlsym(library, "ref_attn_dense"));
            expect(reference_gemm && reference_bias && reference_attention, "libref symbols");
        }
        hipStream_t stream; hip_check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
        const int64_t shapes[][2]{{1152,1536},{3456,1152},{1152,1152},{4304,1152},{1152,4304},{4608,4608},{2560,4608}};
        const char* names[]{"v.patch", "v.blk.0.qkv", "v.blk.0.proj", "v.blk.0.fc1", "v.blk.0.fc2", "v.merger.fc1", "v.merger.fc2"};
        Model* model = argc == 3 ? new Model(argv[2]) : nullptr;
        for (size_t shape = 0; shape < 7; ++shape) {
            std::vector<uint16_t> weight, bias;
            if (model) { weight = model->weight((std::string(names[shape]) + ".weight").c_str(), shapes[shape][0], shapes[shape][1]); bias = model->weight((std::string(names[shape]) + ".bias").c_str(), shapes[shape][0]); }
            for (int64_t rows : {1,3,8,16,256}) {
                gemm_case(stream, rows, shapes[shape][0], shapes[shape][1], 0,
                    false, false, false, model ? &weight : nullptr);
                gemm_case(stream, rows, shapes[shape][0], shapes[shape][1], shape == 3 ? 2 : shape == 5 ? 1 : 0,
                    true, shape == 2 || shape == 4, shape == 2 || shape == 4, model ? &weight : nullptr, model ? &bias : nullptr);
            }
        }
        delete model;
        if (argc == 2) {
            gemm_case(stream, 3, 1152, 1152, 0, false, false, false);
            gemm_case(stream, 3, 1152, 1152, 1, true, true, false);
            gemm_case(stream, 3, 1152, 1152, 2, true, true, false);
            for (int64_t rows : {1,3,8,16,256}) { attention_case(stream, rows, false); attention_case(stream, rows, true); }
        }
        hip_check(hipStreamDestroy(stream));
        std::printf("VISION_GPU_PASS source=%s cases=%llu comparisons=%llu checks=%llu mismatches=0\n", argc == 3 ? "model" : "synthetic", (unsigned long long)cases, (unsigned long long)comparisons, (unsigned long long)checks);
        std::printf("VISION_LIBREF_PASS comparisons=%llu\n", (unsigned long long)reference_comparisons);
    } catch (const std::exception& error) { std::fprintf(stderr, "VISION_FAIL %s\n", error.what()); return 1; }
    return 0;
}
