#include <hip/hip_runtime.h>
#include "r3dh_vision_rope.h"
#include "r3dh_vision_strict.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <dlfcn.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
extern "C" int r3dh_vision_rope_registry_matches(const RadParam*, int);

namespace {
uint64_t checks = 0, comparisons = 0, reference_comparisons = 0, cases = 0;
uint64_t invariance_comparisons = 0;
using Reference = int (*)(const RadArgs*, RadStream);
void expect(bool condition, const char* message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}
void hip_check(hipError_t status) {
    if (status != hipSuccess) throw std::runtime_error(hipGetErrorString(status));
}
uint16_t bf16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof bits);
    if ((bits & 0x7fffffffu) > 0x7f800000u) return uint16_t((bits >> 16) | 0x40);
    return uint16_t((bits + 32767u + ((bits >> 16) & 1u)) >> 16);
}
float fp32(uint16_t value) {
    uint32_t bits = uint32_t(value) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof result);
    return result;
}
RadParam integer(const char* key, int64_t value) {
    RadParam result{}; result.key = key; result.kind = RAD_P_INT; result.ival = value; return result;
}
RadParam real(const char* key, double value) {
    RadParam result{}; result.key = key; result.kind = RAD_P_F64; result.dval = value; return result;
}
RadParam text(const char* key, const char* value) {
    RadParam result{}; result.key = key; result.kind = RAD_P_STR; result.sval = value; return result;
}
std::vector<RadParam> parameters(int rows, double theta = 10000, double scale = 1) {
    return {integer("M", rows), integer("head_dim", 72), integer("n_head", 16),
        integer("n_head_kv", 16), real("theta", theta), real("scale", scale), text("mode", "axial"),
        integer("rotary_dim", 72), text("sections", "18 18")};
}
RadTensor matrix(void* memory, int64_t rows, int64_t width, int64_t pitch, uint32_t dtype = RAD_BF16) {
    RadTensor result{}; result.data = memory; result.dtype = dtype; result.rank = 2;
    result.shape[0] = rows; result.shape[1] = width; result.stride[0] = pitch; result.stride[1] = 1;
    return result;
}
RadArgs arguments(const std::vector<RadTensor>& tensors, const std::vector<RadParam>& params) {
    RadArgs result{}; result.t = tensors.data(); result.n_t = int(tensors.size());
    result.p = params.data(); result.n_p = int(params.size()); return result;
}
void host_tests() {
    const uint64_t start = checks;
    int count = 0;
    const auto* old_schemas = r3dh_vision_strict_schemas(&count);
    expect(count == 3 && r3d_vision_schema_count() == 4, "aggregate schema count");
    for (int index = 0; index < count; ++index)
        expect(r3d_vision_schema_at(index) == old_schemas + index, "preserve strict schema pointers");
    const auto* old_rows = r3dh_vision_strict_kernels(&count);
    expect(count == 15 && r3d_vision_kernel_count() == 16, "aggregate kernel count");
    for (int index = 0; index < count; ++index)
        expect(r3d_vision_kernel_at(index) == old_rows + index, "preserve strict row pointers");
    expect(!r3d_vision_schema_at(-1) && !r3d_vision_schema_at(4), "schema index bounds");
    expect(!r3d_vision_kernel_at(-1) && !r3d_vision_kernel_at(16), "kernel index bounds");
    const auto* row = r3d_vision_kernel_at(15);
    expect(row == r3dh_vision_rope_kernel() && row->launch == r3dh_vision_rope, "native candidate");
    expect(row->domain == RAD_DOMAIN_DEVICE && !std::strcmp(row->op, "rope"), "device only rope");
    expect(row->opd_shape == r3dh_vision_rope_shape && row->n_constraints == 7, "candidate ABI");
    const auto* schema = r3d_vision_schema_at(3);
    expect(schema == r3dh_vision_rope_schema() && schema->n_operands == 2, "canonical schema");
    expect(schema->operands[0].role == RAD_OPD_INOUT, "canonical in-place operand");
    for (int rows : {1, 3, 8, 16, 256}) {
        auto params = parameters(rows);
        expect(r3dh_vision_rope_registry_matches(params.data(), int(params.size())) == 1, "actual engine constraints accept ViT parameters");
        for (int operand = 0; operand < 2; ++operand) {
            RadOpdDesc output{};
            expect(row->opd_shape(params.data(), int(params.size()), operand, &output) == RAD_OK, "shape success");
            expect(output.rank == 2 && output.dtype == (operand ? RAD_I32 : RAD_BF16), "shape dtype/rank");
            expect(output.shape[0] == (operand ? 2 : rows) && output.shape[1] == (operand ? rows : 3456), "shape geometry");
        }
    }
    auto params = parameters(3);
    auto range = params;
    range[0].kind = RAD_P_RANGE; range[0].ival = 1; range[0].ihi = 256;
    RadOpdDesc output{};
    expect(row->opd_shape(range.data(), int(range.size()), 0, &output) == RAD_OK && output.shape[0] == 256, "capacity range");
    std::vector<uint16_t> storage(3 * 3456, 0x3f80);
    std::vector<int32_t> coordinates(9, 0);
    std::vector<RadTensor> tensors{matrix(storage.data(), 3, 3456, 3456), matrix(coordinates.data(), 2, 3, 3, RAD_I32)};
    const auto unchanged = storage;
    uint64_t rejects = 0;
    auto reject = [&](std::vector<RadParam> values, std::vector<RadTensor> operands, int wanted, bool views = false) {
        auto args = arguments(operands, values);
        const int status = views ? r3dh_vision_rope_qk(&args, nullptr) : r3dh_vision_rope(&args, nullptr);
        expect(status == wanted, "exact rejection status"); expect(storage == unchanged, "reject before mutation"); ++rejects;
    };
    for (int rows : {0, -1, 257, 2048}) {
        auto values = params; values[0].ival = rows; reject(values, tensors, RAD_E_SHAPE);
        expect(r3dh_vision_rope_registry_matches(values.data(), int(values.size())) == 0, "actual engine constraints reject unsupported M");
    }
    for (int field : {1, 2, 3, 7}) { auto values = params; --values[field].ival; reject(values, tensors, RAD_E_UNSUPPORTED); }
    for (int field : {4, 5}) for (double value : {0.0, -1.0, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN(), 1e300}) {
        auto values = params; values[field].dval = value; reject(values, tensors, RAD_E_INVAL);
    }
    for (size_t field = 0; field < params.size(); ++field) { auto values = params; values[field].kind = 99; reject(values, tensors, RAD_E_INVAL); }
    for (const char* sections : {"18", "18 17", "18 18 1", "18 18junk", "", "-18 18"}) {
        auto values = params; values[8].sval = sections; reject(values, tensors, RAD_E_UNSUPPORTED);
    }
    for (const char* mode : {"neox", "mrope", "imrope", "gptj", ""}) { auto values = params; values[6].sval = mode; reject(values, tensors, RAD_E_UNSUPPORTED); }
    for (const char* dtype : {"f16", "f32"}) { auto values = params; values.push_back(text("dtype", dtype)); reject(values, tensors, RAD_E_DTYPE); }
    for (const char* key : {"yarn_factor", "llama3_factor"}) { auto values = params; values.push_back(real(key, 2)); reject(values, tensors, RAD_E_UNSUPPORTED); }
    { auto values = params; values.push_back(text("ladder", "bogus")); reject(values, tensors, RAD_E_UNSUPPORTED); }
    { auto values = params; values.push_back(integer("unknown", 1)); reject(values, tensors, RAD_E_UNSUPPORTED); }
    { auto values = params; values.push_back(params[0]); reject(values, tensors, RAD_E_INVAL); }
    { auto values = params; values[0].key = nullptr; reject(values, tensors, RAD_E_INVAL); }
    { auto values = params; values[6].sval = nullptr; reject(values, tensors, RAD_E_INVAL); }
    reject(range, tensors, RAD_E_INVAL);
    for (int operand = 0; operand < 2; ++operand) {
        auto changed = tensors; changed[operand].data = nullptr; reject(params, changed, RAD_E_SHAPE);
        changed = tensors; changed[operand].dtype = RAD_F16; reject(params, changed, RAD_E_DTYPE);
        changed = tensors; changed[operand].rank = 1; reject(params, changed, RAD_E_SHAPE);
        changed = tensors; changed[operand].stride[1] = 2; reject(params, changed, RAD_E_SHAPE);
        changed = tensors; ++changed[operand].stride[0]; reject(params, changed, RAD_E_SHAPE);
        changed = tensors; ++changed[operand].shape[1]; reject(params, changed, RAD_E_SHAPE);
        changed = tensors; changed[operand].data = static_cast<char*>(changed[operand].data) + 1; reject(params, changed, RAD_E_SHAPE);
    }
    for (int components : {1, 4, 8}) { auto changed = tensors; changed[1].shape[0] = components; reject(params, changed, RAD_E_SHAPE); }
    { auto changed = tensors; changed[0].shape[0] = 2; reject(params, changed, RAD_E_SHAPE); }
    { auto changed = tensors; changed.push_back(changed[0]); reject(params, changed, RAD_E_INVAL); }
    { auto changed = tensors; changed.pop_back(); reject(params, changed, RAD_E_INVAL); }
    std::vector<RadTensor> views{matrix(storage.data(), 3, 1152, 3456), matrix(storage.data() + 1152, 3, 1152, 3456), tensors[1]};
    { auto changed = views; changed[1].data = storage.data(); reject(params, changed, RAD_E_SHAPE, true); }
    { auto changed = views; changed[1].data = storage.data() + 8; reject(params, changed, RAD_E_SHAPE, true); }
    expect(r3dh_vision_rope(nullptr, nullptr) == RAD_E_INVAL, "null args"); ++rejects;
    auto args = arguments(tensors, params); args.t = nullptr;
    expect(r3dh_vision_rope(&args, nullptr) == RAD_E_INVAL, "null operands"); ++rejects;
    args = arguments(tensors, params); args.p = nullptr;
    expect(r3dh_vision_rope(&args, nullptr) == RAD_E_INVAL, "null params"); ++rejects;
    expect(row->opd_shape(params.data(), int(params.size()), 2, &output) == RAD_E_INVAL, "operand bounds");
    expect(row->opd_shape(params.data(), int(params.size()), 0, nullptr) == RAD_E_INVAL, "null shape output");
    for (uint32_t bits = 0; bits < 65536; ++bits) {
        if ((bits & 0x7f80u) != 0x7f80u) expect(bf16(fp32(uint16_t(bits))) == bits, "BF16 roundtrip");
    }
    expect(bf16(1.00390625f) == 0x3f80 && bf16(1.01171875f) == 0x3f82, "BF16 ties to even");
    std::printf("VISION_ROPE_HOST_PASS checks=%llu rejects=%llu schemas=4 kernels=16\n",
        (unsigned long long)(checks - start), (unsigned long long)rejects);
}
struct Buffer {
    void* allocation = nullptr;
    uint8_t* data = nullptr;
    size_t bytes;
    explicit Buffer(size_t size) : bytes(size) {
        hip_check(hipMalloc(&allocation, bytes + 256)); data = static_cast<uint8_t*>(allocation) + 128;
    }
    ~Buffer() { if (allocation) static_cast<void>(hipFree(allocation)); }
    void upload(const void* source, hipStream_t stream) {
        hip_check(hipMemsetAsync(allocation, 0xa5, bytes + 256, stream));
        hip_check(hipMemcpyAsync(data, source, bytes, hipMemcpyHostToDevice, stream));
    }
    void download(void* target, hipStream_t stream) {
        hip_check(hipMemcpyAsync(target, data, bytes, hipMemcpyDeviceToHost, stream));
    }
    void guards(hipStream_t stream) {
        uint8_t before[128], after[128];
        hip_check(hipMemcpyAsync(before, allocation, 128, hipMemcpyDeviceToHost, stream));
        hip_check(hipMemcpyAsync(after, data + bytes, 128, hipMemcpyDeviceToHost, stream));
        hip_check(hipStreamSynchronize(stream));
        for (size_t index = 0; index < 128; ++index) expect(before[index] == 0xa5 && after[index] == 0xa5, "device guards");
    }
};
void oracle(std::vector<uint16_t>& data, const std::vector<int32_t>& coordinates, int rows, float theta, float scale) {
    for (int token = 0; token < rows; ++token) for (int head = 0; head < 32; ++head)
        for (int axis = 0; axis < 2; ++axis) for (int frequency = 0; frequency < 18; ++frequency) {
            const float exponent = float(frequency) / 18.0f;
            const float inverse = 1.0f / float(std::pow(double(theta), double(exponent)));
            const float coordinate = float(coordinates[size_t(axis * rows + token)]) / scale;
            const float angle = coordinate * inverse;
            const float cosine = float(std::cos(double(angle))), sine = float(std::sin(double(angle)));
            const size_t offset = size_t(token * 3456 + head * 72 + axis * 18 + frequency);
            const float left = fp32(data[offset]), right = fp32(data[offset + 36]);
            data[offset] = bf16(left * cosine - right * sine);
            data[offset + 36] = bf16(left * sine + right * cosine);
        }
}
void compare(const std::vector<uint16_t>& actual, const std::vector<uint16_t>& wanted, bool reference) {
    expect(actual.size() == wanted.size(), "comparison size");
    for (size_t index = 0; index < actual.size(); ++index) {
        if (reference) ++reference_comparisons; else ++comparisons;
        if (actual[index] != wanted[index]) {
            std::fprintf(stderr, "VISION_ROPE_MISMATCH index=%zu actual=%04x wanted=%04x libref=%d\n", index, actual[index], wanted[index], reference);
            throw std::runtime_error("bit-exact axial rotation");
        }
    }
}
void gpu_case(hipStream_t stream, Reference reference, int rows, int pattern, int layout, int components) {
    const double theta = pattern == 4 ? 500000.0 : 10000.0;
    const double scale = pattern == 4 ? 1.25 : 1.0;
    auto params = parameters(rows, theta, scale);
    std::vector<uint16_t> input(size_t(rows * 3456)), actual(input.size()), previous;
    std::vector<int32_t> coordinates(size_t(rows * components)), returned(coordinates.size());
    Buffer fused(input.size() * 2), query(size_t(rows * 1152 * 2)), key(size_t(rows * 1152 * 2));
    Buffer position(coordinates.size() * sizeof(int32_t));
    for (int repeat = 0; repeat < 2; ++repeat) {
        uint32_t seed = uint32_t(19937 + repeat * 931 + rows * 71 + pattern * 115);
        for (auto& value : input) {
            seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5;
            value = bf16(float(int(seed % 8193) - 4096) / 1024.0f);
        }
        for (int token = 0; token < rows; ++token) {
            coordinates[size_t(token)] = pattern == 0 || pattern == 3 ? 0 : pattern == 1 ? token / 16 : pattern == 2 ? token + 1 : (token % 2 ? -65537 : 16777217) + repeat;
            coordinates[size_t(rows + token)] = pattern == 0 || pattern == 2 ? 0 : pattern == 1 ? token % 16 : pattern == 3 ? 47 - token % 48 : (token % 3 ? 32769 : -1048577) - repeat;
            if (components == 3) coordinates[size_t(2 * rows + token)] = repeat ? std::numeric_limits<int32_t>::min() : std::numeric_limits<int32_t>::max();
        }
        fused.upload(input.data(), stream); position.upload(coordinates.data(), stream);
        auto* base = reinterpret_cast<uint16_t*>(fused.data);
        std::vector<RadTensor> tensors{matrix(base, rows, 3456, 3456), matrix(position.data, components, rows, rows, RAD_I32)};
        std::vector<uint16_t> packed_query(size_t(rows * 1152)), packed_key(packed_query.size());
        if (layout) {
            if (layout == 2) {
                for (int token = 0; token < rows; ++token) {
                    std::memcpy(packed_query.data() + token * 1152, input.data() + token * 3456, 2304);
                    std::memcpy(packed_key.data() + token * 1152, input.data() + token * 3456 + 1152, 2304);
                }
                query.upload(packed_query.data(), stream); key.upload(packed_key.data(), stream);
            }
            tensors = {matrix(layout == 1 ? base : reinterpret_cast<uint16_t*>(query.data), rows, 1152, layout == 1 ? 3456 : 1152),
                matrix(layout == 1 ? base + 1152 : reinterpret_cast<uint16_t*>(key.data), rows, 1152, layout == 1 ? 3456 : 1152), tensors[1]};
        }
        auto args = arguments(tensors, params);
        expect((layout ? r3dh_vision_rope_qk(&args, reinterpret_cast<RadStream>(stream)) : r3dh_vision_rope(&args, reinterpret_cast<RadStream>(stream))) == RAD_OK, "GPU launch");
        fused.download(actual.data(), stream);
        if (layout == 2) { query.download(packed_query.data(), stream); key.download(packed_key.data(), stream); }
        position.download(returned.data(), stream);
        hip_check(hipStreamSynchronize(stream));
        if (layout == 2) for (int token = 0; token < rows; ++token) {
            std::memcpy(actual.data() + token * 3456, packed_query.data() + token * 1152, 2304);
            std::memcpy(actual.data() + token * 3456 + 1152, packed_key.data() + token * 1152, 2304);
        }
        auto expected = input;
        oracle(expected, coordinates, rows, float(theta), float(scale)); compare(actual, expected, false);
        auto ref_output = input;
        std::vector<RadTensor> ref_tensors{matrix(ref_output.data(), rows, 3456, 3456), matrix(coordinates.data(), components, rows, rows, RAD_I32)};
        auto ref_args = arguments(ref_tensors, params);
        expect(reference(&ref_args, nullptr) == RAD_OK, "libref launch"); compare(actual, ref_output, true);
        expect(returned == coordinates, "positions unchanged including unused third plane");
        if (pattern == 0) expect(actual == input, "zero coordinates identity");
        if (repeat) expect(actual != previous, "changed input changes result");
        previous = actual;
        fused.guards(stream); position.guards(stream);
        if (layout == 2) { query.guards(stream); key.guards(stream); }
    }
    ++cases;
    std::printf("VISION_ROPE_CASE_PASS M=%d coords=%d layout=%d components=%d repeats=2 exhaustive=1\n", rows, pattern, layout, components);
}
void third_plane_case(hipStream_t stream, int rows) {
    std::vector<uint16_t> input(size_t(rows * 3456)), baseline(input.size()), changed(input.size());
    std::vector<int32_t> coordinates(size_t(3 * rows));
    for (size_t index = 0; index < input.size(); ++index) input[index] = bf16(float(int(index % 3071) - 1535) / 1024.0f);
    for (int token = 0; token < rows; ++token) {
        coordinates[size_t(token)] = token + 7;
        coordinates[size_t(rows + token)] = 47 - token % 48;
    }
    Buffer data(input.size() * 2), position(coordinates.size() * sizeof(int32_t));
    auto params = parameters(rows);
    std::vector<RadTensor> tensors{matrix(data.data, rows, 3456, 3456), matrix(position.data, 3, rows, rows, RAD_I32)};
    auto args = arguments(tensors, params);
    for (int repeat = 0; repeat < 2; ++repeat) {
        for (int token = 0; token < rows; ++token)
            coordinates[size_t(2 * rows + token)] = repeat ? std::numeric_limits<int32_t>::min() : std::numeric_limits<int32_t>::max();
        data.upload(input.data(), stream); position.upload(coordinates.data(), stream);
        expect(r3dh_vision_rope(&args, reinterpret_cast<RadStream>(stream)) == RAD_OK, "third-plane-only launch");
        data.download(repeat ? changed.data() : baseline.data(), stream);
        hip_check(hipStreamSynchronize(stream));
        data.guards(stream); position.guards(stream);
    }
    for (size_t index = 0; index < baseline.size(); ++index) {
        ++invariance_comparisons;
        expect(baseline[index] == changed[index], "third plane must not affect rotation");
    }
    std::printf("VISION_ROPE_THIRD_PLANE_PASS M=%d comparisons=%zu\n", rows, baseline.size());
}
}
int main(int argc, char** argv) {
    try {
        expect(argc == 2, "usage: --host | --gpu");
        host_tests();
        if (!std::strcmp(argv[1], "--host")) return 0;
        expect(!std::strcmp(argv[1], "--gpu"), "GPU mode required");
        int devices = 0; hip_check(hipGetDeviceCount(&devices)); expect(devices == 1, "one visible GPU");
        hipDeviceProp_t properties{}; hip_check(hipGetDeviceProperties(&properties, 0));
        expect(!std::strncmp(properties.gcnArchName, "gfx1100", 7) && properties.pciBusID == 0xa3, "GPU5 gfx1100 PCI a3");
        std::printf("VISION_ROPE_DEVICE arch=%s bus=%02x name=%s strict=1\n", properties.gcnArchName, properties.pciBusID, properties.name);
        const char* path = std::getenv("VISION_ROPE_REF_LIBRARY"); expect(path != nullptr, "libref path required");
        void* library = dlopen(path, RTLD_NOW | RTLD_LOCAL);
        if (!library) throw std::runtime_error(dlerror());
        auto reference = reinterpret_cast<Reference>(dlsym(library, "ref_rope")); expect(reference != nullptr, "actual libref rope symbol");
        hipStream_t streams[2];
        for (auto& stream : streams) hip_check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
        for (int rows : {1, 3, 8, 16, 256}) for (int pattern = 0; pattern < 5; ++pattern)
            for (int layout = 0; layout < 3; ++layout) for (int components : {2, 3})
                gpu_case(streams[(rows + pattern + layout + components) % 2], reference, rows, pattern, layout, components);
        for (int rows : {1, 3, 8, 16, 256}) third_plane_case(streams[rows % 2], rows);
        for (auto& stream : streams) hip_check(hipStreamDestroy(stream));
        dlclose(library);
        std::printf("VISION_ROPE_GPU_PASS cases=%llu comparisons=%llu libref_comparisons=%llu checks=%llu mismatches=0\n",
            (unsigned long long)cases, (unsigned long long)comparisons, (unsigned long long)reference_comparisons, (unsigned long long)checks);
        std::printf("VISION_ROPE_INVARIANCE_PASS cases=5 comparisons=%llu nonblocking_streams=2 launches=310\n",
            (unsigned long long)invariance_comparisons);
    } catch (const std::exception& error) { std::fprintf(stderr, "VISION_ROPE_FAIL %s\n", error.what()); return 1; }
    return 0;
}
