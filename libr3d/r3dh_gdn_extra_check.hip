#include "r3dh_gdn_extra.h"
#include "ref_ops.h"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>
#if defined(__FAST_MATH__)
#error checker requires strict floating point
#endif
namespace {
void expect(bool condition, const char* text) { if (!condition) throw std::runtime_error(text); }
void hip_ok(hipError_t status) { if (status != hipSuccess) throw std::runtime_error(hipGetErrorString(status)); }
void rad_ok(int status) { if (status != RAD_OK) { std::fprintf(stderr, "status=%d\n", status); throw std::runtime_error("RAD failure"); } }
uint16_t bf16(float value) {
    const uint32_t bits = std::bit_cast<uint32_t>(value);
    const uint32_t tail = bits & 65535u;
    return static_cast<uint16_t>((bits >> 16) + (tail > 32768u || (tail == 32768u && ((bits >> 16) & 1u))));
}
struct Buffer {
    RadTensor descriptor{};
    std::vector<uint8_t> reference;
    uint8_t* allocation = nullptr;
    size_t bytes = 0;
    Buffer(uint32_t type, std::initializer_list<int64_t> shape, int padding = 0) {
        descriptor.dtype = type; descriptor.rank = shape.size();
        std::copy(shape.begin(), shape.end(), descriptor.shape);
        int64_t stride = 1;
        for (int axis = descriptor.rank - 1; axis >= 0; --axis) {
            descriptor.stride[axis] = stride;
            stride *= descriptor.shape[axis];
            if (axis == 1) stride += padding;
        }
        const int size = type == RAD_BF16 ? 2 : type == RAD_I8 ? 1 : 4;
        bytes = stride * size; reference.resize(bytes, 0xa5);
        hip_ok(hipMalloc(&allocation, bytes + 512));
        hip_ok(hipMemset(allocation, 0xa5, bytes + 512));
    }
    ~Buffer() { if (allocation) static_cast<void>(hipFree(allocation)); }
    RadTensor host() { RadTensor tensor = descriptor; tensor.data = reference.data(); return tensor; }
    RadTensor device() { RadTensor tensor = descriptor; tensor.data = allocation + 256; return tensor; }
    void upload(hipStream_t stream) { hip_ok(hipMemcpyAsync(allocation + 256, reference.data(), bytes, hipMemcpyHostToDevice, stream)); }
    void fill(int seed, float magnitude = 1) {
        const int size = descriptor.dtype == RAD_BF16 ? 2 : descriptor.dtype == RAD_I8 ? 1 : 4;
        for (size_t element = 0; element < bytes / size; ++element) {
            const float value = static_cast<float>(static_cast<int>((element * 73 + seed * 31) % 257) - 128) * magnitude / 128;
            if (descriptor.dtype == RAD_BF16) reinterpret_cast<uint16_t*>(reference.data())[element] = bf16(value);
            else if (descriptor.dtype == RAD_F32) reinterpret_cast<float*>(reference.data())[element] = value;
            else if (descriptor.dtype == RAD_I32) reinterpret_cast<int32_t*>(reference.data())[element] = 0;
        }
    }
    void integers(const std::vector<int>& values) {
        expect(values.size() * 4 <= bytes, "integer fixture capacity");
        std::memcpy(reference.data(), values.data(), values.size() * 4);
    }
    void compare(const char* label) {
        std::vector<uint8_t> actual(bytes);
        hip_ok(hipMemcpy(actual.data(), allocation + 256, bytes, hipMemcpyDeviceToHost));
        size_t differences = 0, first = bytes;
        for (size_t index = 0; index < bytes; ++index)
            if (actual[index] != reference[index]) { ++differences; if (first == bytes) first = index; }
        if (differences) {
            std::fprintf(stderr, "DIFF %s bytes=%zu first=%zu expected=%02x actual=%02x\n", label, differences, first, reference[first], actual[first]);
            if (descriptor.dtype == RAD_F32) {
                const size_t element = first / 4;
                std::fprintf(stderr, "F32 expected=%.9g actual=%.9g\n", reinterpret_cast<const float*>(reference.data())[element], reinterpret_cast<const float*>(actual.data())[element]);
            }
            throw std::runtime_error("bit-exact libref comparison failed");
        }
    }
    void guards() {
        uint8_t prefix[256], suffix[256];
        hip_ok(hipMemcpy(prefix, allocation, 256, hipMemcpyDeviceToHost));
        hip_ok(hipMemcpy(suffix, allocation + bytes + 256, 256, hipMemcpyDeviceToHost));
        for (int index = 0; index < 256; ++index) expect(prefix[index] == 0xa5 && suffix[index] == 0xa5, "allocation guard");
    }
};
struct Fixture {
    std::vector<std::unique_ptr<Buffer>> buffers;
    hipStream_t stream = nullptr;
    Fixture() { hip_ok(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking)); }
    ~Fixture() { if (stream) static_cast<void>(hipStreamDestroy(stream)); }
    Buffer& add(uint32_t type, std::initializer_list<int64_t> shape, int padding = 0) {
        buffers.push_back(std::make_unique<Buffer>(type, shape, padding)); return *buffers.back();
    }
    void upload() { for (auto& buffer : buffers) buffer->upload(stream); hip_ok(hipStreamSynchronize(stream)); }
    void guards() { for (auto& buffer : buffers) buffer->guards(); }
};
std::vector<RadTensor> descriptors(const std::vector<Buffer*>& buffers, bool device) {
    std::vector<RadTensor> result;
    for (Buffer* buffer : buffers) result.push_back(buffer ? device ? buffer->device() : buffer->host() : RadTensor{});
    return result;
}
RadArgs arguments(std::vector<RadTensor>& tensors, std::vector<RadParam>& parameters) {
    RadArgs result{}; result.t = tensors.data(); result.n_t = tensors.size();
    result.p = parameters.data(); result.n_p = parameters.size(); return result;
}
std::vector<RadParam> parameters(int rows, int heads, int bound, bool anchor, int group, const char* act) {
    return {RAD_INT("M", rows), RAD_INT("q_len", bound), RAD_INT("head_k", 128), RAD_INT("head_v", 128),
        RAD_INT("conv_width", 4), RAD_INT("chunk", 64), RAD_INT("n_head_v", heads),
        RAD_INT("group", group), RAD_INT("n", heads * 128), RAD_F64("scale", 1.0 / std::sqrt(128.0)),
        RAD_F64("eps", 1e-6), RAD_STR("state_form", anchor ? "anchor" : "per_candidate"), RAD_STR("act", act)};
}
int64_t scratch_expected(int rows, int heads) {
    const int keys = heads == 24 ? 8 : 16;
    return (static_cast<int64_t>(rows) * (2 * keys + heads) * 256 + 255) & ~int64_t{255};
}
void scratch_host() {
    int positives = 0, negatives = 0;
    for (int heads : {24, 48}) {
        for (int rows : {1, 3, 8, 16, 2048}) {
            std::vector<RadTensor> tensors(18);
            auto params = std::vector<RadParam>{RAD_INT("head_k", 128), RAD_INT("head_v", 128),
                RAD_INT("conv_width", 4), RAD_INT("n_head_v", heads)};
            auto& output = tensors[15];
            output.dtype = RAD_BF16; output.rank = 3;
            output.shape[0] = rows; output.shape[1] = heads; output.shape[2] = 128;
            auto args = arguments(tensors, params);
            for (const auto& tensor : tensors) expect(tensor.data == nullptr, "scratch host null tensor data");
            expect(args.scratch == nullptr && args.scratch_bytes == 0, "scratch host null scratch");
            const int64_t actual = r3dh_gdn_extra_scratch(&args);
            expect(actual == scratch_expected(rows, heads) && actual % 256 == 0,
                "descriptor-only scratch byte count");
            ++positives;
            std::printf("PASS scratch_host M=%d heads=%d keys=%d tensors=18 data=null bytes=%lld aligned=256\n",
                rows, heads, heads / 3, static_cast<long long>(actual));
        }
    }
    std::vector<RadTensor> tensors(18);
    auto params = std::vector<RadParam>{RAD_INT("head_k", 128), RAD_INT("head_v", 128),
        RAD_INT("conv_width", 4), RAD_INT("n_head_v", 24)};
    auto& output = tensors[15];
    output.dtype = RAD_BF16; output.rank = 3;
    output.shape[0] = 3; output.shape[1] = 24; output.shape[2] = 128;
    auto args = arguments(tensors, params);
    auto reject = [&](int expected, const char* label) {
        expect(r3dh_gdn_extra_scratch(&args) == expected, label); ++negatives;
    };
    expect(r3dh_gdn_extra_scratch(nullptr) == RAD_E_INVAL, "scratch rejects null args"); ++negatives;
    args.t = nullptr; reject(RAD_E_INVAL, "scratch rejects null tensors"); args.t = tensors.data();
    for (int count : {0, 17, 19}) {
        args.n_t = count; reject(RAD_E_INVAL, "scratch rejects tensor count");
    }
    args.n_t = 18;
    for (int rank : {0, 2, 4}) {
        output.rank = rank; reject(RAD_E_SHAPE, "scratch rejects output rank");
    }
    output.rank = 3;
    for (uint32_t dtype : {RAD_F32, RAD_I8, RAD_I32}) {
        output.dtype = dtype; reject(RAD_E_SHAPE, "scratch rejects output dtype");
    }
    output.dtype = RAD_BF16;
    for (int heads : {0, 8, 16, 23, 25, 47, 49}) {
        output.shape[1] = heads; reject(RAD_E_SHAPE, "scratch rejects bad heads");
    }
    output.shape[1] = 24;
    for (int rows : {-1, 0, 2049}) {
        output.shape[0] = rows; reject(RAD_E_SHAPE, "scratch rejects rows");
    }
    output.shape[0] = 3;
    output.shape[2] = 127; reject(RAD_E_SHAPE, "scratch rejects head width"); output.shape[2] = 128;
    params[0] = RAD_INT("head_k", 64); reject(RAD_E_SHAPE, "scratch rejects head_k"); params[0] = RAD_INT("head_k", 128);
    params[1] = RAD_INT("head_v", 64); reject(RAD_E_SHAPE, "scratch rejects head_v"); params[1] = RAD_INT("head_v", 128);
    params[2] = RAD_INT("conv_width", 3); reject(RAD_E_SHAPE, "scratch rejects conv width"); params[2] = RAD_INT("conv_width", 4);
    params[3] = RAD_INT("n_head_v", 48); reject(RAD_E_SHAPE, "scratch rejects mismatched heads"); params[3] = RAD_INT("n_head_v", 24);
    expect(r3dh_gdn_extra_scratch(&args) == scratch_expected(3, 24), "scratch negative fixture restored");
    std::printf("PASS scratch_host positives=%d negatives=%d gpu_calls=0\n", positives, negatives);
}
void scratch_runtime() {
    for (int heads : {24, 48}) {
        for (int rows : {1, 3, 8, 16, 2048}) {
            Fixture fixture;
            const int channels = (2 * (heads / 3) + heads) * 128;
            auto params = parameters(rows, heads, rows, false, 128, "sigmoid");
            std::vector<Buffer*> operands{
                &fixture.add(RAD_BF16, {rows, channels}),
                &fixture.add(RAD_BF16, {channels, 4}),
                &fixture.add(RAD_BF16, {channels}),
                &fixture.add(RAD_BF16, {1, channels, rows + 2}),
                &fixture.add(RAD_I32, {1}),
                &fixture.add(RAD_BF16, {rows, heads}),
                &fixture.add(RAD_BF16, {rows, heads}),
                &fixture.add(RAD_F32, {heads}),
                &fixture.add(RAD_F32, {heads}),
                &fixture.add(RAD_F32, {1, heads, 128, 128}),
                &fixture.add(RAD_I32, {1, rows}),
                &fixture.add(RAD_I32, {1}),
                &fixture.add(RAD_I32, {2}),
                &fixture.add(RAD_BF16, {rows, heads * 128}),
                &fixture.add(RAD_F32, {128}),
                &fixture.add(RAD_BF16, {rows, heads, 128}),
                &fixture.add(RAD_I8, {rows, heads * 128}),
                &fixture.add(RAD_F32, {rows, heads})};
            auto device = descriptors(operands, true);
            auto args = arguments(device, params);
            for (const auto& tensor : device) expect(tensor.data != nullptr, "scratch runtime allocated tensor data");
            const int64_t actual = r3dh_gdn_extra_scratch(&args);
            expect(actual == scratch_expected(rows, heads), "runtime scratch byte count");
            auto planning = device;
            for (auto& tensor : planning) tensor.data = nullptr;
            auto planning_args = arguments(planning, params);
            expect(r3dh_gdn_extra_scratch(&planning_args) == actual, "runtime and planning scratch match");
            Buffer& scratch = fixture.add(RAD_I8, {actual});
            args.scratch = scratch.device().data; args.scratch_bytes = actual;
            expect(r3dh_gdn_extra_scratch(&args) == actual, "allocated scratch does not change size");
            fixture.guards();
            std::printf("PASS scratch_runtime M=%d heads=%d keys=%d tensors=18 data=allocated bytes=%lld planning_equal guards\n",
                rows, heads, heads / 3, static_cast<long long>(actual));
        }
    }
}
void identity() {
    const char* physical = std::getenv("ROCR_VISIBLE_DEVICES");
    expect(physical && (!std::strcmp(physical, "2") || !std::strcmp(physical, "1")
        || !std::strcmp(physical, "4")), "physical GPU1, GPU2 or GPU4 required");
    expect(std::getenv("HIP_VISIBLE_DEVICES") && !std::strcmp(std::getenv("HIP_VISIBLE_DEVICES"), "0"), "logical GPU0 required");
    int count = 0; hip_ok(hipGetDeviceCount(&count)); expect(count == 1, "one visible GPU");
    hip_ok(hipSetDevice(0)); hipDeviceProp_t property{}; hip_ok(hipGetDeviceProperties(&property, 0));
    expect(!std::strncmp(property.gcnArchName, "gfx1100", 7), "native gfx1100 required");
    char pci[64]{}; hip_ok(hipDeviceGetPCIBusId(pci, sizeof(pci), 0));
    if (!std::strcmp(physical, "4"))
        expect(!std::strcmp(pci, "0000:83:00.0"), "GPU4 PCI 83:00.0 required");
    std::printf("gpu physical=%s logical=0 arch=%s pci=%s contract=off tolerance=0\n", physical, property.gcnArchName, pci);
}
void decode(int heads, bool anchor, int bound, int sequences, int group, const char* act) {
    Fixture fixture; const int keys = heads / 3, channels = (keys * 2 + heads) * 128;
    const int rows = sequences == 1 ? bound : bound + 1, slots = sequences * 10;
    auto params = parameters(rows, heads, bound, anchor, group, act);
    Buffer& input = fixture.add(RAD_BF16, {rows, channels}, heads * 128 + 13);
    Buffer& weight = fixture.add(RAD_BF16, {channels, 4}); weight.fill(3, 0.19f);
    Buffer& bias = fixture.add(RAD_BF16, {channels}); bias.fill(4, 0.015f);
    Buffer& conv = fixture.add(RAD_BF16, {sequences + 1, channels, bound + 2}, 17); conv.fill(5, 0.11f);
    Buffer& conv_index = fixture.add(RAD_I32, {sequences});
    std::vector<int> conv_slots(sequences); for (int index = 0; index < sequences; ++index) conv_slots[index] = index + 1;
    conv_index.integers(conv_slots);
    Buffer& avalue = fixture.add(RAD_BF16, {rows, heads}, 7);
    Buffer& bvalue = fixture.add(RAD_BF16, {rows, heads}, 7);
    Buffer& alog = fixture.add(RAD_F32, {heads}); alog.fill(7, 0.3f);
    Buffer& dtbias = fixture.add(RAD_F32, {heads}); dtbias.fill(8, 0.09f);
    Buffer& cache = fixture.add(RAD_F32, {slots, heads, 128, 128}, 32); cache.fill(9, 0.001f);
    Buffer& state_index = fixture.add(RAD_I32, {sequences, anchor ? 6 : bound});
    std::vector<int> indices(sequences * (anchor ? 6 : bound));
    for (int sequence = 0; sequence < sequences; ++sequence)
        for (int column = 0; column < (anchor ? 6 : bound); ++column)
            indices[sequence * (anchor ? 6 : bound) + column] = sequence * 10 + column;
    state_index.integers(indices);
    if (anchor) {
        for (int sequence = 0; sequence < sequences; ++sequence)
            for (int slot = 3; slot <= 5; ++slot)
                for (int head = 0; head < heads; ++head)
                    std::memset(cache.reference.data() + ((sequence * 10 + slot) * cache.descriptor.stride[0]
                        + head * cache.descriptor.stride[1]) * 4, 0, 16384 * 4);
    }
    Buffer& accepted = fixture.add(RAD_I32, {sequences});
    accepted.integers(std::vector<int>(sequences, 1));
    Buffer& cumulative = fixture.add(RAD_I32, {sequences + 1});
    cumulative.integers(sequences == 1 ? std::vector<int>{0, rows} : std::vector<int>{0, bound, rows});
    Buffer& gate = fixture.add(RAD_BF16, {rows, heads * 128}, channels + 19);
    Buffer& norm = fixture.add(RAD_F32, {128}); norm.fill(11, 0.9f);
    Buffer& output = fixture.add(RAD_BF16, {rows, heads, 128}, 11);
    Buffer& codes = fixture.add(RAD_I8, {rows, heads * 128}, 13);
    Buffer& scales = fixture.add(RAD_F32, {rows, heads * 128 / group}, 3);
    std::vector<Buffer*> operands{&input, &weight, &bias, &conv, &conv_index, &avalue, &bvalue,
        &alog, &dtbias, &cache, &state_index, &accepted, &cumulative, &gate, &norm, &output, &codes, &scales};
    auto device = descriptors(operands, true), host = descriptors(operands, false);
    auto device_args = arguments(device, params), host_args = arguments(host, params);
    if (anchor) {
        accepted.integers(std::vector<int>(sequences, 0));
        expect(ref_gdn_conv_recurrent_update(&host_args, nullptr) == RAD_E_INVAL,
            "fused libref rejects zero accepted convolution offset");
        accepted.integers(std::vector<int>(sequences, 1));
        device[10].shape[1] = 5;
        expect(r3dh_gdn_extra_validate(&device_args, 3) == RAD_E_SHAPE, "reject short anchor index");
        device[10].shape[1] = 6;
        device[9].dtype = RAD_BF16;
        expect(r3dh_gdn_extra_validate(&device_args, 3) == RAD_E_SHAPE, "reject non-F32 anchor state");
        device[9].dtype = RAD_F32;
        std::printf("PASS anchor_contract M=%d seq=%d q_len=%d group=%d assertions=3 fused_min_accepted=1\n", rows, sequences, bound, group);
    }
    const int64_t scratch_bytes = r3dh_gdn_extra_scratch(&device_args); expect(scratch_bytes > 0, "scratch geometry");
    Buffer& scratch = fixture.add(RAD_I8, {scratch_bytes});
    device_args.scratch = scratch.device().data; device_args.scratch_bytes = scratch_bytes;
    fixture.upload();
    const int saved_depth = device[3].shape[2]; device[3].shape[2] = bound + 1;
    expect(r3dh_gdn_extra_validate(&device_args, 3) != RAD_OK, "reject shallow state depth"); device[3].shape[2] = saved_depth;
    device_args.scratch_bytes = scratch_bytes - 1;
    expect(r3dh_gdn_extra_conv_recurrent_update_had_strict(&device_args, reinterpret_cast<RadStream>(fixture.stream)) != RAD_OK, "reject short scratch");
    device_args.scratch_bytes = scratch_bytes;
    for (int step = 0; step < 3; ++step) {
        input.fill(13 + step * 7, 0.7f); avalue.fill(17 + step * 11, 0.8f);
        bvalue.fill(19 + step * 13, 0.9f); gate.fill(23 + step * 17, 1.1f);
        if (step) {
            std::vector<int> counts(sequences, 1); counts[0] = bound > 1 ? 2 : 1;
            accepted.integers(counts);
        }
        input.upload(fixture.stream); avalue.upload(fixture.stream); bvalue.upload(fixture.stream);
        gate.upload(fixture.stream); accepted.upload(fixture.stream);
        host[16] = RadTensor{}; host[17] = RadTensor{};
        rad_ok(ref_gdn_conv_recurrent_update(&host_args, nullptr));
        std::vector<RadTensor> quant_host{output.host(), codes.host(), scales.host()};
        quant_host[0].rank = 2;
        quant_host[0].shape[1] = heads * 128;
        quant_host[0].stride[1] = 1;
        quant_host[0].shape[2] = 0;
        auto quant_args = arguments(quant_host, params); rad_ok(ref_had_quant_act_i8g(&quant_args, nullptr));
        if (step == 1) hip_ok(hipStreamBeginCapture(fixture.stream, hipStreamCaptureModeThreadLocal));
        rad_ok(r3dh_gdn_extra_conv_recurrent_update_had_strict(&device_args, reinterpret_cast<RadStream>(fixture.stream)));
        if (step == 1) {
            hipGraph_t graph = nullptr; hipGraphExec_t executable = nullptr;
            hip_ok(hipStreamEndCapture(fixture.stream, &graph));
            hip_ok(hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
            hip_ok(hipGraphLaunch(executable, fixture.stream)); hip_ok(hipStreamSynchronize(fixture.stream));
            hip_ok(hipGraphExecDestroy(executable)); hip_ok(hipGraphDestroy(graph));
        } else hip_ok(hipStreamSynchronize(fixture.stream));
        conv.compare("conv_state"); cache.compare("recurrent_state_and_factors"); output.compare("bf16_output");
        codes.compare("row_major_integer_FWHT"); scales.compare("group_scales"); fixture.guards();
        std::printf("PASS decode heads=%d keys=%d dims=128 M=%d seq=%d q_len=%d depth=%d form=%s group=%d act=%s step=%d exact guards stream%s\n",
            heads, keys, rows, sequences, bound, saved_depth, anchor ? "anchor" : "per_candidate", group, act, step, step == 1 ? " graph" : "");
    }
}
void prefill(int heads, int rows, bool warm) {
    Fixture fixture; const int keys = heads / 3, channels = (2 * keys + heads) * 128;
    auto params = parameters(rows, heads, rows, false, 128, "sigmoid");
    Buffer& input = fixture.add(RAD_BF16, {rows, channels}, heads * 128 + 13); input.fill(31, 0.7f);
    Buffer& weight = fixture.add(RAD_BF16, {channels, 4}); weight.fill(32, 0.2f);
    Buffer& bias = fixture.add(RAD_BF16, {channels}); bias.fill(33, 0.01f);
    Buffer& alog = fixture.add(RAD_F32, {heads}); alog.fill(34, 0.15f);
    Buffer& dtbias = fixture.add(RAD_F32, {heads}); dtbias.fill(35, 0.07f);
    Buffer& conv = fixture.add(RAD_BF16, {3, channels, 3}, 17); conv.fill(36, 0.1f);
    Buffer& cumulative = fixture.add(RAD_I32, {2}); cumulative.integers({0, rows});
    Buffer& avalue = fixture.add(RAD_BF16, {rows, heads}, 7); avalue.fill(37, 0.8f);
    Buffer& bvalue = fixture.add(RAD_BF16, {rows, heads}, 7); bvalue.fill(38, 0.9f);
    Buffer& index = fixture.add(RAD_I32, {1}); index.integers({2});
    Buffer& initialized = fixture.add(RAD_I32, {1}); initialized.integers({warm ? 1 : 0});
    Buffer& query = fixture.add(RAD_BF16, {rows, keys, 128}, 9);
    Buffer& key = fixture.add(RAD_BF16, {rows, keys, 128}, 9);
    Buffer& value = fixture.add(RAD_BF16, {rows, heads, 128}, 9);
    Buffer& decay = fixture.add(RAD_F32, {rows, heads}, 5);
    Buffer& beta = fixture.add(RAD_F32, {rows, heads}, 5);
    Buffer& inverse = fixture.add(RAD_BF16, {rows, heads, 64}, 9);
    Buffer& initial = fixture.add(RAD_F32, {3, heads, 128, 128}, 32); initial.fill(39, 0.001f);
    Buffer& final = fixture.add(RAD_F32, {3, heads, 128, 128}, 32); final.fill(40, 0.001f);
    Buffer& output = fixture.add(RAD_BF16, {rows, heads, 128}, 9);
    fixture.upload();
    std::vector<Buffer*> prep{&input, &weight, &bias, &alog, &dtbias, &conv, &cumulative, &avalue, &bvalue, &index, &initialized, &query, &key, &value, &decay, &beta};
    auto prep_host = descriptors(prep, false), prep_device = descriptors(prep, true);
    auto prep_ref = arguments(prep_host, params), prep_gpu = arguments(prep_device, params);
    rad_ok(ref_gdn_conv_prep(&prep_ref, nullptr));
    rad_ok(r3dh_gdn_extra_conv_prep_strict(&prep_gpu, reinterpret_cast<RadStream>(fixture.stream)));
    hip_ok(hipStreamSynchronize(fixture.stream));
    for (Buffer* buffer : {&conv, &query, &key, &value, &decay, &beta}) buffer->compare("conv_prep");
    std::vector<Buffer*> solve{&key, &beta, &decay, &cumulative, &inverse};
    auto solve_host = descriptors(solve, false), solve_device = descriptors(solve, true);
    auto solve_ref = arguments(solve_host, params), solve_gpu = arguments(solve_device, params);
    solve_gpu.n_t = 4;
    expect(r3dh_gdn_extra_kkt_strict(&solve_gpu, reinterpret_cast<RadStream>(fixture.stream)) == RAD_E_INVAL, "reject missing KKT operand");
    solve_gpu.n_t = 5;
    rad_ok(ref_gdn_kkt_solve(&solve_ref, nullptr));
    rad_ok(r3dh_gdn_extra_kkt_strict(&solve_gpu, reinterpret_cast<RadStream>(fixture.stream)));
    hip_ok(hipStreamSynchronize(fixture.stream)); inverse.compare("kkt_BF16");
    std::vector<Buffer*> scan{&query, &key, &value, &inverse, &decay, &beta, &initial, &cumulative, &output, &final, &index};
    auto scan_host = descriptors(scan, false), scan_device = descriptors(scan, true);
    if (!warm) { scan_host[6] = RadTensor{}; scan_device[6] = RadTensor{}; }
    auto scan_ref = arguments(scan_host, params), scan_gpu = arguments(scan_device, params);
    scan_device[9].dtype = RAD_BF16;
    expect(r3dh_gdn_extra_chunk_strict(&scan_gpu, reinterpret_cast<RadStream>(fixture.stream)) == RAD_E_SHAPE, "reject non-F32 chunk state");
    scan_device[9].dtype = RAD_F32;
    scan_device[10].dtype = RAD_F32;
    expect(r3dh_gdn_extra_chunk_strict(&scan_gpu, reinterpret_cast<RadStream>(fixture.stream)) == RAD_E_SHAPE, "reject non-I32 chunk index");
    scan_device[10].dtype = RAD_I32;
    rad_ok(ref_gdn_chunk_scan(&scan_ref, nullptr));
    rad_ok(r3dh_gdn_extra_chunk_strict(&scan_gpu, reinterpret_cast<RadStream>(fixture.stream)));
    hip_ok(hipStreamSynchronize(fixture.stream)); output.compare("chunk_BF16"); final.compare("chunk_F32_state"); fixture.guards();
    std::printf("PASS chunk_contract M=%d heads=%d assertions=3 initial=%s\n", rows, heads, warm ? "F32" : "zero");
    std::printf("PASS prefill heads=%d keys=%d dims=128 M=%d chunk=64 warm=%d exact conv kkt scan state guards padded_slot\n", heads, keys, rows, warm);
}
#include "r3dh_gdn_extra_check_chunk.h"
#include "r3dh_gdn_extra_check_chunk_suite.h"
}
int main(int argc, char** argv) {
    try {
        if (argc == 2 && (!std::strcmp(argv[1], "--chunk-smoke") || !std::strcmp(argv[1], "--chunk")
                || !std::strcmp(argv[1], "--chunk-registry"))) {
            chunk_registry = !std::strcmp(argv[1], "--chunk-registry");
            chunk_identity();
            if (!std::strcmp(argv[1], "--chunk-smoke")) chunk_case({65, 66}, true, false);
            else chunk_suite();
            return 0;
        }
        expect(argc == 1 || (argc == 2 && !std::strcmp(argv[1], "--host")), "usage: r3dh_gdn_extra_check [--host]");
        scratch_host();
        if (argc == 2) return 0;
        identity();
        scratch_runtime();
        for (int group : {128, 256}) {
            decode(24, false, 1, 1, group, "sigmoid");
            decode(24, false, 3, 2, group, "silu");
            decode(24, true, 3, 2, group, "sigmoid");
            decode(24, true, 8, 1, group, "silu");
        }
        decode(48, false, 1, 1, 256, "sigmoid");
        for (int rows : {1, 63, 64, 65}) prefill(24, rows, rows != 63);
        prefill(48, 1, false);
        std::puts("PASS GDN_EXTRA all strict libref bit-exact checks; synthetic weights actual TP2/full head geometry; no full-model certification");
        return 0;
    } catch (const std::exception& error) { std::fprintf(stderr, "FAIL GDN_EXTRA %s\n", error.what()); return 1; }
}
