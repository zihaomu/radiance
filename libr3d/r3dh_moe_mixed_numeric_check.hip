#include <hip/hip_runtime.h>
#include "r3dh_moe_mixed.h"
#include "r3dh_moe_mixed_model.h"
#include <array>
#include <cmath>
#include <dlfcn.h>
#include <memory>

namespace {
void require(bool passed, const char* message) {
    if (!passed) throw std::runtime_error(message);
}
void hip_ok(hipError_t status) { require(status == hipSuccess, hipGetErrorString(status)); }
float unpack(uint16_t value) {
    const uint32_t bits = uint32_t(value) << 16;
    float result; std::memcpy(&result, &bits, sizeof(result)); return result;
}
uint16_t pack(float value) {
    uint32_t bits; std::memcpy(&bits, &value, sizeof(bits));
    uint32_t high = bits >> 16, low = bits & 65535u;
    if ((bits & 0x7fffffffu) > 0x7f800000u) return uint16_t(high | 64u);
    if (low > 32768u || (low == 32768u && (high & 1u))) ++high;
    return uint16_t(high);
}
RadTensor matrix(void* data, int rows, int columns) {
    RadTensor result{}; result.data = data; result.dtype = RAD_BF16; result.rank = 2;
    result.shape[0] = rows; result.shape[1] = columns;
    result.stride[0] = columns; result.stride[1] = 1; return result;
}
RadTensor vector(std::vector<int32_t>& data) {
    RadTensor result{}; result.data = data.data(); result.dtype = RAD_I32; result.rank = 1;
    result.shape[0] = data.size(); result.stride[0] = 1; return result;
}
std::vector<uint16_t> weight(Model& model, int layer, int expert, bool down, int rank) {
    const std::string wanted = "blk." + std::to_string(layer) + (down ? ".ffn_down_exps." : ".ffn_gate_up_exps.")
        + std::to_string(expert) + ".weight";
    for (const auto& entry : model.entries) {
        if (model.name(entry) != wanted) continue;
        require(entry.enc < model.encodings.size(), "BF16 encoding index");
        const auto& encoding = model.encodings[entry.enc];
        require(entry.rank == 2 && entry.shape[0] == (down ? 2560 : 1280)
            && entry.shape[1] == (down ? 640 : 2560) && entry.n_planes == 1
            && encoding.n_planes == 1 && encoding.plane[0].dtype == RAD_BF16
            && !encoding.transform[0] && entry.plane_first < model.planes.size(), "actual BF16 schema");
        const auto& plane = model.planes[entry.plane_first];
        require(plane.bytes == uint64_t(entry.shape[0] * entry.shape[1] * 2)
            && plane.offset >= entry.offset && plane.offset - entry.offset <= entry.bytes
            && plane.bytes <= entry.bytes - (plane.offset - entry.offset), "BF16 plane bounds");
        std::vector<uint16_t> full(plane.bytes / 2);
        model.read(plane.offset, full.data(), plane.bytes);
        std::vector<uint16_t> shard(size_t(down ? 2560 * 320 : 640 * 2560));
        if (down) {
            for (int row = 0; row < 2560; ++row)
                std::copy_n(full.data() + row * 640 + rank * 320, 320, shard.data() + row * 320);
        } else {
            for (int part = 0; part < 2; ++part)
                std::copy_n(full.data() + (part * 640 + rank * 320) * 2560,
                    320 * 2560, shard.data() + part * 320 * 2560);
        }
        return shard;
    }
    throw std::runtime_error("missing actual protected weight");
}
struct Device {
    void* pointer = nullptr;
    size_t bytes;
    explicit Device(size_t count) : bytes(count) { hip_ok(hipMalloc(&pointer, bytes + 512)); }
    ~Device() { if (pointer) static_cast<void>(hipFree(pointer)); }
    void* data() const { return static_cast<uint8_t*>(pointer) + 256; }
    void upload(const void* source, hipStream_t stream) {
        hip_ok(hipMemsetAsync(pointer, 0xa5, bytes + 512, stream));
        hip_ok(hipMemcpyAsync(data(), source, bytes, hipMemcpyHostToDevice, stream));
    }
    void read(void* output, hipStream_t stream) {
        std::array<uint8_t, 256> before{}, after{};
        hip_ok(hipMemcpyAsync(output, data(), bytes, hipMemcpyDeviceToHost, stream));
        hip_ok(hipMemcpyAsync(before.data(), pointer, 256, hipMemcpyDeviceToHost, stream));
        hip_ok(hipMemcpyAsync(after.data(), static_cast<uint8_t*>(data()) + bytes,
            256, hipMemcpyDeviceToHost, stream));
        hip_ok(hipStreamSynchronize(stream));
        require(std::all_of(before.begin(), before.end(), [](uint8_t value) { return value == 0xa5; })
            && std::all_of(after.begin(), after.end(), [](uint8_t value) { return value == 0xa5; }), "GPU guards");
    }
};
RadLaunchFn reference(const char* path) {
    void* plugin = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!plugin) throw std::runtime_error(dlerror());
    const auto count = reinterpret_cast<int(*)()>(dlsym(plugin, "rad_kernel_count"));
    const auto at = reinterpret_cast<const RadKernelInfo*(*)(int)>(dlsym(plugin, "rad_kernel_at"));
    require(count && at, "libref plugin ABI");
    for (int index = 0; index < count(); ++index) {
        const auto* kernel = at(index);
        if (kernel && !std::strcmp(kernel->op, "moe_gemm")) return kernel->launch;
    }
    throw std::runtime_error("libref moe_gemm unavailable");
}
uint64_t comparisons = 0, cases = 0, negatives = 0;
void run(Model& model, RadLaunchFn cpu, int layer, const std::vector<int>& experts,
         int rank, bool down, int repeat, bool gpu, int tokens) {
    const int rows = tokens * 10, columns = down ? 2560 : 640, inner = down ? 320 : 2560;
    const int count = int(experts.size()), begin = tokens == 1 ? 0 : 5;
    const int end = begin + count * 2;
    std::vector<std::vector<uint16_t>> weights;
    for (int expert : experts) weights.push_back(weight(model, layer, expert, down, rank));
    std::vector<const uint16_t*> table;
    for (auto& values : weights) table.push_back(values.data());
    std::vector<uint16_t> activation(size_t(down ? rows : tokens) * inner);
    for (size_t index = 0; index < activation.size(); ++index)
        activation[index] = pack(float(int((index * 17 + repeat * 29) % 103) - 51) / 64.0f);
    std::vector<int32_t> sorted(rows, -1), offsets(count + 1);
    for (int expert = 0; expert <= count; ++expert) offsets[expert] = begin + expert * 2;
    for (int row = begin; row < end; ++row) sorted[row] = (row * 7 + repeat * 3) % rows;
    sorted[begin + 1] = -1;
    const uint16_t sentinel = 0x42c6;
    std::vector<uint16_t> expected(size_t(rows) * columns, sentinel), output(expected), libref(expected);
    uint64_t wrong = 0, noop = 0;
    for (int row = begin; row < end; ++row) {
        const int expert = (row - begin) / 2;
        for (int column = 0; column < columns; ++column) {
            float sum = 0.0f, wrong_sum = 0.0f;
            if (sorted[row] >= 0) {
                const int source = down ? row : sorted[row] / 10;
                for (int reduction = 0; reduction < inner; ++reduction) {
                    const float value = unpack(activation[size_t(source) * inner + reduction]);
                    sum = std::fma(value, unpack(weights[expert][size_t(column) * inner + reduction]), sum);
                    wrong_sum = std::fma(value,
                        unpack(weights[(expert + 1) % count][size_t(column) * inner + reduction]), wrong_sum);
                }
            }
            expected[size_t(row) * columns + column] = pack(sum);
            wrong += pack(sum) != pack(wrong_sum); noop += pack(sum) != sentinel;
        }
    }
    require(wrong && noop, "wrong-expert/noop controls distinguish"); negatives += 2;
    std::vector<RadParam> params{RAD_INT("M", 64), RAD_INT("N", columns), RAD_INT("K", inner),
        RAD_INT("n_expert", count), RAD_INT("top_k", 10), RAD_STR("dtype", "bf16"),
        RAD_STR("a_order", down ? "sorted" : "token")};
    std::array<RadTensor, 5> tensors{matrix(activation.data(), down ? rows : tokens, inner), {},
        vector(sorted), vector(offsets), matrix(libref.data(), rows, columns)};
    tensors[1] = matrix(table.data(), columns, inner); tensors[1].rank = 3;
    tensors[1].shape[0] = count; tensors[1].shape[1] = columns; tensors[1].shape[2] = inner;
    tensors[1].stride[0] = 0; tensors[1].stride[1] = inner; tensors[1].stride[2] = 1;
    RadArgs args{}; args.t = tensors.data(); args.n_t = tensors.size();
    args.p = params.data(); args.n_p = params.size(); args.rank = rank; args.world_size = 2;
    args.kv_used_max_h = -1;
    require(cpu(&args, nullptr) == RAD_OK, "CPU libref launch");
    for (int row = begin; row < end; ++row) for (int column = 0; column < columns; ++column) {
        const size_t index = size_t(row) * columns + column;
        require(libref[index] == expected[index], "actual BF16 libref mismatch"); ++comparisons;
    }
    if (gpu) {
        hipStream_t stream = nullptr; hip_ok(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
        std::vector<std::unique_ptr<Device>> buffers;
        std::vector<const uint16_t*> pointers;
        for (auto& values : weights) {
            buffers.push_back(std::make_unique<Device>(values.size() * 2));
            buffers.back()->upload(values.data(), stream);
            pointers.push_back(static_cast<const uint16_t*>(buffers.back()->data()));
        }
        Device input(activation.size() * 2), pointers_device(pointers.size() * sizeof(void*));
        Device order_device(sorted.size() * 4), offsets_device(offsets.size() * 4), result(output.size() * 2);
        input.upload(activation.data(), stream); pointers_device.upload(pointers.data(), stream);
        order_device.upload(sorted.data(), stream); offsets_device.upload(offsets.data(), stream);
        result.upload(output.data(), stream);
        tensors[0].data = input.data(); tensors[1].data = pointers_device.data();
        tensors[2].data = order_device.data(); tensors[3].data = offsets_device.data(); tensors[4].data = result.data();
        require(r3dh_moe_mixed_bf16_launch(&args, reinterpret_cast<RadStream>(stream)) == RAD_OK, "GPU BF16 launch");
        result.read(output.data(), stream);
        require(output == expected, "GPU BF16 result/regular-row preservation"); comparisons += output.size();
        std::vector<uint16_t> unchanged(activation.size()); input.read(unchanged.data(), stream);
        require(unchanged == activation, "input changed");
        hip_ok(hipStreamDestroy(stream));
    }
    ++cases;
    std::printf("MIXED_BF16_CASE layer=%d rank=%d order=%s repeat=%d experts=%d M=%d band=64 libref=bitexact gpu=%d\n",
        layer, rank, down ? "sorted" : "token", repeat, count, tokens, int(gpu));
    std::printf("MIXED_PROTECTED_IDS layer=%d", layer);
    for (int expert : experts) std::printf(" expert=%d", expert);
    std::puts("");
}
#include "r3dh_moe_mixed_compact_check.h"
#include "r3dh_moe_mixed_down_check.h"
}
int main(int argc, char** argv) {
    try {
        require(argc == 4 || argc == 5, "usage: mixed_numeric MODEL LIBREF --host|--gpu [layer]");
        const bool gpu = !std::strcmp(argv[3], "--gpu");
        require(gpu || !std::strcmp(argv[3], "--host"), "unknown mode");
        if (gpu) {
            int count = 0; hip_ok(hipGetDeviceCount(&count)); require(count == 1, "one visible GPU");
            hipDeviceProp_t properties{}; hip_ok(hipGetDeviceProperties(&properties, 0));
            char pci[32]{}; hip_ok(hipDeviceGetPCIBusId(pci, sizeof(pci), 0));
            require(!std::strncmp(properties.gcnArchName, "gfx1100", 7), "gfx1100 device");
            const char* visible = std::getenv("ROCR_VISIBLE_DEVICES");
            require(visible && *visible && !std::strchr(visible, ','), "single physical GPU selection");
            std::printf("MIXED_GPU_ID physical=%s HIP=0 pci=%s arch=%s name=%s\n", visible, pci,
                properties.gcnArchName, properties.name);
        }
        Model model(argv[1]); const auto cpu = reference(argv[2]);
        const std::array<int, 3> layers{34, 44, 47};
        const std::array<std::vector<int>, 3> experts{{{407, 496}, {292, 350}, {122, 143, 399, 445}}};
        for (size_t layer = 0; layer < layers.size(); ++layer)
            for (int tokens : {1, 3, 63, 64})
            for (int rank : {0, 1}) for (bool down : {false, true}) for (int repeat : {0, 1})
                run(model, cpu, layers[layer], experts[layer], rank, down, repeat, gpu, tokens);
        std::printf("MIXED_BF16_PASS cases=%llu comparisons=%llu negative_controls=%llu gpu=%d\n",
            (unsigned long long)cases, (unsigned long long)comparisons, (unsigned long long)negatives, int(gpu));
        for (int layer : {34, 44, 47}) for (int rank : {0, 1}) for (int repeat : {0, 1})
            for (int tokens : {1, 3, 63, 64})
            { compact_run(model, layer, rank, repeat, gpu, tokens);
              compact_down_run(model, layer, rank, repeat, gpu, tokens); }
        std::printf("MIXED_ALL_PASS cases=%llu comparisons=%llu gpu=%d\n", (unsigned long long)cases, (unsigned long long)comparisons, int(gpu));
    } catch (const std::exception& error) { std::fprintf(stderr, "MIXED_FAIL %s\n", error.what()); return 1; }
}
