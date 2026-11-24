#include "rad_test.h"
#include "rad_abi.h"
#include "rad_device.h"
#include "rad_plugin.h"
#include "device/device.h"
#include "../libr3d/r3dh_i8.h"
#include "../libr3d/r3dh_hc_api.h"
#include "../libr3d/r3d_flags.h"
#include <cfenv>
#include <limits>
#include <dlfcn.h>
#include <memory>

// libr3d runs its bit-exact fast paths by default. These cases compare against the strict
// kernels and switch on the path under test themselves, so start from strict; a variable
// already set in the environment wins. The fused attention rows stay registered for their tests.
static const bool r3d_strict_defaults = [] {
    for (const char* name : r3d_exact_flags)
        if (std::strncmp(name, "RAD_ATTN_FUSE_", 14)) setenv(name, "0", 0);
    for (const char* name : r3d_tolerance_defaults) setenv(name, "0", 0);
    return true;
}();

namespace {
struct Library {
    void* handle;
    explicit Library(const char* name) {
        handle = dlopen((home() + "/kernels/" + name + ".so").c_str(),
                        RTLD_NOW | RTLD_LOCAL);
        if (!handle) throw std::runtime_error(dlerror());
    }
    ~Library() { dlclose(handle); }
    const RadKernelInfo* ngram(const char* operation = "ngram_ids") {
        auto count = reinterpret_cast<int (*)()>(dlsym(handle, "rad_kernel_count"));
        auto at = reinterpret_cast<const RadKernelInfo* (*)(int)>(dlsym(handle, "rad_kernel_at"));
        if (!count || !at) throw std::runtime_error("missing plugin enumeration");
        for (int index = 0; index < count(); ++index) {
            const auto* row = at(index);
            if (!std::strcmp(row->op, operation)) return row;
        }
        throw std::runtime_error("missing ngram_ids row");
    }
};
struct Queue {
    RadStream stream = nullptr;
    explicit Queue(int priority) { REQUIRE_EQ(rad_stream_create(&stream, priority), RAD_OK); }
    ~Queue() { rad_stream_destroy(stream); }
};
struct Buffer {
    std::vector<uint8_t> expected;
    void* allocation;
    int kind;
    RadTensor host{}, device{};
    Buffer(int dtype, std::initializer_list<int64_t> shape, int memory, int pitch = 0)
        : kind(memory) {
        host.dtype = dtype;
        host.rank = shape.size();
        int64_t elements = 1;
        std::copy(shape.begin(), shape.end(), host.shape);
        for (int dim = host.rank - 1; dim >= 0; --dim) {
            host.stride[dim] = elements;
            elements *= host.shape[dim];
        }
        if (pitch) {
            host.stride[0] = pitch;
            elements = host.shape[0] * pitch;
        }
        expected.assign(128 + elements * (dtype == RAD_I64 ? 8 : dtype == RAD_BF16 ? 2 : 4), 0x5a);
        host.data = expected.data() + 64;
        allocation = rad_dev_alloc(expected.size(), kind);
        REQUIRE(allocation);
        device = host;
        device.data = static_cast<uint8_t*>(allocation) + 64;
    }
    ~Buffer() { rad_dev_free(allocation, kind); }
    template <class Value> Value* data() { return static_cast<Value*>(host.data); }
    void push(RadStream stream) {
        REQUIRE_EQ(rad_memcpy_async(allocation, expected.data(), expected.size(), stream), RAD_OK);
    }
    std::vector<uint8_t> read(RadStream stream) {
        std::vector<uint8_t> bytes(expected.size());
        if (rad_memcpy_async(bytes.data(), allocation, bytes.size(), stream) != RAD_OK ||
            rad_stream_sync(stream) != RAD_OK)
            throw std::runtime_error("buffer read failed");
        return bytes;
    }
};
RadParam integer(const char* key, int64_t value) {
    RadParam param{};
    param.key = key;
    param.kind = RAD_P_INT;
    param.ival = value;
    return param;
}
}

TEST(r3d_mtp_ple_actual_aql_mapped_ngram) {
    RadDeviceProps props{};
    REQUIRE_EQ(rad_dev_props(0, &props), RAD_OK);
    if (props.is_host_backend) {
        REQUIRE(!std::getenv("RAD_R3D_REQUIRE_DEVICE"));
        SKIP("explicit gfx1100 card run required");
    }
    REQUIRE_EQ(std::strcmp(props.arch, "gfx1100"), 0);
    REQUIRE_EQ(rad_dev_set(0), RAD_OK);
    Library native("libr3d"), reference("libref");
    auto* row = native.ngram();
    auto* oracle = reference.ngram();
    REQUIRE_EQ(std::strcmp(row->name, "r3d_ngram_ids_i32"), 0);
    Queue compute(0), other(1);
    uint64_t compared = 0, launches = 0, rejects = 0;
    std::vector<int> counts;
    for (int count = 1; count <= 64; ++count) counts.push_back(count);
    for (int count : {127, 511, 2047, 2048}) counts.push_back(count);
    for (int memory : {RAD_MEM_HOST_MAPPED, RAD_MEM_HOST_PINNED})
        for (int count : counts) {
            RadStream stream = count % 2 ? compute.stream : other.stream;
            std::vector<std::unique_ptr<Buffer>> buffers;
            auto add = [&](int type, std::initializer_list<int64_t> shape, int kind, int pitch = 0) {
                buffers.push_back(std::make_unique<Buffer>(type, shape, kind, pitch));
            };
            add(RAD_I32, {count}, memory);
            add(RAD_I32, {3, 1, 2}, RAD_MEM_DEVICE);
            add(RAD_I32, {4}, memory);
            add(RAD_I64, {3}, RAD_MEM_DEVICE);
            add(RAD_I64, {16}, RAD_MEM_DEVICE);
            add(RAD_I64, {16}, RAD_MEM_DEVICE);
            add(RAD_I32, {3}, memory);
            add(RAD_I32, {3}, memory);
            add(RAD_I32, {3}, memory);
            add(RAD_I32, {count, 16}, memory, 21);
            auto ints = [&](int index) { return buffers[index]->data<int32_t>(); };
            auto longs = [&](int index) { return buffers[index]->data<int64_t>(); };
            for (int index = 0; index < 6; ++index) ints(1)[index] = index * 7919;
            const int32_t cumulative[]{0, count / 2, count / 2, count};
            std::memcpy(ints(2), cumulative, sizeof cumulative);
            longs(3)[0] = 24000000000001ll;
            longs(3)[1] = -51000000000007ll;
            longs(3)[2] = INT64_MAX - 2;
            for (int head = 0; head < 16; ++head) {
                longs(4)[head] = 101 + head * 2;
                longs(5)[head] = head * 200;
            }
            for (int sequence = 0; sequence < 3; ++sequence) {
                ints(6)[sequence] = 2 - sequence;
                ints(7)[sequence] = 1;
                ints(8)[sequence] = 1;
            }
            std::vector<RadTensor> host, device;
            for (auto& buffer : buffers) {
                host.push_back(buffer->host);
                device.push_back(buffer->device);
            }
            host.push_back({});
            device.push_back({});
            host[8] = {};
            device[8] = {};
            std::vector<RadParam> params{integer("M", 2048), integer("heads", 16),
                                         integer("ngram", 3), integer("eos", 248044)};
            RadArgs args{};
            args.t = device.data(); args.n_t = device.size();
            args.p = params.data(); args.n_p = params.size();
            for (int step = 0; step < 4; ++step) {
                for (int token = 0; token < count; ++token)
                    ints(0)[token] = (token + step) % 7 == 0 ? 248044 :
                        int32_t((uint32_t(token * 7919) ^ uint32_t((step + 1) * 2246822519u)) % 248000);
                for (int sequence = 0; sequence < 3; ++sequence)
                    ints(7)[sequence] = step == 0 ? sequence % 2 : 1;
                std::fill(buffers[9]->expected.begin() + 64,
                          buffers[9]->expected.end() - 64, uint8_t(0xa5));
                for (size_t index = 0; index < buffers.size(); ++index)
                    if (index != 1 || step == 0) buffers[index]->push(stream);
                for (int repeat = 0; repeat < 2; ++repeat) {
                    args.t = host.data();
                    REQUIRE_EQ(oracle->launch(&args, nullptr), RAD_OK);
                    args.t = device.data();
                    REQUIRE_EQ(row->launch(&args, stream), RAD_OK);
                    ++launches;
                }
                REQUIRE_EQ(rad_stream_sync(stream), RAD_OK);
                for (auto& buffer : buffers) {
                    auto actual = buffer->read(stream);
                    REQUIRE(actual == buffer->expected);
                    compared += actual.size();
                }
            }
        }
    RadStream closed = nullptr;
    REQUIRE_EQ(rad_stream_create(&closed, 0), RAD_OK);
    rad_stream_destroy(closed);
    std::fprintf(stderr, "AQL_NGRAM_PASS fixtures=136 band=2048 launches=%llu rejects=%llu compared_bytes=%llu\n",
                 (unsigned long long)launches, (unsigned long long)rejects,
                 (unsigned long long)compared);
}

TEST(r3d_mtp_ple_actual_aql_shared_validator_and_gate) {
    RadDeviceProps props{};
    REQUIRE_EQ(rad_dev_props(0, &props), RAD_OK);
    if (props.is_host_backend) {
        REQUIRE(!std::getenv("RAD_R3D_REQUIRE_DEVICE"));
        SKIP("explicit gfx1100 card run required");
    }
    REQUIRE_EQ(rad_dev_set(0), RAD_OK);
    Library native("libr3d"), reference("libref");
    using Launch = int (*)(const RadArgs*, RadStream);
    auto validate = reinterpret_cast<Launch>(dlsym(native.handle, "r3dh_mtp_ple_device"));
    auto gate = reinterpret_cast<Launch>(dlsym(native.handle, "r3dh_mtp_ple_ple_gate"));
    REQUIRE(validate && gate);
    const auto* oracle = reference.ngram("ple_gate");
    Queue queue(1);
    std::vector<std::unique_ptr<Buffer>> buffers;
    for (int index = 0; index < 8; ++index) {
        const bool weight = index >= 3 && index <= 5;
        buffers.push_back(std::make_unique<Buffer>(weight ? RAD_F32 : RAD_BF16,
            weight ? std::initializer_list<int64_t>{16} :
                     std::initializer_list<int64_t>{3, index == 2 ? 8 : 16},
            weight ? RAD_MEM_DEVICE : RAD_MEM_HOST_MAPPED));
    }
    std::vector<RadTensor> host, device;
    for (auto& buffer : buffers) {
        host.push_back(buffer->host);
        device.push_back(buffer->device);
    }
    std::vector<RadParam> params{integer("M", 3), integer("n", 8), integer("hc", 2)};
    RadArgs args{};
    args.t = device.data(); args.n_t = device.size();
    args.p = params.data(); args.n_p = params.size();
    args.scratch = device[0].data;
    REQUIRE_EQ(validate(&args, queue.stream), RAD_OK);
    args.scratch = device[3].data;
    REQUIRE_EQ(validate(&args, queue.stream), RAD_OK);
    args.scratch = nullptr;
    const auto* backend = rad::active_backend();
    REQUIRE(backend->vmem_granularity && backend->vmem_reserve && backend->vmem_commit &&
            backend->vmem_decommit && backend->vmem_release);
    const int64_t granule = backend->vmem_granularity();
    REQUIRE(granule > 0);
    void* mapping = backend->vmem_reserve(granule, granule);
    REQUIRE(mapping);
    REQUIRE_EQ(backend->vmem_commit(mapping, granule), RAD_OK);
    args.scratch = mapping;
    CHECK_EQ(validate(&args, queue.stream), RAD_OK);
    args.scratch = nullptr;
    REQUIRE_EQ(backend->vmem_decommit(mapping, granule), RAD_OK);
    backend->vmem_release(mapping, granule);
    double maximum = 0;
    for (int step = 0; step < 4; ++step) {
        for (size_t index = 0; index < buffers.size(); ++index) {
            auto& buffer = *buffers[index];
            for (int64_t element = 0; element < rad_tensor_numel(&buffer.host); ++element) {
                if (buffer.host.dtype == RAD_F32)
                    buffer.data<float>()[element] = float(1 + element % 3) / 4;
                else
                    buffer.data<uint16_t>()[element] = uint16_t(0x3e80 + ((element + step) % 7) * 64);
            }
            buffer.push(queue.stream);
        }
        args.t = host.data();
        REQUIRE_EQ(oracle->launch(&args, nullptr), RAD_OK);
        args.t = device.data();
        REQUIRE_EQ(gate(&args, queue.stream), RAD_OK);
        REQUIRE_EQ(rad_stream_sync(queue.stream), RAD_OK);
        for (size_t index = 0; index < buffers.size(); ++index) {
            auto actual = buffers[index]->read(queue.stream);
            auto expected = buffers[index]->expected;
            if (index >= 6) {
                for (int64_t element = 0; element < rad_tensor_numel(&host[index]); ++element) {
                    uint16_t value = 0;
                    std::memcpy(&value, actual.data() + 64 + element * 2, 2);
                    double error = std::abs(double(rad_bf16_to_f32(value)) -
                        rad_bf16_to_f32(buffers[index]->data<uint16_t>()[element]));
                    REQUIRE(std::isfinite(error));
                    maximum = std::max(maximum, error);
                    std::memcpy(expected.data() + 64 + element * 2, &value, 2);
                }
            }
            CHECK(actual == expected);
        }
    }
    CHECK(maximum <= 0.015);
    std::fprintf(stderr, "AQL_PLE_GATE_PASS launches=4 max_abs=%.9g\n", maximum);
}

static void r3d_bf16_aql_gemm(bool dense, bool router = false) {
    RadDeviceProps props{};
    REQUIRE_EQ(rad_dev_props(0, &props), RAD_OK);
    if (props.is_host_backend) {
        REQUIRE(!std::getenv("RAD_R3D_REQUIRE_DEVICE"));
        SKIP("explicit gfx1100 card run required");
    }
    REQUIRE_EQ(std::strcmp(props.arch, "gfx1100"), 0);
    REQUIRE_EQ(rad_dev_set(0), RAD_OK);
    Library native("libr3d"), reference("libref");
    using Launch = int (*)(const RadArgs*, RadStream);
    auto gemm = reinterpret_cast<Launch>(dlsym(native.handle, "r3d_rad_gemm_nt_bf16_tiled"));
    auto logits = reinterpret_cast<Launch>(dlsym(native.handle, "r3d_rad_logits_gemm_bf16"));
    REQUIRE(gemm && logits);
    if (router) {
        auto count = reinterpret_cast<int (*)()>(dlsym(native.handle, "rad_kernel_count"));
        auto at = reinterpret_cast<const RadKernelInfo* (*)(int)>(dlsym(native.handle, "rad_kernel_at"));
        REQUIRE(count && at);
        gemm = nullptr;
        for (int index = 0; index < count(); ++index) {
            const auto* row = at(index);
            if (row && !std::strcmp(row->name, "model_router_gemm_strict")) gemm = row->launch;
        }
        REQUIRE(gemm);
    }
    Queue compute(0), other(1);
    const auto* backend = rad::active_backend();
    REQUIRE(backend->vmem_granularity && backend->vmem_reserve && backend->vmem_commit_units &&
            backend->vmem_decommit && backend->vmem_release);
    struct Mapping {
        const rad::DeviceBackend* backend;
        void* address = nullptr;
        int64_t bytes = 0;
        bool committed = false;
        ~Mapping() {
            if (committed) backend->vmem_decommit(address, bytes);
            if (address) backend->vmem_release(address, bytes);
        }
    };
    uint64_t fixtures = 0, launches = 0, compared = 0, rejects = 0;
    for (bool virtual_weights : {false, true})
        for (bool f32_output : {false, true})
            for (int rows : {1, 3, 63, 64})
                for (int columns : {48, 96, 1, 512}) {
                    if (router && (f32_output || (columns != 1 && columns != 512))) continue;
                    constexpr int inner = 2560;
                    const int x_pitch = inner + 16;
                    const int y_pitch = ((columns + 7) / 8 + 1) * 8;
                    Buffer activation(RAD_BF16, {rows, inner}, RAD_MEM_DEVICE, x_pitch);
                    Buffer weight(RAD_BF16, {columns, inner}, RAD_MEM_DEVICE);
                    Buffer output(f32_output ? RAD_F32 : RAD_BF16,
                                  {rows, columns}, RAD_MEM_DEVICE, y_pitch);
                    Mapping mapping{backend};
                    if (virtual_weights) {
                        const int64_t granule = backend->vmem_granularity();
                        REQUIRE(granule > 0);
                        mapping.bytes = (int64_t(weight.expected.size()) + granule - 1) /
                                        granule * granule;
                        mapping.address = backend->vmem_reserve(mapping.bytes, granule);
                        REQUIRE(mapping.address);
                        REQUIRE(backend->vmem_commit_units(mapping.address, mapping.bytes,
                                                           granule) >= 0);
                        mapping.committed = true;
                        weight.device.data = static_cast<uint8_t*>(mapping.address) + 64;
                    }
                    RadTensor host[]{activation.host, weight.host, output.host};
                    RadTensor device[]{activation.device, weight.device, output.device};
                    std::vector<RadParam> params{integer("M", rows <= 16 ? 16 : 64),
                        integer(f32_output ? "n_vocab" : "N", columns),
                        integer(f32_output ? "n_embd" : "K", inner), integer("impl", 0)};
                    RadArgs args{};
                    args.t = device; args.n_t = 3;
                    args.p = params.data(); args.n_p = params.size();
                    const auto* oracle = reference.ngram(f32_output ? "logits_gemm" : "gemm_nt");
                    Launch launch = f32_output ? logits : gemm;
                    RadStream stream = rows % 2 ? other.stream : compute.stream;
                    for (int step = 0; step < 2; ++step) {
                        for (int row = 0; row < rows; ++row)
                            for (int index = 0; index < inner; ++index) {
                                const uint32_t mixed = uint32_t(index + row * 3571 + step * 65537) *
                                                       2246822519u;
                                activation.data<uint16_t>()[row * x_pitch + index] =
                                    rad_f32_to_bf16(dense ? float(int((mixed >> 19) % 33) - 16) / 32 :
                                        float((row % 3 + 1) * (step ? -2 : 1)) / 8);
                            }
                        for (int column = 0; column < columns; ++column)
                            for (int index = 0; index < inner; ++index) {
                                const uint32_t mixed = uint32_t(index + column * 7919) * 3266489917u;
                                weight.data<uint16_t>()[column * inner + index] =
                                    rad_f32_to_bf16(dense ? float(int((mixed >> 17) % 17) - 8) / 16 :
                                        float(column % 3 + 1));
                            }
                        std::fill(output.expected.begin(), output.expected.end(), uint8_t(0x5a));
                        activation.push(stream);
                        output.push(stream);
                        if (virtual_weights)
                            REQUIRE_EQ(rad_memcpy_async(mapping.address, weight.expected.data(),
                                weight.expected.size(), stream), RAD_OK);
                        else weight.push(stream);
                        args.t = host;
                        REQUIRE_EQ(oracle->launch(&args, nullptr), RAD_OK);
                        args.t = device;
                        REQUIRE_EQ(launch(&args, stream), RAD_OK);
                        ++launches;
                        REQUIRE_EQ(rad_stream_sync(stream), RAD_OK);
                        auto actual = output.read(stream);
                        uint64_t differences = 0;
                        for (size_t byte = 0; byte < actual.size(); ++byte)
                            differences += actual[byte] != output.expected[byte];
                        std::fprintf(stderr,
                            "AQL_BF16 memory=%s op=%s M=%d band=%lld N=%d K=%d step=%d diff_bytes=%llu\n",
                            virtual_weights ? "VMEM" : "DEVICE", f32_output ? "logits_gemm" : "gemm_nt",
                            rows, (long long)params[0].ival, columns, inner, step,
                            (unsigned long long)differences);
                        if (differences) {
                            int printed = 0;
                            for (int row = 0; row < rows && printed < 4; ++row)
                                for (int column = 0; column < columns && printed < 4; ++column) {
                                    const size_t offset = 64 + (row * y_pitch + column) *
                                                          (f32_output ? 4 : 2);
                                    if (!std::memcmp(actual.data() + offset,
                                            output.expected.data() + offset, f32_output ? 4 : 2)) continue;
                                    double dot = 0;
                                    for (int index = 0; index < inner; ++index)
                                        dot += double(rad_bf16_to_f32(activation.data<uint16_t>()[
                                            row * x_pitch + index])) * rad_bf16_to_f32(
                                            weight.data<uint16_t>()[column * inner + index]);
                                    uint32_t got = 0, want = 0;
                                    std::memcpy(&got, actual.data() + offset, f32_output ? 4 : 2);
                                    std::memcpy(&want, output.expected.data() + offset, f32_output ? 4 : 2);
                                    std::fprintf(stderr, "AQL_BF16_DIFF row=%d column=%d gpu=0x%x ref=0x%x dot=%.17g\n",
                                                 row, column, got, want, dot);
                                    ++printed;
                                }
                        }
                        CHECK_EQ(differences, uint64_t(0));
                        auto guarded = output.expected;
                        for (int row = 0; row < rows; ++row)
                            for (int column = 0; column < columns; ++column) {
                                const size_t offset = 64 + (row * y_pitch + column) *
                                                      (f32_output ? 4 : 2);
                                std::memcpy(guarded.data() + offset, actual.data() + offset,
                                            f32_output ? 4 : 2);
                            }
                        CHECK(actual == guarded);
                        CHECK(activation.read(stream) == activation.expected);
                        if (virtual_weights) {
                            std::vector<uint8_t> actual_weight(weight.expected.size());
                            REQUIRE_EQ(rad_memcpy_async(actual_weight.data(), mapping.address,
                                actual_weight.size(), stream), RAD_OK);
                            REQUIRE_EQ(rad_stream_sync(stream), RAD_OK);
                            CHECK(actual_weight == weight.expected);
                        } else CHECK(weight.read(stream) == weight.expected);
                        compared += actual.size();
                    }
                    const auto before = output.read(stream);
                    if (router) {
                        for (int invalid_band : {rows - 1, 2049}) {
                            params[0].ival = invalid_band;
                            CHECK_EQ(launch(&args, stream), RAD_E_SHAPE);
                            ++rejects;
                        }
                        params[0].ival = rows <= 16 ? 16 : 64;
                    }
                    for (uint32_t dtype : {uint32_t(RAD_F32), uint32_t(RAD_F16),
                            uint32_t(RAD_I8), uint32_t(RAD_F8E4M3), uint32_t(0x7fffffffu)}) {
                        device[1].dtype = dtype;
                        device[1].data = reinterpret_cast<void*>(uintptr_t(0x123450000000ull));
                        CHECK_EQ(launch(&args, stream), RAD_E_DTYPE);
                        ++rejects;
                    }
                    device[1] = weight.device;
                    CHECK(output.read(stream) == before);
                    ++fixtures;
                }
    std::fprintf(stderr,
        "AQL_BF16_SUMMARY fixture=%s fixtures=%llu launches=%llu rejects=%llu compared_output_guard_bytes=%llu\n",
        router ? "router_band_exact" : dense ? "dense_exact" : "representable_exact",
        (unsigned long long)fixtures, (unsigned long long)launches,
        (unsigned long long)rejects, (unsigned long long)compared);
}

TEST(r3d_bf16_actual_aql_device_and_vmem_gemm) {
    r3d_bf16_aql_gemm(false);
}

TEST(r3d_bf16_actual_aql_device_and_vmem_gemm_dense_exact) {
    r3d_bf16_aql_gemm(true);
}

TEST(r3d_i8_quick_dot4_exact_bench) {
    if (!std::getenv("RAD_I8_BENCH")) SKIP("optional dense int8 decode benchmark");
    Library native("libr3d");
    auto launch = reinterpret_cast<int (*)(const RadArgs*, RadStream)>(dlsym(native.handle, "r3dh_i8_rad_gemm_nt_q"));
    REQUIRE(launch);
    Queue queue(0);
    const char* previous = std::getenv("RAD_I8_DOT4");
    const bool present = previous != nullptr;
    const std::string saved = previous ? previous : "";
    constexpr int columns = 2560;
    for (int reduction : {2560, 10240}) {
        const int groups = reduction / 128;
        Buffer weight(RAD_I8, {columns, reduction}, RAD_MEM_DEVICE);
        Buffer weight_scale(RAD_F32, {columns, groups}, RAD_MEM_DEVICE);
        for (int column = 0; column < columns; ++column) {
            for (int position = 0; position < reduction; ++position)
                weight.data<int8_t>()[r3dh_i8::packed_index(column, position, reduction)]
                    = int8_t(int((uint32_t(column * 3571 + position) * 2654435761u) % 256) - 128);
            for (int group = 0; group < groups; ++group)
                weight_scale.data<float>()[column * groups + group] = float((column + group) % 17 + 1) / 8192.0f;
        }
        weight.push(queue.stream); weight_scale.push(queue.stream);
        for (int rows : {1, 2, 4}) {
            Buffer activation(RAD_I8, {rows, reduction}, RAD_MEM_DEVICE, reduction + 16);
            Buffer activation_scale(RAD_F32, {rows, groups}, RAD_MEM_DEVICE, groups + 4);
            Buffer output(RAD_BF16, {rows, columns}, RAD_MEM_DEVICE, columns + 16);
            for (int row = 0; row < rows; ++row) {
                for (int position = 0; position < reduction; ++position)
                    activation.data<int8_t>()[row * (reduction + 16) + position]
                        = int8_t(int((uint32_t(row * 71 + position) * 2246822519u) % 256) - 128);
                for (int group = 0; group < groups; ++group)
                    activation_scale.data<float>()[row * (groups + 4) + group] = float((row + group) % 13 + 1) / 4096.0f;
            }
            activation.push(queue.stream); activation_scale.push(queue.stream); output.push(queue.stream);
            RadTensor tensors[]{activation.device, activation_scale.device, weight.device, weight_scale.device, output.device};
            tensors[2].dtype = r3dh_i8::weight_dtype; tensors[3].dtype = r3dh_i8::scale_dtype;
            RadParam params[]{RAD_INT("M", 16), RAD_INT("N", columns), RAD_INT("K", reduction),
                RAD_INT("group", 128), RAD_STR("dtype", "i8a8h")};
            RadArgs args{}; args.t = tensors; args.n_t = 5; args.p = params; args.n_p = 5;
            std::vector<uint8_t> baseline;
            float timings[2]{};
            for (int variant = 0; variant < 2; ++variant) {
                setenv("RAD_I8_DOT4", variant ? "1" : "0", 1);
                REQUIRE_EQ(launch(&args, queue.stream), RAD_OK);
                auto actual = output.read(queue.stream);
                if (variant) CHECK(actual == baseline); else baseline = std::move(actual);
                RadEvent start{}, finish{};
                REQUIRE_EQ(rad_event_create(&start), RAD_OK); REQUIRE_EQ(rad_event_create(&finish), RAD_OK);
                for (int repeat = 0; repeat < 3; ++repeat) REQUIRE_EQ(launch(&args, queue.stream), RAD_OK);
                REQUIRE_EQ(rad_event_record(start, queue.stream), RAD_OK);
                for (int repeat = 0; repeat < 20; ++repeat) REQUIRE_EQ(launch(&args, queue.stream), RAD_OK);
                REQUIRE_EQ(rad_event_record(finish, queue.stream), RAD_OK); REQUIRE_EQ(rad_event_sync(finish), RAD_OK);
                REQUIRE_EQ(rad_event_elapsed_ms(start, finish, &timings[variant]), RAD_OK);
                rad_event_destroy(start); rad_event_destroy(finish);
            }
            std::printf("I8_DOT4_BENCH M=%d N=%d K=%d baseline_us=%.3f dot4_us=%.3f speedup=%.3f exact=1\n",
                rows, columns, reduction, timings[0] * 50.0f, timings[1] * 50.0f, timings[0] / timings[1]);
        }
    }
    if (present) setenv("RAD_I8_DOT4", saved.c_str(), 1); else unsetenv("RAD_I8_DOT4");
}

TEST(r3d_hc_quick_wmma_candidate) {
    const bool packed_down = std::getenv("RAD_HC_PACKED_BENCH") != nullptr;
    const bool inject_parallel = std::getenv("RAD_HC_INJECT_BENCH") != nullptr;
    const bool norm_parallel = std::getenv("RAD_HC_NORM_BENCH") != nullptr || packed_down;
    const bool exact_tile = std::getenv("RAD_HC_EXACT_BENCH") != nullptr;
    if (!std::getenv("RAD_HC_BENCH") && !exact_tile && !norm_parallel && !inject_parallel) SKIP("optional HC matrix candidate");
    const char* override_switch = std::getenv("RAD_HC_SWITCH_BENCH");
    const char* switch_name = override_switch ? override_switch : packed_down ? "RAD_HC_DOWN_PACKED" : inject_parallel ? "RAD_HC_INJECT_PARALLEL"
        : norm_parallel ? "RAD_HC_NORM_PARALLEL" : exact_tile ? "RAD_HC_EXACT_TILE" : "RAD_HC_WMMA";
    Library native("libr3d");
    using Launch = int (*)(const RadArgs*, RadStream, const R3dhHcTrace*);
    auto launch = reinterpret_cast<Launch>(dlsym(native.handle, "r3dh_hc_rad_hc_read_strict_trace"));
    auto plain = reinterpret_cast<int (*)(const RadArgs*, RadStream)>(dlsym(native.handle, "r3dh_hc_rad_hc_read_strict"));
    auto scratch_size = reinterpret_cast<int64_t (*)(const RadArgs*)>(dlsym(native.handle, "r3dh_hc_rad_hc_read_strict_scratch"));
    REQUIRE(launch && plain && scratch_size);
    auto profile = reinterpret_cast<int (*)(const RadArgs*, RadStream, float*)>(dlsym(native.handle, "r3dh_hc_rad_hc_read_profile"));
    Queue queue(0);
    Buffer down(RAD_F8E4M3, {320, 10560}, RAD_MEM_DEVICE);
    Buffer up(RAD_F8E4M3, {10240, 336}, RAD_MEM_DEVICE);
    for (Buffer* matrix : {&down, &up}) {
        const int depth = matrix == &down ? 10240 : 320;
        const int group = matrix == &down ? 128 : 80;
        for (int row = 0; row < matrix->host.shape[0]; ++row) {
            uint8_t* line = matrix->data<uint8_t>() + row * matrix->host.stride[0];
            for (int position = 0; position < depth; ++position) {
                const uint32_t hash = uint32_t(row * 3571 + position) * 2654435761u;
                line[position] = uint8_t(24 + (hash % 80)) | uint8_t((hash >> 17) & 128);
            }
            for (int block = 0; block < depth / group; ++block) {
                const float scale = (1.0f + float((row * 13 + block) % 17) / 32.0f) * 0x1p-10f;
                std::memcpy(line + depth + block * 4, &scale, 4);
            }
        }
        matrix->push(queue.stream);
    }
    const char* previous = std::getenv(switch_name);
    const bool present = previous != nullptr;
    const std::string saved = previous ? previous : "";
    for (int pattern = 0; pattern < (inject_parallel ? 15 : norm_parallel ? 9 : 1); ++pattern)
    for (int rows : {1, 2, 3, 4, 5, 8, 16, 17}) {
        if (!packed_down && rows != 1 && rows != 4 && rows != 16 && rows != 17) continue;
        if (rows == 17 && (!inject_parallel || pattern)) continue;
        Buffer hidden(RAD_BF16, {rows, 10240}, RAD_MEM_DEVICE);
        Buffer gain(RAD_F32, {10240}, RAD_MEM_DEVICE);
        Buffer output(RAD_BF16, {rows, 2560}, RAD_MEM_DEVICE);
        Buffer traced_down(RAD_BF16, {rows, 320}, RAD_MEM_DEVICE);
        Buffer traced_logits(RAD_F32, {rows, 10240}, RAD_MEM_DEVICE);
        Buffer traced_gates(RAD_BF16, {rows, 10240}, RAD_MEM_DEVICE);
        Buffer inject_weights(RAD_BF16, {4, 10240}, RAD_MEM_DEVICE);
        Buffer inject_output(RAD_BF16, {rows, 4}, RAD_MEM_DEVICE);
        Buffer quantized(RAD_I8, {rows, 2560}, RAD_MEM_DEVICE);
        Buffer quant_scales(RAD_F32, {rows, 20}, RAD_MEM_DEVICE);
        Buffer rotated(RAD_I8, {rows, 2560}, RAD_MEM_DEVICE);
        Buffer rotated_scales(RAD_F32, {rows, 20}, RAD_MEM_DEVICE);
        for (int index = 0; index < rows * 10240; ++index)
            hidden.data<uint16_t>()[index] = uint16_t(0x3e00u | ((uint32_t(index) * 2246822519u) & 0x83ffu));
        if (pattern) {
            for (int index = 0; index < rows * 10240; ++index) {
                const uint16_t sign = index & 1 ? 0x8000 : 0;
                int exponent = pattern == 1 ? 120 + index % 13
                    : pattern == 2 ? 120 + index % 14
                    : pattern == 3 ? index % 14 : index % 2 ? 100 : 150;
                uint16_t bits = sign | uint16_t(exponent << 7) | uint16_t((index * 37) & 127);
                if (pattern == 5) bits = sign;
                if (pattern == 6 && index % 2560 == 0) bits = 0x7f80;
                if (pattern == 6 && index % 2560 == 1) bits = 0x7fc1;
                if (pattern >= 7) {
                    const int position = index % 2560;
                    bits = position < 5 ? 0x3f80 : position < 10 ? 0x3980 : 0;
                    if (position == 10) bits = pattern == 7 ? 0x2b80 : 0x3780;
                }
                if (inject_parallel && pattern >= 9) bits = 0x3f80;
                hidden.data<uint16_t>()[index] = bits;
            }
        }
        for (int index = 0; index < 10240; ++index) gain.data<float>()[index] = 0.75f + float(index % 17) / 32.0f;
        for (int index = 0; index < 40960; ++index)
            inject_weights.data<uint16_t>()[index] = uint16_t(0x3900u | ((uint32_t(index) * 37u) & 0x807fu));
        if (inject_parallel && pattern >= 9) {
            for (int index = 0; index < 10240; ++index) gain.data<float>()[index] = 1.0f;
            for (int channel_stream = 0; channel_stream < 4; ++channel_stream) {
                for (int position = 0; position < 10240; ++position) {
                    uint16_t bits = 0;
                    if (pattern == 9) bits = position % 3 == 0 ? 0x4b80 : position % 3 == 1 ? 0x3f80 : 0xcb80;
                    if (pattern == 10) bits = channel_stream & 1 ? 0xbf80 : 0x3f80;
                    if (pattern == 11) {
                        if (position == 0) bits = 0x3d00;
                        if (position == 1) bits = channel_stream & 1 ? 0xb500 : 0x3500;
                    }
                    if (pattern == 12) bits = uint16_t((position % 127 + 1) | ((position & 1) << 15));
                    if (pattern == 13 && position == 0) bits = channel_stream & 1 ? 0x7fc1 : 0x7f80;
                    if (pattern == 14) bits = position & 1 ? 0x8000 : 0;
                    inject_weights.data<uint16_t>()[channel_stream * 10240 + position] = bits;
                }
            }
        }
        for (Buffer* buffer : {&hidden, &gain, &output, &traced_down, &traced_logits, &traced_gates,
                &inject_weights, &inject_output, &quantized, &quant_scales, &rotated, &rotated_scales})
            buffer->push(queue.stream);
        RadTensor tensors[11]{};
        tensors[0] = hidden.device; tensors[1] = gain.device;
        tensors[2] = down.device; tensors[2].shape[1] = 10240; tensors[2].stride[0] = 10240;
        tensors[3] = up.device; tensors[3].shape[1] = 320; tensors[3].stride[0] = 320;
        tensors[5] = output.device;
        tensors[4] = inject_weights.device; tensors[6] = inject_output.device;
        tensors[7] = quantized.device; tensors[8] = quant_scales.device;
        tensors[9] = rotated.device; tensors[10] = rotated_scales.device;
        RadParam params[]{RAD_INT("M", std::max(rows, 16)), RAD_INT("n", 2560), RAD_INT("hc", 4),
            RAD_INT("lowrank", 320), RAD_STR("dtype", "bf16"), RAD_STR("mix", "e4m3"),
            RAD_INT("inject", 1), RAD_INT("group", 128), RAD_INT("rotate", 1)};
        RadArgs args{}; args.t = tensors; args.n_t = 11; args.p = params; args.n_p = 9;
        const int64_t bytes = scratch_size(&args);
        REQUIRE(bytes > 0);
        Buffer scratch(RAD_I8, {bytes}, RAD_MEM_DEVICE);
        args.scratch = scratch.allocation; args.scratch_bytes = bytes;
        R3dhHcTrace trace{static_cast<uint16_t*>(traced_down.device.data), static_cast<float*>(traced_logits.device.data),
            static_cast<uint16_t*>(traced_gates.device.data), rows, 2560, 4, 320};
        float timings[2]{};
        std::vector<std::vector<uint8_t>> baseline;
        std::vector<uint8_t> normalized_baseline;
        for (int variant = 0; variant < 2; ++variant) {
            setenv(switch_name, variant ? "1" : "0", 1);
            REQUIRE_EQ(launch(&args, queue.stream, &trace), RAD_OK);
            if (norm_parallel || inject_parallel) {
                auto actual = scratch.read(queue.stream);
                actual.resize(size_t(rows) * 10240 * 2);
                if (variant) CHECK(actual == normalized_baseline); else normalized_baseline = std::move(actual);
            }
            if (!pattern && (!variant || norm_parallel || inject_parallel) && std::getenv("RAD_HC_STAGE_BENCH")) {
                REQUIRE(profile);
                float stages[6]{};
                REQUIRE_EQ(profile(&args, queue.stream, stages), RAD_OK);
                std::printf("HC_STAGES M=%d variant=%d norm_us=%.3f down_us=%.3f up_us=%.3f mix_us=%.3f inject_us=%.3f quant_us=%.3f\n",
                    rows, variant, stages[0], stages[1], stages[2], stages[3], stages[4], stages[5]);
            }
            int stage = 0;
            for (Buffer* buffer : {&traced_down, &traced_logits, &traced_gates, &output,
                    &inject_output, &quantized, &quant_scales, &rotated, &rotated_scales}) {
                auto actual = buffer->read(queue.stream);
                const int element_bytes = buffer->host.dtype == RAD_F32 ? 4 : buffer->host.dtype == RAD_I8 ? 1 : 2;
                const size_t end = 64 + size_t(buffer->host.shape[0] * buffer->host.shape[1]) * element_bytes;
                CHECK(std::equal(actual.begin(), actual.begin() + 64, buffer->expected.begin()));
                CHECK(std::equal(actual.begin() + end, actual.end(), buffer->expected.begin() + end));
                if (!variant) baseline.push_back(actual);
                else {
                    if (exact_tile || norm_parallel || inject_parallel) CHECK(actual == baseline[stage]);
                    float maximum = 0, reference_max = 0;
                    const int elements = int(buffer->host.shape[0] * buffer->host.shape[1]);
                    for (int index = 0; index < elements; ++index) {
                        float value = 0, expected = 0;
                        if (buffer->host.dtype == RAD_I8) {
                            value = float(int8_t(actual[64 + index]));
                            expected = float(int8_t(baseline[stage][64 + index]));
                        } else if (buffer->host.dtype == RAD_F32) {
                            std::memcpy(&value, actual.data() + 64 + index * 4, 4);
                            std::memcpy(&expected, baseline[stage].data() + 64 + index * 4, 4);
                        } else {
                            uint16_t bits = 0; uint32_t wide_bits = 0;
                            std::memcpy(&bits, actual.data() + 64 + index * 2, 2);
                            wide_bits = uint32_t(bits) << 16; std::memcpy(&value, &wide_bits, 4);
                            std::memcpy(&bits, baseline[stage].data() + 64 + index * 2, 2);
                            wide_bits = uint32_t(bits) << 16; std::memcpy(&expected, &wide_bits, 4);
                        }
                        if ((norm_parallel || inject_parallel) && (pattern == 6 || pattern == 13) && !std::isfinite(expected)) continue;
                        REQUIRE(std::isfinite(value));
                        maximum = std::max(maximum, std::abs(value - expected));
                        reference_max = std::max(reference_max, std::abs(expected));
                    }
                    std::printf("HC_ERROR M=%d pattern=%d stage=%d max_abs=%g peak_relative=%g\n", rows, pattern, stage, maximum,
                        maximum / std::max(reference_max, 1e-8f));
                    CHECK(maximum <= 0.02f * std::max(reference_max, 1e-4f));
                }
                ++stage;
            }
            RadEvent start{}, finish{};
            REQUIRE_EQ(rad_event_create(&start), RAD_OK); REQUIRE_EQ(rad_event_create(&finish), RAD_OK);
            for (int repeat = 0; repeat < 3; ++repeat) REQUIRE_EQ(plain(&args, queue.stream), RAD_OK);
            REQUIRE_EQ(rad_event_record(start, queue.stream), RAD_OK);
            for (int repeat = 0; repeat < (pattern ? 1 : 20); ++repeat) REQUIRE_EQ(plain(&args, queue.stream), RAD_OK);
            REQUIRE_EQ(rad_event_record(finish, queue.stream), RAD_OK); REQUIRE_EQ(rad_event_sync(finish), RAD_OK);
            REQUIRE_EQ(rad_event_elapsed_ms(start, finish, &timings[variant]), RAD_OK);
            rad_event_destroy(start); rad_event_destroy(finish);
        }
        if (pattern) std::printf("HC_%s_BOUNDARY M=%d pattern=%d exact=1\n",
            packed_down ? "DOWN_PACKED" : inject_parallel ? "INJECT" : "NORM", rows, pattern);
        else std::printf("HC_%s_BENCH M=%d strict_us=%.3f candidate_us=%.3f speedup=%.3f exact=%d\n",
            packed_down ? "DOWN_PACKED" : inject_parallel ? "INJECT_PARALLEL" : norm_parallel ? "NORM_PARALLEL" : exact_tile ? "EXACT_TILE" : "WMMA", rows,
            timings[0] * 50.0f, timings[1] * 50.0f, timings[0] / timings[1], int(exact_tile || norm_parallel || inject_parallel));
    }
    if (present) setenv(switch_name, saved.c_str(), 1); else unsetenv(switch_name);
}

TEST(r3d_logits_quick_staged_exact_bench) {
    if (!std::getenv("RAD_LOGITS_BENCH")) SKIP("optional actual-shard logits benchmark");
    Library native("libr3d");
    using Launch = int (*)(const RadArgs*, RadStream);
    auto launch = reinterpret_cast<Launch>(dlsym(native.handle, "r3d_rad_logits_gemm_bf16"));
    REQUIRE(launch);
    Queue queue(0);
    constexpr int columns = 124160, inner = 2560;
    Buffer weight(RAD_BF16, {columns, inner}, RAD_MEM_DEVICE);
    for (int64_t index = 0; index < int64_t(columns) * inner; ++index)
        weight.data<uint16_t>()[index] = uint16_t(0x3c00u | ((uint32_t(index) * 2654435761u) & 0x83ffu));
    weight.push(queue.stream);
    const char* previous = std::getenv("RAD_LOGITS_STAGED");
    const bool present = previous != nullptr;
    const std::string saved = previous ? previous : "";
    for (int rows : {1, 4, 16}) {
        Buffer activation(RAD_BF16, {rows, inner}, RAD_MEM_DEVICE, inner + 16);
        Buffer output(RAD_F32, {rows, columns}, RAD_MEM_DEVICE, columns + 8);
        for (int row = 0; row < rows; ++row)
            for (int index = 0; index < inner; ++index)
                activation.data<uint16_t>()[row * (inner + 16) + index]
                    = uint16_t(0x3d00u | ((uint32_t(index + row * 3571) * 2246822519u) & 0x83ffu));
        activation.push(queue.stream);
        output.push(queue.stream);
        RadTensor tensors[]{activation.device, weight.device, output.device};
        std::vector<RadParam> params{integer("M", 16), integer("n_vocab", columns), integer("n_embd", inner)};
        RadArgs args{}; args.t = tensors; args.n_t = 3; args.p = params.data(); args.n_p = params.size();
        std::vector<uint8_t> expected;
        float timings[2]{};
        for (int variant = 0; variant < 2; ++variant) {
            REQUIRE_EQ(setenv("RAD_LOGITS_STAGED", variant ? "1" : "0", 1), 0);
            REQUIRE_EQ(launch(&args, queue.stream), RAD_OK);
            auto actual = output.read(queue.stream);
            if (variant) CHECK(actual == expected); else expected = std::move(actual);
            RadEvent start{}, finish{};
            REQUIRE_EQ(rad_event_create(&start), RAD_OK);
            REQUIRE_EQ(rad_event_create(&finish), RAD_OK);
            for (int warmup = 0; warmup < 3; ++warmup) REQUIRE_EQ(launch(&args, queue.stream), RAD_OK);
            REQUIRE_EQ(rad_event_record(start, queue.stream), RAD_OK);
            for (int repeat = 0; repeat < 20; ++repeat) REQUIRE_EQ(launch(&args, queue.stream), RAD_OK);
            REQUIRE_EQ(rad_event_record(finish, queue.stream), RAD_OK);
            REQUIRE_EQ(rad_event_sync(finish), RAD_OK);
            REQUIRE_EQ(rad_event_elapsed_ms(start, finish, &timings[variant]), RAD_OK);
            rad_event_destroy(start); rad_event_destroy(finish);
        }
        std::printf("LOGITS_STAGED_BENCH M=%d N=%d K=%d strict_us=%.3f staged_us=%.3f speedup=%.3f exact=1\n",
            rows, columns, inner, timings[0] * 50.0f, timings[1] * 50.0f, timings[0] / timings[1]);
    }
    if (present) setenv("RAD_LOGITS_STAGED", saved.c_str(), 1); else unsetenv("RAD_LOGITS_STAGED");
}

TEST(r3d_router_actual_aql_ragged_band) {
    r3d_bf16_aql_gemm(true, true);
}

TEST(r3d_tp2_topk_scatter_exact) {
    RadDeviceProps props{};
    REQUIRE_EQ(rad_dev_props(0, &props), RAD_OK);
    if (props.is_host_backend || std::strcmp(props.arch, "gfx1100")) SKIP("gfx1100 required");
    REQUIRE_EQ(rad_dev_set(0), RAD_OK);
    Library native("libr3d");
    auto launch = reinterpret_cast<RadLaunchFn>(dlsym(native.handle, "r3d_rad_router_topk_scatter"));
    REQUIRE(launch);
    struct RestoreFlag {
        const char* name = "RAD_TP2_TOPK_SCATTER";
        bool present = std::getenv(name) != nullptr;
        std::string value = present ? std::getenv(name) : "";
        ~RestoreFlag() { if (present) setenv(name, value.c_str(), 1); else unsetenv(name); }
    } flag;
    Queue queue(0);
    constexpr int experts = 512, kept = 10;
    for (const int rows : {1, 2}) for (const bool protect : {false, true}) {
        Buffer logits(RAD_BF16, {rows, experts}, RAD_MEM_DEVICE);
        Buffer ids(RAD_I32, {rows, kept}, RAD_MEM_DEVICE), weights(RAD_BF16, {rows, kept}, RAD_MEM_DEVICE);
        Buffer sorted(RAD_I32, {rows * kept}, RAD_MEM_DEVICE), offsets(RAD_I32, {experts + 1}, RAD_MEM_DEVICE);
        Buffer counts(RAD_I32, {experts}, RAD_MEM_DEVICE);
        std::fill_n(logits.data<uint16_t>(), rows * experts, uint16_t(0x3f80));
        logits.push(queue.stream);
        RadTensor tensors[]{logits.device, ids.device, weights.device, sorted.device, offsets.device, counts.device};
        RadParam params[]{RAD_INT("M", rows == 1 ? 64 : rows), RAD_INT("n_expert", experts), RAD_INT("n_local", experts),
                          RAD_INT("top_k", kept), RAD_INT("expert_base", 0), RAD_INT("norm", 1), RAD_STR("protect", protect ? "7,9" : "")};
        RadArgs args{}; args.t = tensors; args.n_t = 6; args.p = params; args.n_p = 7; args.world_size = 2;
        for (const char* setting : {"0", "1"}) {
            REQUIRE_EQ(setenv(flag.name, setting, 1), 0);
            for (Buffer* buffer : {&ids, &weights, &sorted, &offsets, &counts}) {
                std::fill(buffer->expected.begin(), buffer->expected.end(), uint8_t(0x5a));
                buffer->push(queue.stream);
            }
            REQUIRE_EQ(launch(&args, queue.stream), RAD_OK);
            std::fill_n(counts.data<int32_t>(), experts, 0);
            std::vector<std::pair<int, int>> ordering;
            for (int slot = 0; slot < rows * kept; ++slot) {
                const int id = slot % kept;
                ids.data<int32_t>()[slot] = id;
                weights.data<uint16_t>()[slot] = rad_f32_to_bf16(0.1f);
                const int place = !protect ? id : id == 7 ? 510 : id == 9 ? 511 : id - (id > 7);
                ++counts.data<int32_t>()[place];
                ordering.emplace_back(place, slot);
            }
            std::sort(ordering.begin(), ordering.end());
            for (size_t index = 0; index < ordering.size(); ++index) sorted.data<int32_t>()[index] = ordering[index].second;
            offsets.data<int32_t>()[0] = 0;
            for (int index = 0; index < experts; ++index)
                offsets.data<int32_t>()[index + 1] = offsets.data<int32_t>()[index] + counts.data<int32_t>()[index];
            for (Buffer* buffer : {&logits, &ids, &weights, &sorted, &offsets, &counts})
                REQUIRE(buffer->read(queue.stream) == buffer->expected);
        }
    }
}

TEST(r3d_bf16_actual_aql_joined_layout_compatibility) {
    RadDeviceProps props{};
    REQUIRE_EQ(rad_dev_props(0, &props), RAD_OK);
    if (props.is_host_backend) {
        REQUIRE(!std::getenv("RAD_R3D_REQUIRE_DEVICE"));
        SKIP("explicit gfx1100 card run required");
    }
    REQUIRE_EQ(std::strcmp(props.arch, "gfx1100"), 0);
    REQUIRE_EQ(rad_dev_set(0), RAD_OK);
    Library native("libr3d"), reference("libref");
    using Launch = int (*)(const RadArgs*, RadStream);
    auto launch = reinterpret_cast<Launch>(dlsym(native.handle, "r3d_rad_logits_gemm_bf16"));
    auto layout = reinterpret_cast<RadLayoutFn>(dlsym(native.handle, "r3d_layout_logits_head"));
    auto relayout = reinterpret_cast<RadRelayoutFn>(dlsym(native.handle, "r3d_relayout_logits_head"));
    REQUIRE(launch && layout && relayout);
    const auto* oracle = reference.ngram("logits_gemm");
    Queue queue(1);
    uint64_t launches = 0;
    for (uint32_t dtype : {uint32_t(RAD_I8), uint32_t(RAD_F8E4M3)}) {
        Buffer codes(dtype, {8, 128}, RAD_MEM_HOST_MAPPED);
        Buffer scale(RAD_F32, {1, 1}, RAD_MEM_HOST_MAPPED);
        std::fill_n(codes.data<uint8_t>(), 8 * 128, dtype == RAD_I8 ? 1 : 0x38);
        scale.data<float>()[0] = 1.0f;
        RadTensor planes[]{codes.host, scale.host};
        const auto encoding = rad_enc_affine(dtype, RAD_F32, 128, 128);
        const int selected[]{0, 1};
        RadLayout stored{};
        REQUIRE_EQ(layout(nullptr, 0, 1, &encoding, selected, planes, 2, &stored), RAD_OK);
        REQUIRE(stored.dtype != RAD_BF16 && stored.dtype != dtype);
        CHECK_EQ(std::strcmp(stored.tag, "rad.planes"), 0);
        Buffer joined(RAD_I8, {stored.bytes}, RAD_MEM_HOST_MAPPED);
        REQUIRE_EQ(relayout(nullptr, 0, 1, &encoding, selected, planes, 2,
                            joined.host.data, stored.bytes), RAD_OK);
        joined.device.dtype = stored.dtype;
        joined.host.dtype = stored.dtype;
        Buffer activation(RAD_BF16, {3, 128}, RAD_MEM_DEVICE, 144);
        Buffer output(RAD_F32, {3, 8}, RAD_MEM_DEVICE, 16);
        RadTensor host[]{activation.host, joined.host, output.host};
        RadTensor device[]{activation.device, joined.device, output.device};
        RadParam params[]{integer("M", 16), integer("n_vocab", 8),
                          integer("n_embd", 128), integer("impl", 0)};
        RadArgs args{};
        args.t = device; args.n_t = 3; args.p = params; args.n_p = 4;
        for (int step = 0; step < 2; ++step) {
            for (int row = 0; row < 3; ++row)
                for (int index = 0; index < 128; ++index)
                    activation.data<uint16_t>()[row * 144 + index] =
                        rad_f32_to_bf16(step ? -0.25f : 0.125f);
            std::fill(output.expected.begin(), output.expected.end(), uint8_t(0x5a));
            activation.push(queue.stream);
            joined.push(queue.stream);
            output.push(queue.stream);
            args.t = host;
            REQUIRE_EQ(oracle->launch(&args, nullptr), RAD_OK);
            args.t = device;
            REQUIRE_EQ(launch(&args, queue.stream), RAD_OK);
            REQUIRE_EQ(rad_stream_sync(queue.stream), RAD_OK);
            CHECK(output.read(queue.stream) == output.expected);
            CHECK(activation.read(queue.stream) == activation.expected);
            CHECK(joined.read(queue.stream) == joined.expected);
            ++launches;
        }
        const auto before = output.read(queue.stream);
        std::memset(joined.host.data, 0, 4);
        joined.push(queue.stream);
        REQUIRE_EQ(rad_stream_sync(queue.stream), RAD_OK);
        CHECK_EQ(launch(&args, queue.stream), RAD_E_FORMAT);
        CHECK(output.read(queue.stream) == before);
    }
    std::fprintf(stderr, "AQL_BF16_JOINED_SUMMARY launches=%llu codes=I8,E4M3 exact_assertions=1\n",
                 (unsigned long long)launches);
}

TEST(r3d_bf16_inaccessible_pointer_format_refusals) {
    Library native("libr3d");
    using Launch = int (*)(const RadArgs*, RadStream);
    for (const char* symbol : {"r3d_rad_gemm_nt_bf16_tiled", "r3d_rad_logits_gemm_bf16"}) {
        auto launch = reinterpret_cast<Launch>(dlsym(native.handle, symbol));
        REQUIRE(launch);
        RadTensor tensors[3]{};
        for (auto& tensor : tensors) tensor.data = reinterpret_cast<void*>(uintptr_t(0x123450000000ull));
        tensors[0].dtype = RAD_I32;
        RadArgs args{};
        RadParam impl = integer("impl", 0);
        args.t = tensors; args.n_t = 3; args.p = &impl; args.n_p = 1;
        for (uint32_t dtype : {uint32_t(RAD_BF16), uint32_t(RAD_I8),
                              uint32_t(RAD_F8E4M3), uint32_t(0x7fffffffu)}) {
            tensors[1].dtype = dtype;
            CHECK_EQ(launch(&args, nullptr), RAD_E_DTYPE);
        }
    }
}

RAD_TEST_MAIN()

TEST(r3d_gdn_actual_aql_compact_conv_prep) {
    RadDeviceProps props{};
    REQUIRE_EQ(rad_dev_props(0, &props), RAD_OK);
    if (props.is_host_backend) {
        REQUIRE(!std::getenv("RAD_R3D_REQUIRE_DEVICE"));
        SKIP("explicit gfx1100 card run required");
    }
    REQUIRE_EQ(std::strcmp(props.arch, "gfx1100"), 0);
    REQUIRE_EQ(rad_dev_set(0), RAD_OK);
    Library native("libr3d"), reference("libref");
    auto count = reinterpret_cast<int (*)()>(dlsym(native.handle, "rad_kernel_count"));
    auto at = reinterpret_cast<const RadKernelInfo* (*)(int)>(dlsym(native.handle, "rad_kernel_at"));
    REQUIRE(count && at);
    const RadKernelInfo* row = nullptr;
    for (int index = 0; index < count(); ++index) {
        const auto* candidate = at(index);
        if (!std::strcmp(candidate->name, "model_gdn_conv_prep_strict")) {
            row = candidate;
            break;
        }
    }
    REQUIRE(row);
    const auto* oracle = reference.ngram("gdn_conv_prep");
    REQUIRE_EQ(std::strcmp(row->name, "model_gdn_conv_prep_strict"), 0);
    std::printf("GDN_AQL rows native=%s reference=%s\n", row->name, oracle->name);
    Queue compute(0), other(1);
    uint64_t fixtures = 0, launches = 0, compared = 0, rejects = 0;
    for (int rows : {1, 63, 64, 65})
        for (int sequences : {1, 4}) {
            std::vector<std::unique_ptr<Buffer>> buffers;
            auto add = [&](int type, std::initializer_list<int64_t> shape, int pitch = 0) {
                buffers.push_back(std::make_unique<Buffer>(type, shape, RAD_MEM_DEVICE, pitch));
            };
            add(RAD_BF16, {rows, 5120}, 8192);
            add(RAD_BF16, {5120, 4});
            add(RAD_F32, {24});
            add(RAD_F32, {24});
            add(RAD_BF16, {sequences, 5120, 3});
            add(RAD_I32, {sequences + 1});
            add(RAD_BF16, {rows, 48});
            add(RAD_I32, {sequences});
            add(RAD_I32, {sequences});
            add(RAD_BF16, {rows, 8, 128});
            add(RAD_BF16, {rows, 8, 128});
            add(RAD_BF16, {rows, 24, 128});
            add(RAD_F32, {rows, 24}, 32);
            add(RAD_F32, {rows, 24}, 32);
            auto fill_bf16 = [&](int index, int count, int pitch, int width, int seed) {
                auto* destination = buffers[index]->data<uint16_t>();
                for (int token = 0; token < count; ++token)
                    for (int column = 0; column < width; ++column) {
                        const float value = float((token * 7 + column * 13 + seed) % 23 - 11) / 32.0f;
                        uint32_t bits;
                        std::memcpy(&bits, &value, sizeof bits);
                        destination[token * pitch + column] = uint16_t(bits >> 16);
                    }
            };
            fill_bf16(4, sequences, 5120 * 3, 5120 * 3, 19);
            auto* cumulative = buffers[5]->data<int32_t>();
            cumulative[0] = 0;
            if (sequences == 1) cumulative[1] = rows;
            else {
                cumulative[1] = rows > 1 ? 1 : 0;
                cumulative[2] = cumulative[1];
                cumulative[3] = rows / 2;
                cumulative[4] = rows;
            }
            for (int sequence = 0; sequence < sequences; ++sequence)
                buffers[7]->data<int32_t>()[sequence] = sequences - sequence - 1;
            RadParam params[]{integer("M", rows <= 64 ? 64 : 128),
                integer("head_k", 128), integer("head_v", 128), integer("n_head_v", 24),
                integer("conv_width", 4), integer("chunk", 64), integer("group", 128)};
            RadArgs args{};
            args.p = params; args.n_p = int(sizeof params / sizeof params[0]); args.n_t = 16;
            for (int step = 0; step < 3; ++step) {
                RadStream stream = step % 2 ? compute.stream : other.stream;
                fill_bf16(0, rows, 8192, 5120, step * 29 + 3);
                fill_bf16(1, 5120, 4, 4, step * 31 + 5);
                fill_bf16(6, rows, 48, 48, step * 37 + 7);
                for (int head = 0; head < 24; ++head) {
                    buffers[2]->data<float>()[head] = float(head % 7 - 3 + step) / 16.0f;
                    buffers[3]->data<float>()[head] = float(head % 5 - 2 - step) / 32.0f;
                }
                for (int sequence = 0; sequence < sequences; ++sequence)
                    buffers[8]->data<int32_t>()[sequence] = step != 0;
                std::vector<RadTensor> host, device;
                for (auto& buffer : buffers) {
                    host.push_back(buffer->host); device.push_back(buffer->device);
                }
                auto operands = [&](const std::vector<RadTensor>& source) {
                    std::vector<RadTensor> result{source[0], source[1], RadTensor{}, source[2],
                        source[3], source[4], source[5], source[6], source[6], source[7],
                        source[8], source[9], source[10], source[11], source[12], source[13]};
                    result[7].shape[1] = result[8].shape[1] = 24;
                    result[8].data = static_cast<uint16_t*>(result[8].data) + 24;
                    return result;
                };
                auto host_args = operands(host), device_args = operands(device);
                for (int index = 0; index < int(buffers.size()); ++index)
                    if (index != 4 || step == 0) buffers[index]->push(stream);
                args.t = host_args.data();
                REQUIRE_EQ(oracle->launch(&args, nullptr), RAD_OK);
                args.t = device_args.data();
                REQUIRE_EQ(row->launch(&args, stream), RAD_OK);
                REQUIRE_EQ(rad_stream_sync(stream), RAD_OK);
                ++launches;
                for (auto& buffer : buffers) {
                    REQUIRE(buffer->read(stream) == buffer->expected);
                    compared += buffer->expected.size();
                }
                const auto before = buffers[9]->read(stream);
                device_args[0].stride[0] = (int64_t(1) << 36) + 1;
                CHECK_EQ(row->launch(&args, stream), RAD_E_SHAPE);
                CHECK(buffers[9]->read(stream) == before);
                ++rejects;
            }
            ++fixtures;
        }
    std::printf("PASS GDN_AQL fixtures=%llu launches=%llu rejects=%llu exact_bytes=%llu queues=2\n",
        (unsigned long long)fixtures, (unsigned long long)launches,
        (unsigned long long)rejects, (unsigned long long)compared);
}

TEST(r3d_gdn_actual_aql_compact_kkt_chunk) {
    RadDeviceProps props{};
    REQUIRE_EQ(rad_dev_props(0, &props), RAD_OK);
    if (props.is_host_backend) {
        REQUIRE(!std::getenv("RAD_R3D_REQUIRE_DEVICE"));
        SKIP("explicit gfx1100 card run required");
    }
    REQUIRE_EQ(std::strcmp(props.arch, "gfx1100"), 0);
    REQUIRE_EQ(rad_dev_count(), 1);
    REQUIRE_EQ(rad_dev_set(0), RAD_OK);
    Library native("libr3d"), reference("libref");
    auto count = reinterpret_cast<int (*)()>(dlsym(native.handle, "rad_kernel_count"));
    auto at = reinterpret_cast<const RadKernelInfo* (*)(int)>(dlsym(native.handle, "rad_kernel_at"));
    REQUIRE(count && at);
    const RadKernelInfo* solve = nullptr;
    const RadKernelInfo* scan = nullptr;
    for (int index = 0; index < count(); ++index) {
        const auto* candidate = at(index);
        if (!std::strcmp(candidate->name, "model_gdn_kkt_strict")) solve = candidate;
        if (!std::strcmp(candidate->name, "model_gdn_chunk_strict")) scan = candidate;
    }
    REQUIRE(solve && scan);
    REQUIRE_EQ(solve->domain, RAD_DOMAIN_DEVICE);
    REQUIRE_EQ(scan->domain, RAD_DOMAIN_DEVICE);
    const auto* solve_ref = reference.ngram("gdn_kkt_solve");
    const auto* scan_ref = reference.ngram("gdn_chunk_scan");
    Queue compute(0), other(1);
    REQUIRE(compute.stream && other.stream && compute.stream != other.stream);
    uint64_t fixtures = 0, pipelines = 0, compared = 0, rejects = 0;
    const std::vector<std::vector<int>> batches{{1}, {3}, {63}, {64}, {65}, {128},
        {65, 63}, {0, 65}, {0, 1, 63, 64, 65, 127, 128, 1},
        {0, 63, 0, 64, 0, 65, 0, 1}};
    for (const auto& lengths : batches)
        for (int padded : {0, 1})
            for (int mode : {0, 1, 2}) {
                const int sequences = int(lengths.size()), slots = sequences + 2;
                std::vector<int32_t> cumulative{0};
                for (int length : lengths) cumulative.push_back(cumulative.back() + length);
                const int rows = cumulative.back();
                std::vector<std::unique_ptr<Buffer>> buffers;
                auto add = [&](int type, std::initializer_list<int64_t> shape, int pitch = 0) {
                    buffers.push_back(std::make_unique<Buffer>(type, shape, RAD_MEM_DEVICE, pitch));
                };
                auto plane = [&](int heads, int width) {
                    const int stored = width + padded * 8;
                    add(RAD_BF16, {rows, heads, stored}, heads * stored + padded * 16);
                    buffers.back()->host.shape[2] = buffers.back()->device.shape[2] = width;
                };
                auto state = [&] {
                    const int stored = 128 + padded;
                    add(RAD_F32, {slots, 24, stored, 128}, 24 * stored * 128 + padded * 32);
                    buffers.back()->host.shape[2] = buffers.back()->device.shape[2] = 128;
                };
                plane(8, 128);
                plane(8, 128);
                plane(24, 128);
                plane(24, 64);
                add(RAD_F32, {rows, 24}, 24 + padded * 8);
                add(RAD_F32, {rows, 24}, 24 + padded * 16);
                state();
                add(RAD_I32, {sequences + 1});
                plane(24, 128);
                state();
                if (padded) add(RAD_I32, {sequences, 3}, 5);
                else add(RAD_I32, {sequences});
                std::memcpy(buffers[7]->host.data, cumulative.data(), cumulative.size() * sizeof(int32_t));
                for (int sequence = 0; sequence < sequences; ++sequence)
                    buffers[10]->data<int32_t>()[sequence * buffers[10]->host.stride[0]] =
                        padded && sequences == 8 && sequence == 7 ? -1 : sequences - sequence;
                auto fill_plane = [&](int operand, int seed, float amplitude) {
                    const auto& tensor = buffers[operand]->host;
                    for (int token = 0; token < rows; ++token)
                        for (int head = 0; head < tensor.shape[1]; ++head)
                            for (int column = 0; column < tensor.shape[2]; ++column)
                                buffers[operand]->data<uint16_t>()[token * tensor.stride[0]
                                    + head * tensor.stride[1] + column] = rad_f32_to_bf16(
                                        float((token * 7 + head * 13 + column * 11 + seed) % 31 - 15) * amplitude);
                };
                auto fill_state = [&](int operand, int seed) {
                    const auto& tensor = buffers[operand]->host;
                    for (int slot = 0; slot < slots; ++slot)
                        for (int head = 0; head < 24; ++head)
                            for (int column = 0; column < 16384; ++column)
                                buffers[operand]->data<float>()[slot * tensor.stride[0]
                                    + head * tensor.stride[1] + column] =
                                        float((slot * 3 + head * 7 + column * 11 + seed) % 29 - 14) / 16384.0f;
                };
                std::vector<uint8_t> previous_inverse, previous_output, previous_state;
                RadParam params[]{integer("M", rows <= 64 ? 64 : rows <= 128 ? 128 : 512),
                    integer("head_k", 128), integer("head_v", 128), integer("n_head_v", 24),
                    integer("chunk", 64), {}};
                params[5].key = "scale"; params[5].kind = RAD_P_F64;
                params[5].dval = padded ? 0.125 : 1.0;
                RadArgs args{};
                args.p = params; args.n_p = int(sizeof params / sizeof params[0]);
                for (int step = 0; step < 2; ++step) {
                    RadStream stream = step ? compute.stream : other.stream;
                    fill_plane(0, 3 + step * 17, 1.0f / 128);
                    fill_plane(1, 5 + step * 19, 1.0f / 512);
                    fill_plane(2, 7 + step * 23, 1.0f / 32);
                    fill_state(6, 11 + step * 31);
                    for (int sequence = 0; sequence < sequences; ++sequence)
                        for (int local = 0; local < lengths[sequence]; ++local) {
                            const int token = cumulative[sequence] + local;
                            for (int head = 0; head < 24; ++head) {
                                buffers[4]->data<float>()[token * buffers[4]->host.stride[0] + head] =
                                    -0.001f * (local % 64 + 1) * (1 + head % 3 + step);
                                buffers[5]->data<float>()[token * buffers[5]->host.stride[0] + head] =
                                    0.125f + float((token + head + step) % 17) / 128;
                            }
                        }
                    for (int operand : {3, 8})
                        std::fill(buffers[operand]->expected.begin(), buffers[operand]->expected.end(), uint8_t(0x5a));
                    if (mode != 2 || step == 0) {
                        std::fill(buffers[9]->expected.begin(), buffers[9]->expected.end(), uint8_t(0x5a));
                        if (mode == 2) fill_state(9, 37);
                    }
                    std::vector<RadTensor> host, device;
                    for (auto& buffer : buffers) {
                        host.push_back(buffer->host); device.push_back(buffer->device);
                    }
                    if (mode == 0) host[6] = device[6] = {};
                    if (mode == 2) { host[6] = host[9]; device[6] = device[9]; }
                    std::vector<RadTensor> host_solve{host[1], host[5], host[4], host[7], host[3]};
                    std::vector<RadTensor> device_solve{device[1], device[5], device[4], device[7], device[3]};
                    for (int operand = 0; operand < int(buffers.size()); ++operand)
                        if (operand != 9 || mode != 2 || step == 0) buffers[operand]->push(stream);
                    args.n_t = 5; args.t = host_solve.data();
                    REQUIRE_EQ(solve_ref->launch(&args, nullptr), RAD_OK);
                    args.t = device_solve.data();
                    REQUIRE_EQ(solve->launch(&args, stream), RAD_OK);
                    args.n_t = 11; args.t = host.data();
                    REQUIRE_EQ(scan_ref->launch(&args, nullptr), RAD_OK);
                    args.t = device.data();
                    REQUIRE_EQ(scan->launch(&args, stream), RAD_OK);
                    REQUIRE_EQ(rad_stream_sync(stream), RAD_OK);
                    for (auto& buffer : buffers) {
                        REQUIRE(buffer->read(stream) == buffer->expected);
                        compared += buffer->expected.size();
                    }
                    if (step) {
                        if (rows > 1) CHECK(buffers[3]->expected != previous_inverse);
                        CHECK(buffers[8]->expected != previous_output);
                        CHECK(buffers[9]->expected != previous_state);
                    }
                    previous_inverse = buffers[3]->expected;
                    previous_output = buffers[8]->expected;
                    previous_state = buffers[9]->expected;
                    params[1].ival = 127;
                    args.n_t = 5; args.t = device_solve.data();
                    CHECK_EQ(solve->launch(&args, stream), RAD_E_UNSUPPORTED);
                    args.n_t = 11; args.t = device.data();
                    CHECK_EQ(scan->launch(&args, stream), RAD_E_UNSUPPORTED);
                    params[1].ival = 128;
                    device[9].rank = 3;
                    CHECK_EQ(scan->launch(&args, stream), RAD_E_SHAPE);
                    device[9].rank = 4;
                    device[7].shape[0] = 10;
                    CHECK_EQ(scan->launch(&args, stream), RAD_E_UNSUPPORTED);
                    device[7] = buffers[7]->device;
                    for (int operand : {3, 8, 9}) CHECK(buffers[operand]->read(stream) == buffers[operand]->expected);
                    rejects += 4;
                    ++pipelines;
                }
                ++fixtures;
                std::printf("PASS GDN_CHUNK_AQL_CASE M=%d seq=%d padded=%d state=%d steps=2\n",
                    rows, sequences, padded, mode);
                std::fflush(stdout);
            }
    std::printf("PASS GDN_CHUNK_AQL fixtures=%llu pipelines=%llu kkt=%llu scan=%llu rejects=%llu exact_bytes=%llu queues=2 tolerance=0\n",
        (unsigned long long)fixtures, (unsigned long long)pipelines,
        (unsigned long long)pipelines, (unsigned long long)pipelines,
        (unsigned long long)rejects, (unsigned long long)compared);
}

TEST(r3d_i8_actual_aql_quant_boundary) {
    RadDeviceProps props{};
    REQUIRE_EQ(rad_dev_props(0, &props), RAD_OK);
    if (props.is_host_backend) {
        REQUIRE(!std::getenv("RAD_R3D_REQUIRE_DEVICE"));
        SKIP("explicit gfx1100 card run required");
    }
    REQUIRE_EQ(std::strcmp(props.arch, "gfx1100"), 0);
    REQUIRE_EQ(rad_dev_set(0), RAD_OK);
    struct Rounding {
        int saved = std::fegetround();
        ~Rounding() { std::fesetround(saved); }
    } rounding;
    REQUIRE_EQ(std::fesetround(FE_TONEAREST), 0);
    Library native("libr3d");
    using Launch = int (*)(const RadArgs*, RadStream);
    auto quant = reinterpret_cast<Launch>(dlsym(native.handle, "r3dh_i8_rad_quant_act_i8g"));
    auto had = reinterpret_cast<Launch>(dlsym(native.handle, "r3dh_i8_rad_had_quant_act_i8g"));
    auto gemm = reinterpret_cast<Launch>(dlsym(native.handle, "r3dh_i8_rad_gemm_nt_q"));
    REQUIRE(quant && had && gemm);
    auto round_even = [](float value) {
        const float lower = std::floor(value);
        int result = static_cast<int>(lower);
        const float fraction = value - lower;
        if (fraction > 0.5f || (fraction == 0.5f && (result & 1))) ++result;
        return static_cast<int8_t>(result);
    };
    CHECK_EQ(round_even(0.5f), 0); CHECK_EQ(round_even(-0.5f), 0);
    CHECK_EQ(round_even(1.5f), 2); CHECK_EQ(round_even(-1.5f), -2);
    CHECK_EQ(round_even(2.5f), 2); CHECK_EQ(round_even(-2.5f), -2);
    Queue compute(0), other(1);
    uint64_t fixtures = 0, pipelines = 0, rejects = 0, compared = 0;
    constexpr int columns = 128;
    for (int width : {2560, 3072, 6144}) {
        const int groups = width / 128;
        std::vector<int8_t> real_codes(columns * width);
        std::vector<uint16_t> real_scales(columns * groups);
        std::unique_ptr<FILE, int (*)(FILE*)> model(std::fopen(
            "/home/zihaomu/bigssd2/radiance_workspace/models/qwen3.8-next-flash-iqr-v4.rad", "rb"), std::fclose);
        REQUIRE(model);
        const int model_width = width == 2560 ? 2560 : 6144;
        const int shard = width == 3072 ? 3072 : 0;
        const int64_t code_offset = width == 2560 ? 58022457344ll : 53973250048ll;
        const int64_t scale_offset = width == 2560 ? 58023768064ll : 53988978688ll;
        for (int column = 0; column < columns; ++column) {
            REQUIRE_EQ(fseeko(model.get(), code_offset + int64_t(column) * model_width + shard, SEEK_SET), 0);
            REQUIRE_EQ(std::fread(real_codes.data() + column * width, 1, width, model.get()), size_t(width));
            REQUIRE_EQ(fseeko(model.get(), scale_offset + int64_t(column) * model_width / 64 + shard / 64, SEEK_SET), 0);
            REQUIRE_EQ(std::fread(real_scales.data() + column * groups, 2, groups, model.get()), size_t(groups));
        }
        for (int rows : {1, 3, 63, 64})
            for (int padded : {0, 1})
                for (int rotated : {0, 1}) {
                    const int input_pitch = width + padded * 32;
                    const int code_pitch = width + padded * 12;
                    const int scale_pitch = groups + padded * 5;
                    const int output_pitch = columns + padded * 16;
                    Buffer input(RAD_BF16, {rows + 2, groups, 128}, RAD_MEM_DEVICE, input_pitch);
                    Buffer codes(RAD_I8, {rows + 2, width}, RAD_MEM_DEVICE, code_pitch);
                    Buffer scales(RAD_F32, {rows + 2, groups}, RAD_MEM_DEVICE, scale_pitch);
                    Buffer weight(RAD_I8, {columns, width}, RAD_MEM_DEVICE);
                    Buffer weight_scale(RAD_F32, {columns, groups}, RAD_MEM_DEVICE);
                    Buffer output(RAD_BF16, {rows + 2, columns}, RAD_MEM_DEVICE, output_pitch);
                    for (Buffer* buffer : {&input, &codes, &scales, &output})
                        buffer->host.shape[0] = buffer->device.shape[0] = rows;
                    weight.device.dtype = r3dh_i8::weight_dtype;
                    weight_scale.device.dtype = r3dh_i8::scale_dtype;
                    std::vector<int8_t> natural(columns * width);
                    std::vector<float> factors(columns * groups);
                    for (int column = 0; column < columns; ++column) {
                        for (int position = 0; position < width; ++position) {
                            const int index = column * width + position;
                            natural[index] = rotated ? real_codes[index] :
                                static_cast<int8_t>((column * 73 + position * 37 + position / 128 * 19) % 255 - 127);
                            const int packed = (((column / 16 * groups + position / 128) * 8
                                + position % 128 / 16) * 16 + column % 16) * 16 + position % 16;
                            weight.data<int8_t>()[packed] = natural[index];
                        }
                        for (int group = 0; group < groups; ++group) {
                            const int index = column * groups + group;
                            const uint16_t bits = rotated ? real_scales[index] : rad_f32_to_bf16(
                                float((column * 11 + group * 7) % 31 - 15) / 8192.0f);
                            factors[index] = rad_bf16_to_f32(bits);
                            weight_scale.data<float>()[index] = factors[index];
                        }
                    }
                    RadParam params[]{integer("M", 64), integer("n", width), integer("group", 128)};
                    RadParam gemm_params[]{integer("M", 64), integer("N", columns), integer("K", width),
                        integer("group", 128), integer("w_rb", 1), {}};
                    gemm_params[5].key = "dtype"; gemm_params[5].kind = RAD_P_STR;
                    gemm_params[5].sval = rotated ? "i8a8h" : "i8a8";
                    Launch producer = rotated ? had : quant;
                    std::vector<uint8_t> previous_codes, previous_scales, previous_output;
                    for (int step = 0; step < 2; ++step) {
                        RadStream stream = step ? compute.stream : other.stream;
                        params[0].ival = gemm_params[0].ival = step ? 2048 : 64;
                        for (int row = 0; row < rows; ++row)
                            for (int position = 0; position < width; ++position) {
                                float value = float((row * 197 + position * 29 + position / 128 * 17 + step * 71) % 509 - 254)
                                    / float(8 << ((position / 128 + step) % 5));
                                if (row == 0 && position < 128) value = 0.0f;
                                if (!rotated && row == 0 && position >= 128 && position < 256) {
                                    const float ties[]{127.0f, 0.5f, -0.5f, 1.5f, -1.5f, 2.5f, -2.5f};
                                    value = ties[(position - 128 + step) % 7];
                                }
                                input.data<uint16_t>()[row * input_pitch + position] = rad_f32_to_bf16(value);
                            }
                        for (Buffer* buffer : {&codes, &scales, &output})
                            std::fill(buffer->expected.begin(), buffer->expected.end(), uint8_t(0x5a));
                        for (Buffer* buffer : {&input, &codes, &scales, &weight, &weight_scale, &output}) buffer->push(stream);
                        for (int row = 0; row < rows; ++row)
                            for (int group = 0; group < groups; ++group) {
                                float values[128];
                                for (int position = 0; position < 128; ++position)
                                    values[position] = rad_bf16_to_f32(input.data<uint16_t>()[row * input_pitch + group * 128 + position]);
                                if (rotated)
                                    for (int distance = 1; distance < 128; distance *= 2)
                                        for (int base = 0; base < 128; base += distance * 2)
                                            for (int offset = 0; offset < distance; ++offset) {
                                                const float left = values[base + offset];
                                                const float right = values[base + distance + offset];
                                                values[base + offset] = left + right;
                                                values[base + distance + offset] = left - right;
                                            }
                                float maximum = 0.0f;
                                for (float value : values) maximum = std::max(maximum, std::fabs(value));
                                if (!(maximum > 0.0f)) maximum = 1e-8f;
                                scales.data<float>()[row * scale_pitch + group] = maximum / 127.0f;
                                const float inverse = 127.0f / maximum;
                                for (int position = 0; position < 128; ++position)
                                    codes.data<int8_t>()[row * code_pitch + group * 128 + position] =
                                        round_even(std::max(-127.0f, std::min(127.0f, values[position] * inverse)));
                            }
                        for (int row = 0; row < rows; ++row)
                            for (int column = 0; column < columns; ++column) {
                                float accumulator = 0.0f;
                                for (int group = 0; group < groups; ++group) {
                                    int subtotal = 0;
                                    for (int position = 0; position < 128; ++position)
                                        subtotal += int(codes.data<int8_t>()[row * code_pitch + group * 128 + position])
                                            * int(natural[column * width + group * 128 + position]);
                                    const float scaled = static_cast<float>(subtotal)
                                        * scales.data<float>()[row * scale_pitch + group];
                                    accumulator = accumulator + scaled * factors[column * groups + group];
                                }
                                output.data<uint16_t>()[row * output_pitch + column] = rad_f32_to_bf16(accumulator);
                            }
                        RadTensor tensors[]{input.device, codes.device, scales.device};
                        const RadTensor original = tensors[0];
                        RadArgs args{}; args.t = tensors; args.n_t = 3; args.p = params; args.n_p = 3;
                        REQUIRE_EQ(producer(&args, stream), RAD_OK);
                        REQUIRE(std::memcmp(&original, &tensors[0], sizeof original) == 0);
                        REQUIRE(codes.read(stream) == codes.expected);
                        REQUIRE(scales.read(stream) == scales.expected);
                        RadTensor gemm_tensors[]{codes.device, scales.device, weight.device, weight_scale.device, output.device};
                        RadArgs gemm_args{}; gemm_args.t = gemm_tensors; gemm_args.n_t = 5;
                        gemm_args.p = gemm_params; gemm_args.n_p = 6;
                        REQUIRE_EQ(gemm(&gemm_args, stream), RAD_OK);
                        int operand = 0;
                        for (Buffer* buffer : {&input, &codes, &scales, &weight, &weight_scale, &output}) {
                            const auto actual = buffer->read(stream);
                            if (actual != buffer->expected) {
                                const size_t offset = std::mismatch(actual.begin(), actual.end(), buffer->expected.begin()).first - actual.begin();
                                std::fprintf(stderr, "I8_BOUNDARY_DIFF n=%d rows=%d padded=%d rotated=%d step=%d operand=%d byte=%zu actual=%u expected=%u\n",
                                    width, rows, padded, rotated, step, operand, offset,
                                    unsigned(actual[offset]), unsigned(buffer->expected[offset]));
                            }
                            REQUIRE(actual == buffer->expected);
                            compared += buffer->expected.size();
                            ++operand;
                        }
                        if (step) {
                            CHECK(codes.expected != previous_codes);
                            CHECK(scales.expected != previous_scales);
                            CHECK(output.expected != previous_output);
                        }
                        previous_codes = codes.expected; previous_scales = scales.expected; previous_output = output.expected;
                        auto reject = [&](int expected) {
                            CHECK_EQ(producer(&args, stream), expected); ++rejects;
                            tensors[0] = input.device; tensors[1] = codes.device; tensors[2] = scales.device;
                        };
                        tensors[0].rank = 1; reject(RAD_E_SHAPE);
                        tensors[0].rank = 7; reject(RAD_E_SHAPE);
                        tensors[0].shape[1]++; reject(RAD_E_SHAPE);
                        tensors[0].shape[1] = 0; reject(RAD_E_SHAPE);
                        tensors[0].shape[1] = std::numeric_limits<int64_t>::max(); reject(RAD_E_SHAPE);
                        tensors[0].stride[1]++; reject(RAD_E_STRIDE);
                        tensors[0].stride[2] = 2; reject(RAD_E_STRIDE);
                        tensors[0].stride[0] = width - 1; reject(RAD_E_STRIDE);
                        tensors[0].stride[0] = -width; reject(RAD_E_STRIDE);
                        if (rows > 1) { tensors[0].stride[0] = std::numeric_limits<int64_t>::max(); reject(RAD_E_STRIDE); }
                        tensors[0].shape[0] = 2049; reject(RAD_E_SHAPE);
                        tensors[1].rank = 3; reject(RAD_E_SHAPE);
                        tensors[1].stride[1] = 2; reject(RAD_E_STRIDE);
                        tensors[2].shape[1]++; reject(RAD_E_SHAPE);
                        tensors[2].stride[0] = groups - 1; reject(RAD_E_STRIDE);
                        params[0].ival = rows - 1; reject(RAD_E_SHAPE);
                        params[0].ival = 2049; reject(RAD_E_SHAPE);
                        params[0].ival = step ? 2048 : 64;
                        params[1].ival = width + 128; reject(RAD_E_SHAPE); params[1].ival = width;
                        tensors[0].dtype = RAD_F32; reject(RAD_E_DTYPE);
                        tensors[1].dtype = RAD_I32; reject(RAD_E_DTYPE);
                        params[0].ival = rows;
                        tensors[0].rank = 2; tensors[0].shape[1] = width; tensors[0].stride[1] = 1;
                        REQUIRE_EQ(producer(&args, stream), RAD_OK);
                        tensors[0] = input.device;
                        tensors[0].rank = 4; tensors[0].shape[1] = groups / 4; tensors[0].shape[2] = 4;
                        tensors[0].shape[3] = 128; tensors[0].stride[1] = 512;
                        tensors[0].stride[2] = 128; tensors[0].stride[3] = 1;
                        REQUIRE_EQ(producer(&args, stream), RAD_OK);
                        for (Buffer* buffer : {&codes, &scales}) REQUIRE(buffer->read(stream) == buffer->expected);
                        gemm_params[0].ival = rows - 1;
                        CHECK_EQ(gemm(&gemm_args, stream), RAD_E_SHAPE); ++rejects;
                        gemm_params[0].ival = 2049;
                        CHECK_EQ(gemm(&gemm_args, stream), RAD_E_SHAPE); ++rejects;
                        gemm_params[0].ival = rows;
                        gemm_tensors[0].rank = 3;
                        CHECK_EQ(gemm(&gemm_args, stream), RAD_E_SHAPE); ++rejects;
                        gemm_tensors[0] = codes.device;
                        gemm_tensors[3].shape[1] = groups + 1;
                        CHECK_EQ(gemm(&gemm_args, stream), RAD_E_SHAPE); ++rejects;
                        gemm_tensors[3] = weight_scale.device;
                        REQUIRE_EQ(gemm(&gemm_args, stream), RAD_OK);
                        REQUIRE(output.read(stream) == output.expected);
                        ++pipelines;
                    }
                    ++fixtures;
                }
        std::printf("PASS I8_BOUNDARY_AQL_WIDTH n=%d real_weight=%s queues=2\n", width,
            width == 2560 ? "attn_k_first128" : width == 3072 ? "ssm_out_rank1_k_shard_first128" : "ssm_out_first128");
        std::fflush(stdout);
    }
    std::printf("PASS I8_BOUNDARY_AQL fixtures=%llu pipelines=%llu producer_launches=%llu gemm_launches=%llu rejects=%llu exact_bytes=%llu queues=2 tolerance=0\n",
        (unsigned long long)fixtures, (unsigned long long)pipelines,
        (unsigned long long)(pipelines * 3), (unsigned long long)(pipelines * 2),
        (unsigned long long)rejects, (unsigned long long)compared);
}

#include "r3d_moe_boundary_aql.h"
#include "r3d_attn_prepare_aql.h"
#include "r3d_qsa_index_aql.h"
#include "r3d_qsa_ring_aql.h"
#include "r3d_iqr_output_aql.h"
#include "r3d_attn_fused_actual_aql.h"
#include "r3d_joined_head_aql.h"
#include "r3d_rerank_aql.h"
#include "r3d_sample_pair_aql.h"
#include "r3d_sample_merge32_aql.h"
#include "r3d_allgather_pair_aql.h"
#include "r3d_gdn_decode_actual_aql.h"
#include "r3d_iqr_group_aql.h"
#include "r3d_w4nl_gate_aql.h"
#include "r3d_i8_gated_aql.h"
#include "r3d_ple_domain_aql.h"
#include "r3d_mtp_fc_aql.h"
#include "r3d_w2_head_aql.h"
#include "r3d_vision_actual_aql.h"
#include "r3d_attn_fused_paged_aql.h"
#include "r3d_paged_wide_aql.h"
#include "r3d_tp1_quant_aql.h"
