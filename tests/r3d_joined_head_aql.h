#include "../libr3d/r3d_logits_planes.h"
#include <chrono>

namespace joined_head_test {
using Launch = int (*)(const RadArgs*, RadStream);
float multiply(float left, float right) { volatile float value = left * right; return value; }
float add(float left, float right) { volatile float value = left + right; return value; }
float decode(uint8_t bits, uint32_t dtype) {
    if (dtype == RAD_I8) return bits < 128 ? float(bits) : float(int(bits) - 256);
    const int magnitude = bits & 127;
    const int exponent = magnitude / 8;
    const int mantissa = magnitude % 8;
    const float value = !exponent ? std::ldexp(float(mantissa), -9)
        : magnitude == 127 ? std::numeric_limits<float>::quiet_NaN()
                            : std::ldexp(float(8 + mantissa), exponent - 10);
    return bits & 128 ? -value : value;
}
struct Api {
    Library native{"libr3d"};
    Library reference{"libref"};
    Launch launch = reinterpret_cast<Launch>(dlsym(native.handle, "r3d_rad_logits_gemm_bf16"));
    RadLayoutFn layout = reinterpret_cast<RadLayoutFn>(dlsym(native.handle, "r3d_layout_logits_head"));
    RadRelayoutFn relayout = reinterpret_cast<RadRelayoutFn>(dlsym(native.handle, "r3d_relayout_logits_head"));
    const RadKernelInfo* oracle = reference.ngram("logits_gemm");
    Api() { REQUIRE(launch && layout && relayout && oracle); }
};
struct Storage {
    const rad::DeviceBackend* backend = rad::active_backend();
    void* allocation = nullptr;
    int64_t bytes = 0;
    int memory;
    std::vector<uint8_t> expected;
    Storage(size_t payload, int kind) : memory(kind), expected(payload + 128, 0x5a) {
        if (kind >= 0) allocation = rad_dev_alloc(expected.size(), kind);
        else {
            const int64_t granule = backend->vmem_granularity();
            REQUIRE(granule > 0);
            bytes = (int64_t(expected.size()) + granule - 1) / granule * granule;
            allocation = backend->vmem_reserve(bytes, granule);
            REQUIRE(allocation);
            REQUIRE(backend->vmem_commit_units(allocation, bytes, granule) >= 0);
        }
        REQUIRE(allocation);
    }
    ~Storage() {
        if (memory >= 0) rad_dev_free(allocation, memory);
        else { backend->vmem_decommit(allocation, bytes); backend->vmem_release(allocation, bytes); }
    }
    void* device() { return static_cast<uint8_t*>(allocation) + 64; }
    void push(RadStream stream) {
        REQUIRE_EQ(rad_memcpy_async(allocation, expected.data(), expected.size(), stream), RAD_OK);
    }
    std::vector<uint8_t> read(RadStream stream) {
        std::vector<uint8_t> actual(expected.size());
        if (rad_memcpy_async(actual.data(), allocation, actual.size(), stream) != RAD_OK ||
            rad_stream_sync(stream) != RAD_OK) throw std::runtime_error("joined storage read");
        return actual;
    }
};
RadTensor matrix(void* pointer, uint32_t dtype, int64_t rows, int64_t columns, int64_t pitch = 0) {
    RadTensor result{};
    result.data = pointer; result.dtype = dtype; result.rank = 2;
    result.shape[0] = rows; result.shape[1] = columns;
    result.stride[0] = pitch ? pitch : columns; result.stride[1] = 1;
    return result;
}
struct Fixture {
    int columns, reduction;
    RadEncoding encoding;
    RadLayout stored{};
    std::vector<uint8_t> codes, scale, joined;
    RadTensor planes[2]{};
    static constexpr int selected[2]{0, 1};
    Fixture(Api& api, int width, int inner, uint32_t code_dtype, uint32_t scale_dtype,
            int row_block, int col_block)
        : columns(width), reduction(inner), encoding(rad_enc_affine(code_dtype, scale_dtype,
              row_block, col_block)), codes(size_t(width) * inner) {
        const int rows = row_block ? (width + row_block - 1) / row_block : 1;
        const int cols = col_block ? (inner + col_block - 1) / col_block : 1;
        const int bytes = scale_dtype == RAD_F32 ? 4 : 2;
        scale.resize(size_t(rows) * cols * bytes);
        for (size_t index = 0; index < codes.size(); ++index)
            codes[index] = code_dtype == RAD_I8 ? uint8_t(index * 73 + index / inner)
                : uint8_t((index * 37 + index / inner) % 127 | ((index & 1) << 7));
        for (int index = 0; index < rows * cols; ++index) {
            const float value = std::ldexp(float(129 + index % 123), -15 - index % 3)
                + (scale_dtype == RAD_F32 ? float(1 + index % 7) * 0x1p-27f : 0.0f);
            if (scale_dtype == RAD_F32) std::memcpy(scale.data() + index * 4, &value, 4);
            else {
                const uint16_t bits = scale_dtype == RAD_BF16 ? rad_f32_to_bf16(value) : rad_f32_to_f16(value);
                std::memcpy(scale.data() + index * 2, &bits, 2);
            }
        }
        planes[0] = matrix(codes.data(), code_dtype, width, inner);
        planes[1] = matrix(scale.data(), scale_dtype, rows, cols);
        pack(api);
    }
    void pack(Api& api) {
        planes[0].data = codes.data(); planes[1].data = scale.data();
        REQUIRE_EQ(api.layout(nullptr, 0, 1, &encoding, selected, planes, 2, &stored), RAD_OK);
        REQUIRE_EQ(stored.dtype, r3d_head::dtype);
        joined.resize(stored.bytes);
        REQUIRE_EQ(api.relayout(nullptr, 0, 1, &encoding, selected, planes, 2,
                               joined.data(), joined.size()), RAD_OK);
    }
    float factor(int column, int position) const {
        const auto& spec = encoding.plane[1];
        const int row = spec.block[0] ? column / spec.block[0] : 0;
        const int col = spec.block[1] ? position / spec.block[1] : 0;
        return rad_load_f32(scale.data(), spec.dtype, int64_t(row) * planes[1].shape[1] + col);
    }
};
uint64_t launches = 0, values = 0, guards = 0, rejects = 0, witnesses = 0, fma_witnesses = 0;
void execute(Api& api, Fixture& fixture, int rows, bool full_model) {
    const int columns = fixture.columns, reduction = fixture.reduction;
    const int x_pitch = reduction + 16, y_pitch = ((columns + 7) / 8 + 1) * 8;
    Queue first(0), second(1);
    Buffer activation(RAD_BF16, {rows, reduction}, RAD_MEM_DEVICE, x_pitch);
    Buffer output(RAD_F32, {rows, columns}, RAD_MEM_DEVICE, y_pitch);
    for (int step = 0; step < 2; ++step) {
        for (int row = 0; row < rows; ++row)
            for (int position = 0; position < reduction; ++position)
                activation.data<uint16_t>()[row * x_pitch + position] = rad_f32_to_bf16(
                    float(((position * 47 + row * 31 + step * 71) % 257) - 128) / 64.0f);
        std::vector<uint8_t> expected = output.expected;
        const auto start = std::chrono::steady_clock::now();
        uint64_t f16_differences = 0;
        for (int row = 0; row < rows; ++row)
            for (int column = 0; column < columns; ++column) {
                float accumulator = 0, narrowed = 0, fused = 0;
                for (int position = 0; position < reduction; ++position) {
                    const float left = rad_bf16_to_f32(activation.data<uint16_t>()[row * x_pitch + position]);
                    const float weight = multiply(decode(fixture.codes[size_t(column) * reduction + position],
                        fixture.encoding.plane[0].dtype), fixture.factor(column, position));
                    accumulator = add(accumulator, multiply(left, weight));
                    if (!full_model) {
                        narrowed = add(narrowed, multiply(left, rad_f16_to_f32(rad_f32_to_f16(weight))));
                        fused = std::fma(left, weight, fused);
                    }
                }
                if (!full_model && std::memcmp(&accumulator, &narrowed, 4)) ++f16_differences;
                if (!full_model && std::memcmp(&accumulator, &fused, 4)) ++fma_witnesses;
                std::memcpy(expected.data() + 64 + (size_t(row) * y_pitch + column) * 4, &accumulator, 4);
            }
        witnesses += f16_differences;
        if (full_model) std::printf("JOINED_CPU_EXHAUSTIVE N=%d K=%d M=%d step=%d seconds=%.3f\n",
            columns, reduction, rows, step, std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
        if (!full_model) {
            RadTensor host[]{activation.host, matrix(fixture.joined.data(), fixture.stored.dtype, columns, reduction), output.host};
            RadParam params[]{integer("M", 64), integer("n_vocab", columns), integer("n_embd", reduction), integer("impl", 0)};
            RadArgs args{}; args.t = host; args.n_t = 3; args.p = params; args.n_p = 4;
            REQUIRE_EQ(api.oracle->launch(&args, nullptr), RAD_OK);
            CHECK(output.expected == expected);
            std::fill(output.expected.begin(), output.expected.end(), 0x5a);
        }
        for (int memory : {int(RAD_MEM_DEVICE), -1, int(RAD_MEM_HOST_MAPPED)}) {
            Storage weight(fixture.joined.size(), memory);
            std::memcpy(weight.expected.data() + 64, fixture.joined.data(), fixture.joined.size());
            for (bool logical : {false, true}) {
                RadStream stream = logical ? second.stream : first.stream;
                weight.push(stream); activation.push(stream); output.push(stream);
                REQUIRE_EQ(rad_stream_sync(stream), RAD_OK);
                RadTensor descriptor = matrix(weight.device(), fixture.stored.dtype, columns, reduction);
                if (!logical) { descriptor.rank = 1; descriptor.shape[0] = fixture.stored.bytes; descriptor.stride[0] = 1; }
                RadTensor device[]{activation.device, descriptor, output.device};
                RadParam params[]{integer("M", step == 0 && rows == 1 ? 1 : 64), integer("n_vocab", columns),
                                  integer("n_embd", reduction), integer("impl", step)};
                RadArgs args{}; args.t = device; args.n_t = 3; args.p = params; args.n_p = 4;
                const auto before = std::chrono::steady_clock::now();
                REQUIRE_EQ(api.launch(&args, stream), RAD_OK);
                REQUIRE_EQ(rad_stream_sync(stream), RAD_OK);
                const double elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - before).count();
                auto actual = output.read(stream);
                CHECK(actual == expected);
                CHECK(weight.read(stream) == weight.expected);
                CHECK(activation.read(stream) == activation.expected);
                ++launches; values += uint64_t(rows) * columns;
                guards += actual.size() + weight.expected.size() + activation.expected.size();
                if (full_model) std::printf("JOINED_GPU_FULL N=%d K=%d memory=%d logical=%d band=%lld step=%d wall_ms=%.3f\n",
                    columns, reduction, memory, logical, (long long)(step == 0 && rows == 1 ? 1 : 64), step, elapsed);
            }
        }
    }
}
bool device_available() {
    RadDeviceProps props{};
    if (rad_dev_props(0, &props) != RAD_OK) throw std::runtime_error("joined device props");
    if (props.is_host_backend) {
        if (std::getenv("RAD_R3D_REQUIRE_DEVICE")) throw std::runtime_error("joined requires actual GPU");
        return false;
    }
    if (std::strcmp(props.arch, "gfx1100") || rad_dev_set(0) != RAD_OK)
        throw std::runtime_error("joined gfx1100 device selection");
    return true;
}
}

TEST(r3d_joined_head_actual_aql_strict_matrix) {
    using namespace joined_head_test;
    if (!device_available()) SKIP("explicit gfx1100 card run required");
    Api api;
    for (uint32_t code : {uint32_t(RAD_I8), uint32_t(RAD_F8E4M3)})
        for (uint32_t scale : {uint32_t(RAD_BF16), uint32_t(RAD_F16), uint32_t(RAD_F32)})
            for (const auto block : {std::array<int, 2>{1, 128}, {128, 128}, {0, 128}, {1, 0},
                                     {0, 0}, {7, 73}}) {
                Fixture fixture(api, 131, 256, code, scale, block[0], block[1]);
                execute(api, fixture, 1, false);
                execute(api, fixture, 3, false);
            }
    CHECK(witnesses > 0);
    CHECK(fma_witnesses > 0);
    std::printf("JOINED_STRICT_MATRIX_PASS launches=%llu values=%llu guarded_bytes=%llu non_f16_exact_witnesses=%llu\n",
        (unsigned long long)launches, (unsigned long long)values, (unsigned long long)guards, (unsigned long long)witnesses);
    std::printf("JOINED_NO_FMA_WITNESSES values=%llu\n", (unsigned long long)fma_witnesses);
}

TEST(r3d_joined_head_actual_aql_rejections) {
    using namespace joined_head_test;
    if (!device_available()) SKIP("explicit gfx1100 card run required");
    Api api;
    Fixture fixture(api, 131, 256, RAD_I8, RAD_BF16, 1, 128);
    for (int defect = 0; defect < 13; ++defect) {
        auto encoding = fixture.encoding;
        RadTensor planes[]{fixture.planes[0], fixture.planes[1]};
        int selected[]{0, 1};
        int status = RAD_E_DTYPE;
        switch (defect) {
        case 0: encoding.plane[1].block[0] = -1; status = RAD_E_SHAPE; break;
        case 1: encoding.plane[1].block[1] = -1; status = RAD_E_SHAPE; break;
        case 2: encoding.plane[0].role[0] = 'x'; break;
        case 3: encoding.transform[0] = 'x'; break;
        case 4: selected[1] = 0; break;
        case 5: planes[1].dtype = RAD_I8; break;
        case 6: ++planes[1].shape[0]; status = RAD_E_SHAPE; break;
        case 7: ++planes[0].stride[0]; status = RAD_E_STRIDE; break;
        case 8: encoding.plane[0].kind = RAD_PLANE_TABLE; break;
        case 9: encoding.plane[1].dtype = RAD_I8; break;
        case 10: planes[0].shape[0] = INT64_MAX; status = RAD_E_SHAPE; break;
        case 11: planes[0].shape[1] = INT64_MAX; status = RAD_E_SHAPE; break;
        case 12: planes[1].shape[1] = 1; planes[1].stride[0] = 1; status = RAD_E_SHAPE; break;
        }
        RadLayout output{};
        std::vector<uint8_t> untouched(fixture.stored.bytes, 0x5a), destination = untouched;
        CHECK_EQ(api.layout(nullptr, 0, 1, &encoding, selected, planes, 2, &output), status);
        CHECK_EQ(api.relayout(nullptr, 0, 1, &encoding, selected, planes, 2,
                              destination.data(), destination.size()), status);
        CHECK(destination == untouched);
        rejects += 2;
    }
    Queue queue(1);
    Buffer activation(RAD_BF16, {1, 256}, RAD_MEM_DEVICE);
    Buffer output(RAD_F32, {1, 131}, RAD_MEM_DEVICE, 144);
    RadParam params[]{integer("M", 64), integer("n_vocab", 131), integer("n_embd", 256), integer("impl", 0)};
    for (int memory : {int(RAD_MEM_DEVICE), -1, int(RAD_MEM_HOST_MAPPED)}) {
        Storage weight(fixture.joined.size(), memory);
        RadTensor tensors[]{activation.device, matrix(weight.device(), fixture.stored.dtype, 131, 256), output.device};
        RadArgs args{}; args.t = tensors; args.n_t = 3; args.p = params; args.n_p = 4;
        for (int defect = 0; defect < 17; ++defect) {
            std::memcpy(weight.expected.data() + 64, fixture.joined.data(), fixture.joined.size());
            auto* header = reinterpret_cast<RadPlanesHeader*>(weight.expected.data() + 64);
            auto* entry = reinterpret_cast<RadPlanesEntry*>(header + 1);
            int status = RAD_E_FORMAT;
            switch (defect) {
            case 0: header->magic = 0; break;
            case 1: header->n_planes = 3; break;
            case 2: header->transform[0] = 'x'; break;
            case 3: --header->bytes; break;
            case 4: entry[0].dtype = RAD_F16; status = RAD_E_DTYPE; break;
            case 5: entry[0].block[0] = 2; status = RAD_E_DTYPE; break;
            case 6: entry[1].dtype = RAD_I8; status = RAD_E_DTYPE; break;
            case 7: entry[1].block[0] = -1; status = RAD_E_SHAPE; break;
            case 8: ++entry[1].rows; status = RAD_E_SHAPE; break;
            case 9: ++entry[0].row_bytes; status = RAD_E_STRIDE; break;
            case 10: entry[0].offset += 64; break;
            case 11: entry[1].offset -= 64; break;
            case 12: entry[1].block[1] = -1; status = RAD_E_SHAPE; break;
            case 13: entry[0].role[0] = 'x'; status = RAD_E_DTYPE; break;
            case 14: entry[1].row_bytes = INT64_MAX; status = RAD_E_STRIDE; break;
            case 15: header->bytes = INT64_MAX; break;
            case 16: entry[0].rows = INT64_MAX; status = RAD_E_SHAPE; break;
            }
            weight.push(queue.stream); activation.push(queue.stream); output.push(queue.stream);
            REQUIRE_EQ(rad_stream_sync(queue.stream), RAD_OK);
            CHECK_EQ(api.launch(&args, queue.stream), memory == RAD_MEM_HOST_MAPPED ? status : RAD_OK);
            REQUIRE_EQ(rad_stream_sync(queue.stream), RAD_OK);
            CHECK(output.read(queue.stream) == output.expected);
            CHECK(weight.read(queue.stream) == weight.expected);
            ++rejects;
        }
        std::memcpy(weight.expected.data() + 64, fixture.joined.data(), fixture.joined.size());
        weight.push(queue.stream); REQUIRE_EQ(rad_stream_sync(queue.stream), RAD_OK);
        for (int index = 0; index < 3; ++index) {
            void* saved = tensors[index].data;
            tensors[index].data = reinterpret_cast<void*>(0x10000);
            CHECK_EQ(api.launch(&args, queue.stream), RAD_E_DEVICE);
            tensors[index].data = saved;
            ++rejects;
        }
        CHECK_EQ(api.launch(&args, reinterpret_cast<RadStream>(0x12345)), RAD_E_DEVICE);
        ++rejects;
        auto saved = tensors[1];
        tensors[1].rank = 1; tensors[1].shape[0] = sizeof(RadPlanesHeader); tensors[1].stride[0] = 1;
        CHECK_EQ(api.launch(&args, queue.stream), RAD_E_SHAPE);
        tensors[1] = saved;
        tensors[1].shape[0] = 130;
        CHECK_EQ(api.launch(&args, queue.stream), RAD_E_SHAPE);
        tensors[1] = saved;
        auto input = tensors[0]; tensors[0].stride[0] = INT64_MAX;
        CHECK_EQ(api.launch(&args, queue.stream), RAD_E_STRIDE);
        tensors[0] = input;
        REQUIRE_EQ(rad_stream_sync(queue.stream), RAD_OK);
        CHECK(output.read(queue.stream) == output.expected);
        rejects += 3;
    }
    std::printf("JOINED_REJECTIONS_PASS checks=%llu device_corruption=enqueue_ok_no_writes_not_host_error\n",
        (unsigned long long)rejects);
}

TEST(r3d_joined_head_actual_model_tp2) {
    using namespace joined_head_test;
    if (!device_available()) SKIP("explicit gfx1100 card run required");
    Api api;
    Model model("/home/zihaomu/bigssd2/radiance_workspace/models/qwen3.8-next-flash-iqr-v4.rad");
    REQUIRE_EQ(model.header.file_bytes, 121678372864ull);
    bool found = false;
    for (const auto& entry : model.entries) {
        if (model.name(entry) != "output.weight") continue;
        found = true;
        REQUIRE_EQ(entry.rank, 2u); REQUIRE_EQ(entry.shape[0], 248320); REQUIRE_EQ(entry.shape[1], 2560);
        REQUIRE(entry.enc < model.encodings.size()); REQUIRE_EQ(entry.n_planes, 2u);
        const auto encoding = model.encodings[entry.enc];
        REQUIRE_EQ(encoding.n_planes, 2);
        REQUIRE_EQ(encoding.plane[0].dtype, uint32_t(RAD_I8));
        REQUIRE_EQ(encoding.plane[1].dtype, uint32_t(RAD_BF16));
        REQUIRE_EQ(encoding.plane[1].block[0], 1); REQUIRE_EQ(encoding.plane[1].block[1], 128);
        for (int rank = 0; rank < 2; ++rank) {
            Fixture fixture(api, 124160, 2560, RAD_I8, RAD_BF16, 1, 128);
            fixture.encoding = encoding;
            for (unsigned index = 0; index < 2; ++index) {
                REQUIRE(entry.plane_first + index < model.planes.size());
                const auto& location = model.planes[entry.plane_first + index];
                REQUIRE(location.offset >= entry.offset);
                REQUIRE(location.offset - entry.offset <= entry.bytes);
                REQUIRE(location.bytes <= entry.bytes - (location.offset - entry.offset));
                auto& destination = index ? fixture.scale : fixture.codes;
                const uint64_t offset = uint64_t(rank) * destination.size();
                REQUIRE_EQ(location.bytes, destination.size() * 2);
                REQUIRE(offset <= location.bytes && destination.size() <= location.bytes - offset);
                model.read(location.offset + offset, destination.data(), destination.size());
            }
            fixture.pack(api);
            std::printf("JOINED_ACTUAL_TP_SHARD rank=%d vocab_offset=%d N=124160 K=2560 code_bytes=%zu scale_bytes=%zu\n",
                rank, rank * 124160, fixture.codes.size(), fixture.scale.size());
            execute(api, fixture, 1, true);
            Fixture slice(api, 131, 2560, RAD_I8, RAD_BF16, 1, 128);
            std::copy_n(fixture.codes.begin(), slice.codes.size(), slice.codes.begin());
            std::copy_n(fixture.scale.begin(), slice.scale.size(), slice.scale.begin());
            slice.pack(api);
            execute(api, slice, 3, false);
        }
    }
    REQUIRE(found);
    std::printf("JOINED_ACTUAL_TP2_PASS launches=%llu values=%llu guarded_bytes=%llu\n",
        (unsigned long long)launches, (unsigned long long)values, (unsigned long long)guards);
}
