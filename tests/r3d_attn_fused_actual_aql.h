#include "../libr3d/r3dh_attn_qsa.h"
namespace attn_fused_actual {
struct Bytes {
    std::vector<uint8_t> expected;
    void* allocation = nullptr;
    const rad::DeviceBackend* backend = nullptr;
    int64_t mapped = 0;
    explicit Bytes(size_t bytes, bool vmem = false) : expected(bytes + 256, 0xa5) {
        if (vmem) {
            backend = rad::active_backend();
            REQUIRE(backend->vmem_granularity && backend->vmem_reserve && backend->vmem_commit_units);
            const int64_t granule = backend->vmem_granularity();
            REQUIRE(granule > 0);
            mapped = (int64_t(expected.size()) + granule - 1) / granule * granule;
            allocation = backend->vmem_reserve(mapped, granule);
            REQUIRE(allocation);
            REQUIRE(backend->vmem_commit_units(allocation, mapped, granule) >= 0);
        } else allocation = rad_dev_alloc(expected.size(), RAD_MEM_DEVICE);
        REQUIRE(allocation);
    }
    ~Bytes() {
        if (backend) {
            backend->vmem_decommit(allocation, mapped);
            backend->vmem_release(allocation, mapped);
        } else rad_dev_free(allocation, RAD_MEM_DEVICE);
    }
    void* device() { return static_cast<uint8_t*>(allocation) + 64; }
    void* host() { return expected.data() + 64; }
    void push(RadStream stream) {
        REQUIRE_EQ(rad_memcpy_async(allocation, expected.data(), expected.size(), stream), RAD_OK);
    }
    std::vector<uint8_t> read(RadStream stream) {
        std::vector<uint8_t> result(expected.size());
        if (rad_memcpy_async(result.data(), allocation, result.size(), stream) != RAD_OK
            || rad_stream_sync(stream) != RAD_OK)
            throw std::runtime_error("fused AQL bytes read failed");
        return result;
    }
};
RadTensor matrix(void* data, int dtype, int rows, int width) {
    RadTensor result{};
    result.data = data; result.dtype = dtype; result.rank = 2;
    result.shape[0] = rows; result.shape[1] = width;
    result.stride[0] = width; result.stride[1] = 1;
    return result;
}
void copy_rows(const RadTensor& source, const RadTensor& destination, int width) {
    if (!destination.data) return;
    for (int row = 0; row < source.shape[0]; ++row)
        std::memcpy(static_cast<uint16_t*>(destination.data) + row * destination.stride[0],
            static_cast<const uint16_t*>(source.data) + row * source.stride[0], width * 2);
}
void cache_oracle(std::vector<uint8_t>& bytes, const std::vector<uint16_t>& key,
                  const std::vector<uint16_t>& value, const int32_t* slots, int rows) {
    for (int row = 0; row < rows; ++row) {
        if (slots[row] < 0 || size_t(slots[row]) >= (bytes.size() - 256) / 516) continue;
        for (int plane = 0; plane < 2; ++plane) {
            const auto& input = plane ? value : key;
            float maximum = 0;
            for (int column = 0; column < 256; ++column)
                maximum = std::max(maximum, std::fabs(rad_bf16_to_f32(input[row * 256 + column])));
            const _Float16 half = static_cast<_Float16>(std::min(maximum / 127.0f, 515.0f));
            const float scale = static_cast<float>(half);
            const size_t offset = 64 + size_t(slots[row]) * 516;
            std::memcpy(bytes.data() + offset + 512 + plane * 2, &half, 2);
            for (int column = 0; column < 256; ++column) {
                const float source = rad_bf16_to_f32(input[row * 256 + column]);
                const float code = scale > 0 ? std::nearbyint(source / scale) : 0;
                const int8_t quantized = int8_t(std::max(-127.0f, std::min(code, 127.0f)));
                std::memcpy(bytes.data() + offset + plane * 256 + column, &quantized, 1);
            }
        }
    }
}
}

void attn_fused_actual_run(bool shared_pool) {
    using namespace attn_fused_actual;
    RadDeviceProps properties{};
    REQUIRE_EQ(rad_dev_props(0, &properties), RAD_OK);
    if (properties.is_host_backend) {
        REQUIRE(!std::getenv("RAD_R3D_REQUIRE_DEVICE"));
        SKIP("explicit gfx1100 card run required");
    }
    REQUIRE_EQ(std::strcmp(properties.arch, "gfx1100"), 0);
    REQUIRE_EQ(rad_dev_set(0), RAD_OK);
    REQUIRE_EQ(std::fesetround(FE_TONEAREST), 0);
    Library native("libr3d"), reference("libref");
    const auto* fused = attn_prepare_boundary::registered(native, "r3dh_attn_fused_prepare_store");
    REQUIRE_EQ(std::strcmp(fused->op, "qk_norm_rope_gate_kv_store"), 0);
    const auto prepare = attn_prepare_boundary::registered(native, "model_qk_prepare_strict")->launch;
    const auto store = attn_prepare_boundary::registered(native, "model_kv_i8_strict")->launch;
    const auto norm = reference.ngram("rmsnorm")->launch;
    const auto rope = reference.ngram("rope")->launch;
    const auto refstore = reference.ngram("kv_store")->launch;
    Queue compute(0), other(1);
    REQUIRE(compute.stream != other.stream);
    uint64_t fixtures = 0, launches = 0, pipelines = 0, rejects = 0, compared = 0, normalized = 0;
    for (int rows : shared_pool ? std::vector<int>{3} : std::vector<int>{1, 3, 63, 64})
    for (int band : shared_pool ? std::vector<int>{64} : std::vector<int>{64, 2048})
    for (int variant = shared_pool ? 4 : 0; variant < (shared_pool ? 8 : 32); ++variant) {
        const bool pitched = variant & 2, gate_present = shared_pool || (variant & 8);
        const bool query_present = shared_pool || !(variant & 4);
        const bool inplace = variant < 16, overwrite_value = variant >= 24;
        const bool virtual_cache = !(variant & 1);
        std::vector<std::unique_ptr<Buffer>> buffers;
        const int pages = shared_pool ? (variant < 6 ? 400026 : r3dh_attn_qsa_max_cache_pages)
            : rows == 63 && band == 64 && variant < 4 ? (variant < 2 ? 50015 : 50512) : 1045;
        if (shared_pool) REQUIRE_EQ(r3dh_attn_qsa_max_cache_pages, 400584);
        auto add = [&](int type, std::initializer_list<int64_t> shape, int pitch = 0) {
            buffers.push_back(std::make_unique<Buffer>(type, shape, RAD_MEM_DEVICE, pitch));
            return buffers.back().get();
        };
        Buffer* qg = add(RAD_BF16, {rows, 6144}, pitched ? 6176 : 6144);
        Buffer* key = add(RAD_BF16, {rows, 256}, pitched ? 288 : 256);
        Buffer* table = add(RAD_F32, {4097, 64});
        Buffer* positions = add(RAD_I32, {rows});
        Buffer* qw = add(RAD_F32, {256});
        Buffer* kw = add(RAD_F32, {256});
        Buffer* query = add(RAD_BF16, {rows, 3072}, pitched ? 3104 : 3072);
        Buffer* outkey = add(RAD_BF16, {rows, 256}, pitched ? 304 : 256);
        Buffer* gated = add(RAD_BF16, {rows, 3072}, pitched ? 3120 : 3072);
        Buffer* value = add(RAD_BF16, {rows, 256}, pitched ? 320 : 256);
        Buffer* slots = add(RAD_I32, {rows});
        Bytes cache(size_t(pages) * 2064, virtual_cache), scratch(size_t(band) * 6656 * 2);
        std::array<RadTensor, 12> host{}, device{};
        for (int index = 0; index < 11; ++index) {
            host[index] = buffers[index]->host; device[index] = buffers[index]->device;
        }
        host[11] = matrix(cache.host(), RAD_I8, pages, 2064);
        device[11] = matrix(cache.device(), RAD_I8, pages, 2064);
        for (auto* tensor : {&host[11], &device[11]}) {
            tensor->rank = 4; tensor->shape[1] = 1; tensor->shape[2] = 4; tensor->shape[3] = 516;
            tensor->stride[1] = 2064; tensor->stride[2] = 516; tensor->stride[3] = 1;
        }
        if (!(variant & 1)) for (auto* tensor : {&host[0], &device[0]}) {
            tensor->rank = 3; tensor->shape[1] = 12; tensor->shape[2] = 512;
            tensor->stride[1] = 512; tensor->stride[2] = 1;
        }
        if (inplace || overwrite_value) {
            host[7] = host[overwrite_value ? 9 : 1]; device[7] = device[overwrite_value ? 9 : 1];
        }
        if (!query_present) host[6] = device[6] = {};
        if (!gate_present) host[8] = device[8] = {};
        RadParam parameters[]{RAD_INT("M", band), RAD_INT("head_dim", 256), RAD_INT("n_head", 12),
            RAD_INT("n_head_kv", 1), RAD_INT("rot", 64), RAD_INT("cs_f32", 1), RAD_STR("q_dtype", "bf16"),
            RAD_INT("block_size", 4), RAD_STR("kv_dtype", "i8"), RAD_F64("eps", 1e-6), RAD_F64("wadd", 0)};
        RadArgs args{}; args.t = device.data(); args.n_t = 12; args.p = parameters; args.n_p = 11;
        const int64_t capacity = int64_t(band) * 6656 * 2;
        REQUIRE_EQ(fused->scratch(&args), capacity);
        args.scratch = scratch.device(); args.scratch_bytes = capacity;
        Buffer direct_query(RAD_BF16, {rows, 3072}, RAD_MEM_DEVICE);
        Buffer direct_key(RAD_BF16, {rows, 256}, RAD_MEM_DEVICE);
        Buffer direct_gate(RAD_BF16, {rows, 3072}, RAD_MEM_DEVICE);
        for (int step = 0; step < 2; ++step) {
            RadStream stream = step ? other.stream : compute.stream;
            for (Buffer* buffer : {qg, key, value, query, outkey, gated}) {
                const int salt = buffer == value ? 131 : buffer == key ? 47 : 0;
                for (size_t index = 0; index < (buffer->expected.size() - 128) / 2; ++index)
                    buffer->data<uint16_t>()[index] = rad_f32_to_bf16(
                        float(int((index * 37 + step * 73 + variant * 19 + salt) % 509) - 254) / 128.0f);
            }
            for (int column = 0; column < 256; ++column) {
                qw->data<float>()[column] = float(column % 19 - 9) / 32.0f;
                kw->data<float>()[column] = float(column % 17 - 8) / 16.0f;
            }
            for (int row = 0; row < rows; ++row) {
                positions->data<int32_t>()[row] = row == 0 ? (step ? 4096 : 0)
                    : row == rows - 1 ? (step ? 0 : 4096) : (row * 67 + step * 13) % 4097;
                slots->data<int32_t>()[row] = shared_pool
                    ? (step ? (row == 0 ? -1 : row == 1 ? pages * 4 : pages * 4 - 1)
                            : (row == 0 ? 0 : pages * 4 - row))
                    : row == 0 ? 0 : row == rows - 1 ? pages * 4 - 1 : row * 53;
            }
            for (int position = 0; position < 4097; ++position) for (int pair = 0; pair < 32; ++pair) {
                const float frequency = 1.0f / float(std::pow(1e7, double(float(2 * pair) / 64.0f)));
                const float angle = attn_prepare_boundary::product(float(position), frequency);
                table->data<float>()[position * 64 + pair] = float(std::cos(double(angle)));
                table->data<float>()[position * 64 + pair + 32] = float(std::sin(double(angle)));
            }
            parameters[0] = step ? RAD_RANGE("M", 1, band) : RAD_INT("M", band);
            parameters[10].dval = step;
            std::vector<std::vector<uint8_t>> initial;
            for (auto& buffer : buffers) { initial.push_back(buffer->expected); buffer->push(stream); }
            cache.push(stream); scratch.push(stream);
            REQUIRE_EQ(rad_stream_sync(stream), RAD_OK);
            const auto cache_initial = cache.expected, scratch_initial = scratch.expected;
            std::vector<uint16_t> query_cpu(rows * 3072), key_cpu(rows * 256), gate_cpu(rows * 3072), saved_value(rows * 256);
            copy_rows(host[9], matrix(saved_value.data(), RAD_BF16, rows, 256), 256);
            std::array<RadTensor, 9> oracle{};
            std::copy_n(host.begin(), 6, oracle.begin());
            oracle[6] = matrix(query_cpu.data(), RAD_BF16, rows, 3072);
            oracle[7] = matrix(key_cpu.data(), RAD_BF16, rows, 256);
            oracle[8] = matrix(gate_cpu.data(), RAD_BF16, rows, 3072);
            attn_prepare_boundary::expected(oracle, 12, 256, float(step), norm, rope, normalized);
            cache_oracle(cache.expected, key_cpu, saved_value, slots->data<int32_t>(), rows);
            auto reference_cache = cache_initial;
            std::vector<int32_t> reference_slots(rows);
            for (int row = 0; row < rows; ++row)
                reference_slots[row] = slots->data<int32_t>()[row] >= 0 && slots->data<int32_t>()[row] < pages * 4
                    ? slots->data<int32_t>()[row] : -1;
            RadTensor reference_store[]{oracle[7], matrix(saved_value.data(), RAD_BF16, rows, 256), matrix(reference_slots.data(), RAD_I32, rows, 1), host[11]};
            reference_store[3].data = reference_cache.data() + 64;
            RadArgs reference_args = args; reference_args.t = reference_store; reference_args.n_t = 4;
            REQUIRE_EQ(refstore(&reference_args, nullptr), RAD_OK);
            REQUIRE(reference_cache == cache.expected);
            copy_rows(oracle[6], host[6], 3072); copy_rows(oracle[7], host[7], 256); copy_rows(oracle[8], host[8], 3072);
            auto* scratch_cpu = static_cast<uint16_t*>(scratch.host());
            std::copy(query_cpu.begin(), query_cpu.end(), scratch_cpu);
            std::copy(gate_cpu.begin(), gate_cpu.end(), scratch_cpu + rows * 3072);
            std::copy(key_cpu.begin(), key_cpu.end(), scratch_cpu + rows * 6144);
            std::copy(saved_value.begin(), saved_value.end(), scratch_cpu + rows * 6400);
            REQUIRE_EQ(fused->launch(&args, stream), RAD_OK); ++launches;
            for (auto& buffer : buffers) {
                REQUIRE(buffer->read(stream) == buffer->expected); compared += buffer->expected.size();
            }
            REQUIRE(cache.read(stream) == cache.expected); compared += cache.expected.size();
            REQUIRE(scratch.read(stream) == scratch.expected); compared += scratch.expected.size();
            for (size_t index = 0; index < buffers.size(); ++index)
                REQUIRE_EQ(rad_memcpy_async(buffers[index]->allocation, initial[index].data(), initial[index].size(), stream), RAD_OK);
            REQUIRE_EQ(rad_memcpy_async(cache.allocation, cache_initial.data(), cache_initial.size(), stream), RAD_OK);
            RadTensor direct_prepare[]{device[0], device[1], device[2], device[3], device[4], device[5],
                direct_query.device, direct_key.device, direct_gate.device};
            RadArgs call = args; call.t = direct_prepare; call.n_t = 9;
            REQUIRE_EQ(prepare(&call, stream), RAD_OK);
            RadTensor direct_store[]{direct_key.device, device[9], device[10], device[11]};
            call.t = direct_store; call.n_t = 4;
            REQUIRE_EQ(store(&call, stream), RAD_OK);
            for (int index : {6, 7, 8}) if (device[index].data) {
                const RadTensor source = index == 6 ? direct_query.device : index == 7 ? direct_key.device : direct_gate.device;
                REQUIRE_EQ(rad_memcpy_2d_async(device[index].data, device[index].stride[0] * 2,
                    source.data, source.stride[0] * 2, source.shape[1] * 2, rows, stream), RAD_OK);
            }
            for (auto& buffer : buffers) {
                REQUIRE(buffer->read(stream) == buffer->expected); compared += buffer->expected.size();
            }
            REQUIRE(cache.read(stream) == cache.expected); compared += cache.expected.size(); ++pipelines;
            const auto scratch_before = scratch.read(stream);
            auto reject = [&](std::array<RadTensor, 12> malformed, int error = RAD_E_SHAPE,
                              int count = 12, int64_t supplied_scratch = -1) {
                RadArgs bad = args; bad.t = malformed.data(); bad.n_t = count;
                if (supplied_scratch >= 0) bad.scratch_bytes = supplied_scratch;
                REQUIRE_EQ(fused->launch(&bad, reinterpret_cast<RadStream>(uintptr_t(1))), error); ++rejects;
            };
            auto malformed = device;
            malformed[0].shape[0] = band + 1; reject(malformed);
            malformed = device; malformed[0].shape[0] = 0; reject(malformed);
            malformed = device; malformed[7].shape[0] = rows + 1; reject(malformed);
            malformed = device; malformed[1].shape[0] = rows + 1; reject(malformed);
            malformed = device; malformed[11].shape[0] = r3dh_attn_qsa_max_cache_pages + 1; reject(malformed);
            malformed = device; malformed[9].stride[0] = 255; reject(malformed);
            malformed = device; malformed[0].stride[0] = 6143; reject(malformed);
            malformed = device; malformed[0].stride[0] = (1ll << 30) + 1; reject(malformed);
            malformed = device; malformed[0].rank = 4; reject(malformed);
            malformed = device; malformed[0].rank = 3; malformed[0].shape[1] = 12; malformed[0].shape[2] = 512;
            malformed[0].stride[1] = 513; malformed[0].stride[2] = 1; reject(malformed);
            malformed = device; malformed[0].dtype = RAD_F32; reject(malformed);
            malformed = device; malformed[11].stride[0] = 2065; reject(malformed);
            malformed = device; malformed[7].data = device[11].data; reject(malformed);
            if (shared_pool) {
                malformed = device; malformed[11].dtype = RAD_BF16; reject(malformed);
                malformed = device; malformed[11].shape[2] = 5; reject(malformed);
                malformed = device; malformed[11].stride[2] = 517; reject(malformed);
                for (int invalid = 0; invalid < 4; ++invalid) {
                    RadTensor bad_store[]{direct_key.device, device[9], device[10], device[11]};
                    if (invalid == 0) bad_store[3].shape[0] = r3dh_attn_qsa_max_cache_pages + 1;
                    if (invalid == 1) bad_store[3].dtype = RAD_BF16;
                    if (invalid == 2) bad_store[3].stride[0] = 2065;
                    if (invalid == 3) bad_store[3].shape[2] = 5;
                    RadArgs bad = args; bad.t = bad_store; bad.n_t = 4;
                    REQUIRE_EQ(store(&bad, reinterpret_cast<RadStream>(uintptr_t(1))), RAD_E_SHAPE);
                    ++rejects;
                }
            }
            malformed = device; malformed[8] = query->device; malformed[6] = query->device; reject(malformed);
            reject(device, RAD_E_INVAL, 11); reject(device, RAD_E_INVAL, 13);
            reject(device, RAD_E_INVAL, 12, capacity - 1);
            if (rows < band) reject(device, RAD_E_INVAL, 12, int64_t(rows) * 6656 * 2);
            RadArgs missing = args; missing.t = nullptr;
            REQUIRE_EQ(fused->launch(&missing, reinterpret_cast<RadStream>(uintptr_t(1))), RAD_E_INVAL); ++rejects;
            REQUIRE_EQ(fused->launch(nullptr, reinterpret_cast<RadStream>(uintptr_t(1))), RAD_E_INVAL); ++rejects;
            const auto saved_band = parameters[0];
            parameters[0] = RAD_INT("M", rows - 1); reject(device, rows == 1 ? RAD_E_UNSUPPORTED : RAD_E_SHAPE);
            parameters[0] = RAD_INT("M", 2049); reject(device, RAD_E_UNSUPPORTED);
            parameters[0] = saved_band;
            for (auto& buffer : buffers) REQUIRE(buffer->read(stream) == buffer->expected);
            REQUIRE(cache.read(stream) == cache.expected);
            REQUIRE(scratch.read(stream) == scratch_before);
            cache.expected = cache_initial; scratch.expected = scratch_initial;
        }
        ++fixtures;
    }
    std::printf("PASS %s fixtures=%llu fused_launches=%llu direct_pipelines=%llu rejects=%llu exact_device_bytes=%llu independent_norm_rope_values=%llu queues=2 cache=DEVICE+VMEM context_positions=0,4096 tolerance=0\n",
        shared_pool ? "ATTN_SHARED_PHYSICAL_POOL_AQL pages=400026,400584 rows=3 band=64 final_slot_and_invalid_indices" : "ATTN_FUSED_ACTUAL_AQL rows=1,3,63,64 bands=64,2048",
        (unsigned long long)fixtures, (unsigned long long)launches, (unsigned long long)pipelines,
        (unsigned long long)rejects, (unsigned long long)compared, (unsigned long long)normalized);
}
TEST(r3d_attn_fused_actual_aql_rows) { attn_fused_actual_run(false); }
TEST(r3d_attn_shared_physical_pool_aql) { attn_fused_actual_run(true); }
