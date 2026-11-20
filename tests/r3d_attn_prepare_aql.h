#include <array>

namespace attn_prepare_boundary {
using Launch = int (*)(const RadArgs*, RadStream);
void ensure(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
const RadKernelInfo* registered(Library& library, const char* name) {
    auto count = reinterpret_cast<int (*)()>(dlsym(library.handle, "rad_kernel_count"));
    auto at = reinterpret_cast<const RadKernelInfo* (*)(int)>(dlsym(library.handle, "rad_kernel_at"));
    ensure(count && at, "attention plugin enumeration");
    for (int index = 0; index < count(); ++index) {
        const auto* row = at(index);
        if (row && !std::strcmp(row->name, name)) {
            ensure(row->domain == RAD_DOMAIN_DEVICE && row->launch, "attention registered device row");
            return row;
        }
    }
    throw std::runtime_error(std::string("missing attention row: ") + name);
}
float product(float left, float right) {
    volatile float result = left * right;
    return result;
}
std::vector<float> weight(Model& model, const char* suffix, int dimension) {
    for (const auto& entry : model.entries) {
        const std::string name = model.name(entry);
        const size_t length = std::strlen(suffix);
        if (name.size() < length || name.compare(name.size() - length, length, suffix)) continue;
        ensure(entry.rank == 1 && entry.shape[0] == dimension && entry.n_planes == 1
            && entry.enc < model.encodings.size() && entry.plane_first < model.planes.size(),
            "attention model norm descriptor");
        const auto& encoding = model.encodings[entry.enc];
        const auto& plane = model.planes[entry.plane_first];
        const int dtype = encoding.plane[0].dtype;
        ensure(encoding.n_planes == 1 && rad_enc_is(&encoding, "plain") && !encoding.transform[0]
            && (dtype == RAD_F32 || dtype == RAD_BF16), "attention plain norm encoding");
        ensure(plane.bytes == uint64_t(dimension) * (dtype == RAD_F32 ? 4 : 2)
            && plane.offset >= entry.offset && plane.offset - entry.offset <= entry.bytes
            && plane.bytes <= entry.bytes - (plane.offset - entry.offset), "attention norm plane bounds");
        std::vector<uint8_t> bytes(plane.bytes);
        model.read(plane.offset, bytes.data(), bytes.size());
        std::vector<float> result(dimension);
        uint64_t hash = 14695981039346656037ull;
        for (auto byte : bytes) hash = (hash ^ byte) * 1099511628211ull;
        for (int column = 0; column < dimension; ++column) {
            if (dtype == RAD_F32) std::memcpy(&result[column], bytes.data() + column * 4, 4);
            else {
                uint16_t value;
                std::memcpy(&value, bytes.data() + column * 2, 2);
                result[column] = rad_bf16_to_f32(value);
            }
            ensure(std::isfinite(result[column]), "attention finite model norm");
        }
        std::printf("ATTN_PREPARE_WEIGHT name=%s dim=%d dtype=%d offset=%llu fnv1a64=%016llx\n",
            name.c_str(), dimension, dtype, (unsigned long long)plane.offset, (unsigned long long)hash);
        return result;
    }
    throw std::runtime_error(std::string("missing attention norm: ") + suffix);
}
void expected(std::array<RadTensor, 9>& tensors, int heads, int dimension, float wadd,
              Launch norm, Launch rope, uint64_t& normalized) {
    const int keyheads = heads ? heads / 12 : 4;
    for (int row = 0; row < tensors[1].shape[0]; ++row) {
        for (int head = 0; head < heads + keyheads; ++head) {
            const bool key = head >= heads;
            const int local = key ? head - heads : head;
            const auto& input = tensors[key ? 1 : 0];
            auto& output = tensors[key ? 7 : 6];
            const auto* source = static_cast<const uint16_t*>(input.data)
                + row * input.stride[0] + local * dimension * (key ? 1 : 2);
            const auto* gain = static_cast<const float*>(tensors[key ? 5 : 4].data);
            std::vector<uint16_t> original(source, source + dimension), stage(dimension), reference(dimension);
            std::vector<uint16_t> gate;
            if (!key && tensors[8].data) gate.assign(source + dimension, source + 2 * dimension);
            float sum = 0;
            for (auto value : original) {
                const float decoded = rad_bf16_to_f32(value);
                sum = sum + product(decoded, decoded);
            }
            const float inverse = 1.0f / std::sqrt(sum / float(dimension) + 1e-6f);
            for (int column = 0; column < dimension; ++column)
                stage[column] = rad_f32_to_bf16(product(product(rad_bf16_to_f32(original[column]), inverse),
                    gain[column] + wadd));
            RadTensor source_view = input, output_view = output;
            source_view.rank = output_view.rank = 2;
            source_view.shape[0] = output_view.shape[0] = 1;
            source_view.shape[1] = output_view.shape[1] = dimension;
            source_view.stride[0] = output_view.stride[0] = dimension;
            source_view.stride[1] = output_view.stride[1] = 1;
            source_view.data = original.data(); output_view.data = reference.data();
            RadTensor operands[]{source_view, tensors[key ? 5 : 4], output_view};
            RadParam parameters[]{RAD_INT("n", dimension), RAD_F64("eps", 1e-6), RAD_F64("wadd", wadd)};
            RadArgs args{};
            args.t = operands; args.n_t = 3; args.p = parameters; args.n_p = 3;
            ensure(norm(&args, nullptr) == RAD_OK, "attention libref rmsnorm launch");
            ensure(stage == reference, "attention CPU/libref norm BF16 bit exact");
            normalized += dimension;
            auto* destination = static_cast<uint16_t*>(output.data) + row * output.stride[0] + local * dimension;
            std::copy(stage.begin(), stage.end(), destination);
            int32_t position = static_cast<const int32_t*>(tensors[3].data)[row];
            if (position >= 0 && position < tensors[2].shape[0]) {
                const auto* table = static_cast<const float*>(tensors[2].data) + position * 64;
                for (int pair = 0; pair < 32; ++pair) {
                    const float first = rad_bf16_to_f32(stage[pair]);
                    const float second = rad_bf16_to_f32(stage[pair + 32]);
                    destination[pair] = rad_f32_to_bf16(product(first, table[pair]) - product(second, table[pair + 32]));
                    destination[pair + 32] = rad_f32_to_bf16(product(first, table[pair + 32]) + product(second, table[pair]));
                }
            }
            RadTensor position_view = tensors[3];
            position_view.shape[0] = 1; position_view.data = &position;
            RadTensor rope_operands[]{output_view, position_view};
            RadParam rope_parameters[]{RAD_INT("head_dim", dimension), RAD_INT("n_head", 1),
                RAD_INT("n_head_kv", 0), RAD_INT("rotary_dim", 64), RAD_STR("mode", "neox"),
                RAD_STR("ladder", "rotary"), RAD_F64("theta", 1e7), RAD_F64("scale", 1)};
            RadArgs rope_args{};
            rope_args.t = rope_operands; rope_args.n_t = 2;
            rope_args.p = rope_parameters; rope_args.n_p = 8;
            ensure(rope(&rope_args, nullptr) == RAD_OK, "attention libref rope launch");
            ensure(std::equal(reference.begin(), reference.end(), destination),
                "attention CPU/libref staged norm and rope BF16 bit exact");
            if (!gate.empty()) std::copy(gate.begin(), gate.end(), static_cast<uint16_t*>(tensors[8].data)
                + row * tensors[8].stride[0] + local * dimension);
        }
    }
}
}

TEST(r3d_attn_prepare_actual_aql_views) {
    using namespace attn_prepare_boundary;
    RadDeviceProps properties{};
    REQUIRE_EQ(rad_dev_props(0, &properties), RAD_OK);
    if (properties.is_host_backend) {
        REQUIRE(!std::getenv("RAD_R3D_REQUIRE_DEVICE"));
        SKIP("explicit gfx1100 card run required");
    }
    REQUIRE_EQ(std::strcmp(properties.arch, "gfx1100"), 0);
    REQUIRE_EQ(rad_dev_set(0), RAD_OK);
    Library native("libr3d"), reference("libref");
    const auto* indexer = attn_prepare_boundary::registered(native, "model_qsa_prepare_strict");
    const auto* main = attn_prepare_boundary::registered(native, "model_qk_prepare_strict");
    const auto norm = reference.ngram("rmsnorm")->launch;
    const auto rope = reference.ngram("rope")->launch;
    Model model("/home/zihaomu/bigssd2/radiance_workspace/models/qwen3.8-next-flash-iqr-v4.rad");
    const auto index_weight = weight(model, ".qsa_q_norm.weight", 128);
    const auto query_weight = weight(model, ".attn_q_norm.weight", 256);
    const auto key_weight = weight(model, ".attn_k_norm.weight", 256);
    Queue compute(0), other(1);
    REQUIRE(compute.stream != other.stream);
    uint64_t fixtures = 0, launches = 0, rejects = 0, compared = 0, normalized = 0;
    for (bool full : {false, true}) for (int rows : {1, 3, 63, 64})
        for (int variant = 0; variant < 8; ++variant) for (bool actual_weight : {false, true}) {
            const bool single = std::getenv("RAD_TP1_PREPARE") != nullptr;
            if (single && (!full || rows > 3)) continue;
            const int heads = full ? (single ? 24 : 12) : 0, dimension = full ? 256 : 128;
            const int keyheads = full ? heads / 12 : 4;
            const int query_width = full ? heads * 256 : 3072;
            const int width = full ? keyheads * 256 : (variant & 1 ? 640 : 512);
            const int storage_width = full ? width * 2 : width;
            const int key_pitch = storage_width + (variant & 2 ? 32 : 0);
            const bool inplace = variant & 4;
            const bool gate = full && (variant & 1);
            const auto* kernel = full ? main : indexer;
            std::vector<std::unique_ptr<Buffer>> buffers;
            auto add = [&](int type, std::initializer_list<int64_t> shape, int pitch = 0) {
                buffers.push_back(std::make_unique<Buffer>(type, shape, RAD_MEM_DEVICE, pitch));
                return buffers.back().get();
            };
            Buffer* qg = add(RAD_BF16, {rows, query_width * 2}, query_width * 2 + 16);
            Buffer* key = add(RAD_BF16, {rows, storage_width}, key_pitch);
            Buffer* table = add(RAD_F32, {4097, 64});
            Buffer* positions = add(RAD_I32, {rows});
            Buffer* qw = add(RAD_F32, {dimension});
            Buffer* kw = add(RAD_F32, {dimension});
            Buffer* query = add(RAD_BF16, {rows, query_width}, query_width + 17);
            Buffer* outkey = add(RAD_BF16, {rows, keyheads * dimension}, keyheads * dimension + 17);
            Buffer* outgate = add(RAD_BF16, {rows, query_width}, query_width + 19);
            std::array<RadTensor, 9> host{}, device{};
            for (int index = 0; index < 9; ++index) {
                host[index] = buffers[index]->host;
                device[index] = buffers[index]->device;
            }
            host[1].shape[1] = device[1].shape[1] = width;
            if (inplace) {
                host[7] = host[1]; device[7] = device[1];
                host[7].shape[1] = device[7].shape[1] = keyheads * dimension;
            }
            if (full && (variant & 2)) {
                host[0].rank = device[0].rank = 3;
                host[0].shape[1] = device[0].shape[1] = heads;
                host[0].shape[2] = device[0].shape[2] = 512;
                host[0].stride[1] = device[0].stride[1] = 512;
                host[0].stride[2] = device[0].stride[2] = 1;
            }
            if (!full) for (int index : {0, 4, 6, 8}) host[index] = device[index] = {};
            else if (!gate) host[8] = device[8] = {};
            RadParam parameters[]{RAD_INT("M", 64), RAD_INT("head_dim", dimension), RAD_INT("n_head", heads),
                RAD_INT("n_head_kv", keyheads), RAD_INT("rot", 64), RAD_INT("cs_f32", 1),
                RAD_STR("q_dtype", "bf16"), RAD_F64("eps", 1e-6), RAD_F64("wadd", 0)};
            RadArgs args{};
            args.t = device.data(); args.n_t = 9; args.p = parameters; args.n_p = 9;
            for (int step = 0; step < 2; ++step) {
                RadStream stream = step ? other.stream : compute.stream;
                for (Buffer* buffer : {qg, key, query, outkey, outgate}) {
                    const size_t count = (buffer->expected.size() - 128) / 2;
                    for (size_t index = 0; index < count; ++index)
                        buffer->data<uint16_t>()[index] = rad_f32_to_bf16(
                            float(int((index * 37 + step * 71 + variant * 19) % 509) - 254) / 128.0f);
                }
                for (int column = 0; column < dimension; ++column) {
                    qw->data<float>()[column] = actual_weight ? query_weight[column]
                        : float(column % 19 - 9) / 32.0f;
                    kw->data<float>()[column] = actual_weight ? (full ? key_weight : index_weight)[column]
                        : float(column % 17 - 8) / 16.0f;
                }
                for (int row = 0; row < rows; ++row)
                    positions->data<int32_t>()[row] = row == 0 ? (step ? 4096 : 0)
                        : row == rows - 1 ? (step ? 0 : 4096) : (row * 67 + step * 13) % 4097;
                for (int position = 0; position < 4097; ++position) for (int pair = 0; pair < 32; ++pair) {
                    const float frequency = 1.0f / float(std::pow(1e7, double(float(2 * pair) / 64.0f)));
                    const float angle = product(float(position), frequency);
                    table->data<float>()[position * 64 + pair] = float(std::cos(double(angle)));
                    table->data<float>()[position * 64 + pair + 32] = float(std::sin(double(angle)));
                }
                parameters[8].dval = actual_weight ? 0 : step;
                for (auto& buffer : buffers) buffer->push(stream);
                REQUIRE_EQ(kernel->launch(&args, stream), RAD_OK);
                ++launches;
                attn_prepare_boundary::expected(host, heads, dimension, float(parameters[8].dval), norm, rope, normalized);
                auto reject = [&](std::array<RadTensor, 9> malformed, int error = RAD_E_SHAPE) {
                    RadArgs bad = args; bad.t = malformed.data();
                    CHECK_EQ(kernel->launch(&bad, stream), error);
                    ++rejects;
                };
                for (int bad_width : {255, 511, 513, 639, 641, 768}) {
                    auto bad = device; bad[1].shape[1] = bad_width; reject(bad);
                }
                for (int index : {1, 7}) {
                    auto bad = device; bad[index].stride[0] = bad[index].shape[1] - 1; reject(bad);
                    bad = device; bad[index].stride[1] = 2; reject(bad);
                    bad = device; bad[index].dtype = RAD_F32; reject(bad);
                    bad = device; bad[index].shape[0] = rows + 1; reject(bad);
                    bad = device; bad[index].rank = 4; reject(bad);
                }
                for (int index : {2, 3, 5}) {
                    auto bad = device; bad[index].dtype = RAD_BF16; reject(bad);
                    bad = device; bad[index].data = nullptr; reject(bad);
                }
                auto bad = device; bad[7].shape[1] = keyheads * dimension + 128; reject(bad);
                if (!full) {
                    for (int index : {0, 4, 6, 8}) {
                        bad = device; bad[index] = buffers[index]->device; reject(bad);
                    }
                    if (width == 640) {
                        bad = device; bad[1].rank = 3; bad[1].shape[1] = 5; bad[1].shape[2] = 128;
                        bad[1].stride[1] = 128; bad[1].stride[2] = 1; reject(bad);
                    }
                } else {
                    bad = device; bad[1].shape[1] = keyheads * dimension + 256; reject(bad);
                    for (int index : {0, 4, 6}) {
                        bad = device; bad[index].data = nullptr; reject(bad);
                    }
                    bad = device; bad[8] = outgate->device; bad[8].dtype = RAD_F32; reject(bad);
                }
                for (auto& buffer : buffers) {
                    const auto observed = buffer->read(stream);
                    if (observed != buffer->expected) {
                        size_t offset = 0;
                        while (observed[offset] == buffer->expected[offset]) ++offset;
                        std::fprintf(stderr, "ATTN_PREPARE_DIFF full=%d M=%d width=%d pitch=%d variant=%d actual=%d step=%d buffer=%zu offset=%zu expected=%u actual=%u\n",
                            full, rows, width, key_pitch, variant, actual_weight, step,
                            size_t(&buffer - buffers.data()), offset, buffer->expected[offset], observed[offset]);
                    }
                    REQUIRE(observed == buffer->expected);
                    compared += observed.size();
                }
            }
            ++fixtures;
            std::printf("ATTN_PREPARE_CASE full=%d M=%d band=64 input_width=%d pitch=%d inplace=%d gate=%d actual_norm=%d launches=2 exact=1\n",
                full, rows, width, key_pitch, inplace, gate, actual_weight);
        }
    std::printf("ATTN_PREPARE_AQL_PASS fixtures=%llu launches=%llu rejects=%llu exact_bytes=%llu cpu_libref_pipeline_values=%llu queues=2\n",
        (unsigned long long)fixtures, (unsigned long long)launches, (unsigned long long)rejects,
        (unsigned long long)compared, (unsigned long long)normalized);
}
