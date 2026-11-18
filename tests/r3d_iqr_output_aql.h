#include "../libr3d/r3d_iqr.h"

namespace iqr_output_boundary {
RadTensor matrix(void* data, uint32_t dtype, int64_t rows, int64_t columns) {
    RadTensor result{};
    result.data = data; result.dtype = dtype; result.rank = 2;
    result.shape[0] = rows; result.shape[1] = columns;
    rad_tensor_pack(&result);
    return result;
}
int independent_mad(uint32_t state, uint32_t multiplier, uint32_t increment) {
    const uint32_t word = uint32_t(uint64_t(state % 16777216u)
        * (multiplier % 16777216u) + increment);
    int total = -126;
    for (unsigned byte = 0; byte < 4; ++byte) total += (word >> (8 * byte)) & 63u;
    return total;
}
struct Fixture {
    int columns, reduction;
    std::vector<uint8_t> weight, scale;
    std::vector<int8_t> decoded;
    std::vector<float> factors, column_factors;
    Fixture(const RadKernelInfo* row, int width, int inner, bool actual)
        : columns(width), reduction(inner), factors(width), column_factors(inner / 128) {
        constexpr unsigned bits = 5, run = 256;
        const size_t code_pitch = size_t(inner) * bits / 8;
        std::vector<uint8_t> codes(size_t(width) * code_pitch);
        std::vector<int8_t> table(65536);
        uint32_t coefficients[2]{0xcd123457u, 0xfedcba98u};
        RadEncoding encoding{};
        rad_enc_copy_str(encoding.scheme, "trellis");
        rad_enc_copy_str(encoding.transform, "fwht128");
        rad_enc_add_plane(&encoding, "codes", RAD_U5, 1, 1);
        rad_enc_add_plane(&encoding, "scale", RAD_F32, 1, 0);
        const int trellis = rad_enc_add_table(&encoding, "trellis", RAD_I8, 65536, 1);
        encoding.plane[trellis].block[0] = run;
        encoding.plane[trellis].block[1] = 1;
        rad_enc_add_table(&encoding, "trellis.mad", RAD_U32, 2, 1);
        rad_enc_add_plane(&encoding, "scale.col", RAD_F32, 0, 128);
        uint32_t random_state = 20261118u;
        auto random = [&] { random_state = random_state * 1664525u + 1013904223u; return random_state; };
        for (auto& value : codes) value = uint8_t(random() >> 16);
        for (auto& value : factors) value = float(1 + random() % 127) / 8192.0f;
        for (auto& value : column_factors) value = float(1 + random() % 127) / 128.0f;
        if (actual) {
            Model model("/home/zihaomu/bigssd2/radiance_workspace/models/qwen3.8-next-flash-iqr-v4.rad");
            bool found = false;
            for (const auto& entry : model.entries) {
                if (model.name(entry) != "blk.3.attn_qg.weight") continue;
                REQUIRE_EQ(entry.rank, 2u);
                REQUIRE_EQ(entry.shape[0], 12288);
                REQUIRE_EQ(entry.shape[1], inner);
                REQUIRE_EQ(width, 6144);
                REQUIRE(entry.enc < model.encodings.size());
                encoding = model.encodings[entry.enc];
                REQUIRE(rad_enc_valid(&encoding));
                REQUIRE_EQ(entry.n_planes, 5u);
                REQUIRE_EQ(encoding.n_planes, 5);
                for (unsigned index = 0; index < entry.n_planes; ++index) {
                    REQUIRE(entry.plane_first + index < model.planes.size());
                    const auto& location = model.planes[entry.plane_first + index];
                    REQUIRE(location.offset >= entry.offset);
                    REQUIRE(location.offset - entry.offset <= entry.bytes);
                    REQUIRE(location.bytes <= entry.bytes - (location.offset - entry.offset));
                    const char* role = encoding.plane[index].role;
                    void* destination = nullptr;
                    size_t bytes = 0;
                    if (!std::strcmp(role, "codes")) { destination = codes.data(); bytes = codes.size(); }
                    else if (!std::strcmp(role, "scale")) { destination = factors.data(); bytes = factors.size() * 4; }
                    else if (!std::strcmp(role, "trellis")) { destination = table.data(); bytes = table.size(); }
                    else if (!std::strcmp(role, "trellis.mad")) { destination = coefficients; bytes = sizeof(coefficients); }
                    else if (!std::strcmp(role, "scale.col")) { destination = column_factors.data(); bytes = column_factors.size() * 4; }
                    REQUIRE(destination);
                    REQUIRE(bytes <= location.bytes);
                    model.read(location.offset, destination, bytes);
                }
                found = true;
                std::printf("IQR_OUTPUT_WEIGHT name=blk.3.attn_qg.weight source_N=12288 row_slice=0:6144 K=%d encoding=actual not_full_tp2_shard=1\n", inner);
                break;
            }
            REQUIRE(found);
        } else {
            for (uint32_t state = 0; state < 65536; ++state)
                table[state] = int8_t(independent_mad(state, coefficients[0], coefficients[1]));
        }
        const int weight_selected[]{0, 2, 3}, scale_selected[]{1, 4};
        RadTensor weight_planes[]{matrix(codes.data(), RAD_U5, width, inner),
            matrix(table.data(), RAD_I8, 65536, 1), matrix(coefficients, RAD_U32, 2, 1)};
        RadTensor scale_planes[]{matrix(factors.data(), RAD_F32, width, 1),
            matrix(column_factors.data(), RAD_F32, 1, inner / 128)};
        const RadParam params[]{RAD_INT("N", width), RAD_INT("K", inner),
            RAD_INT("group", 128), RAD_INT("w_rb", 1), RAD_STR("dtype", "i8a8h")};
        RadLayout weight_layout{}, scale_layout{};
        REQUIRE(row->layout && row->relayout);
        REQUIRE_EQ(row->layout(params, 5, 2, &encoding, weight_selected, weight_planes, 3, &weight_layout), RAD_OK);
        REQUIRE_EQ(row->layout(params, 5, 3, &encoding, scale_selected, scale_planes, 2, &scale_layout), RAD_OK);
        REQUIRE_EQ(weight_layout.dtype, r3d_iqr::weight_dtype);
        REQUIRE_EQ(scale_layout.dtype, r3d_iqr::scale_dtype);
        weight.resize(weight_layout.bytes); scale.resize(scale_layout.bytes);
        REQUIRE_EQ(row->relayout(params, 5, 2, &encoding, weight_selected, weight_planes, 3,
            weight.data(), weight.size()), RAD_OK);
        REQUIRE_EQ(row->relayout(params, 5, 3, &encoding, scale_selected, scale_planes, 2,
            scale.data(), scale.size()), RAD_OK);
        decoded.resize(size_t(width) * inner);
        for (int start = 0; start < width; start += run)
            for (int position = 0; position < inner; ++position) {
                uint32_t state = 0;
                auto code = [&](int column) {
                    unsigned result = 0;
                    for (unsigned digit = 0; digit < bits; ++digit) {
                        const unsigned bit = position * bits + digit;
                        result |= ((codes[size_t(column) * code_pitch + bit / 8] >> (bit % 8)) & 1u) << digit;
                    }
                    return result;
                };
                for (int column = start + run - (16 + bits - 1) / bits; column < start + int(run); ++column)
                    state = ((state << bits) | code(column)) & 65535u;
                for (int column = start; column < start + int(run); ++column) {
                    state = ((state << bits) | code(column)) & 65535u;
                    decoded[size_t(column) * inner + position] = int8_t(independent_mad(state, coefficients[0], coefficients[1]));
                }
            }
    }
};
int8_t round_even(float value) {
    const float lower = std::floor(value);
    int result = int(lower);
    const float fraction = value - lower;
    if (fraction > 0.5f || (fraction == 0.5f && (result & 1))) ++result;
    return int8_t(result);
}
void quant_expected(Buffer& input, Buffer& codes, Buffer& scales, int rows, int reduction) {
    for (int row = 0; row < rows; ++row)
        for (int group = 0; group < reduction / 128; ++group) {
            float values[128];
            for (int position = 0; position < 128; ++position)
                values[position] = rad_bf16_to_f32(input.data<uint16_t>()[row * input.host.stride[0] + group * 128 + position]);
            for (int distance = 1; distance < 128; distance *= 2)
                for (int base = 0; base < 128; base += distance * 2)
                    for (int offset = 0; offset < distance; ++offset) {
                        const float left = values[base + offset], right = values[base + distance + offset];
                        values[base + offset] = left + right;
                        values[base + distance + offset] = left - right;
                    }
            float maximum = 0;
            for (float value : values) maximum = std::max(maximum, std::fabs(value));
            if (!(maximum > 0)) maximum = 1e-8f;
            scales.data<float>()[row * scales.host.stride[0] + group] = maximum / 127.0f;
            const float inverse = 127.0f / maximum;
            for (int position = 0; position < 128; ++position)
                codes.data<int8_t>()[row * codes.host.stride[0] + group * 128 + position] =
                    round_even(std::max(-127.0f, std::min(127.0f, values[position] * inverse)));
        }
}
uint16_t oracle(const Fixture& fixture, Buffer& activation, Buffer& scales, int row, int column) {
    float accumulator = 0;
    for (int group = 0; group < fixture.reduction / 128; ++group) {
        int subtotal = 0;
        for (int offset = 0; offset < 128; ++offset) {
            const int position = group * 128 + offset;
            subtotal += int(activation.data<int8_t>()[row * activation.host.stride[0] + position])
                * int(fixture.decoded[size_t(column) * fixture.reduction + position]);
        }
        volatile float scaled = float(subtotal) * scales.data<float>()[row * scales.host.stride[0] + group];
        volatile float product = scaled * fixture.column_factors[group];
        accumulator = accumulator + product;
    }
    return rad_f32_to_bf16(accumulator * fixture.factors[column]);
}
}

TEST(r3d_iqr_actual_aql_output_views) {
    using namespace iqr_output_boundary;
    RadDeviceProps properties{};
    REQUIRE_EQ(rad_dev_props(0, &properties), RAD_OK);
    if (properties.is_host_backend) {
        REQUIRE(!std::getenv("RAD_R3D_REQUIRE_DEVICE"));
        SKIP("explicit gfx1100 card run required");
    }
    REQUIRE_EQ(std::strcmp(properties.arch, "gfx1100"), 0);
    REQUIRE_EQ(rad_dev_set(0), RAD_OK);
    REQUIRE_EQ(std::fesetround(FE_TONEAREST), 0);
    Library native("libr3d");
    const auto* prefill = moe_boundary::registered(native, "gemm_nt_q_iqr_prefill");
    const auto* decode = moe_boundary::registered(native, "gemm_nt_q_iqr_decode");
    const auto* producer = moe_boundary::registered(native, "qwen_had_quant_i8_rowgroup");
    Queue compute(0), other(1);
    uint64_t fixtures = 0, launches = 0, rejects = 0, exact_bytes = 0, independent = 0, producers = 0;
    for (int source = 0; source < 3; ++source) {
        const int columns = source ? 6144 : 256;
        const int reduction = source ? 2560 : 256;
        const int groups = reduction / 128;
        Fixture fixture(prefill, columns, reduction, source == 2);
        Buffer weight(RAD_I32, {1, int64_t((fixture.weight.size() + 3) / 4)}, RAD_MEM_DEVICE);
        Buffer weight_scale(RAD_F32, {1, int64_t(fixture.scale.size() / 4)}, RAD_MEM_DEVICE);
        std::memcpy(weight.host.data, fixture.weight.data(), fixture.weight.size());
        std::memcpy(weight_scale.host.data, fixture.scale.data(), fixture.scale.size());
        weight.device = matrix(weight.device.data, r3d_iqr::weight_dtype, columns, reduction);
        for (int rows : {1, 3, 63, 64, 65}) {
            if (source && rows != 3 && rows != 63) continue;
            const auto* gemm = rows <= 2 ? decode : prefill;
            const int band = rows <= 64 ? 64 : 128;
            for (int padded : {0, 1})
                for (bool logical_scale : {false, true}) {
                    Buffer input(RAD_BF16, {rows + 2, groups, 128}, RAD_MEM_DEVICE, reduction + padded * 16);
                    Buffer codes(RAD_I8, {rows + 2, reduction}, RAD_MEM_DEVICE, reduction + padded * 16);
                    Buffer scales(RAD_F32, {rows + 2, groups}, RAD_MEM_DEVICE, groups + padded * 4);
                    Buffer output(RAD_BF16, {rows + 2, columns}, RAD_MEM_DEVICE, columns + padded * 32);
                    for (Buffer* buffer : {&input, &codes, &scales, &output})
                        buffer->host.shape[0] = buffer->device.shape[0] = rows;
                    weight_scale.device = matrix(weight_scale.device.data, r3d_iqr::scale_dtype,
                        columns, logical_scale ? reduction : groups);
                    std::vector<uint8_t> previous;
                    for (int step = 0; step < 2; ++step) {
                        RadStream stream = step ? compute.stream : other.stream;
                        REQUIRE_EQ(rad_stream_device_access(stream, 0, nullptr), RAD_OK);
                        for (int row = 0; row < rows; ++row)
                            for (int position = 0; position < reduction; ++position)
                                input.data<uint16_t>()[row * input.host.stride[0] + position] = rad_f32_to_bf16(
                                    float((row * 197 + position * 29 + position / 128 * 17 + step * 71) % 509 - 254) / 32.0f);
                        for (Buffer* buffer : {&codes, &scales, &output})
                            std::fill(buffer->expected.begin(), buffer->expected.end(), uint8_t(0x5a));
                        for (Buffer* buffer : {&input, &codes, &scales, &weight, &weight_scale, &output}) buffer->push(stream);
                        REQUIRE_EQ(rad_stream_sync(stream), RAD_OK);
                        quant_expected(input, codes, scales, rows, reduction);
                        RadTensor quant_tensors[]{input.device, codes.device, scales.device};
                        RadParam quant_params[]{integer("M", band), integer("n", reduction), integer("group", 128)};
                        RadArgs quant_args{};
                        quant_args.t = quant_tensors; quant_args.n_t = 3;
                        quant_args.p = quant_params; quant_args.n_p = 3;
                        REQUIRE_EQ(producer->launch(&quant_args, stream), RAD_OK);
                        ++producers;
                        REQUIRE(codes.read(stream) == codes.expected);
                        REQUIRE(scales.read(stream) == scales.expected);
                        std::vector<uint16_t> witness(size_t(rows) * columns);
                        for (int row = 0; row < rows; ++row)
                            for (int column = 0; column < columns; ++column)
                                if (rows <= 3 || source == 0 || column % 97 == 0 || column == columns - 1) {
                                    witness[size_t(row) * columns + column] = oracle(fixture, codes, scales, row, column);
                                    ++independent;
                                }
                        std::vector<uint8_t> canonical;
                        for (int implementation = 0; implementation < 3; ++implementation) {
                            std::vector<uint8_t> baseline;
                            for (int rank : {2, 3, 4}) {
                                output.push(stream);
                                std::array<RadTensor, 7> tensors{{codes.device, scales.device, weight.device, weight_scale.device, output.device, {}, {}}};
                                if (rank == 3) {
                                    tensors[4].rank = 3;
                                    tensors[4].shape[1] = source ? 12 : 2;
                                    tensors[4].shape[2] = source ? 512 : 128;
                                    tensors[4].stride[1] = tensors[4].shape[2]; tensors[4].stride[2] = 1;
                                } else if (rank == 4) {
                                    tensors[4].rank = 4;
                                    tensors[4].shape[1] = source ? 3 : 1;
                                    tensors[4].shape[2] = source ? 4 : 2;
                                    tensors[4].shape[3] = source ? 512 : 128;
                                    tensors[4].stride[1] = tensors[4].shape[2] * tensors[4].shape[3];
                                    tensors[4].stride[2] = tensors[4].shape[3]; tensors[4].stride[3] = 1;
                                }
                                RadParam params[]{RAD_INT("M", band), RAD_INT("N", columns), RAD_INT("K", reduction),
                                    RAD_INT("group", 128), RAD_INT("w_rb", 1), RAD_INT("impl", implementation), RAD_STR("dtype", "i8a8h")};
                                RadArgs args{}; args.t = tensors.data(); args.n_t = 7; args.p = params; args.n_p = 7;
                                const auto original = tensors;
                                REQUIRE_EQ(gemm->launch(&args, stream), RAD_OK);
                                REQUIRE_EQ(std::memcmp(tensors.data(), original.data(), sizeof(tensors)), 0);
                                ++launches;
                                auto actual = output.read(stream);
                                if (rank == 2) {
                                    auto guarded = output.expected;
                                    for (int row = 0; row < rows; ++row) {
                                        const size_t offset = 64 + row * output.host.stride[0] * 2;
                                        std::memcpy(guarded.data() + offset, actual.data() + offset, columns * 2);
                                        for (int column = 0; column < columns; ++column)
                                            if (rows <= 3 || source == 0 || column % 97 == 0 || column == columns - 1) {
                                                uint16_t value;
                                                std::memcpy(&value, actual.data() + offset + column * 2, 2);
                                                if (value != witness[size_t(row) * columns + column])
                                                    throw std::runtime_error("IQR independent output mismatch source=" + std::to_string(source)
                                                        + " M=" + std::to_string(rows) + " impl=" + std::to_string(implementation));
                                            }
                                    }
                                    REQUIRE(actual == guarded);
                                    baseline = actual;
                                    if (!implementation) canonical = actual;
                                    else REQUIRE(actual == canonical);
                                } else REQUIRE(actual == baseline);
                                exact_bytes += actual.size();
                                if (rank != 3) continue;
                                auto reject = [&](auto mutate, int expected) {
                                    auto bad = tensors;
                                    std::array<RadParam, 7> bad_params;
                                    std::copy(std::begin(params), std::end(params), bad_params.begin());
                                    RadArgs invalid = args; invalid.t = bad.data(); invalid.p = bad_params.data();
                                    mutate(bad, bad_params);
                                    REQUIRE_EQ(gemm->launch(&invalid, stream), expected);
                                    ++rejects;
                                };
                                reject([](auto& bad, auto&) { bad[4].stride[1]++; }, RAD_E_STRIDE);
                                reject([](auto& bad, auto&) { bad[4].stride[2] = -1; }, RAD_E_STRIDE);
                                reject([](auto& bad, auto&) { bad[4].stride[0] = -1; }, RAD_E_STRIDE);
                                reject([](auto& bad, auto&) { bad[4].stride[0] = 0; }, RAD_E_STRIDE);
                                reject([&](auto& bad, auto&) { bad[4].stride[0] = columns - 1; }, RAD_E_STRIDE);
                                reject([](auto& bad, auto&) { bad[4].stride[0] = INT64_MAX; }, RAD_E_STRIDE);
                                if (rows > 1)
                                    reject([](auto& bad, auto&) { bad[4].stride[0] = INT64_MAX / 4; }, RAD_E_STRIDE);
                                reject([](auto& bad, auto&) { bad[4].shape[1]++; }, RAD_E_SHAPE);
                                reject([](auto& bad, auto&) { bad[4].shape[0]++; }, RAD_E_SHAPE);
                                reject([](auto& bad, auto&) { bad[4].shape[1] = 0; }, RAD_E_SHAPE);
                                reject([](auto& bad, auto&) { bad[4].shape[1] = INT64_MAX; }, RAD_E_SHAPE);
                                reject([](auto& bad, auto&) { bad[4].rank = 1; }, RAD_E_SHAPE);
                                reject([](auto& bad, auto&) { bad[4].rank = 0; }, RAD_E_SHAPE);
                                reject([](auto& bad, auto&) { bad[4].rank = 99; }, RAD_E_SHAPE);
                                reject([](auto& bad, auto&) { bad[4].data = nullptr; }, RAD_E_DTYPE);
                                reject([](auto& bad, auto&) { bad[4].dtype = RAD_F32; }, RAD_E_DTYPE);
                                reject([](auto& bad, auto&) { bad[4].data = reinterpret_cast<void*>(UINTPTR_MAX - 1); }, RAD_E_STRIDE);
                                reject([&](auto&, auto& bad) { bad[0].ival = rows - 1; }, RAD_E_SHAPE);
                                reject([](auto&, auto& bad) { bad[0].ival = 2049; }, RAD_E_SHAPE);
                                reject([](auto& bad, auto&) { bad[3].shape[1] = 17; }, RAD_E_SHAPE);
                                REQUIRE(output.read(stream) == baseline);
                            }
                        }
                        if (step) REQUIRE(canonical != previous);
                        previous = canonical;
                        for (Buffer* buffer : {&input, &codes, &scales, &weight, &weight_scale})
                            REQUIRE(buffer->read(stream) == buffer->expected);
                    }
                    ++fixtures;
                    std::printf("IQR_OUTPUT_CASE source=%s M=%d band=%d N=%d K=%d heads=%d width=%d pitched=%d logical_scale=%d impl=0,1,2 rank=2,3,4 steps=2 exact=1\n",
                        source == 2 ? "actual_attn_qg_row_slice" : "synthetic", rows, band, columns, reduction,
                        source ? 12 : 2, source ? 512 : 128, padded, int(logical_scale));
                    std::fflush(stdout);
                }
        }
    }
    std::printf("PASS IQR_OUTPUT_AQL fixtures=%llu producer_launches=%llu gemm_launches=%llu rejects=%llu exact_bytes=%llu independent_values=%llu queues=2 tolerance=0 full_M63_cpu=sampled full_M3_cpu=exhaustive\n",
        (unsigned long long)fixtures, (unsigned long long)producers, (unsigned long long)launches,
        (unsigned long long)rejects, (unsigned long long)exact_bytes, (unsigned long long)independent);
}
