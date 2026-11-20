#include "format/radfile.h"
#include "format/share.h"
#include "format/encoding.h"
#include "../arch/common/rad_fp8.h"
#include "rad_quant.h"

namespace iqr_group {
struct Fixture {
    RadEncoding encoding{};
    int columns = 0, reduction = 0;
    std::vector<std::vector<uint8_t>> storage;
    std::vector<RadTensor> planes;
    std::vector<uint8_t> weight, scale;
    std::vector<int8_t> decoded;
    int selected[3]{0, 2, 3}, scale_selected[1]{1};
    Fixture() = default;
    Fixture(const RadKernelInfo* row, rad::RadFile& file, int rank) {
        const auto* entry = file.find("blk.48.attn_output.weight");
        REQUIRE(entry);
        encoding = file.encoding(*entry);
        char text[512]; rad_enc_format(&encoding, text, sizeof(text));
        std::printf("IQR_MTP_MODEL rank=%d full_N=%lld full_K=%lld encoding=%s class=%d retired=%s\n",
            rank, (long long)entry->shape[0], (long long)entry->shape[1], text,
            int(rad::arch::linear_w8h_form(encoding)),
            rad::arch::linear_retired(encoding) ? rad::arch::linear_retired(encoding) : "none");
        REQUIRE_EQ(entry->rank, 2u); REQUIRE_EQ(entry->shape[0], 2560);
        REQUIRE_EQ(entry->shape[1], 6144); REQUIRE_EQ(encoding.n_planes, 4);
        REQUIRE_EQ(encoding.plane[0].dtype, uint32_t(RAD_U4));
        REQUIRE_EQ(encoding.plane[1].dtype, uint32_t(RAD_BF16));
        columns = int(entry->shape[0]); reduction = int(entry->shape[1] / 2);
        storage.resize(4); planes.resize(4);
        for (int operand : {2, 3}) {
            rad::WeightInfo info;
            info.decl.rank = 2; info.decl.shard = RAD_SHARD_COL;
            info.decl.dtype = operand == 2 ? RAD_U4 : RAD_BF16;
            info.decl.shape[0] = columns;
            info.decl.shape[1] = operand == 2 ? reduction : reduction / 128;
            info.n_sel = operand == 2 ? 3 : 1;
            for (int index = 0; index < info.n_sel; ++index) {
                info.sel[index] = operand == 2 ? selected[index] : scale_selected[index];
                rad_enc_plane_dims(&encoding.plane[info.sel[index]], columns, reduction,
                    &info.sel_rows[index], &info.sel_cols[index]);
            }
            rad::PlaneShare shares[RAD_ENC_MAX_PLANES];
            std::string why;
            REQUIRE_EQ(rad::weight_share(info, encoding, entry->rank, entry->shape,
                2, rank, shares, &why), RAD_OK);
            for (int index = 0; index < info.n_sel; ++index) {
                const int plane = info.sel[index];
                const auto& share = shares[index];
                storage[plane].resize(share.dense_bytes());
                size_t cursor = 0;
                const auto* source = file.data(file.plane(*entry, plane));
                for (const auto& rect : share.rects)
                    for (int64_t row_index = 0; row_index < rect.rows; ++row_index) {
                        REQUIRE(cursor + rect.bytes <= storage[plane].size());
                        std::memcpy(storage[plane].data() + cursor,
                            source + (rect.row0 + row_index) * share.row_bytes + rect.byte0, rect.bytes);
                        cursor += rect.bytes;
                    }
                REQUIRE_EQ(cursor, storage[plane].size());
                planes[plane] = iqr_output_boundary::matrix(storage[plane].data(), share.dtype,
                    share.rows, share.cols);
                std::printf("IQR_MTP_PART rank=%d plane=%s rows=%lld cols=%lld bytes=%zu rects=%zu\n",
                    rank, encoding.plane[plane].role, (long long)share.rows,
                    (long long)share.cols, cursor, share.rects.size());
            }
        }
        RadParam params[]{RAD_INT("N", columns), RAD_INT("K", reduction),
            RAD_INT("group", 128), RAD_STR("dtype", "i8a8h")};
        RadTensor weight_planes[]{planes[0], planes[2], planes[3]};
        for (int operand : {2, 3}) {
            RadLayout layout{};
            const auto* tensors = operand == 2 ? weight_planes : &planes[1];
            const int* selection = operand == 2 ? selected : scale_selected;
            const int count = operand == 2 ? 3 : 1;
            REQUIRE_EQ(row->layout(params, 4, operand, &encoding, selection, tensors, count, &layout), RAD_OK);
            auto& target = operand == 2 ? weight : scale;
            target.resize(layout.bytes);
            REQUIRE_EQ(row->relayout(params, 4, operand, &encoding, selection, tensors,
                count, target.data(), target.size()), RAD_OK);
            if (operand == 3) REQUIRE_EQ(layout.dtype, r3d_iqr::group_scale_dtype);
        }
        decoded.resize(size_t(columns) * reduction);
        const unsigned bits = rad_dtype_bits(encoding.plane[0].dtype);
        const int run = int(encoding.plane[2].block[0]);
        const size_t pitch = size_t(reduction) * bits / 8;
        const auto* table = reinterpret_cast<const int8_t*>(storage[2].data());
        auto code = [&](int row_index, int column) {
            unsigned value = 0;
            for (unsigned digit = 0; digit < bits; ++digit) {
                const size_t bit = size_t(column) * bits + digit;
                value |= unsigned((storage[0][row_index * pitch + bit / 8] >> (bit % 8)) & 1) << digit;
            }
            return value;
        };
        for (int start = 0; start < columns; start += run)
            for (int column = 0; column < reduction; ++column) {
                unsigned state = 0;
                for (int previous = run - (16 + bits - 1) / bits; previous < run; ++previous)
                    state = ((state << bits) | code(start + previous, column)) % 65536;
                for (int row_index = start; row_index < start + run; ++row_index) {
                    state = ((state << bits) | code(row_index, column)) % 65536;
                    decoded[size_t(row_index) * reduction + column] = table[state];
                }
            }
        void* library = dlopen((home() + "/quantizers/libquant.so").c_str(), RTLD_NOW | RTLD_LOCAL);
        REQUIRE(library);
        auto count = reinterpret_cast<int (*)()>(dlsym(library, "rad_quant_count"));
        auto at = reinterpret_cast<const RadQuantizerInfo* (*)(int)>(dlsym(library, "rad_quant_at"));
        REQUIRE(count && at);
        RadQuantDecodeFn decoder = nullptr;
        for (int index = 0; index < count(); ++index)
            if (std::strcmp(at(index)->name, "gptq") == 0) decoder = at(index)->decode;
        REQUIRE(decoder);
        RadEncoding grid = encoding;
        grid.transform[0] = 0;
        std::vector<uint16_t> unity(size_t(columns) * (reduction / 128), 0x3f80);
        const void* data[]{planes[0].data, unity.data(), planes[2].data, planes[3].data};
        std::vector<float> canonical(decoded.size());
        RadQuantWeight descriptor{}; descriptor.rows = columns; descriptor.cols = reduction;
        const int status = decoder(&grid, &descriptor, data, 0, columns, canonical.data());
        REQUIRE_EQ(status, RAD_OK);
        size_t differences = 0;
        for (size_t index = 0; index < decoded.size(); ++index)
            differences += canonical[index] != float(decoded[index]);
        CHECK_EQ(differences, 0u);
        std::printf("IQR_MTP_CANONICAL quantizer=gptq decoded=%zu differences=%zu run=%lld\n",
            decoded.size(), differences, (long long)encoding.plane[2].block[0]);
        dlclose(library);
    }
};
}

TEST(r3d_iqr_group_model_contract) {
    Library native("libr3d");
    const auto* row = moe_boundary::registered(native, "gemm_nt_q_iqr_prefill");
    rad::RadFile file;
    REQUIRE_EQ(file.open("/home/zihaomu/bigssd2/radiance_workspace/models/qwen3.8-next-flash-iqr-v4.rad"), RAD_OK);
    REQUIRE_EQ(file.size(), 121678372864ll);
    for (int rank : {0, 1}) { iqr_group::Fixture fixture(row, file, rank); }
}

namespace iqr_group {
uint16_t oracle(const Fixture& fixture, Buffer& activation, Buffer& scales, int row, int column) {
    float accumulator = 0;
    const int groups = fixture.reduction / 128;
    for (int group = 0; group < groups; ++group) {
        int32_t subtotal = 0;
        for (int offset = 0; offset < 128; ++offset) {
            const int position = group * 128 + offset;
            subtotal += int(activation.data<int8_t>()[row * activation.host.stride[0] + position])
                * int(fixture.decoded[size_t(column) * fixture.reduction + position]);
        }
        uint16_t bits;
        std::memcpy(&bits, fixture.storage[1].data() + (size_t(column) * groups + group) * 2, 2);
        volatile float scaled = float(subtotal) * scales.data<float>()[row * scales.host.stride[0] + group];
        volatile float product = scaled * rad_bf16_to_f32(bits);
        accumulator = accumulator + product;
    }
    return rad_f32_to_bf16(accumulator);
}
}

TEST(r3d_iqr_group_actual_aql) {
    using namespace iqr_group;
    using iqr_output_boundary::matrix;
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
    const auto* decode = moe_boundary::registered(native, "gemm_nt_q_iqr_decode");
    const auto* prefill = moe_boundary::registered(native, "gemm_nt_q_iqr_prefill");
    const auto* producer = moe_boundary::registered(native, "qwen_had_quant_i8_rowgroup");
    Library reference("libref");
    auto reference_count = reinterpret_cast<int (*)()>(dlsym(reference.handle, "rad_kernel_count"));
    auto reference_at = reinterpret_cast<const RadKernelInfo* (*)(int)>(dlsym(reference.handle, "rad_kernel_at"));
    REQUIRE(reference_count && reference_at);
    const RadKernelInfo* reference_gemm = nullptr;
    for (int index = 0; index < reference_count(); ++index)
        if (std::strcmp(reference_at(index)->name, "ref_gemm_nt_q") == 0) reference_gemm = reference_at(index);
    REQUIRE(reference_gemm && reference_gemm->launch);
    Queue compute(0), other(1);
    uint64_t fixtures = 0, launches = 0, producers = 0, rejects = 0, values = 0, exact_bytes = 0;
    rad::RadFile file;
    REQUIRE_EQ(file.open("/home/zihaomu/bigssd2/radiance_workspace/models/qwen3.8-next-flash-iqr-v4.rad"), RAD_OK);
    for (int rank : {0, 1}) {
        Fixture fixture(prefill, file, rank);
        const int columns = fixture.columns, reduction = fixture.reduction, groups = reduction / 128;
        Buffer weight(RAD_I32, {1, int64_t((fixture.weight.size() + 3) / 4)}, RAD_MEM_DEVICE);
        Buffer weight_scale(RAD_BF16, {1, int64_t(fixture.scale.size() / 2)}, RAD_MEM_DEVICE);
        std::memcpy(weight.host.data, fixture.weight.data(), fixture.weight.size());
        std::memcpy(weight_scale.host.data, fixture.scale.data(), fixture.scale.size());
        weight.device = matrix(weight.device.data, r3d_iqr::weight_dtype, columns, reduction);
        for (int rows : {1, 2, 3, 4, 63, 64}) for (int mode : {0, 1}) {
            const auto* gemm = rows <= 2 ? decode : prefill;
            Buffer input(RAD_BF16, {rows + 2, groups, 128}, RAD_MEM_DEVICE, reduction + 16);
            Buffer codes(RAD_I8, {rows + 2, reduction}, RAD_MEM_DEVICE, reduction + 16);
            Buffer scales(RAD_F32, {rows + 2, groups}, RAD_MEM_DEVICE, groups + 4);
            Buffer output(RAD_BF16, {rows + 2, columns}, RAD_MEM_DEVICE, columns + 32);
            for (Buffer* buffer : {&input, &codes, &scales, &output})
                buffer->host.shape[0] = buffer->device.shape[0] = rows;
            std::vector<uint8_t> previous;
            for (int step = 0; step < 2; ++step) {
                RadStream stream = step ? compute.stream : other.stream;
                for (Buffer* buffer : {&codes, &scales, &output})
                    std::fill(buffer->expected.begin(), buffer->expected.end(), uint8_t(0x5a));
                for (int row = 0; row < rows; ++row) {
                    for (int position = 0; position < reduction; ++position) {
                        input.data<uint16_t>()[row * input.host.stride[0] + position] = rad_f32_to_bf16(
                            float((row * 197 + position * 29 + position / 128 * 17 + step * 71 + rank * 13) % 509 - 254) / 32.0f);
                        if (!mode) codes.data<int8_t>()[row * codes.host.stride[0] + position] =
                            int8_t((position * 37 + row * 53 + step * 71 + rank * 19) % 256 - 128);
                    }
                    if (!mode) for (int group = 0; group < groups; ++group)
                        scales.data<float>()[row * scales.host.stride[0] + group] =
                            float((group * 13 + row * 7 + step * 17) % 127 + 1) / 1024.0f;
                }
                for (Buffer* buffer : {&input, &codes, &scales, &weight, &weight_scale, &output}) buffer->push(stream);
                REQUIRE_EQ(rad_stream_sync(stream), RAD_OK);
                if (mode) {
                    iqr_output_boundary::quant_expected(input, codes, scales, rows, reduction);
                    RadTensor quant_tensors[]{input.device, codes.device, scales.device};
                    RadParam quant_params[]{RAD_INT("M", 64), RAD_INT("n", reduction), RAD_INT("group", 128)};
                    RadArgs quant_args{}; quant_args.t = quant_tensors; quant_args.n_t = 3;
                    quant_args.p = quant_params; quant_args.n_p = 3;
                    REQUIRE_EQ(producer->launch(&quant_args, stream), RAD_OK);
                    ++producers;
                    REQUIRE(codes.read(stream) == codes.expected);
                    REQUIRE(scales.read(stream) == scales.expected);
                }
                auto golden = output.expected;
                for (int row = 0; row < rows; ++row) for (int column = 0; column < columns; ++column) {
                    const uint16_t value = oracle(fixture, codes, scales, row, column);
                    std::memcpy(golden.data() + 64 + (row * output.host.stride[0] + column) * 2, &value, 2);
                    ++values;
                }
                std::vector<uint16_t> ref_values(size_t(rows) * columns);
                RadTensor ref_tensors[]{codes.host, scales.host,
                    matrix(fixture.decoded.data(), RAD_I8, columns, reduction),
                    matrix(fixture.storage[1].data(), RAD_BF16, columns, groups),
                    matrix(ref_values.data(), RAD_BF16, rows, columns)};
                RadParam ref_params[]{RAD_INT("M", rows), RAD_INT("N", columns), RAD_INT("K", reduction),
                    RAD_INT("group", 128), RAD_STR("dtype", "i8a8h")};
                RadArgs ref_args{}; ref_args.t = ref_tensors; ref_args.n_t = 5;
                ref_args.p = ref_params; ref_args.n_p = 5;
                REQUIRE_EQ(reference_gemm->launch(&ref_args, nullptr), RAD_OK);
                for (int row = 0; row < rows; ++row)
                    REQUIRE_EQ(std::memcmp(ref_values.data() + row * columns,
                        golden.data() + 64 + row * output.host.stride[0] * 2, columns * 2), 0);
                if (step) REQUIRE(golden != previous);
                previous = golden;
                for (int implementation : {0, 1, 2}) for (bool logical : {false, true}) for (int output_rank : {2, 3}) {
                    weight_scale.device = matrix(weight_scale.device.data, r3d_iqr::group_scale_dtype,
                        columns, logical ? reduction : groups);
                    output.push(stream);
                    std::array<RadTensor, 5> tensors{{codes.device, scales.device, weight.device, weight_scale.device, output.device}};
                    if (output_rank == 3) {
                        tensors[4].rank = 3; tensors[4].shape[1] = 5; tensors[4].shape[2] = 512;
                        tensors[4].stride[1] = 512; tensors[4].stride[2] = 1;
                    }
                    RadParam params[]{RAD_INT("M", logical ? 2048 : 64), RAD_INT("N", columns), RAD_INT("K", reduction),
                        RAD_INT("group", 128), RAD_INT("impl", implementation), RAD_STR("dtype", "i8a8h")};
                    RadArgs args{}; args.t = tensors.data(); args.n_t = 5; args.p = params; args.n_p = 6;
                    const auto original = tensors;
                    REQUIRE_EQ(gemm->launch(&args, stream), RAD_OK);
                    REQUIRE_EQ(std::memcmp(tensors.data(), original.data(), sizeof(tensors)), 0);
                    ++launches;
                    REQUIRE(output.read(stream) == golden);
                    exact_bytes += golden.size();
                    if (output_rank != 3) continue;
                    auto reject = [&](auto mutate, int expected) {
                        auto bad = tensors;
                        std::array<RadParam, 6> bad_params;
                        std::copy(std::begin(params), std::end(params), bad_params.begin());
                        RadArgs invalid = args; invalid.t = bad.data(); invalid.p = bad_params.data();
                        mutate(bad, bad_params);
                        REQUIRE_EQ(gemm->launch(&invalid, reinterpret_cast<RadStream>(0x12345)), expected);
                        ++rejects;
                    };
                    for (uint32_t dtype : {uint32_t(RAD_F32), uint32_t(RAD_BF16), 0x33000301u})
                        reject([&](auto& bad, auto&) { bad[3].dtype = dtype; }, RAD_E_DTYPE);
                    reject([](auto& bad, auto&) { bad[3].rank = 3; }, RAD_E_DTYPE);
                    reject([](auto& bad, auto&) { bad[3].stride[1] = 2; }, RAD_E_DTYPE);
                    reject([](auto& bad, auto&) { bad[3].shape[0]++; }, RAD_E_SHAPE);
                    reject([](auto& bad, auto&) { bad[3].shape[1] = 17; }, RAD_E_SHAPE);
                    reject([](auto& bad, auto&) { bad[4].stride[1]++; }, RAD_E_STRIDE);
                    reject([&](auto& bad, auto&) { bad[4].stride[0] = columns - 1; }, RAD_E_STRIDE);
                    reject([](auto& bad, auto&) { bad[4].shape[0]++; }, RAD_E_SHAPE);
                    reject([&](auto&, auto& bad) { bad[0].ival = rows - 1; }, RAD_E_SHAPE);
                    reject([](auto&, auto& bad) { bad[3].ival = 64; }, RAD_E_SHAPE);
                    reject([](auto&, auto& bad) { bad[4].ival = 4; }, RAD_E_INVAL);
                    reject([](auto&, auto& bad) { bad[5].sval = "bf16"; }, RAD_E_DTYPE);
                    REQUIRE(output.read(stream) == golden);
                }
                for (Buffer* buffer : {&input, &codes, &scales, &weight, &weight_scale})
                    REQUIRE(buffer->read(stream) == buffer->expected);
            }
            ++fixtures;
            std::printf("IQR_MTP_AQL rank=%d M=%d N=%d K=%d producer=%d exhaustive=1 strict_scalar=1\n",
                rank, rows, columns, reduction, mode);
        }
    }
    std::printf("PASS IQR_MTP_GROUP_AQL fixtures=%llu gemms=%llu producers=%llu rejects=%llu independent_values=%llu exact_bytes=%llu queues=2 tolerance=0 all_M_exhaustive=1 libref_normalized_i8_bf16_exhaustive=1 impls_share_strict_scalar=1\n",
        (unsigned long long)fixtures, (unsigned long long)launches, (unsigned long long)producers,
        (unsigned long long)rejects, (unsigned long long)values, (unsigned long long)exact_bytes);
}

TEST(r3d_iqr_group_synthetic_aql_edges) {
    using namespace iqr_group;
    using iqr_output_boundary::matrix;
    RadDeviceProps properties{};
    REQUIRE_EQ(rad_dev_props(0, &properties), RAD_OK);
    if (properties.is_host_backend) {
        REQUIRE(!std::getenv("RAD_R3D_REQUIRE_DEVICE"));
        SKIP("explicit gfx1100 card run required");
    }
    REQUIRE_EQ(std::strcmp(properties.arch, "gfx1100"), 0);
    REQUIRE_EQ(rad_dev_set(0), RAD_OK);
    Library native("libr3d");
    const auto* gemm = moe_boundary::registered(native, "gemm_nt_q_iqr_prefill");
    Queue compute(0), other(1);
    uint64_t launches = 0, values = 0, guarded = 0, ties = 0;
    for (bool halfway : {false, true}) {
        Fixture fixture;
        fixture.columns = 256; fixture.reduction = halfway ? 128 : 65536;
        const int columns = fixture.columns, reduction = fixture.reduction, groups = reduction / 128;
        const int rows = halfway ? 4 : 1;
        rad_enc_copy_str(fixture.encoding.scheme, "trellis");
        rad_enc_copy_str(fixture.encoding.transform, "fwht128");
        rad_enc_add_plane(&fixture.encoding, "codes", RAD_U4, 1, 1);
        rad_enc_add_plane(&fixture.encoding, "scale", RAD_BF16, 1, 128);
        const int trellis = rad_enc_add_table(&fixture.encoding, "trellis", RAD_I8, 65536, 1);
        fixture.encoding.plane[trellis].block[0] = 256; fixture.encoding.plane[trellis].block[1] = 1;
        rad_enc_add_table(&fixture.encoding, "trellis.mad", RAD_U32, 2, 1);
        fixture.storage.resize(4); fixture.planes.resize(4);
        fixture.storage[0].resize(size_t(columns) * reduction / 2);
        fixture.storage[1].resize(size_t(columns) * groups * 2);
        fixture.storage[2].resize(65536); fixture.storage[3].resize(8);
        uint32_t coefficients[]{halfway ? 0u : 0xff123457u, halfway ? 0x3f3f0100u : 0xfedcba98u};
        std::memcpy(fixture.storage[3].data(), coefficients, 8);
        for (uint32_t state = 0; state < 65536; ++state)
            fixture.storage[2][state] = uint8_t(iqr_output_boundary::independent_mad(state, coefficients[0], coefficients[1]));
        for (int column = 0; column < reduction; ++column) for (int row = 0; row < columns; ++row) {
            unsigned code = unsigned(row * 37 + column * 19 + (column >> 8)) & 15u;
            if (!halfway) {
                if (row == 0) code = unsigned(column) & 15u;
                if (row >= 253) code = (unsigned(column) >> ((256 - row) * 4)) & 15u;
            }
            fixture.storage[0][size_t(row) * (reduction / 2) + column / 2] |= uint8_t(code << ((column % 2) * 4));
        }
        const uint16_t scale_bits[]{0x3f80, 0xbf80, 0x0000, 0x8000, 0x3f81, 0xbf81};
        for (size_t index = 0; index < size_t(columns) * groups; ++index) {
            const uint16_t scale = halfway ? scale_bits[index % 6] : uint16_t(0x3c00 + index % 256);
            std::memcpy(fixture.storage[1].data() + index * 2, &scale, 2);
        }
        fixture.planes[0] = matrix(fixture.storage[0].data(), RAD_U4, columns, reduction);
        fixture.planes[1] = matrix(fixture.storage[1].data(), RAD_BF16, columns, groups);
        fixture.planes[2] = matrix(fixture.storage[2].data(), RAD_I8, 65536, 1);
        fixture.planes[3] = matrix(fixture.storage[3].data(), RAD_U32, 2, 1);
        fixture.decoded.resize(size_t(columns) * reduction);
        for (int column = 0; column < reduction; ++column) {
            auto code = [&](int row) { return unsigned((fixture.storage[0][size_t(row) * (reduction / 2) + column / 2] >> ((column % 2) * 4)) & 15u); };
            unsigned state = 0;
            for (int row = 252; row < 256; ++row) state = ((state << 4) | code(row)) % 65536;
            for (int row = 0; row < columns; ++row) {
                state = ((state << 4) | code(row)) % 65536;
                if (!halfway && !row) REQUIRE_EQ(state, unsigned(column));
                fixture.decoded[size_t(row) * reduction + column] = int8_t(fixture.storage[2][state]);
            }
        }
        RadParam params[]{RAD_INT("M", 64), RAD_INT("N", columns), RAD_INT("K", reduction),
            RAD_INT("group", 128), RAD_INT("impl", 0), RAD_STR("dtype", "i8a8h")};
        RadTensor weight_planes[]{fixture.planes[0], fixture.planes[2], fixture.planes[3]};
        for (int operand : {2, 3}) {
            RadLayout layout{};
            const auto* planes = operand == 2 ? weight_planes : &fixture.planes[1];
            const int* selected = operand == 2 ? fixture.selected : fixture.scale_selected;
            const int count = operand == 2 ? 3 : 1;
            REQUIRE_EQ(gemm->layout(params, 6, operand, &fixture.encoding, selected, planes, count, &layout), RAD_OK);
            auto& target = operand == 2 ? fixture.weight : fixture.scale;
            target.resize(layout.bytes);
            REQUIRE_EQ(gemm->relayout(params, 6, operand, &fixture.encoding, selected, planes, count, target.data(), target.size()), RAD_OK);
        }
        Buffer weight(RAD_I32, {1, int64_t(fixture.weight.size() / 4)}, RAD_MEM_DEVICE);
        Buffer weight_scale(RAD_BF16, {1, int64_t(fixture.scale.size() / 2)}, RAD_MEM_DEVICE);
        Buffer codes(RAD_I8, {rows + 2, reduction}, RAD_MEM_DEVICE, reduction + 16);
        Buffer scales(RAD_F32, {rows + 2, groups}, RAD_MEM_DEVICE, groups + 4);
        Buffer output(RAD_BF16, {rows + 2, columns}, RAD_MEM_DEVICE, columns + 32);
        std::memcpy(weight.host.data, fixture.weight.data(), fixture.weight.size());
        std::memcpy(weight_scale.host.data, fixture.scale.data(), fixture.scale.size());
        weight.device = matrix(weight.device.data, r3d_iqr::weight_dtype, columns, reduction);
        for (Buffer* buffer : {&codes, &scales, &output}) buffer->host.shape[0] = buffer->device.shape[0] = rows;
        RadTensor tensors[]{codes.device, scales.device, weight.device, {}, output.device};
        RadArgs args{}; args.t = tensors; args.n_t = 5; args.p = params; args.n_p = 6;
        for (int step = 0; step < 2; ++step) {
            RadStream stream = step ? compute.stream : other.stream;
            for (int row = 0; row < rows; ++row) {
                const int totals[]{257, 259, -257, -259};
                const int total = totals[(row + step) % 4], sign = total < 0 ? -1 : 1;
                for (int position = 0; position < reduction; ++position)
                    codes.data<int8_t>()[row * codes.host.stride[0] + position] = halfway
                        ? int8_t(position < 2 ? sign * 127 : position == 2 ? total - sign * 254 : 0)
                        : int8_t((position * 37 + step * 71) % 256 - 128);
                for (int group = 0; group < groups; ++group)
                    scales.data<float>()[row * scales.host.stride[0] + group] = halfway ? 1.0f / 256 : float(group % 31 + 1) / 1024;
            }
            for (Buffer* buffer : {&codes, &scales, &weight, &weight_scale, &output}) buffer->push(stream);
            REQUIRE_EQ(rad_stream_sync(stream), RAD_OK);
            auto golden = output.expected;
            for (int row = 0; row < rows; ++row) for (int column = 0; column < columns; ++column) {
                const uint16_t value = oracle(fixture, codes, scales, row, column);
                std::memcpy(golden.data() + 64 + (row * output.host.stride[0] + column) * 2, &value, 2);
                if (halfway && !column) {
                    const uint16_t expected[]{0x3f80, 0x3f82, 0xbf80, 0xbf82};
                    REQUIRE_EQ(value, expected[(row + step) % 4]); ++ties;
                }
                ++values;
            }
            for (int implementation : {0, 1, 2}) for (bool logical : {false, true}) {
                params[4].ival = implementation;
                tensors[3] = matrix(weight_scale.device.data, r3d_iqr::group_scale_dtype, columns, logical ? reduction : groups);
                output.push(stream);
                REQUIRE_EQ(gemm->launch(&args, stream), RAD_OK);
                REQUIRE(output.read(stream) == golden); ++launches;
            }
            for (Buffer* buffer : {&codes, &scales, &weight, &weight_scale}) REQUIRE(buffer->read(stream) == buffer->expected);
        }
        r3d_iqr::Header original;
        std::memcpy(&original, weight.host.data, sizeof(original));
        for (int defect = 0; defect < 7; ++defect) {
            auto bad = original;
            if (defect == 0) bad.magic = 0;
            if (defect == 1) bad.bits = 2;
            if (defect == 2) bad.run_rows = 64;
            if (defect == 3) bad.state_bits = 15;
            if (defect == 4) bad.rows++;
            if (defect == 5) bad.columns++;
            if (defect == 6) bad.run_rows = 0;
            std::memcpy(weight.host.data, &bad, sizeof(bad));
            weight.push(compute.stream); output.push(compute.stream);
            REQUIRE_EQ(gemm->launch(&args, compute.stream), RAD_OK);
            REQUIRE(output.read(compute.stream) == output.expected);
            REQUIRE(weight.read(compute.stream) == weight.expected); ++guarded;
        }
    }
    std::printf("PASS IQR_MTP_EDGES_AQL gemms=%llu independent_values=%llu full_state_row=65536 low24_overflow=1 bf16_halfway_witnesses=%llu invalid_device_headers=%llu tolerance=0\n",
        (unsigned long long)launches, (unsigned long long)values, (unsigned long long)ties, (unsigned long long)guarded);
}
