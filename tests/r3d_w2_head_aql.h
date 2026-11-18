#include "../libr3d/r3d_w2_head.h"
#include <numeric>

namespace w2_head_test {
using joined_head_test::Storage;
using joined_head_test::device_available;
using joined_head_test::matrix;
using joined_head_test::multiply;
using joined_head_test::add;
using mtp_fc_test::Model;
using Launch = int (*)(const RadArgs*, RadStream);
uint64_t launches = 0, values = 0, rejects = 0, producer_calls = 0, oracle_calls = 0;
const RadKernelInfo* named(Library& library, const char* name) {
    auto count = reinterpret_cast<int (*)()>(dlsym(library.handle, "rad_kernel_count"));
    auto at = reinterpret_cast<const RadKernelInfo* (*)(int)>(dlsym(library.handle, "rad_kernel_at"));
    if (!count || !at) throw std::runtime_error("W2 plugin enumeration");
    for (int index = 0; index < count(); ++index)
        if (!std::strcmp(at(index)->name, name)) return at(index);
    throw std::runtime_error(name);
}
struct Api {
    Library native{"libr3d"}, reference{"libref"};
    const RadKernelInfo* row = named(native, "gemm_nt_q_w2a8_m16");
    const RadKernelInfo* producer = named(native, "quant_act_i8_bf16");
    const RadKernelInfo* coarse = named(native, "row_topk_bf16");
    Launch oracle = reinterpret_cast<Launch>(dlsym(reference.handle, "ref_gemm_nt_q"));
    Launch quant_oracle = reinterpret_cast<Launch>(dlsym(reference.handle, "ref_quant_act_i8"));
    Api() { REQUIRE(row->layout && row->relayout && row->launch && oracle && quant_oracle); }
};
struct Fixture {
    int columns, reduction;
    RadEncoding encoding{};
    int selected[2]{1, 2};
    RadTensor planes[2]{};
    RadLayout layout{};
    std::vector<uint8_t> codes, zeros, joined;
    std::vector<uint16_t> scales;
    std::vector<RadParam> params;
    Fixture(Api& api, int n, int k) : columns(n), reduction(k),
        codes(size_t(n) * k / 4), zeros(size_t(n) * (k / 128)), scales(zeros.size()),
        params{integer("N", n), integer("K", k), integer("group", 128),
               RAD_STR("dtype", "w2a8"), integer("M", 8)} {
        std::strcpy(encoding.scheme, "affine"); encoding.n_planes = 3;
        for (int index = 0; index < 3; ++index) {
            auto& plane = encoding.plane[index];
            std::strcpy(plane.role, index == 0 ? "codes" : index == 1 ? "scale" : "zero");
            plane.dtype = index == 0 ? RAD_U2 : index == 1 ? RAD_F16 : RAD_U8;
            plane.kind = RAD_PLANE_TILED; plane.block[0] = 1; plane.block[1] = index ? 128 : 1;
        }
        for (size_t index = 0; index < codes.size(); ++index) codes[index] = uint8_t(index * 31 + index / 37);
        for (size_t index = 0; index < zeros.size(); ++index) {
            zeros[index] = uint8_t(index % 4);
            scales[index] = rad_f32_to_f16(float(1 + index % 17) / 256.0f);
        }
        pack(api);
    }
    void pack(Api& api) {
        planes[0] = matrix(scales.data(), RAD_F16, columns, reduction / 128);
        planes[1] = matrix(zeros.data(), RAD_U8, columns, reduction / 128);
        REQUIRE_EQ(api.row->layout(params.data(), params.size(), 3, &encoding, selected, planes, 2, &layout), RAD_OK);
        REQUIRE_EQ(layout.dtype, r3d_w2::dtype); REQUIRE_EQ(std::strcmp(layout.tag, "rad.planes"), 0);
        joined.resize(layout.bytes);
        REQUIRE_EQ(api.row->relayout(params.data(), params.size(), 3, &encoding, selected, planes, 2,
                                    joined.data(), joined.size()), RAD_OK);
        REQUIRE_EQ(r3d_w2::header(reinterpret_cast<const RadPlanesHeader*>(joined.data()), columns, reduction,
                                joined.size()), RAD_OK);
        const auto* entries = reinterpret_cast<const RadPlanesEntry*>(joined.data() + sizeof(RadPlanesHeader));
        REQUIRE_EQ(std::memcmp(joined.data() + entries[0].offset, scales.data(), scales.size() * 2), 0);
        REQUIRE_EQ(std::memcmp(joined.data() + entries[1].offset, zeros.data(), zeros.size()), 0);
    }
};
int step(int position) { return (position & ~12) | ((position & 4) << 1) | ((position & 8) >> 1); }
void equal(const std::vector<uint8_t>& actual, const std::vector<uint8_t>& expected, const char* label) {
    if (actual != expected) {
        const auto mismatch = std::mismatch(actual.begin(), actual.end(), expected.begin());
        std::printf("W2_MISMATCH %s byte=%zu actual=%u expected=%u\n", label,
            size_t(mismatch.first - actual.begin()), unsigned(*mismatch.first), unsigned(*mismatch.second));
        REQUIRE(false);
    }
}
std::vector<int> top(const uint16_t* logits, int columns) {
    std::vector<int> ids(columns); std::iota(ids.begin(), ids.end(), 0);
    std::partial_sort(ids.begin(), ids.begin() + 8, ids.end(), [&](int left, int right) {
        const float first = rad_bf16_to_f32(logits[left]), second = rad_bf16_to_f32(logits[right]);
        return first > second || (first == second && left < right);
    });
    ids.resize(8); return ids;
}
void execute(Api& api, Fixture& fixture, int rows, int rank, bool full, int band = 8) {
    const int columns = fixture.columns, reduction = fixture.reduction, groups = reduction / 128;
    Queue first(0), second(1);
    const int live_pad = ((rows + 15) / 16) * 16;
    const int capacity_pad = ((band + 15) / 16) * 16;
    if (band == 32) api.row = named(api.native, "gemm_nt_q_w2a8_head_strict");
    Buffer input(RAD_BF16, {rows, reduction}, RAD_MEM_DEVICE, reduction + 16);
    Buffer activation(RAD_I8, {rows, reduction}, RAD_MEM_DEVICE);
    Buffer scale(RAD_F32, {rows, 1}, RAD_MEM_DEVICE, 4);
    Buffer sums(RAD_F32, {groups, capacity_pad}, RAD_MEM_DEVICE);
    Buffer output(RAD_BF16, {rows, columns}, RAD_MEM_DEVICE, columns + 16);
    for (int repeat = 0; repeat < 2; ++repeat) {
        const RadStream stream = repeat ? second.stream : first.stream;
        std::fill(activation.expected.begin(), activation.expected.end(), 0x5a);
        std::fill(sums.expected.begin(), sums.expected.end(), 0x5a);
        for (int row = 0; row < rows; ++row) {
            scale.data<float>()[row * 4] = std::ldexp(1.0f, (row % 3) - 1);
            for (int position = 0; position < reduction; ++position) {
                const int code = position == 0 ? 127 : ((position * 47 + row * 31 + repeat * 73) % 255) - 127;
                input.data<uint16_t>()[row * (reduction + 16) + position] = rad_f32_to_bf16(
                    float(code) * scale.data<float>()[row * 4]);
                activation.data<int8_t>()[row * reduction + step(position)] = int8_t(code);
            }
        }
        for (int group = 0; group < groups; ++group)
            for (int row = 0; row < rows; ++row) {
                int sum = 0;
                for (int inner = 0; inner < 128; ++inner)
                    sum += activation.data<int8_t>()[row * reduction + step(group * 128 + inner)];
                sums.data<float>()[group * live_pad + row] = float(sum);
            }
        const auto expected_activation = activation.expected, expected_scale = scale.expected, expected_sums = sums.expected;
        input.push(stream);
        std::fill(activation.expected.begin(), activation.expected.end(), 0x5a);
        std::fill(scale.expected.begin(), scale.expected.end(), 0x5a);
        std::fill(sums.expected.begin(), sums.expected.end(), 0x5a);
        activation.push(stream); scale.push(stream); sums.push(stream);
        REQUIRE_EQ(rad_stream_sync(stream), RAD_OK);
        RadTensor quant_host[]{input.host, activation.host, scale.host, sums.host};
        RadParam quant_params[]{integer("M", band), integer("n", reduction), integer("group", 128), RAD_STR("dtype", "bf16")};
        RadArgs quant_args{}; quant_args.t = quant_host; quant_args.n_t = 4; quant_args.p = quant_params; quant_args.n_p = 4;
        REQUIRE_EQ(api.quant_oracle(&quant_args, nullptr), RAD_OK);
        equal(activation.expected, expected_activation, "libref producer codes");
        equal(scale.expected, expected_scale, "libref producer scales");
        equal(sums.expected, expected_sums, "libref producer sums");
        RadTensor quant_device[]{input.device, activation.device, scale.device, sums.device}; quant_args.t = quant_device;
        REQUIRE_EQ(api.producer->launch(&quant_args, stream), RAD_OK); ++producer_calls;
        equal(activation.read(stream), expected_activation, "native step-order producer");
        equal(scale.read(stream), expected_scale, "native row scale");
        equal(sums.read(stream), expected_sums, "native group-major M16 sums");
        auto expected = output.expected;
        for (int row = 0; row < rows; ++row)
            for (int column = 0; column < columns; ++column) {
                float accumulator = 0;
                for (int group = 0; group < groups; ++group) {
                    int32_t dot = 0;
                    for (int inner = 0; inner < 128; ++inner) {
                        const int position = group * 128 + inner;
                        const size_t index = size_t(column) * (reduction / 4) + position / 4;
                        const int code = (fixture.codes[index] >> (2 * (position % 4))) & 3;
                        dot += int32_t(activation.data<int8_t>()[row * reduction + step(position)]) * code;
                    }
                    const size_t index = size_t(column) * groups + group;
                    const int32_t sub = dot - int32_t(fixture.zeros[index]) * int32_t(sums.data<float>()[group * live_pad + row]);
                    accumulator = add(accumulator, multiply(multiply(float(sub), scale.data<float>()[row * 4]),
                                                            rad_f16_to_f32(fixture.scales[index])));
                }
                const uint16_t result = rad_f32_to_bf16(accumulator);
                std::memcpy(expected.data() + 64 + (size_t(row) * (columns + 16) + column) * 2, &result, 2);
            }
        if (rows == 1 || !full) {
            RadTensor host[]{activation.host, scale.host, matrix(fixture.codes.data(), RAD_U2, columns, reduction),
                matrix(fixture.joined.data(), RAD_F16, columns, reduction), output.host, sums.host, {}};
            RadArgs args{}; args.t = host; args.n_t = 7; args.p = fixture.params.data(); args.n_p = fixture.params.size();
            REQUIRE_EQ(api.oracle(&args, nullptr), RAD_OK); ++oracle_calls;
            equal(output.expected, expected, "libref ZP_I8_ROW exhaustive");
            std::fill(output.expected.begin(), output.expected.end(), 0x5a);
        }
        for (int kind : {int(RAD_MEM_DEVICE), -1}) {
            Storage codes(fixture.codes.size(), kind), joined(fixture.joined.size(), kind);
            std::copy(fixture.codes.begin(), fixture.codes.end(), codes.expected.begin() + 64);
            std::copy(fixture.joined.begin(), fixture.joined.end(), joined.expected.begin() + 64);
            codes.push(stream); joined.push(stream);
            for (int blob = 0; blob < 2; ++blob) {
                int issued = 2;
                while (issued < rows) issued *= 2;
                fixture.params[4].ival = band == 8 && rows == 1 && !blob ? 2 : band == 32 && !blob ? issued : band;
                output.push(stream); REQUIRE_EQ(rad_stream_sync(stream), RAD_OK);
                RadTensor tensors[]{activation.device, scale.device, matrix(codes.device(), RAD_U2, columns, reduction),
                    blob ? matrix(joined.device(), r3d_w2::dtype, fixture.joined.size(), 1)
                         : matrix(joined.device(), r3d_w2::dtype, columns, reduction), output.device, sums.device, {}};
                if (blob) { tensors[3].rank = 1; tensors[3].shape[0] = fixture.joined.size(); tensors[3].stride[0] = 1; }
                RadArgs args{}; args.t = tensors; args.n_t = 7; args.p = fixture.params.data(); args.n_p = fixture.params.size();
                REQUIRE_EQ(api.row->launch(&args, stream), RAD_OK); ++launches;
                REQUIRE(expected != output.expected);
                const auto actual = output.read(stream); equal(actual, expected, "strict coarse BF16 all outputs and guards");
                values += uint64_t(rows) * columns;
                Buffer candidate_ids(RAD_I32, {rows, 8}, RAD_MEM_DEVICE, 16);
                Buffer candidate_values(RAD_F32, {rows, 8}, RAD_MEM_DEVICE, 16);
                candidate_ids.push(stream); candidate_values.push(stream);
                REQUIRE_EQ(rad_stream_sync(stream), RAD_OK);
                RadTensor coarse_tensors[]{output.device, candidate_ids.device, candidate_values.device};
                RadParam coarse_params[]{integer("M", band), integer("N", columns), integer("R", 8),
                    integer("vocab_off", rank * 124160), RAD_STR("dtype", "bf16")};
                RadArgs coarse_args{}; coarse_args.t = coarse_tensors; coarse_args.n_t = 3;
                coarse_args.p = coarse_params; coarse_args.n_p = 5;
                REQUIRE_EQ(api.coarse->launch(&coarse_args, stream), RAD_OK);
                const auto device_ids = candidate_ids.read(stream), device_values = candidate_values.read(stream);
                for (int row = 0; row < rows; ++row) {
                    const auto* got = reinterpret_cast<const uint16_t*>(actual.data() + 64) + row * (columns + 16);
                    const auto* want = reinterpret_cast<const uint16_t*>(expected.data() + 64) + row * (columns + 16);
                    const auto actual_ids = top(got, columns), expected_ids = top(want, columns);
                    REQUIRE(actual_ids == expected_ids);
                    for (int index = 0; index < 8; ++index) {
                        REQUIRE_EQ(actual_ids[index] + rank * 124160, expected_ids[index] + rank * 124160);
                        REQUIRE_EQ(got[actual_ids[index]], want[expected_ids[index]]);
                        candidate_ids.data<int32_t>()[row * 16 + index] = expected_ids[index] + rank * 124160;
                        candidate_values.data<float>()[row * 16 + index] = rad_bf16_to_f32(want[expected_ids[index]]);
                    }
                }
                equal(device_ids, candidate_ids.expected, "device R8 stable global IDs and guards");
                equal(device_values, candidate_values.expected, "device R8 scores and guards");
                if (!blob && repeat == 0) {
                    for (int invalid = 0; invalid < 6; ++invalid) {
                        const auto saved_activation = tensors[0];
                        const auto saved_sum = tensors[5];
                        const auto saved_band = fixture.params[4];
                        switch (invalid) {
                        case 0: fixture.params[4].ival = 33; break;
                        case 1: fixture.params[4].ival = rows - 1; break;
                        case 2: tensors[0].shape[0] = INT64_MAX; break;
                        case 3: tensors[0].stride[1] = 2; break;
                        case 4: tensors[5].stride[0] = INT64_MAX; break;
                        case 5: tensors[5].shape[0]++; break;
                        }
                        REQUIRE(api.row->launch(&args, stream) < 0); ++rejects;
                        tensors[0] = saved_activation; tensors[5] = saved_sum; fixture.params[4] = saved_band;
                    }
                    equal(output.read(stream), expected, "descriptor rejects preserve output");
                    equal(activation.read(stream), expected_activation, "immutable activation");
                    equal(sums.read(stream), expected_sums, "immutable live Mpad sums");
                    for (int index = 0; index < 6; ++index) {
                        const auto saved = tensors[index]; tensors[index].dtype = RAD_I64;
                        REQUIRE(api.row->launch(&args, stream) < 0); ++rejects; tensors[index] = saved;
                    }
                    const auto saved = tensors[5]; tensors[5].shape[1] = 15;
                    REQUIRE(api.row->launch(&args, stream) < 0); ++rejects;
                    tensors[5] = saved;
                    tensors[0].data = reinterpret_cast<void*>(0x12340);
                    REQUIRE_EQ(api.row->launch(&args, stream), RAD_E_DEVICE); ++rejects;
                    tensors[0] = activation.device;
                    auto corrupt = fixture.joined;
                    reinterpret_cast<RadPlanesHeader*>(corrupt.data())->magic = 0;
                    REQUIRE_EQ(rad_memcpy_async(joined.device(), corrupt.data(), corrupt.size(), stream), RAD_OK);
                    output.push(stream); REQUIRE_EQ(rad_stream_sync(stream), RAD_OK);
                    REQUIRE_EQ(api.row->launch(&args, stream), RAD_OK);
                    equal(output.read(stream), output.expected, "corrupt device header no writes"); ++rejects;
                    auto corrupt_expected = joined.expected;
                    std::copy(corrupt.begin(), corrupt.end(), corrupt_expected.begin() + 64);
                    equal(joined.read(stream), corrupt_expected, "corrupt joined metadata immutable");
                    equal(codes.read(stream), codes.expected, "corrupt launch codes immutable");
                    joined.push(stream);
                }
            }
            equal(codes.read(stream), codes.expected, "packed U2 immutable");
            equal(joined.read(stream), joined.expected, "joined header immutable");
        }
        std::printf("W2_CASE rank=%d offset=%d N=%d K=%d M=%d band=%d repeat=%d exhaustive=1\n",
                    rank, rank * 124160, columns, reduction, rows, band, repeat); std::fflush(stdout);
    }
}
TEST(r3d_w2_head_actual_tp2_aql) {
    if (!device_available()) SKIP("explicit gfx1100 card run required");
    Api api; Model model("/home/zihaomu/bigssd2/radiance_workspace/models/qwen3.8-next-flash-iqr-v4.rad");
    const RadFileEntry* head = nullptr;
    for (const auto& entry : model.directory) {
        const auto& encoding = model.encodings[entry.enc];
        if (entry.rank == 2 && entry.shape[0] == 248320 && entry.shape[1] == 2560 &&
            encoding.n_planes == 3 && encoding.plane[0].dtype == RAD_U2) {
            REQUIRE(!head); head = &entry;
        }
    }
    REQUIRE(head);
    std::printf("W2_ACTUAL_CHECKPOINT %s logical=mtp.draft_head.weight true_TP2_row_halves\n", model.name(*head));
    for (int rank = 0; rank < 2; ++rank) {
        Fixture fixture(api, 124160, 2560); fixture.encoding = model.encodings[head->enc];
        std::vector<std::pair<void*, size_t>> destinations{{fixture.codes.data(), fixture.codes.size()},
            {fixture.scales.data(), fixture.scales.size() * 2}, {fixture.zeros.data(), fixture.zeros.size()}};
        REQUIRE_EQ(head->n_planes, 3u);
        for (int index = 0; index < 3; ++index) {
            REQUIRE(head->plane_first + index < model.planes.size());
            const auto& plane = model.planes[head->plane_first + index];
            REQUIRE_EQ(plane.bytes, destinations[index].second * 2);
            REQUIRE(plane.offset >= head->offset && plane.bytes <= head->bytes - (plane.offset - head->offset));
            model.read(plane.offset + rank * destinations[index].second, destinations[index].first, destinations[index].second);
        }
        fixture.pack(api);
        for (int rows : {1, 2, 3, 8}) execute(api, fixture, rows, rank, true);
        for (int rows : {1, 3, 8, 16, 31, 32}) execute(api, fixture, rows, rank, true, 32);
        api.row = named(api.native, "gemm_nt_q_w2a8_m16");
    }
    std::printf("W2_ACTUAL_PASS launches=%llu producer=%llu libref_gemm=%llu values=%llu rejects=%llu R8=stable_global_offsets_0_124160 device_corruption=enqueue_ok_no_writes\n",
        (unsigned long long)launches, (unsigned long long)producer_calls, (unsigned long long)oracle_calls,
        (unsigned long long)values, (unsigned long long)rejects);
}
TEST(r3d_w2_head_synthetic_aql) {
    if (!device_available()) SKIP("explicit gfx1100 card run required");
    Api api; Fixture fixture(api, 131, 2560);
    for (int rows : {1, 2, 3, 8}) execute(api, fixture, rows, 0, false);
    const auto encoding = fixture.encoding;
    for (int index = 0; index < 3; ++index) {
    std::fill(fixture.scales.begin(), fixture.scales.end(), 0);
    fixture.pack(api); execute(api, fixture, 3, 1, false);
        fixture.encoding.plane[index].dtype = RAD_I64;
        RadLayout layout{};
        REQUIRE(api.row->layout(fixture.params.data(), fixture.params.size(), 3, &fixture.encoding,
                                fixture.selected, fixture.planes, 2, &layout) < 0);
        fixture.encoding = encoding;
    }
    std::printf("W2_SYNTHETIC_PASS launches=%llu values=%llu rejects=%llu\n",
        (unsigned long long)launches, (unsigned long long)values, (unsigned long long)rejects);
}
TEST(r3d_w2_head_header_rejections_aql) {
    if (!device_available()) SKIP("explicit gfx1100 card run required");
    Api api; Fixture fixture(api, 131, 2560); Queue queue(0);
    Buffer activation(RAD_I8, {1, 2560}, RAD_MEM_DEVICE);
    Buffer scale(RAD_F32, {1, 1}, RAD_MEM_DEVICE);
    Buffer sums(RAD_F32, {20, 16}, RAD_MEM_DEVICE);
    Buffer output(RAD_BF16, {1, 131}, RAD_MEM_DEVICE, 144);
    Storage codes(fixture.codes.size(), RAD_MEM_DEVICE);
    codes.push(queue.stream); REQUIRE_EQ(rad_stream_sync(queue.stream), RAD_OK);
    for (int kind : {int(RAD_MEM_DEVICE), -1, int(RAD_MEM_HOST_MAPPED)}) {
        std::fill(activation.expected.begin() + 64, activation.expected.end() - 64, 0);
        std::fill(sums.expected.begin() + 64, sums.expected.end() - 64, 0);
        scale.data<float>()[0] = 1.0f;
        Storage joined(fixture.joined.size(), kind);
        RadTensor tensors[]{activation.device, scale.device, matrix(codes.device(), RAD_U2, 131, 2560),
            matrix(joined.device(), kind == RAD_MEM_HOST_MAPPED ? RAD_F16 : r3d_w2::dtype, 131, 2560),
            output.device, sums.device, {}};
        RadArgs args{}; args.t = tensors; args.n_t = 7; args.p = fixture.params.data(); args.n_p = fixture.params.size();
        for (int defect = 0; defect < 12; ++defect) {
            auto corrupt = fixture.joined;
            std::copy(fixture.joined.begin(), fixture.joined.end(), joined.expected.begin() + 64);
            activation.push(queue.stream); scale.push(queue.stream); sums.push(queue.stream);
            joined.push(queue.stream); output.push(queue.stream);
            REQUIRE_EQ(rad_stream_sync(queue.stream), RAD_OK);
            REQUIRE_EQ(api.row->launch(&args, queue.stream), RAD_OK);
            REQUIRE(output.read(queue.stream) != output.expected);
            auto* header = reinterpret_cast<RadPlanesHeader*>(corrupt.data());
            auto* entries = reinterpret_cast<RadPlanesEntry*>(header + 1);
            switch (defect) {
            case 0: header->magic = 0; break;
            case 1: header->n_planes = 3; break;
            case 2: header->bytes = INT64_MAX; break;
            case 3: std::strcpy(entries[0].role, "zero"); break;
            case 4: entries[1].dtype = RAD_I8; break;
            case 5: entries[0].block[1] = 64; break;
            case 6: entries[1].rows++; break;
            case 7: entries[0].cols++; break;
            case 8: entries[1].row_bytes++; break;
            case 9: entries[1].offset = entries[0].offset; break;
            case 10: entries[0].offset = INT64_MAX; break;
            case 11: std::strcpy(header->transform, "bad"); break;
            }
            const int status = r3d_w2::header(header, 131, 2560, corrupt.size()); REQUIRE(status < 0);
            std::copy(corrupt.begin(), corrupt.end(), joined.expected.begin() + 64);
            joined.push(queue.stream); output.push(queue.stream); REQUIRE_EQ(rad_stream_sync(queue.stream), RAD_OK);
            REQUIRE_EQ(api.row->launch(&args, queue.stream), kind == RAD_MEM_HOST_MAPPED ? status : RAD_OK);
            equal(output.read(queue.stream), output.expected, "all malformed headers no writes"); ++rejects;
        }
            equal(joined.read(queue.stream), joined.expected, "malformed header immutable");
        std::copy(fixture.joined.begin(), fixture.joined.end(), joined.expected.begin() + 64);
        joined.push(queue.stream); REQUIRE_EQ(rad_stream_sync(queue.stream), RAD_OK);
        for (int band : {0, 17}) {
            fixture.params[4].ival = band; REQUIRE_EQ(api.row->launch(&args, queue.stream), RAD_E_SHAPE); ++rejects;
        }
        fixture.params[4].ival = 8;
        tensors[5].stride[0] = INT64_MAX;
        REQUIRE_EQ(api.row->launch(&args, queue.stream), RAD_E_STRIDE); ++rejects;
        tensors[5] = sums.device;
        if (kind == RAD_MEM_HOST_MAPPED) {
            tensors[3].dtype = r3d_w2::dtype;
            REQUIRE_EQ(api.row->launch(&args, queue.stream), RAD_OK);
            REQUIRE_EQ(rad_stream_sync(queue.stream), RAD_OK);
        }
    }
    std::printf("W2_HEADER_REJECTIONS_PASS rejects=%llu mapped_host_errors=explicit device_errors=no_write_async\n",
                (unsigned long long)rejects);
}
}
