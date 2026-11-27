#include "../libr3d/r3dh_mtp_ple.h"
#include "rad_format.h"
#include <fstream>

namespace mtp_fc_test {
using joined_head_test::Storage;
using joined_head_test::matrix;
void expect(bool condition, const char* text) {
    if (!condition) throw std::runtime_error(text);
}
struct Blob {
    uint32_t dtype;
    std::vector<uint8_t> bytes;
    Blob(size_t count, uint32_t type) : dtype(type), bytes(count * (rad_dtype_bits(type) / 8)) {}
    size_t size() const { return bytes.size() / (rad_dtype_bits(dtype) / 8); }
    template<class Value> Value* data() { return reinterpret_cast<Value*>(bytes.data()); }
};
#include "../libr3d/r3dh_mtp_ple_check_model.h"
struct Api {
    Library native{"libr3d"}, reference{"libref"};
    const RadKernelInfo* row = native.ngram("mtp_enter");
    const RadKernelInfo* oracle = reference.ngram("mtp_enter");
    Api() {
        REQUIRE_EQ(std::strcmp(row->name, "r3d_mtp_enter_e4m3"), 0);
        REQUIRE(row->layout && row->relayout && row->unrelayout && row->scratch && oracle);
    }
};
std::vector<RadParam> parameters() {
    return {RAD_INT("M", 64), RAD_INT("n", 2560), RAD_INT("hc", 4),
        RAD_INT("ngroup", 4), RAD_STR("fc", "e4m3"), RAD_F64("eps", 1e-6),
        RAD_F64("wadd", 1), RAD_STR("dtype", "bf16")};
}
void equal_bytes(const std::vector<uint8_t>& actual, const std::vector<uint8_t>& expected,
                 const char* label, uint64_t& compared) {
    REQUIRE_EQ(actual.size(), expected.size());
    const auto mismatch = std::mismatch(actual.begin(), actual.end(), expected.begin());
    if (mismatch.first != actual.end()) {
        std::printf("MTP_FC_MISMATCH %s byte=%zu actual=%02x expected=%02x\n", label,
            size_t(mismatch.first - actual.begin()), unsigned(*mismatch.first), unsigned(*mismatch.second));
        REQUIRE(false);
    }
    compared += actual.size();
}
}

TEST(r3d_mtp_fc_private_layout_host) {
    using namespace mtp_fc_test;
    Api api;
    auto params = parameters(); params[1] = RAD_INT("n", 128);
    const RadEncoding encoding = rad_enc_affine(RAD_F8E4M3, RAD_F32, 1, 128);
    const int selection[]{0, 1};
    std::vector<uint8_t> codes(128 * 128), packed(128 * 144 + 128, 0x5a);
    std::vector<float> scales(128);
    for (size_t index = 0; index < codes.size(); ++index) codes[index] = uint8_t(index * 19);
    for (size_t index = 0; index < scales.size(); ++index) scales[index] = float(index + 1) / 8192;
    const auto original_codes = codes; const auto original_scales = scales;
    RadTensor planes[]{matrix(codes.data(), RAD_F8E4M3, 128, 128), matrix(scales.data(), RAD_F32, 128, 1)};
    for (int operand : {4, 5}) {
        RadLayout layout{};
        REQUIRE_EQ(api.row->layout(params.data(), int(params.size()), operand, &encoding, selection, planes, 2, &layout), RAD_OK);
        CHECK_EQ(layout.dtype, r3dh_mtp_fc_e4m3_dtype);
        CHECK_EQ(std::strcmp(layout.tag, "r4d.mtp.e4m3.g128"), 0);
        CHECK_EQ(layout.rank, 2); CHECK_EQ(layout.shape[0], 128); CHECK_EQ(layout.shape[1], 144);
        CHECK_EQ(layout.bytes, 128 * 144); CHECK_EQ(layout.align, 16);
        REQUIRE_EQ(api.row->relayout(params.data(), int(params.size()), operand, &encoding, selection, planes, 2,
            packed.data() + 64, layout.bytes), RAD_OK);
        std::fill(codes.begin(), codes.end(), 0); std::fill(scales.begin(), scales.end(), 0);
        REQUIRE_EQ(api.row->unrelayout(params.data(), int(params.size()), operand, &encoding, selection,
            packed.data() + 64, layout.bytes, planes, 2), RAD_OK);
        CHECK(codes == original_codes); CHECK(scales == original_scales);
        CHECK(std::all_of(packed.begin(), packed.begin() + 64, [](uint8_t value) { return value == 0x5a; }));
        CHECK(std::all_of(packed.end() - 64, packed.end(), [](uint8_t value) { return value == 0x5a; }));
        CHECK_EQ(api.row->relayout(params.data(), int(params.size()), operand, &encoding, selection, planes, 2,
            packed.data() + 64, layout.bytes - 1), RAD_E_SHAPE);
        for (int malformed = 0; malformed < 5; ++malformed) {
            auto bad = planes[0];
            if (malformed == 0) planes[0].dtype = r3dh_mtp_fc_e4m3_dtype;
            if (malformed == 1) planes[0].rank = 1;
            if (malformed == 2) --planes[0].shape[1];
            if (malformed == 3) ++planes[0].stride[0];
            if (malformed == 4) planes[0].stride[1] = 2;
            CHECK(api.row->layout(params.data(), int(params.size()), operand, &encoding, selection, planes, 2, &layout) != RAD_OK);
            planes[0] = bad;
        }
    }
}

TEST(r3d_mtp_fc_actual_aql_private_views) {
    using namespace mtp_fc_test;
    if (!joined_head_test::device_available()) SKIP("explicit gfx1100 card run required");
    Api api;
    mtp_fc_test::Model model("/home/zihaomu/bigssd2/radiance_workspace/models/qwen3.8-next-flash-iqr-v4.rad");
    constexpr int width = 2560, wide = 10240, physical = 2640;
    Blob codes(width * width, RAD_F8E4M3), scales(width * (width / 128), RAD_F32);
    Blob gain_h(wide, RAD_F32), gain_e(width, RAD_F32);
    model.gain("mtp.pre_fc_norm_hidden.weight", gain_h);
    model.gain("mtp.pre_fc_norm_embedding.weight", gain_e);
    std::vector<uint8_t> packed[2];
    std::vector<float> expanded[2];
    auto params = parameters();
    const RadEncoding encoding = rad_enc_affine(RAD_F8E4M3, RAD_F32, 1, 128);
    const int selection[]{0, 1};
    uint64_t compared = 0, outputs = 0, launches = 0, rejects = 0, fixtures = 0, guards = 0;
    for (int half : {0, 1}) {
        model.matrix(half ? "mtp.fc_embedding.weight" : "mtp.fc_hidden.weight", codes, scales, width);
        RadTensor planes[]{matrix(codes.data<void>(), RAD_F8E4M3, width, width),
            matrix(scales.data<void>(), RAD_F32, width, width / 128)};
        RadLayout layout{};
        REQUIRE_EQ(api.row->layout(params.data(), int(params.size()), half + 4, &encoding, selection, planes, 2, &layout), RAD_OK);
        REQUIRE_EQ(layout.dtype, r3dh_mtp_fc_e4m3_dtype);
        REQUIRE_EQ(layout.bytes, int64_t(width) * physical); REQUIRE_EQ(layout.shape[1], physical);
        packed[half].resize(size_t(layout.bytes)); expanded[half].resize(width * width);
        REQUIRE_EQ(api.row->relayout(params.data(), int(params.size()), half + 4, &encoding, selection, planes, 2,
            packed[half].data(), layout.bytes), RAD_OK);
        for (int row = 0; row < width; ++row) for (int column = 0; column < width; ++column)
            expanded[half][row * width + column] = joined_head_test::multiply(
                joined_head_test::decode(codes.data<uint8_t>()[row * width + column], RAD_F8E4M3),
                scales.data<float>()[row * (width / 128) + column / 128]);
        const auto saved_codes = codes.bytes, saved_scales = scales.bytes;
        std::fill(codes.bytes.begin(), codes.bytes.end(), 0); std::fill(scales.bytes.begin(), scales.bytes.end(), 0);
        REQUIRE_EQ(api.row->unrelayout(params.data(), int(params.size()), half + 4, &encoding, selection,
            packed[half].data(), layout.bytes, planes, 2), RAD_OK);
        CHECK(codes.bytes == saved_codes); CHECK(scales.bytes == saved_scales);
    }
    Queue first(0), second(1);
    for (int memory : {int(RAD_MEM_DEVICE), -1}) {
        Storage fc_h(packed[0].size(), memory), fc_e(packed[1].size(), memory);
        std::copy(packed[0].begin(), packed[0].end(), fc_h.expected.begin() + 64);
        std::copy(packed[1].begin(), packed[1].end(), fc_e.expected.begin() + 64);
        fc_h.push(first.stream); fc_e.push(first.stream); REQUIRE_EQ(rad_stream_sync(first.stream), RAD_OK);
        for (int padding : {0, 16}) for (int count : {1, 2, 3, 63, 64}) {
            RadStream stream = (count + padding) % 2 ? first.stream : second.stream;
            Buffer hidden(RAD_BF16, {count, wide}, RAD_MEM_DEVICE, wide + padding);
            Buffer embedding(RAD_BF16, {count, width}, RAD_MEM_DEVICE, width + padding);
            Buffer wh(RAD_F32, {wide}, RAD_MEM_DEVICE), we(RAD_F32, {width}, RAD_MEM_DEVICE);
            Buffer output(RAD_BF16, {count, wide}, RAD_MEM_DEVICE, wide + padding);
            std::memcpy(wh.host.data, gain_h.data<void>(), gain_h.bytes.size());
            std::memcpy(we.host.data, gain_e.data<void>(), gain_e.bytes.size());
            RadArgs args{}; args.p = params.data(); args.n_p = int(params.size());
            REQUIRE_EQ(api.row->scratch(&args), 2293760);
            Storage scratch(size_t(api.row->scratch(&args)), RAD_MEM_DEVICE);
            std::vector<RadTensor> operands{hidden.device, embedding.device, wh.device, we.device,
                matrix(fc_h.device(), r3dh_mtp_fc_e4m3_dtype, width, width),
                matrix(fc_e.device(), r3dh_mtp_fc_e4m3_dtype, width, width), output.device};
            args.t = operands.data(); args.n_t = int(operands.size());
            args.scratch = scratch.device(); args.scratch_bytes = int64_t(scratch.expected.size() - 128);
            for (int changed : {0, 1}) {
                for (Buffer* input : {&hidden, &embedding}) {
                    auto* values = input->data<uint16_t>();
                    for (int row = 0; row < count; ++row) for (int column = 0; column < input->host.shape[1]; ++column) {
                        uint32_t state = uint32_t(row * 65537 + column * 97 + changed * 104729 + (input == &hidden ? 19 : 23));
                        state = (state ^ (state >> 16)) * 0x7feb352du; state ^= state >> 15;
                        values[row * input->host.stride[0] + column] = rad_f32_to_bf16(float(int(state % 2048) - 1024) / 512);
                    }
                    input->push(stream);
                }
                wh.push(stream); we.push(stream); REQUIRE_EQ(rad_stream_sync(stream), RAD_OK);
                std::vector<uint8_t> golden = output.expected;
                RadTensor oracle_t[]{hidden.host, embedding.host, wh.host, we.host,
                    matrix(expanded[0].data(), RAD_F32, width, width), matrix(expanded[1].data(), RAD_F32, width, width), output.host};
                oracle_t[6].data = golden.data() + 64;
                RadArgs oracle_args = args; oracle_args.t = oracle_t; oracle_args.scratch = nullptr; oracle_args.scratch_bytes = 0;
                REQUIRE_EQ(api.oracle->launch(&oracle_args, nullptr), RAD_OK);
                for (int view : {0, 1, 2}) {
                    for (int half : {4, 5}) {
                        operands[half].dtype = view == 1 ? RAD_F8E4M3 : r3dh_mtp_fc_e4m3_dtype;
                        operands[half].shape[1] = operands[half].stride[0] = view ? physical : width;
                    }
                    const auto saved = operands;
                    output.push(stream); scratch.push(stream); REQUIRE_EQ(rad_stream_sync(stream), RAD_OK);
                    REQUIRE_EQ(api.row->launch(&args, stream), RAD_OK); REQUIRE_EQ(rad_stream_sync(stream), RAD_OK);
                    REQUIRE_EQ(std::memcmp(saved.data(), operands.data(), saved.size() * sizeof(RadTensor)), 0);
                    equal_bytes(output.read(stream), golden, "full_output_padding_guards", compared);
                    const auto observed = scratch.read(stream);
                    REQUIRE(std::equal(observed.begin(), observed.begin() + 64, scratch.expected.begin()));
                    const size_t used = size_t(count) * 35840;
                    REQUIRE(std::equal(observed.begin() + 64 + used, observed.end(), scratch.expected.begin() + 64 + used));
                    guards += observed.size() - used; ++launches; outputs += uint64_t(count) * wide;
                }
            }
            operands[4] = matrix(fc_h.device(), r3dh_mtp_fc_e4m3_dtype, width, width);
            operands[5] = matrix(fc_e.device(), r3dh_mtp_fc_e4m3_dtype, width, width);
            output.push(stream); scratch.push(stream); REQUIRE_EQ(rad_stream_sync(stream), RAD_OK);
            const RadStream invalid_stream = reinterpret_cast<RadStream>(uintptr_t(0x12345));
            for (int half : {4, 5}) for (int malformed = 0; malformed < 15; ++malformed) {
                const auto saved = operands[half]; auto& bad = operands[half];
                if (malformed == 0) bad.dtype = RAD_F8E4M3;
                if (malformed == 1) bad.dtype = RAD_BF16;
                if (malformed == 2) bad.dtype = RAD_F32;
                if (malformed == 3) bad.dtype = RAD_F16;
                if (malformed == 4) bad.dtype = 0x33000401u;
                if (malformed == 5) bad.rank = 1;
                if (malformed == 6) bad.rank = 3;
                if (malformed == 7) --bad.shape[0];
                if (malformed == 8) --bad.shape[1];
                if (malformed == 9) ++bad.stride[0];
                if (malformed == 10) bad.stride[1] = 2;
                if (malformed == 11) bad.stride[0] = physical;
                if (malformed == 12) bad.shape[1] = physical;
                if (malformed == 13) bad.data = static_cast<uint8_t*>(bad.data) + 1;
                if (malformed == 14) { bad.dtype = RAD_F8E4M3; bad.shape[1] = physical; bad.stride[0] = physical + 16; }
                REQUIRE_EQ(api.row->launch(&args, invalid_stream), RAD_E_DTYPE); ++rejects;
                operands[half] = saved;
            }
            for (const char* value : {"bad", ""}) {
                params[4] = RAD_STR("fc", value);
                REQUIRE_EQ(api.row->launch(&args, invalid_stream), RAD_E_UNSUPPORTED); ++rejects;
            }
            params[4] = RAD_STR("missing_fc", "e4m3");
            REQUIRE_EQ(api.row->launch(&args, invalid_stream), RAD_E_UNSUPPORTED); ++rejects;
            params[4] = RAD_STR("fc", "e4m3");
            args.scratch_bytes = int64_t(count) * 35840 - 1;
            REQUIRE_EQ(api.row->launch(&args, invalid_stream), RAD_E_SCRATCH); ++rejects;
            args.scratch_bytes = int64_t(scratch.expected.size() - 128);
            args.scratch = static_cast<uint8_t*>(scratch.device()) + 1;
            REQUIRE_EQ(api.row->launch(&args, invalid_stream), RAD_E_SCRATCH); ++rejects;
            equal_bytes(output.read(stream), output.expected, "rejected_output_untouched", compared);
            equal_bytes(scratch.read(stream), scratch.expected, "rejected_scratch_untouched", compared);
            equal_bytes(hidden.read(stream), hidden.expected, "hidden_immutable", compared);
            equal_bytes(embedding.read(stream), embedding.expected, "embedding_immutable", compared);
            equal_bytes(wh.read(stream), wh.expected, "gain_h_immutable", compared);
            equal_bytes(we.read(stream), we.expected, "gain_e_immutable", compared);
            ++fixtures;
            std::printf("PASS MTP_FC_ACTUAL_CASE memory=%s M=%d band=64 padding=%d ngroup=4 wadd=1 changed_inputs=2 views=private_logical,legacy_F8_physical,private_physical full_output=1\n",
                memory < 0 ? "VMEM" : "DEVICE", count, padding);
            std::fflush(stdout);
        }
        equal_bytes(fc_h.read(first.stream), fc_h.expected, "FC_H_immutable", compared);
        equal_bytes(fc_e.read(first.stream), fc_e.expected, "FC_E_immutable", compared);
    }
    std::printf("PASS MTP_FC_ACTUAL_AQL fixtures=%llu registered_launches=%llu preenqueue_rejects=%llu exhaustive_output_values=%llu exact_bytes=%llu scratch_guard_bytes=%llu actual_weights=both_FC_and_gains TP=replicated queues=2 tolerance=0\n",
        (unsigned long long)fixtures, (unsigned long long)launches, (unsigned long long)rejects,
        (unsigned long long)outputs, (unsigned long long)compared, (unsigned long long)guards);
}

TEST(r3d_mtp_fc_exact_tp2_draft_aql) {
    using namespace mtp_fc_test;
    using rad::RadTape;
    if (!joined_head_test::device_available()) SKIP("explicit gfx1100 card run required");
    struct Flag {
        const char* name;
        bool present = std::getenv(name) != nullptr;
        std::string saved = present ? std::getenv(name) : "";
        void set(const char* value) { REQUIRE_EQ(value ? setenv(name, value, 1) : unsetenv(name), 0); }
        ~Flag() { if (present) setenv(name, saved.c_str(), 1); else unsetenv(name); }
    } flag{"RAD_MTP_FC_EXACT"}, precision{"RAD_PRECISION"};
    Api api;
    const auto* backend = rad::active_backend();
    REQUIRE(backend->tape_begin && backend->tape_end && backend->tape_diff && backend->tape_len && backend->tape_destroy);
    mtp_fc_test::Model model("/home/zihaomu/bigssd2/radiance_workspace/models/qwen3.8-next-flash-iqr-v4.rad");
    constexpr int width = 2560, streams = 4, wide = width * streams, physical = 2640;
    auto params = parameters();
    Queue queue(0);
    Blob codes(width * width, RAD_F8E4M3), scales(width * (width / 128), RAD_F32);
    Blob gain_h(wide, RAD_F32), gain_e(width, RAD_F32);
    model.gain("mtp.pre_fc_norm_hidden.weight", gain_h);
    model.gain("mtp.pre_fc_norm_embedding.weight", gain_e);
    Buffer wh(RAD_F32, {wide}, RAD_MEM_DEVICE), we(RAD_F32, {width}, RAD_MEM_DEVICE);
    std::memcpy(wh.host.data, gain_h.data<void>(), gain_h.bytes.size());
    std::memcpy(we.host.data, gain_e.data<void>(), gain_e.bytes.size());
    wh.push(queue.stream); we.push(queue.stream);
    Storage fc_h(width * physical, -1), fc_e(width * physical, -1);
    const RadEncoding encoding = rad_enc_affine(RAD_F8E4M3, RAD_F32, 1, 128);
    const int selection[]{0, 1};
    for (int half : {0, 1}) {
        auto& fc = half ? fc_e : fc_h;
        model.matrix(half ? "mtp.fc_embedding.weight" : "mtp.fc_hidden.weight", codes, scales, width);
        RadTensor planes[]{matrix(codes.data<void>(), RAD_F8E4M3, width, width),
            matrix(scales.data<void>(), RAD_F32, width, width / 128)};
        RadLayout layout{};
        REQUIRE_EQ(api.row->layout(params.data(), int(params.size()), half + 4, &encoding, selection, planes, 2, &layout), RAD_OK);
        REQUIRE_EQ(layout.dtype, r3dh_mtp_fc_e4m3_dtype); REQUIRE_EQ(layout.bytes, int64_t(width) * physical);
        REQUIRE_EQ(api.row->relayout(params.data(), int(params.size()), half + 4, &encoding, selection, planes, 2,
            fc.expected.data() + 64, layout.bytes), RAD_OK);
        fc.push(queue.stream);
    }
    uint64_t compared = 0;
    bool special = false;
    auto trial = [&](int count, int padding, int world, int rank, int groups, bool legacy_hidden,
                     const char* setting, bool candidate) {
        const int hp = wide + padding, ep = width + 2 * padding, op = wide + 3 * padding;
        Buffer hidden(RAD_BF16, {count + 1, wide}, RAD_MEM_DEVICE, hp);
        Buffer embedding(RAD_BF16, {count + 1, width}, RAD_MEM_DEVICE, ep);
        Buffer output(RAD_BF16, {count + 1, wide}, RAD_MEM_DEVICE, op);
        for (Buffer* input : {&hidden, &embedding}) {
            for (int row = 0; row < count; ++row) {
                auto* values = input->data<uint16_t>() + row * input->host.stride[0];
                for (int col = 0; col < input->host.shape[1]; ++col) {
                    uint32_t bits = uint32_t(col * 97 + row * 65537 + rank * 7919 + (input == &hidden ? 19 : 23));
                    bits = (bits ^ (bits >> 16)) * 0x7feb352du; bits ^= bits >> 15;
                    values[col] = rad_f32_to_bf16(float(int(bits % 2049) - 1024) / 512);
                }
                values[0] = 0x8000; values[127] = 0x0001; values[128] = 0;
                if (special && input == &hidden) values[(row % streams) * width + 129] = row & 1 ? 0xffc5 : 0x7f80;
            }
            input->push(queue.stream);
        }
        RadTensor tensors[]{hidden.device, embedding.device, wh.device, we.device,
            matrix(fc_h.device(), r3dh_mtp_fc_e4m3_dtype, width, width),
            matrix(fc_e.device(), r3dh_mtp_fc_e4m3_dtype, width, width), output.device};
        for (int slot : {0, 1, 6}) tensors[slot].shape[0] = count;
        if (legacy_hidden) tensors[4] = matrix(fc_h.device(), RAD_F8E4M3, width, physical);
        params[3] = RAD_INT("ngroup", groups);
        RadArgs args{}; args.p = params.data(); args.n_p = int(params.size()); args.t = tensors; args.n_t = 7;
        args.world_size = world; args.rank = rank;
        args.scratch_bytes = api.row->scratch(&args);
        REQUIRE_EQ(args.scratch_bytes, 2293760);
        Storage scratch(size_t(args.scratch_bytes), RAD_MEM_DEVICE); args.scratch = scratch.device();
        auto record = [&](const char* value) {
            flag.set(value); output.push(queue.stream); scratch.push(queue.stream);
            if (rad_stream_sync(queue.stream) != RAD_OK || backend->tape_begin() != RAD_OK)
                throw std::runtime_error("MTP FC tape begin");
            const int status = api.row->launch(&args, queue.stream);
            RadTape tape = backend->tape_end(status == RAD_OK);
            if (status != RAD_OK || !tape || backend->tape_len(tape) != 3)
                throw std::runtime_error("MTP FC tape did not capture three launches");
            return tape;
        };
        const RadTape strict = record("0");
        const auto strict_out = output.read(queue.stream), strict_work = scratch.read(queue.stream);
        auto golden_out = output.expected, golden_work = scratch.expected;
        auto arm = [](auto& poison, auto& golden, const auto& actual, size_t begin, size_t bytes) {
            std::copy_n(actual.begin() + begin, bytes, golden.begin() + begin);
            for (size_t index = begin; index < begin + bytes; ++index) poison[index] = uint8_t(~actual[index]);
        };
        for (int row = 0; row < count; ++row)
            arm(output.expected, golden_out, strict_out, 64 + size_t(row) * op * 2, wide * 2);
        arm(scratch.expected, golden_work, strict_work, 64, size_t(count) * width * ((streams + 1) * 2 + 4));
        equal_bytes(strict_out, golden_out, "strict_output_guards", compared);
        equal_bytes(strict_work, golden_work, "strict_scratch_guards", compared);
        const RadTape experiment = record(setting);
        char why[512]{};
        const int difference = backend->tape_diff(strict, experiment, why, sizeof(why));
        backend->tape_destroy(experiment); backend->tape_destroy(strict);
        if (difference != (candidate ? 1 : -1)) std::printf("MTP_FC_ROUTE M=%d setting=%s diff=%d %s\n",
            count, setting ? setting : "unset", difference, why);
        REQUIRE_EQ(difference, candidate ? 1 : -1);
        if (candidate) { REQUIRE(std::strstr(why, "grid 2560x")); REQUIRE(std::strstr(why, "against 81920x")); }
        equal_bytes(output.read(queue.stream), golden_out, "draft_all_output_rows_and_guards", compared);
        equal_bytes(scratch.read(queue.stream), golden_work, "draft_norm_fp32_product_and_guards", compared);
        equal_bytes(hidden.read(queue.stream), hidden.expected, "draft_hidden_immutable", compared);
        equal_bytes(embedding.read(queue.stream), embedding.expected, "draft_embedding_immutable", compared);
    };
    precision.set("exact");
    for (int rank : {0, 1}) for (int count : {1, 4}) for (int padding : {0, 17})
        trial(count, padding, 2, rank, streams, false, "1", true);
    for (const char* disabled : {static_cast<const char*>(nullptr), "", "0", "01"})
        trial(4, 0, 2, 0, streams, false, disabled, false);
    for (int count : {2, 3, 5}) trial(count, 17, 2, 0, streams, false, "1", false);
    trial(4, 0, 1, 0, streams, false, "1", false);
    trial(4, 0, 2, 0, 1, false, "1", false);
    trial(4, 0, 2, 0, streams, true, "1", false);
    precision.set("tolerance"); trial(4, 0, 2, 1, streams, false, "1", true);
    equal_bytes(fc_h.read(queue.stream), fc_h.expected, "model_fc_h_immutable", compared);
    equal_bytes(fc_e.read(queue.stream), fc_e.expected, "model_fc_e_immutable", compared);
    special = true; precision.set("exact");
    fc_e.expected[64 + 127] = 0x7f; fc_e.expected[64 + physical + 128] = 0xff;
    const uint32_t scale_bits[]{0x80000000u, 0x7f800000u, 0xff800000u, 0xffc51234u};
    for (int col = 0; col < 4; ++col)
        std::memcpy(fc_h.expected.data() + 64 + col * physical + width + 19 * sizeof(float), &scale_bits[col], sizeof(float));
    fc_h.push(queue.stream); fc_e.push(queue.stream);
    trial(4, 17, 2, 1, streams, false, "1", true);
    equal_bytes(fc_h.read(queue.stream), fc_h.expected, "special_fc_h_immutable", compared);
    equal_bytes(fc_e.read(queue.stream), fc_e.expected, "special_fc_e_immutable", compared);
    equal_bytes(wh.read(queue.stream), wh.expected, "draft_hidden_gain_immutable", compared);
    equal_bytes(we.read(queue.stream), we.expected, "draft_embedding_gain_immutable", compared);
    std::printf("PASS MTP_FC_EXACT_TP2_DRAFT_AQL exact_bytes=%llu\n", (unsigned long long)compared);
}
