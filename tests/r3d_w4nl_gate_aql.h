#include "rad_planes.h"

namespace w4nl_gate {
using joined_head_test::Storage;
using joined_head_test::multiply;
using joined_head_test::add;
using iqr_output_boundary::matrix;
struct Counts { uint64_t fixtures = 0, calls = 0, values = 0, bytes = 0, rejects = 0, references = 0; };
std::vector<RadParam> parameters(int rank, bool down = false) {
    return {RAD_INT("M", 64), RAD_INT("N", down ? 2560 : rank ? 512 : 768),
        RAD_INT("N_odd", down ? 2560 : rank ? 768 : 512), RAD_INT("K", down ? rank ? 256 : 384 : 2560),
        RAD_INT("K_odd", down ? rank ? 384 : 256 : 2560), RAD_INT("n_expert", 512), RAD_INT("top_k", 10),
        RAD_INT("group", 128), RAD_INT("a_group", 128), RAD_INT("parts", down ? 1 : 2),
        RAD_STR("dtype", "w4nl64a8i8h"), RAD_STR("a_order", down ? "sorted" : "token")};
}
struct Weight {
    int rows, inner;
    RadEncoding encoding{};
    std::array<std::vector<uint8_t>, 4> raw;
    std::array<RadTensor, 4> planes{};
    std::array<std::vector<uint8_t>, 2> native, reference;
    std::array<int, 2> selected[2];
    float level = 0;
    std::array<float, 16> codebook{};
    Weight(rad::RadFile& file, const RadKernelInfo* kernel, Library& libref, int rank, int expert, bool down) {
        rows = down ? 2560 : (rank ^ (expert & 1)) ? 512 : 768;
        inner = down ? (rank ^ (expert & 1)) ? 256 : 384 : 2560;
        const auto name = std::string("blk.48.") + (down ? "ffn_down_exps." : "ffn_gate_up_exps.")
            + std::to_string(expert) + ".weight";
        const auto* entry = file.find(name.c_str()); REQUIRE(entry);
        REQUIRE_EQ(entry->rank, 2u); REQUIRE_EQ(entry->shape[0], down ? 2560 : 1280);
        REQUIRE_EQ(entry->shape[1], down ? 640 : 2560);
        encoding = file.encoding(*entry); REQUIRE_EQ(encoding.n_planes, 4);
        REQUIRE_EQ(std::strcmp(encoding.scheme, "affine"), 0);
        REQUIRE_EQ(std::strcmp(encoding.transform, "fwht128"), 0);
        const char* roles[]{"codes", "table", "scale", "scale.1"};
        for (int index = 0; index < 4; ++index) REQUIRE(rad_enc_plane(&encoding, roles[index]));
        auto params = parameters(rank, down);
        for (int bundle = 0; bundle < 2; ++bundle) {
            rad::WeightInfo info;
            info.decl.rank = 2; info.decl.shape[0] = rows; info.decl.shape[1] = inner;
            info.decl.dtype = bundle ? RAD_F8E4M3 : RAD_U4;
            info.decl.shard = down ? RAD_SHARD_COL : RAD_SHARD_ROW;
            if (!down) {
                info.decl.n_row_parts = 2;
                info.decl.row_parts[0] = rows / 2; info.decl.row_parts[1] = rows / 2;
            }
            info.shard_lo = (rank ^ (expert & 1)) ? 384 : 0;
            info.shard_hi = info.shard_lo + (down ? inner : rows / 2);
            info.n_sel = 2;
            for (int index = 0; index < 2; ++index) {
                const int ordinal = bundle * 2 + index;
                const int plane = int(rad_enc_plane(&encoding, roles[ordinal]) - encoding.plane);
                selected[bundle][index] = plane; info.sel[index] = plane;
                rad_enc_plane_dims(&encoding.plane[plane], rows, inner, &info.sel_rows[index], &info.sel_cols[index]);
            }
            rad::PlaneShare shares[RAD_ENC_MAX_PLANES]; std::string why;
            const int shared = rad::weight_share(info, encoding, entry->rank, entry->shape, 2, rank, shares, &why);
            if (shared) std::printf("W4NL_GATE_SHARE_ERROR %s %s\n", name.c_str(), why.c_str());
            REQUIRE_EQ(shared, RAD_OK);
            for (int index = 0; index < 2; ++index) {
                const int ordinal = bundle * 2 + index, plane = selected[bundle][index];
                const auto& share = shares[index]; raw[ordinal].resize(share.dense_bytes());
                const auto* source = file.data(file.plane(*entry, plane)); size_t cursor = 0;
                for (const auto& rect : share.rects) for (int64_t row = 0; row < rect.rows; ++row) {
                    REQUIRE(cursor + rect.bytes <= raw[ordinal].size());
                    std::memcpy(raw[ordinal].data() + cursor, source + (rect.row0 + row) * share.row_bytes + rect.byte0, rect.bytes);
                    cursor += rect.bytes;
                }
                REQUIRE_EQ(cursor, raw[ordinal].size());
                planes[ordinal] = matrix(raw[ordinal].data(), share.dtype, share.rows, share.cols);
                uint64_t hash = 14695981039346656037ull;
                for (auto byte : raw[ordinal]) hash = (hash ^ byte) * 1099511628211ull;
                std::printf("W4NL_GATE_PART name=%s rank=%d parts=%d span=%lld:%lld role=%s rows=%lld cols=%lld bytes=%zu rects=%zu fnv64=%llu\n",
                    name.c_str(), rank, down ? 1 : 2, (long long)info.shard_lo, (long long)info.shard_hi,
                    roles[ordinal], (long long)share.rows, (long long)share.cols, cursor, share.rects.size(), (unsigned long long)hash);
            }
            const int operand = 2 + (expert & 1) * 4 + bundle;
            RadLayout layout{};
            REQUIRE_EQ(kernel->layout(params.data(), params.size(), operand, &encoding, selected[bundle].data(), planes.data() + bundle * 2, 2, &layout), RAD_OK);
            native[bundle].resize(layout.bytes);
            REQUIRE_EQ(kernel->relayout(params.data(), params.size(), operand, &encoding, selected[bundle].data(), planes.data() + bundle * 2, 2,
                native[bundle].data(), native[bundle].size()), RAD_OK);
            const auto reference_layout = reinterpret_cast<RadLayoutFn>(dlsym(libref.handle, "ref_moe_cb_layout"));
            const auto reference_pack = reinterpret_cast<RadRelayoutFn>(dlsym(libref.handle, "ref_moe_cb_relayout"));
            REQUIRE(reference_layout && reference_pack);
            REQUIRE_EQ(reference_layout(params.data(), params.size(), operand, &encoding, selected[bundle].data(), planes.data() + bundle * 2, 2, &layout), RAD_OK);
            reference[bundle].resize(layout.bytes);
            REQUIRE_EQ(reference_pack(params.data(), params.size(), operand, &encoding, selected[bundle].data(), planes.data() + bundle * 2, 2,
                reference[bundle].data(), reference[bundle].size()), RAD_OK);
        }
        REQUIRE_EQ(raw[0].size(), size_t(rows) * inner / 2); REQUIRE_EQ(raw[1].size(), 64u);
        REQUIRE_EQ(raw[2].size(), size_t(rows) * inner / 64); REQUIRE_EQ(raw[3].size(), 4u);
        std::memcpy(codebook.data(), raw[1].data(), 64); std::memcpy(&level, raw[3].data(), 4);
        REQUIRE_EQ(level, 0x1p-13f);
    }
    uint16_t oracle(const int8_t* activation, const float* scale, int row) const {
        float total = 0;
        for (int group = 0; group < inner / 64; ++group) {
            float subtotal = 0;
            for (int offset = 0; offset < 64; ++offset) {
                const int position = group * 64 + offset;
                const size_t bit = (size_t(row) * inner + position) * 4;
                unsigned code = 0;
                for (unsigned digit = 0; digit < 4; ++digit) code |= unsigned((raw[0][(bit + digit) / 8] >> ((bit + digit) % 8)) & 1) << digit;
                subtotal = add(subtotal, multiply(float(activation[position]), codebook[code]));
            }
            const float factor = multiply(joined_head_test::decode(raw[2][size_t(row) * (inner / 64) + group], RAD_F8E4M3), level);
            total = add(total, multiply(multiply(subtotal, scale[group / 2]), factor));
        }
        return moe_boundary::pack(total);
    }
};
struct Table {
    Storage blob, pointers;
    int64_t pitch;
    uint32_t dtype;
    int rows, inner;
    Table(const std::array<std::unique_ptr<Weight>, 4>& weights, int parity, int bundle, bool stacked, RadStream stream)
        : blob(stacked ? 256 * int64_t((weights[parity]->native[bundle].size() + 15) / 16 * 16)
                       : weights[parity]->native[bundle].size() + 16 + weights[parity + 2]->native[bundle].size(), RAD_MEM_DEVICE),
          pointers(256 * 8, RAD_MEM_DEVICE),
          pitch(stacked ? int64_t((weights[parity]->native[bundle].size() + 15) / 16 * 16) : 0),
          dtype(bundle ? r3dh_moe_w4nl::scales_type : r3dh_moe_w4nl::codes_type),
          rows(weights[parity]->rows), inner(weights[parity]->inner) {
        std::fill(pointers.expected.begin() + 64, pointers.expected.end() - 64, 0);
        for (int index = 0; index < 2; ++index) {
            const auto& packed = weights[parity + index * 2]->native[bundle];
            const size_t offset = stacked ? index * pitch : index ? weights[parity]->native[bundle].size() + 16 : 0;
            std::memcpy(blob.expected.data() + 64 + offset, packed.data(), packed.size());
            const uint64_t address = reinterpret_cast<uintptr_t>(blob.device()) + offset;
            std::memcpy(pointers.expected.data() + 64 + index * 8, &address, 8);
        }
        blob.push(stream); pointers.push(stream);
    }
    RadTensor view() {
        auto value = matrix(pitch ? blob.device() : pointers.device(), dtype, rows, inner);
        value.rank = 3; value.shape[0] = 256; value.shape[1] = rows; value.shape[2] = inner;
        value.stride[0] = pitch; value.stride[1] = inner; value.stride[2] = 1; return value;
    }
    void verify(RadStream stream, Counts& counts) {
        REQUIRE(blob.read(stream) == blob.expected); REQUIRE(pointers.read(stream) == pointers.expected);
        counts.bytes += blob.expected.size() + pointers.expected.size();
    }
};
void equal(Buffer& buffer, RadStream stream, Counts& counts) {
    const auto actual = buffer.read(stream);
    if (actual != buffer.expected) {
        size_t byte = 0; while (byte < actual.size() && actual[byte] == buffer.expected[byte]) ++byte;
        std::printf("W4NL_GATE_DIFF byte=%zu expected=%u actual=%u\n", byte, unsigned(buffer.expected[byte]), unsigned(actual[byte]));
    }
    REQUIRE(actual == buffer.expected); counts.bytes += actual.size();
}
void exercise(rad::RadFile& file, Library& native, Library& reference, int rank, bool stacked, Counts& counts) {
    const std::string suffix = rank ? "512_768_strict" : "768_512_strict";
    const auto* gate = moe_boundary::registered(native, "model_moe_w4nl_gate_" + suffix);
    const auto* down = moe_boundary::registered(native, rank ? "model_moe_w4nl_256_384_strict" : "model_moe_w4nl_384_256_strict");
    const auto* quant = moe_boundary::registered(native, "model_gated_had_384_strict");
    const auto* ref_gemm = reference.ngram("moe_gemm_q");
    const auto ref_quant = reinterpret_cast<RadLaunchFn>(dlsym(reference.handle, "ref_had_quant_act_i8g"));
    REQUIRE(ref_quant);
    std::array<std::unique_ptr<Weight>, 4> weights, down_weights;
    for (int expert = 0; expert < 4; ++expert) {
        weights[expert] = std::make_unique<Weight>(file, gate, reference, rank, expert, false);
        down_weights[expert] = std::make_unique<Weight>(file, down, reference, rank, expert, true);
    }
    Queue first(0), second(0); std::array<std::unique_ptr<Table>, 8> tables;
    for (int parity = 0; parity < 2; ++parity) for (int bundle = 0; bundle < 2; ++bundle) {
        tables[parity * 2 + bundle] = std::make_unique<Table>(weights, parity, bundle, stacked, first.stream);
        tables[4 + parity * 2 + bundle] = std::make_unique<Table>(down_weights, parity, bundle, stacked, first.stream);
    }
    REQUIRE_EQ(rad_stream_sync(first.stream), RAD_OK);
    for (int tokens : {1, 3, 63, 64}) {
        const int rows = tokens * 10;
        Buffer input(RAD_I8, {tokens, 2560}, RAD_MEM_DEVICE, 2573), scale(RAD_F32, {tokens, 20}, RAD_MEM_DEVICE, 25);
        Buffer slots(RAD_I32, {rows}, RAD_MEM_DEVICE), offsets(RAD_I32, {513}, RAD_MEM_DEVICE);
        Buffer output(RAD_BF16, {rows, 768}, RAD_MEM_DEVICE, 781), codes(RAD_I8, {rows, 384}, RAD_MEM_DEVICE, 397);
        Buffer factors(RAD_F32, {rows, 3}, RAD_MEM_DEVICE, 7), result(RAD_BF16, {rows, 2560}, RAD_MEM_DEVICE, 2577);
        std::vector<uint8_t> previous;
        for (int empty = 0; empty < 2; ++empty) for (int step = 0; step < 2; ++step) {
            const auto stream = step ? second.stream : first.stream;
            for (Buffer* buffer : {&input, &scale, &slots, &offsets, &output, &codes, &factors, &result})
                std::fill(buffer->expected.begin(), buffer->expected.end(), 0x5a);
            for (int row = 0; row < tokens; ++row) {
                for (int position = 0; position < 2560; ++position) input.data<int8_t>()[row * 2573 + position] = int8_t(int((position * 71 + row * 37 + step * 29) % 256) - 128);
                for (int group = 0; group < 20; ++group) scale.data<float>()[row * 25 + group] = float(group % 7 + row % 3 + step + 1) / 8192.0f;
            }
            for (int index = 0; index <= 512; ++index) offsets.data<int32_t>()[index] = empty ? 0 : std::min(index * 2, 8);
            for (int index = 0; index < rows; ++index) slots.data<int32_t>()[index] = index < 8 ? (index * 37 + step * 19) % rows : -1;
            slots.data<int32_t>()[1] = -1; slots.data<int32_t>()[6] = rows;
            for (Buffer* buffer : {&input, &scale, &slots, &offsets, &output, &codes, &factors, &result}) buffer->push(stream);
            REQUIRE_EQ(rad_stream_sync(stream), RAD_OK);
            auto params = parameters(rank);
            std::array<RadTensor, 9> device{input.device, scale.device, {}, {}, slots.device, offsets.device, {}, {}, output.device};
            for (int parity = 0; parity < 2; ++parity) for (int bundle = 0; bundle < 2; ++bundle)
                device[2 + parity * 4 + bundle] = tables[parity * 2 + bundle]->view();
            RadArgs args{}; args.t = device.data(); args.n_t = 9; args.p = params.data(); args.n_p = params.size(); args.rank = rank; args.world_size = 2;
            REQUIRE_EQ(gate->launch(&args, stream), RAD_OK); ++counts.calls;
            for (int row = 0; row < rows; ++row) for (int column = 0; column < 768; ++column) {
                uint16_t expected = 0; const int slot = slots.data<int32_t>()[row];
                if (!empty && row < 8 && slot >= 0 && slot < rows) {
                    const auto& weight = *weights[row / 2]; const int local = column % 384;
                    if (local < weight.rows / 2) expected = weight.oracle(input.data<int8_t>() + (slot / 10) * 2573,
                        scale.data<float>() + (slot / 10) * 25, (column / 384) * (weight.rows / 2) + local);
                }
                output.data<uint16_t>()[row * 781 + column] = expected;
            }
            equal(output, stream, counts); counts.values += uint64_t(rows) * 768;
            if (!empty && step) REQUIRE(output.expected != previous);
            previous = output.expected;
            auto compare_reference = [&](bool is_down) {
                auto host = device; host[0] = is_down ? codes.host : input.host; host[1] = is_down ? factors.host : scale.host;
                host[4] = slots.host; host[5] = offsets.host;
                std::vector<int32_t> reference_slots(rows);
                for (int index = 0; index < rows; ++index) {
                    const int slot = slots.data<int32_t>()[index];
                    reference_slots[index] = !empty && index < 8 && slot >= 0 && slot < rows ? slot : -1;
                }
                host[4].data = reference_slots.data();
                const Buffer& expected = is_down ? result : output;
                std::vector<uint8_t> actual(expected.expected.size(), 0x5a);
                for (int row = 0; row < rows; ++row)
                    std::memset(actual.data() + 64 + row * expected.host.stride[0] * 2, 0,
                        size_t(expected.host.shape[1]) * 2);
                host[8] = is_down ? result.host : output.host; host[8].data = actual.data() + 64;
                std::array<std::array<uint64_t, 256>, 4> pointers{};
                auto& source = is_down ? down_weights : weights;
                for (int parity = 0; parity < 2; ++parity) for (int bundle = 0; bundle < 2; ++bundle) {
                    auto& table = pointers[parity * 2 + bundle];
                    table.fill(reinterpret_cast<uintptr_t>(source[parity]->reference[bundle].data()));
                    for (int index = 0; index < 2; ++index) table[index] = reinterpret_cast<uintptr_t>(source[parity + index * 2]->reference[bundle].data());
                    auto value = matrix(table.data(), bundle ? RAD_F8E4M3 : RAD_U4, source[parity]->rows, source[parity]->inner);
                    value.rank = 3; value.shape[0] = 256; value.shape[1] = source[parity]->rows; value.shape[2] = source[parity]->inner;
                    value.stride[0] = 0; value.stride[1] = source[parity]->inner; value.stride[2] = 1;
                    host[2 + parity * 4 + bundle] = value;
                }
                auto geometry = parameters(rank, is_down); RadArgs call = args; call.t = host.data(); call.p = geometry.data(); call.n_p = geometry.size();
                geometry[0] = RAD_INT("M", tokens);
                REQUIRE_EQ(ref_gemm->launch(&call, nullptr), RAD_OK);
                if (actual != expected.expected) {
                    size_t byte = 0;
                    while (byte < actual.size() && actual[byte] == expected.expected[byte]) ++byte;
                    std::printf("W4NL_GATE_REF_DIFF down=%d rank=%d M=%d empty=%d step=%d byte=%zu row=%zu col=%zu expected=%u actual=%u\n",
                        is_down, rank, tokens, empty, step, byte, (byte - 64) / (host[8].stride[0] * 2),
                        ((byte - 64) / 2) % host[8].stride[0], unsigned(expected.expected[byte]), unsigned(actual[byte]));
                    throw std::runtime_error("W4NL reference mismatch");
                }
                ++counts.references;
            };
            compare_reference(false);
            const RadParam quant_params[]{RAD_INT("M", rows), RAD_INT("n", 384), RAD_INT("group", 128), RAD_STR("dtype", "bf16"), RAD_STR("act", "silu")};
            RadTensor quant_device[]{output.device, codes.device, factors.device}; RadArgs quant_args{};
            quant_args.t = quant_device; quant_args.n_t = 3; quant_args.p = quant_params; quant_args.n_p = 5;
            REQUIRE_EQ(quant->launch(&quant_args, stream), RAD_OK); ++counts.calls;
            std::vector<uint16_t> product(size_t(rows) * 384);
            for (int row = 0; row < rows; ++row) for (int column = 0; column < 384; ++column) {
                const auto gate_bits = output.data<uint16_t>()[row * 781 + column];
                const auto up_bits = output.data<uint16_t>()[row * 781 + 384 + column];
                const float gate_value = rad_load_f32(&gate_bits, RAD_BF16, 0);
                const float up_value = rad_load_f32(&up_bits, RAD_BF16, 0);
                const float sigmoid = 1.0f / (1.0f + std::exp(-gate_value));
                product[size_t(row) * 384 + column] = moe_boundary::pack(multiply(multiply(gate_value, sigmoid), up_value));
            }
            RadTensor quant_host[]{matrix(product.data(), RAD_BF16, rows, 384), codes.host, factors.host};
            RadArgs quant_ref = quant_args; quant_ref.t = quant_host;
            REQUIRE_EQ(ref_quant(&quant_ref, nullptr), RAD_OK); ++counts.references;
            equal(codes, stream, counts); equal(factors, stream, counts);
            auto down_params = parameters(rank, true); auto down_tensors = device;
            down_tensors[0] = codes.device; down_tensors[1] = factors.device; down_tensors[8] = result.device;
            for (int parity = 0; parity < 2; ++parity) for (int bundle = 0; bundle < 2; ++bundle)
                down_tensors[2 + parity * 4 + bundle] = tables[4 + parity * 2 + bundle]->view();
            RadArgs down_args = args; down_args.t = down_tensors.data(); down_args.p = down_params.data(); down_args.n_p = down_params.size();
            REQUIRE_EQ(down->launch(&down_args, stream), RAD_OK); ++counts.calls;
            for (int row = 0; row < rows; ++row) for (int column = 0; column < 2560; ++column) {
                const int slot = slots.data<int32_t>()[row]; uint16_t expected = 0;
                if (!empty && row < 8 && slot >= 0 && slot < rows)
                    expected = down_weights[row / 2]->oracle(codes.data<int8_t>() + row * 397, factors.data<float>() + row * 7, column);
                result.data<uint16_t>()[row * 2577 + column] = expected;
            }
            equal(result, stream, counts); counts.values += uint64_t(rows) * 2560; compare_reference(true);
            for (auto* buffer : {&input, &scale, &slots, &offsets}) equal(*buffer, stream, counts);
            if (!empty && !step) {
                for (int invalid = 0; invalid < 20; ++invalid) {
                    auto broken = device; auto geometry = params; int status = RAD_E_SHAPE;
                    switch (invalid) {
                    case 0: broken[0].shape[0] = rows; break;
                    case 1: broken[1].shape[1] = 19; break;
                    case 2: broken[8].shape[1] = 767; break;
                    case 3: broken[2].shape[0] = 255; break;
                    case 4: broken[3].shape[2] = 20; break;
                    case 5: broken[6].stride[0] = -1; status = RAD_E_STRIDE; break;
                    case 6: broken[7].dtype = RAD_F32; status = RAD_E_DTYPE; break;
                    case 7: broken[4].stride[0] = 2; status = RAD_E_STRIDE; break;
                    case 8: broken[5].shape[0] = 512; break;
                    case 9: geometry[0] = RAD_INT("M", 0); break;
                    case 10: geometry[1] = RAD_INT("N", 640); break;
                    case 11: geometry[3] = RAD_INT("K", 2432); break;
                    case 12: geometry[5] = RAD_INT("n_expert", 510); break;
                    case 13: geometry[6] = RAD_INT("top_k", 9); break;
                    case 14: geometry[7] = RAD_INT("group", 64); break;
                    case 15: geometry[9] = RAD_INT("parts", 1); status = RAD_E_UNSUPPORTED; break;
                    case 16: geometry[11] = RAD_STR("a_order", "sorted"); break;
                    case 17: broken[0].data = nullptr; status = RAD_E_INVAL; break;
                    case 18: broken[2].dtype = 0x7f001234u; status = RAD_E_DTYPE; break;
                    case 19: broken[7].dtype = 0x7f001234u; status = RAD_E_DTYPE; break;
                    }
                    RadArgs rejected = args; rejected.t = broken.data(); rejected.p = geometry.data();
                    REQUIRE_EQ(gate->launch(&rejected, reinterpret_cast<RadStream>(uintptr_t(1))), status); ++counts.rejects;
                }
                equal(output, stream, counts);
            }
            ++counts.fixtures;
            std::printf("W4NL_GATE_MATRIX rank=%d layout=%s M=%d band=64 empty=%d input=%d gate=0 quant=0 down=0 cpu_diff=0 libref_diff=0 guards=0\n",
                rank, stacked ? "stacked" : "pointers", tokens, empty, step);
        }
    }
    for (auto& table : tables) table->verify(first.stream, counts);
}
}
TEST(r3d_w4nl_gate_actual_aql) {
    if (!joined_head_test::device_available()) SKIP("explicit gfx1100 card required");
    rad::RadFile file;
    REQUIRE_EQ(file.open("/home/zihaomu/bigssd2/radiance_workspace/models/qwen3.8-next-flash-iqr-v4.rad"), RAD_OK);
    Model metadata("/home/zihaomu/bigssd2/radiance_workspace/models/qwen3.8-next-flash-iqr-v4.rad");
    int experts = 0;
    for (const auto& entry : metadata.entries) if (metadata.name(entry).find("blk.48.ffn_gate_up_exps.") == 0) {
        REQUIRE(entry.enc < metadata.encodings.size()); const auto& encoding = metadata.encodings[entry.enc];
        REQUIRE_EQ(encoding.n_planes, 4); REQUIRE_EQ(entry.shape[0], 1280); REQUIRE_EQ(entry.shape[1], 2560);
        const auto* code = rad_enc_plane(&encoding, "codes"); REQUIRE(code); REQUIRE_EQ(code->dtype, uint32_t(RAD_U4));
        const auto* scale = rad_enc_plane(&encoding, "scale"); REQUIRE(scale); REQUIRE_EQ(scale->dtype, uint32_t(RAD_F8E4M3));
        REQUIRE_EQ(scale->block[1], 64); ++experts;
    }
    REQUIRE_EQ(experts, 512); std::printf("W4NL_GATE_MODEL experts=512 protected_bf16=0 full_N=1280 K=2560\n");
    Library native("libr3d"), reference("libref"); w4nl_gate::Counts counts;
    for (int rank = 0; rank < 2; ++rank) for (bool stacked : {false, true}) w4nl_gate::exercise(file, native, reference, rank, stacked, counts);
    REQUIRE_EQ(counts.fixtures, 64u); REQUIRE_EQ(counts.calls, 192u); REQUIRE_EQ(counts.references, 192u);
    REQUIRE_EQ(counts.rejects, 320u);
    std::printf("W4NL_GATE_PASS fixtures=%llu calls=%llu values=%llu exact_bytes=%llu rejects=%llu libref=%llu gpu=4 tp_shards=2 cpu_fallback=0\n",
        (unsigned long long)counts.fixtures, (unsigned long long)counts.calls, (unsigned long long)counts.values,
        (unsigned long long)counts.bytes, (unsigned long long)counts.rejects, (unsigned long long)counts.references);
}
TEST(r3d_w4nl_gate_layout_host) {
    Library native("libr3d"), reference("libref"); rad::RadFile file;
    REQUIRE_EQ(file.open("/home/zihaomu/bigssd2/radiance_workspace/models/qwen3.8-next-flash-iqr-v4.rad"), RAD_OK);
    uint64_t rejected = 0;
    for (int rank = 0; rank < 2; ++rank) {
        const auto* kernel = moe_boundary::registered(native, rank ? "model_moe_w4nl_gate_512_768_strict" : "model_moe_w4nl_gate_768_512_strict");
        for (int expert : {0, 1}) {
            w4nl_gate::Weight fixture(file, kernel, reference, rank, expert, false);
            auto params = w4nl_gate::parameters(rank);
            for (int bundle = 0; bundle < 2; ++bundle) {
                const int operand = 2 + (expert & 1) * 4 + bundle;
                const int first = bundle * 2; RadLayout layout{};
                for (int invalid = 0; invalid < 8; ++invalid) {
                    auto encoding = fixture.encoding; auto selected = fixture.selected[bundle];
                    auto planes = fixture.planes; int status = RAD_E_DTYPE;
                    switch (invalid) {
                    case 0: std::strcpy(encoding.transform, "fwht64"); break;
                    case 1: std::strcpy(encoding.scheme, "plain"); break;
                    case 2: encoding.plane[fixture.selected[0][0]].dtype = RAD_I4; break;
                    case 3: encoding.plane[fixture.selected[1][0]].block[1] = 32; break;
                    case 4: encoding.plane[fixture.selected[0][1]].extent[1] = 32; break;
                    case 5: std::swap(selected[0], selected[1]); break;
                    case 6: planes[first].shape[0] -= 1; status = RAD_E_SHAPE; break;
                    case 7: selected[1] = selected[0]; status = RAD_E_SHAPE; break;
                    }
                    REQUIRE_EQ(kernel->layout(params.data(), params.size(), operand, &encoding, selected.data(), planes.data() + first, 2, &layout), status);
                    ++rejected;
                }
                const auto saved = fixture.raw[bundle ? 3 : 1];
                if (bundle) { const float bad = 1.0f; std::memcpy(fixture.raw[3].data(), &bad, 4); }
                else { const float bad = std::numeric_limits<float>::quiet_NaN(); std::memcpy(fixture.raw[1].data(), &bad, 4); }
                auto destination = fixture.native[bundle];
                REQUIRE_EQ(kernel->relayout(params.data(), params.size(), operand, &fixture.encoding, fixture.selected[bundle].data(), fixture.planes.data() + first,
                    2, destination.data(), destination.size()), RAD_E_FORMAT);
                ++rejected; std::memcpy(fixture.raw[bundle ? 3 : 1].data(), saved.data(), saved.size());
                REQUIRE_EQ(kernel->relayout(params.data(), params.size(), operand, &fixture.encoding, fixture.selected[bundle].data(), fixture.planes.data() + first,
                    2, destination.data(), destination.size()), RAD_OK);
                REQUIRE(destination == fixture.native[bundle]);
            }
        }
    }
    REQUIRE_EQ(rejected, 72u);
    std::printf("W4NL_GATE_HOST_PASS rejects=%llu shards=2 parities=2 exact_roundtrip=8\n", (unsigned long long)rejected);
}
