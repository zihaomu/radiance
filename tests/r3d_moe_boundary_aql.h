#include "../libr3d/r3dh_moe_iqr.h"
#include "../libr3d/r3dh_moe_w4nl.h"
#include "../libr3d/r3dh_moe_mixed_model.h"
#include <array>

namespace moe_boundary {
void ensure(bool condition) {
    if (!condition) throw std::runtime_error("MoE boundary predicate failed");
}
template<class Left, class Right> void ensure_equal(const Left& left, const Right& right) {
    if (!(left == right)) throw std::runtime_error("MoE boundary equality: " + radtest::show(left, right));
}
const RadKernelInfo* registered(Library& library, const std::string& name) {
    auto count = reinterpret_cast<int (*)()>(dlsym(library.handle, "rad_kernel_count"));
    auto at = reinterpret_cast<const RadKernelInfo* (*)(int)>(dlsym(library.handle, "rad_kernel_at"));
    moe_boundary::ensure(count && at);
    for (int index = 0; index < count(); ++index) {
        const auto* row = at(index);
        if (row && name == row->name) {
            moe_boundary::ensure_equal(row->domain, RAD_DOMAIN_DEVICE);
            moe_boundary::ensure(row->launch);
            return row;
        }
    }
    throw std::runtime_error("missing registered MoE row: " + name);
}
std::vector<uint8_t> plane(Model& model, int layer, int expert, bool down, const char* role) {
    const std::string name = "blk." + std::to_string(layer)
        + (down ? ".ffn_down_exps." : ".ffn_gate_up_exps.") + std::to_string(expert) + ".weight";
    for (const auto& entry : model.entries) {
        if (model.name(entry) != name) continue;
        moe_boundary::ensure(entry.enc < model.encodings.size());
        moe_boundary::ensure_equal(entry.rank, 2u);
        moe_boundary::ensure_equal(entry.shape[0], down ? 2560 : 1280);
        moe_boundary::ensure_equal(entry.shape[1], down ? 640 : 2560);
        const auto& encoding = model.encodings[entry.enc];
        moe_boundary::ensure_equal(entry.n_planes, down ? 4u : 5u);
        moe_boundary::ensure_equal(encoding.n_planes, int(entry.n_planes));
        moe_boundary::ensure(rad_enc_valid(&encoding));
        for (unsigned index = 0; index < entry.n_planes; ++index) {
            if (std::strcmp(encoding.plane[index].role, role)) continue;
            moe_boundary::ensure(entry.plane_first + index < model.planes.size());
            const auto& location = model.planes[entry.plane_first + index];
            moe_boundary::ensure(location.offset >= entry.offset);
            moe_boundary::ensure(location.offset - entry.offset <= entry.bytes);
            moe_boundary::ensure(location.bytes <= entry.bytes - (location.offset - entry.offset));
            std::vector<uint8_t> result(location.bytes);
            model.read(location.offset, result.data(), result.size());
            return result;
        }
    }
    throw std::runtime_error("missing checked model plane: " + name + ":" + role);
}
template<class Value> std::vector<Value> values(const std::vector<uint8_t>& bytes, size_t count) {
    moe_boundary::ensure_equal(bytes.size(), count * sizeof(Value));
    std::vector<Value> result(count);
    std::memcpy(result.data(), bytes.data(), bytes.size());
    return result;
}
struct Expert {
    int width, inner;
    std::vector<uint8_t> gate, scale, down, down_scale;
    std::vector<int8_t> decoded;
    std::vector<float> row, column, codebook, factors;
};
Expert load(Model& model, int layer, int expert, int rank, int parity, bool single = false) {
    using namespace r3dh_moe_iqr;
    Expert result{};
    result.width = single ? 1280 : (rank ^ parity) ? 512 : 768;
    result.inner = result.width / 2;
    const int low = !single && (rank ^ parity) ? 384 : 0;
    auto codes = plane(model, layer, expert, false, "codes");
    moe_boundary::ensure_equal(codes.size(), 1280u * 1280u);
    const auto table = values<int8_t>(plane(model, layer, expert, false, "trellis"), 65536);
    const auto coefficients = values<uint32_t>(plane(model, layer, expert, false, "trellis.mad"), 2);
    const auto full_row = values<float>(plane(model, layer, expert, false, "scale"), 1280);
    result.column = values<float>(plane(model, layer, expert, false, "scale.col"), 20);
    result.row.resize(result.width);
    result.gate.resize(sizeof(Header) + size_t(result.width) * 1280);
    const Header header{magic, 1, uint32_t(result.width), 2560,
        coefficients[0], coefficients[1], 128, 4};
    std::memcpy(result.gate.data(), &header, sizeof(header));
    auto* sharded = result.gate.data() + sizeof(Header);
    for (int row = 0; row < result.width; ++row) {
        const int from = (row / result.inner) * 640 + low + row % result.inner;
        std::memcpy(sharded + size_t(row) * 1280, codes.data() + size_t(from) * 1280, 1280);
        result.row[row] = full_row[from];
    }
    result.scale.resize((result.width + 20) * sizeof(float));
    std::memcpy(result.scale.data(), result.row.data(), result.width * sizeof(float));
    std::memcpy(result.scale.data() + result.width * sizeof(float), result.column.data(), 80);
    result.decoded.resize(size_t(result.width) * 2560);
    for (int column = 0; column < 2560; ++column) {
        auto nibble = [&](int row) { return (sharded[size_t(row) * 1280 + column / 2]
            >> ((column % 2) * 4)) & 15u; };
        for (int first = 0; first < result.width; first += 128) {
            uint32_t state = 0;
            for (int back = 0; back < 4; ++back) state |= nibble(first + 127 - back) << (back * 4);
            for (int offset = 0; offset < 128; ++offset) {
                state = ((state << 4) | nibble(first + offset)) & 65535u;
                result.decoded[size_t(first + offset) * 2560 + column] = table[state];
            }
        }
    }
    using namespace r3dh_moe_w4nl;
    codes = plane(model, layer, expert, true, "codes");
    const auto scales = plane(model, layer, expert, true, "scale");
    moe_boundary::ensure_equal(codes.size(), 2560u * 320u); moe_boundary::ensure_equal(scales.size(), 2560u * 10u);
    result.codebook = values<float>(plane(model, layer, expert, true, "table"), 16);
    const auto level = values<float>(plane(model, layer, expert, true, "scale.1"), 1);
    moe_boundary::ensure_equal(level[0], 0x1p-13f);
    CodesHeader code_header{codes_magic, 1, 2560, uint32_t(result.inner), {}};
    std::copy(result.codebook.begin(), result.codebook.end(), code_header.table);
    const ScalesHeader scale_header{scales_magic, 1, 2560, uint32_t(result.inner), level[0], {}};
    result.down.resize(sizeof(code_header) + size_t(2560) * result.inner / 2);
    result.down_scale.resize(sizeof(scale_header) + size_t(2560) * result.inner / 64);
    std::memcpy(result.down.data(), &code_header, sizeof(code_header));
    std::memcpy(result.down_scale.data(), &scale_header, sizeof(scale_header));
    result.factors.resize(2560 * result.inner / 64);
    for (int row = 0; row < 2560; ++row) {
        std::memcpy(result.down.data() + sizeof(code_header) + size_t(row) * result.inner / 2,
            codes.data() + row * 320 + low / 2, result.inner / 2);
        std::memcpy(result.down_scale.data() + sizeof(scale_header) + size_t(row) * result.inner / 64,
            scales.data() + row * 10 + low / 64, result.inner / 64);
        for (int group = 0; group < result.inner / 64; ++group) {
            const uint8_t bits = scales[row * 10 + low / 64 + group];
            const int exponent = (bits >> 3) & 15, fraction = bits & 7;
            moe_boundary::ensure(exponent != 15 || fraction != 7);
            const float positive = exponent ? std::ldexp(float(8 + fraction), exponent - 10)
                : std::ldexp(float(fraction), -9);
            result.factors[row * (result.inner / 64) + group] = (bits & 128 ? -positive : positive) * level[0];
        }
    }
    return result;
}
struct Table {
    Buffer blob, pointers;
    int entries;
    Table(const std::vector<uint8_t>& data, int count, RadStream stream)
        : blob(RAD_I32, {int64_t((data.size() + 3) / 4)}, RAD_MEM_DEVICE),
          pointers(RAD_I64, {count}, RAD_MEM_DEVICE), entries(count) {
        std::memcpy(blob.host.data, data.data(), data.size());
        for (int index = 0; index < count; ++index)
            pointers.data<uint64_t>()[index] = reinterpret_cast<uintptr_t>(blob.device.data);
        blob.push(stream); pointers.push(stream);
    }
    RadTensor view(uint32_t dtype, int rows, int columns, int64_t bytes = 0) {
        auto tensor = pointers.device; tensor.dtype = dtype; tensor.rank = bytes ? 2 : 3;
        tensor.shape[0] = entries; tensor.shape[1] = bytes ? bytes : rows;
        tensor.shape[2] = bytes ? 0 : columns;
        tensor.stride[0] = 0; tensor.stride[1] = bytes ? 1 : columns; tensor.stride[2] = bytes ? 0 : 1;
        return tensor;
    }
};
RadTensor live(Buffer& buffer, int rows) { auto tensor = buffer.device; tensor.shape[0] = rows; return tensor; }
uint16_t pack(float value) {
    uint32_t bits; std::memcpy(&bits, &value, 4);
    uint32_t high = bits >> 16, low = bits & 65535u;
    if ((bits & 0x7fffffffu) > 0x7f800000u) return uint16_t(high | 64u);
    if (low > 32768u || (low == 32768u && (high & 1u))) ++high;
    return uint16_t(high);
}
void capture(Buffer& buffer, RadStream stream, int rows, int columns, int bytes) {
    auto actual = buffer.read(stream);
    auto expected = buffer.expected;
    for (int row = 0; row < rows; ++row) {
        const size_t begin = 64 + size_t(row) * buffer.host.stride[0] * bytes;
        std::copy_n(actual.data() + begin, columns * bytes, expected.data() + begin);
    }
    moe_boundary::ensure(actual == expected);
    buffer.expected = std::move(actual); buffer.host.data = buffer.expected.data() + 64;
}
void chain(Model& model, Library& library, int layer, int rank, RadStream stream,
           uint64_t& launches, uint64_t& rejects, uint64_t& compared, bool single = false) {
    const int experts = layer == 0 ? 512 : layer == 47 ? 508 : 510;
    std::array<Expert, 2> weights{load(model, layer, layer == 34 ? 409 : 0, rank, 0, single),
        load(model, layer, layer == 34 ? 408 : 1, rank, 1, single)};
    const int width = single ? 1280 : 768, inner = width / 2;
    const int output_pitch = width + 13, code_pitch = inner + 13, factor_pitch = inner / 128 + 4;
    const std::string suffix = single ? "1280_1280_strict" : rank ? "512_768_strict" : "768_512_strict";
    const auto* gate = registered(library, layer == 0 ? "model_moe_iqr_" + suffix
        : "mixed_iqr_" + std::to_string(experts) + "_" + suffix);
    const std::string down_suffix = single ? "640_640_strict" : rank ? "256_384_strict" : "384_256_strict";
    const auto* down = registered(library, layer == 0 ? "model_moe_w4nl_" + down_suffix
        : "mixed_w4nl_" + std::to_string(experts) + "_" + down_suffix);
    const auto* quant = registered(library, single ? "model_gated_had_640_strict" : "model_gated_had_384_strict");
    std::array<std::unique_ptr<Table>, 8> tables;
    for (int parity = 0; parity < 2; ++parity) {
        auto& weight = weights[parity];
        tables[parity * 4] = std::make_unique<Table>(weight.gate, experts / 2, stream);
        tables[parity * 4 + 1] = std::make_unique<Table>(weight.scale, experts / 2, stream);
        tables[parity * 4 + 2] = std::make_unique<Table>(weight.down, experts / 2, stream);
        tables[parity * 4 + 3] = std::make_unique<Table>(weight.down_scale, experts / 2, stream);
    }
    for (int tokens : {1, 3, 4, 16, 63, 64, 2048}) {
        if (single && tokens != 1 && tokens != 4) continue;
        if (std::getenv("RAD_IQR_QUICK") && tokens != 1 && tokens != 4 && tokens != 16 && tokens != 64) continue;
        const int band = std::max(tokens, 64);
        const int rows = tokens * 10;
        Buffer input(RAD_I8, {tokens + 2, 2560}, RAD_MEM_DEVICE, 2573);
        Buffer scale(RAD_F32, {tokens + 2, 20}, RAD_MEM_DEVICE, 25);
        Buffer sorted(RAD_I32, {rows}, RAD_MEM_DEVICE), offsets(RAD_I32, {experts + 1}, RAD_MEM_DEVICE);
        Buffer output(RAD_BF16, {rows + 2, width}, RAD_MEM_DEVICE, output_pitch);
        Buffer codes(RAD_I8, {rows + 2, inner}, RAD_MEM_DEVICE, code_pitch);
        Buffer factors(RAD_F32, {rows + 2, inner / 128}, RAD_MEM_DEVICE, factor_pitch);
        Buffer result(RAD_BF16, {rows + 2, 2560}, RAD_MEM_DEVICE, 2577);
        for (int index = 0; index <= experts; ++index) offsets.data<int32_t>()[index] = std::min(index * 2, 8);
        offsets.push(stream);
        std::vector<uint8_t> previous;
        for (int step = 0; step < 2; ++step) {
            for (Buffer* buffer : {&input, &scale, &output, &codes, &factors, &result})
                std::fill(buffer->expected.begin(), buffer->expected.end(), 0x5a);
            for (int row = 0; row < tokens; ++row) {
                for (int column = 0; column < 2560; ++column)
                    input.data<int8_t>()[row * 2573 + column] = int8_t(int((column * 19 + row * 37 + step * 29) % 256) - 128);
                for (int group = 0; group < 20; ++group)
                    scale.data<float>()[row * 25 + group] = float(group % 7 + row % 3 + step + 1) / 8192.0f;
            }
            for (int row = 0; row < rows; ++row) sorted.data<int32_t>()[row] = row < 8 ? (row * 37 + step * 19) % rows : -1;
            sorted.data<int32_t>()[1] = -1; sorted.data<int32_t>()[6] = rows + 1;
            for (Buffer* buffer : {&input, &scale, &sorted, &output, &codes, &factors, &result}) buffer->push(stream);
            std::vector<RadParam> gate_params{RAD_INT("M", band), RAD_INT("N", weights[0].width), RAD_INT("N_odd", weights[1].width),
                RAD_INT("K", 2560), RAD_INT("K_odd", 2560), RAD_INT("n_expert", experts), RAD_INT("top_k", 10),
                RAD_INT("group", 128), RAD_INT("a_group", 128), RAD_INT("parts", 2), RAD_STR("dtype", "i8a8h"), RAD_STR("a_order", "token")};
            std::array<RadTensor, 9> tensors{live(input, tokens), live(scale, tokens), {}, {}, sorted.device, offsets.device, {}, {}, live(output, rows)};
            for (int parity = 0; parity < 2; ++parity) {
                tensors[2 + parity * 4] = tables[parity * 4]->view(r3dh_moe_iqr::weight_dtype, weights[parity].width, 2560);
                tensors[3 + parity * 4] = tables[parity * 4 + 1]->view(r3dh_moe_iqr::scale_dtype, weights[parity].width, step ? 2560 : 20);
            }
            RadArgs args{}; args.t = tensors.data(); args.n_t = tensors.size(); args.p = gate_params.data(); args.n_p = gate_params.size(); args.rank = rank; args.world_size = single ? 1 : 2;
            moe_boundary::ensure_equal(gate->launch(&args, stream), RAD_OK); ++launches;
            for (int row = 0; row < rows; ++row) for (int column = 0; column < width; ++column) {
                uint16_t expected = 0;
                const int slot = sorted.data<int32_t>()[row];
                if (row < 8 && slot >= 0 && slot < rows) {
                    const auto& weight = weights[(row / 2) % 2];
                    const int local = column % inner, source = slot / 10;
                    if (local < weight.inner) {
                        const int weight_row = (column / inner) * weight.inner + local;
                        float sum = 0;
                        for (int group = 0; group < 20; ++group) {
                            int32_t dot = 0;
                            for (int position = group * 128; position < (group + 1) * 128; ++position)
                                dot += int32_t(input.data<int8_t>()[source * 2573 + position]) * int32_t(weight.decoded[size_t(weight_row) * 2560 + position]);
                            const float scaled = float(dot) * scale.data<float>()[source * 25 + group];
                            sum = sum + scaled * weight.column[group];
                        }
                        expected = pack(sum * weight.row[weight_row]);
                    }
                }
                output.data<uint16_t>()[row * output_pitch + column] = expected;
            }
            moe_boundary::ensure(output.read(stream) == output.expected); compared += uint64_t(rows) * width;
            moe_boundary::ensure(gate->scratch);
            const int64_t scratch_bytes = gate->scratch(&args);
            moe_boundary::ensure(scratch_bytes > 0);
            Buffer scratch(RAD_I32, {(scratch_bytes + 3) / 4}, RAD_MEM_DEVICE);
            scratch.push(stream);
            const auto expected_gate = output.expected;
            auto poison_gate = [&] {
                std::fill(output.expected.begin(), output.expected.end(), 0x5a);
                output.push(stream);
                std::copy(expected_gate.begin(), expected_gate.end(), output.expected.begin());
            };
            poison_gate();
            args.scratch = scratch.device.data;
            args.scratch_bytes = scratch_bytes;
            moe_boundary::ensure_equal(gate->launch(&args, stream), RAD_OK); ++launches;
            moe_boundary::ensure(output.read(stream) == expected_gate); compared += uint64_t(rows) * width;
            const auto scratch_actual = scratch.read(stream);
            for (size_t index = 0; index < scratch_actual.size(); ++index)
                if (index < 64 || index >= 64 + size_t(scratch_bytes))
                    moe_boundary::ensure_equal(scratch_actual[index], uint8_t(0x5a));
            poison_gate();
            const int64_t live_scratch_bytes = int64_t(std::min(rows, experts) + (rows + 63) / 64 + 1) * 8;
            moe_boundary::ensure(scratch_bytes >= live_scratch_bytes);
            args.scratch_bytes = live_scratch_bytes - 1;
            moe_boundary::ensure_equal(gate->launch(&args, stream), RAD_OK); ++launches;
            moe_boundary::ensure(output.read(stream) == expected_gate);
            args.scratch = nullptr;
            args.scratch_bytes = 0;
            std::array<RadTensor, 3> quant_tensors{live(output, rows), live(codes, rows), live(factors, rows)};
            std::vector<RadParam> quant_params{RAD_INT("M", rows), RAD_INT("n", inner), RAD_INT("group", 128), RAD_STR("dtype", "bf16"), RAD_STR("act", "silu")};
            RadArgs quant_args{}; quant_args.t = quant_tensors.data(); quant_args.n_t = 3; quant_args.p = quant_params.data(); quant_args.n_p = quant_params.size();
            moe_boundary::ensure_equal(quant->launch(&quant_args, stream), RAD_OK); ++launches;
            capture(codes, stream, rows, inner, 1); capture(factors, stream, rows, inner / 128, 4);
            std::vector<RadParam> down_params{RAD_INT("M", band), RAD_INT("N", 2560), RAD_INT("N_odd", 2560),
                RAD_INT("K", weights[0].inner), RAD_INT("K_odd", weights[1].inner), RAD_INT("n_expert", experts), RAD_INT("top_k", 10),
                RAD_INT("group", 128), RAD_INT("a_group", 128), RAD_INT("parts", 1), RAD_STR("dtype", "w4nl64a8i8h"), RAD_STR("a_order", "sorted")};
            auto down_tensors = tensors; down_tensors[0] = live(codes, rows); down_tensors[1] = live(factors, rows); down_tensors[8] = live(result, rows);
            for (int parity = 0; parity < 2; ++parity) {
                const auto& weight = weights[parity];
                down_tensors[2 + parity * 4] = tables[parity * 4 + 2]->view(r3dh_moe_w4nl::codes_type, 2560, weight.inner, step ? 0 : weight.down.size());
                down_tensors[3 + parity * 4] = tables[parity * 4 + 3]->view(r3dh_moe_w4nl::scales_type, 2560, weight.inner, step ? 0 : weight.down_scale.size());
            }
            RadArgs down_args = args; down_args.t = down_tensors.data(); down_args.p = down_params.data(); down_args.n_p = down_params.size();
            moe_boundary::ensure_equal(down->launch(&down_args, stream), RAD_OK); ++launches;
            for (int row = 0; row < rows; ++row) for (int column = 0; column < 2560; ++column) {
                uint16_t expected = 0;
                const int slot = sorted.data<int32_t>()[row];
                if (row < 8 && slot >= 0 && slot < rows) {
                    const auto& weight = weights[(row / 2) % 2];
                    const auto* packed = weight.down.data() + sizeof(r3dh_moe_w4nl::CodesHeader) + size_t(column) * weight.inner / 2;
                    float sum = 0;
                    for (int group = 0; group < weight.inner / 64; ++group) {
                        float dot = 0;
                        for (int position = group * 64; position < (group + 1) * 64; ++position) {
                            const int code = (packed[position / 2] >> ((position % 2) * 4)) & 15;
                            const float product = float(codes.data<int8_t>()[row * code_pitch + position]) * weight.codebook[code];
                            dot = dot + product;
                        }
                        const float first = dot * factors.data<float>()[row * factor_pitch + group / 2];
                        sum = sum + first * weight.factors[column * (weight.inner / 64) + group];
                    }
                    expected = pack(sum);
                }
                result.data<uint16_t>()[row * 2577 + column] = expected;
            }
            const auto actual = result.read(stream);
            moe_boundary::ensure(actual == result.expected); compared += uint64_t(rows) * 2560;
            if (step) moe_boundary::ensure(actual != previous);
            previous = actual;
            if (std::getenv("RAD_W4NL_BENCH")) {
                const char* original = std::getenv("RAD_W4NL_DOT4");
                const bool present = original != nullptr;
                const std::string saved = original ? original : "";
                RadEvent start{}, finish{};
                ensure_equal(rad_event_create(&start), RAD_OK);
                ensure_equal(rad_event_create(&finish), RAD_OK);
                float times[2]{};
                for (int variant : {0, 1, 1, 0}) {
                    ensure_equal(setenv("RAD_W4NL_DOT4", variant ? "1" : "0", 1), 0);
                    for (int warmup = 0; warmup < 3; ++warmup)
                        ensure_equal(down->launch(&down_args, stream), RAD_OK);
                    ensure_equal(rad_event_record(start, stream), RAD_OK);
                    for (int repeat = 0; repeat < 20; ++repeat)
                        ensure_equal(down->launch(&down_args, stream), RAD_OK);
                    ensure_equal(rad_event_record(finish, stream), RAD_OK);
                    ensure_equal(rad_event_sync(finish), RAD_OK);
                    float elapsed = 0;
                    ensure_equal(rad_event_elapsed_ms(start, finish, &elapsed), RAD_OK);
                    times[variant] += elapsed * 25.0f;
                    ensure(result.read(stream) == result.expected);
                }
                rad_event_destroy(start);
                rad_event_destroy(finish);
                if (present) setenv("RAD_W4NL_DOT4", saved.c_str(), 1); else unsetenv("RAD_W4NL_DOT4");
                std::printf("W4NL_BENCH layer=%d rank=%d tokens=%d K=%d/%d step=%d scalar_us=%.3f dot4_us=%.3f speedup=%.3f\n",
                    layer, rank, tokens, weights[0].inner, weights[1].inner, step,
                    times[0], times[1], times[0] / times[1]);
            }
            if (single && layer == 0 && tokens == 1 && step == 0 && std::getenv("RAD_W4NL_STRESS")) {
                const char* original = std::getenv("RAD_W4NL_DOT4");
                const bool present = original != nullptr;
                const std::string saved = original ? original : "";
                const auto saved_codes = codes.expected, saved_factors = factors.expected;
                const auto saved_result = result.expected;
                for (int pattern = 0; pattern < 6; ++pattern) {
                    for (int row = 0; row < rows; ++row) {
                        for (int position = 0; position < inner; ++position)
                            codes.data<int8_t>()[row * code_pitch + position] = pattern == 0 ? -128
                                : pattern == 1 ? 127 : position % 2 ? -128 : 127;
                        for (int group = 0; group < inner / 128; ++group) {
                            const uint32_t bits[] = {0u, 0x80000000u, 1u, 0x7f7fffffu, 0x7f800000u, 0x7fc01234u};
                            factors.data<float>()[row * factor_pitch + group] = pattern >= 2
                                ? __builtin_bit_cast(float, bits[(row + group) % 6]) : 0x1p-10f;
                        }
                    }
                    codes.push(stream); factors.push(stream);
                    for (int parity = 0; parity < 2; ++parity) {
                        auto changed = weights[parity].down;
                        auto* header = reinterpret_cast<r3dh_moe_w4nl::CodesHeader*>(changed.data());
                        if (pattern == 3)
                            for (size_t position = sizeof(*header); position < changed.size(); ++position)
                                changed[position] = uint8_t(position);
                        if (pattern == 4) { header->table[0] = 1.12345f; header->table[15] = -22.03125f; }
                        if (pattern == 5) header->version = 2;
                        auto& blob = tables[parity * 4 + 2]->blob;
                        std::memcpy(blob.host.data, changed.data(), changed.size()); blob.push(stream);
                    }
                    const char* run = std::getenv("RAD_W4NL_RUN");
                    const std::string saved_run = run ? run : "";
                    ensure_equal(setenv("RAD_W4NL_DOT4", "0", 1), 0);
                    ensure_equal(setenv("RAD_W4NL_RUN", "0", 1), 0);
                    ensure_equal(down->launch(&down_args, stream), RAD_OK); ++launches;
                    const auto reference = result.read(stream);
                    if (run) setenv("RAD_W4NL_RUN", saved_run.c_str(), 1); else unsetenv("RAD_W4NL_RUN");
                    ensure_equal(setenv("RAD_W4NL_DOT4", "1", 1), 0);
                    result.push(stream);
                    ensure_equal(down->launch(&down_args, stream), RAD_OK); ++launches;
                    ensure(result.read(stream) == reference); compared += uint64_t(rows) * 2560;
                }
                std::copy(saved_codes.begin(), saved_codes.end(), codes.expected.begin()); codes.push(stream);
                std::copy(saved_factors.begin(), saved_factors.end(), factors.expected.begin()); factors.push(stream);
                std::copy(saved_result.begin(), saved_result.end(), result.expected.begin()); result.push(stream);
                for (int parity = 0; parity < 2; ++parity) {
                    auto& blob = tables[parity * 4 + 2]->blob;
                    std::memcpy(blob.host.data, weights[parity].down.data(), weights[parity].down.size()); blob.push(stream);
                }
                if (present) setenv("RAD_W4NL_DOT4", saved.c_str(), 1); else unsetenv("RAD_W4NL_DOT4");
                std::printf("W4NL_STRESS_PASS patterns=6 values=%d\n", rows * 2560 * 6);
            }
            moe_boundary::ensure(input.read(stream) == input.expected); moe_boundary::ensure(scale.read(stream) == scale.expected);
            moe_boundary::ensure(sorted.read(stream) == sorted.expected); moe_boundary::ensure(offsets.read(stream) == offsets.expected);
            auto reject = [&](const RadKernelInfo* row, RadArgs& call, int operand, RadTensor invalid, int code) {
                auto* operands = const_cast<RadTensor*>(call.t); const auto saved = operands[operand]; operands[operand] = invalid;
                moe_boundary::ensure_equal(row->launch(&call, stream), code); ++rejects; operands[operand] = saved;
            };
            for (bool is_down : {false, true}) {
                auto& call = is_down ? down_args : args; const auto* row = is_down ? down : gate;
                auto invalid = call.t[8]; invalid.shape[0]++; reject(row, call, 8, invalid, RAD_E_SHAPE);
                invalid = call.t[8]; invalid.shape[0] = 0; reject(row, call, 8, invalid, RAD_E_SHAPE);
                invalid = call.t[8]; invalid.shape[0] = 650; reject(row, call, 8, invalid, RAD_E_SHAPE);
                invalid = call.t[2]; invalid.dtype++; reject(row, call, 2, invalid, RAD_E_DTYPE);
                invalid = call.t[2]; invalid.shape[0]--; reject(row, call, 2, invalid, RAD_E_SHAPE);
                invalid = call.t[3]; invalid.shape[1]++; reject(row, call, 3, invalid, RAD_E_SHAPE);
                invalid = call.t[3]; invalid.stride[1]++; reject(row, call, 3, invalid, RAD_E_STRIDE);
                invalid = call.t[0]; invalid.shape[0]++; reject(row, call, 0, invalid, RAD_E_SHAPE);
                auto* params = const_cast<RadParam*>(call.p); params[0].ival = 2049;
                moe_boundary::ensure_equal(row->launch(&call, stream), RAD_E_SHAPE); ++rejects; params[0].ival = band;
                params[6].ival = 11; moe_boundary::ensure_equal(row->launch(&call, stream), RAD_E_SHAPE); ++rejects; params[6].ival = 10;
            }
            moe_boundary::ensure(result.read(stream) == result.expected);
            if (step == 1) {
                for (int pattern = 0; pattern < 5; ++pattern) {
                    for (int expert = 0; expert <= experts; ++expert)
                        offsets.data<int32_t>()[expert] = pattern == 0 ? std::min(expert * 17, rows - 1)
                            : pattern == 1 ? expert * rows / experts : pattern == 4 ? std::min(expert * 4, rows - 1)
                            : pattern == 2 && expert == experts ? rows : 0;
                    for (int row = 0; row < rows; ++row)
                        sorted.data<int32_t>()[row] = row == 1 ? -1 : row == 6 ? rows + 1 : (row * 37) % rows;
                    offsets.push(stream);
                    sorted.push(stream);
                    args.scratch = nullptr;
                    args.scratch_bytes = 0;
                    poison_gate();
                    moe_boundary::ensure_equal(gate->launch(&args, stream), RAD_OK); ++launches;
                    const auto reference = output.read(stream);
                    poison_gate();
                    args.scratch = scratch.device.data;
                    args.scratch_bytes = scratch_bytes;
                    moe_boundary::ensure_equal(gate->launch(&args, stream), RAD_OK); ++launches;
                    moe_boundary::ensure(output.read(stream) == reference);
                    compared += uint64_t(rows) * width;
                    const auto map_actual = scratch.read(stream);
                    for (size_t index = 0; index < map_actual.size(); ++index)
                        if (index < 64 || index >= 64 + size_t(scratch_bytes))
                            moe_boundary::ensure_equal(map_actual[index], uint8_t(0x5a));
                    if (std::getenv("RAD_IQR_BENCH")) {
                        RadEvent start{}, finish{};
                        moe_boundary::ensure_equal(rad_event_create(&start), RAD_OK);
                        moe_boundary::ensure_equal(rad_event_create(&finish), RAD_OK);
                        float milliseconds[2]{};
                        for (int variant = 0; variant < 2; ++variant) {
                            args.scratch = variant ? scratch.device.data : nullptr;
                            args.scratch_bytes = variant ? scratch_bytes : 0;
                            for (int warmup = 0; warmup < 3; ++warmup)
                                moe_boundary::ensure_equal(gate->launch(&args, stream), RAD_OK);
                            moe_boundary::ensure_equal(rad_event_record(start, stream), RAD_OK);
                            for (int repeat = 0; repeat < 20; ++repeat)
                                moe_boundary::ensure_equal(gate->launch(&args, stream), RAD_OK);
                            moe_boundary::ensure_equal(rad_event_record(finish, stream), RAD_OK);
                            moe_boundary::ensure_equal(rad_event_sync(finish), RAD_OK);
                            moe_boundary::ensure_equal(rad_event_elapsed_ms(start, finish, &milliseconds[variant]), RAD_OK);
                        }
                        rad_event_destroy(start);
                        rad_event_destroy(finish);
                        std::printf("IQR_BENCH layer=%d rank=%d tokens=%d pattern=%d reference_us=%.3f mapped_us=%.3f speedup=%.3f\n",
                            layer, rank, tokens, pattern, milliseconds[0] * 50.0f,
                            milliseconds[1] * 50.0f, milliseconds[0] / milliseconds[1]);
                    }
                    args.scratch = nullptr;
                    args.scratch_bytes = 0;
                }
            }
        }
        std::printf("PASS MOE_BOUNDARY_AQL_CASE layer=%d rank=%d experts=%d entries=%d M=%d band=%d gate=%d/%d down=%d/%d inputs=2 views=physical,logical pitched=1 guards=1\n",
            layer, rank, experts, experts / 2, tokens, band, weights[0].width, weights[1].width, weights[0].inner, weights[1].inner);
        std::fflush(stdout);
    }
    for (auto& table : tables) {
        moe_boundary::ensure(table->blob.read(stream) == table->blob.expected);
        moe_boundary::ensure(table->pointers.read(stream) == table->pointers.expected);
    }
}
}

TEST(r3d_moe_actual_aql_live_band_chain) {
    CHECK(true);
    RadDeviceProps props{}; moe_boundary::ensure_equal(rad_dev_props(0, &props), RAD_OK);
    if (props.is_host_backend) {
        moe_boundary::ensure(!std::getenv("RAD_R3D_REQUIRE_DEVICE"));
        SKIP("explicit gfx1100 card run required");
    }
    moe_boundary::ensure_equal(std::strcmp(props.arch, "gfx1100"), 0); moe_boundary::ensure_equal(rad_dev_set(0), RAD_OK);
    Library library("libr3d"); Queue first(0), second(1);
    Model model("/home/zihaomu/bigssd2/radiance_workspace/models/qwen3.8-next-flash-iqr-v4.rad");
    uint64_t launches = 0, rejects = 0, compared = 0;
    for (int layer : {0, 34, 44, 47}) for (int rank : {0, 1})
        moe_boundary::chain(model, library, layer, rank, rank ? second.stream : first.stream, launches, rejects, compared);
    std::printf("PASS MOE_BOUNDARY_AQL fixtures=40 launches=%llu rejects=%llu exact_values=%llu queues=2 cpu_fallback=0\n",
        (unsigned long long)launches, (unsigned long long)rejects, (unsigned long long)compared);
}

namespace moe_boundary {
std::vector<uint16_t> protected_weight(Model& model, int layer, int expert, bool down, int rank, bool single = false) {
    const std::string name = "blk." + std::to_string(layer)
        + (down ? ".ffn_down_exps." : ".ffn_gate_up_exps.") + std::to_string(expert) + ".weight";
    for (const auto& entry : model.entries) {
        if (model.name(entry) != name) continue;
        ensure(entry.enc < model.encodings.size() && entry.plane_first < model.planes.size());
        const auto& encoding = model.encodings[entry.enc];
        ensure(entry.rank == 2 && entry.n_planes == 1 && encoding.n_planes == 1
            && entry.shape[0] == (down ? 2560 : 1280) && entry.shape[1] == (down ? 640 : 2560)
            && encoding.plane[0].dtype == RAD_BF16 && !encoding.transform[0]);
        const auto& location = model.planes[entry.plane_first];
        ensure(location.bytes == uint64_t(entry.shape[0] * entry.shape[1] * 2)
            && location.offset >= entry.offset && location.offset - entry.offset <= entry.bytes
            && location.bytes <= entry.bytes - (location.offset - entry.offset));
        std::vector<uint16_t> full(location.bytes / 2), shard(down ? 2560 * 320 : 640 * 2560);
        model.read(location.offset, full.data(), location.bytes);
        if (single) return full;
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
    throw std::runtime_error("missing protected BF16 model weight: " + name);
}
float unpack(uint16_t value) {
    const uint32_t bits = uint32_t(value) << 16;
    float result; std::memcpy(&result, &bits, 4); return result;
}
void protected_pair(Model& model, Library& library, int layer, int rank, RadStream stream,
                    uint64_t& launches, uint64_t& rejects, uint64_t& compared, bool single = false) {
    const std::vector<int> experts = layer == 34 ? std::vector<int>{407, 496}
        : layer == 44 ? std::vector<int>{292, 350} : std::vector<int>{122, 143, 399, 445};
    const int count = int(experts.size()), end = count * 2;
    const int width = single ? 1280 : 640, inner_width = width / 2, output_pitch = width + 13;
    const auto* gate = registered(library, "mixed_bf16_gate_up_strict");
    const auto* down = registered(library, "mixed_bf16_down_strict");
    std::array<std::vector<std::vector<uint16_t>>, 2> weights;
    std::vector<std::unique_ptr<Buffer>> blobs;
    Buffer gate_table(RAD_I64, {count}, RAD_MEM_DEVICE), down_table(RAD_I64, {count}, RAD_MEM_DEVICE);
    for (int kind = 0; kind < 2; ++kind) for (int expert : experts) {
        weights[kind].push_back(protected_weight(model, layer, expert, kind != 0, rank, single));
        const auto& data = weights[kind].back();
        blobs.push_back(std::make_unique<Buffer>(RAD_BF16, std::initializer_list<int64_t>{int64_t(data.size())}, RAD_MEM_DEVICE));
        std::memcpy(blobs.back()->host.data, data.data(), data.size() * 2); blobs.back()->push(stream);
        auto& table = kind ? down_table : gate_table;
        table.data<uint64_t>()[weights[kind].size() - 1] = reinterpret_cast<uintptr_t>(blobs.back()->device.data);
    }
    gate_table.push(stream); down_table.push(stream);
    for (int tokens : {1, 3, 63, 64}) {
        if (single && tokens > 3) continue;
        const int rows = tokens * 10;
        Buffer input(RAD_BF16, {tokens + 2, 2560}, RAD_MEM_DEVICE, 2573);
        Buffer sorted(RAD_I32, {rows}, RAD_MEM_DEVICE), offsets(RAD_I32, {count + 1}, RAD_MEM_DEVICE);
        Buffer output(RAD_BF16, {rows + 2, width}, RAD_MEM_DEVICE, output_pitch);
        Buffer result(RAD_BF16, {rows + 2, 2560}, RAD_MEM_DEVICE, 2577);
        for (int index = 0; index <= count; ++index) offsets.data<int32_t>()[index] = index * 2;
        offsets.push(stream);
        std::vector<uint8_t> previous;
        for (int step = 0; step < 2; ++step) {
            for (Buffer* buffer : {&input, &output, &result}) std::fill(buffer->expected.begin(), buffer->expected.end(), 0x5a);
            for (int row = 0; row < tokens; ++row) for (int column = 0; column < 2560; ++column)
                input.data<uint16_t>()[row * 2573 + column] = pack(float(int((column * 17 + row * 31 + step * 29) % 103) - 51) / 64.0f);
            for (int row = 0; row < rows; ++row) sorted.data<int32_t>()[row] = row < end ? (row * 7 + step * 3) % rows : -1;
            sorted.data<int32_t>()[1] = -1;
            for (Buffer* buffer : {&input, &sorted, &output, &result}) buffer->push(stream);
            auto view = [&](Buffer& buffer, int columns, int inner) {
                auto tensor = buffer.device; tensor.dtype = RAD_BF16; tensor.rank = 3;
                tensor.shape[0] = count; tensor.shape[1] = columns; tensor.shape[2] = inner;
                tensor.stride[0] = 0; tensor.stride[1] = inner; tensor.stride[2] = 1; return tensor;
            };
            std::vector<RadParam> gate_params{RAD_INT("M", 64), RAD_INT("N", width), RAD_INT("K", 2560),
                RAD_INT("n_expert", count), RAD_INT("top_k", 10), RAD_STR("dtype", "bf16"), RAD_STR("a_order", "token")};
            std::array<RadTensor, 5> tensors{live(input, tokens), view(gate_table, width, 2560), sorted.device, offsets.device, live(output, rows)};
            RadArgs args{}; args.t = tensors.data(); args.n_t = tensors.size(); args.p = gate_params.data(); args.n_p = gate_params.size(); args.rank = rank; args.world_size = single ? 1 : 2;
            ensure_equal(gate->launch(&args, stream), RAD_OK); ++launches;
            for (int row = 0; row < end; ++row) for (int column = 0; column < width; ++column) {
                float sum = 0;
                const int slot = sorted.data<int32_t>()[row];
                if (slot >= 0) for (int inner = 0; inner < 2560; ++inner)
                    sum = std::fma(unpack(input.data<uint16_t>()[(slot / 10) * 2573 + inner]),
                        unpack(weights[0][row / 2][column * 2560 + inner]), sum);
                output.data<uint16_t>()[row * output_pitch + column] = pack(sum);
            }
            ensure(output.read(stream) == output.expected);
            auto down_input = live(output, rows); down_input.shape[1] = inner_width;
            std::vector<RadParam> down_params{RAD_INT("M", 64), RAD_INT("N", 2560), RAD_INT("K", inner_width),
                RAD_INT("n_expert", count), RAD_INT("top_k", 10), RAD_STR("dtype", "bf16"), RAD_STR("a_order", "sorted")};
            auto down_tensors = tensors; down_tensors[0] = down_input; down_tensors[1] = view(down_table, 2560, inner_width); down_tensors[4] = live(result, rows);
            RadArgs down_args = args; down_args.t = down_tensors.data(); down_args.p = down_params.data(); down_args.n_p = down_params.size();
            ensure_equal(down->launch(&down_args, stream), RAD_OK); ++launches;
            for (int row = 0; row < end; ++row) for (int column = 0; column < 2560; ++column) {
                float sum = 0;
                if (sorted.data<int32_t>()[row] >= 0) for (int inner = 0; inner < inner_width; ++inner)
                    sum = std::fma(unpack(output.data<uint16_t>()[row * output_pitch + inner]),
                        unpack(weights[1][row / 2][column * inner_width + inner]), sum);
                result.data<uint16_t>()[row * 2577 + column] = pack(sum);
            }
            const auto actual = result.read(stream); ensure(actual == result.expected);
            if (step) ensure(actual != previous);
            previous = actual; compared += uint64_t(rows) * (width + 2560);
            for (bool is_down : {false, true}) {
                auto& call = is_down ? down_args : args; const auto* row = is_down ? down : gate;
                auto reject = [&](int operand, RadTensor invalid, int code) {
                    auto* operands = const_cast<RadTensor*>(call.t); const auto saved = operands[operand]; operands[operand] = invalid;
                    ensure_equal(row->launch(&call, stream), code); ++rejects; operands[operand] = saved;
                };
                auto invalid = call.t[4]; invalid.shape[0]++; reject(4, invalid, RAD_E_SHAPE);
                invalid = call.t[4]; invalid.shape[0] = 0; reject(4, invalid, RAD_E_SHAPE);
                invalid = call.t[4]; invalid.shape[0] = 650; reject(4, invalid, RAD_E_SHAPE);
                invalid = call.t[1]; invalid.dtype++; reject(1, invalid, RAD_E_SHAPE);
                invalid = call.t[1]; invalid.shape[0]++; reject(1, invalid, RAD_E_SHAPE);
                invalid = call.t[1]; invalid.stride[1]++; reject(1, invalid, RAD_E_STRIDE);
                invalid = call.t[0]; invalid.shape[0]++; reject(0, invalid, RAD_E_SHAPE);
                auto* params = const_cast<RadParam*>(call.p); params[0].ival = 2049;
                ensure_equal(row->launch(&call, stream), RAD_E_SHAPE); ++rejects; params[0].ival = 64;
            }
            ensure(input.read(stream) == input.expected); ensure(output.read(stream) == output.expected);
            ensure(sorted.read(stream) == sorted.expected); ensure(offsets.read(stream) == offsets.expected);
            ensure(result.read(stream) == result.expected);
        }
        std::printf("PASS MOE_PROTECTED_AQL_CASE layer=%d rank=%d experts=%d M=%d band=64 inputs=2 gate=%d/2560 down=2560/%d pitched=1 guards=1\n", layer, rank, count, tokens, width, inner_width);
        std::fflush(stdout);
    }
    for (auto& blob : blobs) ensure(blob->read(stream) == blob->expected);
    ensure(gate_table.read(stream) == gate_table.expected); ensure(down_table.read(stream) == down_table.expected);
}
}

TEST(r3d_moe_protected_actual_aql_live_band) {
    CHECK(true);
    RadDeviceProps props{}; moe_boundary::ensure_equal(rad_dev_props(0, &props), RAD_OK);
    if (props.is_host_backend) {
        moe_boundary::ensure(!std::getenv("RAD_R3D_REQUIRE_DEVICE"));
        SKIP("explicit gfx1100 card run required");
    }
    moe_boundary::ensure_equal(std::strcmp(props.arch, "gfx1100"), 0);
    moe_boundary::ensure_equal(rad_dev_set(0), RAD_OK);
    Library library("libr3d"); Queue first(0), second(1);
    Model model("/home/zihaomu/bigssd2/radiance_workspace/models/qwen3.8-next-flash-iqr-v4.rad");
    uint64_t launches = 0, rejects = 0, compared = 0;
    for (int layer : {34, 44, 47}) for (int rank : {0, 1})
        moe_boundary::protected_pair(model, library, layer, rank, rank ? second.stream : first.stream, launches, rejects, compared);
    std::printf("PASS MOE_PROTECTED_AQL fixtures=24 launches=%llu rejects=%llu exact_values=%llu queues=2 cpu_fallback=0\n",
        (unsigned long long)launches, (unsigned long long)rejects, (unsigned long long)compared);
}
