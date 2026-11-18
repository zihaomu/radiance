#include <numeric>

namespace gdn_decode_actual_aql {
const RadKernelInfo* named(Library& library, const char* name) {
    auto count = reinterpret_cast<int (*)()>(dlsym(library.handle, "rad_kernel_count"));
    auto at = reinterpret_cast<const RadKernelInfo* (*)(int)>(dlsym(library.handle, "rad_kernel_at"));
    if (!count || !at) throw std::runtime_error("missing GDN plugin enumeration");
    for (int index = 0; index < count(); ++index)
        if (!std::strcmp(at(index)->name, name)) return at(index);
    throw std::runtime_error(name);
}
RadParam real_param(const char* key, double value) {
    RadParam result{}; result.key = key; result.kind = RAD_P_F64; result.dval = value;
    return result;
}
RadParam string_param(const char* key, const char* value) {
    RadParam result{}; result.key = key; result.kind = RAD_P_STR; result.sval = value;
    return result;
}
void equal(const std::vector<uint8_t>& expected, const std::vector<uint8_t>& actual,
           const char* stage, int operand) {
    if (expected != actual) {
        size_t offset = 0;
        while (offset < expected.size() && expected[offset] == actual[offset]) ++offset;
        std::printf("GDN_DECODE_DIFF stage=%s operand=%d byte=%zu expected=%u actual=%u\n",
            stage, operand, offset, unsigned(expected[offset]), unsigned(actual[offset]));
        throw std::runtime_error("GDN decode byte-exact comparison failed");
    }
}
}

TEST(r3d_gdn_decode_actual_aql_persistent_fused) {
    using namespace gdn_decode_actual_aql;
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
    const auto* fused = named(native, "model_gdn_conv_recurrent_had_strict");
    const auto* prep = named(native, "model_gdn_conv_prep_strict");
    const auto* solve = named(native, "model_gdn_kkt_strict");
    const auto* scan = named(native, "model_gdn_chunk_strict");
    const auto* fused_ref = reference.ngram("gdn_conv_recurrent_update");
    const auto* prep_ref = reference.ngram("gdn_conv_prep");
    const auto* solve_ref = reference.ngram("gdn_kkt_solve");
    const auto* scan_ref = reference.ngram("gdn_chunk_scan");
    const auto* quant_ref = reference.ngram("had_quant_act_i8g");
    REQUIRE_EQ(fused->domain, RAD_DOMAIN_DEVICE);
    REQUIRE(fused->scratch);
    Queue first(0), second(1);
    REQUIRE(first.stream && second.stream && first.stream != second.stream);
    uint64_t fixtures = 0, prefills = 0, decodes = 0, rejects = 0, compared = 0, guard_checks = 0;
    uint64_t device_guards = 0;
    const std::vector<std::vector<int>> batches{{1}, {1, 1}, {1, 1, 1},
        {1, 1, 1, 1, 1, 1, 1, 1}, {0, 1}, {0, 1, 0, 1, 0, 1, 0, 1},
        {2, 1}, {0, 2, 1, 0, 3, 1, 2, 1}};
    for (const auto& lengths : batches)
        for (int rank_two : {0, 1}) {
            const int sequences = int(lengths.size());
            const int width = *std::max_element(lengths.begin(), lengths.end());
            if (!rank_two && width != 1) continue;
            const int rows = std::accumulate(lengths.begin(), lengths.end(), 0);
            const int warm_rows = sequences == 1 ? 63 : sequences * 3;
            const int capacity = std::max(rows, warm_rows), slots = sequences * width;
            std::vector<std::unique_ptr<Buffer>> buffers;
            auto add = [&](int type, std::initializer_list<int64_t> shape, int pitch = 0) {
                buffers.push_back(std::make_unique<Buffer>(type, shape, RAD_MEM_DEVICE, pitch));
            };
            add(RAD_BF16, {capacity, 8192});
            add(RAD_BF16, {5120, 4});
            add(RAD_BF16, {sequences, 5120, width + 2});
            add(RAD_I32, {sequences});
            add(RAD_BF16, {capacity, 48});
            add(RAD_F32, {24}); add(RAD_F32, {24});
            add(RAD_F32, {slots, 24, 128, 128});
            if (rank_two) add(RAD_I32, {sequences, width});
            else add(RAD_I32, {sequences});
            add(RAD_I32, {sequences}); add(RAD_I32, {sequences + 1});
            add(RAD_F32, {128});
            add(RAD_BF16, {capacity, 24, 128});
            add(RAD_I8, {rows, 3072}); add(RAD_F32, {rows, 24});
            add(RAD_BF16, {sequences, 5120, width + 2});
            add(RAD_F32, {slots, 24, 128, 128});
            add(RAD_BF16, {capacity, 24, 128});
            add(RAD_I8, {rows, 3072}); add(RAD_F32, {rows, 24});
            add(RAD_BF16, {capacity, 8, 128}); add(RAD_BF16, {capacity, 8, 128});
            add(RAD_BF16, {capacity, 24, 128});
            add(RAD_F32, {capacity, 24}); add(RAD_F32, {capacity, 24});
            add(RAD_BF16, {capacity, 24, 64});
            add(RAD_I32, {sequences + 1}); add(RAD_I32, {sequences});
            Buffer scratch(RAD_I32, {20971520 / 4}, RAD_MEM_DEVICE);
            auto fill = [&](int operand, int seed) {
                auto& buffer = *buffers[operand];
                const int64_t elements = (buffer.expected.size() - 128) / 2;
                for (int64_t element = 0; element < elements; ++element)
                    buffer.data<uint16_t>()[element] = rad_f32_to_bf16(
                        float((element * 13 + seed * 17) % 37 - 18) / 64.0f);
            };
            fill(0, 3); fill(1, 5); fill(4, 7);
            for (int head = 0; head < 24; ++head) {
                buffers[5]->data<float>()[head] = float(head % 7 - 3) / 32.0f;
                buffers[6]->data<float>()[head] = float(head % 5 - 2) / 64.0f;
            }
            for (int column = 0; column < 128; ++column)
                buffers[11]->data<float>()[column] = 0.75f + float(column % 11) / 32.0f;
            for (int operand : {2, 7, 15, 16})
                std::fill(buffers[operand]->expected.begin() + 64, buffers[operand]->expected.end() - 64, 0);
            int total = 0;
            for (int sequence = 0; sequence < sequences; ++sequence) {
                buffers[3]->data<int32_t>()[sequence] = sequences - sequence - 1;
                buffers[9]->data<int32_t>()[sequence] = 1;
                buffers[10]->data<int32_t>()[sequence] = total;
                total += lengths[sequence];
                buffers[26]->data<int32_t>()[sequence] = sequences == 1 ? 0 : sequence * 3;
                buffers[27]->data<int32_t>()[sequence] = 0;
                for (int column = 0; column < width; ++column)
                    buffers[8]->data<int32_t>()[sequence * buffers[8]->host.stride[0] + column] =
                        (sequences - sequence - 1) * width + column;
            }
            buffers[10]->data<int32_t>()[sequences] = rows;
            buffers[26]->data<int32_t>()[sequences] = warm_rows;
            std::vector<RadParam> params{integer("q_len", 8), integer("M", 64),
                integer("head_k", 128), integer("head_v", 128), integer("n_head_v", 24),
                integer("conv_width", 4), integer("chunk", 64), integer("group", 128), integer("n", 3072),
                real_param("scale", 0.08838834765), real_param("eps", 1e-6),
                string_param("act", "sigmoid"), string_param("state_form", "per_candidate")};
            auto views = [&](bool device, int tokens) {
                std::vector<RadTensor> result;
                for (auto& buffer : buffers) result.push_back(device ? buffer->device : buffer->host);
                for (int operand : {0, 4, 12, 17, 20, 21, 22, 23, 24, 25}) result[operand].shape[0] = tokens;
                return result;
            };
            auto ab = [&](RadTensor tensor, bool second_half) {
                tensor.shape[1] = 24;
                if (second_half) tensor.data = static_cast<uint16_t*>(tensor.data) + 24;
                return tensor;
            };
            auto qkv = [&](RadTensor tensor) { tensor.shape[1] = 5120; return tensor; };
            auto execute = [&](const RadKernelInfo* row, std::vector<RadTensor> tensors, RadStream stream) {
                RadArgs args{}; args.t = tensors.data(); args.n_t = int(tensors.size());
                args.p = params.data(); args.n_p = int(params.size());
                REQUIRE_EQ(row->launch(&args, stream), RAD_OK);
            };
            for (auto& buffer : buffers) buffer->push(first.stream);
            scratch.push(first.stream);
            REQUIRE_EQ(rad_stream_sync(first.stream), RAD_OK);
            for (bool device : {false, true}) {
                auto tensor = views(device, warm_rows);
                RadStream stream = device ? first.stream : nullptr;
                execute(device ? prep : prep_ref, {qkv(tensor[0]), tensor[1], {}, tensor[5], tensor[6],
                    tensor[2], tensor[26], ab(tensor[4], false), ab(tensor[4], true), tensor[3], tensor[27],
                    tensor[20], tensor[21], tensor[22], tensor[23], tensor[24]}, stream);
                execute(device ? solve : solve_ref,
                    {tensor[21], tensor[24], tensor[23], tensor[26], tensor[25]}, stream);
                execute(device ? scan : scan_ref, {tensor[20], tensor[21], tensor[22], tensor[25],
                    tensor[23], tensor[24], {}, tensor[26], tensor[12], tensor[7], tensor[8]}, stream);
            }
            REQUIRE_EQ(rad_stream_sync(first.stream), RAD_OK);
            ++prefills;
            for (auto& buffer : buffers) {
                equal(buffer->expected, buffer->read(first.stream), "prefill", int(&buffer - buffers.data()));
                compared += buffer->expected.size();
            }
            for (auto pair : {std::pair{2, 15}, std::pair{7, 16}}) {
                std::copy(buffers[pair.first]->expected.begin(), buffers[pair.first]->expected.end(),
                    buffers[pair.second]->expected.begin());
                REQUIRE_EQ(rad_memcpy_async(buffers[pair.second]->allocation, buffers[pair.first]->allocation,
                    buffers[pair.first]->expected.size(), first.stream), RAD_OK);
            }
            REQUIRE_EQ(rad_stream_sync(first.stream), RAD_OK);
            std::vector<uint8_t> previous_output;
            for (int step = 0; step < 4; ++step) {
                RadStream stream = step % 2 ? first.stream : second.stream;
                REQUIRE_EQ(rad_stream_device_access(stream, 0, nullptr), RAD_OK);
                fill(0, 19 + step * 23); fill(4, 29 + step * 31);
                for (int operand : {0, 4}) buffers[operand]->push(stream);
                for (int operand : {12, 13, 14, 17, 18, 19}) {
                    std::fill(buffers[operand]->expected.begin(), buffers[operand]->expected.end(), 0x5a);
                    buffers[operand]->push(stream);
                }
                REQUIRE_EQ(rad_stream_sync(stream), RAD_OK);
                auto operands = [&](bool device, bool explicit_one) {
                    const auto tensor = views(device, rows);
                    RadTensor gate = tensor[0]; gate.shape[1] = 3072;
                    gate.data = static_cast<uint16_t*>(gate.data) + 5120;
                    RadTensor absent = tensor[9]; absent.data = nullptr;
                    return std::vector<RadTensor>{qkv(tensor[0]), tensor[1], {}, tensor[explicit_one ? 15 : 2],
                        tensor[3], ab(tensor[4], false), ab(tensor[4], true), tensor[5], tensor[6],
                        tensor[explicit_one ? 16 : 7], tensor[8], explicit_one ? tensor[9] : absent,
                        tensor[10], gate, tensor[11], tensor[explicit_one ? 17 : 12],
                        tensor[explicit_one ? 18 : 13], tensor[explicit_one ? 19 : 14]};
                };
                for (bool explicit_one : {false, true}) {
                    auto host = operands(false, explicit_one), device = operands(true, explicit_one);
                    RadTensor quant_input = host[15], quant_codes = host[16], quant_scales = host[17];
                    host[16] = {}; host[17] = {};
                    execute(fused_ref, host, nullptr);
                    quant_input.rank = 2; quant_input.shape[1] = 3072; quant_input.stride[1] = 1;
                    execute(quant_ref, {quant_input, quant_codes, quant_scales}, nullptr);
                    RadArgs args{}; args.t = device.data(); args.n_t = 18;
                    args.p = params.data(); args.n_p = int(params.size());
                    args.scratch = scratch.device.data; args.scratch_bytes = 20971520;
                    REQUIRE_EQ(fused->launch(&args, stream), RAD_OK);
                    ++decodes;
                    REQUIRE_EQ(rad_stream_sync(stream), RAD_OK);
                    if (step == 0) {
                        const auto valid = device;
                        auto reject = [&](int expected) {
                            CHECK_EQ(fused->launch(&args, stream), expected); ++rejects; device = valid;
                        };
                        device[3].shape[2] = 2; reject(RAD_E_SHAPE);
                        device[3].stride[2] = 2; reject(RAD_E_SHAPE);
                        device[3].shape[1] = 5119; reject(RAD_E_SHAPE);
                        device[10].rank = 0; reject(RAD_E_SHAPE);
                        device[10].shape[rank_two ? 1 : 0] = 0; reject(RAD_E_SHAPE);
                        device[10].dtype = RAD_F32; reject(RAD_E_SHAPE);
                        device[11] = valid[4]; device[11].dtype = RAD_F32; reject(RAD_E_SHAPE);
                        device[11] = valid[4]; device[11].rank = 2; reject(RAD_E_SHAPE);
                        device[11] = valid[4]; device[11].shape[0]++; reject(RAD_E_SHAPE);
                        device[11] = valid[4]; device[11].stride[0] = 2; reject(RAD_E_SHAPE);
                        device[9].stride[2] = 129; reject(RAD_E_SHAPE);
                        device[0].stride[0] = 5119; reject(RAD_E_SHAPE);
                        device[13].shape[1] = 3071; reject(RAD_E_SHAPE);
                        device[15].shape[0]++; reject(RAD_E_SHAPE);
                        params[0].ival = 0; reject(RAD_E_SHAPE);
                        params[0].ival = 9; reject(RAD_E_SHAPE); params[0].ival = 8;
                        args.scratch = nullptr; reject(RAD_E_INVAL); args.scratch = scratch.device.data;
                        args.scratch_bytes = fused->scratch(&args) - 1;
                        reject(RAD_E_INVAL); args.scratch_bytes = 20971520;
                        args.scratch = static_cast<uint8_t*>(scratch.device.data) + 8;
                        reject(RAD_E_INVAL); args.scratch = scratch.device.data;
                    }
                }
                for (auto pair : {std::pair{2, 15}, std::pair{7, 16}, std::pair{12, 17},
                                  std::pair{13, 18}, std::pair{14, 19}})
                    equal(buffers[pair.first]->expected, buffers[pair.second]->expected, "absent=one", pair.first);
                for (int operand = 0; operand < int(buffers.size()); ++operand) {
                    equal(buffers[operand]->expected, buffers[operand]->read(stream), "decode", operand);
                    compared += buffers[operand]->expected.size();
                }
                CHECK(previous_output != buffers[12]->expected);
                previous_output = buffers[12]->expected;
                const auto scratch_bytes = scratch.read(stream);
                const size_t live_end = 64 + size_t(rows) * 5120 * 2;
                REQUIRE(std::all_of(scratch_bytes.begin(), scratch_bytes.begin() + 64,
                    [](uint8_t value) { return value == 0x5a; }));
                REQUIRE(std::all_of(scratch_bytes.begin() + live_end, scratch_bytes.end(),
                    [](uint8_t value) { return value == 0x5a; }));
                compared += scratch_bytes.size() - live_end + 64;
                guard_checks += 2;
                if (step == 3) {
                    auto guarded = operands(true, false);
                    RadArgs args{}; args.t = guarded.data(); args.n_t = 18;
                    args.p = params.data(); args.n_p = int(params.size());
                    args.scratch = scratch.device.data; args.scratch_bytes = 20971520;
                    auto check_guard = [&] {
                        REQUIRE_EQ(fused->launch(&args, stream), RAD_OK);
                        REQUIRE_EQ(rad_stream_sync(stream), RAD_OK);
                        for (int operand : {2, 7, 12}) {
                            equal(buffers[operand]->expected, buffers[operand]->read(stream), "device-guard", operand);
                            compared += buffers[operand]->expected.size();
                        }
                        ++device_guards;
                    };
                    const auto saved_cu = buffers[10]->expected;
                    for (int sequence = 0; sequence <= sequences; ++sequence)
                        buffers[10]->data<int32_t>()[sequence] = sequence ? rows : 0;
                    if (rows <= width)
                        for (int sequence = 1; sequence <= sequences; ++sequence)
                            buffers[10]->data<int32_t>()[sequence] = rows + 1;
                    buffers[10]->push(stream); check_guard();
                    std::copy(saved_cu.begin(), saved_cu.end(), buffers[10]->expected.begin());
                    buffers[10]->push(stream);
                    const auto saved_conv = buffers[3]->expected, saved_index = buffers[8]->expected;
                    for (int sequence = 0; sequence < sequences; ++sequence) {
                        buffers[3]->data<int32_t>()[sequence] = sequences;
                        for (int column = 0; column < width; ++column)
                            buffers[8]->data<int32_t>()[sequence * buffers[8]->host.stride[0] + column] = slots;
                    }
                    buffers[3]->push(stream); buffers[8]->push(stream); check_guard();
                    std::copy(saved_conv.begin(), saved_conv.end(), buffers[3]->expected.begin());
                    std::copy(saved_index.begin(), saved_index.end(), buffers[8]->expected.begin());
                    buffers[3]->push(stream); buffers[8]->push(stream);
                    guarded[11] = buffers[9]->device;
                    for (int accepted : {0, 17}) {
                        std::fill_n(buffers[9]->data<int32_t>(), sequences, accepted);
                        buffers[9]->push(stream); check_guard();
                    }
                }
            }
            ++fixtures;
            std::printf("PASS GDN_DECODE_FIXTURE rows=%d seq=%d width=%d index_rank=%d prefill=%d steps=4 band=8 absent=one\n",
                rows, sequences, width, rank_two ? 2 : 1, warm_rows);
            std::fflush(stdout);
        }
    std::printf("PASS GDN_DECODE_DEVICE_GUARDS calls=%llu invalid_span_slot_accepted=0,17 persistent_writes=0\n", (unsigned long long)device_guards);
    std::printf("PASS GDN_DECODE_AQL fixtures=%llu chunk_prefills=%llu decode_calls=%llu rejects=%llu scratch_guards=%llu exact_bytes=%llu queues=2 tolerance=0\n",
        (unsigned long long)fixtures, (unsigned long long)prefills, (unsigned long long)decodes,
        (unsigned long long)rejects, (unsigned long long)guard_checks, (unsigned long long)compared);
}
