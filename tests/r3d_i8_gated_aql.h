#include "format/radfile.h"
#include "format/share.h"
#include "build/startup.h"
#include "sample/sampler.h"

TEST(r3d_i8_gated_normal_device_coverage) {
    using namespace rad;
    if (!joined_head_test::device_available()) SKIP("explicit gfx1100 card run required");
    RadFile file;
    REQUIRE_EQ(file.open("/home/zihaomu/bigssd2/radiance_workspace/models/qwen3.8-next-flash-iqr-v4.rad"), RAD_OK);
    LoadedPlugins plugins;
    REQUIRE_EQ(rad_tools_load_plugins(home(), {"libr3d", "libr4d", "r3dh_ple_host"},
        file.str(file.header().arch_id), file.str(file.header().quant), &plugins), RAD_OK);
    RadModelMeta metadata{};
    metadata.arch_id = file.str(file.header().arch_id);
    metadata.name = file.str(file.header().model_name); metadata.quant = file.str(file.header().quant);
    auto integer = [&](const char* key) { int64_t value = 0; file.get_i(key, &value); return value; };
    metadata.n_layers = integer("n_layers"); metadata.n_embd = integer("n_embd");
    metadata.n_head = integer("n_head"); metadata.n_head_kv = integer("n_head_kv");
    metadata.head_dim = integer("head_dim"); metadata.n_ff = integer("n_ff");
    metadata.n_vocab = integer("n_vocab"); metadata.n_ctx_train = integer("n_ctx_train");
    metadata.n_expert = integer("n_expert"); metadata.n_expert_used = integer("n_expert_used");
    metadata.n_expert_shared = integer("n_expert_shared");
    double epsilon = 1e-6, theta = 10000;
    file.get_f("rms_eps", &epsilon); file.get_f("rope_theta", &theta);
    metadata.rms_eps = float(epsilon); metadata.rope_theta = float(theta); metadata.rope_scale = 1;
    std::vector<std::string> keys, values;
    std::vector<const char*> key_pointers, value_pointers;
    for (int64_t index = 0; index < file.meta_count(); ++index) {
        const auto& item = file.meta_at(index);
        if (item.type != RAD_P_STR) continue;
        keys.push_back(file.str(item.key)); values.push_back(file.str(item.v.s));
    }
    for (auto& key : keys) key_pointers.push_back(key.c_str());
    for (auto& value : values) value_pointers.push_back(value.c_str());
    metadata.n_kv = int(keys.size()); metadata.kv_key = key_pointers.data(); metadata.kv_val = value_pointers.data();
    Registry& registry = rad_tools_registry();
    registry.set_device_fact(RAD_FACT_P2P, 1);
    const Plugin* architecture = registry.plugin(plugins.arch_plugin);
    REQUIRE(architecture && architecture->arch_declare);
    for (int mtp : {0, 1, 3}) for (int rank : {0, 1}) {
        RadBuildCtx context{};
        context.rank = rank; context.world_size = 2; context.max_tok = mtp == 3 ? 2048 : 64;
        context.max_seqs = mtp == 3 ? 8 : 1; context.max_ctx = mtp == 3 ? 200000 : 4096;
        context.max_spec = mtp; context.max_enc_patches = mtp == 3 ? 256 : 0;
        context.kv_dtype = RAD_I8; context.scope = "";
        Builder builder(registry, metadata, context);
        builder.set_instances(false); builder.set_sources(rad_container_sources(file));
        REQUIRE_EQ(architecture->arch_declare(&builder, &metadata, &context), RAD_OK);
        const auto logits_handle = builder.program().logits_buf;
        REQUIRE(logits_handle != RAD_NULL_HANDLE);
        REQUIRE(size_t(logits_handle) < builder.program().buffers.size());
        const auto& logits_shape = builder.program().buffers[logits_handle].decl;
        REQUIRE_EQ(logits_shape.rank, 2);
        REQUIRE_EQ(logits_shape.shape[0], context.max_seqs * (mtp + 1));
        REQUIRE_EQ(logits_shape.shape[1], 124160);
        Sampler serving_sampler;
        SamplerConfig sampler_config;
        sampler_config.n_vocab = metadata.n_vocab;
        sampler_config.n_vocab_local = logits_shape.shape[1];
        sampler_config.vocab_off = rank * sampler_config.n_vocab_local;
        sampler_config.world_size = context.world_size; sampler_config.rank = rank;
        sampler_config.max_seqs = context.max_seqs; sampler_config.max_spec = context.max_spec;
        sampler_config.max_tok = context.max_tok;
        REQUIRE_EQ(serving_sampler.declare(&builder, sampler_config), RAD_OK);
        REQUIRE_EQ(builder.finish(), RAD_OK); REQUIRE(builder.errors().empty());
        const auto& program = builder.program();
        REQUIRE_EQ(rad_gate_reference_kernels(program, false, "strict normal MTP coverage test"), RAD_OK);
        uint64_t device_bands = 0, host_bands = 0, retired = 0, missing = 0, gated = 0;
        uint64_t strict_coarse = 0, strict_rerank = 0, strict_merge = 0;
        bool dense_reported = false;
        for (size_t index = 1; index < program.ops.size(); ++index) {
            const auto& operation = program.ops[index];
            if (operation.optional_retired) { ++retired; continue; }
            REQUIRE(!operation.bands.empty());
            for (const auto& band : operation.bands) {
                const auto* row = band.dom[RAD_DOMAIN_DEVICE].row;
                if (row && row->info && row->info->domain == RAD_DOMAIN_DEVICE && !row->reference) {
                    ++device_bands;
                    if (!dense_reported && std::getenv("RAD_TRACE_DENSE_SELECTION")
                        && operation.op == "gemm_nt_q" && operation.base.str().find("8192") != std::string::npos) {
                        std::printf("DENSE_SELECTED mtp=%d rank=%d kernel=%s geometry=%s\n",
                            mtp, rank, row->info->name, operation.base.str().c_str());
                        dense_reported = true;
                    }
                    if (operation.op == "gemm_nt_q" && operation.base.str().find("w2a8") != std::string::npos) {
                        REQUIRE_EQ(std::strcmp(row->info->name, "gemm_nt_q_w2a8_head_strict"), 0);
                        ++strict_coarse;
                    }
                    if (operation.op == "logit_rerank") {
                        REQUIRE_EQ(std::strcmp(row->info->name, "r3d_logit_rerank_joined_strict"), 0);
                        ++strict_rerank;
                    }
                    if (operation.op == "sample_merge_topk") {
                        REQUIRE_EQ(std::strcmp(row->info->name, "r3dh_sample_merge_topk_f32"), 0);
                        ++strict_merge;
                    }
                    if (!std::strcmp(row->info->name, "r3d_gemm_nt_q_gated_i8a8h_strict")) {
                        ++gated; REQUIRE_EQ(band.dom[RAD_DOMAIN_DEVICE].scratch_bytes, 24576);
                        REQUIRE_EQ(std::strcmp(row->info->op, "gemm_nt_q_gated"), 0);
                    }
                    continue;
                }
                const bool allowed = operation.op == "ngram_ids" || operation.op == "embed_lookup_q" || operation.op == "ngram_hint";
                const auto* host = band.dom[RAD_DOMAIN_HOST].row;
                if (allowed && host && !host->reference) { ++host_bands; continue; }
                ++missing;
                std::printf("I8_GATED_COVERAGE_HOLE mtp=%d rank=%d op=%zu name=%s geometry=%s reason=%s\n",
                    mtp, rank, index, operation.op.c_str(), operation.base.str().c_str(), band.miss[RAD_DOMAIN_DEVICE].c_str());
            }
        }
        std::printf("I8_GATED_NORMAL_DECLARE mtp=%d rank=%d finish=0 missing=%llu device_bands=%llu allowed_host_bands=%llu retired_ops=%llu gated_bands=%llu audit=0 instances=0 p2p=declaration_fact_only\n",
            mtp, rank, (unsigned long long)missing, (unsigned long long)device_bands, (unsigned long long)host_bands,
            (unsigned long long)retired, (unsigned long long)gated);
        REQUIRE_EQ(missing, 0u); REQUIRE(device_bands > 0);
        if (mtp) REQUIRE(gated > 0);
        if (mtp == 3) {
            REQUIRE(strict_coarse > 0); REQUIRE(strict_rerank > 0);
            REQUIRE(strict_merge > 0);
            Geometry sampler_geometry;
            sampler_geometry.set_i("M", 32); sampler_geometry.set_i("n_cand", 1024);
            const auto* sampler = registry.select("sample_merge_topk", sampler_geometry, RAD_DOMAIN_DEVICE);
            REQUIRE(sampler && sampler->info && !sampler->reference);
            REQUIRE_EQ(std::strcmp(sampler->info->name, "r3dh_sample_merge_topk_f32"), 0);
            std::printf("SAMPLE_MERGE32_NORMAL_DECLARE rank=%d seq=8 spec=3 batch=2048 ctx=200000 vision=256 M=32 strict_merge=%llu holes=0 reference_gate=0 audit=0 instances=0\n",
                rank, (unsigned long long)strict_merge);
            std::printf("HEAD_CAPACITY_DECLARE rank=%d seq=8 spec=3 batch=2048 ctx=200000 vision=256 logit_capacity=32 strict_coarse=%llu strict_rerank=%llu holes=0 finish=0\n",
                rank, (unsigned long long)strict_coarse, (unsigned long long)strict_rerank);
        }
    }
}

TEST(r3d_i8_gated_model_contract) {
    Model model("/home/zihaomu/bigssd2/radiance_workspace/models/qwen3.8-next-flash-iqr-v4.rad");
    unsigned found = 0;
    for (const auto& entry : model.entries) {
        const auto name = model.name(entry);
        if (name.find("blk.48.") != 0 || name.find("shexp") == std::string::npos) continue;
        REQUIRE(entry.enc < model.encodings.size());
        char text[1024];
        rad_enc_format(&model.encodings[entry.enc], text, sizeof(text));
        std::printf("I8_GATED_MODEL name=%s N=%lld K=%lld encoding=%s\n", name.c_str(),
            (long long)entry.shape[0], (long long)entry.shape[1], text);
        ++found;
    }
    REQUIRE(found > 0);
}

namespace i8_gated_aql {
using namespace joined_head_test;
struct Weight {
    int columns;
    bool plain;
    RadEncoding encoding{};
    std::array<std::vector<uint8_t>, 5> raw;
    std::array<RadTensor, 5> planes{};
    std::array<std::vector<uint8_t>, 2> packed;
    std::array<RadLayout, 2> layout{};
    std::vector<int8_t> decoded;
    Weight(const RadKernelInfo* kernel, rad::RadFile& file, int rank, bool affine = false)
        : columns(rank ? 512 : 768), plain(affine) {
        if (plain) { load_affine(kernel, file, rank); return; }
        const auto* entry = file.find("blk.48.ffn_gate_up_shexp.weight"); REQUIRE(entry);
        REQUIRE_EQ(entry->shape[0], 1280); REQUIRE_EQ(entry->shape[1], 2560);
        encoding = file.encoding(*entry);
        REQUIRE_EQ(encoding.n_planes, 5); REQUIRE_EQ(encoding.plane[0].dtype, uint32_t(RAD_U3));
        REQUIRE_EQ(encoding.plane[1].dtype, uint32_t(RAD_F32));
        const std::vector<RadParam> params{RAD_INT("M", 16), RAD_INT("N", columns), RAD_INT("K", 2560),
            RAD_INT("group", 128), RAD_STR("dtype", "i8a8h"), RAD_STR("act", "silu")};
        const std::array<std::vector<int>, 2> selected{{{0, 2, 3}, {1, 4}}};
        for (int bundle = 0; bundle < 2; ++bundle) {
            rad::WeightInfo info;
            info.decl.rank = 2; info.decl.shard = RAD_SHARD_ROW;
            info.decl.dtype = bundle ? RAD_F32 : RAD_U3;
            info.decl.shape[0] = columns; info.decl.shape[1] = bundle ? 20 : 2560;
            info.decl.n_row_parts = 2;
            info.decl.row_parts[0] = columns / 2; info.decl.row_parts[1] = columns / 2;
            info.shard_lo = rank ? 384 : 0; info.shard_hi = info.shard_lo + columns / 2;
            info.n_sel = selected[bundle].size();
            for (int index = 0; index < info.n_sel; ++index) {
                const int plane = selected[bundle][index]; info.sel[index] = plane;
                rad_enc_plane_dims(&encoding.plane[plane], columns, 2560, &info.sel_rows[index], &info.sel_cols[index]);
            }
            rad::PlaneShare shares[RAD_ENC_MAX_PLANES]; std::string why;
            const int shared = rad::weight_share(info, encoding, entry->rank, entry->shape, 2, rank, shares, &why);
            if (shared) std::printf("I8_GATED_SHARE_ERROR %s\n", why.c_str());
            REQUIRE_EQ(shared, RAD_OK);
            std::vector<RadTensor> views;
            for (int index = 0; index < info.n_sel; ++index) {
                const int plane = selected[bundle][index]; const auto& share = shares[index];
                raw[plane].resize(share.dense_bytes()); size_t cursor = 0;
                const auto* source = file.data(file.plane(*entry, plane));
                for (const auto& rect : share.rects) for (int64_t row = 0; row < rect.rows; ++row) {
                    REQUIRE(cursor + rect.bytes <= raw[plane].size());
                    std::memcpy(raw[plane].data() + cursor, source + (rect.row0 + row) * share.row_bytes + rect.byte0, rect.bytes);
                    cursor += rect.bytes;
                }
                REQUIRE_EQ(cursor, raw[plane].size());
                planes[plane] = matrix(raw[plane].data(), share.dtype, share.rows, share.cols);
                views.push_back(planes[plane]);
                std::printf("I8_GATED_PART rank=%d parts=2 span=%lld:%lld role=%s rows=%lld cols=%lld bytes=%zu rects=%zu\n",
                    rank, (long long)info.shard_lo, (long long)info.shard_hi, encoding.plane[plane].role,
                    (long long)share.rows, (long long)share.cols, cursor, share.rects.size());
            }
            REQUIRE_EQ(kernel->layout(params.data(), params.size(), bundle + 2, &encoding, selected[bundle].data(),
                views.data(), views.size(), &layout[bundle]), RAD_OK);
            auto malformed_views = views; ++malformed_views[0].shape[0]; RadLayout refused{};
            REQUIRE_EQ(kernel->layout(params.data(), params.size(), bundle + 2, &encoding, selected[bundle].data(),
                malformed_views.data(), malformed_views.size(), &refused), RAD_E_SHAPE);
            std::vector<uint8_t> untouched(size_t(layout[bundle].bytes) + 128, 0x5a), before = untouched;
            REQUIRE_EQ(kernel->relayout(params.data(), params.size(), bundle + 2, &encoding, selected[bundle].data(),
                malformed_views.data(), malformed_views.size(), untouched.data() + 64, layout[bundle].bytes), RAD_E_SHAPE);
            REQUIRE(untouched == before);
            auto unknown_encoding = encoding; unknown_encoding.plane[0].dtype = RAD_U2;
            REQUIRE_EQ(kernel->layout(params.data(), params.size(), bundle + 2, &unknown_encoding, selected[bundle].data(),
                views.data(), views.size(), &refused), RAD_E_DTYPE);
            packed[bundle].resize(layout[bundle].bytes);
            REQUIRE_EQ(kernel->relayout(params.data(), params.size(), bundle + 2, &encoding, selected[bundle].data(),
                views.data(), views.size(), packed[bundle].data(), packed[bundle].size()), RAD_OK);
            std::vector<std::vector<uint8_t>> inverse;
            for (const auto& view : views) inverse.emplace_back(size_t(view.shape[0] * view.shape[1] * rad_dtype_bits(view.dtype) + 7) / 8);
            auto inverse_views = views;
            for (size_t index = 0; index < views.size(); ++index) inverse_views[index].data = inverse[index].data();
            REQUIRE_EQ(kernel->unrelayout(params.data(), params.size(), bundle + 2, &encoding, selected[bundle].data(),
                packed[bundle].data(), packed[bundle].size(), inverse_views.data(), inverse_views.size()), RAD_OK);
            for (size_t index = 0; index < views.size(); ++index) REQUIRE(inverse[index] == raw[selected[bundle][index]]);
        }
        REQUIRE_EQ(layout[0].dtype, r3d_iqr::weight_dtype); REQUIRE_EQ(layout[1].dtype, r3d_iqr::scale_dtype);
        const auto* table = reinterpret_cast<const int8_t*>(raw[2].data());
        uint32_t coefficients[2]; std::memcpy(coefficients, raw[3].data(), sizeof(coefficients));
        for (unsigned state = 0; state < 65536; ++state)
            REQUIRE_EQ(int(table[state]), iqr_output_boundary::independent_mad(state, coefficients[0], coefficients[1]));
        const int run = int(encoding.plane[2].block[0]);
        decoded.resize(size_t(columns) * 2560);
        auto code = [&](int row, int column) {
            unsigned result = 0;
            for (unsigned digit = 0; digit < 3; ++digit) {
                const size_t bit = size_t(column) * 3 + digit;
                result |= ((raw[0][size_t(row) * 960 + bit / 8] >> (bit % 8)) & 1u) << digit;
            }
            return result;
        };
        for (int start = 0; start < columns; start += run) for (int column = 0; column < 2560; ++column) {
            unsigned state = 0;
            for (int previous = run - 6; previous < run; ++previous) state = ((state << 3) | code(start + previous, column)) % 65536;
            for (int row = start; row < start + run; ++row) {
                state = ((state << 3) | code(row, column)) % 65536;
                decoded[size_t(row) * 2560 + column] = table[state];
            }
        }
    }
    void load_affine(const RadKernelInfo* kernel, rad::RadFile& file, int rank) {
        Model model("/home/zihaomu/bigssd2/radiance_workspace/models/qwen3.8-next-flash-iqr-v4.rad");
        std::string source_name;
        for (const auto& entry : model.entries) {
            if (entry.rank != 2 || entry.shape[0] < columns || entry.shape[1] != 2560 || entry.enc >= model.encodings.size()) continue;
            const auto& candidate = model.encodings[entry.enc];
            const auto* code = rad_enc_plane(&candidate, "codes"), *scale = rad_enc_plane(&candidate, "scale");
            if (candidate.n_planes == 2 && rad_enc_is(&candidate, "affine") && !std::strcmp(candidate.transform, "fwht128")
                && code && scale && code->dtype == RAD_I8 && scale->dtype == RAD_BF16 && scale->block[0] == 1 && scale->block[1] == 128) {
                source_name = model.name(entry); break;
            }
        }
        REQUIRE(!source_name.empty());
        const auto* entry = file.find(source_name.c_str()); REQUIRE(entry); encoding = file.encoding(*entry);
        const int64_t start = rank ? entry->shape[0] - columns : 0;
        const std::vector<RadParam> params{RAD_INT("M", 16), RAD_INT("N", columns), RAD_INT("K", 2560),
            RAD_INT("group", 128), RAD_STR("dtype", "i8a8h"), RAD_STR("act", "silu")};
        for (int bundle = 0; bundle < 2; ++bundle) {
            const int selected = int(rad_enc_plane(&encoding, bundle ? "scale" : "codes") - encoding.plane);
            rad::WeightInfo info; info.decl.rank = 2; info.decl.shard = RAD_SHARD_ROW;
            info.decl.dtype = bundle ? RAD_BF16 : RAD_I8;
            info.decl.shape[0] = columns; info.decl.shape[1] = bundle ? 20 : 2560;
            info.shard_lo = start; info.shard_hi = start + columns; info.n_sel = 1; info.sel[0] = selected;
            rad_enc_plane_dims(&encoding.plane[selected], columns, 2560, &info.sel_rows[0], &info.sel_cols[0]);
            rad::PlaneShare shares[RAD_ENC_MAX_PLANES]; std::string why;
            REQUIRE_EQ(rad::weight_share(info, encoding, entry->rank, entry->shape, 2, rank, shares, &why), RAD_OK);
            const auto& share = shares[0]; raw[bundle].resize(share.dense_bytes()); size_t cursor = 0;
            const auto* source = file.data(file.plane(*entry, selected));
            for (const auto& rect : share.rects) for (int64_t row = 0; row < rect.rows; ++row) {
                REQUIRE(cursor + rect.bytes <= raw[bundle].size());
                std::memcpy(raw[bundle].data() + cursor, source + (rect.row0 + row) * share.row_bytes + rect.byte0, rect.bytes); cursor += rect.bytes;
            }
            REQUIRE_EQ(cursor, raw[bundle].size());
            planes[bundle] = matrix(raw[bundle].data(), share.dtype, share.rows, share.cols);
            REQUIRE_EQ(kernel->layout(params.data(), params.size(), bundle + 2, &encoding, &selected, &planes[bundle], 1, &layout[bundle]), RAD_OK);
            packed[bundle].resize(layout[bundle].bytes);
            REQUIRE_EQ(kernel->relayout(params.data(), params.size(), bundle + 2, &encoding, &selected, &planes[bundle], 1,
                packed[bundle].data(), packed[bundle].size()), RAD_OK);
            std::vector<uint8_t> inverse(raw[bundle].size()); RadTensor inverse_view = planes[bundle]; inverse_view.data = inverse.data();
            REQUIRE_EQ(kernel->unrelayout(params.data(), params.size(), bundle + 2, &encoding, &selected, packed[bundle].data(),
                packed[bundle].size(), &inverse_view, 1), RAD_OK);
            REQUIRE(inverse == raw[bundle]);
            std::printf("I8_GATED_AFFINE_ACTUAL name=%s rank=%d full_N=%lld selected_rows=%lld:%lld N=%d K=2560 role=%s bytes=%zu kind=actual_weight_subview_not_shared_ffn\n",
                source_name.c_str(), rank, (long long)entry->shape[0], (long long)start, (long long)(start + columns), columns,
                encoding.plane[selected].role, raw[bundle].size());
        }
        REQUIRE_EQ(layout[0].dtype, r3dh_i8::weight_dtype); REQUIRE_EQ(layout[1].dtype, r3dh_i8::scale_dtype);
        decoded.resize(raw[0].size()); std::memcpy(decoded.data(), raw[0].data(), decoded.size());
    }
    uint16_t dot(const int8_t* activation, const float* scales, int column) const {
        float total = 0;
        const auto* row_scales = reinterpret_cast<const float*>(raw[1].data());
        const auto* column_scales = reinterpret_cast<const float*>(raw[4].data());
        for (int group = 0; group < 20; ++group) {
            int32_t subtotal = 0;
            for (int offset = 0; offset < 128; ++offset)
                subtotal += int(activation[group * 128 + offset]) * int(decoded[size_t(column) * 2560 + group * 128 + offset]);
            const float weight_scale = plain ? rad_load_f32(raw[1].data(), RAD_BF16, column * 20 + group) : column_scales[group];
            total = add(total, multiply(multiply(float(subtotal), scales[group]), weight_scale));
        }
        if (plain) return rad_f32_to_bf16(total);
        return rad_f32_to_bf16(multiply(total, row_scales[column]));
    }
};
void equal(const std::vector<uint8_t>& expected, const std::vector<uint8_t>& actual, const char* name) {
    if (expected != actual) {
        size_t byte = 0; while (byte < expected.size() && expected[byte] == actual[byte]) ++byte;
        std::printf("I8_GATED_DIFF %s offset=%zu expected=%u actual=%u\n", name, byte, unsigned(expected[byte]), unsigned(actual[byte]));
    }
    if (expected != actual) throw std::runtime_error(std::string("I8 gated exact comparison: ") + name);
}
void quant_expected(const RadTensor* tensors, int rows, int width, bool producer) {
    for (int row = 0; row < rows; ++row) for (int group = 0; group < width / 128; ++group) {
        std::array<float, 128> values;
        for (int position = 0; position < 128; ++position)
            values[position] = rad_load_f32(tensors[0].data, RAD_BF16, row * tensors[0].stride[0] + group * 128 + position);
        for (int distance = 1; distance < 128; distance *= 2)
            for (int base = 0; base < 128; base += distance * 2)
                for (int offset = 0; offset < distance; ++offset) {
                    const float left = values[base + offset], right = values[base + distance + offset];
                    values[base + offset] = add(left, right); values[base + distance + offset] = add(left, -right);
                }
        float maximum = 0;
        for (float value : values) maximum = std::max(maximum, std::fabs(value));
        const float bounded = maximum > 0 ? maximum : 1e-8f;
        const float scale = producer ? bounded / 127.0f : maximum > 0 ? multiply(maximum, 1.0f / 127.0f) : 1.0f;
        const float inverse = producer ? 127.0f / bounded : 1.0f / scale;
        static_cast<float*>(tensors[2].data)[row * tensors[2].stride[0] + group] = scale;
        for (int position = 0; position < 128; ++position)
            static_cast<int8_t*>(tensors[1].data)[row * tensors[1].stride[0] + group * 128 + position] =
                iqr_output_boundary::round_even(std::max(-127.0f, std::min(127.0f, multiply(values[position], inverse))));
    }
}
}

TEST(r3d_i8_gated_actual_aql) {
    using namespace i8_gated_aql;
    if (!device_available()) SKIP("explicit gfx1100 card run required");
    REQUIRE_EQ(rad_dev_set(0), RAD_OK); REQUIRE_EQ(std::fesetround(FE_TONEAREST), 0);
    Library native("libr3d"), reference("libref");
    const auto* kernel = moe_boundary::registered(native, "r3d_gemm_nt_q_gated_i8a8h_strict");
    const auto gemm = reinterpret_cast<Launch>(dlsym(native.handle, "r3d_rad_gemm_iqr_scalar"));
    const auto gate = reinterpret_cast<Launch>(dlsym(native.handle, "r3dh_qwen_ops_gated_had_quant_i8g"));
    const auto* producer = moe_boundary::registered(native, "qwen_had_quant_i8_rowgroup");
    const auto* host_producer = reference.ngram("had_quant_act_i8g");
    REQUIRE(gemm && gate && kernel->scratch && kernel->layout && kernel->unrelayout);
    rad::RadFile file;
    REQUIRE_EQ(file.open("/home/zihaomu/bigssd2/radiance_workspace/models/qwen3.8-next-flash-iqr-v4.rad"), RAD_OK);
    Queue first(0), second(1);
    uint64_t fixtures = 0, calls = 0, values = 0, exact_bytes = 0, rejects = 0;
    for (bool affine : {false, true}) for (int parity : {0, 1}) for (int rank : {0, 1}) {
        Weight weight(kernel, file, rank ^ parity, affine);
        const int columns = weight.columns, width = columns / 2;
        for (int rows : {1, 2, 3, 8, 16}) for (int padded : {0, 1}) for (bool optional : {false, true}) {
            const int input_pitch = 2560 + padded * 16, scale_pitch = 20 + padded * 4;
            const int output_pitch = width + padded * 16, output_scale_pitch = width / 128 + padded * 3;
            Storage scratch(24576 + 384, RAD_MEM_DEVICE);
            const size_t scratch_offset = 64 + (256 - (reinterpret_cast<uintptr_t>(scratch.allocation) + 64) % 256) % 256;
            void* scratch_pointer = static_cast<uint8_t*>(scratch.allocation) + scratch_offset;
            Storage packed_weight(weight.packed[0].size(), RAD_MEM_DEVICE), packed_scale(weight.packed[1].size(), RAD_MEM_DEVICE);
            std::memcpy(packed_weight.expected.data() + 64, weight.packed[0].data(), weight.packed[0].size());
            std::memcpy(packed_scale.expected.data() + 64, weight.packed[1].data(), weight.packed[1].size());
            Storage input(rows * input_pitch * 2, RAD_MEM_DEVICE), codes(rows * input_pitch, RAD_MEM_DEVICE), scales(rows * scale_pitch * 4, RAD_MEM_DEVICE);
            Storage output(rows * output_pitch, RAD_MEM_DEVICE), output_scale(rows * output_scale_pitch * 4, RAD_MEM_DEVICE), bf16(rows * output_pitch * 2, RAD_MEM_DEVICE);
            Storage intermediate(rows * columns * 2, RAD_MEM_DEVICE), direct_output(rows * output_pitch, RAD_MEM_DEVICE),
                direct_scale(rows * output_scale_pitch * 4, RAD_MEM_DEVICE), direct_bf16(rows * output_pitch * 2, RAD_MEM_DEVICE);
            const std::vector<RadParam> params{RAD_INT("M", 16), RAD_INT("N", columns), RAD_INT("K", 2560),
                RAD_INT("group", 128), RAD_STR("dtype", "i8a8h"), RAD_STR("act", "silu")};
            RadArgs geometry{}; geometry.p = params.data(); geometry.n_p = params.size();
            REQUIRE_EQ(kernel->scratch(&geometry), 24576);
            for (int operand = 0; operand < 7; ++operand) { RadOpdDesc descriptor{}; REQUIRE_EQ(kernel->opd_shape(params.data(), params.size(), operand, &descriptor), RAD_OK); }
            std::vector<uint8_t> previous;
            for (int step = 0; step < 2; ++step) {
                const auto stream = step ? second.stream : first.stream;
                auto* input_host = reinterpret_cast<uint16_t*>(input.expected.data() + 64);
                for (int row = 0; row < rows; ++row) for (int column = 0; column < 2560; ++column)
                    input_host[row * input_pitch + column] = rad_f32_to_bf16(float((column * 17 + row * 53 + step * 29) % 191 - 95) / 256.0f);
                for (Storage* buffer : {&scratch, &packed_weight, &packed_scale, &input, &codes, &scales, &output, &output_scale, &bf16,
                    &intermediate, &direct_output, &direct_scale, &direct_bf16}) buffer->push(stream);
                REQUIRE_EQ(rad_stream_sync(stream), RAD_OK);
                RadTensor producer_tensors[]{matrix(input.device(), RAD_BF16, rows, 2560, input_pitch),
                    matrix(codes.device(), RAD_I8, rows, 2560, input_pitch), matrix(scales.device(), RAD_F32, rows, 20, scale_pitch)};
                const RadParam producer_params[]{RAD_INT("M", 16), RAD_INT("n", 2560), RAD_INT("group", 128), RAD_STR("dtype", "bf16")};
                RadArgs producer_args{}; producer_args.t = producer_tensors; producer_args.n_t = 3; producer_args.p = producer_params; producer_args.n_p = 4;
                REQUIRE_EQ(producer->launch(&producer_args, stream), RAD_OK);
                RadTensor tensors[]{producer_tensors[1], producer_tensors[2], matrix(packed_weight.device(), weight.layout[0].dtype, columns, 2560),
                    matrix(packed_scale.device(), weight.layout[1].dtype, columns, step ? 2560 : 20),
                    matrix(output.device(), RAD_I8, rows, width, output_pitch), matrix(output_scale.device(), RAD_F32, rows, width / 128, output_scale_pitch),
                    optional ? matrix(bf16.device(), RAD_BF16, rows, width, output_pitch) : RadTensor{}};
                RadArgs args = geometry; args.t = tensors; args.n_t = !optional && step ? 6 : 7; args.scratch = scratch_pointer; args.scratch_bytes = 24576;
                REQUIRE_EQ(kernel->launch(&args, stream), RAD_OK); ++calls;
                RadTensor gemm_tensors[]{tensors[0], tensors[1], tensors[2], tensors[3], matrix(intermediate.device(), RAD_BF16, rows, columns)};
                std::vector<RadParam> gemm_params = params; gemm_params.push_back(RAD_INT("impl", rows > 2 ? 1 : 2));
                RadArgs direct{}; direct.t = gemm_tensors; direct.n_t = 5; direct.p = gemm_params.data(); direct.n_p = gemm_params.size();
                if (affine) gemm_tensors[3] = matrix(packed_scale.device(), weight.layout[1].dtype, columns, 20);
                const auto actual_gemm = affine ? reinterpret_cast<Launch>(dlsym(native.handle, "r3dh_i8_rad_gemm_nt_q")) : gemm;
                REQUIRE(actual_gemm);
                REQUIRE_EQ(actual_gemm(&direct, stream), RAD_OK);
                RadTensor gate_tensors[]{gemm_tensors[4], matrix(direct_output.device(), RAD_I8, rows, width, output_pitch),
                    matrix(direct_scale.device(), RAD_F32, rows, width / 128, output_scale_pitch),
                    optional ? matrix(direct_bf16.device(), RAD_BF16, rows, width, output_pitch) : RadTensor{}};
                const RadParam gate_params[]{RAD_INT("n", width), RAD_INT("group", 128), RAD_STR("dtype", "bf16"), RAD_STR("act", "silu")};
                direct.t = gate_tensors; direct.n_t = 4; direct.p = gate_params; direct.n_p = 4;
                REQUIRE_EQ(gate(&direct, stream), RAD_OK);
                RadTensor host_producer_tensors[]{matrix(input_host, RAD_BF16, rows, 2560, input_pitch),
                    matrix(codes.expected.data() + 64, RAD_I8, rows, 2560, input_pitch), matrix(scales.expected.data() + 64, RAD_F32, rows, 20, scale_pitch)};
                producer_args.t = host_producer_tensors;
                quant_expected(host_producer_tensors, rows, 2560, true);
                auto* intermediate_host = reinterpret_cast<uint16_t*>(intermediate.expected.data() + 64);
                for (int row = 0; row < rows; ++row) for (int column = 0; column < columns; ++column) {
                    intermediate_host[row * columns + column] = weight.dot(reinterpret_cast<const int8_t*>(codes.expected.data() + 64) + row * input_pitch,
                        reinterpret_cast<const float*>(scales.expected.data() + 64) + row * scale_pitch, column); ++values;
                }
                std::memcpy(scratch.expected.data() + scratch_offset, intermediate_host, rows * columns * 2);
                std::vector<uint16_t> product(size_t(rows) * width);
                for (int row = 0; row < rows; ++row) for (int column = 0; column < width; ++column) {
                    const float gate_value = rad_load_f32(intermediate_host + row * columns + column, RAD_BF16, 0);
                    const float up_value = rad_load_f32(intermediate_host + row * columns + width + column, RAD_BF16, 0);
                    const float sigmoid = 1.0f / add(1.0f, std::exp(-gate_value));
                    const uint16_t value = rad_f32_to_bf16(multiply(multiply(gate_value, sigmoid), up_value));
                    product[size_t(row) * width + column] = value;
                    if (optional) std::memcpy(bf16.expected.data() + 64 + (row * output_pitch + column) * 2, &value, 2);
                }
                RadTensor host_gate_tensors[]{matrix(product.data(), RAD_BF16, rows, width),
                    matrix(output.expected.data() + 64, RAD_I8, rows, width, output_pitch),
                    matrix(output_scale.expected.data() + 64, RAD_F32, rows, width / 128, output_scale_pitch)};
                direct.t = host_gate_tensors; direct.n_t = 3;
                REQUIRE_EQ(host_producer->launch(&direct, nullptr), RAD_OK);
                auto independent_codes = output.expected, independent_scales = output_scale.expected;
                auto independent_tensors = std::array<RadTensor, 3>{host_gate_tensors[0], host_gate_tensors[1], host_gate_tensors[2]};
                independent_tensors[1].data = independent_codes.data() + 64;
                independent_tensors[2].data = independent_scales.data() + 64;
                quant_expected(independent_tensors.data(), rows, width, false);
                equal(output.expected, independent_codes, "CPU/libref codes");
                equal(output_scale.expected, independent_scales, "CPU/libref scales");
                direct_output.expected = output.expected; direct_scale.expected = output_scale.expected; direct_bf16.expected = bf16.expected;
                for (auto item : {std::make_pair(&output, "codes"), {&output_scale, "scales"}, {&bf16, "bf16"}, {&scratch, "scratch"},
                    {&intermediate, "intermediate"}, {&direct_output, "direct_codes"}, {&direct_scale, "direct_scales"}, {&direct_bf16, "direct_bf16"},
                    {&input, "input"}, {&codes, "activation"}, {&scales, "activation_scale"}, {&packed_weight, "weight"}, {&packed_scale, "weight_scale"}}) {
                    equal(item.first->expected, item.first->read(stream), item.second); exact_bytes += item.first->expected.size();
                }
                auto reject = [&](auto mutate, int expected = RAD_OK) {
                    std::array<RadTensor, 7> bad_tensors; std::copy(tensors, tensors + 7, bad_tensors.begin());
                    auto bad_params = params; RadArgs invalid = args; invalid.t = bad_tensors.data(); invalid.p = bad_params.data();
                    mutate(invalid, bad_tensors, bad_params);
                    const int status = kernel->launch(&invalid, reinterpret_cast<RadStream>(0x12345));
                    if (status == RAD_OK || status == RAD_E_DEVICE) throw std::runtime_error("invalid gated descriptor reached enqueue");
                    if (expected != RAD_OK && status != expected) throw std::runtime_error("wrong gated rejection status");
                    ++rejects;
                };
                reject([](auto& invalid, auto&, auto&) { invalid.scratch = nullptr; }, RAD_E_INVAL);
                reject([](auto& invalid, auto&, auto&) { invalid.scratch_bytes = 24575; }, RAD_E_INVAL);
                reject([](auto& invalid, auto&, auto&) { invalid.scratch = static_cast<uint8_t*>(invalid.scratch) + 1; }, RAD_E_INVAL);
                reject([](auto& invalid, auto&, auto&) { invalid.n_t = 5; });
                reject([](auto& invalid, auto&, auto&) { invalid.n_t = 8; });
                reject([](auto& invalid, auto&, auto&) { invalid.p = nullptr; });
                reject([](auto&, auto&, auto& parameters) { parameters[0].ival = 0; });
                reject([](auto&, auto&, auto& parameters) { parameters[0].ival = 17; });
                reject([](auto&, auto&, auto& parameters) { parameters[1].ival = 1024; });
                reject([](auto&, auto&, auto& parameters) { parameters[2].ival = 2432; });
                reject([](auto&, auto&, auto& parameters) { parameters[3].ival = 64; });
                reject([](auto&, auto&, auto& parameters) { parameters[4] = RAD_STR("dtype", "bf16"); }, RAD_E_DTYPE);
                reject([](auto&, auto&, auto& parameters) { parameters[5] = RAD_STR("act", "sigmoid"); });
                for (int index = 0; index < (optional ? 7 : 6); ++index) {
                    reject([&](auto&, auto& bad, auto&) { bad[index].dtype = RAD_U32; }, RAD_E_DTYPE);
                    reject([&](auto&, auto& bad, auto&) { bad[index].rank = 1; });
                    reject([&](auto&, auto& bad, auto&) { ++bad[index].shape[1]; });
                    reject([&](auto&, auto& bad, auto&) { bad[index].stride[1] = 2; });
                    reject([&](auto&, auto& bad, auto&) { bad[index].stride[0] = -1; });
                    reject([&](auto& invalid, auto& bad, auto&) { bad[index].data = invalid.scratch; });
                }
                equal(scratch.expected, scratch.read(stream), "reject scratch guard");
                equal(output.expected, output.read(stream), "reject output guard");
                equal(output_scale.expected, output_scale.read(stream), "reject scale guard");
                equal(bf16.expected, bf16.read(stream), "reject BF16 guard");
                if (step) REQUIRE(previous != output.expected);
                previous = output.expected;
            }
            ++fixtures;
        }
    }
    std::printf("PASS I8_GATED_AQL fixtures=%llu calls=%llu rejects=%llu independent_gemm_values=%llu exact_bytes=%llu queues=2 parity_permutations=2 tolerance=0\n",
        (unsigned long long)fixtures, (unsigned long long)calls, (unsigned long long)rejects, (unsigned long long)values, (unsigned long long)exact_bytes);
}
