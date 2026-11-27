#include <array>

namespace qsa_index_boundary {
using Launch = int (*)(const RadArgs*, RadStream);
int sequence(const RadTensor& cu, int row, int fallback) {
    const auto* cuts = static_cast<const int*>(cu.data);
    for (int index = 0; index + 1 < cu.shape[0]; ++index)
        if (row >= cuts[index * cu.stride[0]] && row < cuts[(index + 1) * cu.stride[0]]) return index;
    return fallback;
}
void score_expected(const std::array<RadTensor, 6>& tensors) {
    const auto& query = tensors[0];
    const auto& keys = tensors[1];
    const auto& table = tensors[2];
    for (int row = 0; row < tensors[5].shape[0]; ++row) {
        const int owner = sequence(tensors[3], row, -1);
        if (owner < 0) continue;
        const int count = static_cast<const int*>(tensors[4].data)[owner * tensors[4].stride[0]];
        for (int column = 0; column < std::min<int64_t>({count, table.shape[1], tensors[5].shape[1]}); ++column) {
            const int page = static_cast<const int*>(table.data)[owner * table.stride[0] + column];
            float value = -INFINITY;
            if (page >= 0 && page < keys.shape[0]) {
                value = 0;
                for (int head = 0; head < 4; ++head) {
                    float dot = 0;
                    for (int lane = 0; lane < 128; ++lane) {
                        const float first = rad_bf16_to_f32(static_cast<const uint16_t*>(query.data)
                            [row * query.stride[0] + head * 128 + lane]);
                        const float second = rad_bf16_to_f32(static_cast<const uint16_t*>(keys.data)
                            [page * keys.stride[0] + lane]);
                        volatile float product = first * second;
                        dot = dot + product;
                    }
                    value = value + std::fmax(dot, 0.0f);
                }
                value = value / std::sqrt(128.0f);
            }
            static_cast<float*>(tensors[5].data)[row * tensors[5].stride[0] + column] = value;
        }
    }
}
void select_expected(const std::array<RadTensor, 8>& tensors, int topk, bool signed_score) {
    for (int row = 0; row < tensors[0].shape[0]; ++row) {
        const int owner = sequence(tensors[4], row, 0);
        const int position = static_cast<const int*>(tensors[3].data)[row * tensors[3].stride[0]];
        const int width = int(std::min(tensors[0].shape[1], tensors[1].shape[1]));
        const int complete = std::min({static_cast<const int*>(tensors[2].data)[owner * tensors[2].stride[0]],
            width, (position + 1) / 4});
        std::vector<std::pair<float, int>> live;
        for (int column = 0; column < complete; ++column) {
            const float value = static_cast<const float*>(tensors[0].data)[row * tensors[0].stride[0] + column];
            if (static_cast<const int*>(tensors[1].data)[owner * tensors[1].stride[0] + column] < 0
                || (signed_score ? !std::isfinite(value) : !(value > -1e30f))) continue;
            live.emplace_back(signed_score ? value : std::max(value, 0.0f), column);
        }
        auto ranked = live;
        std::stable_sort(ranked.begin(), ranked.end(), [](const auto& left, const auto& right) {
            return left.first > right.first;
        });
        std::vector<int> chosen;
        if (live.size() <= size_t(topk)) {
            for (const auto& entry : live) chosen.push_back(entry.second);
        } else {
            const float threshold = ranked[topk - 1].first;
            for (const auto& entry : live) if (entry.first > threshold) chosen.push_back(entry.second);
            for (const auto& entry : live)
                if (chosen.size() < size_t(topk) && entry.first == threshold) chosen.push_back(entry.second);
        }
        std::vector<int> pages;
        for (int column : chosen)
            pages.push_back(static_cast<const int*>(tensors[1].data)[owner * tensors[1].stride[0] + column]);
        if (complete < width && (position + 1) % 4) {
            const int partial = static_cast<const int*>(tensors[1].data)[owner * tensors[1].stride[0] + complete];
            if (partial >= 0) pages.push_back(partial);
        }
        for (int column = 0; column <= topk; ++column)
            static_cast<int*>(tensors[5].data)[row * tensors[5].stride[0] + column] =
                column < int(pages.size()) ? pages[column] : -1;
        static_cast<int*>(tensors[6].data)[row * tensors[6].stride[0]] = int(pages.size());
        static_cast<int*>(tensors[7].data)[row * tensors[7].stride[0]] =
            pages.empty() ? 0 : (int(pages.size()) - 1) * 4 + position % 4 + 1;
    }
}
}

#include "r3d_qsa_radix_aql.h"

TEST(r3d_qsa_index_actual_aql_compact_score_select) {
    using namespace qsa_index_boundary;
    RadDeviceProps properties{};
    REQUIRE_EQ(rad_dev_props(0, &properties), RAD_OK);
    if (properties.is_host_backend) {
        REQUIRE(!std::getenv("RAD_R3D_REQUIRE_DEVICE"));
        SKIP("explicit gfx1100 card run required");
    }
    REQUIRE_EQ(std::strcmp(properties.arch, "gfx1100"), 0);
    REQUIRE_EQ(rad_dev_set(0), RAD_OK);
    Library native("libr3d"), reference("libref");
    const auto* score = native.ngram("qsa_score");
    const auto* select = native.ngram("qsa_select");
    REQUIRE_EQ(std::strcmp(score->name, "sparse_qsa_score_bf16_strict"), 0);
    REQUIRE_EQ(std::strcmp(select->name, "sparse_qsa_select_signed_or_relu_strict"), 0);
    REQUIRE_EQ(score->domain, RAD_DOMAIN_DEVICE);
    REQUIRE_EQ(select->domain, RAD_DOMAIN_DEVICE);
    Queue compute(0), other(1);
    REQUIRE(compute.stream && other.stream && compute.stream != other.stream);
    uint64_t fixtures = 0, pipelines = 0, selections = 0, rejects = 0, compared = 0, cpu_reference = 0;
    uint64_t boundary512 = 0, boundary513 = 0, signed_cases = 0;
    for (int rows : {1, 3, 63, 64}) for (int sequences : {1, 2, 8})
        for (int context : {63, 64, 2052}) for (int padded : {0, 1}) for (int signed_score : {0, 1}) {
            const int width = context < 100 ? 17 : 514;
            const int table_width = context < 100 ? 256 : 1024;
            std::vector<std::unique_ptr<Buffer>> buffers;
            auto add = [&](int type, std::initializer_list<int64_t> shape, int pitch = 0) {
                buffers.push_back(std::make_unique<Buffer>(type, shape, RAD_MEM_DEVICE, pitch));
                return buffers.back().get();
            };
            Buffer* query = add(RAD_BF16, {rows, 512}, padded ? 544 : 512);
            Buffer* keys = add(RAD_BF16, {1045, 1, 4, 32});
            Buffer* table = add(RAD_I32, {sequences, table_width}, table_width + padded * 7);
            Buffer* cu = add(RAD_I32, {sequences + 1});
            Buffer* count = add(RAD_I32, {sequences});
            Buffer* scores = add(RAD_F32, {rows, width}, 1024);
            Buffer* positions = add(RAD_I32, {rows});
            Buffer* selected = add(RAD_I32, {rows, 513}, 520 + padded * 9);
            Buffer* nselected = add(RAD_I32, {rows});
            Buffer* used = add(RAD_I32, {rows});
            std::array<RadTensor, 6> score_host{}, score_device{};
            std::array<RadTensor, 8> select_host{}, select_device{};
            for (int index = 0; index < 6; ++index) {
                score_host[index] = buffers[index]->host;
                score_device[index] = buffers[index]->device;
            }
            const int select_indices[]{5, 2, 4, 6, 3, 7, 8, 9};
            for (int index = 0; index < 8; ++index) {
                select_host[index] = buffers[select_indices[index]]->host;
                select_device[index] = buffers[select_indices[index]]->device;
            }
            RadParam score_params[]{RAD_INT("n", 128), RAD_INT("heads", 4), RAD_INT("M", 64),
                RAD_INT("blocks", width), RAD_INT("seqs", sequences), RAD_STR("dtype", "bf16")};
            RadParam select_params[]{RAD_INT("ratio", 4), RAD_INT("topk", 512), RAD_INT("signed", signed_score),
                RAD_INT("M", 64), RAD_INT("blocks", width), RAD_INT("seqs", sequences)};
            RadArgs score_args{}, select_args{};
            score_args.t = score_device.data(); score_args.n_t = 6; score_args.p = score_params; score_args.n_p = 6;
            select_args.t = select_device.data(); select_args.n_t = 8; select_args.p = select_params; select_args.n_p = 6;
            std::vector<uint8_t> previous_scores, previous_selected;
            for (int step = 0; step < 2; ++step) {
                const RadStream stream = step ? other.stream : compute.stream;
                for (int row = 0; row < rows; ++row) for (int column = 0; column < 512; ++column)
                    query->data<uint16_t>()[row * query->host.stride[0] + column] =
                        rad_f32_to_bf16(float((row * 3 + column * 7 + step * 11) % 31 - 15) / 16);
                for (int page = 0; page < 1045; ++page) for (int lane = 0; lane < 128; ++lane)
                    keys->data<uint16_t>()[page * 128 + lane] =
                        rad_f32_to_bf16(float((page * 11 + lane * 3 + step * 5) % 29 - 14) / 32);
                for (int owner = 0; owner <= sequences; ++owner)
                    cu->data<int>()[owner * cu->host.stride[0]] = rows * owner * owner / (sequences * sequences);
                if (padded && rows > 1) cu->data<int>()[0] = std::min(1, cu->data<int>()[1]);
                for (int owner = 0; owner < sequences; ++owner) {
                    count->data<int>()[owner * count->host.stride[0]] = (context + 3) / 4 - (padded ? owner % 3 : 0);
                    for (int column = 0; column < table_width; ++column)
                        table->data<int>()[owner * table->host.stride[0] + column] =
                            padded && (column + owner) % 23 == 7 ? -1 : (column * 17 + owner * 37 + step * 19) % 1045;
                }
                for (int row = 0; row < rows; ++row)
                    positions->data<int>()[row * positions->host.stride[0]] = context == 2052
                        ? 2050 + (row + step) % 2 : std::max(0, context - rows + row - (step ? 0 : 1));
                for (Buffer* output : {scores, selected, nselected, used})
                    std::fill(output->expected.begin(), output->expected.end(), uint8_t(0x5a));
                for (auto& buffer : buffers) buffer->push(stream);
                REQUIRE_EQ(rad_stream_sync(stream), RAD_OK);
                score_expected(score_host);
                const auto expected_score = scores->expected;
                std::fill(scores->expected.begin(), scores->expected.end(), uint8_t(0x5a));
                score_args.t = score_host.data();
                REQUIRE_EQ(reference.ngram("qsa_score")->launch(&score_args, nullptr), RAD_OK);
                REQUIRE(scores->expected == expected_score);
                cpu_reference += expected_score.size();
                score_args.t = score_device.data();
                auto oracle_select = [&](int topk) {
                    select_expected(select_host, topk, signed_score);
                    const std::array<std::vector<uint8_t>, 3> expected{selected->expected, nselected->expected, used->expected};
                    for (Buffer* output : {selected, nselected, used})
                        std::fill(output->expected.begin(), output->expected.end(), uint8_t(0x5a));
                    select_args.t = select_host.data();
                    REQUIRE_EQ(reference.ngram("qsa_select")->launch(&select_args, nullptr), RAD_OK);
                    select_args.t = select_device.data();
                    int index = 0;
                    for (Buffer* output : {selected, nselected, used}) {
                        REQUIRE(output->expected == expected[index++]);
                        cpu_reference += output->expected.size();
                    }
                };
                oracle_select(512);
                const int score_status = score->launch(&score_args, stream);
                if (score_status != RAD_OK) {
                    std::fprintf(stderr, "QSA_SCORE_REFUSAL M=%d seqs=%d ctx=%d padded=%d signed=%d step=%d status=%d\n",
                        rows, sequences, context, padded, signed_score, step, score_status);
                    for (int operand = 0; operand < 6; ++operand) {
                        const auto& tensor = score_device[operand];
                        std::fprintf(stderr, "operand=%d ptr=%p rank=%u shape=%lld,%lld,%lld,%lld stride=%lld,%lld,%lld,%lld\n",
                            operand, tensor.data, tensor.rank, (long long)tensor.shape[0], (long long)tensor.shape[1],
                            (long long)tensor.shape[2], (long long)tensor.shape[3], (long long)tensor.stride[0],
                            (long long)tensor.stride[1], (long long)tensor.stride[2], (long long)tensor.stride[3]);
                    }
                }
                REQUIRE_EQ(score_status, RAD_OK);
                REQUIRE_EQ(select->launch(&select_args, stream), RAD_OK);
                ++pipelines;
                for (auto& buffer : buffers) {
                    const auto actual = buffer->read(stream);
                    if (actual != buffer->expected) {
                        const size_t offset = std::mismatch(actual.begin(), actual.end(), buffer->expected.begin()).first - actual.begin();
                        std::fprintf(stderr, "QSA_INDEX_DIFF M=%d seqs=%d ctx=%d padded=%d signed=%d step=%d operand=%td byte=%zu\n",
                            rows, sequences, context, padded, signed_score, step, &buffer - buffers.data(), offset);
                    }
                    REQUIRE(actual == buffer->expected);
                    compared += actual.size();
                }
                if (context == 2052 && !padded) for (int row = 0; row < rows; ++row) {
                    const int expected = positions->data<int>()[row] == 2050 ? 513 : 512;
                    REQUIRE_EQ(nselected->data<int>()[row], expected);
                    expected == 512 ? ++boundary512 : ++boundary513;
                }
                if (step) {
                    REQUIRE(scores->expected != previous_scores);
                    REQUIRE(selected->expected != previous_selected);
                }
                previous_scores = scores->expected; previous_selected = selected->expected;
                const float special[]{-INFINITY, INFINITY, NAN, -1e31f, -1e29f, -4, -0.0f, 0.0f, 2, 2, 7, -2};
                for (int topk : {2, 512}) {
                    select_params[1].ival = topk;
                    select_host[5].shape[1] = select_device[5].shape[1] = topk + 1;
                    for (int row = 0; row < rows; ++row) for (int column = 0; column < width; ++column)
                        scores->data<float>()[row * 1024 + column] = special[(column + row + step) % 12];
                    for (Buffer* output : {selected, nselected, used})
                        std::fill(output->expected.begin(), output->expected.end(), uint8_t(0x5a));
                    for (Buffer* buffer : {scores, selected, nselected, used}) buffer->push(stream);
                    REQUIRE_EQ(rad_stream_sync(stream), RAD_OK);
                    oracle_select(topk);
                    REQUIRE_EQ(select->launch(&select_args, stream), RAD_OK);
                    for (auto& buffer : buffers) { REQUIRE(buffer->read(stream) == buffer->expected); compared += buffer->expected.size(); }
                    ++selections;
                    if (signed_score) ++signed_cases;
                }
                select_params[1].ival = 512;
                select_host[5].shape[1] = select_device[5].shape[1] = 513;
                score_device[1].stride[2]++;
                CHECK_EQ(score->launch(&score_args, stream), RAD_E_SHAPE); ++rejects;
                score_device[1] = keys->device;
                score_params[1].ival = 3;
                CHECK_EQ(score->launch(&score_args, stream), RAD_E_UNSUPPORTED); ++rejects;
                score_params[1].ival = 4;
                select_params[1].ival = 513;
                CHECK_EQ(select->launch(&select_args, stream), RAD_E_UNSUPPORTED); ++rejects;
                select_params[1].ival = 512;
                select_device[5].shape[1] = 512;
                CHECK_EQ(select->launch(&select_args, stream), RAD_E_SHAPE); ++rejects;
                select_device[5].shape[1] = 513;
                for (auto& buffer : buffers) REQUIRE(buffer->read(stream) == buffer->expected);
            }
            ++fixtures;
        }
    REQUIRE(boundary512 && boundary513 && signed_cases);
    std::printf("PASS QSA_INDEX_AQL fixtures=%llu score_select_pipelines=%llu score_launches=%llu select_launches=%llu rejects=%llu exact_device_bytes=%llu independent_libref_bytes=%llu boundary512=%llu boundary513=%llu queues=2 tolerance=0\n",
        (unsigned long long)fixtures, (unsigned long long)pipelines, (unsigned long long)pipelines,
        (unsigned long long)(pipelines + selections), (unsigned long long)rejects, (unsigned long long)compared,
        (unsigned long long)cpu_reference, (unsigned long long)boundary512, (unsigned long long)boundary513);
}
