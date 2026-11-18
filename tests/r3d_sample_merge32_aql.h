TEST(r3d_sample_merge32_actual_aql) {
    using namespace sample_pair_aql;
    RadDeviceProps properties{};
    REQUIRE_EQ(rad_dev_props(0, &properties), RAD_OK);
    if (properties.is_host_backend) {
        REQUIRE(!std::getenv("RAD_R3D_REQUIRE_DEVICE")); SKIP("gfx1100 AQL required");
    }
    REQUIRE_EQ(std::strcmp(properties.arch, "gfx1100"), 0);
    REQUIRE_EQ(rad_dev_count(), 1);
    REQUIRE_EQ(rad_dev_set(0), RAD_OK);
    REQUIRE_EQ(sizeof(RadSampleParams), 96u);
    REQUIRE_EQ(offsetof(RadSampleParams, seed), 0u);
    REQUIRE_EQ(offsetof(RadSampleParams, pos), 8u);
    Library native("libr3d"), reference("libref");
    const auto* combine = native.ngram("sample_merge_topk");
    const auto* ref_combine = reference.ngram("sample_merge_topk");
    REQUIRE_EQ(std::strcmp(combine->name, "r3dh_sample_merge_topk_f32"), 0);
    REQUIRE_EQ(combine->domain, RAD_DOMAIN_DEVICE);
    REQUIRE_EQ(combine->priority, 200);
    rad::Registry registry;
    REQUIRE_EQ(registry.load_plugin(home() + "/kernels/libr3d.so", 0), RAD_OK);
    Queue first(0), second(1);
    REQUIRE(first.stream != second.stream);
    Counts counts;
    uint64_t fixtures = 0, compared_values = 0, shape_checks = 0;
    for (int rows : {1, 3, 8, 16, 31, 32}) for (int width : {17, 1024, 2048}) {
        rad::Geometry geometry;
        geometry.set_i("M", 32); geometry.set_i("n_cand", width);
        const auto* selected = registry.select("sample_merge_topk", geometry, RAD_DOMAIN_DEVICE);
        REQUIRE(selected && selected->info);
        REQUIRE_EQ(std::strcmp(selected->info->name, combine->name), 0);
        geometry.set_i("M", 33);
        REQUIRE(!registry.select("sample_merge_topk", geometry, RAD_DOMAIN_DEVICE));
        std::vector<RadParam> keys{integer("M", 32), integer("n_cand", width), integer("world_size", 2)};
        for (int operand = 0; operand < 4; ++operand) {
            RadOpdDesc shape{};
            REQUIRE_EQ(combine->opd_shape(keys.data(), int(keys.size()), operand, &shape), RAD_OK);
            REQUIRE_EQ(shape.rank, 2);
            REQUIRE_EQ(shape.shape[0], 32);
            REQUIRE_EQ(shape.shape[1], operand == 0 ? 4 * width : operand == 1 ? 24 : width);
            REQUIRE_EQ(shape.dtype, operand < 2 ? RAD_U32 : operand == 2 ? RAD_I32 : RAD_F32);
            if (operand == 1) REQUIRE_EQ(shape.fill, RAD_FILL_SAMPLE_PARAMS);
            keys[0].ival = 33;
            REQUIRE_EQ(combine->opd_shape(keys.data(), int(keys.size()), operand, &shape), RAD_E_SHAPE);
            keys[0].ival = 32;
            ++shape_checks;
        }
        for (int gather_type : {RAD_U32, RAD_I32}) for (int param_type : {RAD_U32, RAD_I32, RAD_F32}) {
            Buffer gathered(gather_type, {rows, 4 * width}, RAD_MEM_DEVICE, 4 * width + 7);
            Buffer params(param_type, {rows, 24}, RAD_MEM_DEVICE);
            Buffer indices(RAD_I32, {rows, width}, RAD_MEM_DEVICE, width + 9);
            Buffer values(RAD_F32, {rows, width}, RAD_MEM_DEVICE, width + 13);
            const std::vector<Buffer*> buffers{&gathered, &params, &indices, &values};
            std::vector<uint32_t> previous_input, previous_output;
            for (int step = 0; step < 2; ++step) {
                RadStream stream = step ? second.stream : first.stream;
                for (auto* buffer : buffers)
                    std::fill(buffer->expected.begin(), buffer->expected.end(), uint8_t(0xa5 + step * 17));
                parameters(params, rows, 0, 1000 + step);
                std::vector<uint32_t> input_words;
                for (int row = 0; row < rows; ++row) {
                    auto& parameter = params.data<RadSampleParams>()[row];
                    const int mode = (row + int(fixtures) + step) % 4;
                    parameter.flags = mode == 3 ? RAD_SP_GREEDY : 0;
                    parameter.top_k = mode == 0 ? 1 : mode == 1 ? 17 : 0;
                    parameter.temp = mode == 0 ? 0.0f : 1.0f;
                    auto* words = gathered.data<uint32_t>() + row * gathered.host.stride[0];
                    const uint32_t base = 0x01000001u + uint32_t(row * 8192);
                    for (int item = 0; item < 2 * width; ++item) {
                        const int source = step ? (item + 13) % (2 * width) : 2 * width - 1 - item;
                        uint32_t id = base + uint32_t(source);
                        uint32_t logit = bits(float((source * 19 + row * 7) % 67 - 33 + step * 100) / 8.0f);
                        if (source == 2 || source == 3) {
                            id = base + 2; logit = source == 2 ? 0x80000000u : 0u;
                        }
                        if (source == 4 || source == 5) { id = base + 4; logit = bits(42.0f + step); }
                        if (step) {
                            if (source == 0 || source == 10) logit = 0x7f800000u;
                            if (source == 1) logit = 0xff800000u;
                            if (source == 6) id = UINT32_MAX;
                            if (source == 7) id = 0xfffffffeu;
                            if (source == 8) logit = 0x7fc12345u + uint32_t(row);
                            if (source == 9) logit = 0xffc23456u + uint32_t(row);
                            if (source == 11) id = INT32_MAX;
                        }
                        words[2 * item] = id; words[2 * item + 1] = logit;
                        input_words.push_back(id); input_words.push_back(logit);
                    }
                }
                if (step) REQUIRE(input_words != previous_input);
                previous_input = input_words;
                run(combine, ref_combine, buffers, keys, stream, merge, counts);
                std::vector<uint32_t> output_words;
                for (int row = 0; row < rows; ++row) for (int column = 0; column < width; ++column) {
                    output_words.push_back(uint32_t(indices.data<int32_t>()[row * indices.host.stride[0] + column]));
                    output_words.push_back(values.data<uint32_t>()[row * values.host.stride[0] + column]);
                }
                if (step) REQUIRE(output_words != previous_output);
                previous_output = output_words;
                compared_values += uint64_t(rows) * width * 2;
                auto reject = [&](int expected, const std::function<void(std::vector<RadTensor>&,
                        std::vector<RadParam>&)>& change) {
                    std::vector<RadTensor> tensors;
                    for (auto* buffer : buffers) tensors.push_back(buffer->device);
                    auto invalid_keys = keys;
                    change(tensors, invalid_keys);
                    RadArgs args{}; args.t = tensors.data(); args.n_t = int(tensors.size());
                    args.p = invalid_keys.data(); args.n_p = int(invalid_keys.size());
                    REQUIRE_EQ(combine->launch(&args, reinterpret_cast<RadStream>(0x12345)), expected);
                    ++counts.rejects;
                };
                reject(RAD_E_SHAPE, [](auto&, auto& invalid) { invalid[0].ival = 33; });
                reject(RAD_E_SHAPE, [&](auto&, auto& invalid) { invalid[0].ival = rows - 1; });
                reject(RAD_E_SHAPE, [](auto& tensors, auto&) { ++tensors[0].shape[0]; });
                reject(RAD_E_SHAPE, [](auto&, auto& invalid) { invalid[2].ival = 3; });
                for (int operand = 0; operand < 4; ++operand) {
                    reject(RAD_E_DTYPE, [&](auto& tensors, auto&) { tensors[operand].dtype = RAD_U8; });
                    reject(RAD_E_STRIDE, [&](auto& tensors, auto&) { tensors[operand].stride[1] = 2; });
                    reject(RAD_E_STRIDE, [&](auto& tensors, auto&) {
                        tensors[operand].stride[0] = tensors[operand].shape[1] - 1;
                    });
                    reject(RAD_E_INVAL, [&](auto& tensors, auto&) {
                        tensors[operand].data = static_cast<uint8_t*>(tensors[operand].data) + 1;
                    });
                }
                reject(RAD_E_STRIDE, [](auto& tensors, auto&) { tensors[1].stride[0] = 25; });
                for (int output = 2; output < 4; ++output) for (int other = 0; other < output; ++other)
                    reject(RAD_E_INVAL, [&](auto& tensors, auto&) { tensors[output].data = tensors[other].data; });
                for (int operand = 0; operand < 4; ++operand)
                    equal(buffers[operand]->expected, buffers[operand]->read(stream), "merge32-rejected-unchanged", operand);
            }
            ++fixtures;
        }
        std::printf("PASS SAMPLE_MERGE32_SLICE rows=%d band=32 n_cand=%d gather_types=2 params_types=3 changed_steps=2 queues=2\n",
            rows, width);
        std::fflush(stdout);
    }
    REQUIRE_EQ(fixtures, 108u);
    REQUIRE_EQ(counts.calls, 216u);
    REQUIRE_EQ(counts.rejects, 5616u);
    std::printf("PASS SAMPLE_MERGE32_AQL fixtures=%llu calls=%llu rejects=%llu shape_proofs=%llu exact_values=%llu exact_bytes=%llu skips=0 tolerance=0 full_guards=1 independent_cpu=1 libref=1\n",
        (unsigned long long)fixtures, (unsigned long long)counts.calls, (unsigned long long)counts.rejects,
        (unsigned long long)shape_checks, (unsigned long long)compared_values, (unsigned long long)counts.bytes);
}
