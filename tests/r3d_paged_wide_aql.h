namespace paged_wide_test {
using joined_head_test::Storage;
using joined_head_test::matrix;
struct Scenario { int rows, used, width, pages, sequences; };
void run(std::initializer_list<Scenario> scenarios) {
    RadDeviceProps properties{};
    REQUIRE_EQ(rad_dev_props(0,&properties),RAD_OK);
    if (properties.is_host_backend) {
        REQUIRE(!std::getenv("RAD_R3D_REQUIRE_DEVICE")); SKIP("explicit gfx1100 card run required");
    }
    REQUIRE_EQ(std::strcmp(properties.arch,"gfx1100"),0);
    REQUIRE_EQ(rad_dev_count(),1);
    REQUIRE_EQ(rad_dev_set(0),RAD_OK);
    REQUIRE_EQ(std::fesetround(FE_TONEAREST),0);
    Library native("libr3d"), reference("libref");
    const auto* registered = attn_prepare_boundary::registered(native,"sparse_qsa_selected_union_i8_strict");
    const auto oracle = reference.ngram("attn_paged")->launch;
    rad::Registry registry;
    REQUIRE_EQ(registry.load_plugin(home()+"/kernels/libr3d.so",0),RAD_OK);
    Queue compute(0), other(1);
    REQUIRE(compute.stream != other.stream);
    uint64_t fixtures = 0, launches = 0, rejects = 0, compared = 0, values = 0, history = 0;
    for (const auto& scenario : scenarios) for (int memory : {int(RAD_MEM_DEVICE),-1}) {
        const int rows = scenario.rows, sequences = scenario.sequences;
        const bool full_history = scenario.used >= 199999;
        const int pitch = memory < 0 ? 3104 : 3072;
        const int table_pitch = scenario.width+(memory < 0 ? 8 : 0);
        std::vector<std::unique_ptr<Storage>> storage;
        std::vector<RadTensor> host(8), device(8);
        auto operand = [&](int index, uint32_t dtype, int count, int width, int stride, int bytes) {
            storage.push_back(std::make_unique<Storage>(size_t(count)*stride*bytes,memory));
            auto* buffer = storage.back().get();
            host[index] = matrix(buffer->expected.data()+64,dtype,count,width,stride);
            device[index] = host[index]; device[index].data = buffer->device(); return buffer;
        };
        operand(0,RAD_BF16,rows,3072,pitch,2);
        auto* cache = operand(1,RAD_I8,scenario.pages,2064,2064,1);
        host[1].rank = 4; host[1].shape[1] = 1; host[1].shape[2] = 4; host[1].shape[3] = 516;
        host[1].stride[1] = 2064; host[1].stride[2] = 516; host[1].stride[3] = 1;
        device[1] = host[1]; device[1].data = cache->device();
        operand(2,RAD_I32,sequences,scenario.width,table_pitch,4);
        operand(3,RAD_I32,sequences,1,1,4); host[3].rank = device[3].rank = 1;
        operand(6,RAD_I32,sequences+1,1,1,4); host[6].rank = device[6].rank = 1;
        auto* output = operand(7,RAD_BF16,rows,3072,pitch,2);
        if (memory < 0) for (int index : {0,7}) {
            host[index].rank = device[index].rank = 3;
            host[index].shape[1] = device[index].shape[1] = 12;
            host[index].shape[2] = device[index].shape[2] = 256;
            host[index].stride[1] = device[index].stride[1] = 256;
            host[index].stride[2] = device[index].stride[2] = 1;
        }
        auto params = attn_fused_paged_test::parameters(2048,256);
        params.resize(params.size()-2);
        rad::Geometry geometry;
        for (const auto& param : params) {
            if (param.kind == RAD_P_INT) geometry.set_i(param.key,param.ival);
            else geometry.set_s(param.key,param.sval);
        }
        const auto* selected = registry.select("attn_paged",geometry,RAD_DOMAIN_DEVICE);
        REQUIRE(selected && selected->info && selected->info->launch);
        REQUIRE_EQ(std::strcmp(selected->info->name,registered->name),0);
        std::printf("PAGED_WIDE_REGISTRY_SELECTED name=%s q_len=2048 max_ctx=200000 max_seqs=256 priority=%d\n",
            selected->info->name,selected->info->priority);
        RadArgs args{}; args.p = params.data(); args.n_p = int(params.size());
        args.t = device.data(); args.n_t = 8;
        std::vector<uint8_t> previous;
        for (int step = 0; step < 2; ++step) {
            RadStream stream = step ? other.stream : compute.stream;
            for (auto& buffer : storage) std::fill(buffer->expected.begin(),buffer->expected.end(),0x5a);
            auto* query_data = static_cast<uint16_t*>(host[0].data);
            for (int row = 0; row < rows; ++row) for (int column = 0; column < 3072; ++column)
                query_data[size_t(row)*pitch+column] = column < 256 ? 0
                    : rad_f32_to_bf16(float((row*7+column*3+step*17)%47-23)/64.0f);
            auto* cache_data = static_cast<uint8_t*>(host[1].data);
            for (int slot = 0; slot < scenario.pages*4; ++slot) {
                for (int column = 0; column < 512; ++column)
                    cache_data[size_t(slot)*516+column] = uint8_t(int8_t((slot*13+column*7+step*23)%251-125));
                for (int plane = 0; plane < 2; ++plane) {
                    const _Float16 scale = _Float16(float(plane+1+step)/128.0f);
                    std::memcpy(cache_data+size_t(slot)*516+512+plane*2,&scale,2);
                }
            }
            auto* table = static_cast<int32_t*>(host[2].data);
            auto* used = static_cast<int32_t*>(host[3].data);
            auto* prefix = static_cast<int32_t*>(host[6].data);
            prefix[0] = 0;
            for (int sequence = 0; sequence < sequences; ++sequence) {
                const int end = sequences == 1 ? rows : sequence == 0 ? 0
                    : sequence == 1 ? 1 : rows;
                prefix[sequence+1] = end;
                used[sequence] = end-prefix[sequence]+scenario.used-rows;
                std::fill_n(table+sequence*table_pitch,scenario.width,-1);
                for (int block = 0; block < (used[sequence]+3)/4; ++block)
                    table[sequence*table_pitch+block] = full_history
                        ? 1+(block*7+sequence*11+step*13)%(scenario.pages-2)
                        : (block*7+sequence*11+step*13)%scenario.pages;
                if (full_history) {
                    REQUIRE_EQ(scenario.width,50000);
                    table[sequence*table_pitch] = 0;
                    table[sequence*table_pitch+49999] = scenario.pages-1;
                    REQUIRE(std::all_of(table+sequence*table_pitch,table+sequence*table_pitch+50000,
                        [&](int physical) { return physical >= 0 && physical < scenario.pages; }));
                    for (int physical : {0,scenario.pages-1})
                        for (int token = 0; token < 4; ++token) for (int column = 256; column < 512; ++column)
                            cache_data[(size_t(physical)*4+token)*516+column] = uint8_t(120-step*20);
                }
                if (!full_history && used[sequence] > 2048) for (int block = 512; block < (used[sequence]+3)/4; ++block) {
                    table[sequence*table_pitch+block] = scenario.pages-1;
                    for (int token = 0; token < 4; ++token) for (int column = 256; column < 512; ++column)
                        cache_data[(size_t(scenario.pages-1)*4+token)*516+column] = uint8_t(100-step*20);
                }
            }
            for (auto& buffer : storage) buffer->push(stream);
            REQUIRE_EQ(rad_stream_sync(stream),RAD_OK);
            RadArgs expected = args; expected.t = host.data();
            REQUIRE_EQ(oracle(&expected,nullptr),RAD_OK);
            if (full_history) {
                for (int boundary : {0,49999}) {
                    auto alternative = output->expected;
                    auto altered = host; altered[7].data = alternative.data()+64;
                    std::vector<int32_t> saved(sequences);
                    for (int sequence = 0; sequence < sequences; ++sequence) {
                        saved[sequence] = table[sequence*table_pitch+boundary];
                        table[sequence*table_pitch+boundary] = table[sequence*table_pitch+1];
                    }
                    RadArgs witness = args; witness.t = altered.data();
                    REQUIRE_EQ(oracle(&witness,nullptr),RAD_OK);
                    REQUIRE(alternative != output->expected); ++history;
                    for (int sequence = 0; sequence < sequences; ++sequence)
                        table[sequence*table_pitch+boundary] = saved[sequence];
                }
                const int maximum_used = *std::max_element(used,used+sequences);
                if (maximum_used%4) {
                    auto* final_page = cache_data+size_t(scenario.pages-1)*2064;
                    std::array<uint8_t,2064> saved{};
                    std::copy_n(final_page,2064,saved.begin());
                    for (int token = maximum_used%4; token < 4; ++token)
                        std::fill_n(final_page+token*516+256,256,uint8_t(-120));
                    auto alternative = output->expected;
                    auto altered = host; altered[7].data = alternative.data()+64;
                    RadArgs witness = args; witness.t = altered.data();
                    REQUIRE_EQ(oracle(&witness,nullptr),RAD_OK);
                    REQUIRE(alternative == output->expected);
                    std::copy(saved.begin(),saved.end(),final_page);
                }
            }
            if (scenario.used > 2048 && sequences == 1) {
                auto alternative = output->expected;
                auto altered = host;
                altered[7].data = alternative.data()+64;
                std::vector<int32_t> truncated(table,table+table_pitch);
                std::fill(truncated.begin()+512,truncated.end(),table[0]);
                altered[2].data = truncated.data(); expected.t = altered.data();
                REQUIRE_EQ(oracle(&expected,nullptr),RAD_OK);
                REQUIRE(alternative != output->expected); ++history;
            }
            if (step) REQUIRE(previous != output->expected);
            previous = output->expected;
            REQUIRE_EQ(selected->info->launch(&args,stream),RAD_OK); ++launches;
            REQUIRE_EQ(rad_stream_sync(stream),RAD_OK);
            for (auto& buffer : storage) attn_fused_paged_test::exact(*buffer,stream,compared);
            values += uint64_t(rows)*3072;
            std::fill(output->expected.begin(),output->expected.end(),0x5a);
            output->push(stream); REQUIRE_EQ(rad_stream_sync(stream),RAD_OK);
            auto refuse = [&](std::vector<RadTensor> malformed) {
                RadArgs bad = args; bad.t = malformed.data();
                REQUIRE(selected->info->launch(&bad,stream) < 0); ++rejects;
            };
            auto malformed = device;
            malformed[2].shape[1] = 50001; malformed[2].stride[0] = 50001; refuse(malformed);
            malformed = device; malformed[6] = {}; refuse(malformed);
            for (int index : {0,1,2,3,6,7}) {
                malformed = device; malformed[index].data = nullptr; refuse(malformed);
                malformed = device; malformed[index].dtype = RAD_F32; refuse(malformed);
                malformed = device; malformed[index].stride[malformed[index].rank-1] = 2; refuse(malformed);
            }
            malformed = device; malformed[0].stride[0] = 3071; refuse(malformed);
            malformed = device; malformed[7].stride[0] = 3071; refuse(malformed);
            malformed = device; malformed[2].stride[0] = scenario.width-1; refuse(malformed);
            malformed = device; malformed[2].shape[0] = sequences+1; refuse(malformed);
            malformed = device; malformed[6].shape[0] = sequences+2; refuse(malformed);
            malformed = device; malformed[1].shape[0] = r3dh_attn_qsa_max_cache_pages + 1; refuse(malformed);
            malformed = device; malformed[1].shape[1] = 2; refuse(malformed);
            malformed = device; malformed[1].stride[0] += 4; refuse(malformed);
            malformed = device; malformed[1].dtype = RAD_BF16; malformed[1].shape[3] = 512; refuse(malformed);
            for (int index : {4,5}) { malformed = device; malformed[index] = device[3]; refuse(malformed); }
            malformed = device; malformed[7] = device[0]; refuse(malformed);
            for (const char* key : {"head_dim","gqa","n_head","block_size","causal","window","q_len","max_ctx","max_seqs"}) {
                auto invalid = params;
                for (auto& param : invalid) if (!std::strcmp(param.key,key)) param.ival = !std::strcmp(key,"window") ? 1 : 0;
                RadArgs bad = args; bad.p = invalid.data();
                REQUIRE(selected->info->launch(&bad,stream) < 0); ++rejects;
            }
            REQUIRE_EQ(rad_stream_sync(stream),RAD_OK);
            for (auto& buffer : storage) attn_fused_paged_test::exact(*buffer,stream,compared);
        }
        ++fixtures;
        std::printf("PAGED_WIDE_CASE rows=%d used=%d width=%d pages=%d seqs=%d memory=%s changed_inputs=2 queues=2 exact=1\n",
            rows,scenario.used,scenario.width,scenario.pages,sequences,memory < 0 ? "VMEM" : "DEVICE");
    }
    REQUIRE_EQ(radtest::failures().load(),0);
    REQUIRE(fixtures > 0 && launches == fixtures*2 && history > 0);
    std::printf("PAGED_WIDE_AQL_PASS fixtures=%llu launches=%llu rejects=%llu exhaustive_bf16_values=%llu exact_bytes=%llu history_witnesses=%llu skips=0\n",
        (unsigned long long)fixtures,(unsigned long long)launches,(unsigned long long)rejects,
        (unsigned long long)values,(unsigned long long)compared,(unsigned long long)history);
}
}
TEST(r3d_paged_wide_focused_aql) {
    paged_wide_test::run({{63,2111,768,541,1}});
}
TEST(r3d_paged_wide_matrix_aql) {
    paged_wide_test::run({{2048,2048,768,50015,1},{63,2052,513,541,1},
        {63,2111,50000,541,1},{63,2111,768,541,3},{1,1,768,50512,1}});
}
TEST(r3d_paged_wide_history200k_aql) {
    paged_wide_test::run({{1,199999,50000,50000,1},{1,200000,50000,400026,1},
        {3,200000,50000,400026,3}});
}
