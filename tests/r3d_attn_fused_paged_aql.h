namespace attn_fused_paged_test {
using joined_head_test::Storage;
using joined_head_test::matrix;
std::vector<RadParam> parameters(int band, int64_t capacity, int context = 200000) {
    return {RAD_INT("q_len",band),RAD_INT("head_dim",256),RAD_INT("gqa",12),RAD_INT("n_head",12),
        RAD_INT("block_size",4),RAD_INT("causal",1),RAD_INT("window",0),RAD_STR("q_dtype","bf16"),
        RAD_STR("kv_dtype","i8"),RAD_INT("max_ctx",context),RAD_INT("max_seqs",capacity),
        RAD_INT("group",128),RAD_STR("act","sigmoid")};
}
bool matches(const RadKernelInfo* kernel, const std::vector<RadParam>& params) {
    rad::Geometry geometry;
    for (const auto& param : params) {
        if (param.kind == RAD_P_INT) geometry.set_i(param.key,param.ival);
        else geometry.set_s(param.key,param.sval);
    }
    for (int index = 0; index < kernel->n_constraints; ++index)
        if (!rad::constraint_holds(kernel->constraints[index],geometry)) return false;
    return true;
}
void exact(Storage& storage, RadStream stream, uint64_t& compared) {
    const auto actual = storage.read(stream);
    if (actual != storage.expected) {
        const auto mismatch = std::mismatch(actual.begin(),actual.end(),storage.expected.begin());
        std::fprintf(stderr,"PAGED_CAPACITY_MISMATCH byte=%zu actual=%u expected=%u size=%zu\n",
            size_t(mismatch.first-actual.begin()),unsigned(*mismatch.first),unsigned(*mismatch.second),actual.size());
    }
    REQUIRE(actual == storage.expected);
    compared += actual.size();
}
}

TEST(r3d_attn_fused_paged_capacity_contract) {
    using namespace attn_fused_paged_test;
    Library native("libr3d");
    const auto* fused = attn_prepare_boundary::registered(native,"r3dh_attn_fused_paged_gate");
    REQUIRE(fused->scratch && fused->opd_shape);
    REQUIRE_EQ(std::strcmp(fused->op,"attn_paged_gate_had_quant"),0);
    uint64_t accepts = 0, rejects = 0;
    for (int capacity : {256,2048}) for (int band : {1,3,8,32}) for (int context : {1,200000}) {
        auto params = parameters(band,capacity,context);
        RadArgs args{}; args.p = params.data(); args.n_p = int(params.size());
        const int64_t expected = std::min<int64_t>(2048,int64_t(band)*capacity)*6144*2;
        REQUIRE(matches(fused,params)); REQUIRE_EQ(fused->scratch(&args),expected);
        for (int operand = 0; operand < 11; ++operand) {
            RadOpdDesc descriptor{};
            REQUIRE_EQ(fused->opd_shape(params.data(),int(params.size()),operand,&descriptor),RAD_OK);
            if (operand == 0) REQUIRE_EQ(descriptor.shape[0],band);
        }
        ++accepts;
    }
    for (int64_t capacity : {int64_t(0),int64_t(1),int64_t(255),int64_t(257),int64_t(2049),
                             int64_t(2147483648LL),std::numeric_limits<int64_t>::max()}) {
        auto params = parameters(1,capacity);
        RadArgs args{}; args.p = params.data(); args.n_p = int(params.size());
        REQUIRE(!matches(fused,params)); REQUIRE_EQ(fused->scratch(&args),-1);
        RadOpdDesc descriptor{};
        REQUIRE(fused->opd_shape(params.data(),int(params.size()),0,&descriptor) != RAD_OK); ++rejects;
    }
    for (const auto& invalid : std::array<std::array<int,2>,4>{{{{0,200000}},{{33,200000}},{{1,0}},{{1,200001}}}}) {
        auto params = parameters(invalid[0],2048,invalid[1]);
        RadArgs args{}; args.p = params.data(); args.n_p = int(params.size());
        REQUIRE(!matches(fused,params)); REQUIRE_EQ(fused->scratch(&args),-1); ++rejects;
    }
    std::printf("PAGED_CAPACITY_CONTRACT_PASS accepts=%llu rejects=%llu q_len=1,3,8,32 max_seqs=256,2048\n",
        (unsigned long long)accepts,(unsigned long long)rejects);
}

TEST(r3d_attn_fused_paged_capacity_actual_aql) {
    using namespace attn_fused_paged_test;
    RadDeviceProps properties{};
    REQUIRE_EQ(rad_dev_props(0,&properties),RAD_OK);
    if (properties.is_host_backend) {
        REQUIRE(!std::getenv("RAD_R3D_REQUIRE_DEVICE")); SKIP("explicit gfx1100 card run required");
    }
    REQUIRE_EQ(std::strcmp(properties.arch,"gfx1100"),0);
    REQUIRE_EQ(rad_dev_set(0),RAD_OK); REQUIRE_EQ(std::fesetround(FE_TONEAREST),0);
    Library native("libr3d"), reference("libref");
    const auto* fused = attn_prepare_boundary::registered(native,"r3dh_attn_fused_paged_gate");
    const auto attention = reference.ngram("attn_paged")->launch;
    const auto quantize = reference.ngram("had_quant_act_i8g")->launch;
    Queue compute(0), other(1); REQUIRE(compute.stream != other.stream);
    uint64_t fixtures = 0, launches = 0, rejects = 0, compared = 0, values = 0;
    const std::array<std::array<int,4>,10> cases{{
        {{2048,1,1,1}},{{2048,1,63,63}},{{2048,1,256,256}},{{2048,1,2048,2048}},
        {{256,1,1,1}},{{256,1,63,63}},{{256,1,256,256}},
        {{256,3,768,256}},{{256,8,2048,256}},{{2048,3,2048,2048}}}};
    for (const auto& scenario : cases) for (int memory : {int(RAD_MEM_DEVICE),-1}) {
        const int capacity = scenario[0], band = scenario[1], rows = scenario[2], sequences = scenario[3];
        const int pitch = 3104, table_pitch = 520, pages = 4;
        std::vector<std::unique_ptr<Storage>> storage;
        std::vector<RadTensor> host(11), device(11);
        auto operand = [&](int index, uint32_t dtype, int count, int width, int stride, int bytes) {
            storage.push_back(std::make_unique<Storage>(size_t(count)*stride*bytes,memory));
            auto* buffer = storage.back().get();
            host[index] = matrix(buffer->expected.data()+64,dtype,count,width,stride);
            device[index] = host[index]; device[index].data = buffer->device(); return buffer;
        };
        operand(0,RAD_BF16,rows,3072,pitch,2);
        auto* cache = operand(1,RAD_I8,pages,2064,2064,1);
        host[1].rank = 4; host[1].shape[1] = 1; host[1].shape[2] = 4; host[1].shape[3] = 516;
        host[1].stride[1] = 2064; host[1].stride[2] = 516; host[1].stride[3] = 1;
        device[1] = host[1]; device[1].data = cache->device();
        operand(2,RAD_I32,sequences,513,table_pitch,4);
        operand(3,RAD_I32,sequences,1,1,4); host[3].rank = device[3].rank = 1;
        auto* output = operand(7,RAD_BF16,rows,3072,pitch,2);
        operand(8,RAD_BF16,rows,3072,pitch,2);
        operand(9,RAD_I8,rows,3072,pitch,1); operand(10,RAD_F32,rows,24,32,4);
        if (memory < 0) for (int index : {0,7,8}) {
            host[index].rank = device[index].rank = 3;
            host[index].shape[1] = device[index].shape[1] = 12;
            host[index].shape[2] = device[index].shape[2] = 256;
            host[index].stride[1] = device[index].stride[1] = 256;
            host[index].stride[2] = device[index].stride[2] = 1;
        }
        auto params = parameters(band,capacity);
        RadArgs args{}; args.p = params.data(); args.n_p = int(params.size());
        REQUIRE(matches(fused,params));
        const int64_t bytes = std::min<int64_t>(2048,int64_t(band)*capacity)*6144*2;
        REQUIRE_EQ(fused->scratch(&args),bytes);
        Storage scratch(size_t(bytes),memory);
        args.t = device.data(); args.n_t = 11; args.scratch = scratch.device(); args.scratch_bytes = bytes;
        std::vector<uint8_t> previous;
        for (int step = 0; step < 2; ++step) {
            RadStream stream = step ? other.stream : compute.stream;
            for (auto& buffer : storage) std::fill(buffer->expected.begin(),buffer->expected.end(),0x5a);
            std::fill(scratch.expected.begin(),scratch.expected.end(),0x5a);
            auto* query_data = static_cast<uint16_t*>(host[0].data);
            auto* gate_data = static_cast<uint16_t*>(host[8].data);
            for (int row = 0; row < rows; ++row) for (int column = 0; column < 3072; ++column) {
                query_data[size_t(row)*pitch+column] = rad_f32_to_bf16(float((row*7+column*3+step*17)%47-23)/32.0f);
                gate_data[size_t(row)*pitch+column] = rad_f32_to_bf16(float((row*11+column*5+step*19)%53-26)/16.0f);
            }
            auto* cache_data = static_cast<uint8_t*>(host[1].data);
            for (int slot = 0; slot < pages*4; ++slot) {
                for (int column = 0; column < 512; ++column)
                    cache_data[slot*516+column] = uint8_t(int8_t((slot*13+column*7+step*23)%251-125));
                for (int plane = 0; plane < 2; ++plane) {
                    const _Float16 scale = _Float16(float(plane+1+step)/128.0f);
                    std::memcpy(cache_data+slot*516+512+plane*2,&scale,2);
                }
            }
            auto* table_data = static_cast<int32_t*>(host[2].data);
            auto* used_data = static_cast<int32_t*>(host[3].data);
            for (int sequence = 0; sequence < sequences; ++sequence) {
                std::fill_n(table_data+sequence*table_pitch,513,-1);
                table_data[sequence*table_pitch] = (sequence+step)%pages;
                table_data[sequence*table_pitch+1] = (sequence+step+1)%pages;
                table_data[sequence*table_pitch+2] = (sequence+step+2)%pages;
                used_data[sequence] = rows/sequences+(sequence+step)%4;
            }
            for (auto& buffer : storage) buffer->push(stream);
            scratch.push(stream); REQUIRE_EQ(rad_stream_sync(stream),RAD_OK);
            std::vector<uint16_t> attended(size_t(rows)*3072);
            RadTensor attended_tensor = matrix(attended.data(),RAD_BF16,rows,3072);
            std::array<RadTensor,8> attention_tensors{};
            std::copy_n(host.begin(),8,attention_tensors.begin()); attention_tensors[7] = attended_tensor;
            RadArgs oracle{}; oracle.p = params.data(); oracle.n_p = int(params.size());
            oracle.t = attention_tensors.data(); oracle.n_t = 8; REQUIRE_EQ(attention(&oracle,nullptr),RAD_OK);
            auto* product = static_cast<uint16_t*>(host[7].data);
            auto* expected_scratch = reinterpret_cast<uint16_t*>(scratch.expected.data()+64);
            for (int row = 0; row < rows; ++row) for (int column = 0; column < 3072; ++column) {
                const size_t linear = size_t(row)*3072+column;
                const float source = rad_bf16_to_f32(gate_data[size_t(row)*pitch+column]);
                const float sigmoid = rad_bf16_to_f32(rad_f32_to_bf16(1.0f/(1.0f+std::exp(-source))));
                product[size_t(row)*pitch+column] = rad_f32_to_bf16(sigmoid*rad_bf16_to_f32(attended[linear]));
                expected_scratch[linear] = attended[linear];
                expected_scratch[size_t(rows)*3072+linear] = gate_data[size_t(row)*pitch+column];
            }
            std::array<RadTensor,3> quant_tensors{{matrix(host[7].data,RAD_BF16,rows,3072,pitch),host[9],host[10]}};
            const RadParam quant_params[]{RAD_INT("n",3072),RAD_INT("group",128),RAD_STR("dtype","bf16")};
            oracle.t = quant_tensors.data(); oracle.n_t = 3; oracle.p = quant_params; oracle.n_p = 3;
            REQUIRE_EQ(quantize(&oracle,nullptr),RAD_OK);
            if (step) REQUIRE(previous != output->expected);
            previous = output->expected;
            REQUIRE_EQ(fused->launch(&args,stream),RAD_OK); ++launches; REQUIRE_EQ(rad_stream_sync(stream),RAD_OK);
            for (auto& buffer : storage) exact(*buffer,stream,compared);
            exact(scratch,stream,compared); values += uint64_t(rows)*(3072+3072+24);
            auto reject = [&](RadArgs bad, int expected) { REQUIRE_EQ(fused->launch(&bad,stream),expected); ++rejects; };
            RadArgs bad = args; bad.scratch = nullptr; reject(bad,RAD_E_INVAL);
            bad = args; bad.scratch_bytes = bytes-1; reject(bad,RAD_E_INVAL);
            bad = args; bad.scratch_bytes = -1; reject(bad,RAD_E_INVAL);
            bad = args; bad.scratch = static_cast<uint8_t*>(args.scratch)+1; reject(bad,RAD_E_INVAL);
            if (capacity == 2048 && band == 1) {
                bad = args; bad.scratch_bytes = int64_t(256)*6144*2; reject(bad,RAD_E_INVAL);
            }
            auto malformed = device; malformed[3].shape[0] = capacity+1;
            bad = args; bad.t = malformed.data(); reject(bad,RAD_E_SHAPE);
            malformed = device; malformed[0].shape[0] = std::min<int64_t>(2048,int64_t(band)*capacity)+1;
            bad = args; bad.t = malformed.data(); reject(bad,RAD_E_SHAPE);
            malformed = device; malformed[0].shape[0] = 0;
            bad = args; bad.t = malformed.data(); reject(bad,RAD_E_SHAPE);
            auto invalid_params = parameters(band,2049);
            bad = args; bad.p = invalid_params.data(); reject(bad,RAD_E_UNSUPPORTED);
            REQUIRE_EQ(rad_stream_sync(stream),RAD_OK);
            for (auto& buffer : storage) exact(*buffer,stream,compared);
            exact(scratch,stream,compared);
        }
        ++fixtures;
        std::printf("PAGED_CAPACITY_AQL_CASE max_seqs=%d q_len=%d rows=%d sequences=%d memory=%s changed_inputs=2 queues=2 exact=1\n",
            capacity,band,rows,sequences,memory < 0 ? "VMEM" : "DEVICE");
    }
    REQUIRE_EQ(radtest::failures().load(),0);
    REQUIRE_EQ(fixtures,uint64_t(20)); REQUIRE_EQ(launches,uint64_t(40));
    std::printf("PAGED_CAPACITY_ACTUAL_AQL_PASS fixtures=%llu launches=%llu rejects=%llu exhaustive_values=%llu exact_bytes=%llu skips=0\n",
        (unsigned long long)fixtures,(unsigned long long)launches,(unsigned long long)rejects,
        (unsigned long long)values,(unsigned long long)compared);
}
