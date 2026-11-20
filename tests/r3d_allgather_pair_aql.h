#pragma once
#include <chrono>
#include <exception>
#include <thread>

namespace allgather_pair_aql {
struct Devices {
    void* runtime = dlopen("libamdhip64.so", RTLD_NOW | RTLD_LOCAL);
    int (*set)(int) = runtime ? reinterpret_cast<int (*)(int)>(dlsym(runtime,"hipSetDevice")) : nullptr;
    int (*count)(int*) = runtime ? reinterpret_cast<int (*)(int*)>(dlsym(runtime,"hipGetDeviceCount")) : nullptr;
    int (*bus)(char*,int,int) = runtime ? reinterpret_cast<int (*)(char*,int,int)>(dlsym(runtime,"hipDeviceGetPCIBusId")) : nullptr;
    Devices() {
        REQUIRE(runtime && set && count && bus);
        int visible = 0;
        REQUIRE_EQ(count(&visible),0);
        REQUIRE_EQ(visible,2);
        REQUIRE_EQ(rad_dev_count(),2);
        REQUIRE_EQ(std::strcmp(rad::device_backend_name(),"aql"),0);
        for (int rank : {0,1}) {
            char address[32]{};
            REQUIRE_EQ(bus(address,sizeof(address),rank),0);
            RadDeviceProps props{};
            REQUIRE_EQ(rad_dev_props(rank,&props),RAD_OK);
            REQUIRE(!props.is_host_backend);
            REQUIRE_EQ(std::strcmp(props.arch,"gfx1100"),0);
            std::printf("ALLGATHER_AQL_DEVICE rank=%d pci=%s backend=aql arch=%s\n",rank,address,props.arch);
        }
    }
    ~Devices() { if (runtime) dlclose(runtime); }
    void bind(int rank) { REQUIRE_EQ(set(rank),0); REQUIRE_EQ(rad_dev_set(rank),RAD_OK); }
};
template<class Work> void dual(Work work) {
    std::exception_ptr errors[2];
    auto invoke = [&](int rank) { try { work(rank); } catch (...) { errors[rank]=std::current_exception(); } };
    std::thread first(invoke,0),second(invoke,1);
    first.join(); second.join();
    for (auto error : errors) if (error) std::rethrow_exception(error);
}
struct Fixture {
    Devices& devices;
    const RadKernelInfo* kernel;
    int rows;
    std::array<RadParam,4> params;
    void* instances[2]{};
    std::unique_ptr<Queue> queues[2];
    std::unique_ptr<Buffer> inputs[2],outputs[2];
    uint64_t calls=0,bytes=0,rejects=0;
    Fixture(Devices& selected,const RadKernelInfo* row,int count,unsigned dtype,bool pitched)
        : devices(selected),kernel(row),rows(count),params{RAD_INT("world_size",2),
          RAD_INT("numel",2048*count),RAD_STR("dtype","f32"),RAD_INT("row",2048)} {
        std::array<RadParam,5> init{params[0],params[1],params[2],params[3],RAD_INT("tp_group",37)};
        int status[2]{RAD_E_STATE,RAD_E_STATE};
        dual([&](int rank) { devices.bind(rank); status[rank]=kernel->init(init.data(),5,rank,2,&instances[rank]); });
        REQUIRE_EQ(status[0],RAD_OK); REQUIRE_EQ(status[1],RAD_OK);
        REQUIRE(instances[0] && instances[1] && instances[0]!=instances[1]);
        for (int rank : {0,1}) {
            devices.bind(rank);
            queues[rank]=std::make_unique<Queue>(0);
            inputs[rank]=std::make_unique<Buffer>(dtype,std::initializer_list<int64_t>{rows,2048},
                RAD_MEM_DEVICE,pitched ? 2048+4*(rank+1) : 0);
            outputs[rank]=std::make_unique<Buffer>(dtype,std::initializer_list<int64_t>{rows,4096},
                RAD_MEM_DEVICE,pitched ? 4096+8*(rank+1) : 0);
        }
    }
    ~Fixture() {
        for (int rank : {0,1}) {
            (void)devices.set(rank); (void)rad_dev_set(rank);
            outputs[rank].reset(); inputs[rank].reset(); queues[rank].reset();
            if (instances[rank]) kernel->fini(instances[rank]);
        }
        (void)devices.set(0); (void)rad_dev_set(0);
    }
    RadArgs arguments(int rank,RadTensor* tensors) {
        RadArgs args{}; args.t=tensors; args.n_t=2; args.p=params.data(); args.n_p=4;
        args.rank=rank; args.world_size=2; args.instance=instances[rank];
        return args;
    }
    void invalid() {
        devices.bind(0);
        std::array<RadTensor,2> tensors{inputs[0]->device,outputs[0]->device};
        auto args=arguments(0,tensors.data());
        auto rejected=[&](int expected) { REQUIRE_EQ(kernel->launch(&args,queues[0]->stream),expected); ++rejects; };
        for (unsigned dtype : {RAD_U16,RAD_I8,RAD_F16,RAD_BF16}) {
            tensors[0].dtype=tensors[1].dtype=dtype; rejected(RAD_E_DTYPE);
        }
        for (unsigned output : {RAD_F32,RAD_I32}) {
            tensors[0].dtype=RAD_U32; tensors[1].dtype=output; rejected(RAD_E_DTYPE);
        }
        tensors={inputs[0]->device,outputs[0]->device};
        tensors[1].shape[1]=4095; rejected(RAD_E_SHAPE);
        tensors={inputs[0]->device,outputs[0]->device};
        tensors[0].stride[1]=2; rejected(RAD_E_STRIDE);
        tensors={inputs[0]->device,outputs[0]->device};
        devices.bind(1); rejected(RAD_E_DEVICE); devices.bind(0);
    }
    void expected_output() {
        for (int rank : {0,1}) {
            auto& output=*outputs[rank];
            std::fill(output.expected.begin(),output.expected.end(),uint8_t(0x5a));
            devices.bind(rank); output.push(queues[rank]->stream);
            REQUIRE_EQ(rad_stream_sync(queues[rank]->stream),RAD_OK);
            for (int sequence=0;sequence<rows;++sequence) for (int source : {0,1})
                std::memcpy(output.data<uint32_t>()+sequence*output.host.stride[0]+source*2048,
                    inputs[source]->data<uint32_t>()+sequence*inputs[source]->host.stride[0],8192);
        }
    }
    void gather(int step) {
        expected_output();
        for (int rank : {0,1}) { devices.bind(rank); REQUIRE_EQ(rad_stream_sync(queues[rank]->stream),RAD_OK); }
        dual([&](int rank) {
            devices.bind(rank);
            if (rank==step%2) std::this_thread::sleep_for(std::chrono::milliseconds(2));
            std::array<RadTensor,2> tensors{inputs[rank]->device,outputs[rank]->device};
            auto args=arguments(rank,tensors.data());
            REQUIRE_EQ(kernel->launch(&args,queues[rank]->stream),RAD_OK);
        });
        for (int rank : {0,1}) {
            devices.bind(rank);
            REQUIRE_EQ(rad_stream_sync(queues[rank]->stream),RAD_OK);
            sample_pair_aql::equal(inputs[rank]->expected,inputs[rank]->read(queues[rank]->stream),"allgather-input",rank);
            sample_pair_aql::equal(outputs[rank]->expected,outputs[rank]->read(queues[rank]->stream),"allgather-output",rank);
            bytes+=inputs[rank]->expected.size()+outputs[rank]->expected.size(); ++calls;
        }
    }
};
uint32_t pattern(int rank,int sequence,int word,int step) {
    if (!(word%2)) return 0x01000001u+uint32_t(rank*0x100000+sequence*4096+word/2+step*7919);
    const uint32_t values[]{0x7fc00123u,0xffc12345u,0x80000000u,0u,0x7f800001u,0xff800000u,0x3f800000u,0xffffffffu};
    return word<17 ? values[(word/2+rank+sequence+step)%8] :
        uint32_t(word*7919u)^uint32_t((step+1)*0x85ebca6bu)^uint32_t(rank*0x6c8e9cf5u+sequence);
}
}

TEST(r3d_allgather_raw32_actual_aql_tp2) {
    using namespace allgather_pair_aql;
    if (!std::getenv("RAD_R3D_REQUIRE_TP2")) SKIP("explicit two-GPU lease required");
    Devices devices;
    Library native("libr3d");
    const auto* kernel=native.ngram("all_gather");
    REQUIRE_EQ(std::strcmp(kernel->name,"r3dh_allgather_tp2_peer_pull"),0);
    auto seed=reinterpret_cast<int (*)(void*,unsigned)>(dlsym(native.handle,"r3dh_allgather_seed"));
    uint64_t calls=0,bytes=0,rejects=0,fixtures=0;
    for (unsigned dtype : {RAD_U32,RAD_I32}) for (int rows : {1,3,8}) for (bool pitched : {false,true}) {
        Fixture fixture(devices,kernel,rows,dtype,pitched);
        fixture.invalid();
        if (seed) for (int rank : {0,1}) {
            devices.bind(rank); REQUIRE_EQ(seed(fixture.instances[rank],0xfffffffcu),RAD_OK);
        }
        for (int step=0;step<8;++step) {
            for (int rank : {0,1}) {
                devices.bind(rank);
                auto& input=*fixture.inputs[rank];
                for (int sequence=0;sequence<rows;++sequence) for (int word=0;word<2048;++word)
                    input.data<uint32_t>()[sequence*input.host.stride[0]+word]=pattern(rank,sequence,word,step);
                input.push(fixture.queues[rank]->stream);
            }
            fixture.gather(step);
        }
        calls+=fixture.calls; bytes+=fixture.bytes; rejects+=fixture.rejects; ++fixtures;
        std::printf("PASS ALLGATHER_RAW32_AQL_CASE rows=%d dtype=%u pitched=%d wrap=%d ranks=2\n",rows,dtype,int(pitched),int(seed!=nullptr));
        std::fflush(stdout);
    }
    std::printf("PASS ALLGATHER_RAW32_AQL fixtures=%llu calls=%llu rejects=%llu exact_bytes=%llu init_threads_joined=1 rank_delays=1 guards=full nan_payloads=1 signed_zero=1 ids_gt_2p24=1\n",
        (unsigned long long)fixtures,(unsigned long long)calls,(unsigned long long)rejects,(unsigned long long)bytes);
}

TEST(r3d_allgather_sample_pipeline_actual_aql_tp2) {
    using namespace allgather_pair_aql;
    if (!std::getenv("RAD_R3D_REQUIRE_TP2")) SKIP("explicit two-GPU lease required");
    Devices devices;
    Library native("libr3d"),reference("libref");
    const auto* kernel=native.ngram("all_gather");
    sample_pair_aql::Counts counts;
    uint64_t gathered=0,bytes=0;
    for (int rows : {1,3,8}) for (int step=0;step<2;++step) {
        Fixture fixture(devices,kernel,rows,RAD_U32,true);
        std::unique_ptr<Buffer> sampling[2];
        for (int rank : {0,1}) {
            devices.bind(rank);
            auto stream=fixture.queues[rank]->stream;
            sampling[rank]=std::make_unique<Buffer>(RAD_U32,std::initializer_list<int64_t>{rows,24},RAD_MEM_DEVICE);
            sample_pair_aql::parameters(*sampling[rank],rows,64,step);
            Buffer logits(RAD_F32,{rows,124160},RAD_MEM_DEVICE,124167);
            Buffer indices(RAD_I32,{rows,1024},RAD_MEM_DEVICE),values(RAD_F32,{rows,1024},RAD_MEM_DEVICE);
            for (int sequence=0;sequence<rows;++sequence) for (int column=0;column<124160;++column)
                logits.data<float>()[sequence*logits.host.stride[0]+column]=
                    float(int((uint32_t(column)*7919u+uint32_t(sequence*71+step*131+rank*23))%4093u)-2046)/256;
            const int offset=0x01000001+rank*124160;
            sample_pair_aql::run(native.ngram("sample_topk"),reference.ngram("sample_topk"),
                {&logits,sampling[rank].get(),&indices,&values,fixture.inputs[rank].get()},
                {integer("M",8),integer("n_vocab",124160),integer("vocab_off",offset)},stream,
                [&](auto& tensors) { sample_pair_aql::topk(tensors,offset); },counts);
        }
        fixture.gather(step);
        for (int rank : {0,1}) {
            devices.bind(rank);
            auto stream=fixture.queues[rank]->stream;
            Buffer indices(RAD_I32,{rows,1024},RAD_MEM_DEVICE),values(RAD_F32,{rows,1024},RAD_MEM_DEVICE);
            Buffer tokens(RAD_I32,{rows},RAD_MEM_DEVICE);
            sample_pair_aql::run(native.ngram("sample_merge_topk"),reference.ngram("sample_merge_topk"),
                {fixture.outputs[rank].get(),sampling[rank].get(),&indices,&values},
                {integer("M",8),integer("n_cand",1024),integer("world_size",2)},stream,
                sample_pair_aql::merge,counts,false,1);
            sample_pair_aql::run(native.ngram("sample_pick"),reference.ngram("sample_pick"),
                {&indices,&values,sampling[rank].get(),&tokens},
                {integer("M",8),integer("n_cand",1024)},stream,sample_pair_aql::pick,counts,false,2);
            for (int sequence=0;sequence<rows;++sequence) REQUIRE(tokens.data<int32_t>()[sequence]>0x01000000);
            sampling[rank].reset();
        }
        gathered+=fixture.calls; bytes+=fixture.bytes;
        std::printf("PASS ALLGATHER_SAMPLE_PIPELINE_AQL rows=%d step=%d ranks=2 topk_device_pairs=1 no_pair_reupload=1 native_merge_pick=1\n",rows,step);
        std::fflush(stdout);
    }
    std::printf("PASS ALLGATHER_SAMPLE_PIPELINE_AQL_TOTAL gather_calls=%llu sample_calls=%llu exact_bytes=%llu ids_gt_2p24=1\n",
        (unsigned long long)gathered,(unsigned long long)counts.calls,(unsigned long long)(bytes+counts.bytes));
}
