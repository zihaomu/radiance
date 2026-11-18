void queued_lifetime_case() {
    constexpr int launches = 1024, row = 44, maximum_rows = 17;
    constexpr int segment = maximum_rows * row * 8 + 64;
    auto params = parameters(row * maximum_rows,row,"f32");
    void* instances[2]{};
    int status[2]{};
    std::thread init_first([&] { hipcheck(hipSetDevice(0)); status[0] = r3dh_allgather_init(params.data(),4,0,2,&instances[0]); });
    std::thread init_second([&] { hipcheck(hipSetDevice(1)); status[1] = r3dh_allgather_init(params.data(),4,1,2,&instances[1]); });
    init_first.join(); init_second.join();
    require(status[0] == RAD_OK && status[1] == RAD_OK,"handoff init");
    std::exception_ptr failures[2];
    std::barrier barrier(2);
    auto rank_main = [&](int rank) {
        try {
            hipcheck(hipSetDevice(rank));
            Context context; context.instance = instances[rank];
            hipcheck(hipStreamCreateWithFlags(&context.stream,hipStreamNonBlocking));
            Allocation input(maximum_rows * row * 4 + 64),output(size_t(segment)*launches),generation(16);
            hipcheck(hipMemsetAsync(output.pointer,0xa5,size_t(segment)*launches,context.stream));
            hipcheck(hipMemsetAsync(generation.pointer,0,16,context.stream));
            hipcheck(hipStreamSynchronize(context.stream));
            require(r3dh_allgather_seed(context.instance,0xfffffff0u) == RAD_OK,"queued wrap seed");
            barrier.arrive_and_wait();
            for (int launch=0;launch<launches;++launch) {
                const int rows = launch%3 == 0 ? 1 : launch%3 == 1 ? 3 : maximum_rows;
                advance<<<1,1,0,context.stream>>>((unsigned*)generation.pointer);
                fill<<<2,256,0,context.stream>>>(input.pointer+32,(unsigned*)generation.pointer,row*4,rows,row*4,rank);
                if (rank == 1 && launch%7 == 0) delay<<<1,1,0,context.stream>>>(100000ull);
                std::array<RadTensor,2> tensors{tensor(input.pointer+32,RAD_F32,rows,row,row),
                    tensor(output.pointer+size_t(segment)*launch+32,RAD_F32,rows,row*2,row*2)};
                RadArgs args{}; args.t=tensors.data(); args.n_t=2; args.p=params.data(); args.n_p=4;
                args.world_size=2; args.rank=rank; args.instance=context.instance;
                require(r3dh_allgather_launch(&args,(RadStream)context.stream)==RAD_OK,"queued launch");
                ++calls;
            }
            std::vector<unsigned char> actual(size_t(segment)*launches);
            hipcheck(hipMemcpyAsync(actual.data(),output.pointer,actual.size(),hipMemcpyDeviceToHost,context.stream));
            hipcheck(hipStreamSynchronize(context.stream));
            unsigned error=99;
            require(r3dh_allgather_status(context.instance,&error)==RAD_OK && !error,"queued status");
            r3dh_allgather_fini(context.instance); context.instance=nullptr;
            for (int launch=0;launch<launches;++launch) {
                const int rows = launch%3 == 0 ? 1 : launch%3 == 1 ? 3 : maximum_rows;
                for (int offset=0;offset<segment;++offset) {
                    unsigned char expected=0xa5;
                    if (offset>=32 && offset<32+rows*row*8) {
                        const int position=offset-32, column=position%(row*8), source_rank=column/(row*4);
                        expected=pattern(uint64_t(position/(row*8))*row*4+column%(row*4),launch+1,source_rank);
                    }
                    ++comparisons;
                    require(actual[size_t(segment)*launch+offset]==expected,"queued independent row oracle/guards");
                }
            }
            ++cases;
        } catch (...) { failures[rank]=std::current_exception(); barrier.arrive_and_drop(); }
    };
    std::thread run_first(rank_main,0),run_second(rank_main,1);
    run_first.join(); run_second.join();
    for (const auto& failure:failures) if (failure) std::rethrow_exception(failure);
    std::printf("ALLGATHER_QUEUED_LIFETIME_PASS launches_per_rank=1024 varying_rows=1,3,17 init_threads_exited=1 wrap=1 delayed_rank=1\n");
}
