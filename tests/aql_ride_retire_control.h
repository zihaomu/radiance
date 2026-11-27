namespace {
struct RideAuxStream {
    RadStream s = nullptr;
    RideAuxStream() { rad_xp_stream_create(&s); }
    ~RideAuxStream() { if (s) rad_stream_destroy(s); }
};
}

TEST(early_ride_wait_keeps_waiter_until_consumed) {
    if (std::strcmp(rad::device_backend_name(), "aql")) SKIP("AQL card only");
    Stream producer;
    RideAuxStream auxiliary;
    REQUIRE(producer.s && auxiliary.s);
    DevBuf<uint8_t> src(4096, RAD_MEM_DEVICE), dst(4096, RAD_MEM_HOST_PINNED);
    REQUIRE(src && dst);
    std::memset(dst.p, 0, 4096);
    void* gate = nullptr;
    void* token = nullptr;
    REQUIRE_EQ(rad_stream_gate(auxiliary.s, &gate), RAD_OK);
    REQUIRE_EQ(rad_ride(producer.s, &token), RAD_OK);
    REQUIRE_EQ(rad::aql::ride_wait_keep(auxiliary.s, token), RAD_OK);
    CHECK_OK(rad_memset_async(src, 0xa5, 4096, producer.s));
    CHECK_OK(rad_memcpy_async(dst, src, 4096, auxiliary.s));
    CHECK_OK(rad_stream_sync(producer.s));
    CHECK(!rad::aql::ride_reusable(token));
    CHECK_EQ(dst.p[0], 0);
    CHECK_OK(rad_gate_open(gate));
    CHECK_OK(rad_stream_sync(auxiliary.s));
    CHECK(rad::aql::ride_reusable(token));
    rad_ride_release(token, nullptr, 0);
    for (int offset = 0; offset < 4096; ++offset) CHECK_EQ(dst.p[offset], 0xa5);
}

TEST(perf_ride_retire_control) {
    const char* arm = std::getenv("RAD_TEST_RIDE_RETIRE_ARM");
    if (!arm) SKIP("opt-in early/late ride retirement control");
    REQUIRE_EQ(std::strcmp(rad::device_backend_name(), "aql"), 0);
    const bool immediate = !std::strcmp(arm, "early-now");
    const bool late = !std::strcmp(arm, "late-held");
    REQUIRE(immediate || late || !std::strcmp(arm, "early-held"));
    constexpr int slots = 48, bytes = 16384, warm = 8, measured = 24;
    Stream producer;
    RideAuxStream auxiliary;
    DevBuf<uint8_t> src(slots * bytes, RAD_MEM_DEVICE), dst(slots * bytes, RAD_MEM_DEVICE);
    DevBuf<uint8_t> scratch(bytes, RAD_MEM_DEVICE), result(slots * bytes, RAD_MEM_HOST_PINNED);
    REQUIRE(producer.s && auxiliary.s && src && dst && scratch && result);
#if defined(__linux__)
    if (const char* pin = std::getenv("RAD_TEST_RIDE_RETIRE_CPU")) {
        char* end = nullptr;
        const long cpu = std::strtol(pin, &end, 10);
        REQUIRE(end != pin && !*end && cpu >= 0 && cpu < CPU_SETSIZE);
        cpu_set_t cpus;
        CPU_ZERO(&cpus);
        CPU_SET(cpu, &cpus);
        REQUIRE_EQ(pthread_setaffinity_np(pthread_self(), sizeof(cpus), &cpus), 0);
        REQUIRE_EQ(sched_getcpu(), int(cpu));
        std::printf("RIDE_SETUP cpu=%ld auxiliary=xp slots=%d bytes=%d warm=%d measured=%d\n", cpu, slots, bytes, warm, measured);
    }
#endif
    using Clock = std::chrono::steady_clock;
    const auto us = [](auto before, auto after) { return std::chrono::duration<double, std::micro>(after - before).count(); };
    for (int pass = 0; pass < warm + measured; ++pass) {
        void* held[slots]{};
        REQUIRE_EQ(rad_memset_async(dst, 0, slots * bytes, auxiliary.s), RAD_OK);
        REQUIRE_EQ(rad_stream_sync(auxiliary.s), RAD_OK);
#ifdef RAD_AQL_RETIRE_DIAGNOSTICS
        rad::aql::retire_count_begin();
#endif
        const auto start = Clock::now();
        for (int slot = 0; slot < slots; ++slot) {
            void* token = nullptr;
            const int offset = slot * bytes, value = 1 + (slot + 7 * pass) % 251;
            REQUIRE_EQ(rad_ride(producer.s, &token), RAD_OK);
            if (!late) REQUIRE_EQ(rad::aql::ride_wait_keep(auxiliary.s, token), RAD_OK);
            if (immediate) rad_ride_release(token, nullptr, 0);
            else held[slot] = token;
            REQUIRE_EQ(rad_memset_async(src.p + offset, value, bytes, producer.s), RAD_OK);
            REQUIRE_EQ(rad_memset_async(scratch, value, bytes, producer.s), RAD_OK);
            if (late) REQUIRE_EQ(rad::aql::ride_wait_keep(auxiliary.s, token), RAD_OK);
            REQUIRE_EQ(rad_memcpy_async(dst.p + offset, src.p + offset, bytes, auxiliary.s), RAD_OK);
        }
        const auto submitted = Clock::now();
        for (void* token : held) if (token) rad_ride_release(token, nullptr, 0);
        const auto retired = Clock::now();
#ifdef RAD_AQL_RETIRE_DIAGNOSTICS
        const auto counts = rad::aql::retire_count_end();
        REQUIRE_EQ(counts.waits, uint64_t(slots));
        REQUIRE_EQ(counts.retirements, uint64_t(slots));
        if (pass >= warm) {
            REQUIRE_EQ(counts.creates, uint64_t(0));
            if (!immediate) {
                REQUIRE_EQ(counts.fresh_reads, uint64_t(0));
                REQUIRE_EQ(counts.fresh_queue_reads, uint64_t(0));
            }
        }
#endif
        REQUIRE_EQ(rad_stream_sync(producer.s), RAD_OK);
        REQUIRE_EQ(rad_stream_sync(auxiliary.s), RAD_OK);
        const auto finished = Clock::now();
        REQUIRE_EQ(rad_memcpy_async(result, dst, slots * bytes, auxiliary.s), RAD_OK);
        REQUIRE_EQ(rad_stream_sync(auxiliary.s), RAD_OK);
        for (int slot = 0; slot < slots; ++slot) {
            const uint8_t wanted = uint8_t(1 + (slot + 7 * pass) % 251);
            const uint8_t* row = result.p + slot * bytes;
            bool correct = true;
            for (int offset = 0; offset < bytes; ++offset) correct &= row[offset] == wanted;
            REQUIRE(correct);
        }
        if (pass < warm) continue;
        std::printf("RIDE_ARM %s pass=%d waits=%d submit_us=%.3f retired_us=%.3f wall_us=%.3f",
                    arm, pass - warm, slots, us(start,submitted), us(start,retired), us(start,finished));
#ifdef RAD_AQL_RETIRE_DIAGNOSTICS
        std::printf(" gets=%llu creates=%llu retires=%llu value_reads=%llu fresh_reads=%llu fresh_nonzero=%llu queue_reads=%llu fresh_queue_reads=%llu peak=%llu",
                    (unsigned long long)counts.gets, (unsigned long long)counts.creates, (unsigned long long)counts.retirements,
                    (unsigned long long)counts.value_reads, (unsigned long long)counts.fresh_reads, (unsigned long long)counts.fresh_nonzero,
                    (unsigned long long)counts.queue_reads, (unsigned long long)counts.fresh_queue_reads, (unsigned long long)counts.retiring_peak);
#endif
        std::printf("\n");
    }
}