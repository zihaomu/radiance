TEST(r3d_tp2_verify_head_four_rows) {
    using namespace joined_head_test;
    if (!device_available()) SKIP("gfx1100 required");
    struct Restore {
        const char* key;
        bool present;
        std::string value;
        explicit Restore(const char* name) : key(name), present(std::getenv(key) != nullptr),
            value(present ? std::getenv(key) : "") {}
        ~Restore() { if (present) setenv(key, value.c_str(), 1); else unsetenv(key); }
    } precision("RAD_PRECISION"), wave("RAD_LOGITS_WAVE"), verify("RAD_MTP_VERIFY_HEAD4");
    constexpr int columns = 124160, reduction = 2560, rows = 4;
    constexpr int xpitch = reduction + 16, ypitch = columns + 16;
    Api api;
    Queue compute(0);
    Model model("/home/zihaomu/bigssd2/radiance_workspace/models/qwen3.8-next-flash-iqr-v4.rad");
    int shards = 0;
    for (const auto& entry : model.entries) {
        if (model.name(entry) != "output.weight") continue;
        REQUIRE_EQ(entry.shape[0], 2 * columns); REQUIRE_EQ(entry.shape[1], reduction);
        REQUIRE_EQ(entry.n_planes, 2u);
        for (int rank : {0, 1}) {
            Fixture fixture(api, columns, reduction, RAD_I8, RAD_BF16, 1, 128);
            fixture.encoding = model.encodings.at(entry.enc);
            for (unsigned plane : {0u, 1u}) {
                auto& payload = plane ? fixture.scale : fixture.codes;
                const auto& location = model.planes.at(entry.plane_first + plane);
                REQUIRE_EQ(location.bytes, payload.size() * 2);
                model.read(location.offset + uint64_t(rank) * payload.size(), payload.data(), payload.size());
            }
            fixture.pack(api);
            Storage weight(fixture.joined.size(), RAD_MEM_DEVICE);
            std::memcpy(weight.expected.data() + 64, fixture.joined.data(), fixture.joined.size());
            weight.push(compute.stream);
            Buffer input(RAD_BF16, {rows, reduction}, RAD_MEM_DEVICE, xpitch);
            Buffer output(RAD_F32, {rows, columns}, RAD_MEM_DEVICE, ypitch);
            for (int row = 0; row < rows; ++row) for (int pos = 0; pos < reduction; ++pos)
                input.data<uint16_t>()[row * xpitch + pos] = rad_f32_to_bf16(float((pos * (47 + row * 2) + 71 * row) % 257 - 128) / 64);
            input.push(compute.stream);
            auto run = [&](bool split, int world) {
                output.push(compute.stream);
                for (int row = 0; row < (split ? rows : 1); ++row) {
                    RadTensor tensors[]{input.device, matrix(weight.device(), fixture.stored.dtype, columns, reduction), output.device};
                    tensors[0].data = static_cast<uint16_t*>(tensors[0].data) + row * xpitch;
                    tensors[2].data = static_cast<float*>(tensors[2].data) + row * ypitch;
                    tensors[0].shape[0] = tensors[2].shape[0] = split ? 1 : rows;
                    RadParam params[]{integer("M", 64), integer("n_vocab", columns), integer("n_embd", reduction)};
                    RadArgs args{}; args.t = tensors; args.n_t = 3; args.p = params; args.n_p = 3; args.world_size = world;
                    REQUIRE_EQ(api.launch(&args, compute.stream), RAD_OK);
                }
            };
            setenv(precision.key, "tolerance", 1); setenv(wave.key, "1", 1); setenv(verify.key, "0", 1);
            run(true, 2); const auto one = output.read(compute.stream);
            run(false, 2); const auto ordered = output.read(compute.stream);
            REQUIRE(one != ordered);
            auto guarded = output.expected;
            for (int row = 0; row < rows; ++row) {
                const size_t offset = 64 + size_t(row) * ypitch * sizeof(float);
                std::copy_n(one.data() + offset, size_t(columns) * sizeof(float), guarded.data() + offset);
            }
            REQUIRE(guarded == one);
            setenv(verify.key, "1", 1); run(false, 2); REQUIRE(output.read(compute.stream) == one);
            run(false, 1); REQUIRE(output.read(compute.stream) == ordered);
            setenv(precision.key, "exact", 1); run(false, 2); REQUIRE(output.read(compute.stream) == ordered);
            REQUIRE(input.read(compute.stream) == input.expected);
            ++shards;
        }
    }
    REQUIRE_EQ(shards, 2);
}

#include "r3d_mtp_hc_four.h"
