namespace {
void batch_contracts() {
    batch_registry_checks();
    for (int64_t count : {1, 3, 16, 32, 33, 64, 128, 256, 512, 2048}) {
        std::vector<RadParam> params{RAD_INT("M", count), RAD_INT("n", 2560), RAD_INT("hc", 4), RAD_INT("ngroup", 4)};
        RadArgs args{}; args.p = params.data(); args.n_p = int(params.size());
        expect(r3dh_mtp_ple_mtp_enter_scratch(&args) == count * 35840, "exact batch scratch extent");
        for (const char* op : {"mtp_enter", "ple_gate"}) {
            RadOpdDesc description{};
            expect(r3dh_mtp_ple_shape(op, params.data(), int(params.size()), 0, &description) == RAD_OK &&
                description.shape[0] == count && description.shape[1] == 10240, "large batch owning shape");
        }
    }
    std::printf("BATCH_CONTRACT_PASS batches=10 scratch_M2048=73400320\n");
}
void batch_grid_case(Model& model, int64_t count) {
    constexpr int64_t width = 1152, side = 48;
    Blob table(side * side * width, RAD_BF16), output(count * (width + 8), RAD_BF16), coordinates(4 * (count + 7), RAD_I32);
    const auto* entry = model.find("v.pos.weight"); expect(entry, "actual vision grid table exists");
    const auto& encoding = model.encodings[entry->enc];
    expect(rad_enc_is(&encoding, "plain") && !encoding.transform[0] && encoding.n_planes == 1 &&
        entry->rank == 2 && entry->shape[0] == side * side && entry->shape[1] == width, "actual grid n1152 side48 schema");
    model.plane(*entry, "codes", table);
    for (int repeat = 0; repeat < 3; ++repeat) {
        fill(output, 13 + repeat);
        for (int64_t row = 0; row < count; ++row) {
            const int32_t extents[]{1, 2, 47, 48, 49, 1152, 2048};
            for (int axis = 0; axis < 2; ++axis) {
                const int32_t extent = extents[(row + axis * 3 + repeat) % 7];
                const int32_t positions[]{0, extent - 1, -3, extent + 2, extent / 2, int32_t(row % extent)};
                coordinates.data<int32_t>()[axis * (count + 7) + row] = positions[(row + repeat + axis) % 6];
                coordinates.data<int32_t>()[(axis + 2) * (count + 7) + row] = extent;
            }
        }
        const std::string label = "grid_actual M=" + std::to_string(count) + " n=1152 side=48 changed=" + std::to_string(repeat);
        run(label.c_str(), r3dh_mtp_ple_grid_embed, ref_grid_embed,
            {RAD_INT("M", count), RAD_INT("n", width), RAD_INT("side", side)},
            {tensor(output, {count, width}, width + 8), tensor(table, {side * side, width}), tensor(coordinates, {4, count}, count + 7)});
    }
}
void batch_suite(const char* path) {
    batch_mode = true; gpu = true;
    int visible = 0; hipcheck(hipGetDeviceCount(&visible)); expect(visible == 1, "exactly one visible batch GPU");
    hipDeviceProp_t properties{}; hipcheck(hipGetDeviceProperties(&properties, 0));
    expect(!std::strncmp(properties.gcnArchName, "gfx1100", 7) &&
        (properties.pciBusID == 0xc3 || properties.pciBusID == 0x83), "authorized gfx1100 PCI c3 or 83");
    hipcheck(hipStreamCreateWithFlags(&queue, hipStreamNonBlocking));
    std::printf("BATCH_DEVICE %s arch=%s pci_bus=%x nondefault_stream=1 strict_fp=1 input_generator=index_hash32\n",
        properties.name, properties.gcnArchName, properties.pciBusID);
    batch_contracts();
    for (int64_t count : {33, 64, 128, 512, 2048}) mtp_case(count, 128, 4);
    Model model(path); current_model = &model;
    mtp_case(64, 2560, 4);
    for (int64_t count : {128, 512, 2048}) mtp_case(count, 2560, 4, true);
    for (int64_t count : {33, 64, 128, 512, 2048}) gate_case(count);
    for (int64_t count : {1, 3, 16, 256, 2048}) batch_grid_case(model, count);
    current_model = nullptr;
    hipcheck(hipStreamDestroy(queue)); queue = nullptr;
    std::printf("MTP_PLE_BATCH_PASS cases=%d compared_bytes=%llu sampled_not_full=1\n", cases,
        static_cast<unsigned long long>(comparisons));
}
}
