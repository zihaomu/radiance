struct Buffer {
    uint8_t* allocation = nullptr;
    size_t bytes;
    std::vector<uint8_t> mirror;
    explicit Buffer(size_t size) : bytes(size), mirror(size + 512, 0xa5) {
        hip_check(hipMalloc(&allocation, mirror.size()));
    }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    ~Buffer() { if (allocation) static_cast<void>(hipFree(allocation)); }
    void* data() const { return allocation + 256; }
    void upload(const void* source, hipStream_t stream) {
        if (source) std::memcpy(mirror.data() + 256, source, bytes);
        hip_check(hipMemcpyAsync(allocation, mirror.data(), mirror.size(), hipMemcpyHostToDevice, stream));
    }
    std::vector<uint8_t> read(hipStream_t stream, bool unchanged) {
        std::vector<uint8_t> result(mirror.size());
        hip_check(hipMemcpyAsync(result.data(), allocation, result.size(), hipMemcpyDeviceToHost, stream));
        hip_check(hipStreamSynchronize(stream));
        expect(std::equal(result.begin(), result.begin() + 256, mirror.begin()), "leading guard");
        expect(std::equal(result.end() - 256, result.end(), mirror.end() - 256), "trailing guard");
        if (unchanged) expect(result == mirror, "input mutated");
        return result;
    }
};
struct Stream {
    hipStream_t value = nullptr;
    Stream() { hip_check(hipStreamCreateWithFlags(&value, hipStreamNonBlocking)); }
    ~Stream() { if (value) static_cast<void>(hipStreamDestroy(value)); }
};
void gpu_case(int tokens, int active, bool reverse, bool stacked, int columns = 16,
              const std::vector<Weight>* actual = nullptr, bool ties = false,
              int even_override = 0, int odd_override = 0, bool sampled = false) {
    const int even = even_override ? even_override : reverse ? 256 : 384;
    const int odd = odd_override ? odd_override : reverse ? 384 : 256;
    auto params = parameters(tokens, columns, even, odd);
    const int rows = tokens * 10, inner_max = std::max(even, odd);
    const int activation_pitch = inner_max + 13, scale_pitch = inner_max / 128 + 2, output_pitch = columns + 7;
    std::vector<Weight> weights;
    if (actual) weights = *actual;
    else for (int index = 0; index < 4; ++index)
        weights.emplace_back(columns, index % 2 ? odd : even, unsigned(index * 31 + 1));
    if (ties) for (auto& weight : weights) {
        std::fill(weight.codes.begin(), weight.codes.end(), 0);
        std::fill(weight.scales.begin(), weight.scales.end(), 56);
        weight.table.fill(1.0f);
    }
    expect(weights.size() == 4, "four expert fixtures");
    for (int index = 0; index < 4; ++index)
        expect(weights[index].rows == columns && weights[index].inner == (index % 2 ? odd : even),
               "fixture parity geometry");
    std::vector<std::vector<int32_t>> buckets(512);
    for (int flat = 0; flat < rows; ++flat) {
        if (!active || flat % 19 == 0) continue;
        const int chosen = (flat * 73 + flat / 11) % active;
        const int expert = active == 1 ? 511 : chosen * 511 / (active - 1);
        buckets[expert].push_back(rows - 1 - flat);
    }
    std::vector<int32_t> sorted(rows, -1), offsets(513);
    int count = 0;
    for (int expert = 0; expert < 512; ++expert) {
        offsets[expert] = count;
        for (int32_t flat : buckets[expert]) sorted[count++] = flat;
    }
    offsets[512] = count;
    const int active_count = int(std::count_if(buckets.begin(), buckets.end(),
        [](const std::vector<int32_t>& bucket) { return !bucket.empty(); }));
    Stream stream;
    std::vector<std::unique_ptr<Buffer>> owned;
    auto make = [&](size_t bytes, const void* source) -> Buffer* {
        owned.push_back(std::make_unique<Buffer>(bytes));
        owned.back()->upload(source, stream.value);
        return owned.back().get();
    };
    std::array<Buffer*, 4> code_device{}, scale_device{};
    std::array<std::vector<uint8_t>, 4> code_blobs, scale_blobs;
    for (int index = 0; index < 4; ++index) {
        const int operand = index % 2 ? 6 : 2;
        code_blobs[index] = weights[index].encode(params, operand);
        scale_blobs[index] = weights[index].encode(params, operand + 1);
        code_device[index] = make(code_blobs[index].size(), code_blobs[index].data());
        scale_device[index] = make(scale_blobs[index].size(), scale_blobs[index].data());
    }
    std::array<RadTensor, 9> tensors{};
    for (int parity = 0; parity < 2; ++parity) {
        for (int scale = 0; scale < 2; ++scale) {
            const size_t size = scale ? scale_blobs[parity].size() : code_blobs[parity].size();
            Buffer* device = nullptr;
            if (stacked) {
                std::vector<uint8_t> stack(256 * size);
                for (int entry_index = 0; entry_index < 256; ++entry_index) {
                    const int fixture = (entry_index % 2) * 2 + parity;
                    const auto& blob = scale ? scale_blobs[fixture] : code_blobs[fixture];
                    std::memcpy(stack.data() + entry_index * size, blob.data(), size);
                }
                device = make(stack.size(), stack.data());
            } else {
                std::array<void*, 256> pointers{};
                for (int entry_index = 0; entry_index < 256; ++entry_index) {
                    const int fixture = (entry_index % 2) * 2 + parity;
                    pointers[entry_index] = scale ? scale_device[fixture]->data() : code_device[fixture]->data();
                }
                device = make(sizeof(pointers), pointers.data());
            }
            RadTensor tensor{};
            tensor.data = device->data(); tensor.dtype = scale ? scales_type : codes_type;
            tensor.rank = 2; tensor.shape[0] = 256; tensor.shape[1] = int64_t(size);
            tensor.stride[0] = stacked ? int64_t(size) : 0; tensor.stride[1] = 1;
            tensors[(parity ? 6 : 2) + scale] = tensor;
        }
    }
    Buffer* sorted_device = make(sorted.size() * sizeof(int32_t), sorted.data());
    Buffer* offsets_device = make(offsets.size() * sizeof(int32_t), offsets.data());
    tensors[4] = vector_tensor(sorted_device->data(), rows);
    tensors[5] = vector_tensor(offsets_device->data(), 513);
    std::vector<int8_t> activation(size_t(rows) * activation_pitch);
    std::vector<float> activation_scale(size_t(rows) * scale_pitch);
    Buffer activation_device(activation.size()), activation_scale_device(activation_scale.size() * sizeof(float));
    Buffer output_device(size_t(rows) * output_pitch * sizeof(uint16_t));
    tensors[0] = matrix(activation_device.data(), RAD_I8, rows, inner_max, activation_pitch);
    tensors[1] = matrix(activation_scale_device.data(), RAD_F32, rows, inner_max / 128, scale_pitch);
    tensors[8] = matrix(output_device.data(), RAD_BF16, rows, columns, output_pitch);
    RadArgs arguments{};
    arguments.t = tensors.data();
    arguments.n_t = int(tensors.size());
    arguments.p = params.data();
    arguments.n_p = int(params.size());
    arguments.rank = reverse ? 1 : 0;
    arguments.world_size = 2;
    arguments.kv_used_max_h = -1;
    uint64_t previous_checksum = 0;
    for (int repeat = 0; repeat < 2; ++repeat) {
        for (int row = 0; row < rows; ++row) {
            for (int position = 0; position < activation_pitch; ++position)
                activation[size_t(row) * activation_pitch + position] = ties
                    ? int8_t(position == 0 ? 1 : 0)
                    : int8_t((row * 17 + position * 13 + repeat * 47) % 256 - 128);
            for (int group = 0; group < scale_pitch; ++group)
                activation_scale[size_t(row) * scale_pitch + group] = ties
                    ? (repeat ? 8288.0f : 8224.0f)
                    : float((row * 7 + group * 11 + repeat * 3) % 29 + 1) * 0.003171f;
        }
        activation_device.upload(activation.data(), stream.value);
        activation_scale_device.upload(activation_scale.data(), stream.value);
        output_device.upload(nullptr, stream.value);
        if (!repeat) {
            hipGraph_t graph = nullptr;
            hip_check(hipStreamBeginCapture(stream.value, hipStreamCaptureModeThreadLocal));
            expect(r3dh_moe_w4nl_launch(&arguments, reinterpret_cast<RadStream>(stream.value)) == RAD_OK,
                "stream capture launch");
            hip_check(hipStreamEndCapture(stream.value, &graph));
            size_t node_count = 0;
            hip_check(hipGraphGetNodes(graph, nullptr, &node_count));
            expect(node_count == 1, "kernel must use supplied nondefault stream");
            hip_check(hipGraphDestroy(graph));
        }
        expect(r3dh_moe_w4nl_launch(&arguments, reinterpret_cast<RadStream>(stream.value)) == RAD_OK, "GPU launch");
        const auto result = output_device.read(stream.value, false);
        uint64_t checksum = 1469598103934665603ull;
        int expert = 0;
        for (int row = 0; row < rows; ++row) {
            while (expert < 511 && offsets[expert + 1] <= row) ++expert;
            const bool valid = sorted[row] >= 0 && row < offsets[512];
            for (int column = 0; column < output_pitch; ++column) {
                uint16_t got;
                std::memcpy(&got, result.data() + 256 + (size_t(row) * output_pitch + column) * 2, 2);
                checksum = (checksum ^ got) * 1099511628211ull;
                const bool compare = !sampled || column >= columns || column == 0
                    || column == columns / 2 || column == columns - 1 || row < 2 || row == rows - 1;
                if (!compare) continue;
                const uint16_t expected = column >= columns ? 0xa5a5 : !valid ? 0
                    : oracle(weights[expert % 4], activation.data() + size_t(row) * activation_pitch,
                             activation_scale.data() + size_t(row) * scale_pitch, column);
                if (got != expected) {
                    std::fprintf(stderr, "mismatch M=%d active=%d reverse=%d stacked=%d repeat=%d row=%d col=%d expert=%d got=%04x want=%04x\n",
                        tokens, active, reverse, stacked, repeat, row, column, expert, got, expected);
                    throw std::runtime_error("independent CPU oracle mismatch");
                }
                ++outputs;
            }
        }
        if (repeat && count) expect(checksum != previous_checksum, "changed input must change output");
        previous_checksum = checksum;
        static_cast<void>(activation_device.read(stream.value, true));
        static_cast<void>(activation_scale_device.read(stream.value, true));
    }
    for (auto& buffer : owned) static_cast<void>(buffer->read(stream.value, true));
    const auto saved = tensors[2];
    tensors[2].dtype = RAD_I8;
    expect(r3dh_moe_w4nl_launch(&arguments, reinterpret_cast<RadStream>(stream.value)) == RAD_E_DTYPE, "reject table alias");
    tensors[2] = saved; tensors[2].stride[0] = -1;
    expect(r3dh_moe_w4nl_launch(&arguments, reinterpret_cast<RadStream>(stream.value)) == RAD_E_STRIDE, "reject negative table stride");
    tensors[2] = saved; tensors[2].rank = 3;
    expect(r3dh_moe_w4nl_launch(&arguments, reinterpret_cast<RadStream>(stream.value)) == RAD_E_SHAPE, "reject table rank");
    tensors[2] = saved;
    ++gpu_cases;
    std::printf("W4NL_CASE_PASS kind=%s M=%d N=%d K=%d/%d active_requested=%d active=%d live_rows=%d stacked=%d sampled=%d repeats=2\n",
        actual ? "model_slice" : ties ? "bf16_ties" : "synthetic", tokens, columns, even, odd,
        active, active_count, count, stacked, sampled);
    std::fflush(stdout);
}
void gpu_tests() {
    int devices = 0;
    hip_check(hipGetDeviceCount(&devices));
    expect(devices == 1, "exactly one GPU must be exposed");
    hipDeviceProp_t properties{};
    hip_check(hipGetDeviceProperties(&properties, 0));
    expect(!std::strncmp(properties.gcnArchName, "gfx1100", 7), "gfx1100 required");
    std::printf("W4NL_DEVICE %s %s\n", properties.name, properties.gcnArchName);
    const int cases[][2] = {{1, 1}, {1, 10}, {3, 2}, {17, 10}, {128, 64},
        {511, 512}, {1024, 7}, {2048, 512}, {2048, 0}};
    for (int index = 0; index < 9; ++index)
        for (int reverse = 0; reverse < 2; ++reverse)
            gpu_case(cases[index][0], cases[index][1], reverse, (index + reverse) % 2);
    gpu_case(1, 10, false, false, 16, nullptr, true);
    gpu_case(3, 10, false, true, 80, nullptr, false, 512, 128);
    gpu_case(1, 10, false, false, 256, nullptr, false, 128, 128);
}
