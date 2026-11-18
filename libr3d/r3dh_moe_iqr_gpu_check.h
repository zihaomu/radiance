struct Device {
    uint8_t* allocation = nullptr;
    size_t bytes = 0;
    void allocate(size_t count) {
        bytes = count;
        hip_check(hipMalloc(&allocation, bytes + 512));
        hip_check(hipMemset(allocation, 0xa5, bytes + 512));
    }
    void* data() const { return allocation + 256; }
    void upload(const void* input, size_t count, size_t offset = 0) {
        expect(offset <= bytes && count <= bytes - offset, "bounded upload");
        hip_check(hipMemcpy(allocation + 256 + offset, input, count, hipMemcpyHostToDevice));
    }
    void unchanged(const void* expected, size_t count) const {
        std::vector<uint8_t> actual(count);
        hip_check(hipMemcpy(actual.data(), data(), count, hipMemcpyDeviceToHost));
        expect(!std::memcmp(actual.data(), expected, count), "device input unchanged");
    }
    void guards() const {
        std::array<uint8_t, 256> prefix{}, suffix{};
        hip_check(hipMemcpy(prefix.data(), allocation, 256, hipMemcpyDeviceToHost));
        hip_check(hipMemcpy(suffix.data(), allocation + 256 + bytes, 256, hipMemcpyDeviceToHost));
        for (unsigned index = 0; index < 256; ++index)
            expect(prefix[index] == 0xa5 && suffix[index] == 0xa5, "device allocation guards");
    }
    ~Device() { if (allocation) static_cast<void>(hipFree(allocation)); }
};
struct Stream {
    hipStream_t stream = nullptr;
    Stream() { hip_check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking)); }
    ~Stream() { if (stream) static_cast<void>(hipStreamDestroy(stream)); }
};
__global__ void decode_all(const uint8_t* blob, int8_t* output, int elements) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    const auto* header = reinterpret_cast<const Header*>(blob);
    output[index] = mad(state(blob + sizeof(Header), 2560, index / 2560, index % 2560),
                        header->multiplier, header->increment);
}
void gpu_decode_case(Fixture& fixture, std::vector<RadParam>& params, int operand) {
    const auto packed = fixture.encode(params, operand);
    const auto expected = fixture.decode();
    Device input, output;
    input.allocate(packed.size());
    output.allocate(expected.size());
    input.upload(packed.data(), packed.size());
    Stream stream;
    hipLaunchKernelGGL(decode_all, dim3((expected.size() + 255) / 256), dim3(256), 0,
        stream.stream, static_cast<const uint8_t*>(input.data()), static_cast<int8_t*>(output.data()), expected.size());
    hip_check(hipGetLastError());
    hip_check(hipStreamSynchronize(stream.stream));
    output.unchanged(expected.data(), expected.size());
    input.unchanged(packed.data(), packed.size());
    input.guards(); output.guards();
    gpu_decoded_values += expected.size();
    std::printf("PASS gpu_decode N=%d elements=%zu exact guards\n", fixture.rows, expected.size());
}
void gpu_identity() {
    const char* visible = std::getenv("ROCR_VISIBLE_DEVICES");
    const char* hip_visible = std::getenv("HIP_VISIBLE_DEVICES");
    expect(visible && *visible && !std::strchr(visible, ',') && hip_visible && !std::strcmp(hip_visible, "0"),
           "single physical GPU and HIP0 environment required");
    int count = 0;
    hip_check(hipGetDeviceCount(&count));
    expect(count == 1, "exactly one visible GPU");
    hip_check(hipSetDevice(0));
    hipDeviceProp_t properties{};
    hip_check(hipGetDeviceProperties(&properties, 0));
    expect(std::strncmp(properties.gcnArchName, "gfx1100", 7) == 0, "native gfx1100 required");
    char pci[64]{};
    hip_check(hipDeviceGetPCIBusId(pci, sizeof(pci), 0));
    std::printf("gpu physical=%s logical=0 arch=%s pci=%s scratch_bytes=0\n", visible, properties.gcnArchName, pci);
}
RadTensor weight_table(void* data, uint32_t type, int rows, bool stacked, size_t bytes) {
    RadTensor result{};
    result.data = data; result.dtype = type; result.rank = 3;
    result.shape[0] = 256; result.shape[1] = rows;
    result.shape[2] = type == weight_dtype ? 2560 : 20;
    result.stride[0] = stacked ? bytes : 0;
    result.stride[1] = result.shape[2]; result.stride[2] = 1;
    return result;
}
void gpu_case(const char* label, std::array<Fixture, 4>& fixtures,
              int tokens, int scenario, bool stacked, int band = 0, bool logical = false) {
    const int width[2] = {fixtures[0].rows, fixtures[1].rows};
    const int output_width = std::max(width[0], width[1]);
    const int total = tokens * 10, input_pitch = 2573, scale_pitch = 27;
    const int output_pitch = output_width + 13;
    auto params = parameters(band ? band : tokens, width[0], width[1]);
    std::array<std::vector<uint8_t>, 4> weights, scales;
    std::array<std::vector<int8_t>, 4> decoded;
    Device weight_device[4], scale_device[4], weight_tabs[2], scale_tabs[2];
    Device stacked_weight[2], stacked_scale[2];
    std::array<std::array<void*, 256>, 2> weight_addresses{}, scale_addresses{};
    for (unsigned index = 0; index < 4; ++index) {
        weights[index] = fixtures[index].encode(params, (index & 1u) ? 6 : 2);
        scales[index] = fixtures[index].encode(params, (index & 1u) ? 7 : 3);
        decoded[index] = fixtures[index].decode();
        weight_device[index].allocate(weights[index].size());
        scale_device[index].allocate(scales[index].size());
        weight_device[index].upload(weights[index].data(), weights[index].size());
        scale_device[index].upload(scales[index].data(), scales[index].size());
    }
    std::array<RadTensor, 9> tensors{};
    for (unsigned parity = 0; parity < 2; ++parity) {
        auto& weight_pointers = weight_addresses[parity];
        auto& scale_pointers = scale_addresses[parity];
        for (unsigned index = 0; index < 256; ++index) {
            const unsigned fixture = index == 255 ? parity + 2 : parity;
            weight_pointers[index] = scenario == 3 ? nullptr : weight_device[fixture].data();
            scale_pointers[index] = scenario == 3 ? nullptr : scale_device[fixture].data();
        }
        if (stacked) {
            stacked_weight[parity].allocate(256 * weights[parity].size());
            stacked_scale[parity].allocate(256 * scales[parity].size());
            for (unsigned index = 0; index < 256; ++index) {
                const unsigned fixture = index == 255 ? parity + 2 : parity;
                stacked_weight[parity].upload(weights[fixture].data(), weights[fixture].size(),
                                              index * weights[fixture].size());
                stacked_scale[parity].upload(scales[fixture].data(), scales[fixture].size(),
                                             index * scales[fixture].size());
            }
        } else {
            weight_tabs[parity].allocate(sizeof(weight_pointers));
            scale_tabs[parity].allocate(sizeof(scale_pointers));
            weight_tabs[parity].upload(weight_pointers.data(), sizeof(weight_pointers));
            scale_tabs[parity].upload(scale_pointers.data(), sizeof(scale_pointers));
        }
        tensors[2 + parity * 4] = weight_table(stacked ? stacked_weight[parity].data()
            : weight_tabs[parity].data(), weight_dtype, width[parity], stacked, weights[parity].size());
        tensors[3 + parity * 4] = weight_table(stacked ? stacked_scale[parity].data()
            : scale_tabs[parity].data(), scale_dtype, width[parity], stacked, scales[parity].size());
        if (logical) {
            tensors[3 + parity * 4].shape[2] = 2560;
            tensors[3 + parity * 4].stride[1] = 2560;
        }
    }
    std::vector<std::vector<int>> buckets(512);
    std::vector<int32_t> sorted(total, -1), offsets(513);
    const int routed = scenario == 3 ? 0 : total - 2;
    for (int index = 0; index < routed; ++index) {
        const int expert = scenario == 0 ? (index == routed - 1 ? 511 : 0)
            : scenario == 1 ? (index == 0 ? 0 : index < routed - 4 ? 1 : index < routed - 1 ? 510 : 511)
            : scenario == 4 ? (index % 4 < 2 ? index % 4 : index % 4 + 508)
            : (index * 37) % 512;
        buckets[expert].push_back((index * 7 + 3) % total);
    }
    int cursor = 0;
    for (int expert = 0; expert < 512; ++expert) {
        offsets[expert] = cursor;
        for (int slot : buckets[expert]) sorted[cursor++] = slot;
    }
    offsets[512] = cursor;
    if (routed > 4) {
        sorted[2] = -7;
        sorted[4] = total + 17;
    }
    Device activation, activation_scale, sorted_device, offset_device, output;
    activation.allocate(tokens * input_pitch);
    activation_scale.allocate(tokens * scale_pitch * sizeof(float));
    sorted_device.allocate(total * sizeof(int32_t));
    offset_device.allocate(513 * sizeof(int32_t));
    output.allocate(static_cast<size_t>(total) * output_pitch * sizeof(uint16_t));
    sorted_device.upload(sorted.data(), sorted_device.bytes);
    offset_device.upload(offsets.data(), offset_device.bytes);
    tensors[0] = matrix(activation.data(), RAD_I8, tokens, 2560, input_pitch);
    tensors[1] = matrix(activation_scale.data(), RAD_F32, tokens, 20, scale_pitch);
    tensors[4] = RadTensor{}; tensors[4].data = sorted_device.data();
    tensors[4].dtype = RAD_I32; tensors[4].rank = 1;
    tensors[4].shape[0] = total; tensors[4].stride[0] = 1;
    tensors[5] = tensors[4]; tensors[5].data = offset_device.data(); tensors[5].shape[0] = 513;
    tensors[8] = matrix(output.data(), RAD_BF16, total, output_width, output_pitch);
    RadArgs arguments{};
    arguments.t = tensors.data(); arguments.n_t = 9;
    arguments.p = params.data(); arguments.n_p = params.size();
    Stream stream;
    for (int repeat = 0; repeat < 2; ++repeat) {
        if (repeat && !stacked && scenario != 3) {
            for (unsigned parity = 0; parity < 2; ++parity) {
                std::swap(weight_addresses[parity][0], weight_addresses[parity][255]);
                std::swap(scale_addresses[parity][0], scale_addresses[parity][255]);
                weight_tabs[parity].upload(weight_addresses[parity].data(), sizeof(weight_addresses[parity]));
                scale_tabs[parity].upload(scale_addresses[parity].data(), sizeof(scale_addresses[parity]));
            }
        }
        std::vector<int8_t> input(tokens * input_pitch, -91);
        std::vector<float> factors(tokens * scale_pitch, -123.0f);
        for (int token = 0; token < tokens; ++token) {
            for (int position = 0; position < 2560; ++position)
                input[token * input_pitch + position] = static_cast<int8_t>(
                    ((position * 19 + token * 37 + repeat * 29) % 256) - 128);
            for (int group = 0; group < 20; ++group)
                factors[token * scale_pitch + group] = (group % 7 + token % 3 + repeat + 1) * 0.001317f;
        }
        activation.upload(input.data(), input.size());
        activation_scale.upload(factors.data(), factors.size() * sizeof(float));
        if (repeat) hip_check(hipStreamBeginCapture(stream.stream, hipStreamCaptureModeThreadLocal));
        hip_check(hipMemsetAsync(output.data(), 0xa5, output.bytes, stream.stream));
        expect(r3dh_moe_iqr_launch(&arguments, reinterpret_cast<RadStream>(stream.stream)) == RAD_OK,
               "native gate_up launch");
        if (repeat) {
            hipGraph_t graph = nullptr;
            hipGraphExec_t executable = nullptr;
            hip_check(hipStreamEndCapture(stream.stream, &graph));
            hip_check(hipGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
            hip_check(hipGraphLaunch(executable, stream.stream));
            hip_check(hipStreamSynchronize(stream.stream));
            hip_check(hipGraphExecDestroy(executable));
            hip_check(hipGraphDestroy(graph));
        } else hip_check(hipStreamSynchronize(stream.stream));
        std::vector<uint16_t> result(static_cast<size_t>(total) * output_pitch);
        hip_check(hipMemcpy(result.data(), output.data(), output.bytes, hipMemcpyDeviceToHost));
        int expert = 0;
        for (int row = 0; row < total; ++row) {
            while (expert < 512 && offsets[expert + 1] <= row) ++expert;
            const int slot = sorted[row];
            const int source_row = slot >= 0 && slot < total ? slot / 10 : -1;
            const bool remapped = repeat && !stacked && scenario != 3;
            const bool second = expert >= 510 ? !remapped : expert < 2 && remapped;
            const int fixture = expert < 512 ? (expert & 1) + (second ? 2 : 0) : 0;
            for (int column = 0; column < output_width; ++column) {
                uint16_t expected = 0;
                const int part_width = output_width / 2;
                const int local = column % part_width;
                const int weight_row = (column / part_width) * (fixtures[fixture].rows / 2) + local;
                if (expert < 512 && source_row >= 0 && local < fixtures[fixture].rows / 2) {
                    float accumulator = 0.0f;
                    for (int group = 0; group < 20; ++group) {
                        int32_t subtotal = 0;
                        for (int offset = 0; offset < 128; ++offset) {
                            const int position = group * 128 + offset;
                            subtotal += static_cast<int>(input[source_row * input_pitch + position])
                                * decoded[fixture][static_cast<size_t>(weight_row) * 2560 + position];
                        }
                        const float scaled = static_cast<float>(subtotal) * factors[source_row * scale_pitch + group];
                        accumulator = accumulator + scaled * fixtures[fixture].column[group];
                    }
                    expected = reference_bf16(accumulator * fixtures[fixture].row[weight_row]);
                }
                if (result[static_cast<size_t>(row) * output_pitch + column] != expected) {
                    std::fprintf(stderr, "mismatch %s M=%d scenario=%d repeat=%d row=%d col=%d got=%04x want=%04x\n",
                        label, tokens, scenario, repeat, row, column,
                        result[static_cast<size_t>(row) * output_pitch + column], expected);
                    throw std::runtime_error("strict BF16 oracle mismatch");
                }
                ++checks;
            }
            for (int column = output_width; column < output_pitch; ++column)
                expect(result[static_cast<size_t>(row) * output_pitch + column] == 0xa5a5u, "row padding guard");
        }
        for (Device* buffer : {&activation, &activation_scale, &sorted_device, &offset_device, &output}) buffer->guards();
        std::vector<int8_t> input_after(input.size());
        hip_check(hipMemcpy(input_after.data(), activation.data(), activation.bytes, hipMemcpyDeviceToHost));
        expect(input_after == input, "activation unchanged");
        activation_scale.unchanged(factors.data(), factors.size() * sizeof(float));
        sorted_device.unchanged(sorted.data(), sorted_device.bytes);
        offset_device.unchanged(offsets.data(), offset_device.bytes);
        for (unsigned index = 0; index < 4; ++index) {
            weight_device[index].unchanged(weights[index].data(), weights[index].size());
            scale_device[index].unchanged(scales[index].data(), scales[index].size());
        }
        ++gpu_runs;
        gpu_output_values += static_cast<uint64_t>(total) * output_width;
    }
    for (unsigned index = 0; index < 4; ++index) {
        weight_device[index].guards(); scale_device[index].guards();
    }
    for (unsigned parity = 0; parity < 2; ++parity) {
        if (stacked) { stacked_weight[parity].guards(); stacked_scale[parity].guards(); }
        else {
            weight_tabs[parity].guards(); scale_tabs[parity].guards();
            weight_tabs[parity].unchanged(weight_addresses[parity].data(), sizeof(weight_addresses[parity]));
            scale_tabs[parity].unchanged(scale_addresses[parity].data(), sizeof(scale_addresses[parity]));
        }
    }
    const auto saved = tensors[6];
    tensors[6].stride[0] = -1;
    expect(r3dh_moe_iqr_launch(&arguments, reinterpret_cast<RadStream>(stream.stream)) == RAD_E_STRIDE,
           "reject negative table stride");
    tensors[6] = saved;
    auto reject_tensor = [&](int operand, RadTensor invalid, int code) {
        const auto original = tensors[operand]; tensors[operand] = invalid;
        expect(r3dh_moe_iqr_launch(&arguments, reinterpret_cast<RadStream>(stream.stream)) == code,
               "strict boundary rejection");
        tensors[operand] = original;
    };
    auto invalid = tensors[8]; invalid.shape[0]++;
    reject_tensor(8, invalid, RAD_E_SHAPE);
    invalid = tensors[8]; invalid.shape[0] = 0; reject_tensor(8, invalid, RAD_E_SHAPE);
    invalid = tensors[8]; invalid.shape[0] = int64_t(band ? band : tokens) * 10 + 10;
    reject_tensor(8, invalid, RAD_E_SHAPE);
    invalid = tensors[3]; invalid.dtype++; reject_tensor(3, invalid, RAD_E_DTYPE);
    invalid = tensors[3]; invalid.shape[2] = 21; reject_tensor(3, invalid, RAD_E_SHAPE);
    invalid = tensors[3]; invalid.stride[1]++; reject_tensor(3, invalid, RAD_E_STRIDE);
    invalid = tensors[2]; invalid.shape[0]--; reject_tensor(2, invalid, RAD_E_SHAPE);
    invalid = tensors[0]; invalid.shape[0]++; reject_tensor(0, invalid, RAD_E_SHAPE);
    params[0] = RAD_INT("M", 2049);
    expect(r3dh_moe_iqr_launch(&arguments, reinterpret_cast<RadStream>(stream.stream)) == RAD_E_SHAPE,
           "reject band beyond 2048");
    params[0] = RAD_INT("M", band ? band : tokens);
    params[4] = RAD_INT("K_odd", 2432);
    expect(r3dh_moe_iqr_launch(&arguments, reinterpret_cast<RadStream>(stream.stream)) == RAD_E_SHAPE,
           "reject odd K");
    ++gpu_cases;
    std::printf("PASS %s M=%d band=%d N=%d N_odd=%d scenario=%d table=%s scale=%s repeats=2 exact guards stream graph\n",
                label, tokens, band ? band : tokens, width[0], width[1], scenario, stacked ? "stacked" : "pointer", logical ? "logical" : "physical");
    std::fflush(stdout);
}
void synthetic_gpu_tests() {
    gpu_identity();
    for (int even : {512, 768}) {
        const int odd = even == 512 ? 768 : 512;
        std::array<Fixture, 4> fixtures{Fixture(even, 7), Fixture(odd, 19),
                                       Fixture(even, 31), Fixture(odd, 43)};
        gpu_case("synthetic", fixtures, 1, 0, false);
        gpu_case("synthetic", fixtures, 3, 1, false);
        gpu_case("synthetic", fixtures, 8, 2, false);
        gpu_case("synthetic", fixtures, 17, 0, false);
        gpu_case("synthetic", fixtures, 3, 3, false);
        gpu_case("synthetic", fixtures, 3, 1, true);
    }
}
