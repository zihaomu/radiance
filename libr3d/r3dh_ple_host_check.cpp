#include "r3dh_ple_host.h"
#include "r3dh_mtp_ple.h"
#include "r3dh_i8.h"
#include "rad_abi_layout.h"

#include <array>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <stdexcept>
#include <vector>

namespace {
uint64_t checks = 0;
void require(bool condition, const char* message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}
RadParam integer(const char* key, int64_t value) {
    RadParam result{}; result.key = key; result.kind = RAD_P_INT; result.ival = value; return result;
}
RadParam text(const char* key, const char* value) {
    RadParam result{}; result.key = key; result.kind = RAD_P_STR; result.sval = value; return result;
}
RadTensor tensor(void* data, uint32_t dtype, int64_t rows, int64_t columns = 0) {
    RadTensor result{};
    result.data = data; result.dtype = dtype; result.rank = columns ? 2 : 1;
    result.shape[0] = rows; result.shape[1] = columns;
    result.stride[0] = columns ? columns : 1; result.stride[1] = 1;
    return result;
}
RadArgs arguments(const RadTensor* tensors, int tensor_count, const RadParam* params, int count) {
    RadArgs result{};
    result.t = tensors; result.n_t = tensor_count; result.p = params; result.n_p = count;
    result.world_size = 1;
    return result;
}
void gains() {
    const RadParam params[]{integer("M", 3), integer("n", 2560), integer("hc", 4)};
    std::vector<uint16_t> input(10240), roundtrip(10240);
    std::vector<uint32_t> stored(10242, 0xdeadbeef);
    RadEncoding encoding = rad_enc_plain(RAD_BF16);
    int selection = 0;
    for (uint32_t start = 0; start < 65536; start += 10240) {
        for (size_t index = 0; index < input.size(); ++index) input[index] = uint16_t(start + index);
        RadTensor plane = tensor(input.data(), RAD_BF16, input.size());
        for (int operand = 3; operand <= 5; ++operand) {
            RadLayout layout{};
            require(r3dh_ple_host_gain_layout(params, 3, operand, &encoding, &selection, &plane, 1, &layout) == RAD_OK, "gain layout");
            require(layout.dtype == RAD_F32 && layout.bytes == 40960 && layout.shape[0] == 10240, "wide gain extent");
            require(r3dh_ple_host_gain_relayout(params, 3, operand, &encoding, &selection, &plane, 1,
                stored.data() + 1, 40960) == RAD_OK, "gain widen");
            require(stored.front() == 0xdeadbeef && stored.back() == 0xdeadbeef, "gain guards");
            for (size_t index = 0; index < input.size(); ++index)
                require(stored[index + 1] == uint32_t(input[index]) << 16, "exact BF16 widening including NaN");
        }
    }
    for (size_t index = 0; index < input.size(); ++index) input[index] = uint16_t(0x3e00 + index % 256);
    RadTensor plane = tensor(input.data(), RAD_BF16, input.size());
    require(r3dh_ple_host_gain_relayout(params, 3, 3, &encoding, &selection, &plane, 1,
        stored.data() + 1, 40960) == RAD_OK, "gain finite widen");
    plane.data = roundtrip.data();
    require(r3dh_ple_host_gain_unrelayout(params, 3, 3, &encoding, &selection,
        stored.data() + 1, 40960, &plane, 1) == RAD_OK && input == roundtrip, "gain inverse");
    RadLayout layout{};
    plane.shape[0] = 2560;
    require(r3dh_ple_host_gain_layout(params, 3, 3, &encoding, &selection, &plane, 1, &layout) == RAD_E_SHAPE, "reject narrow gains");
    plane.shape[0] = 10240; plane.stride[0] = 2;
    require(r3dh_ple_host_gain_layout(params, 3, 3, &encoding, &selection, &plane, 1, &layout) == RAD_E_STRIDE, "reject strided gains");
    plane.stride[0] = 1;
    require(r3dh_ple_host_gain_layout(params, 3, 2, &encoding, &selection, &plane, 1, &layout) == RAD_E_UNSUPPORTED, "reject non-gain operand");
    require(r3dh_ple_host_gain_relayout(params, 3, 3, &encoding, &selection, &plane, 1,
        stored.data(), 40959) == RAD_E_SHAPE, "reject byte count");
    selection = 1;
    require(r3dh_ple_host_gain_layout(params, 3, 3, &encoding, &selection, &plane, 1, &layout) == RAD_E_UNSUPPORTED, "reject selected plane");
    selection = 0; encoding.transform[0] = 'x';
    require(r3dh_ple_host_gain_layout(params, 3, 3, &encoding, &selection, &plane, 1, &layout) == RAD_E_UNSUPPORTED, "reject transformed gains");
    encoding = rad_enc_plain(RAD_F32);
    plane = tensor(stored.data() + 1, RAD_F32, 10240);
    std::vector<uint32_t> copied(10240);
    require(r3dh_ple_host_gain_relayout(params, 3, 5, &encoding, &selection, &plane, 1,
        copied.data(), 40960) == RAD_OK, "F32 gain copy");
    require(!std::memcmp(copied.data(), stored.data() + 1, 40960), "F32 gain exact");
}
void schemas_and_devices() {
    require(r3dh_ple_host_schema_count() == 7, "schema count");
    const auto* ids = r3dh_ple_host_schema_at(0);
    const auto* gather = r3dh_ple_host_schema_at(1);
    const auto* gate = r3dh_ple_host_schema_at(2);
    require(ids->n_operands == 11 && ids->operands[1].role == RAD_OPD_INOUT
        && ids->operands[10].optional && ids->operands[10].role == RAD_OPD_OUT, "ids state/read-ahead schema");
    require(gather->n_operands == 5 && gather->operands[1].role == RAD_OPD_WEIGHT
        && gather->operands[2].optional && gather->operands[4].optional, "gather schema");
    require(gate->n_operands == 8 && gate->n_params == 7, "gate schema");
    require(!r3dh_ple_host_schema_at(-1) && !r3dh_ple_host_schema_at(7), "schema bounds");
#ifndef R3DH_PLE_HOST_CPU_ONLY
    const RadLaunchFn launches[]{r3dh_mtp_ple_rad_ngram_ids, r3dh_mtp_ple_ple_gate,
        r3dh_i8_rad_gemm_nt_q, r3dh_i8_rad_quant_act_i8g, r3dh_i8_rad_had_quant_act_i8g};
    require(r3dh_ple_host_device_kernel_count() == 5, "device count");
    for (int index = 0; index < 5; ++index) {
        const auto* row = r3dh_ple_host_device_kernel_at(index);
        require(row && row->domain == RAD_DOMAIN_DEVICE && row->launch == launches[index], "native DEVICE launch identity");
        require(std::strcmp(row->op, "embed_lookup_q"), "no DEVICE table gather exposed");
    }
    require(!r3dh_ple_host_device_kernel_at(-1) && !r3dh_ple_host_device_kernel_at(5), "device bounds");
    RadParam params[]{integer("M", 1), integer("n", 2560), integer("hc", 4)};
    params[0].kind = RAD_P_RANGE; params[0].ihi = 32;
    RadOpdDesc desc{};
    const auto* gate_row = r3dh_ple_host_device_kernel_at(1);
    require(gate_row->opd_shape(params, 3, 0, &desc) == RAD_OK && desc.shape[0] == 32
        && desc.shape[1] == 10240, "gate range resolved");
    require(gate_row->layout == r3dh_ple_host_gain_layout, "gate gain hook");
#endif
}
struct Library {
    void* handle;
    explicit Library(const char* path) : handle(dlopen(path, RTLD_NOW | RTLD_LOCAL)) {
        if (!handle) throw std::runtime_error(dlerror());
    }
    ~Library() { dlclose(handle); }
    template<class Function> Function symbol(const char* name) {
        auto result = reinterpret_cast<Function>(dlsym(handle, name));
        require(result != nullptr, name); return result;
    }
};
void host_fixture(const char* path) {
    require(r3dh_ple_host_open(path) == RAD_OK, "retained real provider before inspection");
    Library library(path);
    const auto abi = library.symbol<uint32_t(*)()>("rad_plugin_abi_version");
    const auto fingerprint = library.symbol<uint64_t(*)()>("rad_plugin_abi_layout");
    require(abi() == RAD_ABI_VERSION && fingerprint() == rad_abi_layout(), "libavx ABI before row access");
    const auto count = library.symbol<int(*)()>("rad_kernel_count");
    const auto at = library.symbol<const RadKernelInfo*(*)(int)>("rad_kernel_at");
    if (auto open = reinterpret_cast<int(*)()>(dlsym(library.handle, "rad_plugin_open")))
        require(open() == RAD_OK, "libavx open");
    const RadKernelInfo *ids = nullptr, *gather = nullptr;
    for (int index = 0; index < count(); ++index) {
        const auto* row = at(index);
        if (!std::strcmp(row->op, "ngram_ids")) ids = row;
        if (!std::strcmp(row->op, "embed_lookup_q")) gather = row;
    }
    R3dhPleHost provider{};
    require(r3dh_ple_host_bind(&provider, abi(), ids, gather) == RAD_OK, "bind actual libavx HOST providers");
    const auto* borrowed = r3dh_ple_host_kernel_at(&provider, 1);
    require(borrowed && borrowed->init == gather->init && borrowed->layout == gather->layout
        && borrowed->launch == gather->launch && borrowed->fini == gather->fini
        && borrowed->scratch == gather->scratch && borrowed->capabilities == gather->capabilities,
        "preserve gather reader/cache/lifecycle hooks");
    RadKernelInfo wrong = *gather; wrong.domain = RAD_DOMAIN_DEVICE;
    R3dhPleHost rejected{};
    require(r3dh_ple_host_bind(&rejected, abi(), ids, &wrong) == RAD_E_UNSUPPORTED
        && !r3dh_ple_host_kernel_at(&rejected, 1), "refuse DEVICE gather");
    require(r3dh_ple_host_bind(&rejected, abi() - 1, ids, gather) == RAD_E_UNSUPPORTED, "refuse old ABI");
    int32_t tokens[]{3, 7, 0, 9, 2}, state[]{-99, -99}, cu[]{0, 3}, warm[]{0};
    int64_t multipliers[]{0x7fffffffffffffffLL, 0x7ffffffffffffffdLL, 24000000000001LL};
    int64_t sizes[]{11, 13, 17, 19}, offsets[]{0, 11, 24, 41};
    std::array<int32_t, 14> output; output.fill(-123);
    std::array<int32_t, 10> ahead; ahead.fill(-123);
    RadTensor tensors[]{tensor(tokens, RAD_I32, 5), tensor(state, RAD_I32, 1, 2),
        tensor(cu, RAD_I32, 2), tensor(multipliers, RAD_I64, 3), tensor(sizes, RAD_I64, 4),
        tensor(offsets, RAD_I64, 4), {}, tensor(warm, RAD_I32, 1), {},
        tensor(output.data() + 1, RAD_I32, 3, 4), tensor(ahead.data() + 1, RAD_I32, 2, 4)};
    const RadParam params[]{integer("M", 3), integer("heads", 4), integer("ngram", 3), integer("eos", 0)};
    auto args = arguments(tensors, 11, params, 4);
    require(provider.rows[0].launch(&args, nullptr) == RAD_OK, "actual HOST ngram ids");
    for (int row = 0; row < 5; ++row) {
        uint64_t mixed = uint64_t(tokens[row]) * uint64_t(multipliers[0]);
        bool cut = tokens[row] == 0;
        for (int depth = 1; depth <= 2; ++depth) {
            const int32_t token = cut || row < depth ? 0 : tokens[row - depth];
            cut = cut || token == 0;
            mixed ^= uint64_t(token) * uint64_t(multipliers[depth]);
            int64_t signed_mix;
            std::memcpy(&signed_mix, &mixed, 8);
            for (int head = (depth - 1) * 2; head < depth * 2; ++head) {
                int64_t expected = signed_mix % sizes[head];
                if (expected < 0) expected += sizes[head];
                expected += offsets[head];
                const int32_t actual = row < 3 ? output[1 + row * 4 + head] : ahead[1 + (row - 3) * 4 + head];
                require(actual == expected, "signed wrap/floor-mod/EOS ngram oracle");
            }
        }
    }
    require(state[0] == 7 && state[1] == 0, "read-ahead must not commit future state");
    require(output.front() == -123 && output.back() == -123 && ahead.front() == -123
        && ahead.back() == -123, "ngram guards");
    constexpr int width = 160, vocabulary = 60;
    std::vector<uint8_t> table(vocabulary * width);
    for (size_t index = 0; index < table.size(); ++index) table[index] = uint8_t(0x30 + index % 16);
    std::vector<float> embedding(3 * 4 * width + 2, -123.0f);
    uint16_t scale = 0x3f00;
    RadParam gather_params[]{integer("M", 12), integer("n_embd", width), integer("n_vocab", vocabulary), text("dtype", "f32")};
    RadTensor gather_tensors[]{tensors[9], tensor(table.data(), RAD_F8E4M3, vocabulary, width),
        tensor(&scale, RAD_BF16, 1), tensor(embedding.data() + 1, RAD_F32, 3, 4 * width), tensors[10]};
    auto gather_args = arguments(gather_tensors, 5, gather_params, 4);
    require(borrowed->init(gather_params, 4, 0, 1, &gather_args.instance) == RAD_OK, "HOST gather init");
    for (int repeat = 0; repeat < 2; ++repeat) {
        if (repeat) gather_tensors[2] = {};
        require(borrowed->launch(&gather_args, nullptr) == RAD_OK, "actual HOST E4M3 gather");
        for (int row = 0; row < 12; ++row)
            for (int column = 0; column < width; ++column) {
                const uint8_t code = table[size_t(output[row + 1]) * width + column];
                const float expected = (code < 0x38 ? 0.5f + float(code - 0x30) / 16.0f
                    : 1.0f + float(code - 0x38) / 8.0f) * (repeat ? 1.0f : 0.5f);
                require(embedding[1 + row * width + column] == expected, "HOST embedding independent oracle");
            }
        require(embedding.front() == -123 && embedding.back() == -123, "HOST embedding guards");
    }
    if (borrowed->fini) borrowed->fini(gather_args.instance);
    std::printf("PLE_HOST_REAL_LIBAVX_PASS ids=20 gathered=3840 ahead_state_preserved=1\n");
}
}
int main(int argc, char** argv) {
    try {
        require(argc == 2, "usage: r3dh_ple_host_check /absolute/path/libavx.so");
        gains(); schemas_and_devices(); host_fixture(argv[1]);
        std::printf("PLE_HOST_PASS checks=%llu GPU_used=0 full_model=not_tested\n", static_cast<unsigned long long>(checks));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "PLE_HOST_FAIL %s checks=%llu\n", error.what(), static_cast<unsigned long long>(checks));
        return 1;
    }
}
