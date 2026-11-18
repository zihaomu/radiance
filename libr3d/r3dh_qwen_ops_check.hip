#include <hip/hip_runtime.h>
#include "r3dh_qwen_ops.h"
#include "rad_format.h"
#include <fstream>
#include <string>
#include "ref_ops.h"
#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)

extern "C" int r3d_rad_moe_router_topk_bf16(const RadArgs*, RadStream);
extern "C" int r3d_rad_moe_scatter(const RadArgs*, RadStream);
extern "C" int r3d_rad_moe_gather(const RadArgs*, RadStream);
namespace {
uint64_t comparisons = 0, gate_cases = 0, router_cases = 0, routing_weight_ulps = 0;
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void gpu(hipError_t status) {
    if (status != hipSuccess) throw std::runtime_error(hipGetErrorString(status));
}
float fp32(uint16_t value) {
    const uint32_t bits = static_cast<uint32_t>(value) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}
uint16_t bf16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7fffffffu) > 0x7f800000u) return (bits >> 16) | 0x40u;
    const uint32_t fraction = bits & 0xffffu;
    const uint32_t truncated = bits >> 16;
    return truncated + (fraction > 0x8000u || (fraction == 0x8000u && (truncated & 1u)));
}
int bytes(uint32_t dtype) {
    return dtype == RAD_F32 || dtype == RAD_I32 ? 4 : dtype == RAD_BF16 ? 2 : 1;
}
RadTensor matrix(void* data, uint32_t dtype, int64_t rows, int64_t columns, int64_t pitch) {
    RadTensor result{};
    result.data = data; result.dtype = dtype; result.rank = 2;
    result.shape[0] = rows; result.shape[1] = columns;
    result.stride[0] = pitch; result.stride[1] = 1;
    return result;
}
RadArgs arguments(const std::vector<RadTensor>& tensors, const std::vector<RadParam>& params) {
    RadArgs result{};
    result.t = tensors.data(); result.n_t = tensors.size();
    result.p = params.data(); result.n_p = params.size();
    result.world_size = 1; result.kv_used_max_h = -1;
    return result;
}
struct Buffer {
    std::vector<uint8_t> host;
    uint8_t* device = nullptr;
    RadTensor tensor{};
    int rows, columns, pitch, item_bytes;
    Buffer(uint32_t dtype, int height, int width, int padding = 7)
        : rows(height), columns(width), pitch(width + padding), item_bytes(bytes(dtype)) {
        host.assign(128 + static_cast<size_t>(rows) * pitch * item_bytes, 0xa5);
        gpu(hipMalloc(&device, host.size()));
        tensor = matrix(device + 64, dtype, rows, columns, pitch);
    }
    ~Buffer() { if (device) (void)hipFree(device); }
    Buffer(const Buffer&) = delete;
    template<class Value> Value* data() { return reinterpret_cast<Value*>(host.data() + 64); }
    RadTensor cpu() { RadTensor result = tensor; result.data = host.data() + 64; return result; }
    void upload(hipStream_t stream) {
        gpu(hipMemcpyAsync(device, host.data(), host.size(), hipMemcpyHostToDevice, stream));
    }
    void download(hipStream_t stream) {
        gpu(hipMemcpyAsync(host.data(), device, host.size(), hipMemcpyDeviceToHost, stream));
    }
    void guards() {
        for (size_t position = 0; position < host.size(); ++position) {
            const size_t offset = position >= 64 ? position - 64 : 0;
            const bool payload = position >= 64 && offset < static_cast<size_t>(rows) * pitch * item_bytes
                && (offset / item_bytes) % pitch < static_cast<size_t>(columns);
            if (!payload) require(host[position] == 0xa5, "allocation or row-padding guard overwritten");
        }
    }
};
void compare(const Buffer& buffer, const std::vector<uint8_t>& expected, const char* label) {
    for (int row = 0; row < buffer.rows; ++row)
        for (int column = 0; column < buffer.columns; ++column) {
            const size_t offset = 64 + (static_cast<size_t>(row) * buffer.pitch + column) * buffer.item_bytes;
            ++comparisons;
            if (std::memcmp(buffer.host.data() + offset, expected.data() + offset, buffer.item_bytes)) {
                std::fprintf(stderr, "mismatch %s row=%d col=%d actual=%02x expected=%02x\n",
                    label, row, column, buffer.host[offset], expected[offset]);
                throw std::runtime_error(label);
            }
        }
}
#include "r3dh_qwen_ops_check_host.h"
void host_checks() {
    for (uint32_t upper = 0; upper < 65536; ++upper) {
        for (uint32_t lower : {0u, 0x7fffu, 0x8000u, 0x8001u, 0xffffu}) {
            const uint32_t bits = (upper << 16) | lower;
            float value;
            std::memcpy(&value, &bits, 4);
            require(bf16(value) == rad_f32_to_bf16(value), "BF16 independent RNE oracle");
            ++comparisons;
        }
    }
    alignas(16) std::array<uint8_t, 16> placeholder{};
    std::vector<RadTensor> tensors{matrix(placeholder.data(), RAD_BF16, 1, 768, 768),
        matrix(placeholder.data(), RAD_I8, 1, 384, 384), matrix(placeholder.data(), RAD_F32, 1, 3, 3)};
    std::vector<RadParam> params{RAD_INT("n", 384), RAD_INT("group", 128), RAD_STR("dtype", "bf16")};
    RadArgs args = arguments(tensors, params);
    require(r3dh_qwen_ops_gated_had_quant_i8g(nullptr, nullptr) == RAD_E_INVAL, "null args");
    tensors[0].dtype = RAD_F16;
    require(r3dh_qwen_ops_gated_had_quant_i8g(&args, nullptr) == RAD_E_DTYPE, "F16 rejected");
    tensors[0].dtype = RAD_BF16; tensors[0].stride[1] = 2;
    require(r3dh_qwen_ops_gated_had_quant_i8g(&args, nullptr) == RAD_E_STRIDE, "inner stride rejected");
    tensors[0].stride[1] = 1; tensors[2].shape[1] = 1;
    require(r3dh_qwen_ops_gated_had_quant_i8g(&args, nullptr) == RAD_E_SHAPE, "short scales rejected");
    tensors[2].shape[1] = 3; params[1] = RAD_INT("group", 256);
    require(r3dh_qwen_ops_gated_had_quant_i8g(&args, nullptr) == RAD_E_SHAPE, "group256 rejected");
    params[1] = RAD_INT("group", 128); params.push_back(RAD_STR("act", "relu"));
    args = arguments(tensors, params);
    require(r3dh_qwen_ops_gated_had_quant_i8g(&args, nullptr) == RAD_E_UNSUPPORTED, "unknown act rejected");
    extra_host_checks();
    std::printf("QWEN_HOST_PASS independent_bf16_checks=327680 malformed=6\n");
}
void gate_check(int rows, int width, bool separate, int mode, hipStream_t stream, bool head_view = false) {
    Buffer gate(RAD_BF16, rows, separate ? width : width * 2, 13);
    Buffer input(RAD_BF16, rows, width, head_view ? 0 : 11), codes(RAD_I8, rows, width, head_view ? 0 : 17);
    Buffer scales(RAD_F32, rows, width / 128, head_view ? 0 : 5), output(RAD_BF16, rows, width, 19);
    std::vector<uint16_t> product(static_cast<size_t>(rows) * width);
    std::vector<int8_t> oracle_codes(product.size());
    std::vector<float> oracle_scales(static_cast<size_t>(rows) * width / 128);
    std::vector<RadParam> params{RAD_INT("M", rows), RAD_INT("n", width), RAD_INT("group", 128),
        RAD_STR("dtype", "bf16"), RAD_STR("act", mode == 0 ? "silu" : mode == 1 ? "sigmoid" : "none")};
    for (int repeat = 0; repeat < 2; ++repeat) {
        for (int row = 0; row < rows; ++row)
            for (int column = 0; column < width; ++column) {
                const int sequence = (row * 101 + column * 37 + repeat * 211) % 513 - 256;
                float gate_value = sequence / 32.0f;
                float up_value = ((row * 31 + column * 17 + repeat * 157) % 257 - 128) / 16.0f;
                if (column / 128 == 0 && row % 3 == 0) up_value = 0.0f;
                if (width == 384 && (row + repeat) % 2 && column >= 256) up_value = 0.0f;
                if (column % 127 == 0) gate_value = (column & 1) ? -80.0f : 80.0f;
                if (mode == 2 && column % 7 == 0) { gate_value = fp32(0x3f81); up_value = fp32(0x3fc0); }
                if (column / 128 == 1 && row % 5 == 0) up_value *= 1e-20f;
                if (mode == 2 && column / 128 == 0) {
                    gate_value = 1.0f;
                    up_value = (column & 1) ? 126.5f / 128.0f : 127.5f / 128.0f;
                }
                const uint16_t gate_bits = bf16(gate_value), up_bits = bf16(up_value);
                gate.data<uint16_t>()[row * gate.pitch + column] = gate_bits;
                if (separate) input.data<uint16_t>()[row * input.pitch + column] = up_bits;
                else gate.data<uint16_t>()[row * gate.pitch + width + column] = up_bits;
                const float exact_gate = fp32(gate_bits), exact_up = fp32(up_bits);
                float activation = exact_gate;
                if (mode != 2) {
                    const float sigmoid = 1.0f / (1.0f + std::exp(-exact_gate));
                    activation = mode == 0 ? exact_gate * sigmoid : sigmoid;
                    if (separate) activation = fp32(bf16(activation));
                }
                product[static_cast<size_t>(row) * width + column] = bf16(activation * exact_up);
            }
        std::vector<RadTensor> ref_tensors{matrix(product.data(), RAD_BF16, rows, width, width),
            matrix(oracle_codes.data(), RAD_I8, rows, width, width),
            matrix(oracle_scales.data(), RAD_F32, rows, width / 128, width / 128)};
        RadArgs ref_args = arguments(ref_tensors, params);
        require(ref_had_quant_act_i8g(&ref_args, nullptr) == RAD_OK, "libref quantizer rejected fixture");
        if (mode == 2)
            require(oracle_scales[0] == 1.0f && oracle_codes[0] == 127 && oracle_codes[1] == 0,
                "explicit int8 ties-to-even fixture");
        std::vector<uint8_t> expected_codes = codes.host, expected_scales = scales.host, expected_output = output.host;
        for (int row = 0; row < rows; ++row) {
            for (int group = 0; group < width / 128; ++group) {
                std::array<float, 128> values{};
                for (int column = 0; column < 128; ++column)
                    values[column] = fp32(product[static_cast<size_t>(row) * width + group * 128 + column]);
                for (int distance = 1; distance < 128; distance *= 2)
                    for (int base = 0; base < 128; base += distance * 2)
                        for (int offset = 0; offset < distance; ++offset) {
                            const float left = values[base + offset], right = values[base + offset + distance];
                            values[base + offset] = left + right;
                            values[base + offset + distance] = left - right;
                        }
                float maximum = 0;
                for (float value : values) maximum = std::max(maximum, std::fabs(value));
                const float scale = maximum > 0 ? maximum * (1.0f / 127.0f) : 1.0f;
                const float inverse = 1.0f / scale;
                require(scale == oracle_scales[static_cast<size_t>(row) * (width / 128) + group], "CPU/libref scales");
                std::memcpy(expected_scales.data() + 64 + (static_cast<size_t>(row) * scales.pitch + group) * 4, &scale, 4);
                for (int column = 0; column < 128; ++column) {
                    const int index = group * 128 + column;
                    const int8_t code = static_cast<int8_t>(std::nearbyint(std::max(-127.0f, std::min(127.0f, values[column] * inverse))));
                    require(code == oracle_codes[static_cast<size_t>(row) * width + index], "CPU/libref code");
                    expected_codes[64 + static_cast<size_t>(row) * codes.pitch + index] = static_cast<uint8_t>(code);
                    const uint16_t rounded = product[static_cast<size_t>(row) * width + index];
                    std::memcpy(expected_output.data() + 64 + (static_cast<size_t>(row) * output.pitch + index) * 2, &rounded, 2);
                }
            }
        }
        gate.upload(stream); input.upload(stream); codes.upload(stream); scales.upload(stream); output.upload(stream);
        const auto gate_before = gate.host, input_before = input.host;
        std::vector<RadTensor> tensors{gate.tensor};
        if (separate) tensors.push_back(input.tensor);
        tensors.push_back(codes.tensor); tensors.push_back(scales.tensor);
        const bool include_output = repeat == 0;
        if (include_output) tensors.push_back(output.tensor);
        std::vector<uint8_t> expected_inplace = input.host;
        if (head_view) {
            require(separate && rows % 12 == 0, "head view fixture dimensions");
            for (int index : {1, 2, 3}) {
                tensors[index].shape[0] /= 12; tensors[index].shape[1] *= 12;
                tensors[index].stride[0] *= 12;
            }
            params[0] = RAD_INT("M", rows / 12);
            if (include_output) tensors.back() = tensors[1];
            if (repeat) {
                tensors[0].rank = 3; tensors[0].shape[0] = rows / 12; tensors[0].shape[1] = 12; tensors[0].shape[2] = width;
                tensors[0].stride[2] = 1; tensors[0].stride[1] = gate.pitch; tensors[0].stride[0] = 12 * gate.pitch;
            }
            for (int row = 0; row < rows; ++row)
                for (int column = 0; column < width; ++column) {
                    const uint16_t value = product[static_cast<size_t>(row) * width + column];
                    std::memcpy(expected_inplace.data() + 64 + (static_cast<size_t>(row) * input.pitch + column) * 2, &value, 2);
                }
        }
        RadArgs args = arguments(tensors, params);
        const RadStream native_stream = reinterpret_cast<RadStream>(stream);
        const int status = separate ? r3dh_qwen_ops_gate_had_quant_i8g(&args, native_stream)
                                    : r3dh_qwen_ops_gated_had_quant_i8g(&args, native_stream);
        require(status == RAD_OK, "native gate launch rejected fixture");
        gate.download(stream); input.download(stream); codes.download(stream); scales.download(stream); output.download(stream);
        gpu(hipStreamSynchronize(stream));
        gate.guards(); input.guards(); codes.guards(); scales.guards(); output.guards();
        require(gate.host == gate_before, "gate input mutated");
        require(input.host == (head_view && include_output ? expected_inplace : input_before), "gate x input mutation contract");
        compare(codes, expected_codes, "gate codes"); compare(scales, expected_scales, "gate scales");
        if (include_output && !head_view) compare(output, expected_output, "unrotated BF16 product");
        ++gate_cases;
    }
}

#include "r3dh_qwen_ops_check_gather.h"
#include "r3dh_qwen_ops_check_router_n1.h"
#include "r3dh_qwen_ops_check_router_n1_model.h"
void router_check(int rows, int repeat, hipStream_t stream) {
    constexpr int width = 2560, experts = 512, top = 10;
    Buffer input(RAD_BF16, rows, width, 13), weight(RAD_BF16, experts, width, 9);
    Buffer logits(RAD_BF16, rows, experts, 7), ids(RAD_I32, rows, top, 0), routing(RAD_BF16, rows, top, 0);
    Buffer sorted(RAD_I32, 1, rows * top, 0), offsets(RAD_I32, 1, experts + 1, 0), counts(RAD_I32, 1, experts, 0);
    for (int row = 0; row < rows; ++row)
        for (int position = 0; position < width; ++position)
            input.data<uint16_t>()[row * input.pitch + position] = bf16(((row * 47 + position * 13 + repeat * 7) % 97 - 48) / 128.0f);
    for (int expert = 0; expert < experts; ++expert)
        for (int position = 0; position < width; ++position) {
            float value = ((expert * 19 + position * 23 + repeat * 3) % 131 - 65) / 256.0f;
            if (position == 0) value = expert % 2 ? 1e-10f : 1e5f;
            weight.data<uint16_t>()[expert * weight.pitch + position] = bf16(value);
        }
    if (repeat == 1) for (int expert = 0; expert < experts; ++expert)
        std::fill_n(weight.data<uint16_t>() + expert * weight.pitch, width, uint16_t{0});
    std::vector<RadParam> params{RAD_INT("M", rows), RAD_INT("N", experts), RAD_INT("K", width),
        RAD_INT("n_expert", experts), RAD_INT("top_k", top), RAD_INT("norm", 1), RAD_STR("dtype", "bf16")};
    std::vector<uint8_t> expected_logits = logits.host;
    auto* expected_data = reinterpret_cast<uint16_t*>(expected_logits.data() + 64);
    for (int row = 0; row < rows; ++row)
        for (int expert = 0; expert < experts; ++expert) {
            float sum = 0;
            for (int position = 0; position < width; ++position) {
                const float product = fp32(input.data<uint16_t>()[row * input.pitch + position])
                    * fp32(weight.data<uint16_t>()[expert * weight.pitch + position]);
                sum = sum + product;
            }
            expected_data[row * logits.pitch + expert] = bf16(sum);
        }
    std::vector<uint8_t> reference_logits = logits.host;
    std::vector<RadTensor> ref_gemm{input.cpu(), weight.cpu(), logits.cpu()};
    ref_gemm[2].data = reference_logits.data() + 64;
    RadArgs ref_args = arguments(ref_gemm, params);
    require(ref_gemm_nt(&ref_args, nullptr) == RAD_OK, "libref router GEMM");
    require(reference_logits == expected_logits, "independent/libref router projection");
    std::vector<int32_t> expected_ids(rows * top), expected_sorted(rows * top), expected_offsets(experts + 1), expected_counts(experts);
    std::vector<uint16_t> expected_weights(rows * top);
    std::vector<RadTensor> ref_top{matrix(expected_data, RAD_BF16, rows, experts, logits.pitch),
        matrix(expected_ids.data(), RAD_I32, rows, top, top), matrix(expected_weights.data(), RAD_BF16, rows, top, top)};
    ref_args = arguments(ref_top, params);
    require(ref_router_topk(&ref_args, nullptr) == RAD_OK, "libref topk");
    std::vector<RadTensor> ref_sort{ref_top[1], matrix(expected_sorted.data(), RAD_I32, 1, rows * top, rows * top),
        matrix(expected_offsets.data(), RAD_I32, 1, experts + 1, experts + 1), matrix(expected_counts.data(), RAD_I32, 1, experts, experts)};
    ref_args = arguments(ref_sort, params);
    require(ref_moe_scatter(&ref_args, nullptr) == RAD_OK, "libref scatter");
    for (Buffer* buffer : {&input, &weight, &logits, &ids, &routing, &sorted, &offsets, &counts}) buffer->upload(stream);
    std::vector<RadTensor> tensors{input.tensor, weight.tensor, logits.tensor};
    RadArgs args = arguments(tensors, params);
    const RadStream native_stream = reinterpret_cast<RadStream>(stream);
    require(r3dh_qwen_ops_router_gemm(&args, native_stream) == RAD_OK, "native router projection");
    tensors = {logits.tensor, ids.tensor, routing.tensor}; args = arguments(tensors, params);
    require(r3d_rad_moe_router_topk_bf16(&args, native_stream) == RAD_OK, "existing native topk");
    tensors = {ids.tensor, sorted.tensor, offsets.tensor, counts.tensor}; args = arguments(tensors, params);
    require(r3d_rad_moe_scatter(&args, native_stream) == RAD_OK, "existing native scatter");
    for (Buffer* buffer : {&input, &weight, &logits, &ids, &routing, &sorted, &offsets, &counts}) buffer->download(stream);
    gpu(hipStreamSynchronize(stream));
    for (Buffer* buffer : {&input, &weight, &logits, &ids, &routing, &sorted, &offsets, &counts}) buffer->guards();
    compare(logits, expected_logits, "FP32 router projection");
    for (int slot = 0; slot < rows * top; ++slot) {
        require(ids.data<int32_t>()[slot] == expected_ids[slot], "topk ids exact");
        require(sorted.data<int32_t>()[slot] == expected_sorted[slot], "stable scatter exact");
        const int ulps = std::abs(int(routing.data<uint16_t>()[slot]) - int(expected_weights[slot]));
        require(ulps == 0, "existing topk weights must be BF16 bit-exact");
        routing_weight_ulps += ulps;
        comparisons += 3;
    }
    for (int expert = 0; expert < experts; ++expert) {
        require(offsets.data<int32_t>()[expert] == expected_offsets[expert], "expert offsets exact");
        require(counts.data<int32_t>()[expert] == expected_counts[expert], "expert counts exact");
        comparisons += 2;
    }
    require(offsets.data<int32_t>()[experts] == rows * top, "scatter terminal offset");
    gather_check(rows, repeat, routing, sorted, stream);
    ++router_cases;
}
}
int main(int argc, char** argv) {
    try {
        require(std::fesetround(FE_TONEAREST) == 0, "host rounding mode");
        host_checks();
        if (argc == 2 && !std::strcmp(argv[1], "--host")) return 0;
        const bool router_n1 = (argc == 2 || argc == 3) && !std::strcmp(argv[1], "--router-n1");
        require(router_n1 || (argc == 2 && !std::strcmp(argv[1], "--gpu")), "usage: --host|--gpu|--router-n1");
        require(std::getenv("ROCR_VISIBLE_DEVICES") && !std::strcmp(std::getenv("ROCR_VISIBLE_DEVICES"), router_n1 ? "6" : "2"), "unexpected physical GPU");
        require(std::getenv("HIP_VISIBLE_DEVICES") && !std::strcmp(std::getenv("HIP_VISIBLE_DEVICES"), "0"), "logical GPU0 only");
        int devices = 0; gpu(hipGetDeviceCount(&devices)); require(devices == 1, "exactly one visible GPU");
        hipDeviceProp_t properties{}; gpu(hipGetDeviceProperties(&properties, 0));
        require(!std::strncmp(properties.gcnArchName, "gfx1100", 7), "checker requires gfx1100");
        hipStream_t stream{}; gpu(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
        hipStream_t second_stream{}; gpu(hipStreamCreateWithFlags(&second_stream, hipStreamNonBlocking));
        if (router_n1) {
            for (int rows : {1, 3, 8, 16, 32, 64, 128, 512, 2048})
                for (bool padded : {false, true}) router_n1_case(rows, padded, stream, second_stream);
            std::printf("QWEN_N1_SYNTHETIC_PASS cases=%llu launches=%llu cpu_gpu=%llu cpu_libref=%llu libref_gpu=%llu mismatches=0 device=%s\n",
                (unsigned long long)n1_cases, (unsigned long long)n1_launches, (unsigned long long)n1_values,
                (unsigned long long)n1_values, (unsigned long long)n1_values, properties.gcnArchName);
            if (argc == 3) router_n1_model(argv[2], stream, second_stream);
            for (int rows : {1, 3, 64})
                for (int repeat : {0, 1}) router_check(rows, repeat, repeat ? second_stream : stream);
            gpu(hipStreamDestroy(second_stream)); gpu(hipStreamDestroy(stream));
            std::printf("QWEN_N1_QUALIFICATION_PASS N=1 K=2560 dtype=bf16 M=1..2048 sampled_M=1,3,8,16,32,64,128,512,2048 fixtures=%llu launches=%llu cpu_gpu=%llu cpu_libref=%llu libref_gpu=%llu N512_regression_cases=%llu mismatches=0\n",
                (unsigned long long)n1_cases, (unsigned long long)n1_launches, (unsigned long long)n1_values,
                (unsigned long long)n1_values, (unsigned long long)n1_values, (unsigned long long)router_cases);
            return 0;
        }
        for (int rows : {1, 3, 8, 16, 32, 64})
            for (int width : {128, 256, 384, 640, 2560, 4096})
                for (bool separate : {false, true})
                    for (int mode : {0, 1, 2}) gate_check(rows, width, separate, mode, stream);
        for (int rows : {10, 30, 80, 160, 320, 640, 2048, 20480}) gate_check(rows, 384, false, 0, stream);
        for (int rows : {1, 3, 8, 16, 32, 64})
            for (int repeat : {0, 1}) router_check(rows, repeat, repeat ? second_stream : stream);
        for (int tokens : {1, 3, 8, 16, 32, 64, 2048}) gate_check(tokens * 12, 256, true, 1, stream, true);
        gpu(hipStreamDestroy(second_stream));
        gpu(hipStreamDestroy(stream));
        std::printf("QWEN_GPU_PASS gate_cases=%llu router_cases=%llu gather_cases=%llu comparisons=%llu existing_topk_weight_ulp_total=%llu device=%s\n",
            (unsigned long long)gate_cases, (unsigned long long)router_cases,
            (unsigned long long)gather_cases,
            (unsigned long long)comparisons, (unsigned long long)routing_weight_ulps, properties.gcnArchName);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "QWEN_FAIL %s gate_cases=%llu router_cases=%llu\n", error.what(),
            (unsigned long long)gate_cases, (unsigned long long)router_cases);
        return 1;
    }
}
