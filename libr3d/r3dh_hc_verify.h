#include "rad_planes.h"
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <map>
#include <regex>
#include <sstream>

struct ModelTensor {
    std::string dtype, shape;
    uint64_t offset = 0;
    std::map<std::string, uint64_t> planes;
};
struct ModelManifest {
    std::map<std::string, ModelTensor> tensors;
    std::ifstream model;
    uint64_t bytes;
    ModelManifest(const char* path, const char* manifest) : model(path, std::ios::binary),
        bytes(std::filesystem::file_size(path)) {
        require(model.good() && bytes == 121678372864ULL, "verified model size mismatch");
        char magic[4]{}; uint32_t version = 0;
        model.read(magic, 4); model.read(reinterpret_cast<char*>(&version), 4);
        require(!std::memcmp(magic, "RAD1", 4) && version == 2, "model RAD1 version 2 header mismatch");
        std::ifstream input(manifest);
        require(input.good(), "cannot open rad-info manifest");
        const std::regex header(R"(^      (\S+)\s+(\S+)\s+(\S+)\s+.*@([0-9]+).*$)");
        const std::regex plane(R"(^          plane\s+(\S+)\s+(\S+)\s+.*@([0-9]+).*$)");
        std::string line, current;
        std::smatch match;
        while (std::getline(input, line)) {
            if (std::regex_match(line, match, header)) {
                current = match[1];
                if (current.find("hc_") == std::string::npos) { current.clear(); continue; }
                require(!tensors.contains(current), "duplicate model tensor");
                tensors[current] = {match[2], match[3], std::stoull(match[4]), {}};
            } else if (!current.empty() && std::regex_match(line, match, plane)) {
                require(tensors[current].planes.emplace(match[1], std::stoull(match[3])).second,
                        "duplicate model plane");
            }
        }
        std::printf("hc_model,path=%s,manifest=%s,bytes=%llu,identity=previously_sha256_verified\n",
                    path, manifest, static_cast<unsigned long long>(bytes));
    }
    const ModelTensor& get(const std::string& name, const char* dtype, const std::string& shape) {
        require(tensors.contains(name), "missing HC tensor in manifest");
        const auto& result = tensors.at(name);
        require(result.dtype == dtype && result.shape == shape, "HC model dtype/shape mismatch");
        return result;
    }
    void read(uint64_t offset, void* destination, size_t count) {
        require(offset <= bytes && count <= bytes - offset, "model plane outside file");
        model.clear(); model.seekg(static_cast<std::streamoff>(offset));
        model.read(static_cast<char*>(destination), std::streamsize(count));
        require(model.good(), "short model plane read");
    }
    void matrix(const std::string& name, Matrix& result) {
        const std::string dtype = "fp8_e4m3*f32[1x" + std::to_string(result.group) + "]";
        const auto& entry = get(name, dtype.c_str(), std::to_string(result.rows) + "x" +
                               std::to_string(result.columns));
        require(entry.planes.size() == 2 && entry.planes.contains("codes") &&
                entry.planes.contains("scale") && entry.offset == entry.planes.at("codes"),
                "HC codes/scale manifest invalid");
        const uint64_t code_offset = entry.planes.at("codes"), scale_offset = entry.planes.at("scale");
        require(code_offset + result.codes.size() <= scale_offset, "HC model plane overlap");
        read(code_offset, result.codes.data(), result.codes.size());
        read(scale_offset, result.scales.data(), result.scales.size() * sizeof(float));
        for (uint8_t code : result.codes) require((code & 127) != 127, "HC NaN E4M3 code");
        for (float scale : result.scales)
            require(std::isfinite(scale) && scale > 0, "HC invalid model scale");
        std::printf("hc_model_tensor,name=%s,codes=%llu,scales=%llu,group=%d\n", name.c_str(),
                    static_cast<unsigned long long>(code_offset),
                    static_cast<unsigned long long>(scale_offset), result.group);
    }
    std::vector<uint16_t> bf16(const std::string& name, const std::string& shape, size_t count) {
        const auto& entry = get(name, "bf16", shape);
        std::vector<uint16_t> result(count);
        read(entry.offset, result.data(), count * 2);
        for (uint16_t value : result) require(std::isfinite(widen(value)), "HC nonfinite weight");
        std::printf("hc_model_tensor,name=%s,offset=%llu,dtype=bf16\n", name.c_str(),
                    static_cast<unsigned long long>(entry.offset));
        return result;
    }
};
using ReferenceRead = int (*)(const RadArgs*, RadStream);
ReferenceRead reference_read = nullptr;
struct ReferenceTwins {
    std::vector<uint8_t> codes, rotated;
    std::vector<float> scales, rotated_scales;
} reference_twins;
void libref_twins_check(const DeviceBuffer& codes, const DeviceBuffer& scales,
                        const DeviceBuffer& rotated, const DeviceBuffer& rotated_scales) {
    std::vector<uint8_t> actual_codes(reference_twins.codes.size()), actual_rotated(actual_codes.size());
    std::vector<float> actual_scales(reference_twins.scales.size()), actual_rotated_scales(actual_scales.size());
    codes.download(actual_codes.data()); scales.download(actual_scales.data());
    rotated.download(actual_rotated.data()); rotated_scales.download(actual_rotated_scales.data());
    require(actual_codes == reference_twins.codes && actual_rotated == reference_twins.rotated,
            "HC twin codes differ from existing libref");
    require(actual_scales == reference_twins.scales && actual_rotated_scales == reference_twins.rotated_scales,
            "HC twin scales differ from existing libref");
    std::puts("hc_twins_vs_existing_libref,codes_scales_both_pairs=BIT_EXACT");
}
int gate_failures = 0;
const char* fixture_name = "synthetic";
std::vector<uint8_t> joined(Matrix& matrix) {
    RadEncoding encoding{};
    rad_enc_copy_str(encoding.scheme, "affine"); encoding.n_planes = 2;
    rad_enc_copy_str(encoding.plane[0].role, "codes");
    encoding.plane[0].dtype = RAD_F8E4M3; encoding.plane[0].kind = RAD_PLANE_TILED;
    encoding.plane[0].block[0] = encoding.plane[0].block[1] = 1;
    rad_enc_copy_str(encoding.plane[1].role, "scale");
    encoding.plane[1].dtype = RAD_F32; encoding.plane[1].kind = RAD_PLANE_TILED;
    encoding.plane[1].block[0] = 1; encoding.plane[1].block[1] = matrix.group;
    RadTensor planes[] = {tensor(matrix.codes.data(), RAD_F8E4M3, matrix.rows, matrix.columns),
        tensor(matrix.scales.data(), RAD_F32, matrix.rows, matrix.columns / matrix.group)};
    const int selection[] = {0, 1};
    RadLayout layout{};
    require(rad_planes_layout(&encoding, selection, planes, 2, &layout) == RAD_OK, "libref layout");
    std::vector<uint8_t> result(size_t(layout.bytes));
    require(rad_planes_relayout(&encoding, selection, planes, 2, result.data(), layout.bytes) == RAD_OK,
            "libref joined planes");
    return result;
}
void libref_check(Matrix& down, Matrix& up, std::vector<uint16_t>& hidden,
                  std::vector<float>& gain, std::vector<uint16_t>& weights,
                  int rows, bool inject, const std::vector<RadParam>& params,
                  std::vector<uint16_t>& output, std::vector<uint16_t>& gates,
                  uint32_t quant_dtype = RAD_F8E4M3) {
    reference_twins.codes.resize(size_t(rows) * 2560);
    reference_twins.rotated.resize(reference_twins.codes.size());
    reference_twins.scales.resize(size_t(rows) * 20);
    reference_twins.rotated_scales.resize(reference_twins.scales.size());
    auto jd = joined(down), ju = joined(up);
    RadTensor operands[] = {tensor(hidden.data(), RAD_BF16, rows, 10240),
        tensor(gain.data(), RAD_F32, 1, 10240), tensor(jd.data(), RAD_F8E4M3, 1, int64_t(jd.size())),
        tensor(ju.data(), RAD_F8E4M3, 1, int64_t(ju.size())),
        tensor(inject ? weights.data() : nullptr, RAD_BF16, 4, 10240),
        tensor(output.data(), RAD_BF16, rows, 2560),
        tensor(inject ? gates.data() : nullptr, RAD_BF16, rows, 4),
        tensor(reference_twins.codes.data(), quant_dtype, rows, 2560),
        tensor(reference_twins.scales.data(), RAD_F32, rows, 20),
        tensor(reference_twins.rotated.data(), quant_dtype, rows, 2560),
        tensor(reference_twins.rotated_scales.data(), RAD_F32, rows, 20)};
    RadArgs args{}; args.t = operands; args.n_t = 11; args.p = params.data(); args.n_p = int(params.size());
    require(reference_read && reference_read(&args, nullptr) == RAD_OK, "existing libref hc_read failed");
}
struct Checkpoints {
    std::vector<float> scales, normalized, down_sum, intermediate;
};
Checkpoints oracle_points;
void strict_checkpoint_check(const DeviceBuffer& scratch, int rows) {
    std::vector<uint16_t> normalized(size_t(rows) * 10240), intermediate(size_t(rows) * 320);
    check_hip(hipMemcpy(normalized.data(), scratch.data(), normalized.size() * 2, hipMemcpyDeviceToHost));
    const size_t offset = (normalized.size() * 2 + 255) & ~size_t(255);
    check_hip(hipMemcpy(intermediate.data(), static_cast<uint8_t*>(scratch.data()) + offset,
                        intermediate.size() * 2, hipMemcpyDeviceToHost));
    int norm_different = 0, down_different = 0;
    for (size_t index = 0; index < normalized.size(); ++index)
        if (normalized[index] != narrow(oracle_points.normalized[index])) ++norm_different;
    for (size_t index = 0; index < intermediate.size(); ++index)
        if (intermediate[index] != narrow(oracle_points.intermediate[index])) ++down_different;
    std::printf("hc_strict_checkpoints,norm_bf16_different=%d,down_bf16_different=%d\n",
                norm_different, down_different);
    if (norm_different || down_different) ++gate_failures;
}

void checkpoint_check(const DeviceBuffer& scratch, const std::vector<uint16_t>& hidden,
                      const std::vector<float>& gain, const Matrix& down, int rows, float wadd) {
    std::vector<float> scales(size_t(rows) * 4);
    check_hip(hipMemcpy(scales.data(), scratch.data(), scales.size() * 4, hipMemcpyDeviceToHost));
    int scale_changed = 0, norm_changed = 0, down_changed = 0, norm_cause = 0;
    float scale_error = 0, down_error = 0;
    std::vector<float> normalized(size_t(rows) * 10240);
    for (size_t index = 0; index < scales.size(); ++index) {
        if (scales[index] != oracle_points.scales[index]) ++scale_changed;
        scale_error = std::max(scale_error, std::abs(scales[index] - oracle_points.scales[index]));
    }
    for (size_t index = 0; index < normalized.size(); ++index) {
        normalized[index] = rounded(widen(hidden[index]) * scales[index / 2560] *
                                    (gain[index % 10240] + wadd));
        if (normalized[index] != oracle_points.normalized[index]) ++norm_changed;
    }
    constexpr int stride = 324, slices = 16;
    const size_t offset = (size_t(rows) * 16 + 255) & ~size_t(255);
    std::vector<float> partials(size_t(slices) * rows * stride);
    check_hip(hipMemcpy(partials.data(), static_cast<uint8_t*>(scratch.data()) + offset,
                        partials.size() * 4, hipMemcpyDeviceToHost));
    for (int row = 0; row < rows; ++row)
        for (int column = 0; column < 320; ++column) {
            float sum = 0;
            for (int slice = 0; slice < slices; ++slice)
                sum += partials[(size_t(slice) * rows + row) * stride + column];
            sum *= 0.25f;
            const size_t index = size_t(row) * 320 + column;
            down_error = std::max(down_error, std::abs(sum - oracle_points.down_sum[index]));
            const float value = rounded(sum * sigmoid(sum));
            if (value != oracle_points.intermediate[index]) {
                ++down_changed;
                float same_norm = 0;
                for (int channel = 0; channel < 10240; ++channel)
                    same_norm += normalized[size_t(row) * 10240 + channel] * down.at(column, channel);
                same_norm *= 0.25f;
                if (rounded(same_norm * sigmoid(same_norm)) != oracle_points.intermediate[index]) ++norm_cause;
                if (down_changed <= 3)
                    std::printf("hc_down_first,row=%d,col=%d,cpu_sum=%a,gpu_sum=%a,cpu_d=%a,gpu_partial_d=%a,same_norm_serial_d=%a\n",
                        row, column, oracle_points.down_sum[index], sum, oracle_points.intermediate[index],
                        value, rounded(same_norm * sigmoid(same_norm)));
            }
        }
    std::printf("hc_checkpoints,scale_different=%d,scale_max_abs=%g,norm_bf16_different=%d,down_bf16_different=%d,down_norm_cause=%d,down_max_abs=%g\n",
                scale_changed, scale_error, norm_changed, down_changed, norm_cause, down_error);
}

void trace_check(const DeviceBuffer& d_buffer, const DeviceBuffer& u_buffer, const DeviceBuffer& g_buffer,
                 const Matrix& up, const std::vector<uint16_t>& actual,
                 const std::vector<uint16_t>& expected, int rows) {
    std::vector<uint16_t> down(size_t(rows) * 320), gates(size_t(rows) * 10240);
    std::vector<float> logits(gates.size());
    d_buffer.download(down.data()); u_buffer.download(logits.data()); g_buffer.download(gates.data());
    int down_changed = 0, up_changed = 0, sigmoid_changed = 0, gate_changed = 0;
    float max_up_error = 0;
    for (size_t index = 0; index < down.size(); ++index)
        if (down[index] != narrow(oracle_points.intermediate[index])) ++down_changed;
    std::vector<uint16_t> serial_gates(gates.size());
    for (int row = 0; row < rows; ++row)
        for (int channel = 0; channel < 10240; ++channel) {
            const size_t index = size_t(row) * 10240 + channel;
            float sum = 0, baseline = 0;
            for (int column = 0; column < 320; ++column) {
                const float weight = up.at(channel, column);
                sum += widen(down[size_t(row) * 320 + column]) * weight;
                baseline += oracle_points.intermediate[size_t(row) * 320 + column] * weight;
            }
            require(std::isfinite(logits[index]), "nonfinite trace up logit");
            max_up_error = std::max(max_up_error, std::abs(sum - logits[index]));
            serial_gates[index] = narrow(sigmoid(sum));
            if (serial_gates[index] != narrow(sigmoid(logits[index]))) ++up_changed;
            if (gates[index] != narrow(sigmoid(logits[index]))) ++sigmoid_changed;
            if (gates[index] != narrow(sigmoid(baseline))) ++gate_changed;
        }
    int reconstructed_changed = 0, down_only_changed = 0, remaining_changed = 0;
    for (int row = 0; row < rows; ++row)
        for (int column = 0; column < 2560; ++column) {
            float mixed = 0, serial = 0;
            for (int stream = 0; stream < 4; ++stream) {
                const size_t index = size_t(row) * 10240 + stream * 2560 + column;
                mixed += widen(gates[index]) * oracle_points.normalized[index];
                serial += widen(serial_gates[index]) * oracle_points.normalized[index];
            }
            const size_t index = size_t(row) * 2560 + column;
            if (narrow(mixed * 0.25f) != actual[index]) ++reconstructed_changed;
            if (narrow(serial * 0.25f) != expected[index]) ++down_only_changed;
            if (narrow(serial * 0.25f) != actual[index]) ++remaining_changed;
        }
    std::printf("hc_trace,actual_down_bf16_different=%d,up_same_d_gate_different=%d,sigmoid_same_logit_gate_different=%d,total_gate_different=%d,up_same_d_max_abs=%g,mix_reconstruction_different=%d,down_only_output_different=%d,remaining_output_different=%d\n",
                down_changed, up_changed, sigmoid_changed, gate_changed, max_up_error,
                reconstructed_changed, down_only_changed, remaining_changed);
    require(reconstructed_changed == 0, "GPU gates fail independent mix reconstruction");
    if (down_changed || up_changed || sigmoid_changed || gate_changed || max_up_error != 0 ||
        down_only_changed || remaining_changed) ++gate_failures;
}
