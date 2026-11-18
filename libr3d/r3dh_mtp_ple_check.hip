#include <hip/hip_runtime.h>
#include "r3dh_mtp_ple.h"
#include "rad_format.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <vector>
extern "C" int ref_mtp_enter(const RadArgs*, RadStream);
extern "C" int ref_ple_gate(const RadArgs*, RadStream);
extern "C" int ref_grid_embed(const RadArgs*, RadStream);
extern "C" int ref_embed_lookup_q(const RadArgs*, RadStream);
extern "C" int ref_ngram_ids(const RadArgs*, RadStream);
extern "C" int ref_logit_rerank(const RadArgs*, RadStream);
extern "C" int ref_row_topk_merge(const RadArgs*, RadStream);
namespace {
bool gpu = false;
bool batch_mode = false;
hipStream_t queue = nullptr;
int cases = 0;
uint64_t comparisons = 0;
void expect(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }
void hipcheck(hipError_t status) { if (status != hipSuccess) throw std::runtime_error(hipGetErrorString(status)); }
struct Blob;
std::vector<Blob*> allocations;
struct Blob {
    std::vector<uint8_t> bytes;
    uint8_t* device = nullptr;
    uint32_t dtype;
    Blob(size_t count, uint32_t type) : bytes(count * (rad_dtype_bits(type) / 8) + 128, 0xa5), dtype(type) {
        if (gpu) hipcheck(hipMalloc(&device, bytes.size()));
        allocations.push_back(this);
    }
    ~Blob() {
        allocations.erase(std::find(allocations.begin(), allocations.end(), this));
        if (device) (void)hipFree(device);
    }
    size_t size() const { return (bytes.size() - 128) / (rad_dtype_bits(dtype) / 8); }
    template<class Value> Value* data() { return reinterpret_cast<Value*>(bytes.data() + 64); }
    void push() { if (gpu) hipcheck(hipMemcpyAsync(device, bytes.data(), bytes.size(), hipMemcpyHostToDevice, queue)); }
    void compare(const char* label) {
        if (!gpu) return;
        std::vector<uint8_t> observed(bytes.size());
        hipcheck(hipMemcpy(observed.data(), device, observed.size(), hipMemcpyDeviceToHost));
        for (size_t index = 0; index < bytes.size(); ++index) {
            if (bytes[index] != observed[index]) {
                std::fprintf(stderr, "FAIL %s byte=%zu expected=%02x actual=%02x dtype=%u\n", label, index, bytes[index], observed[index], dtype);
                throw std::runtime_error(label);
            }
        }
        comparisons += bytes.size();
    }
};
RadTensor tensor(Blob& blob, std::initializer_list<int64_t> shape, int64_t pitch = 0) {
    RadTensor result{}; result.data = blob.data<void>(); result.dtype = blob.dtype; result.rank = uint32_t(shape.size());
    int axis = 0; for (int64_t length : shape) result.shape[axis++] = length;
    int64_t stride = 1;
    for (axis = int(result.rank) - 1; axis >= 0; --axis) { result.stride[axis] = stride; stride *= result.shape[axis]; }
    if (pitch) result.stride[0] = pitch;
    return result;
}
void fill(Blob& blob, int seed) {
    for (size_t index = 0; index < blob.size(); ++index) {
        uint32_t mixed = uint32_t(index) ^ (uint32_t(seed) * 0x9e3779b9u);
        mixed = (mixed ^ (mixed >> 16)) * 0x7feb352du;
        mixed = (mixed ^ (mixed >> 15)) * 0x846ca68bu;
        mixed ^= mixed >> 16;
        const float value = batch_mode ? float(int(mixed % 1024) - 512) / 512.0f :
            float(int((index * 37 + size_t(seed) * 11) % 103) - 51) / 64.0f;
        if (blob.dtype == RAD_BF16) blob.data<uint16_t>()[index] = rad_f32_to_bf16(value);
        else if (blob.dtype == RAD_F32) blob.data<float>()[index] = value;
        else if (blob.dtype == RAD_F8E4M3) blob.data<uint8_t>()[index] = uint8_t((index * 13 + seed) % 111) | (index % 2 ? 128 : 0);
    }
}
using Function = int (*)(const RadArgs*, RadStream);
#include "r3dh_mtp_ple_check_batch.h"
void run(const char* label, Function native, Function reference, std::vector<RadParam> params,
         std::vector<RadTensor> operands, std::vector<RadTensor> oracle = {}, bool sampled = false) {
    if (batch_mode) {
        params.push_back(RAD_STR("dtype", "bf16"));
        native = registered_launch(native, params);
    }
    RadArgs args{}; args.t = operands.data(); args.n_t = int(operands.size()); args.p = params.data(); args.n_p = int(params.size());
    Blob scratch(native == r3dh_mtp_ple_mtp_enter ? size_t(r3dh_mtp_ple_mtp_enter_scratch(&args)) : 16, RAD_I8);
    for (Blob* blob : allocations) blob->push();
    if (batch_mode && gpu) hipcheck(hipStreamSynchronize(queue));
    std::vector<RadTensor> remote = operands;
    for (RadTensor& item : remote) {
        if (!item.data) continue;
        bool mapped = false;
        for (Blob* blob : allocations) if (item.data == blob->data<void>()) {
            item.data = blob->device ? blob->device + 64 : nullptr; mapped = true; break;
        }
        expect(mapped, "map all native operands");
    }
    if (!oracle.empty()) args.t = oracle.data();
    std::vector<int64_t> checked_rows;
    if (sampled) {
        checked_rows = sample_rows(operands[0].shape[0]);
        sampled_reference(args, reference, params, checked_rows);
    } else expect(reference(&args, nullptr) == RAD_OK, label);
    if (gpu) {
        args.t = operands.data(); args.scratch = scratch.data<void>(); args.scratch_bytes = int64_t(scratch.size());
        expect(native(&args, reinterpret_cast<RadStream>(queue)) == RAD_E_DEVICE, "reject host operands without CPU fallback");
        const RadTensor saved = remote[0]; remote[0] = operands[0];
        args.t = remote.data(); args.scratch = scratch.device + 64;
        expect(native(&args, reinterpret_cast<RadStream>(queue)) == RAD_E_DEVICE, "reject mixed host/device operands");
        remote[0] = saved;
        args.t = remote.data(); args.scratch = scratch.device + 64; args.scratch_bytes = int64_t(scratch.size());
        if (batch_mode && native == r3dh_mtp_ple_mtp_enter) {
            --args.scratch_bytes;
            expect(native(&args, reinterpret_cast<RadStream>(queue)) == RAD_E_SCRATCH, "short native MTP scratch rejected");
            ++args.scratch_bytes;
            args.scratch = scratch.device + 65;
            expect(native(&args, reinterpret_cast<RadStream>(queue)) == RAD_E_SCRATCH, "misaligned native MTP scratch rejected");
            args.scratch = scratch.device + 64;
        }
        expect(native(&args, reinterpret_cast<RadStream>(queue)) == RAD_OK, label);
        hipcheck(hipStreamSynchronize(queue));
        for (Blob* blob : allocations) if (blob != &scratch) {
            if (sampled && blob->data<void>() == operands[6].data)
                compare_sampled(*blob, operands[6], checked_rows, label);
            else blob->compare(label);
        }
        std::vector<uint8_t> guard(scratch.bytes.size());
        hipcheck(hipMemcpy(guard.data(), scratch.device, guard.size(), hipMemcpyDeviceToHost));
        expect(std::equal(guard.begin(), guard.begin() + 64, scratch.bytes.begin()) &&
            std::equal(guard.end() - 64, guard.end(), scratch.bytes.end() - 64), "scratch guards");
    }
    if (batch_mode) std::printf("COVERAGE %s oracle=%s full_output_certified=%d checked_rows=%zu total_rows=%lld all_columns=1 scratch_bytes=%zu\n",
        label, sampled ? "libref_sampled_rows" : "libref_exhaustive", int(!sampled),
        sampled ? checked_rows.size() : size_t(operands[0].shape[0]),
        static_cast<long long>(operands[0].shape[0]), scratch.size());
    std::printf("PASS %s device=%d reference_ok=1 checked_bit_exact=%d exhaustive=%d guarded_inputs_outputs_scratch=%d\n", label, int(gpu), int(gpu), int(!sampled), int(gpu));
    ++cases;
}
#include "r3dh_mtp_ple_check_model.h"
void mtp_case(int64_t count, int64_t width, int64_t groups, bool sampled = false) {
    const int64_t streams = 4, wide = width * streams, row_bytes = (width + 4 * (width / 128) + 15) & ~int64_t(15);
    Blob hidden(count * (wide + 8), RAD_BF16), embedding(count * (width + 8), RAD_BF16), wh(wide, RAD_F32), we(width, RAD_F32);
    Blob codes(width * width, RAD_F8E4M3), scales(width * (width / 128), RAD_F32), packed_h(width * row_bytes, RAD_F8E4M3), packed_e(width * row_bytes, RAD_F8E4M3);
    Blob decoded_h(width * width, RAD_F32), decoded_e(width * width, RAD_F32), output(count * (wide + 8), RAD_BF16);
    std::vector<RadParam> params{RAD_INT("M", count), RAD_INT("n", width), RAD_INT("hc", streams), RAD_INT("ngroup", groups), RAD_STR("fc", "e4m3"), RAD_F64("eps", 1e-6), RAD_F64("wadd", 1.0)};
    RadEncoding encoding = rad_enc_affine(RAD_F8E4M3, RAD_F32, 1, 128);
    const int selection[]{0, 1};
    fill(wh, 3); fill(we, 5);
    if (current_model) {
        current_model->gain("mtp.pre_fc_norm_hidden.weight", wh);
        current_model->gain("mtp.pre_fc_norm_embedding.weight", we);
    }
    for (int half = 0; half < 2; ++half) {
        fill(codes, half + 3);
        for (size_t index = 0; index < scales.size(); ++index) scales.data<float>()[index] = 0.001013f * float(1 + index % 7);
        if (current_model) current_model->matrix(half ? "mtp.fc_embedding.weight" : "mtp.fc_hidden.weight", codes, scales, width);
        RadTensor planes[]{tensor(codes, {width, width}), tensor(scales, {width, width / 128})};
        RadLayout layout{};
        expect(r3dh_mtp_ple_layout_mtp_e4m3(params.data(), int(params.size()), half + 4, &encoding, selection, planes, 2,
            &layout) == RAD_OK && layout.dtype == r3dh_mtp_fc_e4m3_dtype && layout.shape[0] == width &&
            layout.shape[1] == row_bytes && layout.bytes == width * row_bytes, "MTP private packed layout descriptor");
        Blob& packed = half ? packed_e : packed_h; Blob& decoded = half ? decoded_e : decoded_h;
        expect(r3dh_mtp_ple_relayout_mtp_e4m3(params.data(), int(params.size()), half + 4, &encoding, selection, planes, 2,
            packed.data<void>(), int64_t(packed.size())) == RAD_OK, "MTP pack");
        for (int64_t row = 0; row < width; ++row) for (int64_t column = 0; column < width; ++column)
            decoded.data<float>()[row * width + column] = rad_load_f32(codes.data<void>(), RAD_F8E4M3, row * width + column) * scales.data<float>()[row * (width / 128) + column / 128];
        std::vector<uint8_t> original_codes(codes.data<uint8_t>(), codes.data<uint8_t>() + codes.size());
        std::vector<float> original_scales(scales.data<float>(), scales.data<float>() + scales.size());
        expect(r3dh_mtp_ple_unrelayout_mtp_e4m3(params.data(), int(params.size()), half + 4, &encoding, selection,
            packed.data<void>(), int64_t(packed.size()), planes, 2) == RAD_OK, "MTP unpack");
        expect(!std::memcmp(original_codes.data(), codes.data<void>(), codes.size()) &&
            !std::memcmp(original_scales.data(), scales.data<void>(), scales.size() * 4), "MTP layout exact roundtrip");
    }
    auto operands = std::vector<RadTensor>{tensor(hidden, {count, wide}, wide + 8), tensor(embedding, {count, width}, width + 8),
        tensor(wh, {wide}), tensor(we, {width}), tensor(packed_h, {width, row_bytes}), tensor(packed_e, {width, row_bytes}), tensor(output, {count, wide}, wide + 8)};
    auto oracle = operands; oracle[4] = tensor(decoded_h, {width, width}); oracle[5] = tensor(decoded_e, {width, width});
    for (int repeat = 0; repeat < 2; ++repeat) {
        fill(hidden, 2 + repeat); fill(embedding, 9 + repeat);
        if (sampled) for (int64_t row = 0; row < count; ++row)
            std::fill_n(output.data<uint16_t>() + row * (wide + 8), wide, uint16_t(0x7fc1));
        const std::string label = "mtp_e4m3 M=" + std::to_string(count) + " n=" + std::to_string(width) + " ngroup=" + std::to_string(groups) + " changed=" + std::to_string(repeat) + " real_weights=" + std::to_string(current_model != nullptr);
        run(label.c_str(), r3dh_mtp_ple_mtp_enter, ref_mtp_enter, params, operands, oracle, sampled);
    }
}
void gate_case(int64_t count) {
    const int64_t width = 2560, streams = 4, wide = width * streams;
    Blob key(count * (wide + 8), RAD_BF16), query(count * (wide + 8), RAD_BF16), value(count * (width + 8), RAD_BF16);
    Blob wk(wide, RAD_F32), wq(wide, RAD_F32), wc(wide, RAD_F32), gv(count * (wide + 8), RAD_BF16), gvn(count * (wide + 8), RAD_BF16);
    fill(wk, 2); fill(wq, 7); fill(wc, 5);
    if (current_model) {
        current_model->gain("blk.1.ple_norm_key.weight", wk);
        current_model->gain("blk.1.ple_norm_query.weight", wq);
        current_model->gain("blk.1.ple_norm_conv.weight", wc);
    }
    for (int repeat = 0; repeat < 2; ++repeat) {
        fill(key, 1 + repeat); fill(query, 3 + repeat); fill(value, 9 + repeat);
        if (repeat) std::fill(key.data<uint16_t>(), key.data<uint16_t>() + key.size(), uint16_t(0));
        const std::string label = "ple_gate M=" + std::to_string(count) + " zero_gate=" + std::to_string(repeat) + " real_weights=" + std::to_string(current_model != nullptr);
        run(label.c_str(), r3dh_mtp_ple_ple_gate, ref_ple_gate,
            {RAD_INT("M", count), RAD_INT("n", width), RAD_INT("hc", streams), RAD_F64("eps", 1e-6), RAD_F64("wadd", 1.0), RAD_F64("gate_eps", 1e-6)},
            {tensor(key, {count, wide}, wide + 8), tensor(query, {count, wide}, wide + 8), tensor(value, {count, width}, width + 8),
             tensor(wk, {wide}), tensor(wq, {wide}), tensor(wc, {wide}), tensor(gv, {count, wide}, wide + 8), tensor(gvn, {count, wide}, wide + 8)});
    }
}
void embedding_case(int64_t count) {
    const int64_t width = 128, vocabulary = 19, side = 4;
    Blob ids(count, RAD_I32), table(vocabulary * width, RAD_F8E4M3), scale(1, RAD_BF16), output(count * (width + 8), RAD_BF16);
    fill(table, 9); scale.data<uint16_t>()[0] = rad_f32_to_bf16(0.0131f);
    for (int optional = 0; optional < 2; ++optional) {
        for (int64_t row = 0; row < count; ++row) ids.data<int32_t>()[row] = int32_t((row * 3 + optional) % vocabulary);
        run("embed_lookup_q E4M3 optional_scale", r3dh_mtp_ple_embed_lookup_q, ref_embed_lookup_q,
            {RAD_INT("M", 32), RAD_INT("n_embd", width), RAD_INT("n_vocab", vocabulary)},
            {tensor(ids, {count}), tensor(table, {vocabulary, width}), optional ? tensor(scale, {1}) : RadTensor{}, tensor(output, {count, width}, width + 8)});
    }
    Blob grid_table(side * side * width, RAD_BF16), coordinates(4 * count, RAD_I32);
    fill(grid_table, 3);
    for (int repeat = 0; repeat < 2; ++repeat) {
        fill(output, repeat + 2);
        for (int64_t row = 0; row < count; ++row) {
            coordinates.data<int32_t>()[row] = int32_t(row % 7) - 2;
            coordinates.data<int32_t>()[count + row] = int32_t(row % 9) - 1;
            coordinates.data<int32_t>()[2 * count + row] = int32_t(row % 4) + repeat;
            coordinates.data<int32_t>()[3 * count + row] = int32_t(row % 5) + 1;
        }
        run("grid_embed clamped_taps", r3dh_mtp_ple_grid_embed, ref_grid_embed,
            {RAD_INT("M", 32), RAD_INT("n", width), RAD_INT("side", side)},
            {tensor(output, {count, width}, width + 8), tensor(grid_table, {side * side, width}), tensor(coordinates, {4, count})});
    }
}
#include "r3dh_mtp_ple_check_extra.h"
void inspect(const char* path) {
    std::ifstream input(path, std::ios::binary); expect(bool(input), "open RAD");
    input.seekg(0, std::ios::end); const uint64_t size = uint64_t(input.tellg());
    RadFileHeader header{}; input.seekg(0); input.read(reinterpret_cast<char*>(&header), sizeof(header));
    expect(bool(input) && header.magic == RAD_MAGIC && header.version == RAD_FORMAT_VER && header.file_bytes == size, "RAD header identity");
    expect(header.str_bytes < 16000000 && header.dir_count < 1000000 && header.enc_count < 100000, "RAD metadata bounds");
    std::vector<char> strings(size_t(header.str_bytes)); std::vector<RadFileEntry> directory(size_t(header.dir_count)); std::vector<RadEncoding> encodings(size_t(header.enc_count));
    input.seekg(header.str_off); input.read(strings.data(), std::streamsize(strings.size()));
    input.seekg(header.dir_off); input.read(reinterpret_cast<char*>(directory.data()), std::streamsize(directory.size() * sizeof(RadFileEntry)));
    input.seekg(header.enc_off); input.read(reinterpret_cast<char*>(encodings.data()), std::streamsize(encodings.size() * sizeof(RadEncoding))); expect(bool(input), "read RAD metadata");
    std::printf("MODEL bytes=%llu\n", static_cast<unsigned long long>(size));
    for (const auto& entry : directory) {
        expect(entry.name < strings.size() && entry.enc < encodings.size() && std::memchr(strings.data() + entry.name, 0, strings.size() - entry.name), "RAD entry bounds");
        const char* name = strings.data() + entry.name;
        if (!std::strstr(name, "mtp") && !std::strstr(name, "ple") && !std::strstr(name, "output.weight")) continue;
        const auto& encoding = encodings[entry.enc];
        std::printf("WEIGHT %s shape=%lld,%lld scheme=%s transform=%s planes=%u first=%llu\n", name, static_cast<long long>(entry.shape[0]), static_cast<long long>(entry.shape[1]), encoding.scheme, encoding.transform, entry.n_planes, static_cast<unsigned long long>(entry.plane_first));
    }
}
}
#include "r3dh_mtp_ple_check_contract.h"
#include "r3dh_mtp_ple_check_batch_suite.h"
#include "r3dh_mtp_ple_check_ngram.h"
int main(int argc, char** argv) {
    try {
        if (argc == 3 && (!std::strcmp(argv[1], "--ngram-host") || !std::strcmp(argv[1], "--ngram"))) {
            ngram_focused_suite(argv[2], !std::strcmp(argv[1], "--ngram")); return 0;
        }
        if (argc == 3 && !std::strcmp(argv[1], "--inspect")) { inspect(argv[2]); return 0; }
        const bool host_model = argc == 3 && !std::strcmp(argv[1], "--model-host");
        if (argc == 3 && !std::strcmp(argv[1], "--batch")) { batch_suite(argv[2]); return 0; }
        if (argc == 2 && !std::strcmp(argv[1], "--batch-host")) { batch_contracts(); return 0; }
        const bool real_weights = argc == 3 && (!std::strcmp(argv[1], "--model") || host_model);
        expect(real_weights || (argc == 2 && (!std::strcmp(argv[1], "--host") || !std::strcmp(argv[1], "--gpu"))), "usage --host|--gpu|--inspect MODEL|--model MODEL");
        gpu = (real_weights && !host_model) || !std::strcmp(argv[1], "--gpu");
        if (gpu) {
            int count = 0; hipcheck(hipGetDeviceCount(&count)); expect(count == 1, "exactly one visible GPU");
            hipDeviceProp_t properties{}; hipcheck(hipGetDeviceProperties(&properties, 0));
            expect(std::strstr(properties.gcnArchName, "gfx1100"), "native gfx1100 required");
            hipcheck(hipStreamCreateWithFlags(&queue, hipStreamNonBlocking));
            std::printf("DEVICE %s arch=%s nondefault_stream=1\n", properties.name, properties.gcnArchName);
        }
        contract_checks(); analytic_checks();
        for (int64_t count : {1, 3, 8, 16, 32}) { mtp_case(count, gpu ? 2560 : 128, 4); gate_case(count); embedding_case(count); ngram_case(count); select_case(count); }
        if (real_weights) {
            Model model(argv[2]); current_model = &model;
            for (int64_t count : {1, 3, 8, 16, 32}) { mtp_case(count, 2560, 4); gate_case(count); }
            current_model = nullptr;
        }
        mtp_case(3, 128, 1);
        if (gpu) hipcheck(hipStreamDestroy(queue));
        std::printf("MTP_PLE_PASS device=%d cases=%d compared_bytes=%llu\n", int(gpu), cases, static_cast<unsigned long long>(comparisons));
        return 0;
    } catch (const std::exception& error) { std::fprintf(stderr, "MTP_PLE_FAIL %s\n", error.what()); return 1; }
}
