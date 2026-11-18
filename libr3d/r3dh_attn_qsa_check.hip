#include <hip/hip_runtime.h>
#include "r3dh_attn_qsa.h"
#include "rad_plugin.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <vector>
extern "C" int ref_rope_table(const RadArgs*, RadStream);
extern "C" int ref_rope(const RadArgs*, RadStream);
extern "C" int ref_kv_store(const RadArgs*, RadStream);
extern "C" int ref_attn_dense(const RadArgs*, RadStream);
extern "C" int ref_attn_paged(const RadArgs*, RadStream);
namespace {
bool gpu = false;
hipStream_t stream = nullptr;
int cases = 0;
int device_calls = 0;
int graph_replays = 0;
size_t comparisons = 0;
void require(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }
void hipcheck(hipError_t status) { if (status != hipSuccess) throw std::runtime_error(hipGetErrorString(status)); }
struct Blob;
std::vector<Blob*> allocations;
struct Blob {
    std::vector<uint8_t> bytes;
    uint8_t* device = nullptr;
    uint32_t type;
    Blob(size_t count, uint32_t dtype) : bytes(count * (rad_dtype_bits(dtype) / 8) + 128, 0xa5), type(dtype) {
        if (gpu) hipcheck(hipMalloc(&device, bytes.size()));
        allocations.push_back(this);
    }
    ~Blob() {
        allocations.erase(std::find(allocations.begin(), allocations.end(), this));
        if (device) (void)hipFree(device);
    }
    template<class Value> Value* data() { return reinterpret_cast<Value*>(bytes.data() + 64); }
    void push() { if (gpu) hipcheck(hipMemcpyAsync(device, bytes.data(), bytes.size(), hipMemcpyHostToDevice, stream)); }
    void compare(const char* label, bool approximate) {
        if (!gpu) return;
        std::vector<uint8_t> actual(bytes.size());
        hipcheck(hipMemcpy(actual.data(), device, actual.size(), hipMemcpyDeviceToHost));
        size_t differences = 0;
        float maximum = 0;
        for (size_t offset = 0; offset < bytes.size();) {
            ++comparisons;
            if (offset >= 64 && offset < bytes.size() - 64 && approximate && type == RAD_F32) {
                float wanted, observed;
                std::memcpy(&wanted, bytes.data() + offset, 4); std::memcpy(&observed, actual.data() + offset, 4);
                maximum = std::max(maximum, std::fabs(wanted - observed));
                require(std::isfinite(observed) && std::fabs(wanted - observed) <= 3e-6f, label);
                if (!std::isfinite(observed) || std::fabs(wanted - observed) > 3e-6f)
                    std::fprintf(stderr, "%s offset=%zu expected=%.9g actual=%.9g\n", label, offset - 64, wanted, observed);
                if (std::memcmp(bytes.data() + offset, actual.data() + offset, 4)) ++differences;
                offset += 4;
            } else if (offset >= 64 && offset < bytes.size() - 64 && approximate && type == RAD_BF16) {
                uint16_t wanted, observed;
                std::memcpy(&wanted, bytes.data() + offset, 2); std::memcpy(&observed, actual.data() + offset, 2);
                const float expected = rad_bf16_to_f32(wanted), value = rad_bf16_to_f32(observed);
                maximum = std::max(maximum, std::fabs(expected - value));
                require(std::isfinite(value) && (wanted == observed || (std::abs(int(wanted) - int(observed)) <= 1)
                    || std::fabs(expected - value) < 1e-6f), label);
                if (wanted != observed) ++differences;
                offset += 2;
            } else {
                require(bytes[offset] == actual[offset], label); ++offset;
            }
        }
        std::printf("PASS %s differences=%zu max_abs=%g guards=exact\n", label, differences, maximum);
    }
};
RadTensor tensor(Blob& blob, std::initializer_list<long> shape, long pitch = 0) {
    RadTensor result{}; result.data = blob.data<void>(); result.dtype = blob.type; result.rank = int(shape.size());
    int axis = 0; for (long size : shape) result.shape[axis++] = size;
    long stride = 1;
    for (axis = result.rank - 1; axis >= 0; --axis) { result.stride[axis] = stride; stride *= result.shape[axis]; }
    if (pitch) result.stride[0] = pitch;
    return result;
}
using Function = int (*)(const RadArgs*, RadStream);
void run(const char* label, Function device, Function reference, std::vector<RadParam> params,
         std::vector<RadTensor> tensors, std::initializer_list<Blob*> outputs, bool approximate = false) {
    std::vector<RadTensor> remote = tensors;
    for (Blob* blob : allocations) blob->push();
    if (gpu) hipcheck(hipStreamSynchronize(stream));
    for (RadTensor& operand : remote) {
        if (!operand.data) continue;
        bool found = false;
        for (Blob* blob : allocations) if (operand.data == blob->data<void>()) {
            operand.data = blob->device ? blob->device + 64 : nullptr; found = true; break;
        }
        require(found, "unmapped fixture operand");
    }
    RadArgs args{}; args.t = tensors.data(); args.n_t = int(tensors.size());
    args.p = params.data(); args.n_p = int(params.size());
    require(reference(&args, nullptr) == RAD_OK, label);
    if (gpu) {
        args.t = remote.data();
        require(device(&args, reinterpret_cast<RadStream>(stream)) == RAD_OK, label);
        ++device_calls;
        hipcheck(hipStreamSynchronize(stream));
        for (Blob* output : outputs) output->compare(label, approximate);
    }
    ++cases;
}
float gain(const RadArgs* args, const char* key, float fallback) {
    for (int index = 0; index < args->n_p; ++index)
        if (!std::strcmp(args->p[index].key, key)) return float(args->p[index].dval);
    return fallback;
}
int integer(const RadArgs* args, const char* key) {
    long long value = 0; rad_args_geti(args, key, &value); return int(value);
}
int cpu_prepare(const RadArgs* args, RadStream) {
    const int dim = integer(args, "head_dim"), heads = integer(args, "n_head"), keyheads = integer(args, "n_head_kv");
    const RadTensor* tensors = args->t;
    for (int row = 0; row < tensors[1].shape[0]; ++row) for (int head = 0; head < heads + keyheads; ++head) {
        const bool key = head >= heads;
        const int local = key ? head - heads : head;
        const RadTensor& input = tensors[key ? 1 : 0]; const RadTensor& output = tensors[key ? 7 : 6];
        const auto* source = static_cast<const uint16_t*>(input.data) + row * input.stride[0] + local * dim * (key ? 1 : 2);
        const auto* weights = static_cast<const float*>(tensors[key ? 5 : 4].data);
        auto* destination = static_cast<uint16_t*>(output.data) + row * output.stride[0] + local * dim;
        std::vector<float> values(dim);
        float sum = 0;
        for (int column = 0; column < dim; ++column) { const float value = rad_bf16_to_f32(source[column]); sum += value * value; }
        const float inverse = 1.0f / std::sqrt(sum / float(dim) + gain(args, "eps", 1e-6f));
        for (int column = 0; column < dim; ++column)
            values[column] = rad_bf16_to_f32(rad_f32_to_bf16((rad_bf16_to_f32(source[column]) * inverse)
                * (weights[column] + gain(args, "wadd", 0))));
        for (int column = 0; column < dim; ++column) destination[column] = rad_f32_to_bf16(values[column]);
        const int position = static_cast<const int*>(tensors[3].data)[row];
        if (position >= 0 && position < tensors[2].shape[0]) for (int pair = 0; pair < 32; ++pair) {
            const float* table = static_cast<const float*>(tensors[2].data) + position * 64;
            destination[pair] = rad_f32_to_bf16(values[pair] * table[pair] - values[pair + 32] * table[pair + 32]);
            destination[pair + 32] = rad_f32_to_bf16(values[pair] * table[pair + 32] + values[pair + 32] * table[pair]);
        }
        if (!key && tensors[8].data) std::memcpy(static_cast<uint16_t*>(tensors[8].data)
            + row * tensors[8].stride[0] + local * dim, source + dim, dim * 2);
    }
    return RAD_OK;
}
void fill(Blob& blob, int seed) {
    const size_t count = (blob.bytes.size() - 128) / 2;
    for (size_t index = 0; index < count; ++index)
        blob.data<uint16_t>()[index] = rad_f32_to_bf16(float(int((index * 37 + seed * 71) % 509) - 254) / 128.0f);
}
void table_case() {
    Blob positions(7, RAD_I32), table(200000 * 64, RAD_F32);
    const int values[] = {-1, 0, 1, 43, 4097, 199999, 200000};
    std::memcpy(positions.data<int>(), values, sizeof values);
    std::fill(table.data<float>(), table.data<float>() + 200000 * 64, -17.0f);
    run("rope_table_long_positions", r3dh_attn_qsa_rope_table, ref_rope_table,
        {RAD_INT("rot", 64), RAD_F64("theta", 1e7), RAD_F64("scale", 1), RAD_STR("dtype", "f32")},
        {tensor(positions, {7}), tensor(table, {200000, 64})}, {&table}, true);
    require(table.data<float>()[0] == 1 && table.data<float>()[32] == 0
        && table.data<float>()[64 * 2] == -17, "CPU table fixture invariant");
}
void prepare_case(int rows, bool indexer, int seed) {
    const int dim = indexer ? 128 : 256, heads = indexer ? 0 : 12, keyheads = indexer ? 4 : 1;
    const int kpitch = indexer ? 640 : 269, qpitch = 6157, opitch = 3081;
    Blob qg(rows * qpitch, RAD_BF16), key(rows * kpitch, RAD_BF16), query(rows * opitch, RAD_BF16);
    Blob gate(rows * opitch, RAD_BF16), outkey(rows * kpitch, RAD_BF16);
    Blob positions(rows, RAD_I32), weights(dim, RAD_F32), table(64 * 64, RAD_F32);
    fill(qg, seed); fill(key, seed + 3); fill(outkey, seed + 3);
    for (int column = 0; column < dim; ++column) weights.data<float>()[column] = float(column % 11 - 5) / 32.0f;
    for (int row = 0; row < rows; ++row) positions.data<int>()[row] = row % 64;
    for (int row = 0; row < 64; ++row) for (int pair = 0; pair < 32; ++pair) {
        const float angle = float(row) / std::pow(1e7f, float(2 * pair) / 64);
        table.data<float>()[row * 64 + pair] = std::cos(angle);
        table.data<float>()[row * 64 + 32 + pair] = std::sin(angle);
    }
    RadTensor absent{};
    const bool inplace = seed % 2 == 0;
    run(indexer ? "qsa_qprep_h128_pitched" : "qk_norm_rope_gate_h256", r3dh_attn_qsa_prepare, cpu_prepare,
        {RAD_INT("head_dim", dim), RAD_INT("n_head", heads), RAD_INT("n_head_kv", keyheads),
         RAD_INT("rot", 64), RAD_INT("cs_f32", 1), RAD_F64("eps", 1e-6), RAD_F64("wadd", 1)},
        {indexer ? absent : tensor(qg, {rows, 6144}, qpitch), tensor(key, {rows, keyheads * dim}, kpitch),
         tensor(table, {64, 64}), tensor(positions, {rows}), indexer ? absent : tensor(weights, {dim}),
         tensor(weights, {dim}), indexer ? absent : tensor(query, {rows, 3072}, opitch),
         tensor(inplace ? key : outkey, {rows, keyheads * dim}, kpitch),
         indexer || seed % 2 ? absent : tensor(gate, {rows, 3072}, opitch)},
        indexer ? std::initializer_list<Blob*>{inplace ? &key : &outkey}
                : std::initializer_list<Blob*>{&query, inplace ? &key : &outkey, &gate}, true);
}
void rope_case(int rows, int heads, bool multi, int seed) {
    const int dim = heads == 4 ? 128 : 256, pitch = heads * dim + 7;
    Blob values(rows * pitch, RAD_BF16), positions(rows * (multi ? 3 : 1), RAD_I32);
    fill(values, seed);
    for (int component = 0; component < (multi ? 3 : 1); ++component)
        for (int row = 0; row < rows; ++row) positions.data<int>()[component * rows + row] = row * 7 + component * 11;
    run(multi ? "rope_imrope_11_11_10" : "rope_neox_partial64", r3dh_attn_qsa_rope, ref_rope,
        {RAD_INT("head_dim", dim), RAD_INT("n_head", heads), RAD_INT("n_head_kv", 0), RAD_INT("rotary_dim", 64),
         RAD_F64("theta", 1e7), RAD_F64("scale", 1), RAD_STR("mode", multi ? "imrope" : "neox"),
         RAD_STR("sections", "11 11 10")},
        {tensor(values, {rows, heads * dim}, pitch), multi ? tensor(positions, {3, rows}) : tensor(positions, {rows})}, {&values}, true);
}
void attention_case(int rows, int seed) {
    const int sequences = rows > 1 ? 2 : 1, pages = (rows + 3) / 4 + 4;
    Blob query(rows * 3083, RAD_BF16), key(rows * 267, RAD_BF16), value(rows * 271, RAD_BF16);
    Blob output(rows * 3087, RAD_BF16), kv(pages * 4 * 516, RAD_I8), slots(rows, RAD_I32);
    Blob cu(sequences + 1, RAD_I32), used(sequences, RAD_I32), table(sequences * pages, RAD_I32);
    fill(query, seed); fill(key, seed + 1); fill(value, seed + 2);
    cu.data<int>()[0] = 0; cu.data<int>()[1] = rows > 1 ? 1 : rows;
    if (sequences == 2) cu.data<int>()[2] = rows;
    for (int sequence = 0; sequence < sequences; ++sequence) {
        used.data<int>()[sequence] = cu.data<int>()[sequence + 1] - cu.data<int>()[sequence];
        for (int page = 0; page < pages; ++page) table.data<int>()[sequence * pages + page] = pages - 1 - page - sequence;
        for (int row = cu.data<int>()[sequence]; row < cu.data<int>()[sequence + 1]; ++row) {
            const int local = row - cu.data<int>()[sequence];
            slots.data<int>()[row] = table.data<int>()[sequence * pages + local / 4] * 4 + local % 4;
        }
    }
    run("kv_store_i8_bytes", r3dh_attn_qsa_kv_store, ref_kv_store,
        {RAD_INT("head_dim", 256), RAD_INT("n_head_kv", 1), RAD_INT("block_size", 4), RAD_STR("kv_dtype", "i8")},
        {tensor(key, {rows, 256}, 267), tensor(value, {rows, 256}, 271), tensor(slots, {rows}),
         tensor(kv, {pages, 1, 4, 516})}, {&kv});
    const std::vector<RadParam> params = {RAD_INT("head_dim", 256), RAD_INT("gqa", 12), RAD_INT("causal", 1),
        RAD_INT("block_size", 4), RAD_STR("q_dtype", "bf16"), RAD_STR("kv_dtype", "i8")};
    run("attn_dense_h256_ragged", r3dh_attn_qsa_dense, ref_attn_dense, params,
        {tensor(query, {rows, 12, 256}, 3083), tensor(key, {rows, 256}, 267), tensor(value, {rows, 256}, 271),
         tensor(cu, {sequences + 1}), tensor(output, {rows, 3072}, 3087)}, {&output}, true);
    RadTensor absent{};
    run("attn_paged_i8_h256_ragged", r3dh_attn_qsa_paged, ref_attn_paged, params,
        {tensor(query, {rows, 3072}, 3083), tensor(kv, {pages, 1, 4, 516}), tensor(table, {sequences, pages}),
         tensor(used, {sequences}), absent, absent, tensor(cu, {sequences + 1}), tensor(output, {rows, 3072}, 3087)}, {&output}, true);
    std::fill(slots.data<int>(), slots.data<int>() + rows, -1);
    run("kv_store_negative_slots", r3dh_attn_qsa_kv_store, ref_kv_store,
        {RAD_INT("head_dim", 256), RAD_INT("n_head_kv", 1), RAD_INT("block_size", 4)},
        {tensor(key, {rows, 256}, 267), tensor(value, {rows, 256}, 271), tensor(slots, {rows}), tensor(kv, {pages, 1, 4, 516})}, {&kv});
}
}
#include "r3dh_attn_qsa_check_extra.h"
int main(int argc, char** argv) {
    try {
        const bool state_only = argc == 2 && (!std::strcmp(argv[1], "--state-host") || !std::strcmp(argv[1], "--state-gpu"));
        if (argc > 2 || (argc == 2 && !state_only && std::strcmp(argv[1], "--host") && std::strcmp(argv[1], "--gpu"))) {
            std::fprintf(stderr, "usage: r3dh_attn_qsa_check [--host|--gpu|--state-host|--state-gpu]\n");
            return 2;
        }
        gpu = argc != 2 || (std::strcmp(argv[1], "--host") && std::strcmp(argv[1], "--state-host"));
        if (gpu) {
            int count = 0; hipcheck(hipGetDeviceCount(&count)); require(count == 1, "exactly one GPU required");
            hipDeviceProp_t properties{}; hipcheck(hipGetDeviceProperties(&properties, 0));
            require(!std::strncmp(properties.gcnArchName, "gfx1100", 7), "gfx1100 required");
            std::printf("DEVICE %s arch=%s\n", properties.name, properties.gcnArchName);
            hipcheck(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
        }
        if (!state_only) table_case();
        rejection_cases();
        state_case();
        state_matrix();
        if (!state_only) {
            prepare_case(2048, false, 2);
            prepare_case(2048, true, 5);
            for (int rows : {1, 3, 17}) resident_chain_case(rows);
            decode_case(65, 3, 11);
            decode_case(1025, 1, 19);
            for (int rows : {1, 3, 17, 64}) for (int seed : {2, 5}) {
                prepare_case(rows, false, seed); prepare_case(rows, true, seed);
                rope_case(rows, 12, true, seed); rope_case(rows, 1, false, seed); rope_case(rows, 4, true, seed);
                if (rows <= 17) attention_case(rows, seed);
            }
        }
        if (gpu) hipcheck(hipStreamDestroy(stream));
        std::printf("PASS mode=%s cases=%d independent_cpu_libref strict_fp32 no_f16_staging\n", gpu ? "gfx1100" : "host", cases);
        std::printf("QSA_COUNTS device_calls=%d graph_replays=%d comparisons=%zu\n",
            device_calls, graph_replays, comparisons);
        return 0;
    } catch (const std::exception& failure) { std::fprintf(stderr, "FAIL %s\n", failure.what()); return 1; }
}
