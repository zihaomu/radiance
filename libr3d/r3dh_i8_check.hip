#include <hip/hip_runtime.h>
#include "r3dh_i8.h"
#include "rad_encoding.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#ifdef __FAST_MATH__
#error r3dh_i8 checker requires strict floating point compilation
#endif
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)

namespace {
constexpr const char* model_path = "/home/zihaomu/bigssd2/radiance_workspace/models/qwen3.8-next-flash-iqr-v4.rad";
constexpr const char* model_sha = "25aff8c950f4b823bc8b77c8184ed48827cb47058ffce08ae2c83b12e021fea0";
uint64_t assertions = 0;
void require(bool condition, const char* message) {
    ++assertions;
    if (!condition) throw std::runtime_error(message);
}
void hip_check(hipError_t status) {
    if (status != hipSuccess) throw std::runtime_error(hipGetErrorString(status));
}
RadParam integer(const char* key, int64_t value) {
    RadParam result{}; result.key = key; result.kind = RAD_P_INT; result.ival = value; return result;
}
RadParam text(const char* key, const char* value) {
    RadParam result{}; result.key = key; result.kind = RAD_P_STR; result.sval = value; return result;
}
RadTensor tensor(void* data, uint32_t dtype, int64_t rows, int64_t columns, int64_t pitch = 0) {
    RadTensor result{};
    result.data = data; result.dtype = dtype; result.rank = 2;
    result.shape[0] = rows; result.shape[1] = columns;
    result.stride[0] = pitch ? pitch : columns; result.stride[1] = 1;
    return result;
}
uint32_t mix(uint32_t value) {
    value ^= value >> 16; value *= 0x7feb352du; value ^= value >> 15;
    value *= 0x846ca68bu; return value ^ (value >> 16);
}
float from_bf16(uint16_t value) { return std::bit_cast<float>(uint32_t(value) << 16); }
uint16_t to_bf16(float value) {
    const uint32_t bits = std::bit_cast<uint32_t>(value);
    uint16_t upper = bits >> 16;
    if ((bits & 0x7fffffffu) > 0x7f800000u) return upper | 0x40u;
    const unsigned tail = bits & 65535u;
    if (tail > 32768u || (tail == 32768u && (upper & 1u))) ++upper;
    return upper;
}

struct Model {
    int descriptor = -1;
    static bool identity(const struct stat& info) {
        return S_ISREG(info.st_mode) && info.st_size == 121678372864ll
            && info.st_dev == 66307 && info.st_ino == 106456599
            && info.st_mtim.tv_sec == 1794879394 && info.st_ctim.tv_sec == 1794879394;
    }
    Model() {
        descriptor = open(model_path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
        require(descriptor >= 0, "model open");
        struct stat info{};
        require(fstat(descriptor, &info) == 0 && identity(info), "model pinned size/inode/time identity");
        uint8_t magic[8]{}; read(magic, 8, 0);
        const uint8_t expected[] = {'R', 'A', 'D', '1', 2, 0, 0, 0};
        require(std::memcmp(magic, expected, 8) == 0, "model magic/version");
        std::printf("model_identity,pinned,size=121678372864,dev=66307,ino=106456599,known_sha256=%s\n", model_sha);
    }
    ~Model() { if (descriptor >= 0) close(descriptor); }
    void read(void* output, size_t bytes, uint64_t offset) const {
        require(offset <= 121678372864ull && bytes <= 121678372864ull - offset, "plane bounds");
        auto* destination = static_cast<uint8_t*>(output);
        while (bytes) {
            const ssize_t count = pread(descriptor, destination, bytes, offset);
            require(count > 0, "model pread");
            destination += count; bytes -= count; offset += count;
        }
    }
    void verify() const {
        struct stat info{};
        require(fstat(descriptor, &info) == 0 && identity(info), "model changed during tests");
    }
};

struct Fixture {
    std::string name;
    int columns, reduction;
    bool rotated;
    std::vector<int8_t> codes;
    std::vector<uint16_t> scales;
    std::vector<RadParam> params(int rows) const {
        return {integer("M", rows), integer("N", columns), integer("K", reduction),
                integer("group", 128), integer("w_rb", 1), text("dtype", rotated ? "i8a8h" : "i8a8")};
    }
    RadEncoding encoding() const {
        RadEncoding result{};
        rad_enc_copy_str(result.scheme, "affine");
        rad_enc_copy_str(result.transform, rotated ? "fwht128" : "");
        rad_enc_add_plane(&result, "codes", RAD_I8, 1, 1);
        rad_enc_add_plane(&result, "scale", RAD_BF16, 1, 128);
        return result;
    }
};
Fixture synthetic(bool rotated, int reduction) {
    Fixture result{rotated ? "synthetic_fwht" : "synthetic_plain", 128, reduction, rotated, {}, {}};
    result.codes.resize(size_t(result.columns) * reduction);
    result.scales.resize(size_t(result.columns) * reduction / 128);
    for (size_t index = 0; index < result.codes.size(); ++index)
        result.codes[index] = static_cast<int8_t>(mix(index + 17) & 255u);
    for (size_t index = 0; index < result.scales.size(); ++index)
        result.scales[index] = to_bf16((int(mix(index + 37) % 2047) - 1023) / 93771.0f);
    return result;
}
Fixture real_fixture(const Model& model, const char* name, int total_columns, int total_reduction,
    uint64_t code_offset, uint64_t scale_offset, int first_column, int columns,
    int first_position, int reduction, bool rotated) {
    require(first_column % 128 == 0 && columns % 128 == 0 && first_position % 128 == 0
        && reduction % 128 == 0 && first_column + columns <= total_columns
        && first_position + reduction <= total_reduction, "real shard boundaries");
    Fixture result{name, columns, reduction, rotated, {}, {}};
    result.codes.resize(size_t(columns) * reduction);
    result.scales.resize(size_t(columns) * reduction / 128);
    for (int column = 0; column < columns; ++column) {
        model.read(result.codes.data() + size_t(column) * reduction, reduction,
                   code_offset + uint64_t(first_column + column) * total_reduction + first_position);
        model.read(result.scales.data() + size_t(column) * (reduction / 128), reduction / 128 * 2,
                   scale_offset + uint64_t(first_column + column) * (total_reduction / 128) * 2 + first_position / 128 * 2);
    }
    return result;
}

struct Packed { std::vector<int8_t> codes; std::vector<float> scales; };
Packed pack_and_check(const Fixture& fixture) {
    const int columns = fixture.columns, reduction = fixture.reduction, groups = reduction / 128;
    auto params = fixture.params(3);
    const auto encoding = fixture.encoding();
    std::vector<int8_t> pitched_codes(size_t(columns) * (reduction + 5), 71);
    std::vector<uint16_t> pitched_scales(size_t(columns) * (groups + 3), 0x5555u);
    for (int column = 0; column < columns; ++column) {
        std::copy_n(fixture.codes.data() + size_t(column) * reduction, reduction,
                    pitched_codes.data() + size_t(column) * (reduction + 5));
        std::copy_n(fixture.scales.data() + size_t(column) * groups, groups,
                    pitched_scales.data() + size_t(column) * (groups + 3));
    }
    RadTensor planes[] = {tensor(pitched_codes.data(), RAD_I8, columns, reduction, reduction + 5),
                          tensor(pitched_scales.data(), RAD_BF16, columns, groups, groups + 3)};
    Packed packed{std::vector<int8_t>(fixture.codes.size()), std::vector<float>(fixture.scales.size())};
    for (int operand = 2; operand <= 3; ++operand) {
        const int selected = operand - 2;
        RadLayout layout{};
        require(r3dh_i8_rad_layout(params.data(), params.size(), operand, &encoding,
            &selected, &planes[selected], 1, &layout) == RAD_OK, "layout acceptance");
        const int64_t bytes = operand == 2 ? packed.codes.size() : packed.scales.size() * 4;
        require(layout.bytes == bytes && layout.dtype == (operand == 2
            ? r3dh_i8::weight_dtype : r3dh_i8::scale_dtype), "private layout bytes/dtype");
        void* output = operand == 2 ? static_cast<void*>(packed.codes.data()) : packed.scales.data();
        require(r3dh_i8_rad_relayout(params.data(), params.size(), operand, &encoding,
            &selected, &planes[selected], 1, output, bytes) == RAD_OK, "relayout");
        std::vector<uint8_t> inverse(operand == 2 ? pitched_codes.size() : pitched_scales.size() * 2, 0xa5u);
        RadTensor restored = planes[selected]; restored.data = inverse.data();
        require(r3dh_i8_rad_unrelayout(params.data(), params.size(), operand, &encoding,
            &selected, output, bytes, &restored, 1) == RAD_OK, "unrelayout");
        const int width = operand == 2 ? reduction : groups;
        const int element_bytes = operand == 2 ? 1 : 2;
        for (int column = 0; column < columns; ++column) {
            const auto* original = static_cast<const uint8_t*>(planes[selected].data)
                + column * planes[selected].stride[0] * element_bytes;
            const auto* actual = inverse.data() + column * planes[selected].stride[0] * element_bytes;
            require(std::memcmp(original, actual, width * element_bytes) == 0, "inverse values");
            for (int64_t byte = width * element_bytes; byte < planes[selected].stride[0] * element_bytes; ++byte)
                require(actual[byte] == 0xa5u, "inverse pitch guard");
        }
    }
    size_t cursor = 0;
    for (int tile = 0; tile < columns; tile += 16)
        for (int group = 0; group < groups; ++group)
            for (int step = 0; step < 128; step += 16)
                for (int column = tile; column < tile + 16; ++column)
                    for (int position = group * 128 + step; position < group * 128 + step + 16; ++position)
                        require(packed.codes[cursor++] == fixture.codes[size_t(column) * reduction + position], "independent packed map");
    for (size_t index = 0; index < packed.scales.size(); ++index)
        require(std::bit_cast<uint32_t>(packed.scales[index]) == uint32_t(fixture.scales[index]) << 16, "scale bits");
    return packed;
}

void rejects(const Fixture& fixture) {
    auto params = fixture.params(3);
    auto encoding = fixture.encoding();
    auto plane = tensor(const_cast<int8_t*>(fixture.codes.data()), RAD_I8, fixture.columns, fixture.reduction);
    int selected = 0;
    RadLayout layout{};
    const auto check = [&]() { return r3dh_i8_rad_layout(params.data(), params.size(), 2,
        &encoding, &selected, &plane, 1, &layout); };
    require(check() == RAD_OK, "reject control");
    const auto original = encoding;
    rad_enc_copy_str(encoding.scheme, "trellis"); require(check() != RAD_OK, "reject IQR"); encoding = original;
    rad_enc_copy_str(encoding.transform, fixture.rotated ? "" : "fwht128");
    require(check() != RAD_OK, "reject wrong transform"); encoding = original;
    encoding.plane[1].dtype = RAD_F32; require(check() != RAD_OK, "reject f32 scale encoding"); encoding = original;
    encoding.plane[1].block[1] = 256; require(check() != RAD_OK, "reject group256"); encoding = original;
    encoding.plane[1].block[0] = 128; require(check() != RAD_OK, "reject rowblock128"); encoding = original;
    encoding.plane[0].dtype = RAD_U8; require(check() != RAD_OK, "reject unsigned"); encoding = original;
    encoding.plane[0].kind = RAD_PLANE_TABLE; require(check() != RAD_OK, "reject table"); encoding = original;
    rad_enc_add_plane(&encoding, "zero", RAD_I8, 1, 128); require(check() != RAD_OK, "reject extra zero"); encoding = original;
    selected = 1; require(check() != RAD_OK, "reject wrong selected"); selected = -1;
    require(check() != RAD_OK, "reject negative selected"); selected = 0;
    plane.stride[0] = fixture.reduction - 1; require(check() != RAD_OK, "reject short plane pitch"); plane.stride[0]++;
    plane.stride[1] = 2; require(check() != RAD_OK, "reject plane inner pitch"); plane.stride[1] = 1;
    params[1].ival = 127; require(check() != RAD_OK, "reject partial N tile"); params[1].ival = fixture.columns;
    params[2].ival = fixture.reduction - 1; require(check() != RAD_OK, "reject partial K group"); params[2].ival++;
    params[3].ival = 64; require(check() != RAD_OK, "reject non128 group"); params[3].ival = 128;
    params[4].ival = 128; require(check() != RAD_OK, "reject w_rb"); params[4].ival = 1;
    params[5].sval = "w4a8i8_asym"; require(check() != RAD_OK, "reject unrelated dtype");
    params[5].sval = fixture.rotated ? "i8a8h" : "i8a8";
    if (!fixture.rotated) { params[5].sval = "i8"; require(check() == RAD_OK, "plain i8 alias"); params[5].sval = "i8a8"; }
    std::vector<int8_t> output(fixture.codes.size());
    require(r3dh_i8_rad_relayout(params.data(), params.size(), 2, &encoding, &selected,
        &plane, 1, output.data(), output.size() - 1) != RAD_OK, "reject wrong byte count");
    for (int rows : {1, 2, 3, 17, 65, 2048}) {
        params[0].ival = rows;
        for (int operand = 0; operand <= 6; ++operand) {
            RadOpdDesc description{};
            require(r3dh_i8_shape_gemm_nt_q(params.data(), params.size(), operand,
                &description) == RAD_OK, "gemm shape");
            if (operand >= 5) require(description.flags & RAD_OPD_F_ABSENT, "absent shape");
        }
    }
    params[0].ival = 2049;
    RadOpdDesc description{};
    require(r3dh_i8_shape_gemm_nt_q(params.data(), params.size(), 0, &description) != RAD_OK, "reject M2049");
    std::printf("strict_rejects,%s,PASS\n", fixture.name.c_str());
}

void frontend_rejects(const Fixture& fixture) {
    auto params = fixture.params(3);
    void* pointer = reinterpret_cast<void*>(uintptr_t(4096));
    std::array<RadTensor, 7> tensors{
        tensor(pointer, RAD_I8, 3, fixture.reduction),
        tensor(pointer, RAD_F32, 3, fixture.reduction / 128),
        tensor(pointer, r3dh_i8::weight_dtype, fixture.columns, fixture.reduction),
        tensor(pointer, r3dh_i8::scale_dtype, fixture.columns, fixture.reduction / 128),
        tensor(pointer, RAD_BF16, 3, fixture.columns), {}, {}};
    RadArgs arguments{}; arguments.t = tensors.data(); arguments.n_t = 5;
    arguments.p = params.data(); arguments.n_p = params.size();
    const auto rejected = [&]() { require(r3dh_i8_rad_gemm_nt_q(&arguments, nullptr) != RAD_OK, "gemm frontend rejection"); };
    require(r3dh_i8_rad_gemm_nt_q(nullptr, nullptr) == RAD_E_INVAL, "null gemm args");
    arguments.t = nullptr; rejected(); arguments.t = tensors.data();
    arguments.n_t = 4; rejected(); arguments.n_t = 8; rejected(); arguments.n_t = 5;
    arguments.p = nullptr; rejected(); arguments.p = params.data();
    arguments.n_p = -1; rejected(); arguments.n_p = params.size();
    const auto saved = tensors;
    for (int index = 0; index < 5; ++index) {
        tensors[index].data = nullptr; rejected(); tensors = saved;
        tensors[index].rank = 1; rejected(); tensors = saved;
        tensors[index].shape[0]++; rejected(); tensors = saved;
        tensors[index].shape[1]++; rejected(); tensors = saved;
        tensors[index].stride[0]--; rejected(); tensors = saved;
        tensors[index].stride[1] = 2; rejected(); tensors = saved;
        tensors[index].data = reinterpret_cast<void*>(uintptr_t(4097)); rejected(); tensors = saved;
    }
    for (uint32_t dtype : {uint32_t(RAD_I8), 0x33000100u, 0x33000200u}) {
        tensors[2].dtype = dtype; rejected(); tensors = saved;
        tensors[3].dtype = dtype; rejected(); tensors = saved;
    }
    tensors[0].stride[0] = INT64_MAX - 3; rejected(); tensors = saved;
    tensors[1].stride[0] = INT64_MAX; rejected(); tensors = saved;
    arguments.n_t = 6; tensors[5] = saved[0]; rejected(); tensors = saved; arguments.n_t = 5;
    params[0].ival = 0; rejected(); params[0].ival = 2049; rejected(); params[0].ival = 3;
    params[1].ival--; rejected(); params[1].ival++;
    params[2].ival--; rejected(); params[2].ival++;
    params[3].ival = 256; rejected(); params[3].ival = 128;
    params[4].ival = 128; rejected(); params[4].ival = 1;
    params[5].sval = "iqr"; rejected();
    std::array<RadParam, 3> quant_params{integer("M", 3), integer("n", fixture.reduction), integer("group", 128)};
    RadOpdDesc description{};
    for (int operand = 0; operand < 3; ++operand)
        require(r3dh_i8_shape_quant_act_i8g(quant_params.data(), quant_params.size(), operand,
            &description) == RAD_OK, "quant shape");
    RadArgs quant_args{};
    require(r3dh_i8_rad_quant_act_i8g(&quant_args, nullptr) != RAD_OK, "quant invalid args");
    require(r3dh_i8_rad_had_quant_act_i8g(&quant_args, nullptr) != RAD_OK, "had quant invalid args");
    std::printf("frontend_rejects,%s,IQR_isolation=1,PASS\n", fixture.name.c_str());
}

template<class Value> struct Guarded {
    static constexpr size_t guard = 256;
    uint8_t* allocation = nullptr;
    size_t count;
    explicit Guarded(size_t elements, hipStream_t stream) : count(elements) {
        hip_check(hipMalloc(reinterpret_cast<void**>(&allocation), count * sizeof(Value) + guard * 2));
        hip_check(hipMemsetAsync(allocation, 0xa5, count * sizeof(Value) + guard * 2, stream));
    }
    Guarded(const Guarded&) = delete;
    Guarded& operator=(const Guarded&) = delete;
    ~Guarded() { if (allocation) static_cast<void>(hipFree(allocation)); }
    Value* data() const { return reinterpret_cast<Value*>(allocation + guard); }
    void upload(const std::vector<Value>& source, hipStream_t stream) {
        require(source.size() == count, "upload size");
        hip_check(hipMemcpyAsync(data(), source.data(), count * sizeof(Value), hipMemcpyHostToDevice, stream));
    }
    std::vector<Value> download(hipStream_t stream) const {
        std::vector<uint8_t> bytes(count * sizeof(Value) + guard * 2);
        hip_check(hipMemcpyAsync(bytes.data(), allocation, bytes.size(), hipMemcpyDeviceToHost, stream));
        hip_check(hipStreamSynchronize(stream));
        for (size_t index = 0; index < guard; ++index)
            require(bytes[index] == 0xa5u && bytes[bytes.size() - 1 - index] == 0xa5u, "device pre/post guard");
        std::vector<Value> result(count);
        std::memcpy(result.data(), bytes.data() + guard, count * sizeof(Value));
        return result;
    }
};

void cpu_quant(const std::vector<uint16_t>& input, std::vector<int8_t>& codes,
    std::vector<float>& scales, int rows, int reduction, int64_t input_pitch,
    int64_t code_pitch, int64_t scale_pitch, bool rotate) {
    for (int row = 0; row < rows; ++row)
        for (int group = 0; group < reduction / 128; ++group) {
            std::array<float, 128> values{};
            for (int position = 0; position < 128; ++position)
                values[position] = from_bf16(input[row * input_pitch + group * 128 + position]);
            if (rotate)
                for (int width = 1; width < 128; width *= 2)
                    for (int base = 0; base < 128; base += width * 2)
                        for (int offset = 0; offset < width; ++offset) {
                            const float left = values[base + offset], right = values[base + width + offset];
                            values[base + offset] = left + right;
                            values[base + width + offset] = left - right;
                        }
            float maximum = 0.0f;
            for (float value : values) maximum = std::max(maximum, std::fabs(value));
            if (!(maximum > 0.0f)) maximum = 1e-8f;
            const float inverse = 127.0f / maximum;
            scales[row * scale_pitch + group] = maximum / 127.0f;
            for (int position = 0; position < 128; ++position) {
                const float value = std::max(-127.0f, std::min(127.0f, values[position] * inverse));
                const float floor_value = std::floor(value);
                const float fraction = value - floor_value;
                int rounded = static_cast<int>(floor_value);
                if (fraction > 0.5f || (fraction == 0.5f && (rounded & 1))) ++rounded;
                codes[row * code_pitch + group * 128 + position] = static_cast<int8_t>(rounded);
            }
        }
}

std::vector<uint16_t> cpu_gemm(const Fixture& fixture, const std::vector<int8_t>& activation,
    const std::vector<float>& scales, int rows, int64_t activation_pitch,
    int64_t scale_pitch, int64_t output_pitch) {
    std::vector<uint16_t> output(size_t(rows) * output_pitch, 0xa5a5u);
    const int reduction = fixture.reduction, groups = reduction / 128;
    for (int row = 0; row < rows; ++row)
        for (int column = 0; column < fixture.columns; ++column) {
            float accumulator = 0.0f;
            for (int group = 0; group < groups; ++group) {
                int64_t subtotal = 0;
                for (int position = group * 128; position < (group + 1) * 128; ++position)
                    subtotal += int64_t(activation[row * activation_pitch + position])
                              * fixture.codes[size_t(column) * reduction + position];
                const float first = static_cast<float>(subtotal) * scales[row * scale_pitch + group];
                const float second = first * from_bf16(fixture.scales[size_t(column) * groups + group]);
                accumulator = accumulator + second;
            }
            output[row * output_pitch + column] = to_bf16(accumulator);
        }
    return output;
}

void run_gpu(const Fixture& fixture, const Packed& packed, int rows, hipStream_t stream) {
    const bool pitched = rows % 2 != 0;
    const int64_t activation_pitch = fixture.reduction + (pitched ? 12 : 0);
    const int64_t scale_pitch = fixture.reduction / 128 + (pitched ? 3 : 0);
    const int64_t output_pitch = fixture.columns + (pitched ? 7 : 0);
    const int64_t input_pitch = fixture.reduction + (pitched ? 5 : 0);
    Guarded<int8_t> activation(size_t(rows) * activation_pitch, stream);
    Guarded<float> scales(size_t(rows) * scale_pitch, stream);
    Guarded<uint16_t> input(size_t(rows) * input_pitch, stream);
    Guarded<int8_t> weight(packed.codes.size(), stream);
    Guarded<float> weight_scale(packed.scales.size(), stream);
    Guarded<uint16_t> output(size_t(rows) * output_pitch, stream);
    weight.upload(packed.codes, stream); weight_scale.upload(packed.scales, stream);
    auto params = fixture.params(rows);
    if (!fixture.rotated && rows == 2) params[5].sval = "i8";
    std::array<RadTensor, 5> tensors{
        tensor(activation.data(), RAD_I8, rows, fixture.reduction, activation_pitch),
        tensor(scales.data(), RAD_F32, rows, fixture.reduction / 128, scale_pitch),
        tensor(weight.data(), r3dh_i8::weight_dtype, fixture.columns, fixture.reduction),
        tensor(weight_scale.data(), r3dh_i8::scale_dtype, fixture.columns, fixture.reduction / 128),
        tensor(output.data(), RAD_BF16, rows, fixture.columns, output_pitch)};
    RadArgs arguments{}; arguments.t = tensors.data(); arguments.n_t = tensors.size();
    arguments.p = params.data(); arguments.n_p = params.size();
    const auto rad_stream = reinterpret_cast<RadStream>(stream);
    for (int repeat = 0; repeat < 2; ++repeat) {
        std::vector<int8_t> host_activation(size_t(rows) * activation_pitch, 0x35);
        std::vector<float> host_scales(size_t(rows) * scale_pitch, std::bit_cast<float>(0x3e353535u));
        std::vector<uint16_t> host_input(size_t(rows) * input_pitch, 0x3535u);
        for (int row = 0; row < rows; ++row) {
            for (int position = 0; position < fixture.reduction; ++position) {
                const uint32_t random = mix(row * fixture.reduction + position + repeat * 12345u);
                host_activation[row * activation_pitch + position] = static_cast<int8_t>(random & 255u);
                host_input[row * input_pitch + position] = to_bf16((int(random % 4095u) - 2047) / 137.0f);
            }
            for (int position = 0; position < 128; ++position) {
                if (row == 0) host_input[position] = 0;
                if (row == 1) {
                    const float ties[] = {127.0f, 0.5f, 1.5f, 2.5f, -0.5f, -1.5f, -2.5f};
                    host_input[row * input_pitch + position] = to_bf16(position < 7 ? ties[position] : 0.0f);
                }
            }
            for (int group = 0; group < fixture.reduction / 128; ++group)
                host_scales[row * scale_pitch + group] = (mix(row * 71 + group + repeat) % 65535u + 1) / 39001.0f;
        }
        if (repeat) cpu_quant(host_input, host_activation, host_scales, rows, fixture.reduction,
                              input_pitch, activation_pitch, scale_pitch, fixture.rotated);
        activation.upload(host_activation, stream); scales.upload(host_scales, stream);
        input.upload(host_input, stream);
        if (repeat) {
            std::vector<int8_t> initial_codes(host_activation.size(), 0x35);
            std::vector<float> initial_scales(host_scales.size(), std::bit_cast<float>(0x3e353535u));
            activation.upload(initial_codes, stream); scales.upload(initial_scales, stream);
            std::array<RadTensor, 3> quant_tensors{
                tensor(input.data(), RAD_BF16, rows, fixture.reduction, input_pitch), tensors[0], tensors[1]};
            std::array<RadParam, 3> quant_params{integer("M", rows), integer("n", fixture.reduction), integer("group", 128)};
            RadArgs quant_args{}; quant_args.t = quant_tensors.data(); quant_args.n_t = quant_tensors.size();
            quant_args.p = quant_params.data(); quant_args.n_p = quant_params.size();
            if (rows % 2) { quant_args.p++; quant_args.n_p--; }
            require((fixture.rotated ? r3dh_i8_rad_had_quant_act_i8g(&quant_args, rad_stream)
                : r3dh_i8_rad_quant_act_i8g(&quant_args, rad_stream)) == RAD_OK, "quant launch");
            require(activation.download(stream) == host_activation, "quant exact codes and pitch");
            const auto actual_scales = scales.download(stream);
            require(std::memcmp(actual_scales.data(), host_scales.data(), host_scales.size() * 4) == 0, "quant exact scale and pitch");
        }
        const auto expected = cpu_gemm(fixture, host_activation, host_scales, rows,
                                      activation_pitch, scale_pitch, output_pitch);
        hip_check(hipMemsetAsync(output.data(), 0xa5, output.count * 2, stream));
        require(r3dh_i8_rad_gemm_nt_q(&arguments, rad_stream) == RAD_OK, "gemm launch");
        const auto actual = output.download(stream);
        if (actual != expected) {
            for (size_t index = 0; index < actual.size(); ++index)
                if (actual[index] != expected[index]) {
                    std::fprintf(stderr, "mismatch,%s,M=%d,repeat=%d,index=%zu,got=%04x,expected=%04x\n",
                        fixture.name.c_str(), rows, repeat, index, actual[index], expected[index]); break;
                }
            require(false, "BF16 bit exact or output pitch guard");
        }
        require(activation.download(stream) == host_activation, "activation unchanged");
        const auto actual_scales = scales.download(stream);
        require(std::memcmp(actual_scales.data(), host_scales.data(), host_scales.size() * 4) == 0, "activation scale unchanged");
        require(input.download(stream) == host_input, "BF16 input unchanged");
    }
    require(weight.download(stream) == packed.codes, "packed codes unchanged");
    const auto actual_scales = weight_scale.download(stream);
    require(std::memcmp(actual_scales.data(), packed.scales.data(), packed.scales.size() * 4) == 0, "weight scale unchanged");
    std::printf("gpu,%s,M=%d,N=%d,K=%d,pitched=%d,repeats=2,nondefault=1,bitexact=1,guards=1,PASS\n",
        fixture.name.c_str(), rows, fixture.columns, fixture.reduction, pitched);
    std::fflush(stdout);
}
}

int main(int argc, char** argv) {
    try {
        require(argc == 1 || (argc == 2 && (!std::strcmp(argv[1], "--host")
            || !std::strcmp(argv[1], "--gpu4"))), "usage: r3dh_i8_check [--host|--gpu4]");
        const bool host_only = argc == 2 && !std::strcmp(argv[1], "--host");
        const bool gpu4 = argc == 2 && !std::strcmp(argv[1], "--gpu4");
        Model model;
        std::vector<Fixture> fixtures;
        fixtures.push_back(synthetic(false, 128));
        fixtures.push_back(synthetic(true, 640));
        rejects(fixtures[0]); rejects(fixtures[1]);
        frontend_rejects(fixtures[0]); frontend_rejects(fixtures[1]);
        fixtures.push_back(real_fixture(model, "attn_k_full", 512, 2560, 58022457344ull, 58023768064ull, 0, 512, 0, 2560, true));
        fixtures.push_back(real_fixture(model, "attn_v_full", 512, 2560, 58059550720ull, 58060861440ull, 0, 512, 0, 2560, true));
        fixtures.push_back(real_fixture(model, "attn_k_row_slice", 512, 2560, 58022457344ull, 58023768064ull, 256, 256, 0, 2560, true));
        fixtures.push_back(real_fixture(model, "attn_v_nk_slice", 512, 2560, 58059550720ull, 58060861440ull, 128, 256, 1280, 1280, true));
        fixtures.push_back(real_fixture(model, "ssm_out_full", 2560, 6144, 53973250048ull, 53988978688ull, 0, 2560, 0, 6144, true));
        fixtures.push_back(real_fixture(model, "ssm_out_k_shard", 2560, 6144, 53973250048ull, 53988978688ull, 0, 2560, 3072, 3072, true));
        fixtures.push_back(real_fixture(model, "shared_down_full", 2560, 640, 53938688000ull, 53940326400ull, 0, 2560, 0, 640, true));
        fixtures.push_back(real_fixture(model, "shared_down_k256", 2560, 640, 53938688000ull, 53940326400ull, 0, 2560, 384, 256, true));
        fixtures.push_back(real_fixture(model, "shared_down_k384", 2560, 640, 53938688000ull, 53940326400ull, 0, 2560, 0, 384, true));
        fixtures.push_back(real_fixture(model, "output_head_first", 248320, 2560, 51939876864ull, 52575576064ull, 0, 128, 0, 2560, false));
        fixtures.push_back(real_fixture(model, "output_head_rank1", 248320, 2560, 51939876864ull, 52575576064ull, 124160, 256, 0, 2560, false));
        fixtures.push_back(real_fixture(model, "output_head_last", 248320, 2560, 51939876864ull, 52575576064ull, 248192, 128, 0, 2560, false));
        hipStream_t stream = nullptr;
        if (!host_only) {
            int count = 0; hip_check(hipGetDeviceCount(&count)); require(count == 1, "exactly one GPU must be exposed");
            hipDeviceProp_t properties{}; hip_check(hipGetDeviceProperties(&properties, 0));
            require(!std::strncmp(properties.gcnArchName, "gfx1100", 7), "gfx1100 only checker");
            char pci[32]{}; hip_check(hipDeviceGetPCIBusId(pci, sizeof(pci), 0));
            require(!std::strcmp(pci, gpu4 ? "0000:83:00.0" : "0000:43:00.0"), "selected physical GPU PCI identity");
            std::printf("device,%s,%s,%s\n", properties.name, properties.gcnArchName, pci);
            hip_check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
        }
        unsigned cases = 0;
        for (const Fixture& fixture : fixtures) {
            const auto packed = pack_and_check(fixture);
            std::printf("host_layout,%s,N=%d,K=%d,roundtrip=1,map=1,PASS\n", fixture.name.c_str(), fixture.columns, fixture.reduction);
            if (!host_only) {
                for (int rows : {1, 2, 3, 8, 16, 17, 32, 64, 65}) { run_gpu(fixture, packed, rows, stream); ++cases; }
                if (fixture.name == "synthetic_plain")
                    for (int rows : {127, 128, 255, 256, 511, 512, 1023, 1024, 2047, 2048}) {
                        run_gpu(fixture, packed, rows, stream); ++cases;
                    }
            }
        }
        if (stream) hip_check(hipStreamDestroy(stream));
        model.verify();
        std::printf("PASS,mode=%s,fixtures=%zu,gpu_cases=%u,assertions=%llu\n",
            host_only ? "host" : "gpu", fixtures.size(), cases, static_cast<unsigned long long>(assertions));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL,%s\n", error.what()); return 1;
    }
}
