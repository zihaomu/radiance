#include "r3dh_ar.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
constexpr long max_elements = 65536;
constexpr size_t guard = 256;
constexpr unsigned capacity = 512;
const char* names[] = {"bf16", "fp16", "fp32"};
uint64_t compared = 0;
std::atomic<int> assertions{0};
void require(bool condition, const char* message) {
    ++assertions;
    if (!condition) throw std::runtime_error(message);
}
void hip_check(hipError_t error) {
    if (error != hipSuccess) throw std::runtime_error(hipGetErrorString(error));
}
void rad_check(int result) {
    if (result != RAD_OK) throw std::runtime_error("RAD status " + std::to_string(result));
}
template <typename Function>
void both(Function function) {
    std::exception_ptr errors[2];
    std::thread workers[2];
    for (int rank = 0; rank < 2; ++rank)
        workers[rank] = std::thread([&, rank] {
            try { hip_check(hipSetDevice(1 - rank)); function(rank); }
            catch (...) { errors[rank] = std::current_exception(); }
        });
    for (auto& worker : workers) worker.join();
    for (const auto& error : errors) if (error) std::rethrow_exception(error);
}
unsigned encode(float value, int dtype) {
    const unsigned bits = std::bit_cast<unsigned>(value);
    if (dtype == 2) return bits;
    if (dtype == 0) {
        const unsigned upper = bits >> 16, lower = bits & 65535u;
        return upper + (lower > 32768u || (lower == 32768u && (upper & 1u)));
    }
    const unsigned sign = (bits >> 16) & 32768u;
    const double magnitude = std::fabs((double)value);
    if (std::isnan(value)) return sign | 0x7e00u;
    if (std::isinf(value)) return sign | 0x7c00u;
    if (magnitude < std::ldexp(1.0, -14))
        return sign | (unsigned)std::nearbyint(std::ldexp(magnitude, 24));
    int exponent = 0;
    (void)std::frexp(magnitude, &exponent);
    --exponent;
    unsigned mantissa = (unsigned)std::nearbyint(std::ldexp(magnitude, 10 - exponent));
    if (mantissa == 2048) { mantissa = 1024; ++exponent; }
    if (exponent > 15) return sign | 0x7c00u;
    return sign | ((unsigned)(exponent + 15) << 10) | (mantissa - 1024);
}
float decode(unsigned bits, int dtype) {
    if (dtype == 2) return std::bit_cast<float>(bits);
    if (dtype == 0) return std::bit_cast<float>(bits << 16);
    const unsigned fraction = bits & 1023u, exponent = (bits >> 10) & 31u;
    float value = exponent == 0 ? std::ldexp((float)fraction, -24)
        : exponent == 31 ? (fraction ? NAN : INFINITY)
        : std::ldexp(1.0f + (float)fraction / 1024.0f, (int)exponent - 15);
    return bits & 32768u ? -value : value;
}
unsigned read(const unsigned char* pointer, int dtype) {
    unsigned bits = 0;
    std::memcpy(&bits, pointer, dtype == 2 ? 4 : 2);
    return bits;
}
void write(unsigned char* pointer, unsigned bits, int dtype) {
    std::memcpy(pointer, &bits, dtype == 2 ? 4 : 2);
}
std::vector<unsigned char> payload(int rank, int iteration, long elements, int dtype) {
    const size_t width = dtype == 2 ? 4 : 2;
    std::vector<unsigned char> bytes(elements * width);
    for (long index = 0; index < elements; ++index) {
        unsigned mixed = (unsigned)index * 0x9e3779b9u + (unsigned)iteration * 0x85ebca6bu;
        mixed ^= (unsigned)(rank + 1) * 0xc2b2ae35u;
        float value = (float)((int)(mixed % 16381u) - 8190) / 64.0f;
        if (iteration % 5 == 0) value = (float)(iteration * (rank + 1) + rank - 17) / 32.0f;
        else if (index % 97 == 0) value = std::ldexp(1.0f, -149 + (iteration + rank) % 23);
        else if (index % 89 == 0) value = rank ? -0.5f : 0.5f;
        else if (index % 71 == 0) value = rank ? 1.0f / 256 : 1.0f;
        write(bytes.data() + index * width, encode(value, dtype), dtype);
    }
    return bytes;
}
std::vector<unsigned char> oracle(const std::vector<unsigned char>& first,
                                  const std::vector<unsigned char>& second, int dtype) {
    const size_t width = dtype == 2 ? 4 : 2;
    std::vector<unsigned char> result(first.size());
    for (size_t offset = 0; offset < result.size(); offset += width) {
        const float sum = decode(read(first.data() + offset, dtype), dtype) +
                          decode(read(second.data() + offset, dtype), dtype);
        write(result.data() + offset, encode(sum, dtype), dtype);
    }
    return result;
}
struct Rank {
    int device = -1;
    void* instance = nullptr;
    hipStream_t stream = nullptr;
    hipGraph_t graph = nullptr;
    hipGraphExec_t executable = nullptr;
    unsigned char* input = nullptr;
    unsigned char* output = nullptr;
    void* raw_scratch = nullptr;
    unsigned* raw_flags = nullptr;
    unsigned* raw_seq = nullptr;
    size_t allocation = 0;
    ~Rank() {
        if (device < 0) return;
        (void)hipSetDevice(device);
        if (stream) (void)hipStreamSynchronize(stream);
        if (executable) (void)hipGraphExecDestroy(executable);
        if (graph) (void)hipGraphDestroy(graph);
        r3dh_rad_ar_fini(instance);
        if (input) (void)hipFree(input);
        if (output) (void)hipFree(output);
        if (raw_scratch) (void)hipFree(raw_scratch);
        if (raw_flags) (void)hipFree(raw_flags);
        if (raw_seq) (void)hipFree(raw_seq);
        if (stream) (void)hipStreamDestroy(stream);
    }
};
std::array<RadParam, 4> params(int dtype, long elements) {
    return {RAD_INT("world_size", 2), RAD_INT("numel", elements),
            RAD_STR("dtype", names[dtype]), RAD_INT("exact", 1)};
}
RadTensor tensor(void* pointer, long elements, int dtype) {
    RadTensor result{};
    result.data = pointer;
    result.dtype = dtype == 0 ? RAD_BF16 : dtype == 1 ? RAD_F16 : RAD_F32;
    result.rank = 1;
    result.shape[0] = elements;
    result.stride[0] = 1;
    return result;
}
int invoke(Rank& state, int rank, int dtype, long elements, int mode) {
    auto geometry = params(dtype, elements);
    RadTensor operands[] = {tensor(state.input + guard, elements, dtype),
        tensor(mode == 2 ? state.input + guard : state.output + guard, elements, dtype)};
    if (mode == 3) operands[1] = {};
    RadArgs args{};
    args.t = operands; args.n_t = mode == 1 ? 1 : 2;
    args.p = geometry.data(); args.n_p = (int)geometry.size();
    args.instance = state.instance; args.rank = rank; args.world_size = 2;
    require(r3dh_rad_ar_scratch(&args) == 0, "per-call scratch");
    return r3dh_rad_ar_oneshot_2rank_exact(&args, (RadStream)state.stream);
}
void upload(Rank& state, const std::vector<unsigned char>& bytes) {
    hip_check(hipSetDevice(state.device));
    hip_check(hipMemset(state.input, 0xa5, state.allocation));
    hip_check(hipMemset(state.output, 0xa5, state.allocation));
    hip_check(hipMemcpy(state.input + guard, bytes.data(), bytes.size(), hipMemcpyHostToDevice));
}
void verify(Rank& state, const std::vector<unsigned char>& original,
            const std::vector<unsigned char>& expected, int dtype, int mode) {
    hip_check(hipSetDevice(state.device));
    hip_check(hipStreamSynchronize(state.stream));
    std::vector<unsigned char> actual(state.allocation), input(state.allocation);
    hip_check(hipMemcpy(actual.data(), (mode ? state.input : state.output),
                         actual.size(), hipMemcpyDeviceToHost));
    for (size_t offset = 0; offset < guard; ++offset) {
        require(actual[offset] == 0xa5, "leading guard");
        require(actual[guard + expected.size() + offset] == 0xa5, "trailing guard");
    }
    require(!std::memcmp(actual.data() + guard, expected.data(), expected.size()), "CPU oracle bits");
    if (!mode) {
        hip_check(hipMemcpy(input.data(), state.input, input.size(), hipMemcpyDeviceToHost));
        require(!std::memcmp(input.data() + guard, original.data(), original.size()), "out input changed");
        for (size_t offset = 0; offset < guard; ++offset) {
            require(input[offset] == 0xa5, "input leading guard");
            require(input[guard + original.size() + offset] == 0xa5, "input trailing guard");
        }
    }
    compared += expected.size() / (dtype == 2 ? 4 : 2);
}
void delayed(int rank, int iteration) {
    if (rank != 1 || iteration % 17) return;
    const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(2);
    while (std::chrono::steady_clock::now() < until) {}
}
void host_tests() {
    void* instance = (void*)1;
    auto geometry = params(0, 8);
    require(r3dh_rad_ar_init(geometry.data(), 4, 0, 1, &instance) == RAD_E_SHAPE && !instance,
            "world reject");
    require(r3dh_rad_ar_init(geometry.data(), 4, 2, 2, &instance) == RAD_E_SHAPE, "rank reject");
    geometry[3].ival = 0;
    require(r3dh_rad_ar_init(geometry.data(), 4, 0, 2, &instance) == RAD_E_SHAPE, "lossy reject");
    geometry[3].ival = 1; geometry[2].sval = "i8";
    require(r3dh_rad_ar_init(geometry.data(), 4, 0, 2, &instance) == RAD_E_DTYPE, "dtype reject");
    geometry[2].sval = "bf16"; geometry[1].ival = 0;
    require(r3dh_rad_ar_init(geometry.data(), 4, 0, 2, &instance) == RAD_E_SHAPE, "empty reject");
    require(r3dh_rad_ar_oneshot_2rank_exact(nullptr, nullptr) == RAD_E_STATE, "missing instance");
    require(r3dh_rad_ar_scratch(nullptr) == 0, "null scratch");
    for (unsigned bits = 0; bits < 65536; ++bits) {
        if ((bits & 0x7c00u) != 0x7c00u)
            require(encode(decode(bits, 1), 1) == bits, "CPU half roundtrip");
    }
    std::printf("host_contract,all,none,%d,0,PASS\n", assertions.load());
}
void rejects(Rank& state, int rank, int dtype) {
    auto geometry = params(dtype, 128);
    RadTensor operands[] = {tensor(state.input + guard, 128, dtype),
                            tensor(state.output + guard, 128, dtype)};
    RadArgs args{};
    args.t = operands; args.n_t = 2; args.p = geometry.data(); args.n_p = 4;
    args.instance = state.instance; args.rank = rank; args.world_size = 2;
    auto expect = [&](int status) {
        require(r3dh_rad_ar_oneshot_2rank_exact(&args, (RadStream)state.stream) == status,
                "negative tensor contract");
    };
    operands[0].data = state.input + guard + 2; expect(RAD_E_ALIGN);
    operands[0].data = state.input + guard;
    operands[0].stride[0] = 2; expect(RAD_E_STRIDE);
    operands[0].stride[0] = 1;
    operands[0].shape[0] = operands[1].shape[0] = 127; expect(RAD_E_ALIGN);
    operands[0].shape[0] = operands[1].shape[0] = 128;
    operands[1].shape[0] = 120; expect(RAD_E_SHAPE);
    operands[1].shape[0] = 128;
    const unsigned saved_dtype = operands[0].dtype;
    operands[0].dtype = RAD_I8; expect(RAD_E_DTYPE);
    operands[0].dtype = saved_dtype;
    args.world_size = 1; expect(RAD_E_SHAPE); args.world_size = 2;
    args.n_t = 0; expect(RAD_E_INVAL); args.n_t = 2;
    operands[1].data = state.input + guard + 16; expect(RAD_E_INVAL);
    operands[1].data = state.output + guard;
    geometry[3].ival = 0; expect(RAD_E_SHAPE); geometry[3].ival = 1;
    geometry[1].ival = max_elements + 8; expect(RAD_E_SHAPE);
    geometry[1].ival = 128;
    geometry[2].sval = "i8"; expect(RAD_E_DTYPE);
    hip_check(hipSetDevice(1 - state.device));
    expect(RAD_E_DEVICE);
    hip_check(hipSetDevice(state.device));
}
void suite(int dtype, bool smoke = false) {
    Rank states[2];
    both([&](int rank) {
        Rank& state = states[rank];
        state.device = 1 - rank;
        state.allocation = max_elements * (dtype == 2 ? 4 : 2) + 2 * guard;
        std::fprintf(stderr, "init_begin,%s,%d,device=%d\n", names[dtype], rank, state.device);
        hip_check(hipStreamCreateWithFlags(&state.stream, hipStreamNonBlocking));
        hip_check(hipMalloc((void**)&state.input, state.allocation));
        hip_check(hipMalloc((void**)&state.output, state.allocation));
        std::fprintf(stderr, "transport_begin,%s,%d\n", names[dtype], rank);
        auto geometry = params(dtype, max_elements);
        geometry[1].kind = RAD_P_RANGE; geometry[1].ival = 1; geometry[1].ihi = max_elements;
        rad_check(r3dh_rad_ar_init(geometry.data(), 4, rank, 2, &state.instance));
        std::fprintf(stderr, "transport_ready,%s,%d\n", names[dtype], rank);
        rejects(state, rank, dtype);
    });
    for (int mode = 0; mode < 4; ++mode) {
        const uint64_t before = compared;
        const int iterations = smoke ? 8 : 32;
        for (int iteration = 0; iteration < iterations; ++iteration) {
            const long words[] = {1, 5, 31, 256, 257, 2048, 8191, 8192};
            const long elements = words[iteration % 8] * (dtype == 2 ? 4 : 8);
            const auto first = payload(0, iteration, elements, dtype);
            const auto second = payload(1, iteration, elements, dtype);
            const auto expected = oracle(first, second, dtype);
            upload(states[0], first); upload(states[1], second);
            std::fprintf(stderr, "eager_begin,%s,%d,%d,%ld\n", names[dtype], mode, iteration, elements);
            both([&](int rank) {
                delayed(rank, iteration);
                rad_check(invoke(states[rank], rank, dtype, elements, mode));
                hip_check(hipStreamSynchronize(states[rank].stream));
            });
            verify(states[0], first, expected, dtype, mode);
            verify(states[1], second, expected, dtype, mode);
        }
        std::printf("eager,%s,%d,%d,%llu,PASS\n", names[dtype], mode, iterations,
                    (unsigned long long)(compared - before));
    }
    if (smoke) return;
    for (int mode = 0; mode < 2; ++mode) {
        const long elements = 4096;
        const uint64_t before = compared;
        both([&](int rank) {
            Rank& state = states[rank];
            hip_check(hipStreamBeginCapture(state.stream, hipStreamCaptureModeThreadLocal));
            rad_check(invoke(state, rank, dtype, elements, mode));
            hip_check(hipStreamEndCapture(state.stream, &state.graph));
            hip_check(hipGraphInstantiate(&state.executable, state.graph, nullptr, nullptr, 0));
        });
        for (int iteration = 0; iteration < 256; ++iteration) {
            const auto first = payload(0, iteration + 100, elements, dtype);
            const auto second = payload(1, iteration + 100, elements, dtype);
            const auto expected = oracle(first, second, dtype);
            upload(states[0], first); upload(states[1], second);
            both([&](int rank) {
                delayed(rank, iteration);
                hip_check(hipGraphLaunch(states[rank].executable, states[rank].stream));
                hip_check(hipStreamSynchronize(states[rank].stream));
            });
            verify(states[0], first, expected, dtype, mode);
            verify(states[1], second, expected, dtype, mode);
        }
        both([&](int rank) {
            hip_check(hipGraphExecDestroy(states[rank].executable));
            hip_check(hipGraphDestroy(states[rank].graph));
            states[rank].executable = nullptr; states[rank].graph = nullptr;
        });
        std::printf("graph,%s,%d,256,%llu,PASS\n", names[dtype], mode,
                    (unsigned long long)(compared - before));
    }
    constexpr unsigned seed = 0xfffffffcu;
    both([&](int rank) {
        Rank& state = states[rank];
        hip_check(hipExtMallocWithFlags(&state.raw_scratch, 2 * 8192 * 16, hipDeviceMallocFinegrained));
        hip_check(hipExtMallocWithFlags((void**)&state.raw_flags, 3 * capacity * 4,
                                        hipDeviceMallocFinegrained));
        hip_check(hipMalloc((void**)&state.raw_seq, (2 * capacity + 1) * 4));
        std::vector<unsigned> flags(3 * capacity), sequence(2 * capacity + 1, seed);
        std::fill(flags.begin(), flags.begin() + capacity, seed);
        hip_check(hipMemset(state.raw_scratch, 0, 2 * 8192 * 16));
        hip_check(hipMemcpy(state.raw_flags, flags.data(), flags.size() * 4, hipMemcpyHostToDevice));
        hip_check(hipMemcpy(state.raw_seq, sequence.data(), sequence.size() * 4, hipMemcpyHostToDevice));
    });
    std::array<unsigned, capacity> advances{};
    const uint64_t before = compared;
    for (int iteration = 0; iteration < 40; ++iteration) {
        const int grids[] = {1, 24, 3, 7, 2, 17};
        const int blocks = grids[iteration % 6];
        for (int block = 0; block < blocks; ++block) ++advances[block];
        const long elements = (iteration % 2 ? 8192 : 257) * (dtype == 2 ? 4 : 8);
        const auto first = payload(0, iteration + 300, elements, dtype);
        const auto second = payload(1, iteration + 300, elements, dtype);
        const auto expected = oracle(first, second, dtype);
        upload(states[0], first); upload(states[1], second);
        both([&](int rank) {
            Rank& state = states[rank];
            delayed(rank, iteration);
            rad_check(r3dh_ar_launch(states[1 - rank].raw_scratch, state.raw_scratch,
                states[1 - rank].raw_flags, state.raw_flags, state.raw_seq, 8192,
                state.input + guard, state.output + guard, elements, dtype, state.stream,
                blocks, 128, iteration % 2 ? 4 : 1, 0));
            hip_check(hipStreamSynchronize(state.stream));
        });
        verify(states[0], first, expected, dtype, 0);
        verify(states[1], second, expected, dtype, 0);
    }
    for (int rank = 0; rank < 2; ++rank) {
        hip_check(hipSetDevice(states[rank].device));
        std::vector<unsigned> sequence(2 * capacity + 1);
        hip_check(hipMemcpy(sequence.data(), states[rank].raw_seq, sequence.size() * 4,
                            hipMemcpyDeviceToHost));
        for (unsigned block = 0; block < capacity; ++block) {
            require(sequence[block] == seed + advances[block], "handshake wrap");
            require(sequence[capacity + block] == seed + 40u, "parity wrap");
        }
    }
    std::printf("wrap_latepush,%s,out,40,%llu,PASS\n", names[dtype],
                (unsigned long long)(compared - before));
}
}
int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    try {
        if (argc != 2 || (std::strcmp(argv[1], "--host-only") && std::strcmp(argv[1], "--gpu") &&
                          std::strcmp(argv[1], "--smoke")))
            throw std::runtime_error("usage: r3dh_ar_check --host-only|--smoke|--gpu (GPU requires both OJ locks)");
        std::puts("case,dtype,mode,iterations,elements_compared,status");
        host_tests();
        if (!std::strcmp(argv[1], "--host-only")) return 0;
        int devices = 0;
        hip_check(hipGetDeviceCount(&devices));
        require(devices == 2, "exactly two visible devices required");
        for (int device = 0; device < 2; ++device) {
            hipDeviceProp_t properties{};
            hip_check(hipGetDeviceProperties(&properties, device));
            require(!std::strncmp(properties.gcnArchName, "gfx1100", 7), "gfx1100 required");
            std::printf("device,%d,%s,0,0,PASS\n", device, properties.gcnArchName);
        }
        if (!std::strcmp(argv[1], "--smoke")) {
            suite(0, true);
            std::printf("TOTAL,bf16,smoke,32,%llu,PASS\n", (unsigned long long)compared);
            return 0;
        }
        for (int dtype = 0; dtype < 3; ++dtype) suite(dtype);
        std::printf("TOTAL,all,all,2040,%llu,PASS\n", (unsigned long long)compared);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
