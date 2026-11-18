#include <hip/hip_runtime.h>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {
void checked(hipError_t error) {
    if (error != hipSuccess) throw std::runtime_error(hipGetErrorString(error));
}
__global__ void push(const unsigned* input, unsigned* peer, unsigned count,
                     unsigned* peer_flag, unsigned sequence) {
    for (unsigned index = threadIdx.x; index < count; index += blockDim.x)
        peer[index] = input[index];
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0 && peer_flag)
        __hip_atomic_store(peer_flag, sequence, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__global__ void observe(const unsigned* source, unsigned* output, unsigned count,
                        unsigned* flag, unsigned sequence, unsigned* timed_out) {
    if (threadIdx.x == 0) {
        unsigned expired = 0;
        const auto start = wall_clock64();
        while (flag && __hip_atomic_load(flag, __ATOMIC_ACQUIRE,
                                         __HIP_MEMORY_SCOPE_SYSTEM) != sequence) {
            if (wall_clock64() - start > 10000000ull) { expired = 1; break; }
        }
        *timed_out = expired;
    }
    __syncthreads();
    for (unsigned index = threadIdx.x; index < count; index += blockDim.x)
        output[index] = reinterpret_cast<const volatile unsigned*>(source)[index];
}
__global__ void live_exchange(const unsigned* input, unsigned* peer, const unsigned* local,
                              unsigned* output, unsigned count, unsigned* own_flag,
                              unsigned* peer_flag, unsigned sequence, unsigned* timed_out) {
    for (unsigned index = threadIdx.x; index < count; index += blockDim.x)
        peer[index] = input[index];
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        __hip_atomic_store(peer_flag, sequence, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
        const auto start = wall_clock64();
        unsigned expired = 0;
        while (__hip_atomic_load(own_flag, __ATOMIC_ACQUIRE,
                                  __HIP_MEMORY_SCOPE_SYSTEM) != sequence) {
            if (wall_clock64() - start > 10000000ull) { expired = 1; break; }
        }
        *timed_out = expired;
    }
    __syncthreads();
    __threadfence_system();
    for (unsigned index = threadIdx.x; index < count; index += blockDim.x)
        output[index] = reinterpret_cast<const volatile unsigned*>(local)[index];
}
struct Device {
    int number = -1;
    bool host = false;
    unsigned* input = nullptr;
    unsigned* scratch = nullptr;
    unsigned* output = nullptr;
    unsigned* flag = nullptr;
    unsigned* timed_out = nullptr;
    hipStream_t stream = nullptr;
    ~Device() {
        if (number < 0) return;
        (void)hipSetDevice(number);
        if (stream) (void)hipStreamSynchronize(stream);
        if (input) (void)hipFree(input);
        if (scratch) { if (host) (void)hipHostFree(scratch); else (void)hipFree(scratch); }
        if (flag) { if (host) (void)hipHostFree(flag); else (void)hipFree(flag); }
        if (output) (void)hipFree(output);
        if (timed_out) (void)hipFree(timed_out);
        if (stream) (void)hipStreamDestroy(stream);
    }
};
}
int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    try {
        if (argc != 2 || (std::strcmp(argv[1], "coarse") && std::strcmp(argv[1], "fine") &&
                          std::strcmp(argv[1], "host")))
            throw std::runtime_error("usage: r3dh_ar_visibility_check coarse|fine|host");
        const bool host = !std::strcmp(argv[1], "host");
        const bool fine = !std::strcmp(argv[1], "fine");
        const char* override_value = std::getenv("HSA_FORCE_FINE_GRAIN_PCIE");
        std::printf("environment,kind=%s,HSA_FORCE_FINE_GRAIN_PCIE=%s\n", argv[1],
                    override_value ? override_value : "unset");
        int devices = 0;
        checked(hipGetDeviceCount(&devices));
        if (devices != 2) throw std::runtime_error("exactly two visible GPUs required");
        Device states[2];
        for (int rank = 0; rank < 2; ++rank) {
            auto& state = states[rank];
            state.number = rank;
            state.host = host;
            checked(hipSetDevice(rank));
            int accessible = 0;
            checked(hipDeviceCanAccessPeer(&accessible, rank, 1 - rank));
            if (!accessible) throw std::runtime_error("peer access unavailable");
            const auto enabled = hipDeviceEnablePeerAccess(1 - rank, 0);
            if (enabled == hipErrorPeerAccessAlreadyEnabled) (void)hipGetLastError();
            else checked(enabled);
            checked(hipStreamCreateWithFlags(&state.stream, hipStreamNonBlocking));
            checked(hipMalloc((void**)&state.input, 4096));
            checked(hipMalloc((void**)&state.output, 4096));
            checked(hipMalloc((void**)&state.timed_out, sizeof(unsigned)));
            auto allocate = [&](unsigned** pointer) {
                if (host) checked(hipHostMalloc((void**)pointer, 4096,
                    hipHostMallocMapped | hipHostMallocCoherent | hipHostMallocPortable));
                else if (fine) checked(hipExtMallocWithFlags((void**)pointer, 4096,
                                                            hipDeviceMallocFinegrained));
                else checked(hipMalloc((void**)pointer, 4096));
            };
            allocate(&state.scratch);
            allocate(&state.flag);
            char bus[32]{};
            hipDeviceProp_t properties{};
            checked(hipDeviceGetPCIBusId(bus, sizeof(bus), rank));
            checked(hipGetDeviceProperties(&properties, rank));
            std::printf("mapping,rank=%d,pci=%s,arch=%s,peer_access=%d,scratch=%p,flag=%p\n",
                rank, bus, properties.gcnArchName, accessible, (void*)state.scratch, (void*)state.flag);
        }
        unsigned failed = 0, tests = 0;
        const char* methods[] = {"peer_copy", "kernel_push", "kernel_pull", "doorbell"};
        for (int method = 0; method < 4; ++method) {
            for (unsigned iteration = 0; iteration < 8; ++iteration) {
                const unsigned counts[] = {4, 20, 124, 1024};
                const unsigned count = counts[iteration % 4];
                const unsigned sequence = iteration + 1;
                std::array<std::vector<unsigned>, 2> inputs;
                for (int rank = 0; rank < 2; ++rank) {
                    auto& state = states[rank];
                    checked(hipSetDevice(rank));
                    inputs[rank].resize(count);
                    for (unsigned index = 0; index < count; ++index)
                        inputs[rank][index] = (index + 1) * 0x9e3779b9u ^
                            sequence * 0x85ebca6bu ^ (rank + 1) * 0xc2b2ae35u;
                    checked(hipMemcpy(state.input, inputs[rank].data(), count * 4, hipMemcpyHostToDevice));
                    checked(hipMemset(state.scratch, 0, 4096));
                    checked(hipMemset(state.flag, 0, 4096));
                    checked(hipDeviceSynchronize());
                    if (method == 2) {
                        checked(hipMemcpy(state.scratch, inputs[rank].data(), count * 4,
                                          hipMemcpyHostToDevice));
                        checked(hipDeviceSynchronize());
                    }
                }
                for (int rank = 0; rank < 2; ++rank) {
                    auto& state = states[rank];
                    auto& peer = states[1 - rank];
                    checked(hipSetDevice(rank));
                    if (method == 0)
                        checked(hipMemcpyPeerAsync(peer.scratch, 1 - rank, state.input, rank,
                                                   count * 4, state.stream));
                    else if (method == 2)
                        observe<<<1, 256, 0, state.stream>>>(peer.scratch, state.output, count,
                                                            nullptr, sequence, state.timed_out);
                    else if (method == 3)
                        live_exchange<<<1, 256, 0, state.stream>>>(state.input, peer.scratch,
                            state.scratch, state.output, count, state.flag, peer.flag,
                            sequence, state.timed_out);
                    else
                        push<<<1, 256, 0, state.stream>>>(state.input, peer.scratch, count,
                            method == 3 ? peer.flag : nullptr, sequence);
                    checked(hipGetLastError());
                }
                for (int rank = 0; rank < 2; ++rank) {
                    checked(hipSetDevice(rank));
                    checked(hipStreamSynchronize(states[rank].stream));
                }
                if (method != 2 && method != 3) {
                    for (int rank = 0; rank < 2; ++rank) {
                        auto& state = states[rank];
                        checked(hipSetDevice(rank));
                        observe<<<1, 256, 0, state.stream>>>(state.scratch, state.output, count,
                            method == 3 ? state.flag : nullptr, sequence, state.timed_out);
                        checked(hipGetLastError());
                    }
                }
                for (int rank = 0; rank < 2; ++rank) {
                    auto& state = states[rank];
                    checked(hipSetDevice(rank));
                    checked(hipStreamSynchronize(state.stream));
                    std::vector<unsigned> actual(count);
                    unsigned expired = 0, doorbell = 0, different = 0;
                    unsigned long long checksum = 0, expected_checksum = 0;
                    checked(hipMemcpy(actual.data(), state.output, count * 4, hipMemcpyDeviceToHost));
                    checked(hipMemcpy(&expired, state.timed_out, sizeof(expired), hipMemcpyDeviceToHost));
                    checked(hipMemcpy(&doorbell, state.flag, sizeof(doorbell), hipMemcpyDeviceToHost));
                    for (unsigned index = 0; index < count; ++index) {
                        different += actual[index] != inputs[1 - rank][index];
                        checksum += actual[index];
                        expected_checksum += inputs[1 - rank][index];
                    }
                    const bool bad = different || expired || (method == 3 && doorbell != sequence);
                    failed += bad;
                    ++tests;
                    std::printf("visibility,method=%s,iteration=%u,rank=%d,words=%u,different=%u,"
                        "checksum=%llu,expected=%llu,first=%08x,expected_first=%08x,"
                        "doorbell=%u,timeout=%u,status=%s\n", methods[method], iteration, rank,
                        count, different, checksum, expected_checksum, actual[0], inputs[1 - rank][0],
                        doorbell, expired, bad ? "FAIL" : "PASS");
                }
            }
        }
        std::printf("TOTAL,kind=%s,tests=%u,failed=%u\n", argv[1], tests, failed);
        return failed ? 1 : 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 2;
    }
}
