#include <hip/hip_runtime.h>
#include <array>
#include <cstring>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace {
void checked(hipError_t error) {
    if (error != hipSuccess) throw std::runtime_error(hipGetErrorString(error));
}
__global__ void transfer(const unsigned* input, unsigned* peer, const unsigned* local,
                         unsigned* observed, unsigned* own_flag, unsigned* peer_flag,
                         unsigned count, unsigned sequence, int handshake) {
    unsigned* destination = handshake == 2 ? const_cast<unsigned*>(local) : peer;
    if (handshake != 3)
        for (unsigned index = threadIdx.x; index < count; index += blockDim.x)
            destination[index] = input[index];
    __threadfence_system();
    __syncthreads();
    if (!handshake) return;
    if (threadIdx.x == 0) {
        __hip_atomic_store(peer_flag, sequence, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
        const auto start = wall_clock64();
        own_flag[1] = 0;
        while (__hip_atomic_load(own_flag, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) != sequence) {
            if (wall_clock64() - start > 10000000ull) { own_flag[1] = 1; break; }
        }
    }
    __syncthreads();
    for (unsigned index = threadIdx.x; index < count; index += blockDim.x)
        observed[index] = reinterpret_cast<const volatile unsigned*>(
            handshake == 2 ? peer : local)[index];
}
struct Device {
    int number = -1;
    unsigned* input = nullptr;
    unsigned* scratch = nullptr;
    unsigned* observed = nullptr;
    unsigned* flag = nullptr;
    hipStream_t stream = nullptr;
    bool host = false;
    ~Device() {
        if (number < 0) return;
        (void)hipSetDevice(number);
        if (stream) (void)hipStreamSynchronize(stream);
        if (input) (void)hipFree(input);
        if (scratch) { if (host) (void)hipHostFree(scratch); else (void)hipFree(scratch); }
        if (observed) (void)hipFree(observed);
        if (flag) (void)hipHostFree(flag);
        if (stream) (void)hipStreamDestroy(stream);
    }
};
}
int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    try {
        int devices = 0;
        checked(hipGetDeviceCount(&devices));
        if (devices != 2) throw std::runtime_error("two visible GPUs required");
        const char* memory_names[] = {"coarse", "fine", "host"};
        const char* modes[] = {"joined", "push", "pull", "split"};
        int selected_kind = -1, selected_mode = -1;
        if (argc != 1 && argc != 3) throw std::runtime_error("usage: r3dh_ar_transport_check [coarse|fine|host joined|push|pull|split]");
        if (argc == 3) {
            for (int kind = 0; kind < 3; ++kind) if (!std::strcmp(argv[1], memory_names[kind])) selected_kind = kind;
            for (int mode = 0; mode < 4; ++mode) if (!std::strcmp(argv[2], modes[mode])) selected_mode = mode;
            if (selected_kind < 0 || selected_mode < 0) throw std::runtime_error("invalid memory type or mode");
        }
        int runtime = 0, driver = 0;
        checked(hipRuntimeGetVersion(&runtime)); checked(hipDriverGetVersion(&driver));
        std::printf("runtime,hip=%d,driver=%d\n", runtime, driver);
        unsigned failed = 0;
        for (int kind = 0; kind < 3; ++kind) {
            if (selected_kind >= 0 && kind != selected_kind) continue;
            Device states[2];
            for (int rank = 0; rank < 2; ++rank) {
                auto& state = states[rank];
                state.number = 1 - rank;
                state.host = kind == 2;
                checked(hipSetDevice(state.number));
                int accessible = 0;
                checked(hipDeviceCanAccessPeer(&accessible, state.number, rank));
                if (!accessible) throw std::runtime_error("P2P unavailable");
                const auto enabled = hipDeviceEnablePeerAccess(rank, 0);
                if (enabled == hipErrorPeerAccessAlreadyEnabled) (void)hipGetLastError();
                else checked(enabled);
                checked(hipStreamCreateWithFlags(&state.stream, hipStreamNonBlocking));
                checked(hipMalloc((void**)&state.input, 4096));
                checked(hipMalloc((void**)&state.observed, 4096));
                checked(hipHostMalloc((void**)&state.flag, 4096,
                                      hipHostMallocMapped | hipHostMallocCoherent));
                *state.flag = 0;
                if (kind == 2)
                    checked(hipHostMalloc((void**)&state.scratch, 4096,
                                          hipHostMallocMapped | hipHostMallocCoherent));
                else if (kind == 1)
                    checked(hipExtMallocWithFlags((void**)&state.scratch, 4096,
                                                  hipDeviceMallocFinegrained));
                else checked(hipMalloc((void**)&state.scratch, 4096));
                char bus[32]{};
                checked(hipDeviceGetPCIBusId(bus, sizeof(bus), state.number));
                std::printf("mapping,kind=%d,rank=%d,device=%d,pci=%s,scratch=%p,flag=%p\n",
                            kind, rank, state.number, bus, (void*)state.scratch, (void*)state.flag);
            }
            for (int handshake = 0; handshake < 4; ++handshake) {
                if (selected_mode >= 0 && handshake != selected_mode) continue;
                for (unsigned iteration = 0; iteration < 8; ++iteration) {
                    const unsigned counts[] = {4, 20, 124, 1024};
                    const unsigned count = counts[iteration % 4];
                    std::array<std::vector<unsigned>, 2> inputs;
                    for (int rank = 0; rank < 2; ++rank) {
                        auto& state = states[rank];
                        checked(hipSetDevice(state.number));
                        inputs[rank].resize(count);
                        for (unsigned index = 0; index < count; ++index)
                            inputs[rank][index] = (index + 1) * 0x9e3779b9u ^
                                (iteration + 1) * 0x85ebca6bu ^ (rank + 1) * 0xc2b2ae35u;
                        checked(hipMemcpy(state.input, inputs[rank].data(), count * 4, hipMemcpyHostToDevice));
                        checked(hipMemset(state.scratch, 0, 4096));
                        checked(hipDeviceSynchronize());
                    }
                    for (int rank = 0; rank < 2; ++rank) {
                        auto& state = states[rank];
                        auto& peer = states[1 - rank];
                        checked(hipSetDevice(state.number));
                        if (handshake == 3)
                            transfer<<<1, 256, 0, state.stream>>>(state.input, peer.scratch, state.scratch,
                                state.observed, state.flag, peer.flag, count, iteration + 1, 0);
                        transfer<<<1, 256, 0, state.stream>>>(state.input, peer.scratch, state.scratch,
                            state.observed, state.flag, peer.flag, count, iteration + 1, handshake);
                        checked(hipGetLastError());
                    }
                    for (int rank = 0; rank < 2; ++rank) {
                        checked(hipSetDevice(states[rank].number));
                        checked(hipStreamSynchronize(states[rank].stream));
                    }
                    for (int rank = 0; rank < 2; ++rank) {
                        auto& state = states[rank];
                        checked(hipSetDevice(state.number));
                        checked(hipStreamSynchronize(state.stream));
                        std::vector<unsigned> actual(count);
                        checked(hipMemcpy(actual.data(), handshake ? state.observed : state.scratch,
                                          count * 4, hipMemcpyDeviceToHost));
                        unsigned different = 0;
                        for (unsigned index = 0; index < count; ++index)
                            different += actual[index] != inputs[1 - rank][index];
                        std::vector<unsigned> landed(count);
                        checked(hipMemcpy(landed.data(), state.scratch, count * 4, hipMemcpyDeviceToHost));
                        unsigned landed_different = 0;
                        for (unsigned index = 0; index < count; ++index)
                            landed_different += landed[index] != inputs[handshake == 2 ? rank : 1 - rank][index];
                        const bool bad_flag = handshake && (*state.flag != iteration + 1 || state.flag[1]);
                        failed += different != 0 || landed_different != 0 || bad_flag;
                        std::printf("transport,kind=%d,handshake=%d,iteration=%u,rank=%d,words=%u,different=%u,first=%08x,expected=%08x,landed_different=%u,flag=%u,timeout=%u,status=%s\n",
                            kind, handshake, iteration, rank, count, different, actual[0], inputs[1 - rank][0], landed_different,
                            *state.flag, state.flag[1],
                            different || landed_different || bad_flag ? "FAIL" : "PASS");
                    }
                }
            }
        }
        std::printf("TOTAL,failed=%u\n", failed);
        return failed ? 1 : 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 2;
    }
}
