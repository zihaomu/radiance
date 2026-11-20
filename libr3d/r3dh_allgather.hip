#include "r3dh_allgather_private.h"
#include <hip/hip_runtime.h>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>

namespace {
struct Buffer {
    int device = -1;
    void* scratch = nullptr;
    unsigned* flags = nullptr;
    ~Buffer() {
        if (device < 0) return;
        int previous = -1;
        (void)hipGetDevice(&previous);
        if (hipSetDevice(device) == hipSuccess) {
            if (scratch) (void)hipFree(scratch);
            if (flags) (void)hipFree(flags);
        }
        if (previous >= 0) (void)hipSetDevice(previous);
    }
};
struct Pair {
    Buffer buffers[2];
    R3dhAllgatherGeometry geometry[2];
    std::mutex mutex;
    std::condition_variable changed;
    bool claimed[2]{}, allocated[2]{}, registered[2]{};
    int status[2]{RAD_OK,RAD_OK};
    bool broken = false;
};
struct Instance {
    std::shared_ptr<Pair> pair;
    int rank = 0;
};
std::mutex registration_mutex;
std::weak_ptr<Pair> pending;
template<class Predicate>
bool wait(Pair& pair, std::unique_lock<std::mutex>& lock, Predicate predicate) {
    if (!pair.changed.wait_for(lock,std::chrono::seconds(10),[&] { return pair.broken || predicate(); })) {
        pair.broken = true;
        pair.changed.notify_all();
    }
    return !pair.broken;
}
__device__ unsigned load(const unsigned* pointer) {
    return __hip_atomic_load(pointer,__ATOMIC_ACQUIRE,__HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ void publish(unsigned* pointer, unsigned value) {
    __hip_atomic_store(pointer,value,__ATOMIC_RELEASE,__HIP_MEMORY_SCOPE_SYSTEM);
}
__device__ bool await(const unsigned* pointer, unsigned sequence, unsigned* error) {
    const auto start = wall_clock64();
    while (load(pointer) != sequence) {
        if (wall_clock64() - start > 200000000ull) {
            publish(error,1u);
            return false;
        }
        __builtin_amdgcn_s_sleep(1);
    }
    return true;
}
__global__ void gather(const uint4* input, uint4* output, uint4* local_scratch,
                        const uint4* peer_scratch, unsigned* local_flags,
                        unsigned* peer_flags, int words, int row_words,
                        int input_pitch, int output_pitch, int rank) {
    __shared__ unsigned sequence;
    __shared__ bool valid;
    const int lane = threadIdx.x;
    if (!lane) {
        sequence = local_flags[3] + 1u;
        local_flags[3] = sequence;
        valid = load(local_flags + 2) == 0;
    }
    __syncthreads();
    if (!valid) return;
    for (int index = lane; index < words; index += blockDim.x)
        local_scratch[index] = input[size_t(index / row_words) * input_pitch + index % row_words];
    __threadfence_system();
    __syncthreads();
    if (!lane) {
        publish(local_flags,sequence);
        valid = await(peer_flags,sequence,local_flags + 2);
    }
    __syncthreads();
    if (!valid) return;
    __threadfence_system();
    for (int index = lane; index < words; index += blockDim.x) {
        const size_t destination = size_t(index / row_words) * output_pitch + index % row_words;
        const volatile unsigned* peer = reinterpret_cast<const volatile unsigned*>(peer_scratch + index);
        const uint4 remote = make_uint4(peer[0],peer[1],peer[2],peer[3]);
        output[destination + rank * row_words] = local_scratch[index];
        output[destination + (1 - rank) * row_words] = remote;
    }
    __threadfence_system();
    __syncthreads();
    if (!lane) {
        publish(local_flags + 1,sequence);
        valid = await(peer_flags + 1,sequence,local_flags + 2);
    }
}
}
extern "C" int r3dh_allgather_init(const RadParam* params, int count, int rank, int world, void** out) {
    if (!out) return RAD_E_INVAL;
    *out = nullptr;
    if (world != 2 || rank < 0 || rank > 1) return RAD_E_SHAPE;
    R3dhAllgatherGeometry geometry;
    int status = r3dh_allgather_init_geometry(params,count,&geometry);
    if (status != RAD_OK) return status;
    const char* fine = std::getenv("HSA_FORCE_FINE_GRAIN_PCIE");
    if (!fine || std::strcmp(fine,"1")) return RAD_E_DEVICE;
    try {
        auto instance = std::make_unique<Instance>();
        instance->rank = rank;
        {
            std::lock_guard<std::mutex> lock(registration_mutex);
            auto group = pending.lock();
            if (!group) { group = std::make_shared<Pair>(); pending = group; }
            if (group->claimed[rank]) return RAD_E_STATE;
            instance->pair = group;
            group->claimed[rank] = true;
            if (group->claimed[0] && group->claimed[1]) pending.reset();
        }
        Pair& pair = *instance->pair;
        Buffer& buffer = pair.buffers[rank];
        pair.geometry[rank] = geometry;
        if (hipGetDevice(&buffer.device) != hipSuccess) status = RAD_E_DEVICE;
        if (status == RAD_OK && hipExtMallocWithFlags(&buffer.scratch,geometry.high * geometry.bytes,
                                                     hipDeviceMallocFinegrained) != hipSuccess) status = RAD_E_DEVICE;
        if (status == RAD_OK && hipExtMallocWithFlags((void**)&buffer.flags,16,
                                                     hipDeviceMallocFinegrained) != hipSuccess) status = RAD_E_DEVICE;
        if (status == RAD_OK && (hipMemset(buffer.flags,0,16) != hipSuccess ||
            hipStreamSynchronize(nullptr) != hipSuccess)) status = RAD_E_DEVICE;
        std::unique_lock<std::mutex> lock(pair.mutex);
        pair.status[rank] = status;
        pair.allocated[rank] = true;
        pair.changed.notify_all();
        if (!wait(pair,lock,[&] { return pair.allocated[1-rank]; })) return RAD_E_STATE;
        const auto& other = pair.geometry[1-rank];
        if (status == RAD_OK) status = pair.status[1-rank];
        if (status == RAD_OK && (geometry.low != other.low || geometry.high != other.high ||
            geometry.row != other.row || geometry.dtype != other.dtype)) status = RAD_E_SHAPE;
        const int peer_device = pair.buffers[1-rank].device;
        if (status == RAD_OK && peer_device == buffer.device) status = RAD_E_DEVICE;
        lock.unlock();
        if (status == RAD_OK) {
            int accessible = 0;
            if (hipDeviceCanAccessPeer(&accessible,buffer.device,peer_device) != hipSuccess || !accessible)
                status = RAD_E_DEVICE;
            else {
                const hipError_t error = hipDeviceEnablePeerAccess(peer_device,0);
                if (error == hipErrorPeerAccessAlreadyEnabled) (void)hipGetLastError();
                else if (error != hipSuccess) status = RAD_E_DEVICE;
            }
        }
        lock.lock();
        pair.status[rank] = status;
        pair.registered[rank] = true;
        pair.changed.notify_all();
        if (!wait(pair,lock,[&] { return pair.registered[1-rank]; })) return RAD_E_STATE;
        if (status == RAD_OK) status = pair.status[1-rank];
        if (status != RAD_OK) return status;
        char bus[32]{}, peer_bus[32]{};
        (void)hipDeviceGetPCIBusId(bus,sizeof(bus),buffer.device);
        (void)hipDeviceGetPCIBusId(peer_bus,sizeof(peer_bus),peer_device);
        std::fprintf(stderr,"allgather_transport,rank=%d,pci=%s,peer=%s,finegrain_vram=1,p2p=1,ready_done=system_acquire_release\n",rank,bus,peer_bus);
        *out = instance.release();
        return RAD_OK;
    } catch (...) { return RAD_E_DEVICE; }
}
extern "C" void r3dh_allgather_fini(void* instance) { delete static_cast<Instance*>(instance); }
extern "C" int r3dh_allgather_launch(const RadArgs* args, RadStream stream) {
    if (!args || !args->instance) return RAD_E_STATE;
    R3dhAllgatherGeometry geometry;
    R3dhAllgatherView view;
    int status = r3dh_allgather_view(args,&geometry,&view);
    if (status != RAD_OK) return status;
    const Instance& instance = *static_cast<const Instance*>(args->instance);
    const Pair& pair = *instance.pair;
    const auto& bound = pair.geometry[instance.rank];
    if (args->rank != instance.rank || geometry.dtype != bound.dtype ||
        geometry.high > bound.high || geometry.row != bound.row) return RAD_E_SHAPE;
    const Buffer& local = pair.buffers[instance.rank];
    const Buffer& peer = pair.buffers[1-instance.rank];
    int device = -1;
    if (hipGetDevice(&device) != hipSuccess || device != local.device) return RAD_E_DEVICE;
    gather<<<1,256,0,(hipStream_t)stream>>>((const uint4*)args->t[0].data,(uint4*)args->t[1].data,
        (uint4*)local.scratch,(const uint4*)peer.scratch,local.flags,peer.flags,
        view.numel * geometry.bytes / 16,view.row * geometry.bytes / 16,
        view.input_pitch * geometry.bytes / 16,view.output_pitch * geometry.bytes / 16,instance.rank);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
extern "C" int r3dh_allgather_status(void* pointer, unsigned* error) {
    if (!pointer || !error) return RAD_E_INVAL;
    const Instance& instance = *static_cast<Instance*>(pointer);
    const Buffer& buffer = instance.pair->buffers[instance.rank];
    int device = -1;
    if (hipGetDevice(&device) != hipSuccess || device != buffer.device) return RAD_E_DEVICE;
    return hipMemcpy(error,buffer.flags + 2,sizeof(unsigned),hipMemcpyDeviceToHost) == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
extern "C" int r3dh_allgather_seed(void* pointer, unsigned sequence) {
    if (!pointer) return RAD_E_INVAL;
    const Instance& instance = *static_cast<Instance*>(pointer);
    const Buffer& buffer = instance.pair->buffers[instance.rank];
    int device = -1;
    if (hipGetDevice(&device) != hipSuccess || device != buffer.device) return RAD_E_DEVICE;
    const unsigned flags[]{sequence,sequence,0,sequence};
    return hipMemcpy(buffer.flags,flags,sizeof(flags),hipMemcpyHostToDevice) == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
