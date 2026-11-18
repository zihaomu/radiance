#include "r3dh_ar.h"
#include <algorithm>
#include <chrono>
#include <climits>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>

namespace {
constexpr unsigned capacity = 512;
struct Instance {
    int rank = 0, device = -1, dtype = 0;
    long max_numel = 0, slot_words = 0;
    void* scratch = nullptr;
    unsigned* flags = nullptr;
    unsigned* sequence = nullptr;
    void* peer_scratch = nullptr;
    unsigned* peer_flags = nullptr;
    ~Instance() {
        if (device < 0) return;
        int previous = -1;
        (void)hipGetDevice(&previous);
        if (hipSetDevice(device) == hipSuccess) {
            if (scratch) (void)hipFree(scratch);
            if (flags) (void)hipFree(flags);
            if (sequence) (void)hipFree(sequence);
        }
        if (previous >= 0) (void)hipSetDevice(previous);
    }
};
struct Published {
    void* scratch = nullptr;
    unsigned* flags = nullptr;
    long max_numel = 0;
    int device = -1, dtype = -1, status = RAD_OK;
};
struct Rendezvous {
    std::mutex mutex;
    std::condition_variable changed;
    Published peer[2];
    bool published[2] = {}, ready[2] = {};
    int departed = 0;
    bool broken = false;
} rendezvous;
template <typename Predicate>
bool wait_for(std::unique_lock<std::mutex>& lock, Predicate predicate) {
    if (!rendezvous.changed.wait_for(lock, std::chrono::seconds(10), [&] {
            return rendezvous.broken || predicate();
        })) {
        rendezvous.broken = true;
        std::fprintf(stderr, "[r3dh_ar] TP2 init rendezvous timed out\n");
        rendezvous.changed.notify_all();
    }
    return !rendezvous.broken;
}
int exchange(Instance& instance, int status) {
    const int rank = instance.rank, other = 1 - rank;
    std::unique_lock<std::mutex> lock(rendezvous.mutex);
    if (!wait_for(lock, [&] { return !rendezvous.published[rank]; })) return RAD_E_STATE;
    rendezvous.peer[rank] = {instance.scratch, instance.flags, instance.max_numel,
                             instance.device, instance.dtype, status};
    rendezvous.published[rank] = true;
    rendezvous.changed.notify_all();
    if (!wait_for(lock, [&] { return rendezvous.published[other]; })) return RAD_E_STATE;
    const Published peer = rendezvous.peer[other];
    if (status == RAD_OK) status = peer.status;
    if (status == RAD_OK && (peer.max_numel != instance.max_numel ||
        peer.dtype != instance.dtype)) status = RAD_E_SHAPE;
    if (status == RAD_OK && peer.device == instance.device) status = RAD_E_DEVICE;
    lock.unlock();
    if (status == RAD_OK) {
        int accessible = 0;
        if (hipDeviceCanAccessPeer(&accessible, instance.device, peer.device) != hipSuccess ||
            !accessible) status = RAD_E_DEVICE;
        else {
            const hipError_t error = hipDeviceEnablePeerAccess(peer.device, 0);
            if (error == hipErrorPeerAccessAlreadyEnabled) (void)hipGetLastError();
            else if (error != hipSuccess) status = RAD_E_DEVICE;
        }
    }
    lock.lock();
    rendezvous.peer[rank].status = status;
    rendezvous.ready[rank] = true;
    rendezvous.changed.notify_all();
    if (!wait_for(lock, [&] { return rendezvous.ready[other]; })) return RAD_E_STATE;
    if (status == RAD_OK) status = rendezvous.peer[other].status;
    if (status == RAD_OK) {
        instance.peer_scratch = peer.scratch;
        instance.peer_flags = peer.flags;
        char local_bus[32]{}, peer_bus[32]{};
        (void)hipDeviceGetPCIBusId(local_bus, sizeof(local_bus), instance.device);
        (void)hipDeviceGetPCIBusId(peer_bus, sizeof(peer_bus), peer.device);
        std::fprintf(stderr, "transport_map,transport=peer_pull,p2p=1,rank=%d,device=%d,pci=%s,peer_device=%d,peer_pci=%s,scratch=%p,flags_vram=%p,sequence=%p,peer_scratch=%p,peer_flags_vram=%p\n",
            rank, instance.device, local_bus, peer.device, peer_bus, instance.scratch,
            (void*)instance.flags, (void*)instance.sequence, peer.scratch, (void*)peer.flags);
    }
    if (++rendezvous.departed == 2) {
        rendezvous.departed = 0;
        rendezvous.published[0] = rendezvous.published[1] = false;
        rendezvous.ready[0] = rendezvous.ready[1] = false;
        rendezvous.changed.notify_all();
    }
    return status;
}
const RadParam* param(const RadParam* params, int count, const char* key) {
    if (!params || count < 0) return nullptr;
    for (int index = 0; index < count; ++index)
        if (params[index].key && !std::strcmp(params[index].key, key)) return params + index;
    return nullptr;
}
int geometry(const RadParam* params, int count, long* maximum, int* dtype) {
    const RadParam* world = param(params, count, "world_size");
    const RadParam* exact = param(params, count, "exact");
    const RadParam* numel = param(params, count, "numel");
    const RadParam* type = param(params, count, "dtype");
    if (!world || world->kind != RAD_P_INT || world->ival != 2 ||
        !exact || exact->kind != RAD_P_INT || exact->ival != 1) return RAD_E_SHAPE;
    if (!type || type->kind != RAD_P_STR || !type->sval) return RAD_E_DTYPE;
    if (!std::strcmp(type->sval, "bf16")) *dtype = 0;
    else if (!std::strcmp(type->sval, "fp16") || !std::strcmp(type->sval, "f16")) *dtype = 1;
    else if (!std::strcmp(type->sval, "f32") || !std::strcmp(type->sval, "fp32")) *dtype = 2;
    else return RAD_E_DTYPE;
    if (!numel || (numel->kind != RAD_P_INT && numel->kind != RAD_P_RANGE)) return RAD_E_SHAPE;
    const long long high = numel->kind == RAD_P_RANGE ? numel->ihi : numel->ival;
    const long element_bytes = *dtype == 2 ? 4 : 2;
    if (numel->ival <= 0 || high < numel->ival ||
        high > (long long)(INT_MAX - 1024) * 16 / element_bytes)
        return RAD_E_SHAPE;
    *maximum = (long)high;
    return RAD_OK;
}
int tensor_count(const RadTensor& tensor, long* count) {
    if (!tensor.data || tensor.rank == 0 || tensor.rank > RAD_MAX_RANK) return RAD_E_SHAPE;
    long elements = 1;
    for (unsigned axis = tensor.rank; axis-- > 0;) {
        const long dimension = tensor.shape[axis];
        if (dimension <= 0 || dimension > LONG_MAX / elements) return RAD_E_SHAPE;
        if (dimension > 1 && tensor.stride[axis] != elements) return RAD_E_STRIDE;
        elements *= dimension;
    }
    *count = elements;
    return RAD_OK;
}
}
extern "C" int r3dh_rad_ar_init(const RadParam* params, int count, int rank, int world, void** out) {
    if (!out) return RAD_E_INVAL;
    *out = nullptr;
    if (world != 2 || rank < 0 || rank >= 2) return RAD_E_SHAPE;
    try {
        auto instance = std::make_unique<Instance>();
        instance->rank = rank;
        int status = geometry(params, count, &instance->max_numel, &instance->dtype);
        if (status != RAD_OK) return status;
        const char* fine_grain = std::getenv("HSA_FORCE_FINE_GRAIN_PCIE");
        if (!fine_grain || std::strcmp(fine_grain, "1")) {
            std::fprintf(stderr, "[r3dh_ar] native peer-pull requires HSA_FORCE_FINE_GRAIN_PCIE=1 before HIP initialization\n");
            return RAD_E_DEVICE;
        }
        const long element_bytes = instance->dtype == 2 ? 4 : 2;
        instance->slot_words = (instance->max_numel * element_bytes + 15) / 16;
        if (hipGetDevice(&instance->device) != hipSuccess) status = RAD_E_DEVICE;
        auto allocate = [&](void** pointer, size_t bytes, bool fine) {
            if (status != RAD_OK) return;
            const hipError_t error = fine
                ? hipExtMallocWithFlags(pointer, bytes, hipDeviceMallocFinegrained)
                : hipMalloc(pointer, bytes);
            if (error != hipSuccess || hipMemset(*pointer, 0, bytes) != hipSuccess)
                status = RAD_E_DEVICE;
        };
        allocate(&instance->scratch, (size_t)instance->slot_words * 32, true);
        if (status == RAD_OK) {
            allocate((void**)&instance->flags, 3 * capacity * sizeof(unsigned), true);
        }
        allocate((void**)&instance->sequence, (2 * capacity + 1) * sizeof(unsigned), false);
        if (hipStreamSynchronize(nullptr) != hipSuccess) status = RAD_E_DEVICE;
        status = exchange(*instance, status);
        if (status != RAD_OK) return status;
        *out = instance.release();
        return RAD_OK;
    } catch (...) {
        return RAD_E_DEVICE;
    }
}
extern "C" void r3dh_rad_ar_fini(void* instance) { delete (Instance*)instance; }
extern "C" int64_t r3dh_rad_ar_scratch(const RadArgs*) { return 0; }
extern "C" int r3dh_rad_ar_oneshot_2rank_exact(const RadArgs* args, RadStream stream) {
    if (!args || !args->instance) return RAD_E_STATE;
    const Instance& instance = *(const Instance*)args->instance;
    int device = -1;
    if (hipGetDevice(&device) != hipSuccess || device != instance.device) return RAD_E_DEVICE;
    if (args->world_size != 2 || args->rank != instance.rank) return RAD_E_SHAPE;
    if (!args->t || args->n_t < 1 || args->n_t > 2) return RAD_E_INVAL;
    long maximum = 0;
    int dtype = -1;
    int status = geometry(args->p, args->n_p, &maximum, &dtype);
    if (status != RAD_OK) return status;
    if (maximum > instance.max_numel || dtype != instance.dtype) return RAD_E_SHAPE;
    const RadTensor& input = args->t[0];
    const RadTensor* output = args->n_t > 1 && args->t[1].data ? &args->t[1] : &input;
    const unsigned expected_dtype = dtype == 0 ? RAD_BF16 : dtype == 1 ? RAD_F16 : RAD_F32;
    if (input.dtype != expected_dtype || output->dtype != expected_dtype) return RAD_E_DTYPE;
    long elements = 0, output_elements = 0;
    status = tensor_count(input, &elements);
    if (status != RAD_OK) return status;
    status = tensor_count(*output, &output_elements);
    if (status != RAD_OK) return status;
    if (elements != output_elements || elements > maximum) return RAD_E_SHAPE;
    const long bytes = elements * (dtype == 2 ? 4 : 2);
    if (bytes % 16 || ((uintptr_t)input.data | (uintptr_t)output->data) & 15u) return RAD_E_ALIGN;
    const uintptr_t source = (uintptr_t)input.data, target = (uintptr_t)output->data;
    if (source != target && (source < target ? target - source : source - target) < (uintptr_t)bytes)
        return RAD_E_INVAL;
    const long words = bytes / 16;
    const int blocks = (int)std::min(24L, std::max(1L, (words + 255) / 256));
    return r3dh_ar_launch(instance.peer_scratch, instance.scratch, instance.peer_flags,
        instance.flags, instance.sequence, instance.slot_words, input.data, output->data,
        elements, dtype, (hipStream_t)stream, blocks, 256, 1, 0);
}
