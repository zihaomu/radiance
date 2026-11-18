#include <hip/hip_runtime.h>
#include "r3dh_attn_qsa.h"
#include <cstring>
namespace {
long integer(const RadArgs* args, const char* key, long fallback = 0) {
    long long value = fallback; rad_args_geti(args, key, &value); return value;
}
bool plane(const RadTensor& tensor, uint32_t type, long rows, long width) {
    return tensor.data && tensor.dtype == type && tensor.rank == 2 && tensor.shape[0] == rows
        && tensor.shape[1] == width && tensor.stride[1] == 1 && tensor.stride[0] >= width;
}
bool vector(const RadTensor& tensor, uint32_t type, long size) {
    return tensor.data && tensor.dtype == type && tensor.rank == 1
        && tensor.shape[0] == size && tensor.stride[0] == 1;
}
struct State {
    const uint16_t* key;
    uint16_t* tail;
    const int *positions, *cu, *sidx, *table, *rope;
    uint16_t* stage;
    int *page, *bpos, *complete;
    long key_pitch, slot_pitch, ring_pitch, index_pitch, table_pitch, rope_pitch, stage_pitch, bpos_pitch;
    int rows, seqs, slots, work, table_width, ring;
};
__global__ void tail(State call) {
    const int token = blockIdx.x, column = threadIdx.x;
    int sequence = -1;
    for (int candidate = 0; candidate < call.seqs; ++candidate)
        if (call.cu[candidate] >= 0 && call.cu[candidate + 1] <= call.rows
            && token >= call.cu[candidate] && token < call.cu[candidate + 1]) { sequence = candidate; break; }
    if (sequence < 0 || token < call.cu[sequence + 1] - call.ring) return;
    const int slot = call.sidx[sequence * call.index_pitch], position = call.positions[token];
    if (slot < 0 || slot >= call.slots || position < 0) return;
    call.tail[slot * call.slot_pitch + (position % call.ring) * call.ring_pitch + column]
        = call.key[token * call.key_pitch + column];
}
__global__ void work(State call) {
    const int item = blockIdx.x;
    if (item < call.seqs) {
        const int begin = call.cu[item], end = call.cu[item + 1];
        call.complete[item] = begin >= 0 && end <= call.rows && end > begin
            ? (call.positions[end - 1] + 1) / 4 : 0;
    }
    int sequence = -1, block = 0, running = 0;
    for (int candidate = 0; candidate < call.seqs; ++candidate) {
        const int begin = call.cu[candidate], end = call.cu[candidate + 1];
        if (begin < 0 || end > call.rows || end <= begin || call.positions[begin] < 0) continue;
        const int lower = call.positions[begin] / 4, upper = (call.positions[end - 1] + 1) / 4;
        const int count = upper > lower ? upper - lower : 0;
        if (item < running + count) { sequence = candidate; block = lower + item - running; break; }
        running += count;
    }
    call.page[item] = -1;
    for (int component = 0; component < (call.rope ? 3 : 1); ++component)
        call.bpos[item * call.bpos_pitch + component] = 0;
    if (sequence < 0 || block < 0 || block >= call.table_width) return;
    const int begin = call.cu[sequence], position = call.positions[begin], first = block * 4;
    const int slot = call.sidx[sequence * call.index_pitch];
    if (slot >= call.slots) return;
    call.page[item] = slot < 0 ? -1 : call.table[sequence * call.table_pitch + block];
    for (int component = 0; component < (call.rope ? 3 : 1); ++component) {
        int value = first;
        if (call.rope) value = first >= position
            ? call.rope[component * call.rope_pitch + begin + first - position]
            : first + call.rope[begin] - position;
        call.bpos[item * call.bpos_pitch + component] = value;
    }
    if (slot < 0) return;
    for (int offset = 0; offset < 4; ++offset) {
        const int index = first + offset;
        if (index >= position && (begin + index - position >= call.cu[sequence + 1])) return;
        const uint16_t* input = index >= position ? call.key + (begin + index - position) * call.key_pitch
            : call.tail + slot * call.slot_pitch + (index % call.ring) * call.ring_pitch;
        uint16_t* output = call.stage + (item * 4 + offset) * call.stage_pitch;
        for (int column = 0; column < 128; ++column) output[column] = input[column];
    }
}
int common(const RadArgs* args, State* call, bool gathering) {
    if (!args || !args->t || args->n_t != (gathering ? 11 : 5)) return RAD_E_INVAL;
    const char* dtype = rad_args_gets(args, "dtype");
    const long ring = integer(args, "ring");
    if (integer(args, "n") != 128 || integer(args, "ratio") != 4 || ring < 4 || ring > 7
        || (dtype && std::strcmp(dtype, "bf16"))) return RAD_E_UNSUPPORTED;
    const RadTensor* tensors = args->t;
    const long rows = tensors[0].shape[0], seqs = tensors[2].shape[0] - 1;
    const RadTensor& state = tensors[3]; const RadTensor& indices = tensors[4];
    if (rows < 1 || rows > 2048 || seqs < 1 || seqs > 8
        || !plane(tensors[0], RAD_BF16, rows, 128) || !vector(tensors[1], RAD_I32, rows)
        || !vector(tensors[2], RAD_I32, seqs + 1) || !state.data || state.dtype != RAD_BF16
        || state.rank != 4 || state.shape[0] < 1 || state.shape[0] > 65536 || state.shape[1] != 1
        || state.shape[2] != ring || state.shape[3] != 128 || state.stride[3] != 1
        || state.stride[2] < 128 || state.stride[0] < ring * state.stride[2]) return RAD_E_SHAPE;
    if (!vector(indices, RAD_I32, seqs) && !(indices.rank == 2 && indices.shape[1] >= 1
        && plane(indices, RAD_I32, seqs, indices.shape[1]))) return RAD_E_SHAPE;
    call->rows = int(rows); call->seqs = int(seqs); call->slots = int(state.shape[0]); call->ring = int(ring);
    call->key = static_cast<const uint16_t*>(tensors[0].data); call->key_pitch = tensors[0].stride[0];
    call->positions = static_cast<const int*>(tensors[1].data); call->cu = static_cast<const int*>(tensors[2].data);
    call->sidx = static_cast<const int*>(indices.data); call->index_pitch = indices.rank == 2 ? indices.stride[0] : 1;
    call->tail = static_cast<uint16_t*>(state.data); call->slot_pitch = state.stride[0]; call->ring_pitch = state.stride[2];
    return RAD_OK;
}
}
extern "C" int r3dh_attn_qsa_tail_store(const RadArgs* args, RadStream stream) {
    State call{}; const int status = common(args, &call, false); if (status != RAD_OK) return status;
    hipLaunchKernelGGL(tail, dim3(call.rows), dim3(128), 0, reinterpret_cast<hipStream_t>(stream), call);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
extern "C" int r3dh_attn_qsa_work(const RadArgs* args, RadStream stream) {
    State call{}; const int status = common(args, &call, true); if (status != RAD_OK) return status;
    const RadTensor* tensors = args->t;
    const long items = tensors[8].shape[0];
    if (items < (call.rows + 3) / 4 + call.seqs || items > 520 || !vector(tensors[8], RAD_I32, items)
        || !plane(tensors[7], RAD_BF16, items * 4, 128) || !vector(tensors[10], RAD_I32, call.seqs)
        || !plane(tensors[5], RAD_I32, call.seqs, tensors[5].shape[1]) || tensors[5].shape[1] < 1
        || tensors[5].shape[1] > 50000) return RAD_E_SHAPE;
    const RadTensor& positions = tensors[9]; const RadTensor& rope = tensors[6];
    if (!(vector(positions, RAD_I32, items) && !rope.data)
        && !(positions.rank == 2 && (positions.shape[1] == 1 || positions.shape[1] == 3)
            && plane(positions, RAD_I32, items, positions.shape[1]) && (!rope.data || positions.shape[1] == 3)))
        return RAD_E_SHAPE;
    if (rope.data && !plane(rope, RAD_I32, 3, call.rows)) return RAD_E_SHAPE;
    call.work = int(items); call.table = static_cast<const int*>(tensors[5].data);
    call.table_pitch = tensors[5].stride[0]; call.table_width = int(tensors[5].shape[1]);
    call.rope = static_cast<const int*>(rope.data); call.rope_pitch = rope.stride[0];
    call.stage = static_cast<uint16_t*>(tensors[7].data); call.stage_pitch = tensors[7].stride[0];
    call.page = static_cast<int*>(tensors[8].data); call.bpos = static_cast<int*>(positions.data);
    call.bpos_pitch = positions.rank == 2 ? positions.stride[0] : 1; call.complete = static_cast<int*>(tensors[10].data);
    hipLaunchKernelGGL(work, dim3(items), dim3(1), 0, reinterpret_cast<hipStream_t>(stream), call);
    return hipGetLastError() == hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
