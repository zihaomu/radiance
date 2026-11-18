#include <array>
#include "../core/rad_internal.h"

namespace qsa_ring_aql {
struct Counts {
    uint64_t fixtures = 0, work = 0, tail = 0, rejects = 0, guards = 0;
    uint64_t bytes = 0, reference_bytes = 0, persistent = 0, rewinds = 0, wraps = 0;
};
int load(const RadTensor& tensor, int row, int column = 0) {
    return static_cast<const int*>(tensor.data)[row * tensor.stride[0] + column];
}
void work_expected(const std::array<RadTensor, 11>& tensors, int ring) {
    const int sequences = int(tensors[2].shape[0]) - 1;
    const int rows = int(tensors[0].shape[0]), capacity = int(tensors[8].shape[0]);
    const int components = tensors[6].data ? 3 : 1;
    for (int item = 0; item < capacity; ++item) {
        static_cast<int*>(tensors[8].data)[item] = -1;
        for (int component = 0; component < components; ++component)
            static_cast<int*>(tensors[9].data)[item * tensors[9].stride[0] + component] = 0;
    }
    int item = 0;
    for (int sequence = 0; sequence < sequences; ++sequence) {
        const int begin = load(tensors[2], sequence), end = load(tensors[2], sequence + 1);
        const bool held = begin >= 0 && end <= rows && end > begin;
        static_cast<int*>(tensors[10].data)[sequence] = held ? (load(tensors[1], end - 1) + 1) / 4 : 0;
        if (!held || load(tensors[1], begin) < 0) continue;
        const int first = load(tensors[1], begin), last = load(tensors[1], end - 1);
        const int slot = load(tensors[4], sequence);
        for (int block = first / 4; block < (last + 1) / 4 && item < capacity; ++block, ++item) {
            if (block < 0 || block >= tensors[5].shape[1] || slot >= tensors[3].shape[0]) continue;
            static_cast<int*>(tensors[8].data)[item] = slot < 0 ? -1 : load(tensors[5], sequence, block);
            for (int component = 0; component < components; ++component) {
                const int position = block * 4;
                const int value = !tensors[6].data ? position : position >= first
                    ? load(tensors[6], component, begin + position - first)
                    : position + load(tensors[6], 0, begin) - first;
                static_cast<int*>(tensors[9].data)[item * tensors[9].stride[0] + component] = value;
            }
            if (slot < 0) continue;
            for (int offset = 0; offset < 4; ++offset) {
                const int position = block * 4 + offset;
                const auto* source = position >= first
                    ? static_cast<const uint16_t*>(tensors[0].data) + (begin + position - first) * tensors[0].stride[0]
                    : static_cast<const uint16_t*>(tensors[3].data) + slot * tensors[3].stride[0]
                        + (position % ring) * tensors[3].stride[2];
                auto* destination = static_cast<uint16_t*>(tensors[7].data) + (item * 4 + offset) * tensors[7].stride[0];
                std::copy_n(source, 128, destination);
            }
        }
    }
}
void tail_expected(const std::array<RadTensor, 11>& tensors, int ring) {
    for (int sequence = 0; sequence + 1 < tensors[2].shape[0]; ++sequence) {
        const int begin = load(tensors[2], sequence), end = load(tensors[2], sequence + 1);
        const int slot = load(tensors[4], sequence);
        if (begin < 0 || end > tensors[0].shape[0] || end <= begin || slot < 0 || slot >= tensors[3].shape[0]) continue;
        for (int row = std::max(begin, end - ring); row < end; ++row) {
            const int position = load(tensors[1], row);
            if (position < 0) continue;
            auto* destination = static_cast<uint16_t*>(tensors[3].data) + slot * tensors[3].stride[0]
                + (position % ring) * tensors[3].stride[2];
            const auto* source = static_cast<const uint16_t*>(tensors[0].data) + row * tensors[0].stride[0];
            std::copy_n(source, 128, destination);
        }
    }
}
const RadKernelInfo* named(Library& library, const char* name) {
    auto count = reinterpret_cast<int (*)()>(dlsym(library.handle, "rad_kernel_count"));
    auto at = reinterpret_cast<const RadKernelInfo* (*)(int)>(dlsym(library.handle, "rad_kernel_at"));
    if (!count || !at) throw std::runtime_error("missing QSA enumeration");
    for (int index = 0; index < count(); ++index)
        if (!std::strcmp(at(index)->name, name)) return at(index);
    throw std::runtime_error("missing QSA model row");
}
bool matches(const RadKernelInfo* kernel, const std::array<RadParam, 7>& params) {
    rad::Geometry geometry;
    for (const auto& param : params) {
        if (param.kind == RAD_P_INT) geometry.set_i(param.key, param.ival);
        else geometry.set_s(param.key, param.sval);
    }
    for (int index = 0; index < kernel->n_constraints; ++index)
        if (!rad::constraint_holds(kernel->constraints[index], geometry)) return false;
    return true;
}
void fixture(const RadKernelInfo* work, const RadKernelInfo* tail, const RadKernelInfo* reference_work,
             const RadKernelInfo* reference_tail, RadStream first_stream, RadStream second_stream,
             int ring, int sequences, int rows, int capacity_mode, int layout, bool lifecycle, Counts& counts) {
    const int maximum_rows = lifecycle ? 64 : rows;
    const int initial_rows = lifecycle ? 63 : rows;
    const int capacity = capacity_mode == 0 ? (initial_rows + 3) / 4 + sequences : capacity_mode == 1 ? 519 : 520;
    if (capacity < (initial_rows + 3) / 4 + sequences) return;
    const int band = maximum_rows <= 64 ? 64 : 2048;
    const int slots = sequences + 2, ring_pitch = layout ? 139 : 128;
    const int slot_pitch = ring * ring_pitch + (layout ? 17 : 0);
    std::vector<std::unique_ptr<Buffer>> owners;
    auto add = [&](int type, std::initializer_list<int64_t> shape, int pitch = 0) {
        owners.push_back(std::make_unique<Buffer>(type, shape, RAD_MEM_DEVICE, pitch));
        return owners.back().get();
    };
    Buffer* key = add(RAD_BF16, {maximum_rows, 128}, layout ? 640 : 128);
    if (layout) {
        key->host.data = static_cast<uint16_t*>(key->host.data) + 512;
        key->device.data = static_cast<uint16_t*>(key->device.data) + 512;
    }
    Buffer* positions = add(RAD_I32, {maximum_rows});
    Buffer* cu = add(RAD_I32, {sequences + 1});
    Buffer* state = add(RAD_BF16, {slots, 1, ring, ring_pitch}, slot_pitch);
    state->host.shape[3] = state->device.shape[3] = 128;
    Buffer* indices = layout ? add(RAD_I32, {sequences, 3}, 5) : add(RAD_I32, {sequences});
    Buffer* table = add(RAD_I32, {sequences, 1200}, layout ? 1207 : 1200);
    Buffer* rope = add(RAD_I32, {3, maximum_rows}, maximum_rows + (layout ? 7 : 0));
    Buffer* stage = add(RAD_BF16, {capacity * 4, 128}, layout ? 133 : 128);
    Buffer* page = add(RAD_I32, {capacity});
    Buffer* block_positions = layout ? add(RAD_I32, {capacity, 3}, 5) : add(RAD_I32, {capacity});
    Buffer* complete = add(RAD_I32, {sequences});
    const std::array<Buffer*, 5> outputs{state, stage, page, block_positions, complete};
    for (int slot = 0; slot < slots; ++slot) for (int position = 0; position < ring; ++position)
        for (int column = 0; column < 128; ++column)
            state->data<uint16_t>()[slot * slot_pitch + position * ring_pitch + column] =
                rad_f32_to_bf16(float((slot * 61 + position * 17 + column * 3) % 257 - 128) / 64);
    std::array<RadParam, 7> params{RAD_INT("M", band), RAD_INT("n", 128), RAD_INT("ratio", 4),
        RAD_INT("ring", ring), RAD_INT("seqs", sequences), RAD_INT("work", std::max(capacity, (band + 3) / 4 + sequences)),
        RAD_STR("dtype", "bf16")};
    REQUIRE(matches(work, params)); REQUIRE(matches(tail, params));
    std::vector<int> next(sequences, 0);
    const int valid_steps = lifecycle ? 13 : 2;
    const int guard_steps = rows == 3 && !lifecycle && capacity_mode == 0 ? 4 : 0;
    for (int step = 0; step < valid_steps + guard_steps; ++step) {
        const int guard = step >= valid_steps ? step - valid_steps + 1 : 0;
        const int query_length = lifecycle && step ? 1 + (step - 1) % 3 : 0;
        const int actual_rows = query_length ? sequences * query_length : initial_rows;
        const RadStream stream = step % 2 ? second_stream : first_stream;
        for (int row = 0; row < actual_rows; ++row) for (int column = 0; column < 128; ++column)
            key->data<uint16_t>()[row * key->host.stride[0] + column] =
                rad_f32_to_bf16(float((row * 41 + column * 7 + step * 29) % 503 - 251) / 128);
        for (int sequence = 0; sequence <= sequences; ++sequence)
            cu->data<int>()[sequence] = query_length ? sequence * query_length
                : actual_rows * sequence * sequence / (sequences * sequences);
        if (layout && !query_length) cu->data<int>()[0] = std::min(1, cu->data<int>()[1]);
        for (int sequence = 0; sequence < sequences; ++sequence) {
            const int begin = cu->data<int>()[sequence], end = cu->data<int>()[sequence + 1];
            int start = lifecycle ? next[sequence] : 9 + sequence * 13 + step * (rows + 1);
            if (guard == 4) start = 4804 + sequence * 8;
            indices->data<int>()[sequence * indices->host.stride[0]] = guard == 1 ? slots
                : !lifecycle && step == 0 && layout && sequence == sequences - 1 ? -1 : sequence + 1;
            for (int block = 0; block < 1200; ++block)
                table->data<int>()[sequence * table->host.stride[0] + block] =
                    layout && block % 23 == 7 ? -1 : (block * 7 + sequence * 19 + step * 3) % 1045;
            for (int row = begin; row < end; ++row) {
                positions->data<int>()[row] = guard == 2 ? -1 : start + row - begin;
                for (int component = 0; component < 3; ++component)
                    rope->data<int>()[component * rope->host.stride[0] + row] = start + row - begin + 100 + component * 11 + step * 37;
            }
            if (lifecycle && end > begin) {
                if (start / ring != (start + end - begin) / ring) ++counts.wraps;
                const int rejected = step % 3 ? std::min(ring - 4, end - begin - 1) : 0;
                next[sequence] = start + end - begin - rejected;
                if (rejected) ++counts.rewinds;
            }
        }
        if (guard == 3) {
            for (int sequence = 0; sequence < sequences; ++sequence) cu->data<int>()[sequence] = -1;
            cu->data<int>()[sequences] = actual_rows + 1;
        }
        for (Buffer* output : {stage, page, block_positions, complete})
            std::fill(output->expected.begin(), output->expected.end(), uint8_t(0x5a));
        for (auto& owner : owners) if (step == 0 || owner.get() != state) owner->push(stream);
        REQUIRE_EQ(rad_stream_sync(stream), RAD_OK);
        std::array<RadTensor, 11> host{}, device{};
        for (int index = 0; index < 11; ++index) { host[index] = owners[index]->host; device[index] = owners[index]->device; }
        for (auto* view : {&host, &device}) {
            (*view)[0].shape[0] = (*view)[1].shape[0] = actual_rows;
            (*view)[6].shape[1] = actual_rows;
            if (!layout) (*view)[6] = {};
        }
        RadArgs args{}; args.t = device.data(); args.n_t = 11; args.p = params.data(); args.n_p = int(params.size());
        std::array<std::vector<uint8_t>, 5> before;
        for (int index = 0; index < 5; ++index) before[index] = outputs[index]->expected;
        work_expected(host, ring);
        if (!guard) {
            std::array<std::vector<uint8_t>, 5> independent;
            for (int index = 0; index < 5; ++index) {
                independent[index] = outputs[index]->expected;
                std::copy(before[index].begin(), before[index].end(), outputs[index]->expected.begin());
            }
            args.t = host.data(); REQUIRE_EQ(reference_work->launch(&args, nullptr), RAD_OK);
            for (int index = 0; index < 5; ++index) {
                REQUIRE(outputs[index]->expected == independent[index]);
                counts.reference_bytes += independent[index].size();
            }
        }
        args.t = device.data(); REQUIRE_EQ(work->launch(&args, stream), RAD_OK); ++counts.work;
        for (auto& owner : owners) { REQUIRE(owner->read(stream) == owner->expected); counts.bytes += owner->expected.size(); }
        tail_expected(host, ring);
        if (!guard) {
            const auto independent = state->expected;
            std::copy(before[0].begin(), before[0].end(), state->expected.begin());
            args.t = host.data(); args.n_t = 5; REQUIRE_EQ(reference_tail->launch(&args, nullptr), RAD_OK);
            REQUIRE(state->expected == independent); counts.reference_bytes += independent.size();
        }
        args.t = device.data(); args.n_t = 5; REQUIRE_EQ(tail->launch(&args, stream), RAD_OK); ++counts.tail;
        for (auto& owner : owners) { REQUIRE(owner->read(stream) == owner->expected); counts.bytes += owner->expected.size(); }
        if (step) ++counts.persistent;
        if (guard) ++counts.guards;
        if (step == 0 && capacity_mode == 0 && rows == 3 && !lifecycle) {
            for (const auto* kernel : {work, tail}) {
                const int operands = kernel == work ? 11 : 5;
                auto reject = [&](int defect, int error) {
                    auto bad_tensors = device; auto bad_params = params;
                    RadArgs bad{}; bad.t = bad_tensors.data(); bad.n_t = operands;
                    bad.p = bad_params.data(); bad.n_p = int(bad_params.size());
                    if (defect == 0) bad.n_t = 0;
                    if (defect == 1) bad_params[3].ival = 3;
                    if (defect == 2) bad_params[3].ival = 8;
                    if (defect == 3) bad_params[1].ival = 256;
                    if (defect == 4) bad_params[2].ival = 3;
                    if (defect == 5) bad_params[6] = RAD_STR("dtype", "f32");
                    if (defect == 6) bad_tensors[3].shape[2] = ring + 1;
                    if (defect == 7) bad_tensors[3].stride[2] = 127;
                    if (defect == 8) bad_tensors[3].stride[0] = ring * ring_pitch - 1;
                    if (defect == 9) bad_tensors[3].rank = 3;
                    if (defect == 10) bad_tensors[4].dtype = RAD_BF16;
                    if (defect == 11) bad_tensors[0].stride[1] = 2;
                    if (defect == 12) bad_tensors[8].shape[0] = 0;
                    if (defect == 13) bad_tensors[8].shape[0] = 521;
                    CHECK_EQ(kernel->launch(&bad, reinterpret_cast<RadStream>(0x12345)), error); ++counts.rejects;
                };
                for (int defect = 0; defect < (kernel == work ? 14 : 12); ++defect)
                    reject(defect, defect == 0 ? RAD_E_INVAL : defect <= 5 ? RAD_E_UNSUPPORTED : RAD_E_SHAPE);
            }
            for (auto& owner : owners) { REQUIRE(owner->read(stream) == owner->expected); counts.bytes += owner->expected.size(); }
        }
    }
    ++counts.fixtures;
}
}

TEST(r3d_qsa_ring_actual_aql_state_work_tail) {
    using namespace qsa_ring_aql;
    RadDeviceProps properties{}; REQUIRE_EQ(rad_dev_props(0, &properties), RAD_OK);
    if (properties.is_host_backend) {
        REQUIRE(!std::getenv("RAD_R3D_REQUIRE_DEVICE")); SKIP("explicit gfx1100 card run required");
    }
    REQUIRE_EQ(std::strcmp(properties.arch, "gfx1100"), 0); REQUIRE_EQ(rad_dev_set(0), RAD_OK);
    Library native("libr3d"), reference("libref");
    const auto* work = named(native, "model_qsa_work_strict");
    const auto* tail = named(native, "model_qsa_tail_strict");
    REQUIRE_EQ(std::strcmp(work->name, "model_qsa_work_strict"), 0);
    REQUIRE_EQ(std::strcmp(tail->name, "model_qsa_tail_strict"), 0);
    REQUIRE_EQ(work->domain, RAD_DOMAIN_DEVICE); REQUIRE_EQ(tail->domain, RAD_DOMAIN_DEVICE);
    uint64_t registration = 0;
    for (int ring : {0, 3, 4, 5, 6, 7, 8}) for (int band : {1, 3, 16, 32, 64, 2048})
        for (int sequences : {1, 2, 8}) for (int capacity : {0, 1, 16, 17, 519, 520, 521}) {
            std::array<RadParam, 7> params{RAD_INT("M", band), RAD_INT("n", 128), RAD_INT("ratio", 4),
                RAD_INT("ring", ring), RAD_INT("seqs", sequences), RAD_INT("work", capacity), RAD_STR("dtype", "bf16")};
            for (const auto* kernel : {work, tail}) {
                const bool valid_ring = ring >= 4 && ring <= 7;
                REQUIRE_EQ(matches(kernel, params), valid_ring && (kernel == tail || (capacity >= 1 && capacity <= 520)));
                const bool valid_shape = valid_ring && (kernel == tail || (capacity >= (band + 3) / 4 + sequences && capacity <= 520));
                for (int operand = 0; operand < (kernel == work ? 11 : 5); ++operand) {
                    RadOpdDesc descriptor{};
                    REQUIRE_EQ(kernel->opd_shape(params.data(), int(params.size()), operand, &descriptor),
                        valid_shape ? RAD_OK : RAD_E_UNSUPPORTED);
                    if (valid_shape && operand == 3) {
                        REQUIRE_EQ(descriptor.rank, 4u); REQUIRE_EQ(descriptor.shape[2], ring);
                    }
                    ++registration;
                }
            }
        }
    Queue compute(0), other(1);
    REQUIRE(compute.stream && other.stream && compute.stream != other.stream);
    Counts counts;
    for (int ring : {4, 5, 6, 7}) {
        for (int sequences : {1, 2, 8}) for (int rows : {1, 3, 16, 32, 63, 64, 2048})
            for (int capacity : {0, 1, 2}) for (int layout : {0, 1})
                fixture(work, tail, reference.ngram("qsa_work"), reference.ngram("qsa_tail_store"),
                    compute.stream, other.stream, ring, sequences, rows, capacity, layout, false, counts);
        for (int sequences : {1, 2, 8}) for (int layout : {0, 1})
            fixture(work, tail, reference.ngram("qsa_work"), reference.ngram("qsa_tail_store"),
                compute.stream, other.stream, ring, sequences, 63, 0, layout, true, counts);
        std::printf("QSA_RING_AQL_RING ring=%d fixtures=%llu work=%llu tail=%llu exact_bytes=%llu\n", ring,
            (unsigned long long)counts.fixtures, (unsigned long long)counts.work,
            (unsigned long long)counts.tail, (unsigned long long)counts.bytes); std::fflush(stdout);
    }
    REQUIRE(counts.rewinds > 0); REQUIRE(counts.wraps > 0); REQUIRE(counts.guards > 0);
    std::printf("PASS QSA_RING_AQL rings=4,5,6,7 fixtures=%llu work=%llu tail=%llu rejects=%llu device_guards=%llu "
        "registration=%llu exact_bytes=%llu independent_libref_bytes=%llu persistent_steps=%llu rewinds=%llu wraps=%llu "
        "queues=2 state_reupload=0 M=1,3,16,32,63,64,2048 band=64,2048 work17=1 q_len=1,2,3 tolerance=0\n",
        (unsigned long long)counts.fixtures, (unsigned long long)counts.work, (unsigned long long)counts.tail,
        (unsigned long long)counts.rejects, (unsigned long long)counts.guards, (unsigned long long)registration,
        (unsigned long long)counts.bytes, (unsigned long long)counts.reference_bytes,
        (unsigned long long)counts.persistent, (unsigned long long)counts.rewinds, (unsigned long long)counts.wraps);
}
