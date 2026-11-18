#include "r3dh_allgather_private.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#ifndef R3DH_ALLGATHER_HOST_ONLY
#include <hip/hip_runtime.h>
#include <barrier>
#include <thread>
#else
extern "C" int r3dh_allgather_init(const RadParam*,int,int,int,void**) { return RAD_E_UNSUPPORTED; }
extern "C" void r3dh_allgather_fini(void*) {}
extern "C" int r3dh_allgather_launch(const RadArgs*,RadStream) { return RAD_E_UNSUPPORTED; }
#endif
namespace {
std::atomic<uint64_t> checks{0}, comparisons{0}, calls{0}, replays{0}, cases{0};
void require(bool condition,const char* message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}
RadParam integer(const char* key,int64_t value) {
    RadParam result{}; result.key = key; result.kind = RAD_P_INT; result.ival = value; return result;
}
RadParam text(const char* key,const char* value) {
    RadParam result{}; result.key = key; result.kind = RAD_P_STR; result.sval = value; return result;
}
std::array<RadParam,4> parameters(int64_t maximum,int64_t row,const char* dtype) {
    return {integer("world_size",2),integer("numel",maximum),text("dtype",dtype),integer("row",row)};
}
RadTensor tensor(void* pointer,unsigned dtype,int64_t rows,int64_t width,int64_t pitch) {
    RadTensor result{}; result.data = pointer; result.dtype = dtype; result.rank = 2;
    result.shape[0] = rows; result.shape[1] = width; result.stride[0] = pitch; result.stride[1] = 1;
    return result;
}
void host() {
    uint64_t augmented_cases = 0, init_rejections = 0;
    for (const int64_t numel : {2048,44}) {
        for (const bool ranged : {false,true}) {
            const RadParam frozen[]{RAD_INT("world_size",2),
                ranged ? RAD_RANGE("numel",numel,numel*7) : RAD_INT("numel",numel),
                RAD_STR("dtype","f32"),RAD_INT("row",numel)};
            R3dhAllgatherGeometry geometry;
            require(r3dh_allgather_geometry(frozen,4,&geometry) == RAD_OK,"exact frozen geometry");
            require(r3dh_allgather_init_geometry(frozen,4,&geometry) == RAD_OK,"exact frozen init geometry");
            require(geometry.low == numel && geometry.high == numel*(ranged ? 7 : 1) &&
                geometry.row == numel && geometry.dtype == RAD_F32 && geometry.bytes == 4,
                "exact frozen geometry fields");
            for (const int64_t group : {0,37}) {
                std::array<RadParam,5> augmented{frozen[0],frozen[1],frozen[2],frozen[3],RAD_INT("tp_group",group)};
                std::sort(augmented.begin(),augmented.end(),
                    [](const RadParam& left,const RadParam& right) { return std::strcmp(left.key,right.key) < 0; });
                do {
                    ++augmented_cases;
                    require(r3dh_allgather_geometry(augmented.data(),5,&geometry) == RAD_E_INVAL,
                        "launch geometry rejects init metadata");
                    require(r3dh_allgather_init_geometry(augmented.data(),5,&geometry) == RAD_OK,
                        "registry appended tp_group init");
                    require(geometry.low == numel && geometry.high == numel*(ranged ? 7 : 1) &&
                        geometry.row == numel && geometry.dtype == RAD_F32 && geometry.bytes == 4,
                        "registry init preserves frozen fields");
                } while (std::next_permutation(augmented.begin(),augmented.end(),
                    [](const RadParam& left,const RadParam& right) { return std::strcmp(left.key,right.key) < 0; }));
            }
            for (int mutation = 0; mutation < 13; ++mutation) {
                std::array<RadParam,5> invalid{frozen[0],frozen[1],frozen[2],frozen[3],RAD_INT("tp_group",37)};
                switch (mutation) {
                    case 0: invalid[4] = RAD_STR("tp_group","37"); break;
                    case 1: invalid[4] = RAD_RANGE("tp_group",0,37); break;
                    case 2: invalid[4].kind = RAD_P_F64; break;
                    case 3: invalid[4].ival = -1; break;
                    case 4: invalid[4].key = "unknown"; break;
                    case 5: invalid[4].key = RAD_FACT_P2P; break;
                    case 6: invalid[0] = invalid[4]; break;
                    case 7: invalid[4].key = nullptr; break;
                    case 8: invalid[1] = RAD_INT("numel",0); break;
                    case 9: invalid[2] = RAD_STR("dtype","i32"); break;
                    case 10: invalid[3].ival = -1; break;
                    case 11: invalid[3] = frozen[1]; break;
                    case 12: invalid[0].ival = 3; break;
                }
                const int expected[]{RAD_E_INVAL,RAD_E_INVAL,RAD_E_INVAL,RAD_E_INVAL,
                    RAD_E_INVAL,RAD_E_INVAL,RAD_E_INVAL,RAD_E_INVAL,RAD_E_SHAPE,
                    RAD_E_DTYPE,RAD_E_SHAPE,RAD_E_INVAL,RAD_E_SHAPE};
                require(r3dh_allgather_init_geometry(invalid.data(),5,&geometry) == expected[mutation],
                    "init metadata preserves exact rejection status");
                ++init_rejections;
            }
        }
    }
    require(augmented_cases == 960 && init_rejections == 52,"frozen regression coverage");
    std::printf("ALLGATHER_FROZEN_HOST_PASS frozen_cases=4 augmented_permutations=%llu init_rejects=%llu\n",
        (unsigned long long)augmented_cases,(unsigned long long)init_rejections);
    alignas(16) unsigned char storage[8192]{};
    for (const char* dtype : {"f32","fp32","bf16","f16","fp16"}) {
        const unsigned type = !std::strcmp(dtype,"bf16") ? RAD_BF16 :
            (!std::strcmp(dtype,"f16") || !std::strcmp(dtype,"fp16")) ? RAD_F16 : RAD_F32;
        const int row = type == RAD_F32 ? 44 : 88;
        auto params = parameters(row * 7,row,dtype);
        std::array<RadTensor,2> tensors{tensor(storage,type,3,row,row + 8),
            tensor(storage + 4096,type,3,row * 2,row * 2 + 16)};
        RadArgs args{}; args.t = tensors.data(); args.n_t = 2; args.p = params.data(); args.n_p = 4;
        args.world_size = 2;
        require(r3dh_allgather_validate(&args) == RAD_OK,"valid pitched capacity");
        params[1].kind = RAD_P_RANGE; params[1].ival = row; params[1].ihi = row * 7;
        require(r3dh_allgather_validate(&args) == RAD_OK,"valid range");
        for (int operand = 0; operand < 2; ++operand) {
            RadOpdDesc shape{};
            require(r3dh_allgather_shape(params.data(),4,operand,&shape) == RAD_OK,"shape");
            require(shape.dtype == type && shape.rank == 1 && shape.shape[0] == row * 7 * (operand+1),"shape extent");
        }
        for (int mutation = 0; mutation < 25; ++mutation) {
            auto saved_params = params; auto saved_tensors = tensors; RadArgs saved_args = args;
            switch (mutation) {
                case 0: args.n_t = 1; break;
                case 1: args.t = nullptr; break;
                case 2: args.world_size = 3; break;
                case 3: args.rank = 2; break;
                case 4: params[0].ival = 3; break;
                case 5: params[2].sval = "i32"; break;
                case 6: params[1].ival = 0; break;
                case 7: params[1].ihi = 1ll << 60; break;
                case 8: params[3].ival = -1; break;
                case 9: params[3].ival = row-1; break;
                case 10: params[3].kind = RAD_P_STR; break;
                case 11: params[2].sval = nullptr; break;
                case 12: params[0].key = nullptr; break;
                case 13: params[3].key = "numel"; break;
                case 14: tensors[0].data = nullptr; break;
                case 15: tensors[0].data = storage + 1; break;
                case 16: tensors[1].data = storage; break;
                case 17: tensors[1].data = storage + 16; break;
                case 18: tensors[0].stride[1] = 2; break;
                case 19: tensors[0].stride[0] = row-1; break;
                case 20: tensors[1].stride[0] += 1; break;
                case 21: tensors[1].shape[0] = 2; break;
                case 22: tensors[1].dtype = RAD_I32; break;
                case 23: tensors[0].rank = RAD_MAX_RANK+1; break;
                case 24: tensors[0].shape[0] = INT64_MAX; break;
            }
            const int expected[]{RAD_E_INVAL,RAD_E_INVAL,RAD_E_SHAPE,RAD_E_SHAPE,RAD_E_SHAPE,
                RAD_E_DTYPE,RAD_E_SHAPE,RAD_E_SHAPE,RAD_E_SHAPE,RAD_E_SHAPE,RAD_E_SHAPE,
                RAD_E_DTYPE,RAD_E_INVAL,RAD_E_INVAL,RAD_E_SHAPE,RAD_E_ALIGN,RAD_E_INVAL,
                RAD_E_INVAL,RAD_E_STRIDE,RAD_E_STRIDE,RAD_E_STRIDE,RAD_E_SHAPE,RAD_E_DTYPE,
                RAD_E_SHAPE,RAD_E_SHAPE};
            require(r3dh_allgather_validate(&args) == expected[mutation],"exact host rejection status");
            params = saved_params; tensors = saved_tensors; args = saved_args;
        }
        for (int rows = 1; rows <= 7; ++rows) {
            tensors[0].shape[0] = tensors[1].shape[0] = rows;
            require(r3dh_allgather_validate(&args) == RAD_OK,"dynamic rows");
        }
    }
    alignas(16) unsigned char raw_storage[229376]{};
    uint64_t raw_accepts = 0, raw_rejects = 0;
    for (const char* wire : {"f32","fp32"}) for (int rows : {1,3,8}) {
        auto params = parameters(2048*8,2048,wire);
        std::array<RadTensor,2> tensors{tensor(raw_storage+32,RAD_U32,rows,2048,2052),
            tensor(raw_storage+73760,RAD_U32,rows,4096,4104)};
        RadArgs args{}; args.t=tensors.data(); args.n_t=2; args.p=params.data(); args.n_p=4;
        args.world_size=2;
        for (unsigned input : {RAD_F32,RAD_U32,RAD_I32,RAD_U16,RAD_I8,RAD_F16,RAD_BF16,RAD_U8})
            for (unsigned output : {RAD_F32,RAD_U32,RAD_I32,RAD_U16,RAD_I8,RAD_F16,RAD_BF16,RAD_U8}) {
                tensors[0].dtype=input; tensors[1].dtype=output;
                const bool accepted=input==output && (input==RAD_F32 || input==RAD_U32 || input==RAD_I32);
                require(r3dh_allgather_validate(&args)==(accepted ? RAD_OK : RAD_E_DTYPE),"raw32 matching dtype matrix");
                if (accepted) ++raw_accepts; else ++raw_rejects;
            }
        tensors[0].dtype=tensors[1].dtype=RAD_U32;
        tensors[1].shape[1]=4095;
        require(r3dh_allgather_validate(&args)==RAD_E_SHAPE,"raw32 invalid output shape"); ++raw_rejects;
        tensors[1].shape[1]=4096;
        tensors[0].stride[0]=2051;
        require(r3dh_allgather_validate(&args)==RAD_E_STRIDE,"raw32 unaligned pitch"); ++raw_rejects;
        tensors[0].stride[0]=2052;
        for (const char* narrow : {"bf16","f16","fp16"}) {
            params[2].sval=narrow;
            require(r3dh_allgather_validate(&args)==RAD_E_DTYPE,"raw32 rejected under 16-bit wire"); ++raw_rejects;
        }
    }
    std::printf("ALLGATHER_RAW32_HOST_PASS accepts=%llu rejects=%llu mixed_types_fail_closed=1\n",
        (unsigned long long)raw_accepts,(unsigned long long)raw_rejects);
    int count = 0;
    require(r3dh_allgather_kernel_count() == 1 && r3dh_allgather_schema_count() == 1,"count accessors");
    const auto* kernel = r3dh_allgather_kernels(&count);
    require(count == 1 && kernel == r3dh_allgather_kernel_at(0),"kernel accessors");
    require(!r3dh_allgather_kernel_at(-1) && !r3dh_allgather_kernel_at(1),"kernel bounds");
    const auto* schema = r3dh_allgather_schemas(&count);
    require(count == 1 && schema == r3dh_allgather_schema_at(0),"schema accessors");
    require(schema->n_params == 4 && schema->n_operands == 2,"canonical schema");
    require(!r3dh_allgather_schema_at(-1) && !r3dh_allgather_schema_at(1),"schema bounds");
    require(r3dh_allgather_validate(nullptr) == RAD_E_INVAL,"null args");
    std::printf("ALLGATHER_HOST_PASS checks=%llu\n",(unsigned long long)checks.load());
}
#ifndef R3DH_ALLGATHER_HOST_ONLY
void hipcheck(hipError_t error) { require(error == hipSuccess,hipGetErrorString(error)); }
__host__ __device__ unsigned char pattern(uint64_t index,unsigned generation,int rank) {
    const uint32_t special[]{0xffffffffu,0x7f800001u,0xffc00123u,0x80000000u,1u,0x7f800000u,0u,0x4b800001u};
    uint32_t word = index / 4 < 8 ? special[index/4] : uint32_t(index/4) * 0x9e3779b9u;
    word ^= generation * 0x85ebca6bu + unsigned(rank) * 0x6c8e9cf5u;
    return (word >> ((index % 4) * 8)) & 255;
}
__global__ void advance(unsigned* generation) { if (!threadIdx.x) ++*generation; }
__global__ void fill(unsigned char* input,unsigned* generation,int row_bytes,int rows,int pitch,int rank) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < row_bytes * rows;
         index += blockDim.x * gridDim.x)
        input[(index / row_bytes) * pitch + index % row_bytes] = pattern(index,*generation,rank);
}
__global__ void delay(unsigned long long ticks) {
    const auto start = wall_clock64();
    while (wall_clock64() - start < ticks) __builtin_amdgcn_s_sleep(1);
}
struct Allocation {
    unsigned char* pointer = nullptr;
    explicit Allocation(size_t bytes) { hipcheck(hipMalloc(&pointer,bytes)); }
    ~Allocation() { if (pointer) (void)hipFree(pointer); }
};
struct Context {
    void* instance = nullptr;
    hipStream_t stream = nullptr;
    hipGraph_t graph = nullptr;
    hipGraphExec_t executable = nullptr;
    ~Context() {
        if (stream) (void)hipStreamSynchronize(stream);
        if (executable) (void)hipGraphExecDestroy(executable);
        if (graph) (void)hipGraphDestroy(graph);
        r3dh_allgather_fini(instance);
        if (stream) (void)hipStreamDestroy(stream);
    }
};
void gpu_case(int row,int rows,const char* dtype,bool pitched,bool graph,bool wrap,bool flat,int frozen_mode=0,unsigned raw_type=RAD_F32) {
    std::barrier barrier(2);
    std::exception_ptr failures[2];
    auto rank_main = [&](int rank) {
        try {
            hipcheck(hipSetDevice(rank));
            const int bytes = !std::strcmp(dtype,"f32") ? 4 : 2;
            const unsigned type = bytes == 4 ? raw_type : !std::strcmp(dtype,"bf16") ? RAD_BF16 : RAD_F16;
            const int row_bytes = row * bytes;
            const int input_pitch = row_bytes + (pitched ? 16*(rank+1) : 0);
            const int output_pitch = row_bytes * 2 + (pitched ? 32*(rank+1) : 0);
            auto params = parameters(row*std::max(rows,7),flat ? 0 : row,dtype);
            if (frozen_mode) {
                params = {RAD_INT("world_size",2),frozen_mode % 2 ? RAD_INT("numel",row*rows) :
                    RAD_RANGE("numel",row,row*std::max(rows,7)),RAD_STR("dtype",dtype),RAD_INT("row",row)};
            }
            std::vector<RadParam> init_params(params.begin(),params.end());
            if (frozen_mode >= 3) init_params.push_back(RAD_INT("tp_group",37));
            Context context;
            hipcheck(hipStreamCreateWithFlags(&context.stream,hipStreamNonBlocking));
            require(r3dh_allgather_kernel_at(0)->init(init_params.data(),int(init_params.size()),rank,2,
                &context.instance) == RAD_OK,"native registry init");
            Allocation input(size_t(input_pitch)*rows + 64), output(size_t(output_pitch)*rows + 64), generation(16);
            hipcheck(hipMemsetAsync(input.pointer,0xa5,size_t(input_pitch)*rows+64,context.stream));
            hipcheck(hipMemsetAsync(output.pointer,0xa5,size_t(output_pitch)*rows+64,context.stream));
            hipcheck(hipMemsetAsync(generation.pointer,0,16,context.stream));
            hipcheck(hipStreamSynchronize(context.stream));
            if (wrap) require(r3dh_allgather_seed(context.instance,0xfffffffcu) == RAD_OK,"seed wrap");
            barrier.arrive_and_wait();
            std::array<RadTensor,2> tensors{tensor(input.pointer+32,type,rows,row,input_pitch/bytes),
                tensor(output.pointer+32,type,rows,row*2,output_pitch/bytes)};
            RadArgs args{}; args.t = tensors.data(); args.n_t = 2; args.p = params.data(); args.n_p = 4;
            args.rank = rank; args.world_size = 2; args.instance = context.instance;
            auto launch = [&] {
                if (!frozen_mode) {
                    advance<<<1,1,0,context.stream>>>((unsigned*)generation.pointer);
                    fill<<<16,256,0,context.stream>>>(input.pointer+32,(unsigned*)generation.pointer,row_bytes,rows,input_pitch,rank);
                }
                if (rank == 1) delay<<<1,1,0,context.stream>>>(100000ull);
                require(r3dh_allgather_launch(&args,(RadStream)context.stream) == RAD_OK,"native launch");
            };
            if (graph) {
                hipcheck(hipStreamBeginCapture(context.stream,hipStreamCaptureModeThreadLocal));
                launch();
                hipcheck(hipStreamEndCapture(context.stream,&context.graph));
                hipcheck(hipGraphInstantiate(&context.executable,context.graph,nullptr,nullptr,0));
            }
            const int repeats = graph ? 300 : 5;
            std::vector<unsigned char> actual_output(size_t(output_pitch)*rows+64),actual_input(size_t(input_pitch)*rows+64);
            for (int repeat = 1; repeat <= repeats; ++repeat) {
                if (frozen_mode) {
                    std::fill(actual_input.begin(),actual_input.end(),0xa5);
                    for (int input_row = 0; input_row < rows; ++input_row)
                        for (int column = 0; column < row_bytes; ++column)
                            actual_input[32 + input_row * input_pitch + column] =
                                pattern(uint64_t(input_row)*row_bytes+column,repeat,rank);
                    hipcheck(hipMemcpyAsync(input.pointer,actual_input.data(),actual_input.size(),
                        hipMemcpyHostToDevice,context.stream));
                }
                if (graph) { hipcheck(hipGraphLaunch(context.executable,context.stream)); ++replays; }
                else launch();
                ++calls;
                hipcheck(hipMemcpyAsync(actual_output.data(),output.pointer,actual_output.size(),hipMemcpyDeviceToHost,context.stream));
                hipcheck(hipMemcpyAsync(actual_input.data(),input.pointer,actual_input.size(),hipMemcpyDeviceToHost,context.stream));
                hipcheck(hipStreamSynchronize(context.stream));
                unsigned error = 99;
                require(r3dh_allgather_status(context.instance,&error) == RAD_OK && error == 0,"protocol status");
                for (size_t index = 0; index < actual_output.size(); ++index) {
                    unsigned char expected = 0xa5;
                    if (index >= 32 && index < size_t(output_pitch)*rows+32) {
                        const size_t offset = index-32, column = offset % output_pitch, output_row = offset / output_pitch;
                        if (column < size_t(row_bytes)*2) {
                            const int source_rank = column / row_bytes;
                            expected = pattern(output_row * row_bytes + column % row_bytes,repeat,source_rank);
                        }
                    }
                    ++comparisons;
                    if (actual_output[index] != expected) {
                        std::fprintf(stderr,"mismatch rank=%d row=%d rows=%d dtype=%s repeat=%d byte=%zu actual=%02x expected=%02x\n",
                            rank,row,rows,dtype,repeat,index,actual_output[index],expected);
                        require(false,"CPU byte oracle/output guard");
                    }
                }
                for (size_t index = 0; index < actual_input.size(); ++index) {
                    unsigned char expected = 0xa5;
                    if (index >= 32 && index < size_t(input_pitch)*rows+32) {
                        const size_t offset = index-32;
                        if (offset % input_pitch < size_t(row_bytes))
                            expected = pattern((offset/input_pitch)*row_bytes + offset%input_pitch,repeat,rank);
                    }
                    ++comparisons;
                    require(actual_input[index] == expected,"input unchanged/guard");
                }
                barrier.arrive_and_wait();
            }
            ++cases;
        } catch (...) { failures[rank] = std::current_exception(); barrier.arrive_and_drop(); }
    };
    std::thread first(rank_main,0),second(rank_main,1);
    first.join(); second.join();
    for (const auto& failure : failures) if (failure) std::rethrow_exception(failure);
    std::printf("ALLGATHER_CASE_PASS row=%d rows=%d dtype=%s pitched=%d graph=%d wrap=%d flat=%d\n",row,rows,dtype,pitched,graph,wrap,flat);
}
#include "r3dh_allgather_check_queued.h"
void timeout_case() {
    void* instances[2]{};
    int status[2]{};
    auto params = parameters(44,44,"f32");
    std::thread first([&] { hipcheck(hipSetDevice(0)); status[0] = r3dh_allgather_init(params.data(),4,0,2,&instances[0]); });
    std::thread second([&] { hipcheck(hipSetDevice(1)); status[1] = r3dh_allgather_init(params.data(),4,1,2,&instances[1]); });
    first.join(); second.join();
    require(status[0] == RAD_OK && status[1] == RAD_OK,"timeout fixture init");
    hipcheck(hipSetDevice(0));
    r3dh_allgather_fini(instances[1]);
    Allocation input(176),output(352);
    std::array<RadTensor,2> tensors{tensor(input.pointer,RAD_F32,1,44,44),tensor(output.pointer,RAD_F32,1,88,88)};
    RadArgs args{}; args.t=tensors.data(); args.n_t=2; args.p=params.data(); args.n_p=4;
    args.world_size=2; args.instance=instances[0];
    require(r3dh_allgather_launch(&args,nullptr) == RAD_OK,"timeout enqueue");
    hipcheck(hipDeviceSynchronize());
    unsigned error=0;
    require(r3dh_allgather_status(instances[0],&error)==RAD_OK && error==1,"bounded missing-peer timeout");
    r3dh_allgather_fini(instances[0]);
    std::printf("ALLGATHER_TIMEOUT_LIFETIME_PASS peer_finalized_before_launch=1 bounded_wait=1\n");
}
void gpu(bool frozen_only=false) {
    int devices=0; hipcheck(hipGetDeviceCount(&devices)); require(devices==2,"exactly two visible GPUs");
    for (int device=0;device<2;++device) {
        hipDeviceProp_t prop{}; hipcheck(hipGetDeviceProperties(&prop,device));
        require(!std::strncmp(prop.gcnArchName,"gfx1100",7),"gfx1100 only");
        char bus[32]{}; hipcheck(hipDeviceGetPCIBusId(bus,sizeof(bus),device));
        std::printf("ALLGATHER_DEVICE ordinal=%d pci=%s arch=%s\n",device,bus,prop.gcnArchName);
    }
    for (int row : {2048,44}) {
        for (int mode = 1; mode <= 4; ++mode) {
            gpu_case(row,1,"f32",false,false,false,false,mode);
            std::printf("ALLGATHER_FROZEN_DEVICE_PASS numel=%d row=%d kind=%s init_params=%d ranks=2 staged_h2d=1\n",
                row,row,mode%2 ? "int" : "range",mode>=3 ? 5 : 4);
        }
    }
    std::printf("ALLGATHER_FROZEN_GPU_PASS rank_cases=%llu device_calls=%llu byte_comparisons=%llu\n",
        (unsigned long long)cases.load(),(unsigned long long)calls.load(),(unsigned long long)comparisons.load());
    if (frozen_only) return;
    for (const char* dtype : {"f32","bf16","f16"}) {
        for (int row : {!std::strcmp(dtype,"f32") ? 44 : 88,2048,64})
            for (int rows : {1,3,17}) gpu_case(row,rows,dtype,rows!=1,false,false,false);
        gpu_case(2048,3,dtype,false,true,true,false);
        gpu_case(2048,1,dtype,false,false,false,true);
    }
    gpu_case(16,2048,"f32",true,false,false,false);
    gpu_case(64,2048,"f32",true,false,false,false);
    queued_lifetime_case();
    timeout_case();
    std::printf("ALLGATHER_GPU_PASS rank_cases=%llu device_calls=%llu graph_replays=%llu byte_comparisons=%llu checks=%llu\n",
        (unsigned long long)cases.load(),(unsigned long long)calls.load(),(unsigned long long)replays.load(),
        (unsigned long long)comparisons.load(),(unsigned long long)checks.load());
    const auto old_cases=cases.load(), old_calls=calls.load(), old_replays=replays.load(), old_bytes=comparisons.load();
    for (unsigned dtype : {RAD_U32,RAD_I32}) {
        for (int rows : {1,3,8}) gpu_case(2048,rows,"f32",rows!=1,false,false,false,0,dtype);
        gpu_case(2048,3,"f32",true,true,true,false,0,dtype);
    }
    std::printf("ALLGATHER_RAW32_GPU_PASS rank_cases=%llu device_calls=%llu graph_replays=%llu byte_comparisons=%llu U32=1 I32=1\n",
        (unsigned long long)(cases.load()-old_cases),(unsigned long long)(calls.load()-old_calls),
        (unsigned long long)(replays.load()-old_replays),(unsigned long long)(comparisons.load()-old_bytes));
}
#endif
}
int main(int argc,char** argv) {
    try {
        host();
#ifndef R3DH_ALLGATHER_HOST_ONLY
        if (argc==2 && !std::strcmp(argv[1],"--gpu")) gpu();
        else if (argc==2 && !std::strcmp(argv[1],"--frozen")) gpu(true);
        else if (argc!=2 || std::strcmp(argv[1],"--host")) return 2;
#endif
        return 0;
    } catch (const std::exception& error) { std::fprintf(stderr,"ALLGATHER_FAIL %s\n",error.what()); return 1; }
}
