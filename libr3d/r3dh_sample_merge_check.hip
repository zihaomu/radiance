#include "r3dh_sample_merge.h"
#include "ref_ops.h"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
namespace {
size_t checks = 0, comparisons = 0, cases = 0, device_calls = 0, replays = 0;
void require(bool condition, const char* message) {
    ++checks; if (!condition) throw std::runtime_error(message);
}
void hipcheck(hipError_t error) { require(error == hipSuccess, hipGetErrorString(error)); }
uint32_t bits(float value) { uint32_t result; std::memcpy(&result,&value,4); return result; }
float value(uint32_t word) { float result; std::memcpy(&result,&word,4); return result; }
struct Buffer {
    std::vector<uint32_t> words;
    uint32_t* device = nullptr;
    explicit Buffer(size_t size) : words(size + 32,0xa5a5a5a5u) {}
    ~Buffer() { if (device) (void)hipFree(device); }
    uint32_t* data() { return words.data() + 16; }
    void allocate() { hipcheck(hipMalloc(&device,words.size() * 4)); }
    void push(hipStream_t stream) { hipcheck(hipMemcpyAsync(device,words.data(),words.size()*4,hipMemcpyHostToDevice,stream)); }
    void compare(hipStream_t stream, const char* label) {
        std::vector<uint32_t> actual(words.size());
        hipcheck(hipMemcpyAsync(actual.data(),device,actual.size()*4,hipMemcpyDeviceToHost,stream));
        hipcheck(hipStreamSynchronize(stream));
        for (size_t offset = 0; offset < actual.size(); ++offset) {
            ++comparisons;
            if (actual[offset] != words[offset]) {
                std::fprintf(stderr,"%s word=%zu expected=%08x actual=%08x\n",label,offset,words[offset],actual[offset]);
                require(false,label);
            }
        }
    }
};
RadTensor tensor(void* pointer, uint32_t dtype, int rows, int width, int pitch) {
    RadTensor result{}; result.data = pointer; result.dtype = dtype; result.rank = 2;
    result.shape[0] = rows; result.shape[1] = width; result.stride[0] = pitch; result.stride[1] = 1;
    return result;
}
struct Fixture {
    int rows, width, inputs, gp, ip, vp;
    Buffer gathered, params, indices, values;
    std::array<RadTensor,4> tensors;
    std::array<RadParam,2> geometry;
    RadArgs args{};
    Fixture(int batch, int count, int pairs, bool pitched) : rows(batch),width(count),inputs(pairs),
        gp(2*pairs+(pitched?6:0)),ip(count+(pitched?7:0)),vp(count+(pitched?11:0)),
        gathered(batch*gp),params(batch*24),indices(batch*ip),values(batch*vp) {
        tensors = {tensor(gathered.data(),RAD_U32,rows,2*inputs,gp),
            tensor(params.data(),RAD_U32,rows,24,24),tensor(indices.data(),RAD_I32,rows,width,ip),
            tensor(values.data(),RAD_F32,rows,width,vp)};
        geometry = {RAD_INT("M",rows),RAD_INT("n_cand",width)};
        args.t = tensors.data(); args.n_t = 4; args.p = geometry.data(); args.n_p = 2; args.world_size = 2;
    }
    std::array<Buffer*,4> buffers() { return {&gathered,&params,&indices,&values}; }
    void fill(int pattern, int repeat) {
        for (Buffer* buffer : buffers()) std::fill(buffer->words.begin(),buffer->words.end(),0xa5a5a5a5u);
        const uint32_t special[]{0x7f800000u,0xff800000u,0x7fc00001u,0x7f800001u,0xffc00123u,
            0u,0x80000000u,1u,0x80000001u,0x007fffffu,0x807fffffu,0x7f7fffffu,0xff7fffffu};
        for (int row = 0; row < rows; ++row) {
            RadSampleParams parameter{};
            switch (pattern) {
                case 7: case 8: case 9: case 10: parameter.top_k = 0; break;
                case 0: parameter.top_k = 1; break;
                case 1: parameter.top_k = std::min(width,17); break;
                case 2: parameter.top_k = 0; break;
                case 3: parameter.top_k = -7; break;
                case 4: parameter.top_k = width+5; break;
                case 5: parameter.top_k = 1; parameter.flags = RAD_SP_GREEDY; break;
                default: parameter.top_k = (row+repeat)%4 == 0 ? 1 : (row+repeat)%4 == 1 ? 17 : 0;
                    parameter.flags = (row+repeat)%4 == 3 ? RAD_SP_GREEDY : RAD_SP_LOGPROBS;
            }
            std::memcpy(params.data()+row*24,&parameter,sizeof parameter);
            for (int item = 0; item < inputs; ++item) {
                uint32_t id = uint32_t((item*37+repeat*19+row*101)%65537);
                uint32_t word = bits(float(int((item*73+repeat*117+row*7)%1021)-510)/16);
                if (pattern == 7) { id = UINT32_MAX; word = 0x7fc12345u; }
                else if (pattern == 8) { id = uint32_t((inputs-item+repeat)%53); word = bits(2.0f); }
                else if (pattern == 9) { id = uint32_t((item/4+repeat)%97); word = (item%4<2)?0u:0x80000000u; }
                else if (item%31 < 13) {
                    id = item%31 == 4 ? UINT32_MAX : item%31 == 5 ? 0x80000000u
                        : item%31 == 6 ? 0x7fffffffu : uint32_t((item/31+row+repeat)%71);
                    word = special[(item+repeat)%13];
                }
                if (pattern == 10) { id = uint32_t((item/3+repeat)%113); word = special[(item/3)%13]; }
                gathered.data()[row*gp+2*item] = id;
                gathered.data()[row*gp+2*item+1] = word;
            }
        }
        if (pattern == 11) {
            const int vocabulary = width * 2 + 37;
            Buffer logits(rows*vocabulary), local_ids(rows*width), local_values(rows*width), pairs(rows*width*2);
            for (int rank = 0; rank < 2; ++rank) {
                for (int row = 0; row < rows; ++row) for (int column = 0; column < vocabulary; ++column)
                    logits.data()[row*vocabulary+column] = bits(float(int((column*73+row*19+rank*53+repeat*97)%2017)-1008)/32);
                std::array<RadTensor,5> producer_tensors{
                    tensor(logits.data(),RAD_F32,rows,vocabulary,vocabulary),tensors[1],
                    tensor(local_ids.data(),RAD_I32,rows,width,width),tensor(local_values.data(),RAD_F32,rows,width,width),
                    tensor(pairs.data(),RAD_U32,rows,width*2,width*2)};
                RadParam producer_params[]{RAD_INT("M",rows),RAD_INT("n_vocab",vocabulary),RAD_INT("vocab_off",rank*vocabulary)};
                RadArgs producer{}; producer.t = producer_tensors.data(); producer.n_t = 5;
                producer.p = producer_params; producer.n_p = 3;
                require(ref_sample_topk(&producer,nullptr) == RAD_OK,"TP2 libref shard sample_topk");
                for (int row = 0; row < rows; ++row)
                    std::memcpy(gathered.data()+row*gp+rank*width*2,pairs.data()+row*width*2,size_t(width)*8);
            }
        }
    }
    void oracle() {
        struct Candidate { uint32_t id, word; };
        for (int row = 0; row < rows; ++row) {
            RadSampleParams parameter; std::memcpy(&parameter,params.data()+row*24,sizeof parameter);
            std::vector<Candidate> candidates;
            for (int item = 0; item < inputs; ++item) {
                const uint32_t id = gathered.data()[row*gp+2*item], word = gathered.data()[row*gp+2*item+1];
                if (int32_t(id) >= 0 && !std::isnan(value(word))) candidates.push_back({id,word});
            }
            std::stable_sort(candidates.begin(),candidates.end(),[](const Candidate& left,const Candidate& right) {
                return value(left.word) > value(right.word)
                    || (value(left.word) == value(right.word) && left.id < right.id);
            });
            candidates.erase(std::unique(candidates.begin(),candidates.end(),[](const Candidate& left,const Candidate& right) {
                return left.id == right.id && value(left.word) == value(right.word);
            }),candidates.end());
            const int keep = parameter.flags & RAD_SP_GREEDY ? 0
                : parameter.top_k > 0 ? std::min(width,parameter.top_k) : width;
            for (int column = 0; column < width; ++column) {
                const bool valid = column < keep && size_t(column) < candidates.size();
                indices.data()[row*ip+column] = valid ? candidates[column].id : UINT32_MAX;
                values.data()[row*vp+column] = valid ? candidates[column].word : 0xff800000u;
            }
        }
    }
    void reference() {
        const auto old_indices = indices.words, old_values = values.words;
        oracle(); const auto expected_indices = indices.words, expected_values = values.words;
        indices.words = old_indices; values.words = old_values;
        tensors[2].data = indices.data(); tensors[3].data = values.data();
        require(ref_sample_merge_topk(&args,nullptr) == RAD_OK,"libref execution");
        for (size_t index = 0; index < indices.words.size(); ++index) {
            ++comparisons; require(indices.words[index] == expected_indices[index],"independent CPU/libref ids");
        }
        for (size_t index = 0; index < values.words.size(); ++index) {
            ++comparisons; require(values.words[index] == expected_values[index],"independent CPU/libref value bits");
        }
    }
};
void execute(int rows,int width,int inputs,int pattern,bool pitched,bool gpu,hipStream_t stream) {
    Fixture fixture(rows,width,inputs,pitched);
    if (pattern == 12) fixture.geometry[0].ival = 8;
    if (pattern%3 == 1) fixture.tensors[0].dtype = RAD_I32;
    fixture.tensors[1].dtype = pattern%3 == 0 ? RAD_U32 : pattern%3 == 1 ? RAD_I32 : RAD_F32;
    auto remote = fixture.tensors;
    if (gpu) {
        int operand = 0;
        for (Buffer* buffer : fixture.buffers()) { buffer->allocate(); remote[operand++].data = buffer->device+16; }
    }
    RadArgs call = fixture.args; call.t = remote.data();
    hipGraph_t graph = nullptr; hipGraphExec_t executable = nullptr;
    if (gpu) {
        fixture.fill(pattern,0); for (Buffer* buffer : fixture.buffers()) buffer->push(stream);
        hipcheck(hipStreamSynchronize(stream));
        hipcheck(hipStreamBeginCapture(stream,hipStreamCaptureModeThreadLocal));
        require(r3dh_sample_merge_topk(&call,reinterpret_cast<RadStream>(stream)) == RAD_OK,"graph capture launch");
        hipcheck(hipStreamEndCapture(stream,&graph));
        hipcheck(hipGraphInstantiate(&executable,graph,nullptr,nullptr,0));
    }
    for (int repeat = 0; repeat < 2; ++repeat) {
        fixture.fill(pattern,repeat);
        if (gpu) for (Buffer* buffer : fixture.buffers()) buffer->push(stream);
        fixture.reference();
        if (gpu) {
            require(r3dh_sample_merge_topk(&call,reinterpret_cast<RadStream>(stream)) == RAD_OK,"native launch");
            ++device_calls;
            for (Buffer* buffer : fixture.buffers()) buffer->compare(stream,"native/input/output/padding/guards");
            std::vector<uint32_t> poison_indices(fixture.indices.words.size(),0xa5a5a5a5u);
            std::vector<uint32_t> poison_values(fixture.values.words.size(),0xa5a5a5a5u);
            hipcheck(hipMemcpyAsync(fixture.indices.device,poison_indices.data(),poison_indices.size()*4,hipMemcpyHostToDevice,stream));
            hipcheck(hipMemcpyAsync(fixture.values.device,poison_values.data(),poison_values.size()*4,hipMemcpyHostToDevice,stream));
            hipcheck(hipGraphLaunch(executable,stream)); ++replays;
            for (Buffer* buffer : fixture.buffers()) buffer->compare(stream,"replay/input/output/padding/guards");
        }
    }
    if (gpu) { hipcheck(hipGraphExecDestroy(executable)); hipcheck(hipGraphDestroy(graph)); }
    ++cases;
    std::printf("PASS M=%d n_cand=%d gathered_pairs=%d pattern=%d pitched=%d repeats=2%s\n",
        rows,width,inputs,pattern,pitched,gpu?" direct+graph":" independent_cpu+libref");
}
void metadata() {
    int count = 0; const RadKernelInfo* kernel = r3dh_sample_merge_kernels(&count);
    require(count == 1 && r3dh_sample_merge_kernel_count() == 1 && kernel == r3dh_sample_merge_kernel_at(0),"kernel count");
    require(!r3dh_sample_merge_kernel_at(-1) && !r3dh_sample_merge_kernel_at(1),"kernel bounds");
    require(kernel->domain == RAD_DOMAIN_DEVICE && !std::strcmp(kernel->op,"sample_merge_topk")
        && kernel->launch == r3dh_sample_merge_topk && !kernel->scratch,"canonical DEVICE only row");
    const RadOpSchema* schema = r3dh_sample_merge_schemas(&count);
    require(count == 1 && r3dh_sample_merge_schema_count() == 1 && schema == r3dh_sample_merge_schema_at(0),"schema count");
    require(!r3dh_sample_merge_schema_at(-1) && !r3dh_sample_merge_schema_at(1),"schema bounds");
    require(schema->n_params == 2 && schema->n_operands == 4,"canonical schema size");
    require(!std::strcmp(schema->params[0].key,"M") && !std::strcmp(schema->params[1].key,"n_cand"),"schema params");
    const char* names[]{"gathered","params","cand_idx","cand_val"};
    for (int operand = 0; operand < 4; ++operand) require(!std::strcmp(schema->operands[operand].name,names[operand]),"schema operands");
    RadParam geometry[]{RAD_INT("M",1),RAD_INT("n_cand",1024),RAD_INT("world_size",2)};
    RadOpdDesc description{};
    require(r3dh_sample_merge_shape(geometry,2,0,&description) == RAD_E_UNSUPPORTED,"no guessed gather width");
    for (int operand = 0; operand < 4; ++operand) {
        require(r3dh_sample_merge_shape(geometry,3,operand,&description) == RAD_OK,"shape");
        require(description.rank == 2 && description.shape[0] == 1
            && description.shape[1] == (operand == 0 ? 4096 : operand == 1 ? 24 : 1024),"shape geometry");
    }
    require(r3dh_sample_merge_shape(nullptr,1,0,&description) == RAD_E_INVAL,"shape invalid params");
    require(r3dh_sample_merge_shape(geometry,3,4,&description) == RAD_E_SHAPE,"shape invalid operand");
    require(r3dh_sample_merge_topk(nullptr,nullptr) == RAD_E_INVAL,"null args");
    Fixture fixture(1,1024,2048,false);
    auto reject = [&](int expected,const char* label) { require(r3dh_sample_merge_topk(&fixture.args,nullptr) == expected,label); };
    for (int operand = 0; operand < 4; ++operand) {
        const RadTensor saved = fixture.tensors[operand];
        fixture.tensors[operand].data = nullptr; reject(RAD_E_INVAL,"null data"); fixture.tensors[operand] = saved;
        fixture.tensors[operand].dtype = RAD_BF16; reject(RAD_E_DTYPE,"dtype"); fixture.tensors[operand] = saved;
        fixture.tensors[operand].rank = 1; reject(RAD_E_SHAPE,"rank"); fixture.tensors[operand] = saved;
        fixture.tensors[operand].stride[1] = 2; reject(RAD_E_STRIDE,"inner stride"); fixture.tensors[operand] = saved;
        fixture.tensors[operand].stride[0] = 0; reject(RAD_E_STRIDE,"row stride"); fixture.tensors[operand] = saved;
    }
    for (int operand = 2; operand < 4; ++operand) {
        const RadTensor saved = fixture.tensors[operand];
        fixture.tensors[operand].data = fixture.tensors[0].data; reject(RAD_E_INVAL,"output alias");
        fixture.tensors[operand] = saved;
    }
    for (int axis = 0; axis < 2; ++axis) for (long long invalid : {0ll,-1ll,axis?2049ll:9ll}) {
        const auto saved = fixture.geometry[axis]; fixture.geometry[axis].ival = invalid;
        reject(RAD_E_SHAPE,"geometry bounds"); fixture.geometry[axis] = saved;
    }
    fixture.tensors[1].stride[0] = 25; reject(RAD_E_STRIDE,"params must be packed"); fixture.tensors[1].stride[0] = 24;
    fixture.tensors[0].shape[1] = 8194; reject(RAD_E_SHAPE,"too many pairs"); fixture.tensors[0].shape[1] = 4096;
    fixture.tensors[0].shape[1] = 4095; reject(RAD_E_SHAPE,"odd gathered words"); fixture.tensors[0].shape[1] = 4096;
    fixture.args.n_t = 3; reject(RAD_E_INVAL,"operand count"); fixture.args.n_t = 4;
    fixture.args.n_p = -1; reject(RAD_E_INVAL,"negative param count"); fixture.args.n_p = 2;
    const void* saved_params = fixture.args.p; fixture.args.p = nullptr; reject(RAD_E_INVAL,"null params");
    fixture.args.p = static_cast<const RadParam*>(saved_params);
    for (int operand = 0; operand < 4; ++operand) {
        void* saved = fixture.tensors[operand].data;
        fixture.tensors[operand].data = static_cast<uint8_t*>(saved)+1;
        reject(RAD_E_INVAL,"unaligned pointer"); fixture.tensors[operand].data = saved;
    }
    fixture.tensors[3].data = fixture.tensors[2].data; reject(RAD_E_INVAL,"output/output alias");
    fixture.tensors[3].data = fixture.values.data();
    RadParam mismatch[]{RAD_INT("M",1),RAD_INT("n_cand",1024),RAD_INT("world_size",4)};
    fixture.args.p = mismatch; fixture.args.n_p = 3; reject(RAD_E_SHAPE,"world geometry mismatch");
}
}
int main(int argc,char** argv) {
    try {
        require(argc == 2 && (!std::strcmp(argv[1],"--host") || !std::strcmp(argv[1],"--gpu")),"usage: --host|--gpu");
        const bool gpu = !std::strcmp(argv[1],"--gpu"); metadata(); hipStream_t stream = nullptr;
        if (gpu) {
            int count = 0; hipcheck(hipGetDeviceCount(&count)); require(count == 1,"one isolated GPU");
            hipDeviceProp_t properties{}; hipcheck(hipGetDeviceProperties(&properties,0));
            require(!std::strncmp(properties.gcnArchName,"gfx1100",7),"gfx1100 only");
            char pci[32]{}; hipcheck(hipDeviceGetPCIBusId(pci,sizeof pci,0));
            std::printf("DEVICE %s arch=%s pci=%s\n",properties.name,properties.gcnArchName,pci);
            hipcheck(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));
        }
        for (int rows : {1,2,4,8}) for (int width : {256,1024,2048})
            for (int pattern = 0; pattern <= 10; ++pattern) execute(rows,width,width*2,pattern,pattern%2,gpu,stream);
        for (int inputs : {1,255,256,257,511,512,513,1023,1024,1025,2047,2048,2049,4095,4096})
            execute(1,1024,inputs,10,true,gpu,stream);
        for (int width : {1,255,257,1023,1025,2047}) execute(2,width,width*2,6,true,gpu,stream);
        for (int rows : {1,2,4,8}) execute(rows,1024,2048,11,true,gpu,stream);
        execute(1,1024,2048,12,true,gpu,stream);
        if (gpu) hipcheck(hipStreamDestroy(stream));
        std::printf("SAMPLE_MERGE_PASS mode=%s cases=%zu checks=%zu comparisons=%zu device_calls=%zu graph_replays=%zu strict_bits guards inputs_immutable\n",
            gpu?"gpu":"host",cases,checks,comparisons,device_calls,replays); return 0;
    } catch (const std::exception& failure) { std::fprintf(stderr,"SAMPLE_MERGE_FAIL %s\n",failure.what()); return 1; }
}
