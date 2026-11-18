#define main r3dh_attn_fused_unused_base_main
#include "r3dh_attn_qsa_check.hip"
#undef main
#include "r3dh_attn_fused.h"
#include "r3dh_sparse_qsa.h"
#include "r3dh_qwen_ops.h"
extern "C" int ref_had_quant_act_i8g(const RadArgs*,RadStream);
extern "C" int ref_qsa_select(const RadArgs*,RadStream);
namespace {
size_t fused_checks=0;
int fused_cases=0;
std::vector<Blob*> fused_resident;
void insist(bool condition,const char* message) {++fused_checks; require(condition,message);}
std::vector<RadParam> fused_params(int rows,bool paged) {
    if (paged) return {RAD_INT("q_len",rows),RAD_INT("head_dim",256),RAD_INT("gqa",12),RAD_INT("n_head",12),
        RAD_INT("block_size",4),RAD_INT("causal",1),RAD_INT("window",0),RAD_STR("q_dtype","bf16"),
        RAD_STR("kv_dtype","i8"),RAD_INT("max_ctx",4096),RAD_INT("max_seqs",256),RAD_INT("group",128),RAD_STR("act","sigmoid")};
    return {RAD_INT("M",rows),RAD_INT("head_dim",256),RAD_INT("n_head",12),RAD_INT("n_head_kv",1),
        RAD_INT("rot",64),RAD_INT("cs_f32",1),RAD_STR("q_dtype","bf16"),RAD_INT("block_size",4),
        RAD_STR("kv_dtype","i8"),RAD_F64("eps",1e-6),RAD_F64("wadd",1)};
}
void row_copy(const RadTensor& source,const RadTensor& output,int width) {
    if (!output.data) return;
    for (int64_t row=0;row<source.shape[0];++row)
        std::memcpy(static_cast<uint16_t*>(output.data)+row*output.stride[0],
            static_cast<const uint16_t*>(source.data)+row*source.stride[0],width*2);
}
RadTensor local_tensor(void* pointer,uint32_t dtype,int rows,int width) {
    RadTensor result{};result.data=pointer;result.dtype=dtype;result.rank=2;
    result.shape[0]=rows;result.shape[1]=width;result.stride[0]=width;result.stride[1]=1;return result;
}
int fused_prepare_oracle(const RadArgs* args,RadStream) {
    const int rows=args->t[1].shape[0];const auto* values=args->t;
    std::vector<uint16_t> query(rows*3072),gated(rows*3072),rotated(rows*256),saved(rows*256);
    const auto qt=local_tensor(query.data(),RAD_BF16,rows,3072),gt=local_tensor(gated.data(),RAD_BF16,rows,3072);
    const auto kt=local_tensor(rotated.data(),RAD_BF16,rows,256),vt=local_tensor(saved.data(),RAD_BF16,rows,256);
    row_copy(values[9],vt,256);
    RadTensor prepare[]{values[0],values[1],values[2],values[3],values[4],values[5],qt,kt,gt};
    RadArgs call=*args;call.t=prepare;call.n_t=9;
    int status=cpu_prepare(&call,nullptr);if (status) return status;
    RadTensor store[]{kt,vt,values[10],values[11]};call.t=store;call.n_t=4;
    if ((status=ref_kv_store(&call,nullptr))) return status;
    row_copy(qt,values[6],3072);row_copy(kt,values[7],256);row_copy(gt,values[8],3072);return RAD_OK;
}
uint16_t gate_at(const RadTensor& gate,int row,int column,int rows) {
    const auto* data=static_cast<const uint16_t*>(gate.data);
    if (gate.rank==3) return data[row*gate.stride[0]+(column/256)*gate.stride[1]+column%256];
    if (gate.shape[0]==rows*12 && gate.shape[1]==256)
        return data[(row*12+column/256)*gate.stride[0]+column%256];
    if (gate.shape[1]==256) return data[row*gate.stride[0]+(column/256)*512+column%256];
    return data[row*gate.stride[0]+column];
}
int fused_paged_oracle(const RadArgs* args,RadStream) {
    const auto* values=args->t;const int rows=values[0].shape[0];
    std::vector<uint16_t> attention(rows*3072),product(rows*3072);
    RadTensor attended=local_tensor(attention.data(),RAD_BF16,rows,3072);
    RadTensor tensors[]{values[0],values[1],values[2],values[3],values[4],values[5],values[6],attended};
    RadArgs call=*args;call.t=tensors;call.n_t=8;
    int status=ref_attn_paged(&call,nullptr);if (status) return status;
    for (int row=0;row<rows;++row) for (int column=0;column<3072;++column) {
        const float input=rad_bf16_to_f32(gate_at(values[8],row,column,rows));
        const float sigmoid=rad_bf16_to_f32(rad_f32_to_bf16(1.0f/(1.0f+std::exp(-input))));
        product[row*3072+column]=rad_f32_to_bf16(sigmoid*rad_bf16_to_f32(attention[row*3072+column]));
    }
    RadTensor pt=local_tensor(product.data(),RAD_BF16,rows,3072);row_copy(pt,values[7],3072);
    RadTensor quantize[]{pt,values[9],values[10]};
    const RadParam params[]{RAD_INT("n",3072),RAD_INT("group",128),RAD_STR("dtype","bf16")};
    call.t=quantize;call.n_t=3;call.p=params;call.n_p=3;return ref_had_quant_act_i8g(&call,nullptr);
}
void fused_rejections(const std::vector<RadParam>&,const std::vector<RadTensor>&,int,uint64_t);
void fused_run(const char* label,int paged,std::vector<RadParam> params,std::vector<RadTensor> tensors) {
    RadArgs args{};args.p=params.data();args.n_p=params.size();uint64_t needed=0;
    insist(r3dh_attn_fused_workspace_bytes(&args,paged,&needed)==RAD_OK,"geometry-only workspace");
    const int64_t upper=params[0].kind==RAD_P_RANGE ? params[0].ihi : params[0].ival;
    insist(needed==uint64_t(paged ? std::min<int64_t>(2048,upper*256)*6144 : upper*6656)*2,"workspace formula");
    std::vector<RadTensor> remote=tensors;
    for (auto* blob:allocations)
        if (std::find(fused_resident.begin(),fused_resident.end(),blob)==fused_resident.end()) blob->push();
    for (auto& value:remote) if (value.data) {
        bool found=false;const uintptr_t pointer=reinterpret_cast<uintptr_t>(value.data);
        for (auto* blob:allocations) {
            const uintptr_t start=reinterpret_cast<uintptr_t>(blob->data<void>());
            if (pointer>=start && pointer<start+blob->bytes.size()-128) {
                value.data=blob->device ? blob->device+64+(pointer-start) : nullptr;found=true;break;
            }
        }
        insist(found,"fused fixture mapping");
    }
    args.t=tensors.data();args.n_t=tensors.size();
    insist((paged ? fused_paged_oracle(&args,nullptr) : fused_prepare_oracle(&args,nullptr))==RAD_OK,"independent fused oracle");
    if (gpu) {
        uint8_t* scratch=nullptr;hipcheck(hipMalloc(&scratch,needed+128));
        hipcheck(hipMemsetAsync(scratch,0xa5,needed+128,stream));
        args.t=remote.data();args.scratch=scratch+64;args.scratch_bytes=needed;
        const int status=r3dh_attn_fused_kernel_at(paged)->launch(&args,reinterpret_cast<RadStream>(stream));
        if (status) std::fprintf(stderr,"ATTN_FUSED status=%d label=%s rows=%lld\n",status,label,(long long)tensors[0].shape[0]);
        insist(status==RAD_OK,label);++device_calls;
        hipcheck(hipStreamSynchronize(stream));
        for (auto* blob:allocations) blob->compare(label,false);
        uint8_t guards[128];hipcheck(hipMemcpy(guards,scratch,64,hipMemcpyDeviceToHost));
        hipcheck(hipMemcpy(guards+64,scratch+64+needed,64,hipMemcpyDeviceToHost));
        for (uint8_t byte:guards) insist(byte==0xa5,"workspace guards");
        hipcheck(hipFree(scratch));
    }
    fused_rejections(params,tensors,paged,needed);
    ++fused_cases;
}
void fused_prepare_case(int rows,int seed,int variant,bool ranged=false) {
    const int pitch=6157,kpitch=269,opitch=3081,pages=(rows+3)/4+2;
    Blob qg(rows*pitch,RAD_BF16),key(rows*kpitch,RAD_BF16),value(rows*kpitch,RAD_BF16);
    Blob query(rows*opitch,RAD_BF16),rotated(rows*kpitch,RAD_BF16),gated(rows*opitch,RAD_BF16);
    Blob table(4096*64,RAD_F32),position(rows,RAD_I32),slot(rows,RAD_I32),weight(256,RAD_F32),kv(pages*2064,RAD_I8);
    fill(qg,seed);fill(key,seed+1);fill(value,seed+2);
    for (int column=0;column<256;++column) weight.data<float>()[column]=float(column%7-3)/16.0f;
    for (int row=0;row<4096;++row) for (int pair=0;pair<32;++pair) {
        const float angle=float(row)/std::pow(1e7f,float(pair)/32.0f);
        table.data<float>()[row*64+pair]=std::cos(angle);table.data<float>()[row*64+32+pair]=std::sin(angle);
    }
    for (int row=0;row<rows;++row) {position.data<int>()[row]=(row*127+seed)%4096;slot.data<int>()[row]=row+3;}
    RadTensor kt=tensor(variant==1 ? value : variant==2 ? key : rotated,{rows,256},kpitch);
    RadTensor qt=tensor(query,{rows,3072},opitch),gt=tensor(gated,{rows,3072},opitch);
    if (variant==1) qt=tensor(qg,{rows,3072},pitch);
    if (variant==2) {qt={};gt={};}
    if (variant==3) gt={};
    if (variant==4) qt={};
    for (int repeat=0;repeat<2;++repeat) {
    fill(qg,seed+repeat*19);fill(key,seed+1+repeat*13);fill(value,seed+2+repeat*11);
    auto params=fused_params(rows,false);
    if (ranged) params[0]=RAD_RANGE("M",1,rows);
    fused_run("fused_prepare_store",0,params,{tensor(qg,{rows,6144},pitch),tensor(key,{rows,256},kpitch),
        tensor(table,{4096,64}),tensor(position,{rows}),tensor(weight,{256}),tensor(weight,{256}),qt,kt,gt,
        tensor(value,{rows,256},kpitch),tensor(slot,{rows}),tensor(kv,{pages,1,4,516})});
    }
}
void fused_paged_case(int rows,int sequences,int context,int seed,bool selected,bool packed,bool alias,int layout=1,int geometry=0) {
    const int pages=(context+3)/4,width=selected ? 513 : pages,pitch=3081;
    Blob query(rows*pitch,RAD_BF16),gate(rows*(packed ? 6144 : pitch),RAD_BF16),output(rows*pitch,RAD_BF16);
    Blob codes(rows*3072,RAD_I8),scales(rows*24,RAD_F32),cache(pages*2064,RAD_I8);
    Blob table(sequences*width,RAD_I32),used(sequences,RAD_I32),cu(sequences+1,RAD_I32);
    fill(query,seed);fill(gate,seed+2);
    for (int page=0;page<pages;++page) for (int token=0;token<4;++token) {
        auto* data=cache.data<int8_t>()+page*2064+token*516;
        for (int column=0;column<512;++column) data[column]=int8_t((page*13+token*7+column*3+seed)%47-23);
        const uint16_t scale=0x2000;std::memcpy(data+512,&scale,2);std::memcpy(data+514,&scale,2);
    }
    for (int sequence=0;sequence<sequences;++sequence) {
        cu.data<int>()[sequence]=sequence*rows/sequences;
        used.data<int>()[sequence]=selected ? std::min(context/4,512)*4+context%4 : context;
        for (int column=0;column<width;++column) table.data<int>()[sequence*width+column]=selected ? ((column*7+sequence+seed)%pages) : column;
    }
    cu.data<int>()[sequences]=rows;
    if (selected && context>2051) for (int sequence=0;sequence<sequences;++sequence)
        for (int column=0;column<width;++column) table.data<int>()[sequence*width+column]=(column+1)%pages;
    RadTensor gt=packed ? tensor(gate,{rows,256},6144) : tensor(gate,{rows,3072},pitch);
    if (packed && layout==2) {gt=tensor(gate,{rows,12,256},6144);gt.stride[1]=512;}
    if (packed && layout==3) gt=tensor(gate,{rows*12,256},512);
    RadTensor ot=tensor(alias ? query : output,{rows,3072},pitch);
    RadTensor ct=tensor(codes,{rows,3072});
    if (alias) {ct=tensor(output,{rows,3072},pitch*2);ct.dtype=RAD_I8;}
    auto params=fused_params(selected ? 1 : std::min(rows,32),true);
    if (geometry) params[0]=RAD_RANGE("q_len",1,geometry);
    for (int repeat=0;repeat<2;++repeat) {
    fill(query,seed+repeat*17);fill(gate,seed+2+repeat*23);
    fused_run(selected ? "fused_selected_paged_gate" : "fused_normal_paged_gate",1,params,
        {tensor(query,{rows,3072},pitch),tensor(cache,{pages,1,4,516}),tensor(table,{sequences,width}),tensor(used,{sequences}),
        {},{},selected ? RadTensor{} : tensor(cu,{sequences+1}),ot,gt,ct,tensor(scales,{rows,24})});
    }
}
#include "r3dh_attn_fused_check_extra.h"
#include "r3dh_attn_fused_check_selected.h"
#include "r3dh_attn_fused_check_long.h"
void fused_host() {
    insist(r3dh_attn_fused_kernel_count()==2 && r3dh_attn_fused_schema_count()==2,"registry counts");
    insist(!r3dh_attn_fused_kernel_at(-1) && !r3dh_attn_fused_kernel_at(2) && !r3dh_attn_fused_schema_at(2),"registry bounds");
    uint64_t bytes=0;insist(r3dh_attn_fused_workspace_bytes(nullptr,0,&bytes)==RAD_E_INVAL,"null args");
    for (int paged=0;paged<2;++paged) for (int rows=1;rows<=32;++rows) {
        auto params=fused_params(rows,paged);RadArgs args{};args.p=params.data();args.n_p=params.size();
        auto* kernel=r3dh_attn_fused_kernel_at(paged);
        insist(kernel->domain==RAD_DOMAIN_DEVICE && kernel->scratch && kernel->opd_shape && kernel->launch,"device callbacks");
        insist(kernel->scratch(&args)>0,"scratch before tensors");
        const auto* schema=r3dh_attn_fused_schema_at(paged);
        for (int operand=0;operand<schema->n_operands;++operand) {RadOpdDesc desc{};insist(kernel->opd_shape(params.data(),params.size(),operand,&desc)==RAD_OK,"shape callbacks");}
        for (int index=0;index<int(params.size());++index) {
            auto changed=params;changed[index].kind=99;args.p=changed.data();
            insist(r3dh_attn_fused_workspace_bytes(&args,paged,&bytes)!=RAD_OK,"wrong param kind rejected");
        }
        auto duplicate=params;duplicate.push_back(params[0]);args.p=duplicate.data();args.n_p=duplicate.size();
        insist(r3dh_attn_fused_workspace_bytes(&args,paged,&bytes)!=RAD_OK,"duplicate rejected");
    }
}
}
int main(int argc,char** argv) {
    try {
        if (argc!=2 || (std::strcmp(argv[1],"--host") && std::strcmp(argv[1],"--gpu")
            && std::strcmp(argv[1],"--prepare-host") && std::strcmp(argv[1],"--prepare-gpu")
            && std::strcmp(argv[1],"--regression-gpu"))) return 2;
        const bool prepare_only=!std::strncmp(argv[1],"--prepare-",10);
        const bool regression_only=!std::strcmp(argv[1],"--regression-gpu");
        gpu=!std::strcmp(argv[1],"--gpu") || !std::strcmp(argv[1],"--prepare-gpu") || regression_only;
        fused_host();
        fused_query_contracts();
        fused_long_host();
        if (gpu) {
            int count=0;hipcheck(hipGetDeviceCount(&count));insist(count==1,"one visible device");
            hipDeviceProp_t prop{};hipcheck(hipGetDeviceProperties(&prop,0));
            insist(!std::strncmp(prop.gcnArchName,"gfx1100",7),"gfx1100 required");
            const char* expected=std::getenv("R3DH_ATTN_FUSED_EXPECT_BDF");
            insist(expected && (!std::strcmp(expected,"0000:c3:00.0") || !std::strcmp(expected,"0000:83:00.0")),"authorized expected PCI required");
            char pci[32]{};hipcheck(hipDeviceGetPCIBusId(pci,sizeof pci,0));insist(!std::strcmp(pci,expected),"physical GPU identity required");
            std::printf("ATTN_FUSED_DEVICE arch=%s pci=%s\n",prop.gcnArchName,pci);
            hipcheck(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));
        }
        const int prepare_start=fused_cases;
        for (int rows=1;rows<=64;++rows) fused_prepare_case(rows,13,rows%5);
        for (int rows:{64,128,512,2048}) {
            for (int variant=0;variant<5;++variant) fused_prepare_case(rows,29+variant,variant);
            fused_prepare_case(rows,47,0,true);
            fused_schema_case(rows,0);
            std::printf("ATTN_FUSED_PREPARE_CASE M=%d compare=exhaustive outputs=all KV=all inputs=all guards=exact variants=5 ranged=1 schema=1\n",rows);
            std::fflush(stdout);
        }
        std::printf("ATTN_FUSED_PREPARE_PASS mode=%s cases=%d checks=%zu comparisons=%zu device_calls=%d M_upper=2048 q_len_upper=32 weights=synthetic full_model=0\n",
            gpu ? "gpu" : "host",fused_cases-prepare_start,fused_checks,comparisons,device_calls);
        if (prepare_only) {
            if (gpu) hipcheck(hipStreamDestroy(stream));
            return 0;
        }
        for (int rows:{1,3,8,16,32}) for (int paged=0;paged<2;++paged) fused_schema_case(rows,paged);
        fused_paged_case(2048,256,31,17,false,true,true,3,8);
        fused_paged_case(64,64,128,53,true,true,true);
        fused_paged_case(64,1,128,59,false,true,true,3,1);
        for (int rows:{1,3,8,16,32}) for (int layout:{2,3})
            fused_paged_case(rows,rows,67,11,true,true,true,layout);
        for (int rows:{1,3,8,16,32}) {
            for (int sign:{0,1}) {
                fused_selected_case(rows,67,sign,false);
                fused_selected_case(rows,2052,sign,false);
            }
            fused_selected_case(rows,67,0,true);
        }
        for (int rows=1;rows<=16;++rows) for (int seed:{2,5}) {
            fused_prepare_case(rows,seed,(rows+seed)%5);
            fused_paged_case(rows,1,37,seed,false,seed==5,seed==5);
            fused_paged_case(rows,rows,67,seed,true,seed==5,seed==5);
        }
        for (int seed:{2,5}) {
            fused_prepare_case(32,seed,seed%4);fused_paged_case(32,1,64,seed,false,true,true);
            fused_paged_case(32,32,2052,seed,true,true,true);
            fused_paged_case(1,1,4096,seed,true,true,true);
            fused_paged_case(256,256,7,seed,true,true,true);
            fused_paged_case(32,8,31,seed,false,true,true);
        }
        if (gpu) {
            if (!regression_only) fused_long_gpu();
            else std::printf("ATTN_FUSED_REGRESSION_SCOPE legacy=298 prepare=180 M64_paged=4 long_context_run=0 q_len_upper=32 M64_attention_q_len=1 context=128\n");
            hipcheck(hipStreamDestroy(stream));
        }
        std::printf("ATTN_FUSED_PASS mode=%s cases=%d checks=%zu comparisons=%zu device_calls=%d strict_bits guards=exact reused_allocations=2\n",gpu ? "gpu" : "host",fused_cases,fused_checks,comparisons,device_calls);
        return 0;
    } catch (const std::exception& error) {std::fprintf(stderr,"ATTN_FUSED_FAIL %s\n",error.what());return 1;}
}
