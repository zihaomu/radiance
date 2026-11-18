#include "r3dh_attn_fused.h"
#include "r3dh_attn_qsa.h"
#include "r3dh_sparse_qsa.h"
#include "r3dh_qwen_ops.h"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstring>
#include <cstdint>
#include <limits>

namespace {
__global__ void gather_gate(const uint16_t* input,uint16_t* output,int64_t rows,int64_t pitch,int mode,int64_t head_pitch) {
    const int64_t column=blockIdx.x*blockDim.x+threadIdx.x;
    if (column>=rows*3072) return;
    const int64_t row=column/3072, within=column%3072;
    const int64_t offset=mode==0 ? row*pitch+within : mode==1 ? row*pitch+(within/256)*head_pitch+within%256
        : (row*12+within/256)*pitch+within%256;
    output[column]=input[offset];
}
}

namespace {
RadTensor matrix(void* data, uint32_t dtype, int64_t rows, int64_t width) {
    RadTensor value{}; value.data=data; value.dtype=dtype; value.rank=2;
    value.shape[0]=rows; value.shape[1]=width; value.stride[0]=width; value.stride[1]=1;
    return value;
}
bool view(const RadTensor& input, uint32_t dtype, int64_t rows, int64_t width, RadTensor& output) {
    if (!input.data || input.dtype!=dtype || input.rank<2 || input.rank>3
        || input.shape[0]!=rows || input.stride[input.rank-1]!=1
        || input.stride[0]<width || input.stride[0]>(1ll<<30)) return false;
    if (input.rank==2 && input.shape[1]!=width) return false;
    if (input.rank==3 && (input.shape[1]<1 || input.shape[2]<1 || input.shape[1]>width
        || input.shape[2]>width || input.shape[1]*input.shape[2]!=width || input.stride[1]!=input.shape[2])) return false;
    output=matrix(input.data,dtype,rows,width); output.stride[0]=input.stride[0]; return true;
}
bool vector(const RadTensor& value,uint32_t dtype,int64_t rows) {
    return value.data && value.dtype==dtype && ((value.rank==1 && value.shape[0]==rows && value.stride[0]==1)
        || (value.rank==2 && value.shape[0]==rows && value.shape[1]==1 && value.stride[0]==1 && value.stride[1]==1));
}
bool cache(const RadTensor& value) {
    return value.data && value.dtype==RAD_I8 && value.rank==4 && value.shape[0]>0 && value.shape[0]<=r3dh_attn_qsa_max_cache_pages
        && value.shape[1]==1 && value.shape[2]==4 && value.shape[3]==516 && value.stride[0]==2064
        && value.stride[1]==2064 && value.stride[2]==516 && value.stride[3]==1;
}
bool extent(const RadTensor& value,uintptr_t& begin,uint64_t& bytes) {
    begin=reinterpret_cast<uintptr_t>(value.data); bytes=0;
    if (!value.data) return true;
    if (value.rank<1 || value.rank>4) return false;
    const uint64_t item=rad_dtype_bits(value.dtype)/8;
    if (!item) return false;
    uint64_t elements=1;
    for (int axis=0;axis<value.rank;++axis) {
        if (value.shape[axis]<1 || value.stride[axis]<1) return false;
        const uint64_t count=value.shape[axis]-1,pitch=value.stride[axis];
        if (count>(std::numeric_limits<uint64_t>::max()-elements)/pitch) return false;
        elements+=count*pitch;
    }
    if (elements>(std::numeric_limits<uintptr_t>::max()-begin)/item) return false;
    bytes=elements*item; return true;
}
bool overlap(uintptr_t left,uint64_t left_bytes,uintptr_t right,uint64_t right_bytes) {
    return left_bytes && right_bytes && (left<=right ? right-left<left_bytes : left-right<right_bytes);
}
int aliases(const RadArgs* args,uint64_t scratch_bytes,const int* outputs,int count) {
    uintptr_t begins[12]{}; uint64_t sizes[12]{};
    const uintptr_t scratch=reinterpret_cast<uintptr_t>(args->scratch);
    if (scratch_bytes>std::numeric_limits<uintptr_t>::max()-scratch) return RAD_E_SHAPE;
    for (int index=0;index<args->n_t;++index) {
        if (!extent(args->t[index],begins[index],sizes[index])
            || overlap(scratch,scratch_bytes,begins[index],sizes[index])) return RAD_E_SHAPE;
    }
    for (int index=0;index<count;++index)
        for (int other=0;other<index;++other)
            if (overlap(begins[outputs[index]],sizes[outputs[index]],begins[outputs[other]],sizes[outputs[other]]))
                return RAD_E_SHAPE;
    return RAD_OK;
}
int copy(const RadTensor& source,const RadTensor& destination,hipStream_t stream) {
    return hipMemcpy2DAsync(destination.data,destination.stride[0]*2,source.data,source.stride[0]*2,
        source.shape[1]*2,source.shape[0],hipMemcpyDeviceToDevice,stream)==hipSuccess ? RAD_OK : RAD_E_DEVICE;
}
int filter(const RadArgs* args, const char* const* names,int count,RadParam* result) {
    int total=0;
    for (int index=0;index<args->n_p;++index)
        for (int other=0;other<count;++other) if (!std::strcmp(args->p[index].key,names[other])) {
            result[total]=args->p[index];
            if (result[total].kind==RAD_P_RANGE) {
                result[total].kind=RAD_P_INT;result[total].ival=result[total].ihi;
            }
            ++total;
        }
    return total;
}
int workspace(const RadArgs* args,int paged,uint64_t& bytes) {
    const int status=r3dh_attn_fused_workspace_bytes(args,paged,&bytes);
    if (status) return status;
    if (!args->scratch || args->scratch_bytes<0 || uint64_t(args->scratch_bytes)<bytes
        || reinterpret_cast<uintptr_t>(args->scratch)%16) return RAD_E_INVAL;
    return RAD_OK;
}
}
extern "C" int r3dh_attn_fused_prepare_store(const RadArgs* args,RadStream stream) {
    if (!args || !args->t || args->n_t!=12) return RAD_E_INVAL;
    uint64_t bytes=0; int status=workspace(args,0,bytes);
    if (status) return status;
    const RadTensor* values=args->t;
    const int64_t rows=values[0].shape[0];
    const int64_t band=rad_param_getdim(args->p,args->n_p,"M",0);
    if (rows<1 || rows>band || band>2048) return RAD_E_SHAPE;
    RadTensor query_gate{},key{},value{},query_out{},key_out{},gate_out{};
    if (!view(values[0],RAD_BF16,rows,6144,query_gate) || !view(values[1],RAD_BF16,rows,256,key)
        || !view(values[9],RAD_BF16,rows,256,value) || !vector(values[3],RAD_I32,rows)
        || !vector(values[4],RAD_F32,256) || !vector(values[5],RAD_F32,256)
        || !vector(values[10],RAD_I32,rows) || !cache(values[11])
        || (values[6].data && !view(values[6],RAD_BF16,rows,3072,query_out))
        || !view(values[7],RAD_BF16,rows,256,key_out)
        || (values[8].data && !view(values[8],RAD_BF16,rows,3072,gate_out))
        || values[2].dtype!=RAD_F32 || !values[2].data || values[2].rank!=2 || values[2].shape[0]<1
        || values[2].shape[1]!=64 || values[2].stride[0]!=64 || values[2].stride[1]!=1) return RAD_E_SHAPE;
    const int outputs[]{6,7,8,11};
    if ((status=aliases(args,bytes,outputs,4))) return status;
    auto* storage=static_cast<uint16_t*>(args->scratch);
    RadTensor query=matrix(storage,RAD_BF16,rows,3072);
    RadTensor gated=matrix(storage+rows*3072,RAD_BF16,rows,3072);
    RadTensor rotated=matrix(storage+rows*6144,RAD_BF16,rows,256);
    RadTensor saved_value=matrix(storage+rows*6400,RAD_BF16,rows,256);
    const auto native=reinterpret_cast<hipStream_t>(stream);
    if ((status=copy(value,saved_value,native))) return status;
    RadTensor prepare[]{query_gate,key,values[2],values[3],values[4],values[5],query,rotated,gated};
    prepare[3].rank=1; prepare[3].shape[0]=rows; prepare[3].stride[0]=1;
    RadArgs call=*args; call.t=prepare; call.n_t=9;
    if ((status=r3dh_attn_qsa_prepare(&call,stream))) return status;
    RadTensor store[]{rotated,saved_value,values[10],values[11]};
    store[2].rank=1; store[2].shape[0]=rows; store[2].stride[0]=1;
    call.t=store; call.n_t=4;
    if ((status=r3dh_attn_qsa_kv_store(&call,stream))) return status;
    if (values[6].data && (status=copy(query,query_out,native))) return status;
    if ((status=copy(rotated,key_out,native))) return status;
    if (values[8].data && (status=copy(gated,gate_out,native))) return status;
    return RAD_OK;
}
extern "C" int r3dh_attn_fused_paged_gate(const RadArgs* args,RadStream stream) {
    uint64_t bytes=0; int status=workspace(args,1,bytes);
    if (status) return status;
    if (!args->t || args->n_t!=11) return RAD_E_INVAL;
    const RadTensor* values=args->t;
    const int64_t rows=values[0].shape[0], sequences=values[3].shape[0];
    const int64_t max_seqs=rad_param_getdim(args->p,args->n_p,"max_seqs",0);
    const int64_t capacity=std::min<int64_t>(2048,rad_param_getdim(args->p,args->n_p,"q_len",0)*max_seqs);
    RadTensor query{},gate{},output{},codes{},scales{};
    if (rows<1 || rows>capacity || sequences<1 || sequences>max_seqs || !view(values[0],RAD_BF16,rows,3072,query)
        || !view(values[7],RAD_BF16,rows,3072,output)
        || !view(values[9],RAD_I8,rows,3072,codes) || !view(values[10],RAD_F32,rows,24,scales)
        || !cache(values[1]) || values[4].data || values[5].data) return RAD_E_SHAPE;
    if (!vector(values[3],RAD_I32,sequences) || (values[6].data && !vector(values[6],RAD_I32,sequences+1))) return RAD_E_SHAPE;
    if (!values[6].data && rows%sequences) return RAD_E_SHAPE;
    int gate_mode=0;
    if (!view(values[8],RAD_BF16,rows,3072,gate)) {
        const auto& packed=values[8];
        if (!packed.data || packed.dtype!=RAD_BF16 || packed.rank<2 || packed.rank>3 || packed.stride[packed.rank-1]!=1) return RAD_E_SHAPE;
        if (packed.rank==2 && packed.shape[0]==rows && packed.shape[1]==256 && packed.stride[0]>=6144) gate_mode=1;
        else if (packed.rank==3 && packed.shape[0]==rows && packed.shape[1]==12 && packed.shape[2]==256
            && packed.stride[1]==512 && packed.stride[0]>=6144) gate_mode=1;
        else if (packed.rank==2 && packed.shape[0]==rows*12 && packed.shape[1]==256 && packed.stride[0]==512) gate_mode=2;
        else return RAD_E_SHAPE;
    }
    RadTensor temporary=matrix(args->scratch,RAD_BF16,rows,3072);
    RadTensor attention[]{query,values[1],values[2],values[3],{},{},values[6],temporary};
    attention[3].rank=1; attention[3].stride[0]=1;
    if (attention[6].data) {attention[6].rank=1; attention[6].stride[0]=1;}
    const char* const names[]{"q_len","head_dim","gqa","block_size","n_head","max_seqs","max_ctx","causal","window","scale","q_dtype","kv_dtype"};
    RadParam parameters[16]{};
    RadArgs call=*args; call.t=attention; call.n_t=8;
    call.p=parameters; call.n_p=filter(args,names,12,parameters);
    if ((status=r3dh_sparse_qsa_shape("attn_paged",&call))) return status;
    const bool selected=!values[6].data && sequences==rows;
    if (values[2].shape[1]>(selected ? 513 : 50000)) return RAD_E_UNSUPPORTED;
    const int outputs[]{7,9,10};
    if ((status=aliases(args,bytes,outputs,3))) return status;
    status=selected ? r3dh_sparse_qsa_paged(&call,stream) : r3dh_attn_qsa_paged(&call,stream);
    if (status) return status;
    gate=matrix(static_cast<uint16_t*>(args->scratch)+rows*3072,RAD_BF16,rows,3072);
    hipLaunchKernelGGL(gather_gate,dim3((rows*3072+255)/256),dim3(256),0,reinterpret_cast<hipStream_t>(stream),
        static_cast<const uint16_t*>(values[8].data),static_cast<uint16_t*>(gate.data),rows,values[8].stride[0],gate_mode,512);
    if (hipGetLastError()!=hipSuccess) return RAD_E_DEVICE;
    RadTensor quantize[]{gate,temporary,codes,scales,output};
    const RadParam quant_params[]{RAD_INT("n",3072),RAD_INT("group",128),RAD_STR("act","sigmoid"),RAD_STR("dtype","bf16")};
    call.t=quantize; call.n_t=5; call.p=quant_params; call.n_p=4;
    return r3dh_qwen_ops_gate_had_quant_i8g(&call,stream);
}
