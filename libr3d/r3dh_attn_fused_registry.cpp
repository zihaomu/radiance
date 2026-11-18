#include "r3dh_attn_fused.h"
#include <algorithm>
#include <cstring>
#include <initializer_list>
#include <type_traits>
namespace {
#define SIZE(values) int(sizeof(values)/sizeof(values[0]))
#define PI(key) {key,RAD_P_INT,RAD_REQUIRED,RAD_PROLE_NONE}
#define OI(key) {key,RAD_P_INT,RAD_OPTIONAL,RAD_PROLE_NONE}
#define OF(key) {key,RAD_P_F64,RAD_OPTIONAL,RAD_PROLE_NONE}
#define PS(key) {key,RAD_P_STR,RAD_REQUIRED,RAD_PROLE_NONE}
const RadParamSpec prepare_params[]{PI("M"),PI("head_dim"),PI("n_head"),PI("n_head_kv"),PI("rot"),PS("q_dtype"),
    OF("eps"),OF("wadd"),OI("cs_f32"),OI("pos_i64"),PI("block_size"),PS("kv_dtype"),OF("k_scale"),OF("v_scale")};
const RadOperandSpec prepare_operands[]{
    {"q_gate",RAD_OPD_IN,1},{"k",RAD_OPD_IN,0},{"cos_sin",RAD_OPD_IN,0},{"positions",RAD_OPD_IN,0},
    {"q_w",RAD_OPD_WEIGHT,1},{"k_w",RAD_OPD_WEIGHT,0},{"q_out",RAD_OPD_OUT,1},{"k_out",RAD_OPD_OUT,0},
    {"gate_out",RAD_OPD_OUT,1},{"v",RAD_OPD_IN,0},{"slot_mapping",RAD_OPD_IN,0},{"kv_cache",RAD_OPD_INOUT,0}};
const RadParamSpec paged_params[]{PI("q_len"),PI("head_dim"),PI("gqa"),PI("block_size"),PI("causal"),PI("window"),
    PS("q_dtype"),PS("kv_dtype"),OF("scale"),OI("max_ctx"),OI("n_head"),
    {"max_seqs",RAD_P_INT,RAD_OPTIONAL,RAD_PROLE_CAPACITY},PI("group"),PS("act")};
const RadOperandSpec paged_operands[]{
    {"q",RAD_OPD_IN,0},{"kv_cache",RAD_OPD_IN,0},{"block_table",RAD_OPD_IN,0},{"seqused",RAD_OPD_IN,0},
    {"k_descale",RAD_OPD_IN,1},{"v_descale",RAD_OPD_IN,1},{"cu_seqlens",RAD_OPD_IN,1},{"out",RAD_OPD_OUT,0},
    {"gate",RAD_OPD_IN,0},{"q8",RAD_OPD_OUT,0},{"q8_scale",RAD_OPD_OUT,0}};
const RadOpSchema schemas[]{
    {"qk_norm_rope_gate_kv_store",prepare_params,SIZE(prepare_params),prepare_operands,SIZE(prepare_operands),"Native strict BF16 prepare then page4 inline-scale I8 cache store."},
    {"attn_paged_gate_had_quant",paged_params,SIZE(paged_params),paged_operands,SIZE(paged_operands),"Native selected-QSA or ordinary paged attention, sigmoid BF16 product, group128 Hadamard I8 quantization."}};
const RadConstraint prepare_constraints[]{RAD_CGE("M",1),RAD_CLE("M",2048),RAD_CEQ("head_dim",256),RAD_CEQ("n_head",12),
    RAD_CEQ("n_head_kv",1),RAD_CEQ("rot",64),RAD_CEQ("cs_f32",1),RAD_CEQ("block_size",4),{"q_dtype",RAD_C_IN,0,"bf16"},{"kv_dtype",RAD_C_IN,0,"i8"}};
const RadConstraint paged_constraints[]{RAD_CGE("q_len",1),RAD_CLE("q_len",32),RAD_CEQ("head_dim",256),RAD_CEQ("n_head",12),
    RAD_CEQ("gqa",12),RAD_CEQ("block_size",4),RAD_CEQ("causal",1),RAD_CEQ("window",0),RAD_CEQ("group",128),
    RAD_CIN("max_seqs","256 2048"),RAD_CGE("max_ctx",1),RAD_CLE("max_ctx",200000),{"q_dtype",RAD_C_IN,0,"bf16"},{"kv_dtype",RAD_C_IN,0,"i8"},{"act",RAD_C_IN,0,"sigmoid"}};
int64_t scratch_prepare(const RadArgs* args) { uint64_t bytes=0; return r3dh_attn_fused_workspace_bytes(args,0,&bytes)==RAD_OK ? int64_t(bytes) : -1; }
int64_t scratch_paged(const RadArgs* args) { uint64_t bytes=0; return r3dh_attn_fused_workspace_bytes(args,1,&bytes)==RAD_OK ? int64_t(bytes) : -1; }
static_assert(std::is_same_v<decltype(&scratch_prepare),RadScratchFn>);
static_assert(std::is_same_v<decltype(&scratch_paged),RadScratchFn>);
RadOpdDesc tensor(uint32_t dtype,std::initializer_list<int64_t> shape) {
    RadOpdDesc result{}; result.dtype=dtype; result.rank=shape.size(); result.idx_const=-1;
    std::copy(shape.begin(),shape.end(),result.shape); return result;
}
RadOpdDesc indices(std::initializer_list<int64_t> shape,int64_t bound,int64_t constant=-1,uint32_t flags=0) {
    auto result=tensor(RAD_I32,shape); result.fill=RAD_FILL_INDEX; result.idx_max=bound;
    result.idx_const=constant; result.flags=flags; return result;
}
int describe(const RadParam* params,int count,int operand,RadOpdDesc* out,int paged) {
    if (!out || operand<0 || operand>=(paged ? 11 : 12)) return RAD_E_INVAL;
    RadArgs args{}; args.p=params; args.n_p=count; uint64_t bytes=0;
    const int status=r3dh_attn_fused_workspace_bytes(&args,paged,&bytes); if (status) return status;
    const int64_t rows=rad_param_getdim(params,count,paged ? "q_len" : "M",0);
    if (!paged) {
        const int64_t pages=(rows+3)/4+2;
        const RadOpdDesc values[]{tensor(RAD_BF16,{rows,6144}),tensor(RAD_BF16,{rows,256}),tensor(RAD_F32,{4096,64}),
            indices({rows},4096),tensor(RAD_F32,{256}),tensor(RAD_F32,{256}),tensor(RAD_BF16,{rows,3072}),tensor(RAD_BF16,{rows,256}),
            tensor(RAD_BF16,{rows,3072}),tensor(RAD_BF16,{rows,256}),indices({rows},pages*4,-1,RAD_OPD_F_IDX_UNIQUE),tensor(RAD_I8,{pages,1,4,516})};
        *out=values[operand];
    } else {
        const int64_t context=rad_param_getdim(params,count,"max_ctx",0),pages=(context+3)/4;
        const int64_t visible=std::min<int64_t>(context/4,512)*4+context%4;
        const RadOpdDesc values[]{tensor(RAD_BF16,{rows,3072}),tensor(RAD_I8,{pages,1,4,516}),indices({rows,513},pages),
            indices({rows},visible+1,visible),{},{},{},tensor(RAD_BF16,{rows,3072}),tensor(RAD_BF16,{rows,3072}),tensor(RAD_I8,{rows,3072}),tensor(RAD_F32,{rows,24})};
        *out=values[operand];
    }
    return RAD_OK;
}
int prepare_shape(const RadParam* params,int count,int operand,RadOpdDesc* out) {return describe(params,count,operand,out,0);}
int paged_shape(const RadParam* params,int count,int operand,RadOpdDesc* out) {return describe(params,count,operand,out,1);}
const RadKernelInfo kernels[]{
    {"r3dh_attn_fused_prepare_store","qk_norm_rope_gate_kv_store","attn","Native prepare + KV store composition","M1..2048 hd256 qh12 kvh1 rot64 page4","bf16 i8",RAD_DOMAIN_DEVICE,10,prepare_constraints,SIZE(prepare_constraints),nullptr,0,nullptr,nullptr,nullptr,r3dh_attn_fused_prepare_store,scratch_prepare,nullptr,nullptr,prepare_shape,nullptr,nullptr,0},
    {"r3dh_attn_fused_paged_gate","attn_paged_gate_had_quant","attn","Native paged + gate Hadamard quant composition","q_len1..32 max_seqs256/2048 max_ctx200000 hd256 gqa12 page4 group128 selected512+tail","bf16 i8 f32",RAD_DOMAIN_DEVICE,10,paged_constraints,SIZE(paged_constraints),nullptr,0,nullptr,nullptr,nullptr,r3dh_attn_fused_paged_gate,scratch_paged,nullptr,nullptr,paged_shape,nullptr,nullptr,0}};
}
extern "C" int r3dh_attn_fused_kernel_count() {return SIZE(kernels);}
extern "C" const RadKernelInfo* r3dh_attn_fused_kernel_at(int index) {return index>=0 && index<SIZE(kernels) ? &kernels[index] : nullptr;}
extern "C" int r3dh_attn_fused_schema_count() {return SIZE(schemas);}
extern "C" const RadOpSchema* r3dh_attn_fused_schema_at(int index) {return index>=0 && index<SIZE(schemas) ? &schemas[index] : nullptr;}
