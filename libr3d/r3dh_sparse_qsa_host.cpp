#include "r3dh_sparse_qsa_internal.h"
#include <algorithm>
#include <initializer_list>
using namespace r3dh_sparse;
namespace {
#define INT_PARAM(key) {key, RAD_P_INT, RAD_OPTIONAL}
#define REAL_PARAM(key) {key, RAD_P_F64, RAD_OPTIONAL}
#define STR_PARAM(key) {key, RAD_P_STR, RAD_OPTIONAL}
const RadParamSpec key_params[] = {
    INT_PARAM("M"), INT_PARAM("n"), INT_PARAM("ratio"), INT_PARAM("rotary_dim"), INT_PARAM("pos0"),
    INT_PARAM("block_size"), REAL_PARAM("eps"), REAL_PARAM("theta"), REAL_PARAM("scale"), REAL_PARAM("wadd"),
    STR_PARAM("dtype"), STR_PARAM("mode"), STR_PARAM("sections"), REAL_PARAM("yarn_factor"),
    REAL_PARAM("yarn_low"), REAL_PARAM("yarn_high"), REAL_PARAM("yarn_magnitude")};
const RadParamSpec score_params[] = {INT_PARAM("M"), INT_PARAM("n"), INT_PARAM("heads"),
    INT_PARAM("blocks"), INT_PARAM("seqs"), STR_PARAM("dtype")};
const RadParamSpec select_params[] = {INT_PARAM("M"), INT_PARAM("blocks"), INT_PARAM("topk"),
    INT_PARAM("ratio"), INT_PARAM("seqs"), INT_PARAM("signed")};
const RadParamSpec paged_params[] = {INT_PARAM("q_len"), INT_PARAM("head_dim"), INT_PARAM("gqa"),
    INT_PARAM("block_size"), INT_PARAM("n_head"), INT_PARAM("max_seqs"), INT_PARAM("max_ctx"),
    INT_PARAM("causal"), INT_PARAM("window"), REAL_PARAM("scale"), STR_PARAM("q_dtype"), STR_PARAM("kv_dtype")};
const RadOperandSpec key_operands[] = {{"k",RAD_OPD_IN,0},{"w",RAD_OPD_WEIGHT,0},
    {"page",RAD_OPD_IN,1},{"pos",RAD_OPD_IN,1},{"out",RAD_OPD_OUT,0}};
const RadOperandSpec score_operands[] = {{"q",RAD_OPD_IN,0},{"bk",RAD_OPD_IN,0},{"bt",RAD_OPD_IN,0},
    {"cu",RAD_OPD_IN,0},{"nc",RAD_OPD_IN,0},{"score",RAD_OPD_OUT,0}};
const RadOperandSpec select_operands[] = {{"score",RAD_OPD_IN,0},{"bt",RAD_OPD_IN,0},{"nc",RAD_OPD_IN,0},
    {"pos",RAD_OPD_IN,0},{"cu",RAD_OPD_IN,0},{"sel",RAD_OPD_OUT,0},{"nsel",RAD_OPD_OUT,0},{"seqused",RAD_OPD_OUT,0}};
const RadOperandSpec paged_operands[] = {{"q",RAD_OPD_IN,0},{"kv_cache",RAD_OPD_IN,0},{"block_table",RAD_OPD_IN,0},
    {"seqused",RAD_OPD_IN,0},{"k_descale",RAD_OPD_IN,1},{"v_descale",RAD_OPD_IN,1},
    {"cu_seqlens",RAD_OPD_IN,1},{"out",RAD_OPD_OUT,0}};
#define SCHEMA(name, params, operands, text) {name,params,int(sizeof(params)/sizeof(params[0])),operands,int(sizeof(operands)/sizeof(operands[0])),text}
const RadOpSchema schemas[] = {
    SCHEMA("qsa_block_key",key_params,key_operands,"Strict device pooled BF16 key, RMSNorm and NeoX/M-RoPE; n128 ratio4."),
    SCHEMA("qsa_score",score_params,score_operands,"Strict device ascending dot/ReLU sum; index 4x128."),
    SCHEMA("qsa_select",select_params,select_operands,"Strict device deterministic signed or ReLU top-k physical attention-page table."),
    SCHEMA("attn_paged",paged_params,paged_operands,"Strict device canonical selected or union table attention; page4 BF16/I8 KV, BF16 query/output.")};
bool vector(const RadTensor& value, unsigned dtype, int64_t minimum) {
    return tensor(value,dtype,1) && value.shape[0] >= minimum;
}
bool matrix(const RadTensor& value, unsigned dtype, int64_t rows, int64_t columns) {
    return tensor(value,dtype,2) && value.shape[0] >= rows && value.shape[1] == columns;
}
bool head_matrix(const RadTensor& value, int64_t rows, int64_t heads, int64_t dim) {
    if (value.rank == 2) return matrix(value,RAD_BF16,rows,heads*dim);
    return tensor(value,RAD_BF16,3) && value.shape[0] >= rows && value.shape[1] == heads
        && value.shape[2] == dim && value.stride[1] == dim;
}
bool disjoint(const RadTensor& left, const RadTensor& right) {
    if (!left.data || !right.data) return true;
    const uint64_t left_bytes = left.shape[0] * left.stride[0] * (left.dtype == RAD_BF16 ? 2 : left.dtype == RAD_I8 ? 1 : 4);
    const uint64_t right_bytes = right.shape[0] * right.stride[0] * (right.dtype == RAD_BF16 ? 2 : right.dtype == RAD_I8 ? 1 : 4);
    const uintptr_t begin_left = reinterpret_cast<uintptr_t>(left.data), begin_right = reinterpret_cast<uintptr_t>(right.data);
    return begin_left <= begin_right ? begin_right - begin_left >= left_bytes : begin_left - begin_right >= right_bytes;
}
int describe(const RadParam* params, int count, const char* name, int operand, RadOpdDesc* out) {
    if (!out || count < 0 || (count && !params)) return RAD_E_INVAL;
    RadArgs args{}; args.p = params; args.n_p = count;
    const int64_t rows = integer(&args,!std::strcmp(name,"attn_paged") ? "q_len" : "M",1);
    const int64_t blocks = integer(&args,"blocks",16), topk = integer(&args,"topk",512);
    const int64_t sequences = std::max<int64_t>(1,std::min<int64_t>(rows,integer(&args,"seqs",1)));
    if (rows < 1 || rows > 2048 || blocks < 1 || blocks > 50000 || topk < 1 || topk > 512) return RAD_E_UNSUPPORTED;
    *out = {}; out->idx_const = -1;
    auto dims = [&](unsigned dtype, std::initializer_list<int64_t> shape) {
        out->dtype = dtype; out->rank = unsigned(shape.size()); int index = 0;
        for (int64_t extent : shape) out->shape[index++] = extent;
    };
    auto index = [&](std::initializer_list<int64_t> shape, int64_t maximum, int64_t constant) {
        dims(RAD_I32,shape); out->fill = RAD_FILL_INDEX; out->idx_max = maximum; out->idx_const = constant;
    };
    if (!std::strcmp(name,"qsa_block_key")) {
        if (integer(&args,"n",128)!=128 || integer(&args,"ratio",4)!=4) return RAD_E_UNSUPPORTED;
        switch (operand) {
            case 0: dims(RAD_BF16,{rows*4,128}); break;
            case 1: dims(RAD_F32,{128}); break;
            case 2: break;
            case 3: index({rows},rows*4,-1); break;
            case 4: dims(RAD_BF16,{rows,128}); break;
            default: return RAD_E_INVAL;
        }
    } else if (!std::strcmp(name,"qsa_score")) {
        if (integer(&args,"n",128)!=128 || integer(&args,"heads",4)!=4) return RAD_E_UNSUPPORTED;
        switch (operand) {
            case 0: dims(RAD_BF16,{rows,512}); break;
            case 1: dims(RAD_BF16,{sequences*blocks,128}); break;
            case 2: index({sequences,blocks},sequences*blocks,-1); break;
            case 3: index({sequences+1},rows+1,-1); out->flags = RAD_OPD_F_IDX_PREFIX; break;
            case 4: index({sequences},blocks+1,blocks); break;
            case 5: dims(RAD_F32,{rows,blocks}); break;
            default: return RAD_E_INVAL;
        }
    } else if (!std::strcmp(name,"qsa_select")) {
        switch (operand) {
            case 0: dims(RAD_F32,{rows,blocks}); break;
            case 1: index({sequences,blocks},sequences*blocks,-1); break;
            case 2: index({sequences},blocks+1,blocks); break;
            case 3: index({rows},blocks*4,blocks*4-1); break;
            case 4: index({sequences+1},rows+1,-1); out->flags = RAD_OPD_F_IDX_PREFIX; break;
            case 5: dims(RAD_I32,{rows,topk+1}); break;
            case 6: case 7: dims(RAD_I32,{rows}); break;
            default: return RAD_E_INVAL;
        }
    } else {
        const int64_t dim = integer(&args,"head_dim",256), gqa = integer(&args,"gqa",12);
        if ((dim!=128 && dim!=256) || gqa<1 || gqa>16) return RAD_E_UNSUPPORTED;
        const bool i8 = !string_is(&args,"kv_dtype","bf16");
        switch (operand) {
            case 0: case 7: dims(RAD_BF16,{rows,gqa*dim}); break;
            case 1: dims(i8 ? RAD_I8 : RAD_BF16,{blocks,1,4,2*dim+(i8?4:0)}); break;
            case 2: index({1,blocks},blocks,-1); break;
            case 3: index({1},blocks*4+1,blocks*4); break;
            case 4: case 5: break;
            case 6: index({2},rows+1,-1); out->flags=RAD_OPD_F_IDX_PREFIX; break;
            default: return RAD_E_INVAL;
        }
    }
    return RAD_OK;
}
}
extern "C" const RadOpSchema* r3dh_sparse_qsa_schema_at(int index) { return index>=0 && index<4 ? &schemas[index] : nullptr; }
extern "C" int r3dh_sparse_qsa_schema_count() { return 4; }
extern "C" int r3dh_sparse_qsa_shape(const char* name, const RadArgs* args) {
    if (!name || !args || !args->t || args->n_p<0 || (args->n_p && !args->p)) return RAD_E_INVAL;
    const RadOpSchema* schema = nullptr;
    for (const auto& entry : schemas) if (!std::strcmp(name,entry.op)) schema=&entry;
    if (!schema) return RAD_E_UNSUPPORTED;
    if (args->n_t != schema->n_operands) return RAD_E_INVAL;
    for (int index=0; index<args->n_p; ++index) {
        if (!args->p[index].key) return RAD_E_INVAL;
        for (int previous=0;previous<index;++previous)
            if (!std::strcmp(args->p[index].key,args->p[previous].key)) return RAD_E_INVAL;
        bool known=false;
        for (int other=0; other<schema->n_params; ++other) if (!std::strcmp(args->p[index].key,schema->params[other].key)) {
            known=true;
            if (args->p[index].kind != schema->params[other].type) return RAD_E_INVAL;
        }
        if (!known) return RAD_E_UNSUPPORTED;
    }
    const auto* values=args->t;
    if (!std::strcmp(name,"qsa_block_key")) {
        if (integer(args,"n",0)!=128 || integer(args,"ratio",0)!=4 || integer(args,"block_size",4)!=4
            || !string_is(args,"dtype","bf16") || real(args,"yarn_factor",0)!=0) return RAD_E_UNSUPPORTED;
        const char* mode=rad_args_gets(args,"mode");
        if (mode && std::strcmp(mode,"neox") && std::strcmp(mode,"mrope") && std::strcmp(mode,"imrope")) return RAD_E_UNSUPPORTED;
        const int64_t rot=integer(args,"rotary_dim",0), blocks=values[2].data ? values[2].shape[0] : values[4].shape[0];
        if (rot<0 || rot>128 || (rot&1) || blocks<1 || blocks>2048) return RAD_E_SHAPE;
        if (!matrix(values[0],RAD_BF16,blocks*4,128) || values[0].shape[0]!=blocks*4
            || !vector(values[1],RAD_F32,128) || !page_tensor(values[4])) return RAD_E_SHAPE;
        if (values[2].data && !vector(values[2],RAD_I32,blocks)) return RAD_E_SHAPE;
        if (values[3].data) {
            if (values[3].rank==2) {
                if (!tensor(values[3],RAD_I32,2) || values[3].shape[0]<blocks || (values[3].shape[1]!=1 && values[3].shape[1]!=3)) return RAD_E_SHAPE;
            } else if (!vector(values[3],RAD_I32,blocks)) return RAD_E_SHAPE;
        } else if (integer(args,"pos0",-1)<0 || integer(args,"pos0",0)>(1LL<<30)-8192) return RAD_E_INVAL;
        int section[3];
        if (!sections(args,section)) return RAD_E_UNSUPPORTED;
        if (values[3].data && values[3].rank==2 && values[3].shape[1]==3 && rad_args_gets(args,"sections") && mode && std::strcmp(mode,"neox")
            && section[0]+section[1]+section[2]!=rot/2) return RAD_E_SHAPE;
        if (!std::isfinite(real(args,"eps",1e-6)) || real(args,"eps",1e-6)<=0
            || !std::isfinite(real(args,"wadd",1)) || !std::isfinite(real(args,"theta",10000)) || real(args,"theta",10000)<=0
            || !std::isfinite(real(args,"scale",1)) || real(args,"scale",1)<=0) return RAD_E_INVAL;
    } else if (!std::strcmp(name,"qsa_score")) {
        if (integer(args,"n",0)!=128 || integer(args,"heads",0)!=4 || !string_is(args,"dtype","bf16")) return RAD_E_UNSUPPORTED;
        const int64_t rows=values[5].shape[0], sequences=values[3].shape[0]-1;
        if (rows<1 || rows>2048 || sequences<1 || sequences>2048 || !matrix(values[0],RAD_BF16,rows,512)
            || !page_tensor(values[1]) || !tensor(values[2],RAD_I32,2) || values[2].shape[0]<sequences
            || !vector(values[3],RAD_I32,sequences+1) || !vector(values[4],RAD_I32,sequences)
            || !tensor(values[5],RAD_F32,2) || values[5].shape[1]>50000) return RAD_E_SHAPE;
    } else if (!std::strcmp(name,"qsa_select")) {
        if (integer(args,"ratio",0)!=4 || integer(args,"topk",0)<1 || integer(args,"topk",0)>512
            || integer(args,"signed",0)<0 || integer(args,"signed",0)>1) return RAD_E_UNSUPPORTED;
        const int64_t rows=values[0].shape[0], sequences=values[4].shape[0]-1, topk=integer(args,"topk",0);
        if (rows<1 || rows>2048 || sequences<1 || sequences>2048 || !tensor(values[0],RAD_F32,2)
            || values[0].shape[1]>50000 || !tensor(values[1],RAD_I32,2) || values[1].shape[0]<sequences
            || !vector(values[2],RAD_I32,sequences) || !vector(values[3],RAD_I32,rows) || !vector(values[4],RAD_I32,sequences+1)
            || !matrix(values[5],RAD_I32,rows,topk+1) || !vector(values[6],RAD_I32,rows) || !vector(values[7],RAD_I32,rows)) return RAD_E_SHAPE;
    } else {
        const int64_t dim=integer(args,"head_dim",0), gqa=integer(args,"gqa",1);
        if ((dim!=128 && dim!=256) || gqa<1 || gqa>16 || integer(args,"block_size",4)!=4
            || integer(args,"causal",1)!=1 || integer(args,"window",0)!=0
            || !string_is(args,"q_dtype","bf16") || values[4].data || values[5].data) return RAD_E_UNSUPPORTED;
        if (values[1].dtype!=RAD_BF16 && values[1].dtype!=RAD_I8) return RAD_E_UNSUPPORTED;
        if (!string_is(args,"kv_dtype",values[1].dtype==RAD_I8 ? "i8" : "bf16")) return RAD_E_UNSUPPORTED;
        const int64_t heads=values[1].shape[1]*gqa, rows=values[0].shape[0], sequences=values[3].shape[0];
        if (rows<1 || rows>2048 || sequences<1 || sequences>2048 || heads<1 || heads>128
            || !tensor(values[1],values[1].dtype,4) || values[1].shape[2]!=4 || values[1].shape[3]!=2*dim+(values[1].dtype==RAD_I8?4:0)
            || !head_matrix(values[0],rows,heads,dim) || !head_matrix(values[7],rows,heads,dim)
            || !tensor(values[2],RAD_I32,2) || values[2].shape[0]<sequences
            || !vector(values[3],RAD_I32,sequences)) return RAD_E_SHAPE;
        if (values[6].data ? !vector(values[6],RAD_I32,sequences+1) || values[6].shape[0]!=sequences+1 : rows%sequences!=0) return RAD_E_SHAPE;
        if (!std::isfinite(real(args,"scale",1.0/std::sqrt(double(dim)))) || real(args,"scale",1)<=0) return RAD_E_INVAL;
    }
    for (int output=0; output<args->n_t; ++output) {
        if (schema->operands[output].role!=RAD_OPD_OUT) continue;
        for (int input=0; input<args->n_t; ++input)
            if (input!=output && !disjoint(values[output],values[input])) return RAD_E_SHAPE;
    }
    return RAD_OK;
}
#define SHAPE(function,op) extern "C" int function(const RadParam* params,int count,int operand,RadOpdDesc* out) {return describe(params,count,op,operand,out);}
SHAPE(r3dh_sparse_qsa_block_key_shape,"qsa_block_key")
SHAPE(r3dh_sparse_qsa_score_shape,"qsa_score")
SHAPE(r3dh_sparse_qsa_select_shape,"qsa_select")
SHAPE(r3dh_sparse_qsa_paged_shape,"attn_paged")
