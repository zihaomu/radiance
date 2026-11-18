#include "r3dh_attn_fused.h"
#include <cstring>
#include <algorithm>
#include <cmath>
namespace {
int validate(const RadArgs* args,int paged) {
    const auto* schema=r3dh_attn_fused_schema_at(paged);
    for (int index=0;index<args->n_p;++index) {
        const auto& param=args->p[index];
        if (!param.key) return RAD_E_INVAL;
        for (int previous=0;previous<index;++previous)
            if (!std::strcmp(param.key,args->p[previous].key)) return RAD_E_INVAL;
        const RadParamSpec* spec=nullptr;
        for (int entry=0;entry<schema->n_params;++entry)
            if (!std::strcmp(param.key,schema->params[entry].key)) spec=&schema->params[entry];
        if (!spec) return RAD_E_UNSUPPORTED;
        if (param.kind!=spec->type && !(param.kind==RAD_P_RANGE && spec->type==RAD_P_INT)) return RAD_E_INVAL;
        if ((param.kind==RAD_P_STR && !param.sval) || (param.kind==RAD_P_F64 && !std::isfinite(param.dval))
            || (param.kind==RAD_P_RANGE && (param.ival<1 || param.ihi<param.ival))) return RAD_E_INVAL;
    }
    for (int entry=0;entry<schema->n_params;++entry) {
        if (schema->params[entry].required!=RAD_REQUIRED) continue;
        bool found=false;
        for (int index=0;index<args->n_p;++index) if (!std::strcmp(args->p[index].key,schema->params[entry].key)) found=true;
        if (!found) return RAD_E_INVAL;
    }
    if (paged) {
        const char* act=rad_args_gets(args,"act");
        if (rad_args_geti_or(args,"causal",0)!=1 || rad_args_geti_or(args,"window",0)!=0
            || rad_args_geti_or(args,"group",0)!=128 || !act || std::strcmp(act,"sigmoid")) return RAD_E_UNSUPPORTED;
        for (int index=0;index<args->n_p;++index) if (!std::strcmp(args->p[index].key,"scale") && args->p[index].dval<=0) return RAD_E_INVAL;
    } else {
        for (int index=0;index<args->n_p;++index) if (!std::strcmp(args->p[index].key,"eps") && args->p[index].dval<=0) return RAD_E_INVAL;
        if (rad_args_geti_or(args,"pos_i64",0)!=0) return RAD_E_UNSUPPORTED;
        for (int index=0;index<args->n_p;++index) if ((!std::strcmp(args->p[index].key,"k_scale")
            || !std::strcmp(args->p[index].key,"v_scale")) && args->p[index].dval!=1.0) return RAD_E_UNSUPPORTED;
    }
    return RAD_OK;
}
}
extern "C" int r3dh_attn_fused_workspace_bytes(const RadArgs* args, int paged, uint64_t* bytes) {
    if (!args || !bytes || args->n_p < 0 || (args->n_p && !args->p) || (paged != 0 && paged != 1))
        return RAD_E_INVAL;
    *bytes = 0;
    const int validation=validate(args,paged);
    if (validation) return validation;
    const int64_t rows = rad_param_getdim(args->p, args->n_p, paged ? "q_len" : "M", 0);
    if (rows < 1 || rows > (paged ? 32 : 2048) || rad_args_geti_or(args, "head_dim", 0) != 256
        || rad_args_geti_or(args, "n_head", 0) != 12 || rad_args_geti_or(args, "block_size", 0) != 4)
        return RAD_E_UNSUPPORTED;
    const char* dtype = rad_args_gets(args, "kv_dtype");
    const char* query = rad_args_gets(args, "q_dtype");
    if (!dtype || std::strcmp(dtype, "i8") || !query || std::strcmp(query, "bf16"))
        return RAD_E_UNSUPPORTED;
    if (paged) {
        const int64_t max_seqs = rad_param_getdim(args->p, args->n_p, "max_seqs", 0);
        if (rad_args_geti_or(args, "gqa", 0) != 12 || (max_seqs != 256 && max_seqs != 2048)
            || rad_param_getdim(args->p, args->n_p, "max_ctx", 0) < 1
            || rad_param_getdim(args->p, args->n_p, "max_ctx", 0) > 200000) return RAD_E_UNSUPPORTED;
    } else if (rad_args_geti_or(args, "n_head_kv", 0) != 1
        || rad_args_geti_or(args, "rot", 0) != 64 || rad_args_geti_or(args, "cs_f32", 0) != 1)
        return RAD_E_UNSUPPORTED;
    *bytes = uint64_t(paged ? std::min<int64_t>(2048, rows * rad_param_getdim(args->p, args->n_p, "max_seqs", 0)) : rows) * (paged ? 6144 : 6656) * sizeof(uint16_t);
    return RAD_OK;
}
