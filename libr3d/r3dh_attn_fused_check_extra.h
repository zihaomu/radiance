void fused_rejections(const std::vector<RadParam>& params,const std::vector<RadTensor>& tensors,int paged,uint64_t needed) {
    RadArgs args{}; args.p=params.data();args.n_p=params.size();args.t=tensors.data();args.n_t=tensors.size();
    args.scratch=reinterpret_cast<void*>(0x10000);args.scratch_bytes=needed;
    const auto launch=r3dh_attn_fused_kernel_at(paged)->launch;
    auto reject=[&](int expected,const char* label) {insist(launch(&args,nullptr)==expected,label);};
    args.scratch=nullptr;reject(RAD_E_INVAL,"missing scratch");
    args.scratch=reinterpret_cast<void*>(0x10010);args.scratch_bytes=needed-1;reject(RAD_E_INVAL,"short scratch");
    args.scratch=reinterpret_cast<void*>(0x10001);args.scratch_bytes=needed;reject(RAD_E_INVAL,"misaligned scratch");
    args.scratch=reinterpret_cast<void*>(0x10000);args.n_t=-1;reject(RAD_E_INVAL,"negative tensor count");
    args.n_t=tensors.size();args.t=nullptr;reject(RAD_E_INVAL,"missing tensors");args.t=tensors.data();
    args.n_p=-1;reject(RAD_E_INVAL,"negative param count");args.n_p=params.size();
    auto changed=tensors;args.t=changed.data();changed[0].dtype=RAD_F32;reject(RAD_E_SHAPE,"wrong query dtype");
    changed=tensors;changed[0].rank=5;reject(RAD_E_SHAPE,"invalid query rank");
    changed=tensors;changed[0].stride[1]=2;reject(RAD_E_SHAPE,"invalid query stride");
    changed=tensors;args.scratch=changed[0].data;reject(RAD_E_SHAPE,"scratch aliases operand");
    args.scratch=reinterpret_cast<void*>(0x10000);changed=tensors;
    if (paged) {
        changed[9].data=changed[7].data;reject(RAD_E_SHAPE,"paged outputs overlap");
        changed=tensors;changed[8].rank=1;reject(RAD_E_SHAPE,"gate rejected before enqueue");
        changed=tensors;changed[4]=changed[10];reject(RAD_E_SHAPE,"external descale rejected");
    } else {
        changed[6]=local_tensor(changed[0].data,RAD_BF16,tensors[0].shape[0],3072);changed[8]=changed[6];
        reject(RAD_E_SHAPE,"prepare outputs overlap");
        changed=tensors;changed[7].data=changed[11].data;reject(RAD_E_SHAPE,"key output aliases cache");
        changed=tensors;changed[4].data=nullptr;reject(RAD_E_SHAPE,"required query weight");
        changed=tensors;changed[0].data=reinterpret_cast<void*>(UINTPTR_MAX-15);
        reject(RAD_E_SHAPE,"prepare operand pointer wrap");
        changed=tensors;args.scratch=reinterpret_cast<void*>(UINTPTR_MAX-15);
        reject(RAD_E_SHAPE,"prepare scratch pointer wrap");
        args.scratch=reinterpret_cast<void*>(0x10000);changed=tensors;
        changed[7].shape[0]=tensors[0].shape[0]+1;reject(RAD_E_SHAPE,"prepare output row mismatch");
        changed=tensors;changed[0].stride[0]=(1ll<<30)+1;
        reject(RAD_E_SHAPE,"prepare pitch out of bounds");
    }
}
void fused_query_contracts() {
    for (int paged=0;paged<2;++paged) for (int rows=1;rows<=(paged ? 32 : 2048);++rows) {
        auto params=fused_params(rows,paged);RadArgs args{};args.p=params.data();args.n_p=params.size();
        const auto* kernel=r3dh_attn_fused_kernel_at(paged);
        bool lower=false,upper=false;
        for (int index=0;index<kernel->n_constraints;++index) {
            const auto& constraint=kernel->constraints[index];
            if (std::strcmp(constraint.key,paged ? "q_len" : "M")) continue;
            if (constraint.op==RAD_C_GE) {insist(constraint.ival==1,"registry lower cap");lower=true;}
            if (constraint.op==RAD_C_LE) {insist(constraint.ival==(paged ? 32 : 2048),"registry upper cap");upper=true;}
        }
        insist(lower && upper,"registry exposes both row bounds");
        if (!paged) for (int operand=0;operand<12;++operand) {
            RadOpdDesc shape{};insist(kernel->opd_shape(params.data(),params.size(),operand,&shape)==RAD_OK,"prepare shape throughout admitted M");
            if (operand==0 || operand==1 || operand==3 || (operand>=6 && operand<=10))
                insist(shape.shape[0]==rows,"prepare shape rows match M");
            if (operand==11) insist(shape.shape[0]==(rows+3)/4+2,"prepare cache shape matches M");
        }
        args.t=reinterpret_cast<const RadTensor*>(1);args.n_t=paged ? 11 : 12;
        args.scratch=reinterpret_cast<void*>(3);args.scratch_bytes=-1;
        uint64_t bytes=0;insist(r3dh_attn_fused_workspace_bytes(&args,paged,&bytes)==RAD_OK,"pointer independent scratch");
        insist(r3dh_attn_fused_kernel_at(paged)->scratch(&args)==int64_t(bytes),"RadScratchFn adapter bytes");
        params[0]=paged ? RAD_RANGE("q_len",1,rows) : RAD_RANGE("M",1,rows);
        uint64_t ranged=0;insist(r3dh_attn_fused_workspace_bytes(&args,paged,&ranged)==RAD_OK && ranged==bytes,"upper range scratch");
        RadOpdDesc ranged_shape{};insist(kernel->opd_shape(params.data(),params.size(),paged ? 0 : 7,&ranged_shape)==RAD_OK,"ranged shape admission");
        insist(ranged_shape.shape[0]==rows,"ranged shape uses upper bound");
        params[0].ihi=paged ? 33 : 2049;insist(r3dh_attn_fused_kernel_at(paged)->scratch(&args)==-1,"invalid geometry scratch adapter");
        bytes=UINT64_MAX;insist(r3dh_attn_fused_workspace_bytes(&args,paged,&bytes)==RAD_E_UNSUPPORTED && bytes==0,"oversized range resets workspace");
        insist(kernel->opd_shape(params.data(),params.size(),0,&ranged_shape)==RAD_E_UNSUPPORTED,"oversized range rejects shape");
        insist(kernel->launch(&args,nullptr)==RAD_E_UNSUPPORTED,"oversized range rejects before pointer reads");
        params[0]=paged ? RAD_INT("q_len",33) : RAD_INT("M",2049);
        insist(kernel->launch(&args,nullptr)==RAD_E_UNSUPPORTED,"oversized integer rejects before pointer reads");
        params[0].ival=0;insist(kernel->launch(&args,nullptr)==RAD_E_UNSUPPORTED,"zero rows rejected");
        params[0]=paged ? RAD_INT("q_len",rows) : RAD_INT("M",rows);
        params[1].ival=192;insist(r3dh_attn_fused_workspace_bytes(&args,paged,&bytes)==RAD_E_UNSUPPORTED,"hd192 not claimed");
        RadOpdDesc desc{};
        insist(r3dh_attn_fused_kernel_at(paged)->opd_shape(params.data(),params.size(),-1,&desc)==RAD_E_INVAL,"negative shape index");
    }
    std::printf("ATTN_FUSED_ABI_PASS geometry_cases=2080 adapters=RadScratchFn pointer_independent=1 hd192=unsupported\n");
}
void fused_schema_case(int rows,int paged) {
    auto params=fused_params(rows,paged);
    const auto* kernel=r3dh_attn_fused_kernel_at(paged);
    const auto* schema=r3dh_attn_fused_schema_at(paged);
    std::vector<Blob*> owned;std::vector<RadTensor> tensors;
    for (int operand=0;operand<schema->n_operands;++operand) {
        RadOpdDesc desc{};insist(kernel->opd_shape(params.data(),params.size(),operand,&desc)==RAD_OK,"staged shape");
        RadTensor value{};
        if (desc.rank) {
            size_t count=1;for (int axis=0;axis<desc.rank;++axis) count*=desc.shape[axis];
            auto* blob=new Blob(count,desc.dtype);owned.push_back(blob);
            std::memset(blob->data<void>(),0,blob->bytes.size()-128);
            if (desc.dtype==RAD_BF16) fill(*blob,rows+operand);
            if (desc.dtype==RAD_F32) for (size_t index=0;index<count;++index) blob->data<float>()[index]=0.5f;
            value.data=blob->data<void>();value.dtype=desc.dtype;value.rank=desc.rank;
            int64_t pitch=1;
            for (int axis=desc.rank-1;axis>=0;--axis) {value.shape[axis]=desc.shape[axis];value.stride[axis]=pitch;pitch*=desc.shape[axis];}
            if (desc.dtype==RAD_I32) for (size_t index=0;index<count;++index)
                blob->data<int>()[index]=int(desc.idx_const>=0 ? desc.idx_const : index%desc.idx_max);
        }
        tensors.push_back(value);
    }
    fused_run(paged ? "fused_schema_device_paged" : "fused_schema_device_prepare",paged,params,tensors);
    for (auto* blob:owned) delete blob;
}
