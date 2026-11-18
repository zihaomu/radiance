size_t long_samples=0,long_input_samples=0,long_selection_values=0;
int long_cases=0,long_rejects=0;
void fused_long_host() {
    for (int context:{8192,32768,131072,200000}) for (int upper:{1,3,32}) {
        auto params=fused_params(upper,true);
        for (auto& param:params) if (!std::strcmp(param.key,"max_ctx")) param.ival=context;
        RadArgs args{};args.p=params.data();args.n_p=params.size();uint64_t bytes=0;
        insist(r3dh_attn_fused_workspace_bytes(&args,1,&bytes)==RAD_OK,"long context admitted");
        insist(bytes==uint64_t(std::min(2048,upper*256))*12288,"long scratch independent of context");
        RadOpdDesc desc{};
        insist(r3dh_attn_fused_kernel_at(1)->opd_shape(params.data(),params.size(),1,&desc)==RAD_OK
            && desc.shape[0]==(context+3)/4,"long shape physical cache pages");
        params[0]=RAD_RANGE("q_len",1,upper);
        insist(r3dh_attn_fused_kernel_at(1)->scratch(&args)==int64_t(bytes),"long ranged upper scratch");
        params[0]=RAD_INT("q_len",33);
        insist(r3dh_attn_fused_workspace_bytes(&args,1,&bytes)==RAD_E_UNSUPPORTED,"long upper M33 rejected");
        ++long_rejects;
    }
    auto params=fused_params(1,true);
    for (auto& param:params) if (!std::strcmp(param.key,"max_ctx")) param.ival=200000;
    std::vector<RadTensor> tensors;
    for (int operand=0;operand<11;++operand) {
        RadOpdDesc desc{};
        insist(r3dh_attn_fused_kernel_at(1)->opd_shape(params.data(),params.size(),operand,&desc)==RAD_OK,"long reject shape");
        RadTensor value{};value.rank=desc.rank;value.dtype=desc.dtype;
        if (desc.rank) value.data=reinterpret_cast<void*>(uintptr_t(0x100000000ull)+uintptr_t(operand)*0x10000000ull);
        int64_t stride=1;
        for (int axis=int(desc.rank)-1;axis>=0;--axis) {value.shape[axis]=desc.shape[axis];value.stride[axis]=stride;stride*=desc.shape[axis];}
        tensors.push_back(value);
    }
    RadArgs args{};args.p=params.data();args.n_p=params.size();args.n_t=tensors.size();
    args.scratch=reinterpret_cast<void*>(uintptr_t(0x400000000ull));
    uint64_t bytes=0;insist(r3dh_attn_fused_workspace_bytes(&args,1,&bytes)==RAD_OK,"long reject scratch");args.scratch_bytes=bytes;
    auto reject=[&](std::vector<RadTensor>& changed,int expected,const char* label) {
        args.t=changed.data();insist(r3dh_attn_fused_paged_gate(&args,nullptr)==expected,label);++long_rejects;
    };
    auto changed=tensors;changed[1].shape[0]=r3dh_attn_qsa_max_cache_pages+1;reject(changed,RAD_E_SHAPE,"oversized physical cache rejected before enqueue");
    changed=tensors;changed[1].shape[0]=0;reject(changed,RAD_E_SHAPE,"empty cache rejected");
    changed=tensors;changed[2].shape[1]=514;changed[2].stride[0]=514;
    reject(changed,RAD_E_UNSUPPORTED,"selected514 no dense fallback");
    changed=tensors;changed[2].shape[1]=50001;changed[2].stride[0]=50001;
    changed[6]=tensors[3];changed[6].shape[0]=2;
    reject(changed,RAD_E_UNSUPPORTED,"ordinary50001 rejected");
    for (auto& param:params) if (!std::strcmp(param.key,"max_ctx")) param.ival=200001;
    insist(r3dh_attn_fused_workspace_bytes(&args,1,&bytes)==RAD_E_UNSUPPORTED,"context200001 rejected");++long_rejects;
    RadTensor select_tensors[]{local_tensor(tensors[0].data,RAD_F32,3,50000),
        local_tensor(tensors[1].data,RAD_I32,1,50000),tensors[3],tensors[3],tensors[3],
        local_tensor(tensors[7].data,RAD_I32,3,513),tensors[3],tensors[3]};
    for (int operand:{2,3,4,6,7}) select_tensors[operand].data=reinterpret_cast<void*>(uintptr_t(0x500000000ull)+uintptr_t(operand)*0x10000000ull);
    select_tensors[3].shape[0]=3;select_tensors[4].shape[0]=2;
    select_tensors[6].shape[0]=3;select_tensors[7].shape[0]=3;
    for (int sign:{0,1}) {
        const RadParam select_params[]{RAD_INT("ratio",4),RAD_INT("topk",512),RAD_INT("signed",sign)};
        RadArgs select_args{};select_args.t=select_tensors;select_args.n_t=8;select_args.p=select_params;select_args.n_p=3;
        insist(r3dh_sparse_qsa_shape("qsa_select",&select_args)==RAD_OK,"underlying select width50000 admitted");
        select_tensors[0].shape[1]=50001;select_tensors[0].stride[0]=50001;
        insist(r3dh_sparse_qsa_shape("qsa_select",&select_args)==RAD_E_SHAPE,"underlying width50001 rejected");++long_rejects;
        select_tensors[0].shape[1]=50000;select_tensors[0].stride[0]=50000;
    }
    std::printf("ATTN_FUSED_LONG_HOST_PASS geometries=12 exact_rejects=%d max_ctx=200000 cache_pages=50000 cache_bytes=103200000 selector_score_width=50000 selected_complete=512 tail_pages=1 M_upper=32 R_cap=2048\n",long_rejects);
}
void long_read(Blob& blob,std::vector<uint8_t>& actual) {
    actual.resize(blob.bytes.size());hipcheck(hipMemcpy(actual.data(),blob.device,actual.size(),hipMemcpyDeviceToHost));
    for (size_t offset=0;offset<64;++offset) {
        insist(actual[offset]==0xa5 && actual[actual.size()-64+offset]==0xa5,"long allocation guards");
    }
}
void long_input(Blob& blob) {
    uint8_t sample[64];
    for (size_t offset:{size_t(0),blob.bytes.size()-64}) {
        hipcheck(hipMemcpy(sample,blob.device+offset,64,hipMemcpyDeviceToHost));
        for (int index=0;index<64;++index) {insist(sample[index]==blob.bytes[offset+index],"long input guards");++long_input_samples;}
    }
    const size_t payload=blob.bytes.size()-128;
    for (size_t index=0;index<64;++index) {
        const size_t offset=64+(payload-1)*index/63;
        hipcheck(hipMemcpy(sample,blob.device+offset,1,hipMemcpyDeviceToHost));
        insist(sample[0]==blob.bytes[offset],"long unchanged input sample");++long_input_samples;
    }
}
void fused_long_case(int rows,int context,int mode) {
    const int pages=context/4,width=mode<2 ? 513 : mode==2 ? 513 : pages;
    Blob query(rows*3072,RAD_BF16),gate(rows*3072,RAD_BF16),output(rows*3072,RAD_BF16);
    Blob codes(rows*3072,RAD_I8),scales(rows*24,RAD_F32),cache(pages*2064,RAD_I8);
    Blob table(pages,RAD_I32),used(1,RAD_I32),cu(2,RAD_I32),score(rows*pages,RAD_F32);
    Blob count(1,RAD_I32),position(rows,RAD_I32),selected(rows*513,RAD_I32),nselected(rows,RAD_I32),sused(rows,RAD_I32);
    auto params=fused_params(rows,true);
    for (auto& param:params) if (!std::strcmp(param.key,"max_ctx")) param.ival=context;
    RadArgs args{};args.p=params.data();args.n_p=params.size();uint64_t needed=0;
    insist(r3dh_attn_fused_workspace_bytes(&args,1,&needed)==RAD_OK,"long workspace");
    Blob scratch((needed+1)/2,RAD_BF16);
    for (int repeat=0;repeat<2;++repeat) {
        std::fill_n(query.data<uint16_t>(),rows*3072,uint16_t(0));fill(gate,37+repeat*19);
        for (int row=0;row<rows;++row) for (int head=0;head<12;++head) {
            query.data<uint16_t>()[row*3072+head*256]=rad_f32_to_bf16(0.125f*(1+row+repeat));
            query.data<uint16_t>()[row*3072+head*256+255]=rad_f32_to_bf16(-0.25f*(1+head%3));
        }
        for (int page=0;page<pages;++page) {
            table.data<int>()[page]=(page*7+repeat)%pages;
            for (int token=0;token<4;++token) {
                auto* data=cache.data<int8_t>()+page*2064+token*516;
                for (int column=0;column<512;++column) data[column]=int8_t((page*13+token*7+column*3+repeat*11)%47-23);
                const uint16_t key_scale=uint16_t(0x2000+repeat*0x400),value_scale=uint16_t(0x2400+repeat*0x400);
                std::memcpy(data+512,&key_scale,2);std::memcpy(data+514,&value_scale,2);
            }
        }
        cu.data<int>()[0]=0;cu.data<int>()[1]=rows;count.data<int>()[0]=pages;used.data<int>()[0]=mode==2 ? 2051 : context;
        for (int row=0;row<rows;++row) {
            position.data<int>()[row]=context-rows+row;
            for (int page=0;page<pages;++page) score.data<float>()[row*pages+page]=float((page*13+row*7+repeat*3)%47-(repeat ? 47 : 30));
        }
        if (mode==2) for (int page=0;page<width;++page) table.data<int>()[page]=pages-1-(page*11+repeat)%pages;
        for (auto* blob:allocations) blob->push();
        if (mode<2) {
            for (int row=0;row<rows;++row) {
                const int complete=(position.data<int>()[row]+1)/4;
                std::vector<float> ranked;
                for (int page=0;page<complete;++page) ranked.push_back(mode ? score.data<float>()[row*pages+page] : std::max(0.0f,score.data<float>()[row*pages+page]));
                std::sort(ranked.begin(),ranked.end(),std::greater<float>());const float threshold=ranked[511];
                int written=0;
                for (int pass=0;pass<2;++pass) for (int page=0;page<complete && written<512;++page) {
                    const float value=mode ? score.data<float>()[row*pages+page] : std::max(0.0f,score.data<float>()[row*pages+page]);
                    if (pass ? value==threshold : value>threshold) selected.data<int>()[row*513+written++]=table.data<int>()[page];
                }
                if ((position.data<int>()[row]+1)%4) selected.data<int>()[row*513+written++]=table.data<int>()[complete];
                std::fill(selected.data<int>()+row*513+written,selected.data<int>()+(row+1)*513,-1);
                nselected.data<int>()[row]=written;sused.data<int>()[row]=(written-1)*4+position.data<int>()[row]%4+1;
            }
            const RadParam select_params[]{RAD_INT("ratio",4),RAD_INT("topk",512),RAD_INT("signed",mode)};
            RadTensor select_tensors[]{tensor(score,{rows,pages}),tensor(table,{1,pages}),tensor(count,{1}),tensor(position,{rows}),tensor(cu,{2}),tensor(selected,{rows,513}),tensor(nselected,{rows}),tensor(sused,{rows})};
            for (auto& value:select_tensors) for (auto* blob:allocations) if (value.data==blob->data<void>()) {value.data=blob->device+64;break;}
            RadArgs select_args{};select_args.p=select_params;select_args.n_p=3;select_args.t=select_tensors;select_args.n_t=8;
            insist(r3dh_sparse_qsa_select(&select_args,reinterpret_cast<RadStream>(stream))==RAD_OK,"long native selection");++device_calls;
            hipcheck(hipStreamSynchronize(stream));
            for (auto* blob:{&selected,&nselected,&sused}) blob->compare("long_independent_signed_ties",false);
            long_selection_values+=rows*515;
        }
        std::vector<RadTensor> tensors{tensor(query,{rows,3072}),tensor(cache,{pages,1,4,516}),mode<2 ? tensor(selected,{rows,513}) : tensor(table,{1,width}),mode<2 ? tensor(sused,{rows}) : tensor(used,{1}),{},{},mode<2 ? RadTensor{} : tensor(cu,{2}),tensor(output,{rows,3072}),tensor(gate,{rows,3072}),tensor(codes,{rows,3072}),tensor(scales,{rows,24})};
        for (auto& value:tensors) if (value.data) for (auto* blob:allocations) if (value.data==blob->data<void>()) {value.data=blob->device+64;break;}
        args.t=tensors.data();args.n_t=tensors.size();args.scratch=scratch.device+64;args.scratch_bytes=needed;
        insist(r3dh_attn_fused_paged_gate(&args,reinterpret_cast<RadStream>(stream))==RAD_OK,"long fused launch");++device_calls;
        hipcheck(hipStreamSynchronize(stream));
        std::vector<uint8_t> actual_output,actual_codes,actual_scales,actual_scratch;
        long_read(output,actual_output);long_read(codes,actual_codes);long_read(scales,actual_scales);long_read(scratch,actual_scratch);
        for (auto* blob:{&query,&gate,&cache,&table,&used,&cu,&score,&count,&position}) long_input(*blob);
        for (int row=0;row<rows;++row) {
            const int head=(row*5+repeat*3)%12,half=(row+repeat)%2,base=head*256+half*128;
            const int visible=mode<2 ? sused.data<int>()[row] : used.data<int>()[0]-rows+row+1;
            auto token_data=[&](int token) {
                const int page=mode<2 ? selected.data<int>()[row*513+token/4] : table.data<int>()[token/4];
                insist(page>=0 && page<pages,"long oracle physical page");return cache.data<int8_t>()+page*2064+(token%4)*516;
            };
            auto score_at=[&](int token) {
                const auto* data=token_data(token);const float scale=std::ldexp(1.0f,-7+repeat);
                float dot=rad_bf16_to_f32(query.data<uint16_t>()[row*3072+head*256])*(float(data[0])*scale);
                dot+=rad_bf16_to_f32(query.data<uint16_t>()[row*3072+head*256+255])*(float(data[255])*scale);
                return dot*0.0625f;
            };
            float maximum=-INFINITY;for (int token=0;token<visible;++token) maximum=std::max(maximum,score_at(token));
            float denominator=0,accumulator[128]{};
            for (int token=0;token<visible;++token) {
                const float probability=float(std::exp(double(score_at(token)-maximum)));denominator+=probability;
                const auto* data=token_data(token);
                for (int lane=0;lane<128;++lane) accumulator[lane]+=probability*(float(data[256+half*128+lane])*std::ldexp(1.0f,-6+repeat));
            }
            float transformed[128];const float inverse=1.0f/denominator;
            for (int lane=0;lane<128;++lane) {
                const uint16_t attended=rad_f32_to_bf16(accumulator[lane]*inverse);
                const float gated=rad_bf16_to_f32(gate.data<uint16_t>()[row*3072+base+lane]);
                const float sigmoid=rad_bf16_to_f32(rad_f32_to_bf16(1.0f/(1.0f+std::exp(-gated))));
                const uint16_t wanted=rad_f32_to_bf16(rad_bf16_to_f32(attended)*sigmoid);
                uint16_t observed=0;std::memcpy(&observed,actual_output.data()+64+(row*3072+base+lane)*2,2);
                if (observed!=wanted) std::fprintf(stderr,"LONG_MISMATCH ctx=%d rows=%d mode=%d repeat=%d row=%d lane=%d expected=%04x actual=%04x\n",context,rows,mode,repeat,row,lane,wanted,observed);
                insist(observed==wanted,"long independent sampled BF16 product");++long_samples;
                transformed[lane]=rad_bf16_to_f32(wanted);
            }
            for (int distance=1;distance<128;distance*=2) for (int block=0;block<128;block+=2*distance) for (int lane=0;lane<distance;++lane) {
                const float first=transformed[block+lane],second=transformed[block+lane+distance];
                transformed[block+lane]=first+second;transformed[block+lane+distance]=first-second;
            }
            float maximum_abs=0;for (float value:transformed) maximum_abs=std::max(maximum_abs,std::fabs(value));
            const float scale=maximum_abs>0 ? maximum_abs*(1.0f/127.0f) : 1.0f;
            float actual_scale=0;std::memcpy(&actual_scale,actual_scales.data()+64+(row*24+base/128)*4,4);
            insist(std::memcmp(&scale,&actual_scale,4)==0,"long independent sampled quant scale");++long_samples;
            for (int lane=0;lane<128;++lane) {
                const int8_t wanted=int8_t(std::nearbyint(std::clamp(transformed[lane]*(1.0f/scale),-127.0f,127.0f)));
                insist(int8_t(actual_codes[64+row*3072+base+lane])==wanted,"long independent sampled Hadamard I8");++long_samples;
            }
        }
        ++long_cases;
        std::printf("ATTN_FUSED_LONG_CASE context=%d pages=%d R=%d mode=%s repeat=%d sampled_groups=%d selected_width=%d pass=1\n",context,pages,rows,mode==0 ? "relu_ties" : mode==1 ? "signed_ties" : mode==2 ? "shared_union" : "ordinary_dense",repeat,rows,width);
        std::fflush(stdout);
    }
}
void fused_long_gpu() {
    for (int context:{8192,32768,131072,200000}) for (int rows:{1,3}) for (int mode:{0,1,2,3}) fused_long_case(rows,context,mode);
    std::printf("ATTN_FUSED_LONG_PASS cases=%d sampled_values=%zu selector_values=%zu unchanged_input_samples=%zu cpu_full_long_recompute=0 performance=unqualified\n",long_cases,long_samples,long_selection_values,long_input_samples);
}
