void fused_selected_case(int rows,int context,int sign,bool shared_union) {
    const int pages=(context+3)/4;
    Blob query(rows*3072,RAD_BF16),gate(rows*3072,RAD_BF16),output(rows*3072,RAD_BF16);
    Blob codes(rows*3072,RAD_I8),scales(rows*24,RAD_F32),cache(pages*2064,RAD_I8);
    Blob table(pages,RAD_I32),used(1,RAD_I32),cu(2,RAD_I32),score(rows*pages,RAD_F32);
    Blob count(1,RAD_I32),position(rows,RAD_I32),selected(rows*513,RAD_I32),nselected(rows,RAD_I32),sused(rows,RAD_I32);
    for (int repeat=0;repeat<2;++repeat) {
        fill(query,31+repeat*19);fill(gate,37+repeat*13);
        for (int page=0;page<pages;++page) {
            table.data<int>()[page]=(page*7+repeat)%pages;
            for (int token=0;token<4;++token) {
                auto* data=cache.data<int8_t>()+page*2064+token*516;
                for (int column=0;column<512;++column) data[column]=int8_t((page*13+token*7+column*3+repeat)%47-23);
                const uint16_t scale=uint16_t(0x2000+repeat*0x400);
                std::memcpy(data+512,&scale,2);std::memcpy(data+514,&scale,2);
            }
        }
        cu.data<int>()[0]=0;cu.data<int>()[1]=rows;
        count.data<int>()[0]=context/4;used.data<int>()[0]=context;
        for (int row=0;row<rows;++row) {
            position.data<int>()[row]=context-rows+row;
            for (int page=0;page<pages;++page)
                score.data<float>()[row*pages+page]=float((page*13+row*7+repeat*3)%47-23);
        }
        RadTensor bt{},ut{},cut{};
        if (!shared_union) {
            run("fused_signed_selection",r3dh_sparse_qsa_select,ref_qsa_select,
                {RAD_INT("ratio",4),RAD_INT("topk",512),RAD_INT("signed",sign)},
                {tensor(score,{rows,pages}),tensor(table,{1,pages}),tensor(count,{1}),tensor(position,{rows}),tensor(cu,{2}),
                 tensor(selected,{rows,513}),tensor(nselected,{rows}),tensor(sused,{rows})},
                {&selected,&nselected,&sused});
            fused_resident={&selected,&sused};
            bt=tensor(selected,{rows,513});ut=tensor(sused,{rows});
        } else {
            const int union_pages[]{4,1,7,2,6};
            const int union_tokens=std::max(rows,19),union_count=(union_tokens+3)/4;
            for (int page=0;page<pages;++page)
                table.data<int>()[page]=page<5 ? union_pages[page] : page<union_count ? page+3 : -1;
            used.data<int>()[0]=union_tokens;
            bt=tensor(table,{1,pages});ut=tensor(used,{1});cut=tensor(cu,{2});
        }
        auto params=fused_params(shared_union ? rows : 1,true);
        fused_run(shared_union ? "fused_shared_union" : sign ? "fused_selected_signed1" : "fused_selected_signed0",1,params,
            {tensor(query,{rows,3072}),tensor(cache,{pages,1,4,516}),bt,ut,{},{},cut,
             tensor(output,{rows,3072}),tensor(gate,{rows,3072}),tensor(codes,{rows,3072}),tensor(scales,{rows,24})});
        fused_resident.clear();
    }
}
