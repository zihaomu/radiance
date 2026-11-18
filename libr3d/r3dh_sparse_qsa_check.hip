#include <hip/hip_runtime.h>
#include "r3dh_sparse_qsa.h"
#include "r3dh_attn_qsa.h"
#include "rad_plugin.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
extern "C" int ref_qsa_block_key(const RadArgs*,RadStream);
extern "C" int ref_qsa_score(const RadArgs*,RadStream);
extern "C" int ref_qsa_select(const RadArgs*,RadStream);
extern "C" int ref_qsa_work(const RadArgs*,RadStream);
extern "C" int ref_qsa_tail_store(const RadArgs*,RadStream);
extern "C" int ref_attn_paged(const RadArgs*,RadStream);
namespace {
bool gpu=false;
hipStream_t stream=nullptr;
size_t checks=0, comparisons=0, differing=0, launches=0, cases=0;
void require(bool value,const char* message) { ++checks; if (!value) throw std::runtime_error(message); }
void hipcheck(hipError_t status) { if (status!=hipSuccess) throw std::runtime_error(hipGetErrorString(status)); }
struct Blob;
std::vector<Blob*> allocations;
struct Blob {
    std::vector<unsigned char> bytes;
    unsigned char* device=nullptr;
    unsigned dtype;
    bool resident=false, uploaded=false;
    Blob(size_t count,unsigned type) : bytes(count*(type==RAD_BF16?2:type==RAD_I8?1:4)+128,0xa5),dtype(type) {
        if (gpu) hipcheck(hipMalloc(&device,bytes.size()));
        allocations.push_back(this);
    }
    ~Blob() { allocations.erase(std::find(allocations.begin(),allocations.end(),this)); if (device) (void)hipFree(device); }
    template<class Value> Value* data() { return reinterpret_cast<Value*>(bytes.data()+64); }
    void push() { if (gpu && (!resident || !uploaded)) { hipcheck(hipMemcpyAsync(device,bytes.data(),bytes.size(),hipMemcpyHostToDevice,stream)); uploaded=true; } }
    void compare(const char* label,bool typed=false) {
        if (!gpu) return;
        std::vector<unsigned char> actual(bytes.size());
        hipcheck(hipMemcpy(actual.data(),device,actual.size(),hipMemcpyDeviceToHost));
        size_t local_differences=0;
        for (size_t offset=0; offset<bytes.size();) {
            ++comparisons;
            const bool payload=offset>=64 && offset<bytes.size()-64;
            const int size=payload && typed ? dtype==RAD_F32?4:dtype==RAD_BF16?2:1 : 1;
            bool equal=!std::memcmp(bytes.data()+offset,actual.data()+offset,size), valid=equal;
            if (!valid) { std::fprintf(stderr,"FAIL %s offset=%zu expected=%02x observed=%02x\n",label,offset,bytes[offset],actual[offset]); throw std::runtime_error(label); }
            if (!equal) ++local_differences;
            offset+=size;
        }
        differing+=local_differences;
        std::printf("COMPARE %s differences=%zu guards=exact\n",label,local_differences);
    }
};
RadTensor tensor(Blob& blob,std::initializer_list<int64_t> shape,int64_t pitch=0) {
    RadTensor value{}; value.data=blob.data<void>(); value.dtype=blob.dtype; value.rank=shape.size();
    int index=0; for (int64_t extent:shape) value.shape[index++]=extent;
    int64_t stride=1;
    for (index=int(value.rank)-1;index>=0;--index) { value.stride[index]=stride; stride*=value.shape[index]; }
    if (pitch) value.stride[0]=pitch;
    return value;
}
using Function=int(*)(const RadArgs*,RadStream);
void run(const char* label,Function implementation,Function oracle,std::vector<RadParam> params,
         std::vector<RadTensor> operands,std::initializer_list<Blob*> outputs,bool typed=false) {
    for (Blob* blob:allocations) blob->push();
    std::vector<RadTensor> remote=operands;
    for (auto& value:remote) if (value.data) {
        bool mapped=false;
        for (Blob* blob:allocations) {
            const auto pointer=reinterpret_cast<uintptr_t>(value.data),begin=reinterpret_cast<uintptr_t>(blob->data<void>());
            if (pointer>=begin && pointer<begin+blob->bytes.size()-128) {
                value.data=gpu ? blob->device+64+(pointer-begin) : value.data; mapped=true; break;
            }
        }
        require(mapped,"unmapped tensor");
    }
    RadArgs args{}; args.p=params.data(); args.n_p=params.size(); args.t=operands.data(); args.n_t=operands.size();
    const char* op=implementation==r3dh_sparse_qsa_block_key?"qsa_block_key":implementation==r3dh_sparse_qsa_score?"qsa_score":
        implementation==r3dh_sparse_qsa_select?"qsa_select":implementation==r3dh_sparse_qsa_paged?"attn_paged":nullptr;
    if (op) {
        require(r3dh_sparse_qsa_shape(op,&args)==RAD_OK,"fixture host descriptor validation");
        const auto* schema=r3dh_sparse_qsa_schema_at(implementation==r3dh_sparse_qsa_block_key?0:implementation==r3dh_sparse_qsa_score?1:implementation==r3dh_sparse_qsa_select?2:3);
        for (int index=0;index<args.n_t;++index) if (!schema->operands[index].optional) {
            void* saved=operands[index].data;operands[index].data=nullptr;
            require(r3dh_sparse_qsa_shape(op,&args)!=RAD_OK,"missing required operand");operands[index].data=saved;
            const auto saved_type=operands[index].dtype;operands[index].dtype=RAD_I64;
            require(r3dh_sparse_qsa_shape(op,&args)!=RAD_OK,"wrong operand dtype");operands[index].dtype=saved_type;
        }
    }
    require(oracle(&args,nullptr)==RAD_OK,label);
    if (gpu) {
        args.t=remote.data(); require(implementation(&args,reinterpret_cast<RadStream>(stream))==RAD_OK,label);
        ++launches; hipcheck(hipStreamSynchronize(stream));
        for (Blob* blob:allocations) blob->compare(label,typed && std::find(outputs.begin(),outputs.end(),blob)!=outputs.end());
    }
    ++cases; std::printf("PASS %s\n",label);
}
void fill_bf(Blob& blob,size_t count,int seed) {
    for (size_t index=0;index<count;++index) blob.data<uint16_t>()[index]=rad_f32_to_bf16(float(int((index*17+seed*29)%101)-50)/64.0f);
}
void selection_fixture(int width,int topk,int signed_score) {
    const int rows=8,sequences=3;
    Blob scores(rows*(width+3),RAD_F32),table(sequences*(width+2),RAD_I32),count(sequences,RAD_I32),positions(rows,RAD_I32),cu(4,RAD_I32);
    Blob selected(rows*(topk+1),RAD_I32),nselected(rows,RAD_I32),used(rows,RAD_I32);
    int offsets[]={1,3,8,8}; std::memcpy(cu.data<void>(),offsets,sizeof(offsets));
    for (int sequence=0;sequence<sequences;++sequence) {
        count.data<int>()[sequence]=width;
        for (int column=0;column<width;++column) table.data<int>()[sequence*(width+2)+column]=(column*7+sequence*13)%(width+19);
    }
    const int boundary[]={0,3,7,2048,2049,2050,2051,199999};
    for (int repeat=0;repeat<2;++repeat) {
        for (int row=0;row<rows;++row) {
            positions.data<int>()[row]=std::min(boundary[row],width*4-1);
            for (int column=0;column<width;++column) {
                float value=float((column*5+row+repeat*3)%17-8);
                if (column%37==0) value=-INFINITY;
                if (column%41==0) value=INFINITY;
                if (column%43==0) value=NAN;
                if (column%11==0) value=column%2 ? -0.0f:0.0f;
                scores.data<float>()[row*(width+3)+column]=value;
            }
        }
        if (width>4) table.data<int>()[2]=-1;
        std::vector<std::vector<int>> expected(rows);
        for (int row=0;row<rows;++row) {
            const int sequence=row<3?0:1,complete=std::min(width,(positions.data<int>()[row]+1)/4);
            std::vector<std::pair<float,int>> order;
            for (int column=0;column<complete;++column) {
                float value=scores.data<float>()[row*(width+3)+column];
                if (table.data<int>()[sequence*(width+2)+column]<0 || (signed_score ? !std::isfinite(value):!(value>-1e30f))) continue;
                order.push_back({signed_score?value:std::max(value,0.0f),column});
            }
            const int live=order.size();
            std::stable_sort(order.begin(),order.end(),[](auto left,auto right){return left.first>right.first;});
            const float cutoff=live>topk?order[topk-1].first:-INFINITY;
            for (int pass=0;pass<(live>topk?2:1);++pass)
                for (int column=0;column<complete && int(expected[row].size())<topk;++column) {
                    float value=scores.data<float>()[row*(width+3)+column];
                    if (table.data<int>()[sequence*(width+2)+column]<0 || (signed_score?!std::isfinite(value):!(value>-1e30f))) continue;
                    if (!signed_score) value=std::max(value,0.0f);
                    if (live<=topk || (pass==0?value>cutoff:value==cutoff)) expected[row].push_back(table.data<int>()[sequence*(width+2)+column]);
                }
            if (complete<width && (positions.data<int>()[row]+1)%4 && table.data<int>()[sequence*(width+2)+complete]>=0)
                expected[row].push_back(table.data<int>()[sequence*(width+2)+complete]);
        }
        run("select_independent_ties_positions",r3dh_sparse_qsa_select,ref_qsa_select,
            {RAD_INT("ratio",4),RAD_INT("topk",topk),RAD_INT("signed",signed_score)},
            {tensor(scores,{rows,width},width+3),tensor(table,{sequences,width},width+2),tensor(count,{sequences}),tensor(positions,{rows}),
             tensor(cu,{4}),tensor(selected,{rows,topk+1}),tensor(nselected,{rows}),tensor(used,{rows})},{&selected,&nselected,&used});
        for (int row=0;row<rows;++row) {
            require(nselected.data<int>()[row]==int(expected[row].size()),"independent nsel");
            for (int column=0;column<=topk;++column)
                require(selected.data<int>()[row*(topk+1)+column]==(column<int(expected[row].size())?expected[row][column]:-1),"independent page id/order");
        }
    }
}
void key_score_fixture(int width,const char* mode,bool list) {
    const int blocks=5,pages=11,rows=4;
    Blob raw(blocks*4*132,RAD_BF16),weights(128,RAD_F32),page(blocks,RAD_I32),pos(blocks*3,RAD_I32),cache(pages*128,RAD_BF16);
    Blob query(rows*520,RAD_BF16),table(width+2,RAD_I32),count(1,RAD_I32),cu(2,RAD_I32),scores(rows*(width+2),RAD_F32);
    for (int lane=0;lane<128;++lane) weights.data<float>()[lane]=float(lane%7-3)/16;
    for (int block=0;block<blocks;++block) {
        page.data<int>()[block]=list?(block==2?-1:(block*3)%pages):block;
        for (int component=0;component<3;++component) pos.data<int>()[block*3+component]=block*4+component*101+199600;
    }
    for (int column=0;column<width;++column) table.data<int>()[column]=column%blocks;
    table.data<int>()[1]=-1; count.data<int>()[0]=width-1; cu.data<int>()[0]=1; cu.data<int>()[1]=rows;
    for (int repeat=0;repeat<2;++repeat) {
        fill_bf(raw,blocks*4*132,repeat); fill_bf(query,rows*520,repeat+2); fill_bf(cache,pages*128,repeat+4);
        std::vector<RadParam> params={RAD_INT("n",128),RAD_INT("ratio",4),RAD_INT("rotary_dim",64),RAD_INT("pos0",199600),
            RAD_F64("theta",1e7),RAD_F64("eps",1e-6),RAD_F64("wadd",1),RAD_STR("mode",mode),RAD_STR("sections","11 11 10")};
        RadTensor output=list?tensor(cache,{pages,1,4,32}):tensor(cache,{blocks,128});
        RadTensor positions=list?tensor(pos,{blocks,3}):RadTensor{};
        run("block_key_pool_norm_rope_pages",r3dh_sparse_qsa_block_key,ref_qsa_block_key,params,
            {tensor(raw,{blocks*4,128},132),tensor(weights,{128}),list?tensor(page,{blocks}):RadTensor{},positions,output},{&cache},true);
        cache.resident=true;
        run("score_shuffled_bkey_map_padding",r3dh_sparse_qsa_score,ref_qsa_score,{RAD_INT("n",128),RAD_INT("heads",4)},
            {tensor(query,{rows,512},520),tensor(cache,{pages,1,4,32}),tensor(table,{1,width}),tensor(cu,{2}),tensor(count,{1}),
             tensor(scores,{rows,width},width+2)},{&scores},true);
        cache.resident=false;
    }
}
void attention_fixture(int context,bool i8,bool selected_mode,bool signed_score,bool union_mode,int dim=256) {
    const int rows=3,heads=12,width=(context+3)/4,pages=width+7,topk=512;
    const int cache_width=2*dim+(i8?4:0);
    Blob query(rows*(heads*dim+8),RAD_BF16),cache(pages*4*cache_width,i8?RAD_I8:RAD_BF16),table(width,RAD_I32),used(1,RAD_I32),cu(2,RAD_I32);
    Blob scores(rows*width,RAD_F32),count(1,RAD_I32),pos(rows,RAD_I32),selected(rows*(topk+1),RAD_I32),nselected(rows,RAD_I32),sused(rows,RAD_I32);
    Blob output(rows*(heads*dim+8),RAD_BF16),full(rows*(heads*dim+8),RAD_BF16);
    used.data<int>()[0]=context; cu.data<int>()[0]=0; cu.data<int>()[1]=rows; count.data<int>()[0]=context/4;
    for (int column=0;column<width;++column) table.data<int>()[column]=(column*17+5)%pages;
    if (union_mode) {
        used.data<int>()[0]=19;
        const int union_pages[]={4,1,7,2,6};
        for (int column=0;column<5;++column) table.data<int>()[column]=union_pages[column];
        for (int column=5;column<width;++column) table.data<int>()[column]=-1;
    }
    for (int repeat=0;repeat<2;++repeat) {
        fill_bf(query,rows*(heads*dim+8),repeat+3);
        if (i8) {
            for (int row=0;row<pages*4;++row) {
                for (int lane=0;lane<2*dim;++lane) cache.data<signed char>()[row*cache_width+lane]=((row*7+lane*11+repeat)%127)-63;
                const uint16_t scales[2]={uint16_t(0x2000+repeat*0x400),0x2400};
                std::memcpy(cache.data<unsigned char>()+row*cache_width+2*dim,scales,4);
            }
        } else fill_bf(cache,pages*4*cache_width,repeat+5);
        for (int row=0;row<rows;++row) {
            pos.data<int>()[row]=context-rows+row;
            for (int column=0;column<width;++column) scores.data<float>()[row*width+column]=float((column*13+row*7+repeat)%23)-11;
        }
        const std::vector<RadParam> params={RAD_INT("head_dim",dim),RAD_INT("gqa",heads),RAD_INT("block_size",4),RAD_INT("causal",1),RAD_INT("window",0),RAD_STR("kv_dtype",i8?"i8":"bf16")};
        if (selected_mode) {
            run("selected_signed_page_table",r3dh_sparse_qsa_select,ref_qsa_select,
                {RAD_INT("ratio",4),RAD_INT("topk",topk),RAD_INT("signed",signed_score)},
                {tensor(scores,{rows,width}),tensor(table,{1,width}),tensor(count,{1}),tensor(pos,{rows}),tensor(cu,{2}),
                 tensor(selected,{rows,topk+1}),tensor(nselected,{rows}),tensor(sused,{rows})},{&selected,&nselected,&sused});
            selected.resident=true; sused.resident=true;
        }
        std::vector<RadTensor> operands={tensor(query,{rows,heads*dim},heads*dim+8),tensor(cache,{pages,1,4,cache_width}),
            selected_mode?tensor(selected,{rows,topk+1}):tensor(table,{1,width}),selected_mode?tensor(sused,{rows}):tensor(used,{1}),
            {},{},selected_mode?RadTensor{}:tensor(cu,{2}),tensor(output,{rows,heads*dim},heads*dim+8)};
        if (dim==128) {
            operands[0].rank=operands[7].rank=3;
            for (int operand:{0,7}) {
                operands[operand].shape[1]=heads;
                operands[operand].shape[2]=dim;
                operands[operand].stride[1]=dim;
                operands[operand].stride[2]=1;
            }
        }
        run(union_mode?"union_ragged_attention":selected_mode?"selected_attention_actual_geometry":"dense_equivalence_attention",
            r3dh_sparse_qsa_paged,ref_attn_paged,params,operands,{&output},true);
        if (selected_mode) {
            RadArgs args{}; args.p=params.data(); args.n_p=params.size();
            std::vector<RadTensor> dense={operands[0],operands[1],tensor(table,{1,width}),tensor(used,{1}),{},{},tensor(cu,{2}),tensor(full,{rows,heads*dim},heads*dim+8)};
            args.t=dense.data(); args.n_t=dense.size(); require(ref_attn_paged(&args,nullptr)==RAD_OK,"full dense oracle");
            if (context<=2051) require(output.bytes==full.bytes,"below-bound selected equals full-dense bytes");
            else {
                require(nselected.data<int>()[rows-1]<width,"above-bound selection drops pages");
                require(output.bytes!=full.bytes,"above-bound selected must not equal full dense");
            }
            std::printf("BOUND context=%d dim=%d kv=%s signed=%d selected_pages=%d full_pages=%d relation=%s\n",
                context,dim,i8?"i8":"bf16",signed_score,nselected.data<int>()[rows-1],width,
                context<=2051?"dense_exact":"sparse_not_dense");
        }
        selected.resident=false; sused.resident=false;
    }
}
void rollback_fixture() {
    Blob keys(4*132,RAD_BF16),positions(4,RAD_I32),cu(2,RAD_I32),tail(7*128,RAD_BF16),state_index(1,RAD_I32),table(8,RAD_I32);
    Blob stage(3*4*128,RAD_BF16),pages(3,RAD_I32),bpos(3,RAD_I32),count(1,RAD_I32);
    cu.data<int>()[0]=0; cu.data<int>()[1]=4; state_index.data<int>()[0]=0;
    for (int column=0;column<8;++column) table.data<int>()[column]=7-column;
    for (int position=2;position<=8;++position) for (int lane=0;lane<128;++lane) tail.data<uint16_t>()[position%7*128+lane]=rad_f32_to_bf16(float(position*2+lane%3));
    tail.resident=true;
    const std::vector<RadParam> params={RAD_INT("n",128),RAD_INT("ratio",4),RAD_INT("ring",7)};
    for (int pass=0;pass<2;++pass) {
        const int first=pass?10:9;
        for (int row=0;row<4;++row) {
            positions.data<int>()[row]=first+row;
            for (int lane=0;lane<128;++lane) keys.data<uint16_t>()[row*132+lane]=rad_f32_to_bf16(float((first+row)*2+lane%3+(pass?64:0)));
        }
        run(pass?"rollback_actual_work_10_13":"verify_actual_work_9_12",r3dh_attn_qsa_work,ref_qsa_work,params,
            {tensor(keys,{4,128},132),tensor(positions,{4}),tensor(cu,{2}),tensor(tail,{1,1,7,128}),tensor(state_index,{1}),
             tensor(table,{1,8}),{},tensor(stage,{12,128}),tensor(pages,{3}),tensor(bpos,{3}),tensor(count,{1})},{&stage,&pages,&bpos,&count});
        if (pass) for (int token=0;token<2;++token) for (int lane=0;lane<128;++lane)
            require(stage.data<uint16_t>()[token*128+lane]==rad_f32_to_bf16(float((8+token)*2+lane%3)),"rollback retained raw key8/9");
        run("actual_tail_store_ring7",r3dh_attn_qsa_tail_store,ref_qsa_tail_store,params,
            {tensor(keys,{4,128},132),tensor(positions,{4}),tensor(cu,{2}),tensor(tail,{1,1,7,128}),tensor(state_index,{1})},{&tail});
    }
}
void host_rejects() {
    require(r3dh_sparse_qsa_schema_count()==4,"schema count");
    require(!r3dh_sparse_qsa_schema_at(-1) && !r3dh_sparse_qsa_schema_at(4),"schema bounds");
    RadShapeFn shapes[]={r3dh_sparse_qsa_block_key_shape,r3dh_sparse_qsa_score_shape,r3dh_sparse_qsa_select_shape,r3dh_sparse_qsa_paged_shape};
    for (int index=0;index<4;++index) {
        const auto* schema=r3dh_sparse_qsa_schema_at(index);
        std::vector<RadParam> params={RAD_INT("M",3),RAD_INT("q_len",3),RAD_INT("n",128),RAD_INT("heads",4),RAD_INT("blocks",7),RAD_INT("ratio",4),RAD_INT("head_dim",256)};
        for (int operand=0;operand<schema->n_operands;++operand) {
            RadOpdDesc description{}; require(shapes[index](params.data(),params.size(),operand,&description)==RAD_OK,"named shape descriptor");
        }
        require(r3dh_sparse_qsa_shape(schema->op,nullptr)==RAD_E_INVAL,"null args rejection");
    }
    Blob query(256,RAD_BF16),cache(4*516,RAD_I8),table(1,RAD_I32),used(1,RAD_I32),output(256,RAD_BF16);
    std::vector<RadTensor> values={tensor(query,{1,256}),tensor(cache,{1,1,4,516}),tensor(table,{1,1}),tensor(used,{1}),{},{},{},tensor(output,{1,256})};
    std::vector<RadParam> params={RAD_INT("head_dim",256),RAD_INT("gqa",1),RAD_INT("block_size",4),RAD_INT("causal",1),RAD_INT("window",0)};
    RadArgs args{}; args.p=params.data();args.n_p=params.size();args.t=values.data();args.n_t=values.size();
    require(r3dh_sparse_qsa_shape("attn_paged",&args)==RAD_OK,"valid attention shape");
    for (int flag=0;flag<5;++flag) {
        const auto previous=params[flag];params[flag].ival=flag==0?64:flag==1?0:flag==2?16:flag==3?0:1;
        require(r3dh_sparse_qsa_paged(&args,nullptr)==RAD_E_UNSUPPORTED,"unsupported must refuse before launch");params[flag]=previous;
    }
    values[7].data=values[0].data;require(r3dh_sparse_qsa_paged(&args,nullptr)==RAD_E_SHAPE,"alias output reject");
}
}
int main(int argc,char** argv) {
    if (argc!=2 || (std::strcmp(argv[1],"--host") && std::strcmp(argv[1],"--gpu"))) return 2;
    try {
        gpu=!std::strcmp(argv[1],"--gpu");
        if (gpu) {
            int count=0;hipcheck(hipGetDeviceCount(&count));require(count==1,"exactly one visible GPU");
            hipDeviceProp_t prop{};hipcheck(hipGetDeviceProperties(&prop,0));require(!std::strncmp(prop.gcnArchName,"gfx1100",7),"gfx1100 only checker");
            char bdf[32]{};hipcheck(hipDeviceGetPCIBusId(bdf,sizeof(bdf),0));
            const char* expected_bdf=std::getenv("R3DH_SPARSE_QSA_EXPECT_BDF");
            require(expected_bdf && !std::strcmp(bdf,expected_bdf),"physical GPU PCI BDF mismatch");
            size_t free_bytes,total;hipcheck(hipMemGetInfo(&free_bytes,&total));require(total-free_bytes<256ULL*1024*1024,"GPU has another memory user");
            hipcheck(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));std::printf("DEVICE %s arch=%s bdf=%s used_bytes=%zu stream=nondefault\n",prop.name,prop.gcnArchName,bdf,total-free_bytes);
        }
        host_rejects();
        for (int width:{5,516,50000}) for (int signed_score:{0,1}) selection_fixture(width,width==5?2:512,signed_score);
        for (const char* mode:{"neox","mrope","imrope"}) key_score_fixture(7,mode,true);
        key_score_fixture(7,"neox",false);
        key_score_fixture(50000,"imrope",true);
        for (int context:{7,2050,2051,2052}) for (bool i8:{false,true}) attention_fixture(context,i8,true,false,false);
        attention_fixture(2052,true,true,true,false);
        attention_fixture(31,true,false,false,true);
        rollback_fixture();
        attention_fixture(2052,false,true,false,false,128);
        attention_fixture(2052,true,true,true,false,128);
        if (gpu) hipcheck(hipStreamDestroy(stream));
        std::printf("SPARSE_QSA_PASS mode=%s cases=%zu checks=%zu comparisons=%zu nonbitexact=%zu device_calls=%zu\n",gpu?"gpu":"host",cases,checks,comparisons,differing,launches);
        return 0;
    } catch (const std::exception& error) { std::fprintf(stderr,"SPARSE_QSA_FAIL %s\n",error.what());return 1; }
}
