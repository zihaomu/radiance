#include <chrono>
#include <fstream>

namespace qsa_radix_check {
struct SavedFlag {
    const char* name;
    bool present;
    std::string previous;
    explicit SavedFlag(const char* key) : name(key), present(std::getenv(key) != nullptr), previous(present ? std::getenv(key) : "") {}
    ~SavedFlag() { if (present) setenv(name, previous.c_str(), 1); else unsetenv(name); }
};
using Observe = int (*)(const RadArgs*, RadStream, int*);
using Images = std::array<std::vector<uint8_t>,3>;
struct Fixture {
    int rows, sequences, width, topk;
    std::array<std::unique_ptr<Buffer>,8> buffers;
    std::array<RadTensor,8> device{}, host{};
    std::array<RadParam,6> params;
    RadArgs args{};
    Images poison;
    Fixture(int n, int seqs, int sw, int tw, int k, int sign) : rows(n), sequences(seqs), width(std::min(sw,tw)), topk(k),
        params{integer("ratio",4),integer("topk",k),integer("signed",sign),integer("M",n),integer("blocks",sw),integer("seqs",seqs)} {
        auto add = [&](int index,int type,std::initializer_list<int64_t> shape,int pitch) {
            buffers[index]=std::make_unique<Buffer>(type,shape,RAD_MEM_DEVICE,pitch);
            device[index]=buffers[index]->device; host[index]=buffers[index]->host;
        };
        add(0,RAD_F32,{n,sw},sw+7); add(1,RAD_I32,{seqs,tw},tw+9);
        add(2,RAD_I32,{seqs},0); add(3,RAD_I32,{n},0); add(4,RAD_I32,{seqs+1},0);
        add(5,RAD_I32,{n,k+1},k+12); add(6,RAD_I32,{n},0); add(7,RAD_I32,{n},0);
        args.t=device.data(); args.n_t=8; args.p=params.data(); args.n_p=int(params.size());
        for (int index=0;index<3;++index) poison[index]=buffers[5+index]->expected;
    }
    void reset_out() {
        for (int index=0;index<3;++index) std::copy(poison[index].begin(),poison[index].end(),buffers[index+5]->expected.begin());
    }
    Images images() { return {buffers[5]->expected,buffers[6]->expected,buffers[7]->expected}; }
    void init(int complete,int limit,int pattern,int remainder,bool tail_live) {
        uint32_t state=20261125u;
        const uint32_t specials[]={0u,0x80000000u,1u,0x80000001u,0x007fffffu,0x807fffffu,0x7f800000u,0xff800000u,0x7fc00001u,0x3f800000u,0x3f800001u,0xbf800000u};
        for (int seq=0;seq<=sequences;++seq) buffers[4]->data<int>()[seq*host[4].stride[0]]=rows*seq/sequences;
        for (int seq=0;seq<sequences;++seq) {
            buffers[2]->data<int>()[seq*host[2].stride[0]]=std::min(width,complete+1);
            for (int col=0;col<host[1].shape[1];++col) {
                int page=(col*97+seq*13)%997;
                if (col>=limit || (pattern==5 && col%19==3)) page=-1;
                if (col==complete) page=tail_live ? 303 : -1;
                buffers[1]->data<int>()[seq*host[1].stride[0]+col]=page;
            }
        }
        for (int row=0;row<rows;++row) {
            buffers[3]->data<int>()[row*host[3].stride[0]]=std::max(0,4*complete-1+remainder);
            for (int col=0;col<host[0].shape[1];++col) {
                float value=0;
                state=state*1664525u+1013904223u;
                switch(pattern) {
                case 0: value=float(int(state>>8)-0x7fffff); break;
                case 1: value=1; break;
                case 2: value=float(col%7); break;
                case 3: value=float(col); break;
                case 4: value=float(-col); break;
                default: {
                    const auto word=specials[col%(sizeof(specials)/sizeof(specials[0]))];
                    std::memcpy(&value,&word,sizeof(value));
                    if (col%17==0) value=-1e30f;
                    if (col%17==1) value=std::nextafter(-1e30f,0.0f);
                    if (col%17==2) value=std::nextafter(-1e30f,-INFINITY);
                    break;
                }}
                buffers[0]->data<float>()[row*host[0].stride[0]+col]=value;
            }
        }
    }
};
}

TEST(r3d_qsa_radix_dispatch_contract) {
    using namespace qsa_radix_check;
    RadDeviceProps props{};
    REQUIRE_EQ(rad_dev_props(0,&props),RAD_OK);
    if (props.is_host_backend) {REQUIRE(!std::getenv("RAD_R3D_REQUIRE_DEVICE")); SKIP("requires gfx1100");}
    REQUIRE_EQ(std::strcmp(props.arch,"gfx1100"),0); REQUIRE_EQ(rad_dev_set(0),RAD_OK);
    Library native("libr3d"), ref("libref");
    auto observe=reinterpret_cast<Observe>(dlsym(native.handle,"r3dh_sparse_qsa_select_observed"));
    REQUIRE(observe!=nullptr);
    auto reference=ref.ngram("qsa_select")->launch;
    SavedFlag radix("RAD_QSA_SELECT_RADIX"), four("RAD_QSA_SELECT_RADIX4"), parallel("RAD_QSA_SELECT_PARALLEL"), precision("RAD_PRECISION");
    Queue first(0),second(1);
    size_t hits=0, four_hits=0, cases=0, checked=0, fallback=0;
    auto check=[&](Fixture& fixture,RadStream stream,bool old_parallel) {
        fixture.reset_out(); qsa_index_boundary::select_expected(fixture.host,fixture.topk,fixture.params[2].ival);
        auto expected=fixture.images();fixture.reset_out();fixture.args.t=fixture.host.data();
        REQUIRE_EQ(reference(&fixture.args,nullptr),RAD_OK); REQUIRE(fixture.images()==expected);
        fixture.args.t=fixture.device.data();
        setenv("RAD_QSA_SELECT_PARALLEL",old_parallel?"1":"0",1);
        const bool eligible=old_parallel && fixture.sequences==1 && fixture.topk==512 && fixture.width>512 && fixture.width<=9216;
        for (int mode=0;mode<5;++mode) {
            setenv("RAD_QSA_SELECT_RADIX",mode==0?"0":"1",1);
            if (mode==1) unsetenv("RAD_QSA_SELECT_RADIX4");
            else setenv("RAD_QSA_SELECT_RADIX4",mode==2?"0":mode==3?"yes":"1",1);
            fixture.reset_out();
            for (auto& buf:fixture.buffers) buf->push(stream);
            int path=-1;
            REQUIRE_EQ(observe(&fixture.args,stream,&path),RAD_OK);
            const int wanted=mode && eligible && fixture.rows==1?2:mode==4 && eligible && fixture.rows==4?3:old_parallel?1:0;
            REQUIRE_EQ(path,wanted);
            REQUIRE_EQ(rad_stream_sync(stream),RAD_OK);
            for (int index=0;index<8;++index) {
                const auto bytes=fixture.buffers[index]->read(stream);
                REQUIRE(bytes==(index<5?fixture.buffers[index]->expected:expected[index-5])); checked+=bytes.size();
            }
            if (mode==4) { if (path==2) ++hits; else if (path==3) ++four_hits; else ++fallback; }
        }
        ++cases;
    };
    for (int exact : {0,1}) {
        if (exact) setenv("RAD_PRECISION","exact",1); else unsetenv("RAD_PRECISION");
        for (int sign : {0,1}) for (int width : {512,513,514,2049,8192,8193,8194,9216,9217}) {
            Fixture fixture(1,1,width+(width%2?7:0),width+(width%2?0:5),512,sign);
            for (int pattern=0;pattern<6;++pattern) {
                const int complete=width-(pattern%2);
                fixture.init(complete,complete,pattern,pattern%4,pattern!=5);
                check(fixture,pattern%2?second.stream:first.stream,true);
            }
        }
        for (int live : {0,1,511,512,513}) for (int remainder=0;remainder<4;++remainder) {
            Fixture fixture(1,1,8193,8204,512,1);fixture.init(8192,live,2,remainder,live%2!=0);
            check(fixture,remainder%2?first.stream:second.stream,true);
        }
        for (int shape=0;shape<4;++shape) {
            Fixture fixture(shape==0?2:1,shape==1?2:1,1025,1025,shape==2?2:512,0);
            fixture.init(1024,1024,2,1,true);check(fixture,first.stream,shape!=3);
            int path=-1;fixture.params[1].ival=513;
            REQUIRE_EQ(observe(&fixture.args,first.stream,&path),RAD_E_UNSUPPORTED); REQUIRE_EQ(path,-1);
        }
        for (int sign : {0,1}) for (int width : {512,513,2049,8193,9216,9217}) {
            for (int pattern : {0,1,2,5}) {
                Fixture fixture(4,1,width+7,width,512,sign);
                fixture.init(width-2,width,pattern,0,true);
                for (int row=0;row<4;++row) {
                    fixture.buffers[3]->data<int>()[row]=4*(width-2)-2+row;
                    if (pattern==2) for (int col=0;col<width;++col)
                        fixture.buffers[0]->data<float>()[row*fixture.host[0].stride[0]+col]=float((col+row*5)%17);
                }
                check(fixture,pattern%2?second.stream:first.stream,true);
            }
        }
        for (int live : {0,1,511,512,513}) {
            Fixture fixture(4,1,8200,8193,512,1); fixture.init(8192,live,5,2,live!=0);
            check(fixture,second.stream,true);
        }
        for (int shape=0;shape<5;++shape) {
            Fixture fixture(shape==0?3:shape==1?5:4,shape==2?2:1,1025,1025,shape==3?2:512,1);
            fixture.init(1024,1024,2,1,true);check(fixture,first.stream,shape!=4);
        }
    }
    REQUIRE(hits>0 && four_hits>0 && fallback>0);
    std::printf("QSA_RADIX_PASS cases=%zu single=%zu four=%zu fallback=%zu exact_device_bytes=%zu streams=2 precision_modes=2\n",cases,hits,four_hits,fallback,checked);
}

TEST(r3d_qsa_radix_select_benchmark) {
    using namespace qsa_radix_check;
    if (!std::getenv("RAD_QSA_RADIX_BENCH")) SKIP("opt-in select timing");
    RadDeviceProps props{}; REQUIRE_EQ(rad_dev_props(0,&props),RAD_OK);
    REQUIRE_EQ(std::strcmp(props.arch,"gfx1100"),0); REQUIRE_EQ(rad_dev_set(0),RAD_OK);
    SavedFlag radix("RAD_QSA_SELECT_RADIX"), parallel("RAD_QSA_SELECT_PARALLEL");
    setenv("RAD_QSA_SELECT_PARALLEL","1",1);
    Library native("libr3d"), ref("libref"); Queue queue(0);
    const auto observe=reinterpret_cast<Observe>(dlsym(native.handle,"r3dh_sparse_qsa_select_observed"));
    REQUIRE(observe!=nullptr);
    auto measure=[&](Fixture& fixture,const char* kind) {
        fixture.reset_out(); qsa_index_boundary::select_expected(fixture.host,512,fixture.params[2].ival);
        const Images expected=fixture.images();fixture.reset_out();fixture.args.t=fixture.host.data();
        REQUIRE_EQ(ref.ngram("qsa_select")->launch(&fixture.args,nullptr),RAD_OK); REQUIRE(fixture.images()==expected);
        fixture.args.t=fixture.device.data(); fixture.reset_out();
        for (auto& buf:fixture.buffers) buf->push(queue.stream);
        REQUIRE_EQ(rad_stream_sync(queue.stream),RAD_OK);
        RadEvent begin{},end{}; REQUIRE_EQ(rad_event_create(&begin),RAD_OK);REQUIRE_EQ(rad_event_create(&end),RAD_OK);
        std::vector<double> samples[2];
        for (int variant : {0,1,1,0,1,0,0,1}) {
            setenv("RAD_QSA_SELECT_RADIX",variant?"1":"0",1);
            fixture.reset_out();for(int index=5;index<8;++index)fixture.buffers[index]->push(queue.stream);
            int path=-1;
            for(int warm=0;warm<3;++warm) REQUIRE_EQ(observe(&fixture.args,queue.stream,&path),RAD_OK);
            REQUIRE_EQ(path,variant && fixture.width>512 ? 2 : 1);
            REQUIRE_EQ(rad_stream_sync(queue.stream),RAD_OK);
            REQUIRE_EQ(rad_event_record(begin,queue.stream),RAD_OK);
            for(int repeat=0;repeat<20;++repeat) REQUIRE_EQ(observe(&fixture.args,queue.stream,nullptr),RAD_OK);
            REQUIRE_EQ(rad_event_record(end,queue.stream),RAD_OK); REQUIRE_EQ(rad_event_sync(end),RAD_OK);
            float elapsed=0;REQUIRE_EQ(rad_event_elapsed_ms(begin,end,&elapsed),RAD_OK);REQUIRE(elapsed>0);
            samples[variant].push_back(elapsed*50.0);
            for(int index=5;index<8;++index) REQUIRE(fixture.buffers[index]->read(queue.stream)==expected[index-5]);
        }
        for(auto& group:samples) std::sort(group.begin(),group.end());
        const double before=(samples[0][1]+samples[0][2])/2,after=(samples[1][1]+samples[1][2])/2;
        std::printf("QSA_RADIX_BENCH kind=%s width=%d A_us=%.4f B_us=%.4f ratio=%.4f A_range=%.4f:%.4f B_range=%.4f:%.4f\n",kind,fixture.width,before,after,before/after,samples[0].front(),samples[0].back(),samples[1].front(),samples[1].back());
        rad_event_destroy(begin);rad_event_destroy(end);
    };
    const char* file=std::getenv("RAD_QSA_RADIX_REPLAY");
    if (!file) for(int width:{512,513,2048,8192}) for (int pattern:{0,1,5}) {
        Fixture fixture(1,1,width,width+1,512,1);fixture.init(width,width,pattern,0,false);
        measure(fixture,pattern==0?"random":pattern==1?"ties":"masked");
    }
    if (file) {
        std::ifstream input(file,std::ios::binary); REQUIRE(bool(input));
        std::array<int,6> header{}; input.read(reinterpret_cast<char*>(header.data()),sizeof(header));
        REQUIRE(header[0]==0x51534131 && header[1]>512 && header[1]<=9216 && header[2]>=0 && header[2]<=header[1] && header[3]>=0 && (header[4]==0||header[4]==1));
        Fixture fixture(1,1,header[1],header[1],512,header[4]);fixture.init(header[2],header[2],0,0,false);
        fixture.buffers[2]->data<int>()[0]=header[2];fixture.buffers[3]->data<int>()[0]=header[3];
        input.read(reinterpret_cast<char*>(fixture.buffers[0]->data<float>()),4*header[1]);
        input.read(reinterpret_cast<char*>(fixture.buffers[1]->data<int>()),4*header[1]);
        std::array<int,515> real{};input.read(reinterpret_cast<char*>(real.data()),sizeof(real));REQUIRE(bool(input));
        REQUIRE(input.peek()==std::char_traits<char>::eof());
        fixture.reset_out();qsa_index_boundary::select_expected(fixture.host,512,header[4]);
        REQUIRE(std::equal(real.begin(),real.begin()+513,fixture.buffers[5]->data<int>()));
        REQUIRE_EQ(real[513],fixture.buffers[6]->data<int>()[0]); REQUIRE_EQ(real[514],fixture.buffers[7]->data<int>()[0]);
        std::printf("QSA_RADIX_REPLAY layer=%d width=%d count=%d position=%d nselected=%d used=%d\n",header[5],header[1],header[2],header[3],real[513],real[514]);
        measure(fixture,"real");
    }
}