// HW5 native C(256,2) context-memory gate. All counts are learned ONLY
// from training books. The published HW5 full reference must NEVER be used
// as a language-model memory during held-out evaluation (test leakage).
// This is one existing geometric student + a read-only mirrored evidence
// projection and sparse exact ordered-triple overlay. No BPTT or TRAIL.
#include "nanlm/unified_wave.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace nanlm;
static_assert(std::endian::native == std::endian::little);
constexpr unsigned V=255, HW5_HEADS=256;
constexpr size_t HEAD_BYTES=128, PAIR_BYTES=256, HW5_ROOT_BYTES=8388608;
constexpr unsigned CONTEXTS=V*V;
using Row=std::array<uint32_t,V>;
using Counts=std::array<Row,V>;
using Prob=std::array<double,V>;
using TripleMap=std::unordered_map<uint32_t,uint32_t>;
constexpr char MAGIC[8]={'H','W','5','T','R','I','0','1'};
constexpr unsigned HW5_PAIRS=HW5_HEADS*(HW5_HEADS-1)/2;
// H5P1 identifies every physical unordered C(256,2) slot, including zero-
// evidence slots and 255 reserved gateway-pairs. Never fabricate counts.
constexpr uint32_t PAIR_TAG=0x31503548U;
static uint32_t key(unsigned a,unsigned b,unsigned c){
 if(a>=V||b>=V||c>=V)throw std::invalid_argument("invalid triple head");
 return (a<<16)|(b<<8)|c;
}
static uint32_t read32(const std::vector<uint8_t>& root,size_t off){
 if(off+4>root.size())throw std::runtime_error("pair offset overflow");
 uint32_t x=0;std::memcpy(&x,root.data()+off,4);return x;
}
static void write32(std::vector<uint8_t>& root,size_t off,uint32_t x){
 if(off+4>root.size())throw std::runtime_error("pair offset overflow");
 std::memcpy(root.data()+off,&x,4);
}
static size_t pair_id(unsigned a,unsigned b){
 if(!(a<b&&b<HW5_HEADS))throw std::invalid_argument("invalid pair");
 return 1+a*(511-a)/2+(b-a-1);
}
static size_t base(unsigned a,unsigned b){
 const auto x=std::min(a,b), y=std::max(a,b);
 return HW5_HEADS*HEAD_BYTES+(pair_id(x,y)-1)*PAIR_BYTES;
}
// Every physical slot is explicitly addressable, including unseen pairs.
// All statistical evidence must originate in the training books.
static void address_all_pairs(std::vector<uint8_t>& root){
 uint32_t addressed=0;
 for(unsigned a=0;a<HW5_HEADS;++a)for(unsigned b=a+1;b<HW5_HEADS;++b){
  const size_t at=base(a,b);
  write32(root,at+160,a);write32(root,at+164,b);
  write32(root,at+168,PAIR_TAG);
  ++addressed;
 }
 if(addressed!=HW5_PAIRS ||
    HW5_HEADS*HEAD_BYTES+HW5_PAIRS*PAIR_BYTES!=HW5_ROOT_BYTES)
  throw std::runtime_error("incomplete C256 physical addressing");
}
static void verify_all_pairs(const std::vector<uint8_t>& root){
 uint32_t addressed=0;
 for(unsigned a=0;a<HW5_HEADS;++a)for(unsigned b=a+1;b<HW5_HEADS;++b){
  const size_t at=base(a,b);
  if(read32(root,at+160)!=a||read32(root,at+164)!=b||
     read32(root,at+168)!=PAIR_TAG)
   throw std::runtime_error("missing or corrupted C256 physical pair identity");
  if(b==255){
   for(size_t off=0;off<88;off+=4)
    if(read32(root,at+off)!=0)
     throw std::runtime_error("gateway pair has training evidence");
  }
  ++addressed;
 }
 if(addressed!=HW5_PAIRS)throw std::runtime_error("incomplete pair ledger");
}
static uint32_t pair_read(const std::vector<uint8_t>& root,
                          unsigned a,unsigned b,unsigned lane){
 if(a>=V||b>=V||lane>1)throw std::invalid_argument("bad pair lookup");
 if(a==b)return read32(root,a*HEAD_BYTES+(lane==0?12:16));
 const size_t at=base(a,b)+16+4*((a<b?0:1)*16+lane);
 return read32(root,at);
}
static void pair_write(std::vector<uint8_t>& root,
                       unsigned a,unsigned b,unsigned lane,uint32_t count){
 if(a==b){write32(root,a*HEAD_BYTES+(lane==0?12:16),count);return;}
 const size_t at=base(a,b);
 const unsigned dir=a<b?0:1;
 write32(root,at+16+4*(dir*16+lane),count);
 write32(root,at+8+4*dir,read32(root,at+8+4*dir)|(1u<<lane));
 if(lane==0)write32(root,at+4*dir,count);
}
static std::vector<uint64_t> lengths(const std::string& file){
 std::ifstream in(file);
 if(!in)throw std::runtime_error("missing document lengths");
 std::vector<uint64_t> v;uint64_t n=0;
 while(in>>n){if(n<3||n>6000000)throw std::runtime_error("invalid book length");v.push_back(n);}
 if(!in.eof()||v.empty()||v.size()>256)throw std::runtime_error("invalid document boundary ledger");
 return v;
}
template<class F>
static uint64_t visit(const std::string& file,const std::vector<uint64_t>& sizes,F emit){
 std::ifstream in(file,std::ios::binary);
 if(!in)throw std::runtime_error("missing corpus");
 uint64_t total=0;
 std::vector<uint8_t> bytes;
 for(const auto n:sizes){
  bytes.resize(static_cast<size_t>(n));
  in.read(reinterpret_cast<char*>(bytes.data()),static_cast<std::streamsize>(n));
  if(static_cast<uint64_t>(in.gcount())!=n)throw std::runtime_error("truncated source document");
  for(const auto x:bytes)if(x>=V)throw std::runtime_error("0xff ingress forbidden");
  emit(bytes);
  total+=n;
 }
 if(in.peek()!=EOF)throw std::runtime_error("extra source bytes beyond book ledger");
 if(in.bad())throw std::runtime_error("failed source read");
 return total;
}
struct Bank{
 std::vector<uint8_t> root=std::vector<uint8_t>(HW5_ROOT_BYTES);
 TripleMap triple;
 Counts adjacent{},skip{};
 std::array<uint64_t,V> context{},out{},in{},start{},end{};
 std::array<uint64_t,CONTEXTS> triple_context{};
 uint64_t transitions=0,lag2=0,bytes=0,books=0;
 void learn(const std::vector<uint8_t>& x){
  if(x.size()<3)throw std::runtime_error("short document");
  ++books;bytes+=x.size();
  ++start[x.front()];++end[x.back()];
  for(size_t i=1;i<x.size();++i){
   const auto a=x[i-1],b=x[i];++adjacent[a][b];++out[a];++in[b];++transitions;
   if(i>=2){
    const auto p=x[i-2];++skip[p][b];++triple[key(p,a,b)];++lag2;
   }
  }
 }
 void finish(){
  address_all_pairs(root);
  for(unsigned a=0;a<V;++a){
   write32(root,a*HEAD_BYTES,a);
   write32(root,a*HEAD_BYTES+4,static_cast<uint32_t>(out[a]));
   write32(root,a*HEAD_BYTES+8,static_cast<uint32_t>(in[a]));
   write32(root,a*HEAD_BYTES+20,static_cast<uint32_t>(start[a]));
   write32(root,a*HEAD_BYTES+24,static_cast<uint32_t>(end[a]));
   write32(root,a*HEAD_BYTES+28,1);
   for(unsigned b=0;b<V;++b){
    if(adjacent[a][b])pair_write(root,a,b,0,adjacent[a][b]);
    if(skip[a][b])pair_write(root,a,b,1,skip[a][b]);
   }
  }
  if(transitions!=bytes-books||lag2!=bytes-2*books)
   throw std::runtime_error("document boundary count mismatch");
  if(std::any_of(root.begin()+255*HEAD_BYTES,root.begin()+256*HEAD_BYTES,
                  [](uint8_t x){return x!=0;}))
   throw std::runtime_error("gateway head not empty");
  verify_all_pairs(root);
 }
 void save(const std::string& rootpath,const std::string& triplepath)const{
  std::ofstream r(rootpath,std::ios::binary|std::ios::trunc);
  if(!r)throw std::runtime_error("cannot write training projection");
  r.write(reinterpret_cast<const char*>(root.data()),HW5_ROOT_BYTES);
  if(!r)throw std::runtime_error("failed training projection");
  std::vector<std::pair<uint32_t,uint32_t>> sorted(triple.begin(),triple.end());
  std::sort(sorted.begin(),sorted.end());
  std::ofstream t(triplepath,std::ios::binary|std::ios::trunc);
  if(!t)throw std::runtime_error("cannot write training triples");
  t.write(MAGIC,8);
  const uint64_t number=sorted.size();
  t.write(reinterpret_cast<const char*>(&number),8);
  for(const auto& [k,n]:sorted){
   t.write(reinterpret_cast<const char*>(&k),4);
   t.write(reinterpret_cast<const char*>(&n),4);
  }
  if(!t)throw std::runtime_error("failed training triples");
 }
 void load(const std::string& rootpath,const std::string& triplepath){
  std::ifstream r(rootpath,std::ios::binary);
  if(!r)throw std::runtime_error("missing training projection");
  r.read(reinterpret_cast<char*>(root.data()),HW5_ROOT_BYTES);
  if(static_cast<size_t>(r.gcount())!=HW5_ROOT_BYTES||r.peek()!=EOF)
   throw std::runtime_error("wrong training projection size");
  verify_all_pairs(root);
  for(unsigned a=0;a<V;++a){
   if(read32(root,a*HEAD_BYTES)!=a)throw std::runtime_error("wrong canonical head");
   out[a]=read32(root,a*HEAD_BYTES+4);
   in[a]=read32(root,a*HEAD_BYTES+8);
   for(unsigned b=0;b<V;++b){
    adjacent[a][b]=pair_read(root,a,b,0);
    skip[a][b]=pair_read(root,a,b,1);
    context[a]+=adjacent[a][b];
    if(a==b)continue;
    const size_t at=base(a,b);
    const unsigned d=a<b?0:1;
    if(read32(root,at+4*d)!=adjacent[a][b])
      throw std::runtime_error("adjacent header != mirror lane");
    const unsigned mask=read32(root,at+8+4*d);
    if(bool(mask&1)!=(adjacent[a][b]!=0) ||
       bool(mask&2)!=(skip[a][b]!=0))
      throw std::runtime_error("mirror lane occupancy mismatch");
   }
   if(context[a]!=out[a])throw std::runtime_error("row count mismatch");
  }
  if(std::any_of(root.begin()+255*HEAD_BYTES,root.begin()+256*HEAD_BYTES,
                 [](uint8_t x){return x!=0;}))
   throw std::runtime_error("gateway head used");
  std::ifstream t(triplepath,std::ios::binary);
  if(!t)throw std::runtime_error("missing training triple ledger");
  char magic[8]{};
  t.read(magic,8);
  if(std::memcmp(magic,MAGIC,8))throw std::runtime_error("bad triple magic");
  uint64_t number=0;t.read(reinterpret_cast<char*>(&number),8);
  if(number>V*V*V)throw std::runtime_error("oversize trigram ledger");
  uint32_t prev=0;
  for(uint64_t i=0;i<number;++i){
   uint32_t k=0,n=0;
   t.read(reinterpret_cast<char*>(&k),4);
   t.read(reinterpret_cast<char*>(&n),4);
   if(!t||!n||k>>24||((k>>16)&255)>=V||((k>>8)&255)>=V||(k&255)>=V
      ||(i&&k<=prev)||!triple.emplace(k,n).second)
    throw std::runtime_error("malformed sorted exact triples");
   prev=k;
   triple_context[((k>>16)&255)*V+((k>>8)&255)]+=n;
  }
  if(t.peek()!=EOF)throw std::runtime_error("extra trigram bytes");
  for(const auto& [k,n]:triple){
   const unsigned a=k>>16,b=(k>>8)&255,c=k&255;
   if(n>adjacent[a][b]||n>skip[a][c]||triple_context[a*V+b]>adjacent[a][b])
    throw std::runtime_error("inconsistent triple evidence");
  }
 }
 uint32_t get(unsigned a,unsigned b,unsigned c)const{
  const auto it=triple.find(key(a,b,c));
  return it==triple.end()?0:it->second;
 }
};
struct Scores {double nll=0;uint64_t correct=0,items=0;};
struct Config {double beta=16,lambda=0,delta=0,kappa=0;};
struct Eval{
 const Bank& bank;
 std::array<Prob,V> geometry{},bi{},skip{};
 std::array<std::array<float,3>,V> src{},dst{};
 std::map<unsigned,std::array<uint64_t,V>> observed;
 uint64_t n=0;
 Eval(const Bank& b,const std::string& student,const std::string& text,
      const std::vector<uint64_t>& sizes):bank(b){
  Universe universe(student);
  UnifiedWaveLanguage wave(universe);
  FabricTeacher teacher(universe);
  for(unsigned a=0;a<V;++a){
   geometry[a]=teacher.predict(static_cast<uint8_t>(a)).probability;
   for(int k=0;k<3;++k){
    std::memcpy(&src[a][k],universe.root().h[a].data+32+4*k,4);
    std::memcpy(&dst[a][k],universe.root().h[a].data+48+4*k,4);
   }
   double row=0,lag=0;
   for(unsigned j=0;j<V;++j){row+=bank.adjacent[a][j];lag+=bank.skip[a][j];}
   for(unsigned j=0;j<V;++j){
    bi[a][j]=(bank.adjacent[a][j]+0.5)/(row+0.5*V);
    skip[a][j]=(bank.skip[a][j]+0.5)/(lag+0.5*V);
   }
  }
  visit(text,sizes,[&](const std::vector<uint8_t>& bytes){
   for(size_t i=2;i<bytes.size();++i){
    const unsigned a=bytes[i-2],p=bytes[i-1],j=bytes[i];
    ++observed[a*V+p][j];++n;
   }
  });
  if(n<100)throw std::runtime_error("insufficient evaluation triples");
 }
 // Two-byte geometry is a CP rank-three term residing in the EXISTING
 // palindrome registers; never a second model or a separately learned head.
 Prob geom3(unsigned a,unsigned b,double kappa)const{
  if(!kappa)return geometry[b];
  Prob p{};double sum=0;
  for(unsigned j=0;j<V;++j){
   double cp=0;
   for(int k=0;k<3;++k)cp+=double(src[a][k])*double(src[b][k])*double(dst[j][k]);
   p[j]=geometry[b][j]*std::exp(std::clamp(kappa*cp,-20.0,20.0));
   sum+=p[j];
  }
  for(auto& v:p)v/=sum;
  return p;
 }
 // Sparse exact trigram count with its own document-boundary-aware denominator,\n // normalized to a Laplace-smoothed
 // C(256,2) directed pair bank, optional second-step mirror and geometry.
 Prob distribution(unsigned a,unsigned b,const Config& c)const{
  const auto geo=geom3(a,b,c.kappa);
  Prob p{};double sum=0;
  const double support=double(bank.triple_context[a*V+b]);
  for(unsigned j=0;j<V;++j){
   const double tri=(double(bank.get(a,b,j))+c.beta*bi[b][j])/(support+c.beta);
   p[j]=(1-c.lambda-c.delta)*tri+c.delta*skip[a][j]+c.lambda*geo[j];
   sum+=p[j];
  }
  if(!(sum>0&&std::isfinite(sum)))throw std::runtime_error("invalid memory normalization");
  for(auto& v:p)v/=sum;
  return p;
 }
 Scores score(const Config& c)const{
  if(c.beta<=0||c.lambda<0||c.delta<0||c.lambda+c.delta>1)
   throw std::invalid_argument("non-convex interpolation");
  Scores s{};
  for(const auto& [ab,labels]:observed){
   const unsigned a=ab/V,b=ab%V;
   const auto p=distribution(a,b,c);
   const auto best=unsigned(std::max_element(p.begin(),p.end())-p.begin());
   for(unsigned j=0;j<V;++j){
    const uint64_t count=labels[j];
    if(!count)continue;
    s.items+=count;
    s.correct+=count*(j==best);
    s.nll-=double(count)*std::log(std::max(p[j],1.e-300));
   }
  }
  if(s.items!=n||!std::isfinite(s.nll))throw std::runtime_error("scoring coverage mismatch");
  return s;
 }
 Scores score_base(int kind)const{
  Scores s{};
  for(const auto& [ab,labels]:observed){
   const unsigned b=ab%V;
   const auto& p=kind==0?geometry[b]:bi[b];
   const auto best=unsigned(std::max_element(p.begin(),p.end())-p.begin());
   for(unsigned j=0;j<V;++j)if(labels[j]){
    s.items+=labels[j];s.correct+=labels[j]*(best==j);
    s.nll-=double(labels[j])*std::log(std::max(p[j],1.e-300));
   }
  }
  if(s.items!=n)throw std::runtime_error("baseline token mismatch");
  return s;
 }
};
static void output(const std::string& tag,const Config& c,const Scores& s){
 std::cout<<std::setprecision(12)<<"HW5_CONTEXT_SCORE"
  <<" mode="<<tag<<" beta="<<c.beta<<" lambda="<<c.lambda
  <<" delta="<<c.delta<<" kappa="<<c.kappa
  <<" ce_nats="<<(s.nll/double(s.items))
  <<" accuracy="<<(double(s.correct)/double(s.items))
  <<" transitions="<<s.items<<" exact_model=HW5_train_only"
  <<" bptt_attempted=0\n";
}
int main(int argc,char** argv){
 try{
  if(argc<2)throw std::invalid_argument("build TRAIN LENGTHS ROOT TRIPLES | score STUDENT ROOT TRIPLES EVAL LENGTHS sweep|fixed [BETA LAMBDA DELTA KAPPA]");
  const std::string mode=argv[1];
  if(mode=="build"){
   if(argc!=6)throw std::invalid_argument("build TRAIN LENGTHS ROOT TRIPLES");
   Bank bank;
   const auto n=visit(argv[2],lengths(argv[3]),[&](const std::vector<uint8_t>& bytes){bank.learn(bytes);});
   bank.finish();
   bank.save(argv[4],argv[5]);
   uint64_t occupied=0,occupied_adjacent=0,occupied_skip=0;
   uint64_t directed_adjacent=0,directed_skip=0,self_adjacent=0;
   for(unsigned a=0;a<V;++a){
    self_adjacent+=bank.adjacent[a][a]!=0;
    for(unsigned b=0;b<V;++b){
     directed_adjacent+=bank.adjacent[a][b]!=0;
     directed_skip+=bank.skip[a][b]!=0;
     if(a>=b)continue;
     occupied+=(bank.adjacent[a][b]||bank.adjacent[b][a]||bank.skip[a][b]||bank.skip[b][a]);
     occupied_adjacent+=(bank.adjacent[a][b]||bank.adjacent[b][a]);
     occupied_skip+=(bank.skip[a][b]||bank.skip[b][a]);
    }
   }
   std::cout<<"HW5_CONTEXT_BANK_BUILT bytes="<<n
     <<" books="<<bank.books<<" pairs="<<occupied
     <<" addressed_pairs="<<HW5_PAIRS<<" reserved_pairs="<<HW5_HEADS-1
     <<" evidence_pairs="<<occupied<<" absent_active_pairs="<<(HW5_PAIRS-(HW5_HEADS-1)-occupied)
     <<" adjacent_mirror_pairs="<<occupied_adjacent<<" skip_mirror_pairs="<<occupied_skip
     <<" directed_adjacent_pairs="<<directed_adjacent<<" directed_skip_pairs="<<directed_skip
     <<" self_adjacent_heads="<<self_adjacent
     <<" distinct_triples="<<bank.triple.size()
     <<" adjacent="<<bank.transitions<<" lag2="<<bank.lag2
     <<" root_bytes="<<HW5_ROOT_BYTES<<" gateway_head=255\n";
  }else if(mode=="score"){
   if(argc<8)throw std::invalid_argument("score STUDENT ROOT TRIPLES EVAL LENGTHS sweep|fixed");
   Bank bank;bank.load(argv[3],argv[4]);
   const auto sizes=lengths(argv[6]);
   const Eval e(bank,argv[2],argv[5],sizes);
   output("geometry_only",Config{},e.score_base(0));
   output("bigram_add05",Config{},e.score_base(1));
   if(std::string(argv[7])=="sweep"){
    for(double beta:{1.,4.,16.,64.,256.})
     for(double lambda:{0.,0.01,0.03,0.07,0.15,0.3})
      for(double delta:{0.,0.03,0.07,0.15,0.3})
       for(double kappa:{0.,1.,4.}){
        if(lambda==0&&kappa!=0)continue;
        if(lambda+delta>0.8)continue;
        const Config c{beta,lambda,delta,kappa};
        output("context_hybrid",c,e.score(c));
       }
   }else if(std::string(argv[7])=="fixed"){
    if(argc!=12)throw std::invalid_argument("fixed requires BETA LAMBDA DELTA KAPPA");
    const Config c{std::stod(argv[8]),std::stod(argv[9]),std::stod(argv[10]),std::stod(argv[11])};
    output("context_hybrid",c,e.score(c));
   }else throw std::invalid_argument("unknown scoring mode");
  }else throw std::invalid_argument("unknown subcommand");
  return 0;
 }catch(const std::exception& e){
  std::cerr<<"HW5_CONTEXT_ERROR "<<e.what()<<"\n";
  return 2;
 }
}
