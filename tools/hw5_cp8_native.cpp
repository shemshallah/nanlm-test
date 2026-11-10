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
#include <random>
#include <numeric>
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
 // Diagnostic only: same frozen native probabilities, three nonoverlapping
 // support regimes. This must sum back to the ordinary aggregate score.
 std::array<Scores,3> score_strata(const Config& c)const{
  std::array<Scores,3> regimes{};
  for(const auto& [ab,labels]:observed){
   const unsigned a=ab/V,b=ab%V;
   const auto p=distribution(a,b,c);
   const auto best=unsigned(std::max_element(p.begin(),p.end())-p.begin());
   const bool context_seen=bank.triple_context[ab]>0;
   for(unsigned j=0;j<V;++j)if(labels[j]){
    const unsigned group=!context_seen?2:(bank.get(a,b,j)>0?0:1);
    auto& bucket=regimes[group];
    bucket.items+=labels[j];
    bucket.correct+=labels[j]*(best==j);
    bucket.nll-=double(labels[j])*std::log(std::max(p[j],1.e-300));
   }
  }
  uint64_t total=0;
  for(const auto& r:regimes)total+=r.items;
  if(total!=n)throw std::runtime_error("nonexclusive native stratum coverage");
  return regimes;
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
static void output_stratum(const char* name,const Config& c,const Scores& s){
 const auto count=double(s.items);
 std::cout<<std::setprecision(12)<<"HW5_CONTEXT_STRATUM"
  <<" regime="<<name<<" beta="<<c.beta<<" lambda="<<c.lambda
  <<" delta="<<c.delta<<" kappa="<<c.kappa
  <<" ce_nats="<<(s.items?s.nll/count:0.)
  <<" accuracy="<<(s.items?double(s.correct)/count:0.)
  <<" transitions="<<s.items<<" exact_model=HW5_train_only"
  <<" bptt_attempted=0\n";
}

// HW5:0 research extension: shared-rank-8 order-sensitive CP geometry.
// U[previous-previous], V[previous] and W[next] live in the previously
// unused 96 bytes of each existing native cognitive-head register.
// Zero extra memory universes, no floating-point NaN operands, gateway 255
// untouched. Optimizer state exists only during fitting, never persisted.
// Trained with TRAIN-ONLY exact triple counts; the source matrix is read only.
constexpr unsigned CP_RANK=8;
using Factors=std::array<std::array<double,CP_RANK>,V>;
struct CP8 {
 Factors u{},v{},w{};
 std::array<Factors,3> first_m{},second_m{};
 static constexpr double beta1=0.9,beta2=0.999;
 static constexpr double epsilon=1e-8;
 void initialize(uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> dist(-0.12,0.12);
  for(unsigned a=0;a<V;++a)for(unsigned k=0;k<CP_RANK;++k){
   u[a][k]=dist(rng);v[a][k]=dist(rng);w[a][k]=dist(rng);
  }
 }
 double dot(unsigned a,unsigned b,unsigned j)const {
  double val=0;
  for(unsigned k=0;k<CP_RANK;++k)val+=u[a][k]*v[b][k]*w[j][k];
  return val;
 }
 static double prior(const Bank& bank,unsigned b,unsigned j) {
  return (double(bank.adjacent[b][j])+0.5)/(double(bank.out[b])+0.5*V);
 }
 static CP8 from_root(const Bank& bank) {
  if(bank.root[29]!='C'||bank.root[30]!='P'||bank.root[31]!='8')
   throw std::runtime_error("CP8 factors absent from trained canonical head registers");
  CP8 model;
  for(unsigned a=0;a<V;++a)for(unsigned k=0;k<CP_RANK;++k){
   const auto load_float=[&](unsigned role) {
    const uint32_t bits=read32(bank.root,a*HEAD_BYTES+32+4*(role*CP_RANK+k));
    const float value=std::bit_cast<float>(bits);
    if(!std::isfinite(value)||std::abs(value)>1.01f)
     throw std::runtime_error("nonfinite or oversize CP8 factor");
    return double(value);
   };
   model.u[a][k]=load_float(0);
   model.v[a][k]=load_float(1);
   model.w[a][k]=load_float(2);
  }
  return model;
 }
 void persist(const Bank& bank,const std::string& path)const {
  auto root=bank.root;
  if(root.size()!=HW5_ROOT_BYTES)
   throw std::runtime_error("wrong root before native CP8 persistence");
  for(unsigned a=0;a<V;++a)for(unsigned k=0;k<CP_RANK;++k){
   const std::array<double,3> vals{u[a][k],v[a][k],w[a][k]};
   for(unsigned role=0;role<3;++role){
    if(!std::isfinite(vals[role])||std::abs(vals[role])>1.01)
     throw std::runtime_error("invalid learned CP8 weight");
    const float val=float(vals[role]);
    write32(root,a*HEAD_BYTES+32+4*(role*CP_RANK+k),
            std::bit_cast<uint32_t>(val));
   }
  }
  // 28 is preexisting head-0 format flag with low byte 1. The following
  // three previously unused bytes distinguish this experimental learner.
  root[29]='C';root[30]='P';root[31]='8';
  if(std::any_of(root.begin()+255*HEAD_BYTES,root.begin()+256*HEAD_BYTES,
                 [](uint8_t x){return x!=0;}))
   throw std::runtime_error("CP8 touched reserved gateway");
  std::ofstream out(path,std::ios::binary|std::ios::trunc);
  if(!out)throw std::runtime_error("cannot create CP8 checkpoint");
  out.write(reinterpret_cast<const char*>(root.data()),HW5_ROOT_BYTES);
  if(!out)throw std::runtime_error("failed CP8 checkpoint write");
 }
 double learn_epoch(const Bank& bank,double alpha,unsigned epoch,double lr) {
  if(alpha<=0||epoch==0||lr<=0)
   throw std::invalid_argument("invalid training hyperparameters");
  std::array<Factors,3> grad{};
  uint64_t observed=0,contexts=0;
  for(unsigned ab=0;ab<CONTEXTS;++ab)if(bank.triple_context[ab]){
   observed+=bank.triple_context[ab];++contexts;
  }
  if(observed<100||contexts<2)throw std::runtime_error("insufficient train triples");
  const double norm=double(observed)+alpha*double(contexts);
  double total_loss=0;
  for(unsigned ab=0;ab<CONTEXTS;++ab){
   const double support=double(bank.triple_context[ab]);
   if(!support)continue;
   const unsigned a=ab/V,b=ab%V;
   Prob logits{},prob{};
   double maxlog=-std::numeric_limits<double>::infinity();
   for(unsigned j=0;j<V;++j){
    logits[j]=std::log(prior(bank,b,j))+std::clamp(dot(a,b,j),-12.0,12.0);
    maxlog=std::max(maxlog,logits[j]);
   }
   double z=0;
   for(unsigned j=0;j<V;++j){prob[j]=std::exp(logits[j]-maxlog);z+=prob[j];}
   if(!(z>0&&std::isfinite(z)))throw std::runtime_error("invalid CP8 normalization");
   for(unsigned j=0;j<V;++j){
    prob[j]/=z;
    const double target=double(bank.get(a,b,j))+alpha*prior(bank,b,j);
    const double influence=((support+alpha)*prob[j]-target)/norm;
    total_loss-=target*std::log(std::max(prob[j],1e-300));
    for(unsigned k=0;k<CP_RANK;++k){
     grad[0][a][k]+=influence*v[b][k]*w[j][k];
     grad[1][b][k]+=influence*u[a][k]*w[j][k];
     grad[2][j][k]+=influence*u[a][k]*v[b][k];
    }
   }
  }
  const double bias_m=1-std::pow(beta1,double(epoch));
  const double bias_v=1-std::pow(beta2,double(epoch));
  auto one_step=[&](Factors& weights,Factors& moments,Factors& variances,
                    const Factors& derivatives){
   for(unsigned a=0;a<V;++a)for(unsigned k=0;k<CP_RANK;++k){
    const double g=derivatives[a][k]+1e-5*weights[a][k];
    moments[a][k]=beta1*moments[a][k]+(1-beta1)*g;
    variances[a][k]=beta2*variances[a][k]+(1-beta2)*g*g;
    weights[a][k]-=lr*(moments[a][k]/bias_m)/
                    (std::sqrt(variances[a][k]/bias_v)+epsilon);
    weights[a][k]=std::clamp(weights[a][k],-1.0,1.0);
   }
  };
  one_step(u,first_m[0],second_m[0],grad[0]);
  one_step(v,first_m[1],second_m[1],grad[1]);
  one_step(w,first_m[2],second_m[2],grad[2]);
  return total_loss/norm;
 }
};
// A single readout distribution: native exact triple memory + one trained
// rank-eight pair-conditioned geometric backoff. Development-only beta/rho/
// scale selection; no updates or model choice during evaluation.
// Exact-support gate: known train triples keep their original probability.
// Geometry ONLY redistributes the probability mass assigned to unseen
// continuations; it cannot steal probability from a known exact triple.
// 0 = ordinary global mixture; 1 = novelty/unknown-only mixture.
static Prob geometric_gate(const Prob& memory,const Prob& geometry,
                           const std::array<uint8_t,V>& seen,
                           double rho,unsigned gate) {
 if(rho<0||rho>1||gate>1)throw std::invalid_argument("invalid geometric gate");
 Prob out{};
 if(gate==0){
  for(unsigned j=0;j<V;++j)out[j]=(1-rho)*memory[j]+rho*geometry[j];
 }else{
  double unknown_memory_mass=0,unknown_geometry_mass=0;
  for(unsigned j=0;j<V;++j)if(!seen[j]){
   unknown_memory_mass+=memory[j];
   unknown_geometry_mass+=geometry[j];
  }
  if(!(unknown_memory_mass>0&&unknown_geometry_mass>0))
   throw std::runtime_error("invalid unknown continuation support");
  for(unsigned j=0;j<V;++j){
   if(seen[j])out[j]=memory[j];
   else out[j]=(1-rho)*memory[j]+rho*unknown_memory_mass*
                        geometry[j]/unknown_geometry_mass;
  }
 }
 const double sum=std::accumulate(out.begin(),out.end(),0.);
 if(!std::isfinite(sum)||std::abs(sum-1.)>1e-8)
  throw std::runtime_error("native geometric gate lost probability mass");
 return out;
}
static Scores cp_score(const Eval& e,const CP8& cp,
                       double beta,double rho,double scale,unsigned gate,
                       std::array<Scores,3>* by_regime=nullptr) {
 if(beta<=0||rho<0||rho>1||scale<0||scale>4||gate>1)
  throw std::invalid_argument("invalid CP8 dev-frozen interpolation");
 Scores answer{};
 std::array<Scores,3> strata{};
 for(const auto& [ab,labels]:e.observed){
  const unsigned a=ab/V,b=ab%V;
  const double support=double(e.bank.triple_context[ab]);
  Prob q{},memory{};
  std::array<uint8_t,V> seen{};
  double z=0;
  for(unsigned j=0;j<V;++j){
   q[j]=e.bi[b][j]*std::exp(std::clamp(scale*cp.dot(a,b,j),-12.0,12.0));
   z+=q[j];
  }
  if(!(z>0&&std::isfinite(z)))throw std::runtime_error("invalid CP8 probability mass");
  for(unsigned j=0;j<V;++j){
   q[j]/=z;
   const auto count=e.bank.get(a,b,j);
   seen[j]=uint8_t(count>0);
   memory[j]=(double(count)+beta*e.bi[b][j])/(support+beta);
  }
  const auto p=geometric_gate(memory,q,seen,rho,gate);
  const unsigned best=unsigned(std::max_element(p.begin(),p.end())-p.begin());
  for(unsigned j=0;j<V;++j)if(labels[j]){
   const double nll=-double(labels[j])*std::log(std::max(p[j],1e-300));
   answer.items+=labels[j];
   answer.correct+=labels[j]*(best==j);
   answer.nll+=nll;
   const unsigned stratum=seen[j]?0:(support>0?1:2);
   strata[stratum].items+=labels[j];
   strata[stratum].correct+=labels[j]*(best==j);
   strata[stratum].nll+=nll;
  }
 }
 if(answer.items!=e.n||!std::isfinite(answer.nll))
  throw std::runtime_error("CP8 scoring coverage mismatch");
 if(by_regime){
  uint64_t count=0;
  for(auto& stratum:strata)count+=stratum.items;
  if(count!=answer.items)throw std::runtime_error("CP8 support strata missed transitions");
  *by_regime=strata;
 }
 return answer;
}
static void cp_output(const Scores& scores,double beta,double rho,
                      double scale,unsigned gate) {
 std::cout<<std::setprecision(12)
          <<"HW5_CP8_SCORE mode=cp8 beta="<<beta
          <<" rho="<<rho<<" scale="<<scale<<" gate="<<gate
          <<" ce_nats="<<scores.nll/double(scores.items)
          <<" accuracy="<<double(scores.correct)/double(scores.items)
          <<" transitions="<<scores.items
          <<" exact_model=HW5_train_only bptt_attempted=0\n";
}
static void cp_regime_output(const std::array<Scores,3>& s) {
 constexpr std::array<const char*,3> names{
  "seen_exact_triple","seen_context_unseen_continuation","unseen_context"};
 for(unsigned i=0;i<3;++i){
  std::cout<<std::setprecision(12)
           <<"HW5_CP8_REGIME regime="<<names[i]
           <<" transitions="<<s[i].items
           <<" ce_nats="<<(s[i].items?s[i].nll/double(s[i].items):0.)
           <<" accuracy="<<(s[i].items?double(s[i].correct)/double(s[i].items):0.)
           <<" exact_model=HW5_train_only bptt_attempted=0\n";
 }
}
static void cp_gate_selftest() {
 Prob memory{},geometry{};
 std::array<uint8_t,V> seen{};
 memory[0]=0.7;memory[1]=0.2;memory[2]=0.1;
 geometry[0]=0.05;geometry[1]=0.15;geometry[2]=0.8;
 seen[0]=1;
 for(double rho:{0.,0.2,0.5,1.}){
  const auto global=geometric_gate(memory,geometry,seen,rho,0);
  const auto novelty=geometric_gate(memory,geometry,seen,rho,1);
  if(std::abs(novelty[0]-0.7)>1e-14)
   throw std::runtime_error("novelty gate modified seen target");
  if(std::abs(novelty[1]+novelty[2]-0.3)>1e-14)
   throw std::runtime_error("novelty gate modified unknown probability budget");
  if(std::abs(global[0]-((1-rho)*0.7+rho*0.05))>1e-14)
   throw std::runtime_error("global geometric mixture mismatch");
  if(rho==0&&(std::abs(novelty[1]-0.2)>1e-14||
              std::abs(novelty[2]-0.1)>1e-14))
   throw std::runtime_error("zero-geometry identity violated");
 }
 std::cout<<"HW5_CP8_GATE_TEST status=passed known_probability_immutable=1"
          <<" unknown_mass_conserved=1 normalization=1\n";
}

int main(int argc,char** argv){
 try{
  if(argc<2)throw std::invalid_argument("build TRAIN LENGTHS ROOT TRIPLES | score STUDENT ROOT TRIPLES EVAL LENGTHS sweep|fixed [BETA LAMBDA DELTA KAPPA]");
  const std::string mode=argv[1];
  if(mode=="gate-selftest"){
   if(argc!=2)throw std::invalid_argument("gate-selftest takes no args");
   cp_gate_selftest();
  }else if(mode=="fit-cp"){
   if(argc!=8)throw std::invalid_argument("fit-cp TRAIN_ROOT TRIPLES CHECKPOINT_PREFIX EPOCHS SEED ALPHA");
   Bank bank;bank.load(argv[2],argv[3]);
   CP8 cp;cp.initialize(std::stoull(argv[6]));
   const int epochs=std::stoi(argv[5]);
   const double alpha=std::stod(argv[7]);
   if(epochs<1||epochs>24||!(alpha>=1.0&&alpha<=1024.0))
    throw std::invalid_argument("epochs or smoothing outside audit bounds");
   for(int epoch=1;epoch<=epochs;++epoch){
    const double objective=cp.learn_epoch(bank,alpha,unsigned(epoch),0.08);
    const std::string out=std::string(argv[4])+".e"+(epoch<10?"0":"")+std::to_string(epoch)+".root";
    cp.persist(bank,out);
    std::cout<<std::setprecision(12)
             <<"HW5_CP8_TRAIN epoch="<<epoch<<" smoothed_ce_nats="<<objective
             <<" root="<<out<<" input=train_only"
             <<" rank=8 bptt_attempted=0\n";
   }
  }else if(mode=="score-cp-sweep"){
   if(argc!=7)throw std::invalid_argument("score-cp-sweep STUDENT CP8_ROOT TRIPLES DEV LENGTHS");
   Bank bank;bank.load(argv[3],argv[4]);
   const CP8 cp=CP8::from_root(bank);
   const Eval e(bank,argv[2],argv[5],lengths(argv[6]));
   for(unsigned gate:{0u,1u})
    for(double rho:{0.,0.01,0.03,0.1,0.2,0.4,0.7})
     for(double scale:{0.5,1.,2.}){
      if(rho==0.&&(scale!=1.||gate!=0u))continue;
      cp_output(cp_score(e,cp,16.,rho,scale,gate),16.,rho,scale,gate);
     }
  }else if(mode=="score-cp"){
   if(argc!=11)throw std::invalid_argument("score-cp STUDENT CP8_ROOT TRIPLES EVAL LENGTHS BETA RHO SCALE GATE");
   Bank bank;bank.load(argv[3],argv[4]);
   const CP8 cp=CP8::from_root(bank);
   const Eval e(bank,argv[2],argv[5],lengths(argv[6]));
   const double beta=std::stod(argv[7]),rho=std::stod(argv[8]),scale=std::stod(argv[9]);
   const unsigned gate=unsigned(std::stoul(argv[10]));
   std::array<Scores,3> strata{};
   const auto total=cp_score(e,cp,beta,rho,scale,gate,&strata);
   cp_output(total,beta,rho,scale,gate);
   cp_regime_output(strata);
  }else if(mode=="build"){
   if(argc!=6)throw std::invalid_argument("build TRAIN LENGTHS ROOT TRIPLES");
   Bank bank;
   const auto n=visit(argv[2],lengths(argv[3]),[&](const std::vector<uint8_t>& bytes){bank.learn(bytes);});
   bank.finish();
   bank.save(argv[4],argv[5]);
   uint64_t occupied=0;
   for(unsigned a=0;a<V;++a)for(unsigned b=a+1;b<V;++b){
    occupied+=(bank.adjacent[a][b]||bank.adjacent[b][a]||bank.skip[a][b]||bank.skip[b][a]);
   }
   std::cout<<"HW5_CONTEXT_BANK_BUILT bytes="<<n
     <<" books="<<bank.books<<" pairs="<<occupied
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
    const auto by_support=e.score_strata(c);
    output_stratum("seen_exact_triple",c,by_support[0]);
    output_stratum("seen_context_unseen_continuation",c,by_support[1]);
    output_stratum("unseen_context",c,by_support[2]);
   }else throw std::invalid_argument("unknown scoring mode");
  }else throw std::invalid_argument("unknown subcommand");
  return 0;
 }catch(const std::exception& e){
  std::cerr<<"HW5_CONTEXT_ERROR "<<e.what()<<"\n";
  return 2;
 }
}
