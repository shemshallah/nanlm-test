// HW5 native C(256,2) conditional geometric credit: the existing 8 MiB
// NaNLM head vectors learn P(c|a,b); no trigram lookup at inference.
// Native rank-three CP softmax with train-only smoothed trigram targets.
// Not full BPTT, recurrent TRAIL, or a claim of semantic understanding.
#include "nanlm/teacher.hpp"
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
#include <vector>
using namespace nanlm;
static_assert(std::endian::native==std::endian::little);
static constexpr unsigned N=FabricTeacher::VOCAB;
static constexpr double GAIN=4.0, CP_GAIN=4.0;
using Vec=std::array<double,3>;
using Matrix=std::array<Vec,N>;
using Row=std::array<uint32_t,N>;
struct Corpus {
 std::map<unsigned,Row> ctx;
 std::array<Row,N> bigram{};
 std::array<uint64_t,N> rows{};
 uint64_t books=0,bytes=0,triples=0,adjacent=0;
 explicit Corpus(const std::string& txt,const std::string& ledger){
  std::ifstream sizes(ledger);
  if(!sizes)throw std::runtime_error("missing length ledger");
  std::vector<uint64_t> lengths;uint64_t n=0;
  while(sizes>>n){if(n<3||n>6000000)throw std::runtime_error("invalid document length");lengths.push_back(n);}
  if(!sizes.eof()||lengths.empty()||lengths.size()>256)throw std::runtime_error("invalid length ledger");
  std::ifstream input(txt,std::ios::binary);
  if(!input)throw std::runtime_error("missing corpus");
  for(const uint64_t len:lengths){
   std::vector<uint8_t> b(static_cast<size_t>(len));
   input.read(reinterpret_cast<char*>(b.data()),static_cast<std::streamsize>(len));
   if(static_cast<uint64_t>(input.gcount())!=len)throw std::runtime_error("truncated corpus document");
   for(const uint8_t x:b)if(x>=N)throw std::runtime_error("0xff reserved network ingress");
   ++books;bytes+=len;
   for(size_t i=1;i<b.size();++i){
    ++bigram[b[i-1]][b[i]];++rows[b[i-1]];++adjacent;
    if(i>=2){++ctx[static_cast<unsigned>(b[i-2])*N+b[i-1]][b[i]];++triples;}
   }
  }
  if(input.peek()!=EOF||input.bad())throw std::runtime_error("extra or failed document input");
  if(adjacent!=bytes-books||triples!=bytes-2*books)
   throw std::runtime_error("book boundary count error");
 }
 double prior(unsigned a,unsigned c)const{
  return (double(bigram[a][c])+0.5)/(double(rows[a])+0.5*N);
 }
};
struct Params { Matrix src{},dst{}; };
static Params load(Universe& u){
 Params p{};
 for(unsigned a=0;a<N;++a){
  const auto& h=u.root().h[a];
  for(int k=0;k<3;++k){
   float v=0,w=0;
   std::memcpy(&v,h.data+32+4*k,4);
   std::memcpy(&w,h.data+48+4*k,4);
   if(!std::isfinite(v)||!std::isfinite(w)||std::fabs(v)>2.1||std::fabs(w)>2.1)
    throw std::runtime_error("invalid pretrained head coordinate");
   p.src[a][k]=v;p.dst[a][k]=w;
  }
 }
 return p;
}
static void persist(Universe& u,const Params& p){
 for(unsigned a=0;a<N;++a)for(int k=0;k<3;++k){
  const float x=static_cast<float>(p.src[a][k]);
  const float y=static_cast<float>(p.dst[a][k]);
  std::memcpy(u.root().h[a].data+32+4*k,&x,4);
  std::memcpy(u.root().h[a].data+48+4*k,&y,4);
 }
 u.sync();
}
struct Metrics {
 double nll=0,soft_nll=0;
 uint64_t correct=0,items=0;
 double ce()const{return nll/double(items?items:1);}
 double soft_ce()const{return soft_nll/double(items?items:1);}
 double accuracy()const{return double(correct)/double(items?items:1);}
};
struct Score {Metrics metrics{};Matrix gsrc{},gdst{};};
static Score evaluate(const Corpus& c,const Params& p,
                      double beta,bool gradients,unsigned oldest_shift=0){
 if(!(beta>=0&&beta<1e6))throw std::invalid_argument("invalid distillation beta");
 Score out{};
 for(const auto& [ab,row]:c.ctx){
  unsigned a=(ab/N+oldest_shift)%N,b=ab%N;
  uint64_t count=0;
  for(const auto n:row)count+=n;
  if(!count)continue;
  Vec effective{};
  for(int k=0;k<3;++k)
   effective[k]=p.src[b][k]*(GAIN+CP_GAIN*p.src[a][k]);
  std::array<double,N> z{},probs{};
  double high=-std::numeric_limits<double>::infinity();
  unsigned winner=0;
  for(unsigned j=0;j<N;++j){
   for(int k=0;k<3;++k)z[j]+=effective[k]*p.dst[j][k];
   if(z[j]>high){high=z[j];winner=j;}
  }
  double total=0;
  for(unsigned j=0;j<N;++j){probs[j]=std::exp(z[j]-high);total+=probs[j];}
  for(double& x:probs)x/=total;
  Vec g_eff{};
  for(unsigned j=0;j<N;++j){
   const double q=(double(row[j])+beta*c.prior(b,j))/(double(count)+beta);
   const double logp=std::log(std::max(1e-300,probs[j]));
   out.metrics.nll-=double(row[j])*logp;
   out.metrics.soft_nll-=double(count)*q*logp;
   out.metrics.correct+=uint64_t(j==winner)*row[j];
   if(!gradients)continue;
   const double residual=(double(count)*probs[j]-double(count)*q)/double(c.triples);
   for(int k=0;k<3;++k){
    out.gdst[j][k]+=residual*effective[k];
    g_eff[k]+=residual*p.dst[j][k];
   }
  }
  if(gradients)for(int k=0;k<3;++k){
   out.gsrc[b][k]+=g_eff[k]*(GAIN+CP_GAIN*p.src[a][k]);
   out.gsrc[a][k]+=g_eff[k]*CP_GAIN*p.src[b][k];
  }
  out.metrics.items+=count;
 }
 if(out.metrics.items!=c.triples||!std::isfinite(out.metrics.nll)||
    !std::isfinite(out.metrics.soft_nll))
  throw std::runtime_error("nonfinite or incomplete context prediction");
 return out;
}
static Params update(const Params& original,const Score& grad,double rate){
 Params p=original;
 for(unsigned i=0;i<N;++i)for(int k=0;k<3;++k){
  p.src[i][k]=std::clamp(original.src[i][k]-rate*grad.gsrc[i][k],-2.0,2.0);
  p.dst[i][k]=std::clamp(original.dst[i][k]-rate*grad.gdst[i][k],-2.0,2.0);
 }
 return p;
}
static void metrics(const std::string& name,const Corpus& c,const Metrics& m){
 std::cout<<std::setprecision(12)<<"HW5_CP_SCORE"
  <<" mode="<<name<<" ce_nats="<<m.ce()
  <<" accuracy="<<m.accuracy()<<" soft_ce_nats="<<m.soft_ce()
  <<" transitions="<<m.items<<" books="<<c.books
  <<" context_rows="<<c.ctx.size()
  <<" bptt_attempted=0 inference_trigram_lookup=0 root_bytes="<<ROOT_BYTES<<"\n";
}
int main(int argc,char** argv){
 try {
  if(argc<3)throw std::invalid_argument(
   "init ROOT | train ROOT TRAIN LENGTHS [BETA] | score ROOT CORPUS LENGTHS [correct|wrong]");
  const std::string cmd=argv[1];
  if(cmd=="init"){
   if(argc!=3)throw std::invalid_argument("init ROOT");
   Universe u(argv[2]);FabricTeacher teacher(u);
   u.sync();std::cout<<"HW5_CP_INITIALIZED bytes="<<ROOT_BYTES<<"\n";
   return 0;
  }
  if(argc<5)throw std::invalid_argument("train|score ROOT CORPUS LENGTHS");
  const Corpus c(argv[3],argv[4]);
  Universe u(argv[2]);FabricTeacher teacher(u);
  const Params original=load(u);
  const double beta=cmd=="train"?(argc==6?std::stod(argv[5]):16.0):16.0;
  if(cmd=="score"){
   if(argc>6)throw std::invalid_argument("score wrong arity");
   const std::string mode=argc==6?argv[5]:"correct";
   if(mode!="correct"&&mode!="wrong")throw std::invalid_argument("invalid score mode");
   const auto s=evaluate(c,original,beta,false,mode=="wrong"?17:0);
   metrics(mode,c,s.metrics);
   return 0;
  }
  if(cmd!="train"||argc>6)throw std::invalid_argument("invalid command");
  const auto before=evaluate(c,original,beta,true);
  double largest=0;
  for(unsigned i=0;i<N;++i)for(int k=0;k<3;++k)
   largest=std::max({largest,std::fabs(before.gsrc[i][k]),
      std::fabs(before.gdst[i][k])});
  if(!std::isfinite(largest))throw std::runtime_error("nonfinite geometry gradient");
  bool accepted=false;double step=0;Metrics after=before.metrics;
  if(largest>1.e-16){
   const double base=std::min(30.0,0.13/largest);
   for(int attempt=0;attempt<15;++attempt){
    const double rate=std::ldexp(base,-attempt);
    Params candidate=update(original,before,rate);
    const auto trial=evaluate(c,candidate,beta,false);
    if(trial.metrics.soft_ce()+1.e-10<before.metrics.soft_ce()){
     persist(u,candidate);after=trial.metrics;step=rate;accepted=true;break;
    }
   }
  }
  std::cout<<std::setprecision(12)<<"HW5_CP_TRAIN"
   <<" before_ce="<<before.metrics.ce()<<" after_ce="<<after.ce()
   <<" before_soft_ce="<<before.metrics.soft_ce()
   <<" after_soft_ce="<<after.soft_ce()
   <<" before_accuracy="<<before.metrics.accuracy()
   <<" after_accuracy="<<after.accuracy()
   <<" grad_max="<<largest<<" step="<<step
   <<" accepted="<<(accepted?1:0)<<" beta="<<beta
   <<" transitions="<<c.triples<<" books="<<c.books
   <<" context_rows="<<c.ctx.size()
   <<" bptt_attempted=0 root_bytes="<<ROOT_BYTES<<"\n";
  return 0;
 }catch(const std::exception& e){
  std::cerr<<"HW5_CP_ERROR "<<e.what()<<"\n";return 2;
 }
}
