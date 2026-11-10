// Full corpus, TWO-RAIL TRAIL temporal-difference eligibility sweep.
// One immutable HW4-derived source pass; one existing 8MiB learner root.
// No BPTT tape, no writes to NaNROUTER/HW4 source matrix or exact ledger.
// This extends the existing NaNLM head geometry with frozen-value batch
// credit, not a replay of all UnifiedWaveLanguage recurrent training steps.
#include "nanlm/unified_wave.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
using namespace nanlm;
namespace {
constexpr size_t V=255;
constexpr double GAMMA=0.93;
constexpr double LAMBDA=0.82;
constexpr double DECAY=GAMMA*LAMBDA;
using Vector=std::array<double,3>;
using Weights=std::array<Vector,V>;
using Probs=std::array<std::array<double,V>,V>;
using Counts=std::array<std::array<uint64_t,V>,V>;
Vector vec(const Head& h,size_t offset){
 std::array<float,3> f{};
 std::memcpy(f.data(),h.data+offset,sizeof(f));
 return {f[0],f[1],f[2]};
}
void store(Head& h,size_t offset,const Vector& v){
 std::array<float,3> f{};
 for(size_t k=0;k<3;++k)
  f[k]=static_cast<float>(std::clamp(v[k],-2.0,2.0));
 std::memcpy(h.data+offset,f.data(),sizeof(f));
}
double elapsed(std::chrono::steady_clock::time_point start){
 return std::chrono::duration<double>(
   std::chrono::steady_clock::now()-start).count();
}
double ce(const Counts& n,const std::array<uint64_t,V>& rows,
          FabricTeacher& teacher,uint64_t transitions,double& accuracy){
 long double total=0;
 uint64_t top1=0;
 for(size_t a=0;a<V;++a){
  if(rows[a]==0)continue;
  const auto p=teacher.predict(uint8_t(a));
  top1+=n[a][p.choice];
  for(size_t b=0;b<V;++b)
   if(n[a][b]>0)
    total-=long double(n[a][b])*
           std::log(std::max(p.probability[b],1e-300));
 }
 accuracy=double(top1)/double(transitions);
 return double(total/double(transitions));
}
struct TwoRail {
 Weights eligibility_plus{},eligibility_minus{},credit_plus{},credit_minus{};
 void advance(uint8_t previous,uint8_t current,
              const Probs& forward,const Probs& inverse,
              const Weights& source,const Weights& target){
  const auto& pf=forward[previous];
  const auto& pf_next=forward[current];
  const auto& pr=inverse[current];
  const auto& pr_next=inverse[previous];
  const auto& sv=source[previous];
  const auto& dv=target[current];
  for(size_t c=0;c<V;++c){
   // Geometric eligibility is the local derivative of the forward /
   // reverse conditional head value, not an activation-time BPTT tape.
   const double gp=4.0*pf[c]*(1.0-pf[c]);
   const double gm=4.0*pr[c]*(1.0-pr[c]);
   const double delta_plus=(c==current?1.0:0.0)+GAMMA*pf_next[c]-pf[c];
   const double delta_minus=(c==previous?1.0:0.0)+GAMMA*pr_next[c]-pr[c];
   for(size_t k=0;k<3;++k){
    double& ep=eligibility_plus[c][k];
    double& em=eligibility_minus[c][k];
    ep=DECAY*ep+gp*sv[k];
    em=DECAY*em+gm*dv[k];
    credit_plus[c][k]+=delta_plus*ep;
    credit_minus[c][k]+=delta_minus*em;
   }
  }
 }
};
} // namespace

int main(int argc,char** argv){
 try {
  if(argc!=4)
   throw std::invalid_argument(
      "usage: hw4-trail LEARNER_ROOT FULL_HW4_CORPUS EXPECTED_BYTES");
  const auto started=std::chrono::steady_clock::now();
  const std::filesystem::path learner=argv[1],corpus=argv[2];
  const uint64_t expected=std::stoull(argv[3]);
  if(expected<1024||expected>1024ull*1024ull*1024ull)
   throw std::invalid_argument("invalid complete corpus byte count");
  if(learner==corpus||
     learner.filename()=="wordnet-c256.root"||
     learner.string().find("hw4-full-reference")!=std::string::npos||
     corpus==std::filesystem::path(learner.string()+".overlay"))
   throw std::invalid_argument("HW4 immutable reference may not be updated");
  if(!std::filesystem::is_regular_file(corpus)||
     std::filesystem::file_size(corpus)!=expected)
   throw std::runtime_error("HW4 verified corpus byte count mismatch");
  if(!std::filesystem::is_regular_file(learner)||
     std::filesystem::file_size(learner)!=ROOT_BYTES)
   throw std::runtime_error("start with existing 8MiB geometric student root");
  Universe u(learner.string());
  UnifiedWaveLanguage model(u);
  if(model.bptt_enabled())
   throw std::runtime_error("full TRAIL experiment unexpectedly enabled BPTT");
  FabricTeacher teacher(u);
  Weights src{},dst{};
  for(size_t h=0;h<V;++h){
   src[h]=vec(u.root().h[h],32);
   dst[h]=vec(u.root().h[h],48);
  }
  std::array<uint8_t,128> ingress{};
  std::memcpy(ingress.data(),u.root().h[255].data,ingress.size());
  Probs fwd{},rev{},scores{};
  const auto prep=std::chrono::steady_clock::now();
  for(size_t i=0;i<V;++i){
   const auto p=teacher.predict(uint8_t(i));
   for(size_t j=0;j<V;++j){
    fwd[i][j]=p.probability[j];
    scores[i][j]=p.logit[j];
   }
  }
  for(size_t b=0;b<V;++b){
   double largest=-std::numeric_limits<double>::infinity();
   for(size_t a=0;a<V;++a)largest=std::max(largest,scores[a][b]);
   double norm=0;
   for(size_t a=0;a<V;++a){
    rev[b][a]=std::exp(scores[a][b]-largest);
    norm+=rev[b][a];
   }
   for(double& v:rev[b])v/=norm;
  }
  const double precompute_seconds=elapsed(prep);
  const auto scan=std::chrono::steady_clock::now();
  TwoRail trail{};
  Counts counts{};
  std::array<uint64_t,V> rows{};
  std::ifstream input(corpus,std::ios::binary);
  if(!input)throw std::runtime_error("cannot open WordNet full corpus");
  std::array<char,1<<17> buffer{};
  int previous=-1;
  uint64_t read_bytes=0,transitions=0;
  while(input){
   input.read(buffer.data(),std::streamsize(buffer.size()));
   const auto n=input.gcount();
   for(std::streamsize i=0;i<n;++i){
    const auto current=uint8_t(
      static_cast<unsigned char>(buffer[size_t(i)]));
    if(current>=V)throw std::runtime_error("reserved ingress byte in corpus");
    ++read_bytes;
    if(previous>=0){
     ++counts[unsigned(previous)][current];
     ++rows[unsigned(previous)];
     trail.advance(uint8_t(previous),current,fwd,rev,src,dst);
     ++transitions;
    }
    previous=int(current);
   }
  }
  if(input.bad()||read_bytes!=expected||transitions+1!=read_bytes)
   throw std::runtime_error("full TRAIL credit pass incomplete");
  const double scan_seconds=elapsed(scan);
  const auto update=std::chrono::steady_clock::now();
  double old_accuracy=0,new_accuracy=0;
  const double initial_ce=ce(counts,rows,teacher,transitions,old_accuracy);
  double grad_max=0,positive_l1=0,negative_l1=0;
  for(size_t c=0;c<V;++c)
   for(size_t k=0;k<3;++k){
    trail.credit_plus[c][k]/=double(transitions);
    trail.credit_minus[c][k]/=double(transitions);
    const auto p=trail.credit_plus[c][k],m=trail.credit_minus[c][k];
    if(!std::isfinite(p)||!std::isfinite(m))
     throw std::runtime_error("nonfinite TRAIL credit");
    grad_max=std::max({grad_max,std::fabs(p),std::fabs(m)});
    positive_l1+=std::fabs(p);
    negative_l1+=std::fabs(m);
   }
  if(grad_max<1e-17)
   throw std::runtime_error("zero two-rail full-corpus eligibility credit");
  const double trial_max=0.13/grad_max;
  double actual_step=0;
  double after=initial_ce;
  bool accepted=false;
  for(int attempt=0;attempt<22;++attempt){
   const double step=std::ldexp(trial_max,-attempt);
   for(size_t c=0;c<V;++c){
    Vector sn=src[c],dn=dst[c];
    for(size_t k=0;k<3;++k){
     sn[k]-=step*trail.credit_minus[c][k];
     dn[k]+=step*trail.credit_plus[c][k];
    }
    store(u.root().h[c],32,sn);
    store(u.root().h[c],48,dn);
   }
   after=ce(counts,rows,teacher,transitions,new_accuracy);
   if(std::isfinite(after)&&after+1e-9<initial_ce){
    actual_step=step;
    accepted=true;
    break;
   }
   for(size_t c=0;c<V;++c){
    store(u.root().h[c],32,src[c]);
    store(u.root().h[c],48,dst[c]);
   }
  }
  if(!accepted){
   after=initial_ce;
   new_accuracy=old_accuracy;
  }
  if(std::memcmp(ingress.data(),u.root().h[255].data,ingress.size())!=0)
   throw std::runtime_error("TRAIL changed ingress head");
  u.sync();
  const double update_seconds=elapsed(update);
  const double total_seconds=elapsed(started);
  uint64_t nonzero=0;
  for(const auto& row:counts)
   for(uint64_t count:row)nonzero+=count>0;
  std::cout<<std::fixed<<std::setprecision(10)
   <<"HW4_FULL_TRAIL"
   <<" input_bytes="<<read_bytes
   <<" transition_updates="<<transitions
   <<" 255_heads_per_transition=1"
   <<" forward_rail=1 inverse_rail=1"
   <<" gamma="<<GAMMA<<" lambda="<<LAMBDA
   <<" geom_credit_plus_l1="<<positive_l1
   <<" geom_credit_minus_l1="<<negative_l1
   <<" max_credit="<<grad_max
   <<" nonzero_byte_pairs="<<nonzero
   <<" initial_teacher_ce="<<initial_ce
   <<" final_teacher_ce="<<after
   <<" initial_teacher_acc="<<old_accuracy
   <<" final_teacher_acc="<<new_accuracy
   <<" accepted="<<int(accepted)
   <<" step="<<actual_step
   <<" precompute_seconds="<<precompute_seconds
   <<" full_scan_seconds="<<scan_seconds
   <<" commit_seconds="<<update_seconds
   <<" total_seconds="<<total_seconds
   <<" corpus_bytes_per_second="<<(double(read_bytes)/scan_seconds)
   <<" bptt_attempted=0"
   <<" root_bytes="<<ROOT_BYTES
   <<" update_type=one_frozen_value_full_corpus_two_rail_geometric_batch"
   <<" hw4_read_only=1\n";
  return 0;
 }catch(const std::exception& e){
  std::cerr<<"HW4_FULL_TRAIL_ERROR "<<e.what()<<"\n";
  return 2;
 }
}
