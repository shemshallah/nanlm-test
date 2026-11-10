// Native full-capacity C(256,2) conditional-byte plane in the EXISTING
// 8 MiB NaNLM root. Mirrors share one Pair; both directions are independent.
// Never use this experimental plane with graph-journal workloads: slot 31
// is an opt-in language score bank, not a new production relation type.
//
// For a,b in 0..254: a!=b uses p[pair_index(a,b)].f[31] if a<b,
// else .r[31]. Diagonal a==b uses p[pair_index(a,255)].f[31].
// Thus all 32,640 pair records index 255^2 scores, and ingress head 255
// never becomes a language token. Unseen transitions get a PRIOR, not
// fabricated observations. Base logits are the restored FabricTeacher
// geometry; posterior target is (count + concentration*prior)/(row+conc).
#include "nanlm/unified_wave.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
using namespace nanlm;
namespace {
constexpr unsigned V=FabricTeacher::VOCAB;
constexpr size_t SLOT=31;
constexpr char MARKER[9]="PAIRC256";
using Counts=std::array<std::array<uint64_t,V>,V>;
using Matrix=std::array<std::array<double,V>,V>;
struct Corpus {
 Counts counts{};
 std::array<uint64_t,V> row{};
 uint64_t bytes=0,transitions=0;
};
Corpus count_bytes(const std::string& filename){
 std::ifstream source(filename,std::ios::binary);
 if(!source)throw std::runtime_error("cannot open Gutenberg split");
 Corpus out;
 int prev=-1;
 std::array<char,1<<20> buffer{};
 while(source){
  source.read(buffer.data(),static_cast<std::streamsize>(buffer.size()));
  for(std::streamsize k=0;k<source.gcount();++k){
   const unsigned b=static_cast<unsigned char>(buffer[static_cast<size_t>(k)]);
   if(b>=V)throw std::runtime_error("0xFF is a reserved network byte");
   ++out.bytes;
   if(prev>=0){
    ++out.counts[static_cast<unsigned>(prev)][b];
    ++out.row[static_cast<unsigned>(prev)];
    ++out.transitions;
   }
   prev=static_cast<int>(b);
  }
 }
 if(source.bad()||out.bytes<64||out.transitions+1!=out.bytes)
  throw std::runtime_error("incomplete corpus scan");
 return out;
}
float& bias(Root& root,unsigned a,unsigned b){
 if(a>=V||b>=V)throw std::out_of_range("network head not language");
 if(a==b)return root.p[pair_index(static_cast<uint8_t>(a),255)].f[SLOT];
 Pair& p=root.p[pair_index(static_cast<uint8_t>(a),static_cast<uint8_t>(b))];
 return (a<b?p.f:p.r)[SLOT];
}
double bias(const Root& root,unsigned a,unsigned b){
 if(a==b)return root.p[pair_index(static_cast<uint8_t>(a),255)].f[SLOT];
 const Pair& p=root.p[pair_index(static_cast<uint8_t>(a),static_cast<uint8_t>(b))];
 return (a<b?p.f:p.r)[SLOT];
}
bool trained(const Root& root){
 return std::memcmp(root.h[255].data+112,MARKER,8)==0;
}
struct Score {
 double ce=0,accuracy=0;
};
Score score(const Corpus& corpus,const Matrix& base,const Root& root,bool full){
 double nll=0,correct=0;
 for(unsigned a=0;a<V;++a){
  if(!corpus.row[a])continue;
  double peak=-std::numeric_limits<double>::infinity();
  unsigned choice=0;
  for(unsigned b=0;b<V;++b){
   const double z=base[a][b]+(full?bias(root,a,b):0.);
   if(z>peak){peak=z;choice=b;}
  }
  double norm=0;
  for(unsigned b=0;b<V;++b)
   norm+=std::exp(base[a][b]+(full?bias(root,a,b):0.)-peak);
  const double logsum=peak+std::log(norm);
  correct+=static_cast<double>(corpus.counts[a][choice]);
  for(unsigned b=0;b<V;++b)if(corpus.counts[a][b]){
   const double z=base[a][b]+(full?bias(root,a,b):0.);
   nll+=static_cast<double>(corpus.counts[a][b])*(logsum-z);
  }
 }
 return {nll/static_cast<double>(corpus.transitions),
         correct/static_cast<double>(corpus.transitions)};
}
struct Coverage {
 size_t physical=0,observed_physical=0,observed_directed=0;
 unsigned contexts=0;
};
Coverage coverage(const Corpus& c){
 std::array<bool,PAIRS> used{},observed{};
 size_t directed=0;
 for(unsigned a=0;a<V;++a){
  if(c.row[a])++directed; // temporary context counter
  for(unsigned b=0;b<V;++b){
   const size_t k=a==b?pair_index(static_cast<uint8_t>(a),255):
      pair_index(static_cast<uint8_t>(a),static_cast<uint8_t>(b));
   used[k]=true;
   if(c.counts[a][b])observed[k]=true;
  }
 }
 Coverage out;
 out.physical=static_cast<size_t>(std::count(used.begin(),used.end(),true));
 out.observed_physical=static_cast<size_t>(std::count(observed.begin(),observed.end(),true));
 for(unsigned a=0;a<V;++a)
  for(unsigned b=0;b<V;++b)out.observed_directed+=c.counts[a][b]>0;
 out.contexts=static_cast<unsigned>(directed);
 return out;
}
}
int main(int argc,char** argv){
 try{
  if(argc<4||argc>5)throw std::invalid_argument(
   "usage: hw5-fullpair train|eval ROOT BOOK_SPLIT [concentration]");
  const std::string mode=argv[1],model=argv[2],source=argv[3];
  if(mode!="train"&&mode!="eval")throw std::invalid_argument("mode must be train or eval");
  if(mode=="train"&&argc!=5)throw std::invalid_argument("train requires positive concentration");
  if(mode=="eval"&&argc!=4)throw std::invalid_argument("eval takes no concentration");
  if(model==source||model.find("gutenberg-c256.root")!=std::string::npos||
     model.find("wordnet-c256.root")!=std::string::npos)
   throw std::invalid_argument("reference corpus and root are immutable");
  const Corpus c=count_bytes(source);
  const Coverage cv=coverage(c);
  if(cv.physical!=PAIRS)throw std::runtime_error("incomplete C256 mapping");
  Universe u(model);
  UnifiedWaveLanguage mind(u);
  FabricTeacher teacher(u);
  const bool present=trained(u.root());
  Matrix logits{},probabilities{};
  for(unsigned a=0;a<V;++a){
   const auto p=teacher.predict(static_cast<uint8_t>(a));
   logits[a]=p.logit;
   probabilities[a]=p.probability;
  }
  const Score before=score(c,logits,u.root(),present);
  if(mode=="eval"){
   std::cout<<std::setprecision(11)<<"HW5_FULLPAIR_EVAL"
    <<" input_bytes="<<c.bytes<<" transitions="<<c.transitions
    <<" ce_nats="<<before.ce<<" accuracy="<<before.accuracy
    <<" pair_plane="<<(present?1:0)
    <<" physical_pairs="<<cv.physical
    <<" observed_pair_cells="<<cv.observed_physical
    <<" observed_directed_transitions="<<cv.observed_directed
    <<" active_contexts="<<cv.contexts
    <<" root_bytes="<<ROOT_BYTES
    <<" bptt_attempted=0\n";
   return 0;
  }
  if(present)throw std::invalid_argument(
   "full-pair root already trained; clone the unmodified parent checkpoint");
  const double concentration=std::stod(argv[4]);
  if(!std::isfinite(concentration)||concentration<1.||concentration>1e7)
   throw std::invalid_argument("concentration must be in [1,1e7]");
  // Existing isolated student roots must not have relation evidence in
  // the same storage lane. The identity mark at 120..127 stays intact.
  for(const auto& p:u.root().p)
   if(p.f[SLOT]!=0.f||p.r[SLOT]!=0.f)
    throw std::runtime_error("pair-language plane would overwrite graph evidence");
  for(unsigned k=0;k<8;++k)
   if(u.root().h[255].data[112+k]!=0)
    throw std::runtime_error("gateway marker space is already occupied");
  Root& root=u.root();
  size_t nonzero=0;
  for(unsigned a=0;a<V;++a){
   const double denom=static_cast<double>(c.row[a])+concentration;
   for(unsigned b=0;b<V;++b){
    const double q=(static_cast<double>(c.counts[a][b])+
       concentration*probabilities[a][b])/denom;
    if(!(q>0.)||!std::isfinite(q))
     throw std::runtime_error("invalid Dirichlet-posterior target");
    const double delta=std::log(q)-logits[a][b];
    if(!std::isfinite(delta)||std::fabs(delta)>100.)
     throw std::runtime_error("pair logit residual nonfinite/out of range");
    bias(root,a,b)=static_cast<float>(delta);
    nonzero+=std::fabs(delta)>1e-7;
   }
  }
  std::memcpy(root.h[255].data+112,MARKER,8);
  const Score after=score(c,logits,root,true);
  if(!std::isfinite(after.ce)||after.ce>before.ce+1e-5)
   throw std::runtime_error("pair-fitting acceptance did not improve train CE");
  u.sync();
  std::cout<<std::setprecision(11)<<"HW5_FULLPAIR_TRAIN"
   <<" input_bytes="<<c.bytes<<" transitions="<<c.transitions
   <<" ce_before="<<before.ce<<" ce_after="<<after.ce
   <<" accuracy_before="<<before.accuracy
   <<" accuracy_after="<<after.accuracy
   <<" concentration="<<concentration
   <<" physical_pairs="<<cv.physical
   <<" observed_pair_cells="<<cv.observed_physical
   <<" observed_directed_transitions="<<cv.observed_directed
   <<" active_contexts="<<cv.contexts
   <<" learned_directional_logits="<<nonzero
   <<" root_bytes="<<ROOT_BYTES
   <<" bptt_attempted=0"
   <<" optimizer=aggregate_dirichlet_geometric_prior\n";
  return 0;
 }catch(const std::exception& e){
  std::cerr<<"HW5_FULLPAIR_ERROR "<<e.what()<<"\n";
  return 2;
 }
}
