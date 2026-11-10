// HW5:0 immutable WordNet teacher -> ONE sequential corpus scan ->
// one aggregate geometric-gradient update in a SEPARATE existing NaNLM root.
// This is a bounded local FabricTeacher head-vector credit optimization,
// not a second TRAIL implementation and not full per-byte BPTT training.
// The source HW5 root and exact ledger are NEVER opened for writing.
#include "nanlm/unified_wave.hpp"
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
using namespace nanlm;
static constexpr unsigned V=255;
using Counts=std::array<std::array<uint64_t,V>,V>;
using Vec=std::array<double,3>;
static Vec read_vector(const Head& h,size_t off){
 std::array<float,3> f{};
 std::memcpy(f.data(),h.data+off,12);
 return {f[0],f[1],f[2]};
}
static void write_vector(Head& h,size_t off,const Vec& a){
 std::array<float,3> f{};
 for(size_t k=0;k<3;++k)
  f[k]=static_cast<float>(std::clamp(a[k],-2.0,2.0));
 std::memcpy(h.data+off,f.data(),12);
}
static double evaluate_ce(const Counts& counts,const FabricTeacher& teacher,
                          const std::array<uint64_t,V>& rows,
                          uint64_t total,double* accuracy=nullptr){
 double nll=0,correct=0;
 for(unsigned i=0;i<V;++i){
  if(!rows[i])continue;
  const auto predicted=teacher.predict(uint8_t(i));
  correct+=static_cast<double>(counts[i][predicted.choice]);
  for(unsigned j=0;j<V;++j)
   if(counts[i][j])
    nll-=double(counts[i][j])*std::log(
      std::max(predicted.probability[j],1e-300));
 }
 if(accuracy)*accuracy=correct/std::max<uint64_t>(1,total);
 return nll/double(std::max<uint64_t>(1,total));
}
int main(int argc,char** argv){
 try{
  if(argc!=4)throw std::invalid_argument(
      "usage: hw5-onepass train|eval LEARNER_ROOT GUTENBERG_SPLIT");
  const std::string mode=argv[1];
  if(mode!="train" && mode!="eval")throw std::invalid_argument("mode must be train or eval");
  const std::string model=argv[2];
  const std::string source=argv[3];
  if(model==source ||model.find("gutenberg-c256.root")!=std::string::npos ||model.find("wordnet-c256.root")!=std::string::npos)
   throw std::invalid_argument("immutable HW5 reference cannot be trained");
  Counts counts{};
  std::array<uint64_t,V> row{};
  std::ifstream input(source,std::ios::binary);
  if(!input)throw std::runtime_error("HW5 corpus does not exist");
  uint64_t seen=0,transition_count=0;
  int previous=-1;
  std::array<char,1<<20> buffer{};
  while(input){
   input.read(buffer.data(),static_cast<std::streamsize>(buffer.size()));
   const auto read=input.gcount();
   for(std::streamsize k=0;k<read;++k){
    const unsigned symbol=static_cast<unsigned char>(buffer[size_t(k)]);
    if(symbol>=V)throw std::runtime_error("HW5 full text contains ingress byte 0xff");
    ++seen;
    if(previous>=0){
     ++counts[unsigned(previous)][symbol];
     ++row[unsigned(previous)];
     ++transition_count;
    }
    previous=int(symbol);
   }
  }
  if(input.bad()||seen<1024||transition_count+1!=seen)
   throw std::runtime_error("incomplete single-pass HW5 read");
  Universe u(model);
  UnifiedWaveLanguage mind(u); // existing 8 MiB HW0 student layout
  FabricTeacher teacher(u);
  std::array<Vec,V> src{},dst{},gsrc{},gdst{};
  for(unsigned i=0;i<V;++i){
   src[i]=read_vector(u.root().h[i],32);
   dst[i]=read_vector(u.root().h[i],48);
  }
  double accuracy_before=0;
  const double before=evaluate_ce(counts,teacher,row,transition_count,
                                  &accuracy_before);
  if(mode=="eval"){
    uint64_t populated=0;
    for(unsigned i=0;i<V;++i)
     for(unsigned j=0;j<V;++j)populated+=counts[i][j]>0;
    std::cout<<std::setprecision(10)
     <<"HW5_GEOMETRIC_EVAL"
     <<" input_bytes="<<seen<<" transitions="<<transition_count
     <<" nonzero_pairs="<<populated
     <<" ce_nats="<<before<<" accuracy="<<accuracy_before
     <<" bptt_attempted=0 model_role=one_nanlm_student"
     <<" source_role=HW5_gutenberg_prose\n";
    return 0;
  }
  // CREDIT: rank-three geometric derivative of aggregate next-byte CE.
  // Unlike per-byte BPTT, this computes one exact aggregate gradient from
  // one full corpus pass. All samples influence the normalized residual.
  for(unsigned i=0;i<V;++i){
   if(!row[i])continue;
   const auto p=teacher.predict(uint8_t(i));
   for(unsigned j=0;j<V;++j){
    const double residual=4.0*(p.probability[j]*
           double(row[i])-double(counts[i][j]))/double(transition_count);
    for(unsigned k=0;k<3;++k){
     gsrc[i][k]+=residual*dst[j][k];
     gdst[j][k]+=residual*src[i][k];
    }
   }
  }
  double max_gradient=0;
  for(unsigned i=0;i<V;++i)
   for(unsigned k=0;k<3;++k)
    max_gradient=std::max({max_gradient,std::fabs(gsrc[i][k]),
                          std::fabs(gdst[i][k])});
  if(!std::isfinite(max_gradient)||max_gradient<1e-15)
   throw std::runtime_error("degenerate HW5 geometry credit");
  // Normalize one global credit step so no individual 3D head component
  // moves by >0.18. A scratch acceptance gate keeps the full-root update
  // only when the actual native teacher cross-entropy improves.
  const double base_step=0.18/max_gradient;
  double after=before,acc_after=accuracy_before,step=0;
  bool accepted=false;
  for(int retry=0;retry<16;++retry){
   const double trial=std::ldexp(base_step,-retry);
   for(unsigned i=0;i<V;++i){
    Vec a=src[i],b=dst[i];
    for(unsigned k=0;k<3;++k){
     a[k]-=trial*gsrc[i][k];
     b[k]-=trial*gdst[i][k];
    }
    write_vector(u.root().h[i],32,a);
    write_vector(u.root().h[i],48,b);
   }
   after=evaluate_ce(counts,teacher,row,transition_count,&acc_after);
   if(std::isfinite(after)&&after+1e-9<before){
    accepted=true;step=trial;break;
   }
   for(unsigned i=0;i<V;++i){
    write_vector(u.root().h[i],32,src[i]);
    write_vector(u.root().h[i],48,dst[i]);
   }
  }
  if(!accepted){
   after=before;acc_after=accuracy_before;
  }
  u.sync();
  uint64_t populated=0;
  for(unsigned i=0;i<V;++i)
   for(unsigned j=0;j<V;++j)populated+=counts[i][j]>0;
  std::cout<<std::setprecision(10)
   <<"HW5_FULL_GEOMETRIC_PASS"
   <<" input_bytes="<<seen<<" transitions="<<transition_count
   <<" nonzero_pairs="<<populated
   <<" train_before_ce="<<before<<" train_after_ce="<<after
   <<" train_before_accuracy="<<accuracy_before
   <<" train_after_accuracy="<<acc_after
   <<" aggregate_step_accepted="<<(accepted?1:0)
   <<" gradient_max="<<max_gradient<<" step="<<step
   <<" bptt_attempted=0 root_bytes="<<ROOT_BYTES
   <<" source_role=HW5_gutenberg_prose"
   <<" model_role=one_nanlm_student\n";
  if(!accepted)throw std::runtime_error(
      "full-pass geometric credit update rejected every scratch candidate");
  return 0;
 }catch(const std::exception& e){
  std::cerr<<"HW5_FULL_GEOMETRIC_ERROR "<<e.what()<<"\n";
  return 2;
 }
}
