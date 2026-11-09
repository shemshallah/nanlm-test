#pragma once
#include "nanlm/universe.hpp"
#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace nanlm {
// One resident HW0:0 universe; no second model or duplicated state.
// The heads store trainable 3D source/target embeddings in their existing 128-byte records.
// Per-pair typed evidence is the residual fabric. Gauge-covariant phases are shared
// geometry, not 256x256 unaccounted per-token tensors.
class FabricTeacher {
 public:
 static constexpr unsigned VOCAB=255; // raw byte 0xFF is reserved for HW0 nannet ingress
 struct Prediction {
  std::array<double,VOCAB> probability{};
  std::array<double,VOCAB> logit{};
  uint8_t choice=0;
  double entropy=0;
 };
 struct Update {
  double before=0,after=0;
  bool accepted=false;
 };
 private:
 Universe& u_;
 double lr_;
 double angular_weight_;
 double evidence_weight_;
 static constexpr double DOT_GAIN=4.0;
 using Vector=std::array<float,3>;
 using Matrix=std::array<Vector,VOCAB>;
 static Vector get(const Head& h,size_t offset){
  Vector v{};
  std::memcpy(v.data(),h.data+offset,sizeof(float)*3);
  return v;
 }
 static void put(Head& h,size_t offset,const Vector& v){
  std::memcpy(h.data+offset,v.data(),sizeof(float)*3);
 }
 static float dot(const Vector& a,const Vector& b){
  return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
 }
 static double norm_angle(const Vector& a,const Vector& b){
  return wrap(std::atan2(static_cast<double>(b[1]),static_cast<double>(b[0]))-
              std::atan2(static_cast<double>(a[1]),static_cast<double>(a[0])));
 }
 static double clip(double x){return std::clamp(x,-2.0,2.0);}
 uint32_t self_count(uint8_t h)const{
  uint32_t count=0;
  std::memcpy(&count,u_.root().h[h].data+24,4);
  return count;
 }
 uint16_t gates(uint8_t h)const{
  uint16_t bits=0;
  std::memcpy(&bits,u_.root().h[h].data+4,2);
  return bits;
 }
 double score(uint8_t previous,uint8_t candidate,const Vector& src,const Vector& dst,const Vec3& rot)const{
  if((gates(previous)&static_cast<uint16_t>(1u<<(candidate&15)))==0)return -30.;
  const double coupling=DOT_GAIN*dot(src,dst);
  const double phase=previous==candidate?0.:u_.covariant_phase(previous,candidate);
  const double angle=norm_angle(src,dst);
  const double geometry=angular_weight_*std::cos(phase-angle);
  const double evidence=previous==candidate?self_count(previous):
                          u_.query(previous,candidate,PRECEDES).support;
  // Formal refutations are NOT treated as negative language examples.
  const double triply_rotated=0.12*(rot.x*dst[0]+rot.y*dst[1]+rot.z*dst[2]);
  return coupling+geometry+triply_rotated+evidence_weight_*std::log1p(evidence);
 }
 static double logloss(const std::array<double,VOCAB>& logits,uint8_t label){
  const double m=*std::max_element(logits.begin(),logits.end());
  double sum=0;
  for(double x:logits)sum+=std::exp(x-m);
  return std::log(sum)+m-logits[label];
 }
 public:
 explicit FabricTeacher(Universe& u,double learning_rate=0.055,
                        double angular_weight=0.08,double evidence_weight=0.09):
  u_(u),lr_(learning_rate),angular_weight_(angular_weight),evidence_weight_(evidence_weight){
  if(lr_<=0||lr_>1||angular_weight_<0||evidence_weight_<0)
   throw std::invalid_argument("invalid learning rate or gain");
  // Fixed, reproducible, deterministic state for previously blank heads.
  for(unsigned i=0;i<VOCAB;++i){
   Head& h=u_.root().h[i];
   if(h.data[3]!=0)continue;
   const double a=TAU*(i+0.5)/VOCAB;
   const Vector in{static_cast<float>(0.22*std::cos(a)),
                   static_cast<float>(0.22*std::sin(a)),
                   static_cast<float>(0.15*std::cos(3*a))};
   const Vector out{static_cast<float>(0.22*std::cos(2*a)),
                    static_cast<float>(0.22*std::sin(2*a)),
                    static_cast<float>(0.15*std::sin(3*a))};
   put(h,32,in);put(h,48,out);
   const uint16_t enabled=0xffff;std::memcpy(h.data+4,&enabled,2);
   h.data[3]=1;
  }
 }
 Prediction predict(uint8_t previous)const{
  if(previous==255)throw std::invalid_argument("nannet head is not a language symbol");
  Prediction out;
  const auto src=get(u_.root().h[previous],32);
  const Vec3 rotor=u_.transformed(previous);
  double best=-std::numeric_limits<double>::infinity();
  double max_logit=-std::numeric_limits<double>::infinity();
  for(unsigned j=0;j<VOCAB;++j){
   const auto dst=get(u_.root().h[j],48);
   const double z=score(previous,static_cast<uint8_t>(j),src,dst,rotor);
   out.logit[j]=z;
   if(z>best){best=z;out.choice=static_cast<uint8_t>(j);}
   max_logit=std::max(max_logit,z);
  }
  double normalizer=0;
  for(unsigned j=0;j<VOCAB;++j){
   out.probability[j]=std::exp(out.logit[j]-max_logit);
   normalizer+=out.probability[j];
  }
  for(auto& p:out.probability){
   p/=normalizer;
   if(p>0)out.entropy-=p*std::log(p);
  }
  return out;
 }
 double loss(uint8_t previous,uint8_t next)const{
  if(next>=VOCAB)throw std::invalid_argument("nannet head is not a language symbol");
  const auto p=predict(previous);
  return -std::log(std::max(p.probability[next],1e-300));
 }
 Update teach(uint8_t previous,uint8_t next,uint64_t time){
  if(previous>=VOCAB||next>=VOCAB)throw std::invalid_argument("nannet head is reserved");
  const auto prediction=predict(previous);
  const double before=-std::log(std::max(prediction.probability[next],1e-300));
  const Vector old_src=get(u_.root().h[previous],32);
  const Vec3 rotor=u_.transformed(previous);
  const std::array<double,3> rotor_vec{rotor.x,rotor.y,rotor.z};
  Matrix old_dst{};
  Vector grad_src{};
  Matrix grad_dst{};
  for(unsigned j=0;j<VOCAB;++j)old_dst[j]=get(u_.root().h[j],48);
  for(unsigned j=0;j<VOCAB;++j){
   const double residual=prediction.probability[j]-(j==next?1.0:0.0);
   for(unsigned k=0;k<3;++k){
    grad_src[k]+=static_cast<float>(DOT_GAIN*residual*old_dst[j][k]);
    grad_dst[j][k]=static_cast<float>(residual*(DOT_GAIN*old_src[k]+0.12*rotor_vec[k]));
   }
  }
  // Evaluate proposed parameters in scratch, then commit atomically with respect
  // to this one-thread learner. No temporary mutation of the resident root.
  for(int attempt=0;attempt<9;++attempt){
   const double rate=std::ldexp(lr_,-attempt);
   Vector candidate_src{};
   Matrix candidate_dst{};
   std::array<double,VOCAB> trial_logits{};
   for(unsigned k=0;k<3;++k)candidate_src[k]=
     static_cast<float>(clip(old_src[k]-rate*grad_src[k]));
   for(unsigned j=0;j<VOCAB;++j){
    for(unsigned k=0;k<3;++k)candidate_dst[j][k]=
      static_cast<float>(clip(old_dst[j][k]-rate*grad_dst[j][k]));
    const double base=prediction.logit[j];
    const double old_dot=DOT_GAIN*dot(old_src,old_dst[j]);
    const double new_dot=DOT_GAIN*dot(candidate_src,candidate_dst[j]);
    const double old_angle=norm_angle(old_src,old_dst[j]);
    const double new_angle=norm_angle(candidate_src,candidate_dst[j]);
    const double phase=previous==j?0.0:u_.covariant_phase(previous,static_cast<uint8_t>(j));
    trial_logits[j]=base-old_dot+new_dot+
      angular_weight_*(std::cos(phase-new_angle)-std::cos(phase-old_angle))+
      0.12*(rotor_vec[0]*(candidate_dst[j][0]-old_dst[j][0])+
            rotor_vec[1]*(candidate_dst[j][1]-old_dst[j][1])+
            rotor_vec[2]*(candidate_dst[j][2]-old_dst[j][2]));
   }
   const double after=logloss(trial_logits,next);
   if(after+1e-12<before){
    put(u_.root().h[previous],32,candidate_src);
    for(unsigned j=0;j<VOCAB;++j)put(u_.root().h[j],48,candidate_dst[j]);
    // A committed head update also teaches a directed inverse-addressed
    // relation and locally corrects the pair's geometric connection.
    if(previous!=next){
     const double original=u_.covariant_phase(previous,next);
     const double target=norm_angle(candidate_src,candidate_dst[next]);
     u_.set_covariant_phase(previous,next,wrap(original+0.08*wrap(target-original)));
     const float evidence=u_.query(previous,next,PRECEDES).support;
     u_.trial(previous,previous,next,PRECEDES,false,0.125f,evidence+0.125f,time);
    }else{
     auto& h=u_.root().h[previous];
     uint32_t count=self_count(previous);
     ++count;std::memcpy(h.data+24,&count,4);
    }
    u_.observe(next);
    return Update{before,after,true};
   }
  }
  return Update{before,before,false};
 }
};
} // namespace nanlm
