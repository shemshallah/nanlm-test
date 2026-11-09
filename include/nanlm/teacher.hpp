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
 double dot_gain_;
 double quaternion_weight_;
 double rule_gain_;
 bool train_heads_;
 double feature_rotor_gain_;
 static constexpr uint8_t RULE_HEAD=PRECEDES;
 static constexpr uint8_t RULE_INVERSE_HEAD=FOLLOWS;
 static constexpr size_t RULE_QUAT_OFFSET=96;
 static constexpr size_t RULE_INIT_OFFSET=112;
 static constexpr uint8_t RULE_MAGIC=0xA5;
 static Vec3q ring(uint8_t byte) {
  // A fixed circular coordinate prior, not learned natural-language semantics.
  const double theta=TAU*(static_cast<double>(byte)+0.5)/VOCAB;
  return {std::cos(theta),std::sin(theta),0.0};
 }
 Quat shared_rule()const{
  Quat q{};
  std::memcpy(&q,u_.root().h[RULE_HEAD].data+RULE_QUAT_OFFSET,sizeof(q));
  return q.normalized();
 }
 bool teach_shared_rule(uint8_t previous,uint8_t next) {
  if(rule_gain_<=0)return false;
  const Prediction p=predict(previous);
  const Quat old_q=shared_rule();
  const Vec3q old_v=old_q.rotate(ring(previous));
  const Vec3q tangent{-old_v.y,old_v.x,0};
  double gradient=0;
  for(unsigned j=0;j<VOCAB;++j) {
   const double dz=rule_gain_*nanlm::dot(tangent,ring(static_cast<uint8_t>(j)));
   gradient+=(p.probability[j]-(j==next?1.0:0.0))*dz;
  }
  const double original=-std::log(std::max(p.probability[next],1e-300));
  for(int retry=0;retry<10;++retry) {
   const double step=std::clamp(std::ldexp(-lr_*gradient,-retry),-0.30,0.30);
   if(!std::isfinite(step)||std::fabs(step)<1e-13)continue;
   const Quat candidate=Quat::axis_angle({0,0,1},step)*old_q;
   const Vec3q moved=candidate.rotate(ring(previous));
   auto logits=p.logit;
   for(unsigned j=0;j<VOCAB;++j) {
    const Vec3q d=ring(static_cast<uint8_t>(j));
    logits[j]+=rule_gain_*(nanlm::dot(moved,d)-nanlm::dot(old_v,d));
   }
   if(logloss(logits,next)+1e-12<original) {
    std::memcpy(u_.root().h[RULE_HEAD].data+RULE_QUAT_OFFSET,
                &candidate,sizeof(candidate));
    const Quat inverse=candidate.conjugate();
    std::memcpy(u_.root().h[RULE_INVERSE_HEAD].data+RULE_QUAT_OFFSET,
                &inverse,sizeof(inverse));
    return true;
   }
  }
  return false;
 }
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
 double score(uint8_t previous,uint8_t candidate,const Vector& src,const Vector& dst,const Vec3& rot,bool bypass_gate=false)const{
  if(!bypass_gate && (gates(previous)&static_cast<uint16_t>(1u<<(candidate&15)))==0)return -30.;
  const double coupling=dot_gain_*dot(src,dst);
  const double phase=previous==candidate?0.:u_.covariant_phase(previous,candidate);
  const double angle=norm_angle(src,dst);
  const double geometry=angular_weight_*std::cos(phase-angle);
  const double evidence=previous==candidate?self_count(previous):
                          u_.query(previous,candidate,PRECEDES).support;
  // Formal refutations are NOT treated as negative language examples.
  const double triply_rotated=feature_rotor_gain_*(rot.x*dst[0]+rot.y*dst[1]+rot.z*dst[2]);
  const Vec3q shifted=u_.quaternion_field().get(previous,candidate).rotate(
    {dst[0],dst[1],dst[2]});
  const double transported=quaternion_weight_*(double(src[0])*shifted.x+
                                   double(src[1])*shifted.y+
                                   double(src[2])*shifted.z);
  const double rule=rule_gain_>0?
    rule_gain_*nanlm::dot(shared_rule().rotate(ring(previous)),ring(candidate)):0.0;
  return coupling+geometry+triply_rotated+transported+rule+
         evidence_weight_*std::log1p(evidence);
 }
 static double logloss(const std::array<double,VOCAB>& logits,uint8_t label){
  const double m=*std::max_element(logits.begin(),logits.end());
  double sum=0;
  for(double x:logits)sum+=std::exp(x-m);
  return std::log(sum)+m-logits[label];
 }
 public:
 explicit FabricTeacher(Universe& u,double learning_rate=0.055,
                        double angular_weight=0.08,double evidence_weight=0.09,
                        double quaternion_weight=0.45,double dot_gain=4.0,
                        double rule_gain=0.0,bool train_heads=true,
                        double feature_rotor_gain=0.12):
  u_(u),lr_(learning_rate),angular_weight_(angular_weight),
  evidence_weight_(evidence_weight),dot_gain_(dot_gain),
  quaternion_weight_(quaternion_weight),rule_gain_(rule_gain),
  train_heads_(train_heads),feature_rotor_gain_(feature_rotor_gain){
  if(!std::isfinite(lr_)||!std::isfinite(angular_weight_)||
     !std::isfinite(evidence_weight_)||!std::isfinite(dot_gain_)||
     !std::isfinite(quaternion_weight_)||!std::isfinite(rule_gain_)||
     !std::isfinite(feature_rotor_gain_)||lr_<=0||lr_>1||
     angular_weight_<0||evidence_weight_<0||dot_gain_<0||
     quaternion_weight_<0||rule_gain_<0||feature_rotor_gain_<0)
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
  // The shared PRECEDES relation rotor occupies free bytes in head 6.
  // It is part of the same mmap root, and does not steal a language symbol.
  if(rule_gain_>0) {
   Head& h=u_.root().h[RULE_HEAD];
   Head& inverse_head=u_.root().h[RULE_INVERSE_HEAD];
   if(h.data[RULE_INIT_OFFSET]!=RULE_MAGIC) {
    const Quat id=Quat::identity();
    std::memcpy(h.data+RULE_QUAT_OFFSET,&id,sizeof(id));
    h.data[RULE_INIT_OFFSET]=RULE_MAGIC;
   }
   // The inverse follows the forward operator, even when an older root
   // existed before the shared relation circuit was introduced.
   const Quat inverse=shared_rule().conjugate();
   std::memcpy(inverse_head.data+RULE_QUAT_OFFSET,&inverse,sizeof(inverse));
   inverse_head.data[RULE_INIT_OFFSET]=RULE_MAGIC;
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
 // The 16 Boolean selector bits are themselves self-modifying,
 // but only after evaluating the full normalized predictive objective
 // against a scratch version of the mask (never mutating the live head).
 bool trial_gate(uint8_t head,uint8_t nibble_group,bool enable,uint8_t expected,
                 double required_gain=1e-5){
  if(head>=VOCAB||expected>=VOCAB||nibble_group>=16)
   throw std::invalid_argument("invalid Boolean head trial");
  const uint16_t old_mask=gates(head);
  const uint16_t bit=static_cast<uint16_t>(1u<<nibble_group);
  const uint16_t new_mask=enable?static_cast<uint16_t>(old_mask|bit):
                                 static_cast<uint16_t>(old_mask&~bit);
  if(new_mask==old_mask)return false;
  const auto before=predict(head);
  const auto src=get(u_.root().h[head],32);
  const Vec3 rotor=u_.transformed(head);
  auto logits=before.logit; // scratch copied logits, no live-head write
  for(unsigned j=0;j<VOCAB;++j)if((j&15u)==nibble_group){
   const auto dst=get(u_.root().h[j],48);
   logits[j]=enable?score(head,static_cast<uint8_t>(j),src,dst,rotor,true):-30.;
  }
  const double before_loss=-std::log(std::max(before.probability[expected],1e-300));
  if(logloss(logits,expected)+required_gain>=before_loss)return false;
  std::memcpy(u_.root().h[head].data+4,&new_mask,2);
  return true;
 }
 Update teach(uint8_t previous,uint8_t next,uint64_t time){
  if(previous>=VOCAB||next>=VOCAB)throw std::invalid_argument("nannet head is reserved");
  const auto prediction=predict(previous);
  const double before=-std::log(std::max(prediction.probability[next],1e-300));
  // Diagnostic: isolate the shared relation operator from head-specific updates.
  // The operator still trains inside the same persistent physical head fabric.
  if(!train_heads_) {
   const bool accepted=teach_shared_rule(previous,next);
   return Update{before,loss(previous,next),accepted};
  }
  const Vector old_src=get(u_.root().h[previous],32);
  const Vec3 rotor=u_.transformed(previous);
  const std::array<double,3> rotor_vec{rotor.x,rotor.y,rotor.z};
  Matrix old_dst{};
  Vector grad_src{};
  Matrix grad_dst{};
  for(unsigned j=0;j<VOCAB;++j)old_dst[j]=get(u_.root().h[j],48);
  for(unsigned j=0;j<VOCAB;++j){
   const double residual=prediction.probability[j]-(j==next?1.0:0.0);
   const Quat q=u_.quaternion_field().get(previous,static_cast<uint8_t>(j));
   const Vec3q moved=q.rotate({old_dst[j][0],old_dst[j][1],old_dst[j][2]});
   const Vec3q backward=q.conjugate().rotate({old_src[0],old_src[1],old_src[2]});
   const std::array<double,3> move{moved.x,moved.y,moved.z};
   const std::array<double,3> back{backward.x,backward.y,backward.z};
   for(unsigned k=0;k<3;++k){
    grad_src[k]+=static_cast<float>(residual*(dot_gain_*old_dst[j][k]+quaternion_weight_*move[k]));
    grad_dst[j][k]=static_cast<float>(residual*(dot_gain_*old_src[k]+feature_rotor_gain_*rotor_vec[k]+quaternion_weight_*back[k]));
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
    const double old_dot=dot_gain_*dot(old_src,old_dst[j]);
    const double new_dot=dot_gain_*dot(candidate_src,candidate_dst[j]);
    const double old_angle=norm_angle(old_src,old_dst[j]);
    const double new_angle=norm_angle(candidate_src,candidate_dst[j]);
    const Quat q=u_.quaternion_field().get(previous,static_cast<uint8_t>(j));
    const Vec3q oq=q.rotate({old_dst[j][0],old_dst[j][1],old_dst[j][2]});
    const Vec3q nq=q.rotate({candidate_dst[j][0],candidate_dst[j][1],candidate_dst[j][2]});
    const double old_qdot=old_src[0]*oq.x+old_src[1]*oq.y+old_src[2]*oq.z;
    const double new_qdot=candidate_src[0]*nq.x+candidate_src[1]*nq.y+candidate_src[2]*nq.z;
    const double phase=previous==j?0.0:u_.covariant_phase(previous,static_cast<uint8_t>(j));
    trial_logits[j]=base-old_dot+new_dot+
      angular_weight_*(std::cos(phase-new_angle)-std::cos(phase-old_angle))+
      feature_rotor_gain_*(rotor_vec[0]*(candidate_dst[j][0]-old_dst[j][0])+
            rotor_vec[1]*(candidate_dst[j][1]-old_dst[j][1])+
            rotor_vec[2]*(candidate_dst[j][2]-old_dst[j][2]))+
      quaternion_weight_*(new_qdot-old_qdot);
   }
   const double after=logloss(trial_logits,next);
   if(after+1e-12<before){
    put(u_.root().h[previous],32,candidate_src);
    for(unsigned j=0;j<VOCAB;++j)put(u_.root().h[j],48,candidate_dst[j]);
    // The quaternion proposal is a true scratch trial: we modify only the
    // candidate's logit off-heap and never temporarily mutate the live field.
    if(previous!=next && quaternion_weight_>0){
     const auto p=predict(previous);
     const auto& src_head=u_.root().h[previous];
     const auto& dst_head=u_.root().h[next];
     const Vector s=get(src_head,32),d=get(dst_head,48);
     const Quat old_q=u_.quaternion_field().get(previous,next);
     const Vec3q old_moved=old_q.rotate({d[0],d[1],d[2]});
     const Quat correction=Quat::from_to(old_moved,{s[0],s[1],s[2]});
     const Quat proposed=Quat::slerp(old_q,correction*old_q,0.20);
     const Vec3q new_moved=proposed.rotate({d[0],d[1],d[2]});
     const double change=quaternion_weight_*(s[0]*(new_moved.x-old_moved.x)+
                                 s[1]*(new_moved.y-old_moved.y)+
                                 s[2]*(new_moved.z-old_moved.z));
     auto logits=p.logit;logits[next]+=change;
     if(logloss(logits,next)+1e-12 < -std::log(std::max(p.probability[next],1e-300)))
       u_.quaternion_field().set(previous,next,proposed);
    }
    // A committed head update also teaches a directed inverse-addressed
    // relation and locally corrects the scalar phase connection.
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
    teach_shared_rule(previous,next);
    // Return the real committed objective (after head, pair, phase,
    // quaternion, observation and shared-relation writes), not scratch CE.
    return Update{before,loss(previous,next),true};
   }
  }
  return Update{before,before,false};
 }
};
} // namespace nanlm
