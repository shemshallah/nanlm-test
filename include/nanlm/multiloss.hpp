#pragma once
#include "nanlm/teacher.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace nanlm {
// One joint update on the EXISTING HW0 pair fabric, never a separate model.
// A triad is composable only when its first two typed propositions have
// unequivocal support at the requested effective time. Both/false are not
// silently promoted to true. The third proposition may support, refute,
// both or be absent, contributing independent temporal terms.
class JointFieldTrainer {
 public:
 struct Weights {
  double prediction=0.20;
  double triangle=1.0;
  double temporal=0.80;
  double retention=0.08;
  double learning_rate=0.35;
 };
 struct Terms {
  double prediction=0,triangle=0,temporal=0,retention=0,total=0;
 };
 struct Result {
  bool eligible=false,accepted=false;
  Terms before{},after{};
 };
 private:
 Universe& u_;
 FabricTeacher& teacher_;
 Weights weights_;
 static constexpr double QUAT_GAIN=0.45;
 static std::array<float,3> read(const Head& h,size_t offset){
  std::array<float,3> out{};
  std::memcpy(out.data(),h.data+offset,sizeof(float)*3);
  return out;
 }
 static double nll_one_logit(const FabricTeacher::Prediction& old,
                            uint8_t label,double delta){
  // LogSumExp stable even after a large signed candidate perturbation.
  const double original=old.logit[label];
  double m=-1e300;
  for(unsigned i=0;i<FabricTeacher::VOCAB;++i)
   m=std::max(m,old.logit[i]+(i==label?delta:0.0));
  double sum=0;
  for(unsigned i=0;i<FabricTeacher::VOCAB;++i)
   sum+=std::exp(old.logit[i]+(i==label?delta:0.0)-m);
  return std::log(sum)+m-original-delta;
 }
 static double square(double x){return x*x;}
 public:
 JointFieldTrainer(Universe& u,FabricTeacher& teacher):JointFieldTrainer(u,teacher,Weights{}){}
 JointFieldTrainer(Universe& u,FabricTeacher& teacher,Weights w):
  u_(u),teacher_(teacher),weights_(w){
  if(w.prediction<0||w.triangle<0||w.temporal<0||w.retention<0||
     w.learning_rate<=0||w.learning_rate>1)
   throw std::invalid_argument("invalid four-term objective");
 }
 Result teach(uint8_t a,uint8_t b,uint8_t c,uint8_t relation,
              uint64_t effective_time){
  if(a==b||b==c||a==c||a==255||b==255||c==255||relation>=16)
   throw std::invalid_argument("invalid supervised compositional triple");
  Result result;
  const Evidence first=u_.query(a,b,relation,effective_time);
  const Evidence second=u_.query(b,c,relation,effective_time);
  if(first.state()!=1||second.state()!=1)return result;
  result.eligible=true;
  const Evidence inferred=u_.query(a,c,relation,effective_time);
  const Quat qab=u_.quaternion_field().get(a,b);
  const Quat qbc=u_.quaternion_field().get(b,c);
  const Quat old=u_.quaternion_field().get(a,c);
  const Quat composed=qab*qbc;
  // A refuted direct proposition requires learned curvature / exception.
  // A supported proposition pulls toward composition. "Both" remains two
  // separate constraints; neither gets overwritten in the evidence store.
  const double s=inferred.support/(1.0+inferred.support);
  const double r=inferred.refute/(1.0+inferred.refute);
  const auto prediction=teacher_.predict(a);
  const auto src=read(u_.root().h[a],32);
  const auto dst=read(u_.root().h[c],48);
  const auto old_rot=old.rotate({dst[0],dst[1],dst[2]});
  auto evaluate=[&](Quat q)->Terms{
   const auto moved=q.rotate({dst[0],dst[1],dst[2]});
   const double delta=QUAT_GAIN*(src[0]*(moved.x-old_rot.x)+
                                 src[1]*(moved.y-old_rot.y)+
                                 src[2]*(moved.z-old_rot.z));
   Terms terms;
   terms.prediction=nll_one_logit(prediction,c,delta);
   const double curvature=q.distance(composed);
   terms.triangle=square(curvature);
   terms.temporal=s*square(curvature)+r*square(std::max(0.0,0.8-curvature));
   terms.retention=square(old.distance(q));
   terms.total=weights_.prediction*terms.prediction+
               weights_.triangle*terms.triangle+
               weights_.temporal*terms.temporal+
               weights_.retention*terms.retention;
   return terms;
  };
  result.before=evaluate(old);result.after=result.before;
  Quat target=composed;
  if(r>s&&r>0.0001){
   // Counterfactual refutation target: a non-identity rotation,
   // never simply minus-q (which represents the same SO3 element).
   target=composed*Quat::axis_angle({1,0.4,0.3},1.2);
  }
  for(unsigned attempt=0;attempt<8;++attempt){
   const Quat candidate=Quat::slerp(old,target,std::ldexp(weights_.learning_rate,-int(attempt)));
   const auto terms=evaluate(candidate);
   if(terms.total+1e-10<result.before.total){
    u_.quaternion_field().set(a,c,candidate);
    result.accepted=true;
    result.after=terms;
    return result;
   }
  }
  return result;
 }
};
} // namespace nanlm
