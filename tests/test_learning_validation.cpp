#include "nanlm/teacher.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace nanlm;
namespace {
constexpr unsigned FIRST=16;
constexpr unsigned LAST=176;
constexpr unsigned EPOCHS=8;
constexpr double ALPHA=1.0;

uint8_t target(unsigned source) {
 return static_cast<uint8_t>((source+19u)%FabricTeacher::VOCAB);
}
bool unseen_source(unsigned source) {
 return ((source-FIRST)%4u)==0;
}
uint32_t next_random(uint32_t& state) {
 state=1664525u*state+1013904223u;
 return state;
}
void cleanup(const std::string& path) {
 for (const std::string suffix: {"", ".so3", ".overlay", ".overlay.tmp"})
  std::filesystem::remove(path+suffix);
}
struct Metrics {
 double ce=0;
 unsigned correct=0, n=0;
 void add(const FabricTeacher::Prediction& p,uint8_t label) {
  const double probability=p.probability[label];
  if (!std::isfinite(probability)||probability<=0)
   throw std::runtime_error("invalid model probability");
  ce-=std::log(probability);
  correct+=p.choice==label?1u:0u;
  ++n;
 }
 double mean_ce()const{return ce/n;}
 double accuracy()const{return double(correct)/n;}
};
Metrics evaluate(FabricTeacher& model,bool unseen) {
 Metrics m;
 for (unsigned source=FIRST;source<LAST;++source) {
  if (unseen_source(source)!=unseen)continue;
  m.add(model.predict(static_cast<uint8_t>(source)),target(source));
 }
 if (m.n!=(unseen?40u:120u))throw std::runtime_error("split violation");
 return m;
}
struct CountBaseline {
 std::array<std::array<uint16_t,FabricTeacher::VOCAB>,FabricTeacher::VOCAB> count{};
 std::array<uint32_t,FabricTeacher::VOCAB> totals{};
 void add(uint8_t source,uint8_t label) {
  ++count[source][label];++totals[source];
 }
 double loss(uint8_t source,uint8_t label)const {
  return std::log(double(totals[source])+ALPHA*FabricTeacher::VOCAB)-
         std::log(double(count[source][label])+ALPHA);
 }
 double mean_ce(bool unseen)const{
  double ce=0;unsigned n=0;
  for(unsigned source=FIRST;source<LAST;++source) {
   if(unseen_source(source)!=unseen)continue;
   ce+=loss(static_cast<uint8_t>(source),target(source));++n;
  }
  return ce/n;
 }
};
// A closed-form one-parameter circular-offset estimator is an essential
// conventional control: it has the SAME circular inductive bias as rule_frozen.
// This baseline is diagnostics only; it is never a second production universe.
struct OffsetBaseline {
 double sum_cos=0,sum_sin=0;
 void add(uint8_t src,uint8_t label) {
  const double displacement=TAU*(int(label)-int(src))/FabricTeacher::VOCAB;
  sum_cos+=std::cos(displacement);
  sum_sin+=std::sin(displacement);
 }
 double angle()const{return std::atan2(sum_sin,sum_cos);}
 Metrics evaluate(bool unseen)const{
  Metrics result;
  constexpr double gain=12.0;
  const double phi=angle();
  for(unsigned src=FIRST;src<LAST;++src){
   if(unseen_source(src)!=unseen)continue;
   const double at=TAU*(double(src)+0.5)/FabricTeacher::VOCAB+phi;
   std::array<double,FabricTeacher::VOCAB> logits{};
   double max_logit=-std::numeric_limits<double>::infinity();
   unsigned best=0;
   for(unsigned j=0;j<FabricTeacher::VOCAB;++j) {
    const double theta=TAU*(double(j)+0.5)/FabricTeacher::VOCAB;
    logits[j]=gain*std::cos(at-theta);
    if(logits[j]>max_logit){max_logit=logits[j];best=j;}
   }
   double denom=0;
   for(double logit:logits)denom+=std::exp(logit-max_logit);
   const auto label=target(src);
   result.ce+=std::log(denom)+max_logit-logits[label];
   result.correct+=best==label?1u:0u;
   ++result.n;
  }
  return result;
 }
};
struct Configuration {
 const char* name;
 double angle,quaternion,dot_gain;
 bool shuffled=false;
 double rule_gain=0;
 bool train_heads=true;
 double feature_rotor_gain=0.12;
};
struct Outcome {
 Metrics before_seen{},before_unseen{},after_seen{},after_unseen{};
 unsigned accepted=0,attempted=0,regressed=0;
 double max_scratch_gap=0;
 double rule_angle_rad=0;
};
Outcome experiment(uint32_t seed,Configuration config,CountBaseline* counts,OffsetBaseline* offset) {
 const std::string path=(std::filesystem::temp_directory_path()/
   ("nanlm-validation-"+std::to_string(::getpid())+"-"+
    std::to_string(seed)+"-"+config.name)).string();
 cleanup(path);
 Outcome o;
 {
  Universe universe(path);
  FabricTeacher model(universe,0.055,config.angle,0.09,config.quaternion,config.dot_gain,config.rule_gain,config.train_heads,config.feature_rotor_gain);
  o.before_seen=evaluate(model,false);
  o.before_unseen=evaluate(model,true);
  std::vector<uint8_t> sources;
  for(unsigned source=FIRST;source<LAST;++source)
   if(!unseen_source(source))sources.push_back(static_cast<uint8_t>(source));
  if(sources.size()!=120)throw std::runtime_error("training split violation");
  uint32_t rng=seed;
  for(unsigned epoch=0;epoch<EPOCHS;++epoch) {
   // Fisher-Yates shuffles only training sources. Evaluation sources never enter training.
   for(size_t i=sources.size();i>1;--i)
    std::swap(sources[i-1],sources[next_random(rng)%i]);
   for(uint8_t src:sources) {
    uint8_t label=target(src);
    if(config.shuffled)label=static_cast<uint8_t>((unsigned(label)+79u)%FabricTeacher::VOCAB);
    const double before=model.loss(src,label);
    const auto update=model.teach(src,label,++o.attempted);
    if(counts)counts->add(src,label);
    if(offset)offset->add(src,label);
    if(update.accepted) {
     ++o.accepted;
     const double actual=model.loss(src,label);
     if(actual>before+1e-7)++o.regressed;
     o.max_scratch_gap=std::max(o.max_scratch_gap,std::fabs(actual-update.after));
    }
   }
  }
  if(config.rule_gain>0) {
   Quat q;
   std::memcpy(&q,universe.root().h[PRECEDES].data+96,sizeof(q));
   Quat inv;
   std::memcpy(&inv,universe.root().h[FOLLOWS].data+96,sizeof(inv));
   if((q*inv).angle()>1e-5)
    throw std::runtime_error("shared forward/reverse relation quaternions inconsistent");
   const Vec3q v=q.rotate({1,0,0});
   o.rule_angle_rad=std::atan2(v.y,v.x);
  }
  o.after_seen=evaluate(model,false);
  o.after_unseen=evaluate(model,true);
  if(universe.root().h[255].data[3]!=0)
   throw std::runtime_error("gateway head was used as language state");
  universe.sync();
 }
 {
  Universe restored(path);
  FabricTeacher reloaded(restored,0.055,config.angle,0.09,config.quaternion,config.dot_gain,config.rule_gain,config.train_heads,config.feature_rotor_gain);
  if(std::fabs(evaluate(reloaded,false).mean_ce()-o.after_seen.mean_ce())>1e-7 ||
     std::fabs(evaluate(reloaded,true).mean_ce()-o.after_unseen.mean_ce())>1e-7)
   throw std::runtime_error("restart changed held-out predictions");
  if(config.rule_gain>0) {
   Quat q,inv;
   std::memcpy(&q,restored.root().h[PRECEDES].data+96,sizeof(q));
   std::memcpy(&inv,restored.root().h[FOLLOWS].data+96,sizeof(inv));
   if((q*inv).angle()>1e-5)
    throw std::runtime_error("restored inverse relation rotor inconsistent");
  }
 }
 cleanup(path);
 return o;
}
void report(uint32_t seed,Configuration config,const Outcome& o) {
 std::cout<<seed<<','<<config.name<<','
  <<o.before_seen.mean_ce()<<','<<o.after_seen.mean_ce()<<','
  <<o.before_unseen.mean_ce()<<','<<o.after_unseen.mean_ce()<<','
  <<o.after_seen.accuracy()<<','<<o.after_unseen.accuracy()<<','
  <<o.accepted<<','<<o.attempted<<','<<o.regressed<<','<<o.max_scratch_gap
  <<','<<o.rule_angle_rad<<"\n";
}
} // namespace
int main() {
 std::cout<<std::fixed<<std::setprecision(8);
 std::cout<<"seed,model,seen_CE_before,seen_CE_after,unseen_source_CE_before,"
           "unseen_source_CE_after,seen_accuracy,unseen_source_accuracy,"
           "accepted,attempted,post_commit_regressions,max_actual_vs_scratch_gap,rule_angle_rad\n";
 constexpr std::array<uint32_t,3> seeds{11,29,47};
 constexpr std::array<Configuration,7> arms{{
  {"flat",0.0,0.0,4.45,false,0.0},
  {"u1",0.08,0.0,4.45,false,0.0},
  {"so3",0.0,0.45,4.0,false,0.0},
  {"hybrid",0.08,0.45,4.0,false,0.0},
  {"rule",0.0,0.0,4.45,false,12.0},
  {"hybrid_rule",0.08,0.45,4.0,false,12.0},
  {"rule_frozen",0.0,0.0,0.0,false,12.0,false,0.0}
 }};
 for(uint32_t seed:seeds) {
  CountBaseline counts;
  OffsetBaseline offset;
  Outcome frozen{};
  bool saw_frozen=false;
  for(const Configuration config:arms) {
   // Count-only learner is updated once, with the same ordered training observations.
   CountBaseline* baseline=std::string(config.name)=="flat"?&counts:nullptr;
   OffsetBaseline* circular=baseline?&offset:nullptr;
   const Outcome outcome=experiment(seed,config,baseline,circular);
   report(seed,config,outcome);
   if(std::string(config.name)=="rule_frozen") {
    saw_frozen=true;frozen=outcome;
    const double expected_angle=TAU*19.0/FabricTeacher::VOCAB;
    if(outcome.after_unseen.accuracy()<0.999 ||
       outcome.after_unseen.mean_ce()>=outcome.before_unseen.mean_ce()-1.0 ||
       std::fabs(outcome.rule_angle_rad-expected_angle)>0.015)
     throw std::runtime_error("shared relation failed synthetic zero-shot transfer");
   }
  }
  if(!saw_frozen)throw std::runtime_error("missing isolated relation control");
  std::cout<<seed<<",count,,"<<counts.mean_ce(false)
    <<",,"<<counts.mean_ce(true)<<",,,,,,,\n";
  const Metrics circular_seen=offset.evaluate(false);
  const Metrics circular_unseen=offset.evaluate(true);
  if(std::fabs(circular_unseen.mean_ce()-frozen.after_unseen.mean_ce())>0.02 ||
     circular_unseen.accuracy()<0.999)
   throw std::runtime_error("shared field diverged from matched circular-offset control");
  std::cout<<seed<<",cyclic_offset,,"<<circular_seen.mean_ce()<<",,"
    <<circular_unseen.mean_ce()<<","<<circular_seen.accuracy()<<","
    <<circular_unseen.accuracy()<<",,,,,"<<offset.angle()<<"\n";
  if(seed==seeds[0]) {
   constexpr Configuration negative{"shuffled_labels",0.08,0.45,4.0,true,12.0};
   const Outcome corrupted=experiment(seed,negative,nullptr,nullptr);
   report(seed,negative,corrupted);
   if(corrupted.after_unseen.mean_ce()<=corrupted.before_unseen.mean_ce())
    throw std::runtime_error("corrupted-label control must not learn true rule");
  }
 }
 std::cout<<"PASS: deterministic source-disjoint held-out audit, matched training budgets, "
            "count and matched circular-offset baselines, scalar, quaternion, exact zero-shot rule transfer, inverse consistency, corrupted-label failure, "
            "committed-loss audit and persistence.\n";
}
