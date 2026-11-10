#include "nanlm/contextual_roles.hpp"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>
using namespace nanlm;
static void require(bool yes,const char* why) {
 if(!yes)throw std::runtime_error(why);
}
static std::vector<ContextualRoleLearner::Example> corpus(
    const std::vector<std::string>& names) {
 using E=ContextualRoleLearner::Example;
 std::vector<E> result;
 for(size_t i=1;i<names.size();i+=2) {
  const auto& a=names[i-1];
  const auto& b=names[i];
  const std::vector<std::pair<std::string,std::string>> samples={
   {a+" sent parcel to "+b,"transfer"},
   {a+" gave parcel to "+b,"transfer"},
   {a+" delivered parcel to "+b,"transfer"},
   {b+" received parcel from "+a,"transfer"},
   {b+" accepted parcel from "+a,"transfer"},
   {b+" got parcel from "+a,"transfer"},
   {a+" followed "+b,"follow"},
   {a+" walked behind "+b,"follow"},
   {a+" trailed "+b,"follow"},
   {b+" led "+a,"follow"},
   {b+" walked ahead of "+a,"follow"},
   {b+" was followed by "+a,"follow"}
  };
  for(const auto& [sentence,relation]:samples) {
   result.push_back(E{sentence,a,b,relation,1});
   result.push_back(E{sentence,b,a,relation,0});
  }
 }
 return result;
}
int main() {
 const std::string root=(std::filesystem::temp_directory_path()/
                          ("nanlm-role-"+std::to_string(::getpid()))).string();
 auto cleanup=[&](){
  for(const char* suffix:{"",".overlay",".so3",".context-role",".context-role.tmp"})
   std::filesystem::remove(root+suffix);
 };
 cleanup();
 try{
  Universe universe(root);
  require(universe.event_count()==0,"fresh root");
  const auto train=corpus({"ada","bea","cory","dina","elan","faye",
    "gail","hugo","iris","juno","kira","lora","mira","nora","opal",
    "pavi","quinn","ravi","sara","tova","uma","vera","wade","xena"});
  const auto test=corpus({"yara","zena","arun","brin","cato","daro",
    "esme","finn","gora","hexi","ivon","jace"});
  const auto dev=corpus({"kato","lena","mavi","navi","orna","pato",
    "reno","sumi","tali","vito","wila","yori"});
  require(train.size()==288&&dev.size()==144&&test.size()==144,
          "expected disjoint train/dev/test fixtures");
  ContextualRoleLearner learned(universe);
  const auto initial=learned.predict(test.front());
  require(!initial.has_value(),"no pretraining predicate supported");
  require(!learned.predict({"ana gave ball to ben","ana","ben","unknown",0}),
          "unknown predicates must abstain");
  learned.train(train,90,0.065);
  auto dev_result=learned.evaluate(dev);
  auto test_result=learned.evaluate(test);
  require(dev_result.accuracy()>=0.75,"development-role generalization below threshold");
  require(test_result.accuracy()>=0.75,"heldout-entity role generalization below threshold");
  require(test_result.ce()<0.55,"heldout semantic CE failed");
  require(learned.updates()==train.size()*90u,"gradient step accounting");
  require(learned.tokens()>0&&learned.relations()==2,"actual lexical/relation parameters");
  for(size_t i=0;i<test.size();i+=2){
   const auto yes=learned.predict(test[i]);
   const auto no=learned.predict(test[i+1]);
   require(yes.has_value()&&no.has_value(),"role-inversion prediction absent");
   require(*yes>*no,"subject-object inversion must reverse prediction");
  }
  std::vector<ContextualRoleLearner::Example> shuffled=train;
  std::mt19937_64 rng(20261109);
  std::vector<int> labels;
  for(const auto& x:shuffled)labels.push_back(x.label);
  std::shuffle(labels.begin(),labels.end(),rng);
  for(size_t i=0;i<shuffled.size();++i)shuffled[i].label=labels[i];
  const std::string control_root=root+"-random-labels";
  Universe control_universe(control_root);
  ContextualRoleLearner control(control_universe);
  control.train(shuffled,90,0.065);
  const auto control_result=control.evaluate(test);
  require(test_result.accuracy()>control_result.accuracy()+0.15,
          "semantic learner fails scrambled-supervision control");
  const double p_before=*learned.predict(test.front());
  learned.save();
  require(universe.event_count()==0,"trainable overlay cannot alter exact event journal");
  {
   ContextualRoleLearner restored(universe);
   const auto p_after=restored.predict(test.front());
   require(p_after.has_value()&&std::abs(*p_after-p_before)<1e-12,
           "persistent neural weights changed on restart");
   require(restored.updates()==learned.updates(),"training steps persisted");
   require(restored.evaluate(test).correct==test_result.correct,
           "restart changed heldout semantic labels");
  }
  {
   std::ofstream corrupt(root+".context-role",std::ios::app);
   corrupt<<"junk";
   corrupt.close();
   bool rejected=false;
   try{ContextualRoleLearner broken(universe);(void)broken;}
   catch(const std::runtime_error&){rejected=true;}
   require(rejected,"corrupt model checkpoint accepted");
  }
  std::cout<<"CONTEXT_ROLE_NATIVE PASS train="<<train.size()
           <<" dev="<<dev.size()<<" test="<<test.size()
           <<" dev_ce="<<dev_result.ce()
           <<" dev_accuracy="<<dev_result.accuracy()
           <<" test_ce="<<test_result.ce()
           <<" test_accuracy="<<test_result.accuracy()
           <<" shuffled_accuracy="<<control_result.accuracy()
           <<" updates="<<learned.updates()
           <<" vocab="<<learned.tokens()
           <<" relations="<<learned.relations()
           <<" world_events="<<universe.event_count()<<"\n";
  for(const char* suffix:{"",".overlay",".so3",".context-role",".context-role.tmp"})
   std::filesystem::remove(control_root+suffix);
  cleanup();
 }catch(...){cleanup();throw;}
}
