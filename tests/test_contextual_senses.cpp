#include "nanlm/contextual_senses.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <unistd.h>
using namespace nanlm;
static void check(bool good,const char* msg){
 if(!good)throw std::runtime_error(msg);
}
int main(){
 const auto root=(std::filesystem::temp_directory_path()/
                 ("nanlm-wsd-"+std::to_string(::getpid()))).string();
 auto clean=[&](const std::string& name){
  for(const char* suffix:{"",".overlay",".so3",".wordnet.tsv",
                          ".context-role",".context-senses",".context-senses.tmp"})
   std::filesystem::remove(name+suffix);
 };
 clean(root);
 clean(root+"-shuffle");
 try{
  {
   std::ofstream f(root+".wordnet.tsv");
   f<<"NANLM-WN1\t2\t0\n";
   f<<"N\t801\t101\tbank\tfinancial institution taking deposits\n";
   f<<"N\t802\t202\tbank\tsloping land next to a river\n";
   f.flush();
   check(bool(f),"synthetic exact-sense fixture");
  }
  WordNetIndex wn;wn.load(root+".wordnet.tsv");
  check(wn.lookup("bank").size()==2,"exact ambiguous surface form");
  Universe u(root);
  using E=ContextualSenseLearner::Example;
  const std::vector<E> prototypes={
   {"the bank approved a loan","bank",801},
   {"cash was deposited at the bank","bank",801},
   {"the bank handled money and credit","bank",801},
   {"a teller served customers inside the bank","bank",801},
   {"she borrowed money from the bank","bank",801},
   {"the bank issued credit to clients","bank",801},
   {"the water rushed past the bank","bank",802},
   {"a boat floated beside the bank","bank",802},
   {"the bank was wet with river mud","bank",802},
   {"the stream flowed along the bank","bank",802},
   {"the bank collapsed during the flood","bank",802},
   {"the bank had reeds beside the river","bank",802}
  };
  const std::vector<E> test={
   {"the customer requested a loan at the bank","bank",801},
   {"the bank counted cash and money","bank",801},
   {"the teller went back to the bank","bank",801},
   {"credit was provided by the bank","bank",801},
   {"the boat moved by the bank in water","bank",802},
   {"the bank was covered with mud","bank",802},
   {"a river stream flowed near the bank","bank",802},
   {"the flood carried reeds across the bank","bank",802}
  };
  std::vector<E> train;
  for(size_t repetition=0;repetition<10;++repetition)
   for(const auto& e:prototypes)train.push_back(e);
  ContextualSenseLearner learner(u);
  check(!learner.infer("the river reached the bank","bank",wn),
        "untrained synsets must abstain");
  learner.train_senses(train,wn,120,0.07);
  const auto scores=learner.evaluate_senses(test,wn);
  check(scores.accuracy()>=0.875,"word-sense contextual test failed");
  check(scores.ce()<0.45,"held-out sense cross-entropy not learned");
  check(learner.trained_senses()==2,"exact synsets not represented");
  const auto result=learner.infer("the boat moved by the bank in water","bank",wn);
  check(result&&result->synset==802,"river sense not inferred");
  const auto another=learner.infer("credit was provided by the bank","bank",wn);
  check(another&&another->synset==801,"financial sense not inferred");
  std::mt19937 rng(20261109);
  std::vector<E> shuffled=train;
  std::vector<uint32_t> labels;
  for(const auto& r:shuffled)labels.push_back(r.gold);
  std::shuffle(labels.begin(),labels.end(),rng);
  for(size_t i=0;i<shuffled.size();++i)shuffled[i].gold=labels[i];
  Universe control_root(root+"-shuffle");
  ContextualSenseLearner control(control_root);
  control.train_senses(shuffled,wn,120,0.07);
  const auto randomized=control.evaluate_senses(test,wn);
  check(scores.accuracy()>randomized.accuracy()+0.2,
        "does not beat shuffled sense labels");
  const size_t events=u.event_count();
  learner.save_senses();
  check(u.event_count()==events,"sense gradient must not assert WordNet facts");
  {
   ContextualSenseLearner restored(u);
   const auto replay=restored.evaluate_senses(test,wn);
   check(replay.correct==scores.correct&&
         std::abs(replay.nll-scores.nll)<1e-10,
         "contextual/sense shared checkpoint changed on reload");
   check(restored.sense_updates()==learner.sense_updates(),
         "sense gradient count lost");
   ContextualRoleLearner shared(u);
   check(shared.tokens()==learner.tokens(),
         "shared contextual embeddings not visible through role learner");
  }
  {
   std::ofstream corrupt(root+".context-senses",std::ios::app);
   corrupt<<"bad";
   corrupt.close();
   bool blocked=false;
   try{ContextualSenseLearner invalid(u);(void)invalid;}
   catch(const std::runtime_error&){blocked=true;}
   check(blocked,"corrupted exact-synset head accepted");
  }
  std::cout<<"CONTEXT_WSD_NATIVE PASS train="<<train.size()
           <<" test="<<test.size()
           <<" test_ce="<<scores.ce()
           <<" test_accuracy="<<scores.accuracy()
           <<" shuffled_accuracy="<<randomized.accuracy()
           <<" synsets="<<learner.trained_senses()
           <<" updates="<<learner.sense_updates()
           <<" shared_tokens="<<learner.tokens()
           <<" root_events="<<events<<"\n";
  clean(root+"-shuffle");clean(root);
 }catch(...){clean(root+"-shuffle");clean(root);throw;}
}
