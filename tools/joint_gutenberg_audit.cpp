// Native cross-author audit of semantic-to-wave transfer.
// Byte foundation/readout see only Alice training. Alice development gates
// optional recurrent updates. Dickens is held out until final evaluation.
// Synthetic semantic labels are a supervised probe, NOT Gutenberg acquisition.
#include "nanlm/joint_wave_semantics.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

using namespace nanlm;
using J=JointWaveSemanticLearner;
static void check(bool v,const char* why){if(!v)throw std::runtime_error(why);}
static std::string read(const std::string& path,size_t low,size_t high) {
 std::ifstream f(path,std::ios::binary);
 if(!f)throw std::runtime_error("cannot open corpus "+path);
 const std::string s((std::istreambuf_iterator<char>(f)),{});
 if(s.size()<low||s.size()>high||s.find(char(255))!=std::string::npos)
  throw std::runtime_error("invalid corpus size or reserved byte "+path);
 return s;
}
static std::string fixture(const std::string& root){
 const std::string name=root+".wordnet.tsv";
 std::ofstream f(name,std::ios::binary|std::ios::trunc);
 f<<"NANLM-WN1\t2\t0\n"
  <<"N\t801\t101\tbank\tfinancial institution taking deposits\n"
  <<"N\t802\t202\tbank\triver shore\n";
 f.flush();
 if(!f)throw std::runtime_error("failed synthetic WordNet file");
 return name;
}
static void cleanup(const std::string& name){
 for(const char* suffix:{"",".overlay",".overlay.tmp",".so3",".episodes",
   ".wordnet.tsv",".joint-wave",".joint-wave.tmp",
   "-byte.joint-wave","-shuffle.joint-wave"})
  std::filesystem::remove(name+suffix);
}
struct Triple { J::Metrics byte,joint,shuffle; };
static Triple compare(J& control,J& joint,J& shuffled,
                      const std::string& corpus){
 Triple result{control.evaluate_bytes(corpus),joint.evaluate_bytes(corpus),
               shuffled.evaluate_bytes(corpus)};
 check(result.byte.count==result.joint.count&&
       result.joint.count==result.shuffle.count&&
       result.byte.count==corpus.size()-1,
       "model heldout byte counts not matched");
 for(double v:{result.byte.ce(),result.joint.ce(),result.shuffle.ce()})
  check(std::isfinite(v)&&v>0&&v<100,"invalid Gutenberg CE");
 return result;
}
static std::vector<J::Role> roles(){
 return {
  {"ada gave parcel to bea","ada","bea","transfer",1},
  {"ada gave parcel to bea","bea","ada","transfer",0},
  {"bea received parcel from ada","ada","bea","transfer",1},
  {"bea received parcel from ada","bea","ada","transfer",0},
  {"ada followed bea","ada","bea","follow",1},
  {"ada followed bea","bea","ada","follow",0},
  {"bea was followed by ada","ada","bea","follow",1},
  {"bea was followed by ada","bea","ada","follow",0}
 };
}
static std::vector<J::Sense> senses(){
 return {
  {"the bank approved a loan","bank",801},
  {"cash was deposited at the bank","bank",801},
  {"the bank issued credit","bank",801},
  {"the water rushed past the bank","bank",802},
  {"the boat floated near the bank","bank",802},
  {"the bank was wet with river mud","bank",802}
 };
}
static std::vector<J::Role> test_roles(){
 return {
  {"cato gave parcel to dara","cato","dara","transfer",1},
  {"cato gave parcel to dara","dara","cato","transfer",0},
  {"dara received parcel from cato","cato","dara","transfer",1},
  {"dara received parcel from cato","dara","cato","transfer",0}
 };
}
static std::vector<J::Sense> test_senses(){
 return {
  {"the bank offered credit","bank",801},
  {"a loan came from the bank","bank",801},
  {"the river passed the bank","bank",802},
  {"mud covered the bank","bank",802}
 };
}
int main(int argc,char** argv) {
 if(argc!=6){
  std::cerr<<"usage: joint_gutenberg_audit TRAIN_ALICE DEV_ALICE "
              "TEST_CAROL_EARLY TEST_CAROL_LATE SEEDS\n";
  return 2;
 }
 try{
  const std::string train=read(argv[1],1500,6000);
  const std::string dev=read(argv[2],300,2000);
  const std::string early=read(argv[3],400,3000);
  const std::string late=read(argv[4],400,3000);
  const size_t n=std::stoull(argv[5]);
  check(n>=2&&n<=8,"seed count must be 2..8");
  for(size_t k=0;k<n;++k){
   const uint64_t seed=20261109ull+103ull*k;
   const std::string root=(std::filesystem::temp_directory_path()/
    ("nanlm-gutenberg-joint-"+std::to_string(::getpid())+"-"+std::to_string(k))).string();
   cleanup(root);
   try {
    {
     fixture(root);
     WordNetIndex wn;wn.load(root+".wordnet.tsv");
     check(wn.lookup("bank").size()==2,"exact synthetic WordNet senses");
     Universe universe(root);
     UnifiedWaveLanguage wave(universe);
     // All variants see identical fixed Alice-only foundation for this
     // seed. No Carol or Alice DEV bytes are used for gradient updates.
     std::istringstream foundation(train.substr(0,1300));
     const auto trained=wave.train(foundation);
     check(trained.tokens>1000&&std::isfinite(trained.ce()),
           "Alice physical wave foundation failed");
     const auto events=universe.event_count();
     J control(wave,root+"-byte");
     J joint(wave,root);
     J shuffled(wave,root+"-shuffle");
     std::mt19937_64 rng(seed);
     // Randomize *whole causal chunks* of training text; do not reverse
     // or shuffle bytes inside sentences, and keep seed order matched.
     std::vector<size_t> order;
     for(size_t i=0;i<train.size();i+=192)order.push_back(i);
     std::shuffle(order.begin(),order.end(),rng);
     for(size_t start:order) {
      J::State state{};
      const size_t stop=std::min(start+192,train.size());
      for(size_t i=start;i<stop;++i){
       const auto ch=static_cast<uint8_t>(
          static_cast<unsigned char>(train[i]));
       if(state.length){
        control.teach_byte(state,ch,0.02);
        joint.teach_byte(state,ch,0.02);
        shuffled.teach_byte(state,ch,0.02);
       }
       wave.observe(state,ch);
      }
     }
     const auto role_data=roles();
     const auto sense_data=senses();
     auto randomized_roles=role_data;
     auto randomized_senses=sense_data;
     std::vector<int> labels;
     for(const auto& row:role_data)labels.push_back(row.label);
     std::shuffle(labels.begin(),labels.end(),rng);
     for(size_t i=0;i<labels.size();++i)
      randomized_roles[i].label=labels[i];
     std::vector<uint32_t> synset_labels;
     for(const auto& row:sense_data)synset_labels.push_back(row.gold);
     std::shuffle(synset_labels.begin(),synset_labels.end(),rng);
     for(size_t i=0;i<synset_labels.size();++i)
      randomized_senses[i].gold=synset_labels[i];
     std::vector<size_t> ro(role_data.size()),so(sense_data.size());
     std::iota(ro.begin(),ro.end(),0u);
     std::iota(so.begin(),so.end(),0u);
     for(size_t epoch=0;epoch<18;++epoch){
      std::shuffle(ro.begin(),ro.end(),rng);
      std::shuffle(so.begin(),so.end(),rng);
      for(size_t i:ro){
       const auto state=joint.state_for(role_data[i].sentence);
       joint.teach_role(state,role_data[i],0.035);
       shuffled.teach_role(state,randomized_roles[i],0.035);
      }
      for(size_t i:so){
       const auto state=joint.state_for(sense_data[i].sentence);
       joint.teach_sense(state,sense_data[i],wn,0.035);
       shuffled.teach_sense(state,randomized_senses[i],wn,0.035);
      }
     }
     // Check this is a matched comparison of a *single* fixed language
     // wave; the semantic heads are three separate in-memory readouts.
     check(universe.event_count()==events,"semantic head modified symbolic journal");
     const auto control_dev=compare(control,joint,shuffled,dev);
     const auto before_early=compare(control,joint,shuffled,early);
     const auto before_late=compare(control,joint,shuffled,late);
     const auto t_roles=test_roles();
     const auto t_senses=test_senses();
     const auto role_before=joint.evaluate_roles(t_roles);
     const auto sense_before=joint.evaluate_senses(t_senses,wn);
     const auto role_shuf=shuffled.evaluate_roles(t_roles);
     const auto sense_shuf=shuffled.evaluate_senses(t_senses,wn);
     // Optional physical recurrent update: *only Alice dev* is consulted
     // by the rollback gate; Dickens scores above were generated for
     // reporting, not consulted for any acceptance decision.
     const bool accepted=joint.guarded_sense_recurrent_step(
       universe,sense_data.front(),wn,dev,0.0);
     const auto after_dev=joint.evaluate_bytes(dev);
     const auto after_early=joint.evaluate_bytes(early);
     const auto after_late=joint.evaluate_bytes(late);
     const auto role_after=joint.evaluate_roles(t_roles);
     const auto sense_after=joint.evaluate_senses(t_senses,wn);
     check(universe.event_count()==events,
           "recurrent semantic update contaminated exact claims");
     if(accepted)
      check(after_dev.ce()<=control_dev.joint.ce()+1e-10,
            "accepted optimizer exceeded zero-regression budget");
     else
      check(std::abs(after_dev.ce()-control_dev.joint.ce())<1e-12,
            "rejected optimizer failed rollback");
     // Same author's heldout development strictly distinguished from
     // cross-author test and from all supervision examples.
     std::cout<<std::setprecision(10)<<"JOINT_GUTENBERG_SEED"
       <<" seed="<<seed
       <<" train_bytes="<<train.size()
       <<" dev_tokens="<<control_dev.byte.count
       <<" early_tokens="<<before_early.byte.count
       <<" late_tokens="<<before_late.byte.count
       <<" dev_byte_ce="<<control_dev.byte.ce()
       <<" dev_joint_ce="<<control_dev.joint.ce()
       <<" dev_shuffled_ce="<<control_dev.shuffle.ce()
       <<" dev_post_recurrence_ce="<<after_dev.ce()
       <<" carol_early_byte_ce="<<before_early.byte.ce()
       <<" carol_early_joint_ce="<<before_early.joint.ce()
       <<" carol_early_shuffled_ce="<<before_early.shuffle.ce()
       <<" carol_early_post_recurrence_ce="<<after_early.ce()
       <<" carol_late_byte_ce="<<before_late.byte.ce()
       <<" carol_late_joint_ce="<<before_late.joint.ce()
       <<" carol_late_shuffled_ce="<<before_late.shuffle.ce()
       <<" carol_late_post_recurrence_ce="<<after_late.ce()
       <<" semantic_role_accuracy="<<role_before.accuracy()
       <<" semantic_role_shuffle_accuracy="<<role_shuf.accuracy()
       <<" semantic_sense_accuracy="<<sense_before.accuracy()
       <<" semantic_sense_shuffle_accuracy="<<sense_shuf.accuracy()
       <<" semantic_role_test_ce_before="<<role_before.ce()
       <<" semantic_role_test_ce_after="<<role_after.ce()
       <<" semantic_sense_test_ce_before="<<sense_before.ce()
       <<" semantic_sense_test_ce_after="<<sense_after.ce()
       <<" recurrent_accepted="<<int(accepted)
       <<" root_bytes="<<ROOT_BYTES
       <<" journal_events="<<events
       <<"\n"<<std::flush;
     check(std::filesystem::file_size(root)==ROOT_BYTES,
           "physical universe mapping size changed");
    }
    cleanup(root);
   }catch(...){cleanup(root);throw;}
  }
  std::cout<<"JOINT_GUTENBERG_AUDIT PASS seeds="<<n
           <<" leak_free=1 strict_dev_gate=1 exact_root=1\n";
 }catch(const std::exception& e){
  std::cerr<<"JOINT_GUTENBERG_AUDIT FAIL "<<e.what()<<"\n";
  return 1;
 }
}
