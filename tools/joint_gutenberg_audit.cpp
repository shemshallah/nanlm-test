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
#include <memory>
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
   "-byte.joint-wave","-shuffle.joint-wave","-pareto.joint-wave",
   "-frozen.joint-wave","-surgery.joint-wave","-sham.joint-wave"})
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
     J pareto(wave,root+"-pareto");
     J frozen(wave,root+"-frozen");
     J surgery(wave,root+"-surgery");
     J sham(wave,root+"-sham");
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
        pareto.teach_byte(state,ch,0.02);
        frozen.teach_byte(state,ch,0.02);
        surgery.teach_byte(state,ch,0.02);
        sham.teach_byte(state,ch,0.02);
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
     const double pareto_base_dev=pareto.evaluate_bytes(dev).ce();
     double pareto_accepted_dev=pareto_base_dev;
     size_t pareto_batches_accepted=0; // nonzero shared-projection steps
     size_t pareto_batches_rejected=0; // shared credit denied; head-only kept
     size_t pareto_head_only_batches=0;
     double pareto_shared_alpha_sum=0.0;
     double surgery_accepted_dev=surgery.evaluate_bytes(dev).ce();
     size_t surgery_batches_accepted=0,surgery_batches_rejected=0;
     size_t surgery_head_only_batches=0,surgery_conflicting_batches=0;
     double surgery_shared_alpha_sum=0.0;
     auto surgery_candidate=std::make_unique<J>(surgery);
     double sham_accepted_dev=sham.evaluate_bytes(dev).ce();
     size_t sham_batches_accepted=0,sham_batches_rejected=0;
     size_t sham_head_only_batches=0,sham_conflicting_batches=0;
     double sham_shared_alpha_sum=0.0;
     auto sham_candidate=std::make_unique<J>(sham);
     auto pareto_candidate=std::make_unique<J>(pareto);
     for(size_t epoch=0;epoch<18;++epoch){
      std::shuffle(ro.begin(),ro.end(),rng);
      std::shuffle(so.begin(),so.end(),rng);
      for(size_t i:ro){
       const auto state=joint.state_for(role_data[i].sentence);
       joint.teach_role(state,role_data[i],0.035);
       shuffled.teach_role(state,randomized_roles[i],0.035);
       pareto_candidate->teach_role(state,role_data[i],0.035);
       frozen.teach_role(state,role_data[i],0.035,false);
      surgery_candidate->teach_role(state,role_data[i],0.035);
      sham_candidate->teach_role(state,randomized_roles[i],0.035);
      }
      for(size_t i:so){
       const auto state=joint.state_for(sense_data[i].sentence);
       joint.teach_sense(state,sense_data[i],wn,0.035);
       shuffled.teach_sense(state,randomized_senses[i],wn,0.035);
       pareto_candidate->teach_sense(state,sense_data[i],wn,0.035);
       frozen.teach_sense(state,sense_data[i],wn,0.035,false);
      surgery_candidate->teach_sense(state,sense_data[i],wn,0.035);
      sham_candidate->teach_sense(state,randomized_senses[i],wn,0.035);
      }
      if((epoch+1)%3==0){
       // Strict Pareto acceptance: if learning synthetic semantics would
       // raise the disjoint Alice byte-development CE, roll the entire
       // candidate readout back. Never consult either Dickens test slice.
       // Only a nonzero change to the shared projection counts as
       // semantic-to-language credit. Try a fixed line search without
       // consulting Dickens (which stays completely held out).
       bool shared_accepted=false;
       for(const double alpha : {1.0,0.25,0.0625,0.015625,0.00390625}){
        auto trial=std::make_unique<J>(pareto);
        trial->blend_semantics_from(*pareto_candidate,alpha);
        const double trial_ce=trial->evaluate_bytes(dev).ce();
        if(std::isfinite(trial_ce)&&trial_ce<=pareto_accepted_dev+1e-12){
         pareto.adopt_readout_from(*trial);
         pareto_accepted_dev=trial_ce;
         ++pareto_batches_accepted;
         pareto_shared_alpha_sum+=alpha;
         shared_accepted=true;
         break;
        }
       }
       if(!shared_accepted){
        // Preserve usable supervised semantic heads while leaving the
        // shared language projection and byte readout EXACTLY unchanged.
        // Report this as head-only specialization, NOT joint learning.
        auto head_only=std::make_unique<J>(pareto);
        head_only->blend_semantics_from(*pareto_candidate,0.0);
        const double same_ce=head_only->evaluate_bytes(dev).ce();
        check(std::isfinite(same_ce)&&
              std::abs(same_ce-pareto_accepted_dev)<=1e-12,
              "head-only fallback modified language cross-entropy");
        pareto.adopt_readout_from(*head_only);
        ++pareto_head_only_batches;
        ++pareto_batches_rejected;
       }
       pareto_candidate=std::make_unique<J>(pareto);
      // Sixth arm: semantic displacement projected using Alice TRAIN byte
      // gradients only; independent Alice DEV CE remains the nonlinear
      // acceptance gate, and Dickens remains a never-consulted test.
      // Never add byte-gradient descent, or consult Dickens at this gate.
      // Calibrate projection on Alice TRAIN, not on the independent
      // Alice development gate. The disjoint DEV text selects alpha only.
      const double alignment=surgery.shared_byte_directional_derivative(
          *surgery_candidate,train);
      check(std::isfinite(alignment),"nonfinite semantic-byte alignment");
      if(alignment>0)++surgery_conflicting_batches;
      bool surgery_shared=false;
      for(const double alpha : {1.0,0.25,0.0625,0.015625,0.00390625}){
       auto trial=std::make_unique<J>(surgery);
       trial->blend_projected_semantics_from(*surgery_candidate,train,alpha);
       if(trial->shared_displacement_norm(surgery)<=1e-12)continue;
       const double trial_ce=trial->evaluate_bytes(dev).ce();
       if(std::isfinite(trial_ce)&&trial_ce<=surgery_accepted_dev+1e-12){
        surgery.adopt_readout_from(*trial);
        surgery_accepted_dev=trial_ce;
        ++surgery_batches_accepted;
        surgery_shared_alpha_sum+=alpha;
        surgery_shared=true;
        break;
       }
      }
      if(!surgery_shared){
       auto head_only=std::make_unique<J>(surgery);
       head_only->blend_projected_semantics_from(*surgery_candidate,train,0.0);
       const double same_ce=head_only->evaluate_bytes(dev).ce();
       check(std::abs(same_ce-surgery_accepted_dev)<=1e-12,
             "projected rejection modified byte probabilities");
       surgery.adopt_readout_from(*head_only);
       ++surgery_batches_rejected;
       ++surgery_head_only_batches;
      }
      surgery_candidate=std::make_unique<J>(surgery);
      // Fixed-permutation negative control with EXACT matching optimizer.
      const double sham_alignment=sham.shared_byte_directional_derivative(
          *sham_candidate,train);
      check(std::isfinite(sham_alignment),"nonfinite semantic-byte alignment");
      if(sham_alignment>0)++sham_conflicting_batches;
      bool sham_shared=false;
      for(const double alpha : {1.0,0.25,0.0625,0.015625,0.00390625}){
       auto trial=std::make_unique<J>(sham);
       trial->blend_projected_semantics_from(*sham_candidate,train,alpha);
       if(trial->shared_displacement_norm(sham)<=1e-12)continue;
       const double trial_ce=trial->evaluate_bytes(dev).ce();
       if(std::isfinite(trial_ce)&&trial_ce<=sham_accepted_dev+1e-12){
        sham.adopt_readout_from(*trial);
        sham_accepted_dev=trial_ce;
        ++sham_batches_accepted;
        sham_shared_alpha_sum+=alpha;
        sham_shared=true;
        break;
       }
      }
      if(!sham_shared){
       auto head_only=std::make_unique<J>(sham);
       head_only->blend_projected_semantics_from(*sham_candidate,train,0.0);
       const double same_ce=head_only->evaluate_bytes(dev).ce();
       check(std::abs(same_ce-sham_accepted_dev)<=1e-12,
             "projected rejection modified byte probabilities");
       sham.adopt_readout_from(*head_only);
       ++sham_batches_rejected;
       ++sham_head_only_batches;
      }
      sham_candidate=std::make_unique<J>(sham);
      }
     }
     // Check this is a matched comparison of a *single* fixed language
     // wave; the semantic heads are three separate in-memory readouts.
     check(universe.event_count()==events,"semantic head modified symbolic journal");
     const auto control_dev=compare(control,joint,shuffled,dev);
     const auto pareto_dev=pareto.evaluate_bytes(dev);
     const auto frozen_dev=frozen.evaluate_bytes(dev);
    const auto surgery_dev=surgery.evaluate_bytes(dev);
    const auto sham_dev=sham.evaluate_bytes(dev);
     check(frozen_dev.ce()==control_dev.byte.ce(),
           "direct frozen encoder changed byte development CE");
     check(frozen.projection_norm()==control.projection_norm(),
           "direct semantic head optimization altered shared projection");
     check(sham_dev.ce()<=control_dev.byte.ce()+1e-10,
          "shuffled projected arm worsened byte-development baseline");
    check(surgery_dev.ce()<=control_dev.byte.ce()+1e-10,
          "projected semantic arm worsened byte-development baseline");
    check(pareto_dev.ce()<=control_dev.byte.ce()+1e-10,
           "Pareto semantic readout worsened byte-development baseline");
     const auto before_early=compare(control,joint,shuffled,early);
     const auto before_late=compare(control,joint,shuffled,late);
     const auto pareto_early=pareto.evaluate_bytes(early);
     const auto pareto_late=pareto.evaluate_bytes(late);
     const auto frozen_early=frozen.evaluate_bytes(early);
     const auto frozen_late=frozen.evaluate_bytes(late);
    const auto surgery_early=surgery.evaluate_bytes(early);
    const auto surgery_late=surgery.evaluate_bytes(late);
    const auto sham_early=sham.evaluate_bytes(early);
    const auto sham_late=sham.evaluate_bytes(late);
     check(frozen_early.ce()==before_early.byte.ce()&&
           frozen_late.ce()==before_late.byte.ce(),
           "direct frozen encoder changed cross-author byte CE");
     check(pareto_early.count==before_early.byte.count&&
           pareto_late.count==before_late.byte.count,
           "Pareto model changed heldout sample sizes");
     const auto t_roles=test_roles();
     const auto t_senses=test_senses();
     const auto role_before=joint.evaluate_roles(t_roles);
     const auto sense_before=joint.evaluate_senses(t_senses,wn);
     const auto pareto_role=pareto.evaluate_roles(t_roles);
     const auto pareto_sense=pareto.evaluate_senses(t_senses,wn);
     const auto frozen_role=frozen.evaluate_roles(t_roles);
     const auto frozen_sense=frozen.evaluate_senses(t_senses,wn);
    const auto surgery_role=surgery.evaluate_roles(t_roles);
    const auto surgery_sense=surgery.evaluate_senses(t_senses,wn);
    const auto sham_role=sham.evaluate_roles(t_roles);
    const auto sham_sense=sham.evaluate_senses(t_senses,wn);
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
       <<" dev_pareto_ce="<<pareto_dev.ce()
       <<" dev_frozen_ce="<<frozen_dev.ce()
      <<" dev_surgery_ce="<<surgery_dev.ce()
      <<" dev_sham_ce="<<sham_dev.ce()
      <<" sham_batches_accepted="<<sham_batches_accepted
      <<" sham_batches_rejected="<<sham_batches_rejected
      <<" sham_head_only_batches="<<sham_head_only_batches
      <<" sham_conflicting_batches="<<sham_conflicting_batches
      <<" sham_shared_alpha_sum="<<sham_shared_alpha_sum
      <<" sham_role_test_ce="<<sham_role.ce()
      <<" sham_sense_test_ce="<<sham_sense.ce()
      <<" sham_role_test_accuracy="<<sham_role.accuracy()
      <<" sham_sense_test_accuracy="<<sham_sense.accuracy()
      <<" surgery_batches_accepted="<<surgery_batches_accepted
      <<" surgery_batches_rejected="<<surgery_batches_rejected
      <<" surgery_head_only_batches="<<surgery_head_only_batches
      <<" surgery_conflicting_batches="<<surgery_conflicting_batches
      <<" surgery_shared_alpha_sum="<<surgery_shared_alpha_sum
      <<" surgery_role_test_ce="<<surgery_role.ce()
      <<" surgery_sense_test_ce="<<surgery_sense.ce()
      <<" surgery_role_test_accuracy="<<surgery_role.accuracy()
      <<" surgery_sense_test_accuracy="<<surgery_sense.accuracy()
       <<" pareto_batches_accepted="<<pareto_batches_accepted
       <<" pareto_batches_rejected="<<pareto_batches_rejected
       <<" pareto_head_only_batches="<<pareto_head_only_batches
       <<" pareto_shared_alpha_sum="<<pareto_shared_alpha_sum
       <<" pareto_role_test_ce="<<pareto_role.ce()
       <<" pareto_sense_test_ce="<<pareto_sense.ce()
       <<" pareto_role_test_accuracy="<<pareto_role.accuracy()
       <<" pareto_sense_test_accuracy="<<pareto_sense.accuracy()
       <<" frozen_role_test_ce="<<frozen_role.ce()
       <<" frozen_sense_test_ce="<<frozen_sense.ce()
       <<" frozen_role_test_accuracy="<<frozen_role.accuracy()
       <<" frozen_sense_test_accuracy="<<frozen_sense.accuracy()
       <<" frozen_role_updates="<<frozen.role_updates()
       <<" frozen_sense_updates="<<frozen.sense_updates()
       <<" pareto_role_updates="<<pareto.role_updates()
       <<" pareto_sense_updates="<<pareto.sense_updates()
       <<" carol_early_byte_ce="<<before_early.byte.ce()
       <<" carol_early_joint_ce="<<before_early.joint.ce()
       <<" carol_early_shuffled_ce="<<before_early.shuffle.ce()
       <<" carol_early_pareto_ce="<<pareto_early.ce()
       <<" carol_early_frozen_ce="<<frozen_early.ce()
      <<" carol_early_surgery_ce="<<surgery_early.ce()
      <<" carol_early_sham_ce="<<sham_early.ce()
       <<" carol_early_post_recurrence_ce="<<after_early.ce()
       <<" carol_late_byte_ce="<<before_late.byte.ce()
       <<" carol_late_joint_ce="<<before_late.joint.ce()
       <<" carol_late_shuffled_ce="<<before_late.shuffle.ce()
       <<" carol_late_pareto_ce="<<pareto_late.ce()
       <<" carol_late_frozen_ce="<<frozen_late.ce()
      <<" carol_late_surgery_ce="<<surgery_late.ce()
      <<" carol_late_sham_ce="<<sham_late.ce()
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
