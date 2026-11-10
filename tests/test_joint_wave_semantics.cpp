#include "nanlm/joint_wave_semantics.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>
using namespace nanlm;
static void require(bool cond,const char* msg){if(!cond)throw std::runtime_error(msg);}
int main(){
 const std::string root=(std::filesystem::temp_directory_path()/
   ("nanlm-joint-"+std::to_string(::getpid()))).string();
 auto cleanup=[&](){
  for(const char* p:{"",".overlay",".so3",".episodes",".wordnet.tsv",
                     ".joint-wave",".joint-wave.tmp","-byte.joint-wave",
                     "-shuffle.joint-wave","-role.joint-wave","-sense.joint-wave"})
   std::filesystem::remove(root+p);
 };
 cleanup();
 try{
  {
   std::ofstream out(root+".wordnet.tsv");
   out<<"NANLM-WN1\t2\t0\n";
   out<<"N\t801\t101\tbank\tfinancial institution\n";
   out<<"N\t802\t202\tbank\triver shoreline\n";
   out.flush();
   require(bool(out),"wordnet fixture creation");
  }
  WordNetIndex wn;wn.load(root+".wordnet.tsv");
  require(wn.lookup("bank").size()==2,"test synsets");
  Universe universe(root);
  UnifiedWaveLanguage wave(universe);
  const std::string language=
   "ada gave parcel to bea.\n"
   "bea received parcel from ada.\n"
   "cash was deposited at the bank.\n"
   "water rushed past the bank.\n";
  const std::string held=
   "bea sent a parcel to ada.\n"
   "the boat floated near the bank.\n";
  {
   std::istringstream training(language);
   const auto m=wave.train(training);
   require(m.tokens>10&&std::isfinite(m.ce()),"wave language initialization");
  }
  const size_t baseline_events=universe.event_count();
  const std::string probe="the bank approved a loan";
  const auto probe_state= [&](){
   UnifiedWaveLanguage::State state{};
   for(unsigned char c:probe)wave.observe(state,c);
   return state;
  }();
  const auto baseline_wave=[&](){
   std::istringstream in(held);
   return wave.evaluate(in).ce();
  }();
  JointWaveSemanticLearner joint(wave,root);
  JointWaveSemanticLearner byte_only(wave,root+"-byte");
  JointWaveSemanticLearner shuffled(wave,root+"-shuffle");
  JointWaveSemanticLearner role_only(wave,root+"-role");
  JointWaveSemanticLearner sense_only(wave,root+"-sense");
  const double projection_initial=joint.projection_norm();
  // Identical deterministic byte warm-up for all three arms.
  for(size_t epoch=0;epoch<4;++epoch){
   UnifiedWaveLanguage::State state{};
   for(unsigned char c:language){
    if(state.length){
     joint.teach_byte(state,c,0.02);
     byte_only.teach_byte(state,c,0.02);
     shuffled.teach_byte(state,c,0.02);
     role_only.teach_byte(state,c,0.02);
     sense_only.teach_byte(state,c,0.02);
    }
    wave.observe(state,c);
   }
  }
  const auto b0=joint.predict_byte(probe_state);
  const double p0=b0.p[static_cast<uint8_t>(' ')];
  const std::vector<JointWaveSemanticLearner::Role> roles={
   {"ada gave parcel to bea","ada","bea","transfer",1},
   {"ada gave parcel to bea","bea","ada","transfer",0},
   {"bea received parcel from ada","ada","bea","transfer",1},
   {"bea received parcel from ada","bea","ada","transfer",0},
   {"ada followed bea","ada","bea","follow",1},
   {"ada followed bea","bea","ada","follow",0},
   {"bea was followed by ada","ada","bea","follow",1},
   {"bea was followed by ada","bea","ada","follow",0}
  };
  const std::vector<JointWaveSemanticLearner::Role> test_roles={
   {"cato gave parcel to dara","cato","dara","transfer",1},
   {"cato gave parcel to dara","dara","cato","transfer",0},
   {"dara received parcel from cato","cato","dara","transfer",1},
   {"dara received parcel from cato","dara","cato","transfer",0},
   {"cato followed dara","cato","dara","follow",1},
   {"cato followed dara","dara","cato","follow",0},
   {"dara was followed by cato","cato","dara","follow",1},
   {"dara was followed by cato","dara","cato","follow",0}
  };
  const std::vector<JointWaveSemanticLearner::Sense> senses={
   {"the bank approved a loan","bank",801},
   {"cash was deposited at the bank","bank",801},
   {"the bank issued credit","bank",801},
   {"the water rushed past the bank","bank",802},
   {"the boat floated near the bank","bank",802},
   {"the bank was wet with river mud","bank",802}
  };
  const std::vector<JointWaveSemanticLearner::Sense> test_senses={
   {"the bank offered credit","bank",801},
   {"a loan came from the bank","bank",801},
   {"the river passed the bank","bank",802},
   {"mud covered the bank","bank",802}
  };
  // Seeded, fixed *permutation* controls. Keep each randomized label fixed
  // across epochs; changing the target every epoch is not a proper shuffle.
  std::vector<int> role_labels;
  for(const auto& row:roles)role_labels.push_back(row.label);
  std::vector<uint32_t> sense_labels;
  for(const auto& row:senses)sense_labels.push_back(row.gold);
  std::mt19937_64 rng(20261109);
  std::shuffle(role_labels.begin(),role_labels.end(),rng);
  std::shuffle(sense_labels.begin(),sense_labels.end(),rng);
  // Both negative controls and task-specific ablations share the same
  // byte warm-up, random initialization and frozen UnifiedWave foundation.
  for(size_t epoch=0;epoch<35;++epoch) {
   for(size_t i=0;i<roles.size();++i){
    const auto s=joint.state_for(roles[i].sentence);
    joint.teach_role(s,roles[i],0.04);
    role_only.teach_role(s,roles[i],0.04);
    auto random_role=roles[i];
    random_role.label=role_labels[i];
    shuffled.teach_role(s,random_role,0.04);
   }
   for(size_t i=0;i<senses.size();++i){
    const auto s=joint.state_for(senses[i].sentence);
    joint.teach_sense(s,senses[i],wn,0.04);
    sense_only.teach_sense(s,senses[i],wn,0.04);
    auto random_sense=senses[i];
    random_sense.gold=sense_labels[i];
    shuffled.teach_sense(s,random_sense,wn,0.04);
   }
  }
  const auto b1=joint.predict_byte(probe_state);
  const double p1=b1.p[static_cast<uint8_t>(' ')];
  // The semantic error must reach a parameter used by the byte readout.
  require(std::abs(p1-p0)>1e-12,"semantic gradients do not change byte predictions");
  require(std::abs(joint.projection_norm()-projection_initial)>1e-12,
          "joint projection was not trained");
  require(joint.byte_updates()>0&&joint.role_updates()>0&&joint.sense_updates()>0,
          "missing one of three gradients");
  require(universe.event_count()==baseline_events,
          "neural predictions contaminated exact NaN16 evidence");
  // A semantic head-only fallback must not change language predictions.
  // A positive shared-projection blend must reproduce the candidate at 1.0.
  {
   auto proposal=byte_only;
   for(size_t i=0;i<10;++i){
    proposal.teach_role(proposal.state_for(roles[0].sentence),roles[0],0.04);
    proposal.teach_sense(proposal.state_for(senses[0].sentence),senses[0],wn,0.04);
   }
   const double original_ce=byte_only.evaluate_bytes(held).ce();
   const auto original_prob=byte_only.predict_byte(probe_state);
   auto head_only=byte_only;
   head_only.blend_semantics_from(proposal,0.0);
   const auto no_credit=head_only.predict_byte(probe_state);
   for(size_t i=0;i<original_prob.p.size();++i)
    require(no_credit.p[i]==original_prob.p[i],
            "head-only blend changed byte probability");
   require(head_only.evaluate_bytes(held).ce()==original_ce,
           "head-only blend changed heldout byte CE");
   require(head_only.role_updates()>0&&head_only.sense_updates()>0,
           "head-only specialization lost semantic training");
   auto full=byte_only;
   full.blend_semantics_from(proposal,1.0);
   const auto full_prob=full.predict_byte(probe_state);
   const auto direct=proposal.predict_byte(probe_state);
   for(size_t i=0;i<direct.p.size();++i)
    require(std::abs(full_prob.p[i]-direct.p[i])<1e-14,
            "full semantic blend differs from proposal");
   require(std::abs(full_prob.p[' ']-original_prob.p[' '])>1e-12,
           "positive shared blend produced no byte change");
   bool invalid=false;
   try{head_only.blend_semantics_from(proposal,-0.1);}
   catch(const std::invalid_argument&){invalid=true;}
   require(invalid,"negative semantic blend accepted");
   invalid=false;
   try{head_only.blend_semantics_from(joint,0.5);}
   catch(const std::invalid_argument&){invalid=true;}
   require(invalid,"unrelated readout semantics were blended");
  }
  const auto held_joint=joint.evaluate_bytes(held);
  const auto held_byte=byte_only.evaluate_bytes(held);
  const auto held_shuffled=shuffled.evaluate_bytes(held);
  const auto held_role_only=role_only.evaluate_bytes(held);
  const auto held_sense_only=sense_only.evaluate_bytes(held);
  const auto role_score=joint.evaluate_roles(test_roles);
  const auto role_shuffle=shuffled.evaluate_roles(test_roles);
  const auto sense_score=joint.evaluate_senses(test_senses,wn);
  const auto sense_shuffle=shuffled.evaluate_senses(test_senses,wn);
  require(held_joint.count>0&&held_byte.count==held_joint.count,
          "byte evaluation sample parity");
  for(double value:{held_joint.ce(),held_byte.ce(),held_shuffled.ce(),
                    held_role_only.ce(),held_sense_only.ce(),
                    role_score.ce(),role_shuffle.ce(),sense_score.ce(),sense_shuffle.ce()})
   require(std::isfinite(value)&&value>=0,"nonfinite heldout joint benchmark");
  {
   std::istringstream after(held);
   require(std::abs(wave.evaluate(after).ce()-baseline_wave)<1e-12,
           "experimental head altered production byte predictor");
  }
  // Independent finite-difference oracle for the *true* existing wave
  // recurrence. This audit restores raw register bytes, even when the
  // original recurrent gain was encoded by an absent magic marker.
  auto gradient_audit=[&](const std::string& sentence,
                          const UnifiedWaveLanguage::WaveAdjoint& terminal,
                          auto loss)->std::pair<double,double>{
   require(sentence.size()>8,"semantic tape sentence too short");
   auto prefix=joint.state_for(sentence.substr(0,sentence.size()-8));
   std::deque<UnifiedWaveLanguage::State> tape{prefix};
   std::deque<uint8_t> symbols;
   for(unsigned char c:sentence.substr(sentence.size()-8)){
    auto after=tape.back();
    wave.observe(after,c);
    tape.push_back(std::move(after));
    symbols.push_back(c);
   }
   const auto recurrent=wave.semantic_wave_gradient(tape,symbols,terminal);
   require(recurrent.steps==8&&std::isfinite(recurrent.norm),
           "semantic eight-step backprop not finite");
   const uint8_t target=symbols.back();
   const double gain=wave.recurrent_input_gain(target);
   auto& raw=universe.root().h[target].data;
   std::array<uint8_t,5> saved{};
   std::memcpy(saved.data(),raw+UnifiedWaveLanguage::BPTT_INPUT_OFFSET,4);
   saved[4]=raw[UnifiedWaveLanguage::BPTT_MAGIC_OFFSET];
   auto restore=[&](){
    std::memcpy(raw+UnifiedWaveLanguage::BPTT_INPUT_OFFSET,saved.data(),4);
    raw[UnifiedWaveLanguage::BPTT_MAGIC_OFFSET]=saved[4];
   };
   auto trial=[&](double offset){
    wave.set_recurrent_input_gain(target,gain+offset);
    auto state=tape.front();
    for(uint8_t byte:symbols)wave.observe(state,byte);
    return loss(state);
   };
   double plus=0,minus=0;
   try {
    plus=trial(0.005);
    minus=trial(-0.005);
   }catch(...){restore();throw;}
   restore();
   const double numerical=(plus-minus)/0.01;
   const double analytic=recurrent.input[target];
   require(std::isfinite(numerical)&&std::isfinite(analytic)&&
           std::abs(analytic-numerical)<1e-6+0.02*std::abs(numerical),
           "semantic recurrent adjoint disagrees with finite differences");
   require(universe.event_count()==baseline_events,
           "semantic finite difference asserted physical facts");
   return {analytic,numerical};
  };
  const auto ws=joint.state_for(senses.front().sentence);
  const auto sense_audit=gradient_audit(
   senses.front().sentence,
   joint.sense_wave_adjoint(ws,senses.front(),wn),
   [&](const auto& state){return joint.sense_loss(state,senses.front(),wn);});
  const auto wr=joint.state_for(test_roles.front().sentence);
  const auto role_audit=gradient_audit(
   test_roles.front().sentence,
   joint.role_wave_adjoint(wr,test_roles.front()),
   [&](const auto& state){return joint.role_loss(state,test_roles.front());});
  joint.save();
  {
   JointWaveSemanticLearner restored(wave,root);
   require(restored.byte_updates()==joint.byte_updates()&&
           restored.role_updates()==joint.role_updates()&&
           restored.sense_updates()==joint.sense_updates(),
           "joint update counters not persistent");
   const auto reload=restored.predict_byte(probe_state);
   require(std::abs(reload.p[' ']-p1)<1e-12,
           "joint byte distribution changed on restart");
   require(std::abs(restored.evaluate_roles(test_roles).ce()-role_score.ce())<1e-10,
           "joint role predictions changed on restart");
   require(std::abs(restored.evaluate_senses(test_senses,wn).ce()-sense_score.ce())<1e-10,
           "joint sense predictions changed on restart");
  }
  {
   std::ofstream out(root+".joint-wave",std::ios::app);
   out<<"corruption";
   out.close();
   bool rejected=false;
   try{JointWaveSemanticLearner invalid(wave,root);(void)invalid;}
   catch(const std::runtime_error&){rejected=true;}
   require(rejected,"corrupted shared checkpoint was accepted");
  }
  // A live root update is opt-in and guarded by separate byte DEV CE.
  // Run this only after checking earlier fixed-base predictions, checkpoint
  // restart and corruption handling.
  const std::string byte_dev=
   "cash was deposited at the bank.\n"
   "water rushed past the bank.\n";
  const auto sense_before=joint.sense_loss(
   joint.state_for(senses.front().sentence),senses.front(),wn);
  const double dev_before=joint.evaluate_bytes(byte_dev).ce();
  const auto journal_before_step=universe.event_count();
  const bool accepted_recurrent=joint.guarded_sense_recurrent_step(
   universe,senses.front(),wn,byte_dev,0.0);
  const double sense_after=joint.sense_loss(
   joint.state_for(senses.front().sentence),senses.front(),wn);
  const double dev_after=joint.evaluate_bytes(byte_dev).ce();
  require(universe.event_count()==journal_before_step,
          "guarded recurrent step modified symbolic journal");
  if(accepted_recurrent){
   require(sense_after+1e-10<sense_before,
           "accepted recurrence update failed exact semantic replay");
   require(dev_after<=dev_before+1e-10,
           "accepted recurrence update violated byte dev guard");
  }else {
   require(std::abs(sense_after-sense_before)<1e-12&&
           std::abs(dev_after-dev_before)<1e-12,
           "rejected recurrence update did not restore original root");
  }
  bool blocked_regression=false;
  try {
   joint.guarded_role_recurrent_step(universe,test_roles.front(),byte_dev,-0.001);
  }catch(const std::invalid_argument&){blocked_regression=true;}
  require(blocked_regression,"invalid byte regression budget accepted");
  universe.sync();
  // Independent Universe reader proves a committed gain is stored in the
  // original mapped root rather than in an out-of-band recurrent model.
  {
   Universe reread(root);
   for(uint8_t c:std::string("d a loan")){
    const auto& original=universe.root().h[c].data;
    const auto& loaded=reread.root().h[c].data;
    require(std::memcmp(original+UnifiedWaveLanguage::BPTT_INPUT_OFFSET,
                        loaded+UnifiedWaveLanguage::BPTT_INPUT_OFFSET,4)==0&&
            original[UnifiedWaveLanguage::BPTT_MAGIC_OFFSET]==
            loaded[UnifiedWaveLanguage::BPTT_MAGIC_OFFSET],
            "recurrent input gain failed root persistence");
   }
  }
  // Heldout test remains *outside* the acceptance decision; report any
  // post-recurrence interference rather than silently optimizing on test.
  const auto byte_after_root=joint.evaluate_bytes(held);
  const auto roles_after_root=joint.evaluate_roles(test_roles);
  const auto senses_after_root=joint.evaluate_senses(test_senses,wn);
  require(std::isfinite(byte_after_root.ce())&&
          std::isfinite(roles_after_root.ce())&&
          std::isfinite(senses_after_root.ce()),
          "post-recurrence heldout metrics invalid");
  std::cout<<"JOINT_WAVE_NATIVE PASS"
           <<" byte_heldout_ce="<<held_joint.ce()
           <<" byte_only_ce="<<held_byte.ce()
           <<" shuffled_semantic_byte_ce="<<held_shuffled.ce()
           <<" role_only_byte_ce="<<held_role_only.ce()
           <<" sense_only_byte_ce="<<held_sense_only.ce()
           <<" production_wave_ce="<<baseline_wave
           <<" role_heldout_ce="<<role_score.ce()
           <<" role_accuracy="<<role_score.accuracy()
           <<" role_shuffled_accuracy="<<role_shuffle.accuracy()
           <<" sense_heldout_ce="<<sense_score.ce()
           <<" sense_accuracy="<<sense_score.accuracy()
           <<" sense_shuffled_accuracy="<<sense_shuffle.accuracy()
           <<" byte_probability_shift="<<(p1-p0)
           <<" recurrent_update_accepted="<<int(accepted_recurrent)
           <<" recurrent_semantic_loss_before="<<sense_before
           <<" recurrent_semantic_loss_after="<<sense_after
           <<" recurrent_byte_dev_before="<<dev_before
           <<" recurrent_byte_dev_after="<<dev_after
           <<" recurrent_byte_test_ce_before="<<held_joint.ce()
           <<" recurrent_byte_test_ce_after="<<byte_after_root.ce()
           <<" recurrent_role_test_ce_after="<<roles_after_root.ce()
           <<" recurrent_sense_test_ce_after="<<senses_after_root.ce()
           <<" sense_recurrent_grad="<<sense_audit.first
           <<" sense_recurrent_fd="<<sense_audit.second
           <<" role_recurrent_grad="<<role_audit.first
           <<" role_recurrent_fd="<<role_audit.second
           <<" wave_state_shared=1"
           <<" gradient_through_recurrence=1"
           <<" byte_updates="<<joint.byte_updates()
           <<" role_updates="<<joint.role_updates()
           <<" sense_updates="<<joint.sense_updates()
           <<" journal_events="<<baseline_events<<"\n";
  cleanup();
  return 0;
 }catch(const std::exception& e){
  std::cerr<<"JOINT_WAVE_NATIVE FAIL: "<<e.what()<<"\n";
  cleanup();
  return 1;
 }
}
