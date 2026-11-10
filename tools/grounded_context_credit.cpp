// Gutenberg source-grounded contextual word-order discrimination.
// Genuine Alice training text creates originals versus within-sentence
// token swaps; heldout Dickens provides independent test tasks.
// This is NOT a WordNet semantic sense annotation or language understanding.
#include "nanlm/joint_wave_semantics.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
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
static void require(bool x,const char* why){if(!x)throw std::runtime_error(why);}
static std::string read(const std::string& path,size_t low,size_t high){
 std::ifstream in(path,std::ios::binary);
 if(!in)throw std::runtime_error("cannot read pinned split");
 std::string text((std::istreambuf_iterator<char>(in)),{});
 if(text.size()<low||text.size()>high||
    text.find(char(255))!=std::string::npos)
  throw std::runtime_error("invalid pinned split bytes");
 return text;
}
static std::string join(const std::vector<std::string>& words){
 std::string s;
 for(const auto& word:words){if(!s.empty())s.push_back(' ');s+=word;}
 return s;
}
static std::vector<J::Role> grounded_rows(const std::string& source,
                                           size_t max_pairs){
 std::vector<J::Role> rows;
 std::vector<std::string> segment;
 std::string word;
 auto emit=[&](){
  if(!word.empty()){segment.push_back(word);word.clear();}
 };
 auto flush=[&](){
  emit();
  for(size_t at=0;at+12<=segment.size()&&rows.size()/2<max_pairs;at+=4){
   std::vector<std::string> window(segment.begin()+at,
                                    segment.begin()+at+12);
   const std::string a=window.front(),b=window.back();
   if(a==b||window[3]==window[8]||
      std::count(window.begin(),window.end(),a)!=1||
      std::count(window.begin(),window.end(),b)!=1)continue;
   const std::string original=join(window);
   std::swap(window[3],window[8]);
   const std::string disrupted=join(window);
   if(original==disrupted)continue;
   // The two entity anchors are unchanged: role orientation cannot
   // distinguish positive/negative labels; only word context can.
   rows.push_back(J::Role{original,a,b,"corpus_order",1});
   rows.push_back(J::Role{disrupted,a,b,"corpus_order",0});
  }
  segment.clear();
 };
 for(unsigned char c:source){
  if((c>='A'&&c<='Z')||(c>='a'&&c<='z')||
     (c>='0'&&c<='9')||c=='_'){
   if(word.size()>=128)throw std::runtime_error("corpus token too long");
   word.push_back(c>='A'&&c<='Z'?char(c-'A'+'a'):char(c));
  }else if(c=='.'||c=='!'||c=='?'||c==';'||c==':'){
   flush();
  }else emit();
 }
 flush();
 if(rows.empty()||rows.size()%2)
  throw std::runtime_error("no balanced source-grounded contexts");
 return rows;
}
static void cleanup(const std::string& root){
 for(const char* suffix:{"",".overlay",".overlay.tmp",".so3",".episodes",
     "-byte.joint-wave","-true.joint-wave","-sham.joint-wave",
     "-frozen.joint-wave",".joint-wave",".joint-wave.tmp"})
  std::filesystem::remove(root+suffix);
}
static double weighted_byte_ce(const J& model,const std::string& a,
                               const std::string& b){
 const auto p=model.evaluate_bytes(a),q=model.evaluate_bytes(b);
 require(p.count==a.size()-1&&q.count==b.size()-1,
         "mismatched cross-author byte sample sizes");
 return (p.loss+q.loss)/double(p.count+q.count);
}
int main(int argc,char** argv){
 if(argc!=6){
  std::cerr<<"usage: grounded_context_credit ALICE_TRAIN ALICE_DEV "
              "CAROL_EARLY CAROL_LATE SEEDS\n";
  return 2;
 }
 try{
  const auto train=read(argv[1],1500,6000);
  const auto dev=read(argv[2],300,2000);
  const auto early=read(argv[3],400,3000);
  const auto late=read(argv[4],400,3000);
  const size_t seeds=std::stoull(argv[5]);
  require(seeds>=2&&seeds<=4,"seed count must be 2..4");
  const auto source_rows=grounded_rows(train,48);
  const auto early_rows=grounded_rows(early,48);
  const auto late_rows=grounded_rows(late,48);
  std::vector<J::Role> test_rows=early_rows;
  test_rows.insert(test_rows.end(),late_rows.begin(),late_rows.end());
  require(source_rows.size()>=24&&test_rows.size()>=12,
          "source-derived context sample too small");
  for(size_t k=0;k<seeds;++k){
   const uint64_t order_seed=20261109ull+103ull*k;
   const uint64_t init_seed=314159ull+104729ull*k;
   const std::string root=(std::filesystem::temp_directory_path()/
       ("nanlm-grounded-context-"+std::to_string(::getpid())+"-"+
        std::to_string(k))).string();
   cleanup(root);
   try{
    {
     Universe universe(root);
     UnifiedWaveLanguage wave(universe);
     std::istringstream foundation(train.substr(0,1300));
     const auto fit=wave.train(foundation);
     require(fit.tokens>=1000&&std::isfinite(fit.ce()),
             "causal wave Alice foundation failed");
     const auto journal=universe.event_count();
     J control(wave,root+"-byte",init_seed);
     J truth(wave,root+"-true",init_seed);
     J shuffled(wave,root+"-sham",init_seed);
     J frozen(wave,root+"-frozen",init_seed);
     std::mt19937_64 rng(order_seed);
     std::vector<size_t> chunks;
     for(size_t i=0;i<train.size();i+=192)chunks.push_back(i);
     std::shuffle(chunks.begin(),chunks.end(),rng);
     for(size_t at:chunks){
      J::State state{};
      for(size_t i=at;i<std::min(at+192,train.size());++i){
       const auto c=static_cast<uint8_t>(
           static_cast<unsigned char>(train[i]));
       if(state.length){
        control.teach_byte(state,c,0.02);
        truth.teach_byte(state,c,0.02);
        shuffled.teach_byte(state,c,0.02);
        frozen.teach_byte(state,c,0.02);
       }
       wave.observe(state,c);
      }
     }
     require(control.evaluate_bytes(dev).ce()==truth.evaluate_bytes(dev).ce()&&
             control.evaluate_bytes(dev).ce()==shuffled.evaluate_bytes(dev).ce(),
             "matched byte warmup diverged");
     std::vector<J::State> cached;
     cached.reserve(source_rows.size());
     for(const auto& row:source_rows)cached.push_back(truth.state_for(row.sentence));
     std::vector<int> random_labels;
     for(const auto& row:source_rows)random_labels.push_back(row.label);
     std::shuffle(random_labels.begin(),random_labels.end(),rng);
     require(std::count(random_labels.begin(),random_labels.end(),1)==
             source_rows.size()/2,"permuted class balance changed");
     std::vector<size_t> order(source_rows.size());
     std::iota(order.begin(),order.end(),0u);
     auto true_candidate=std::make_unique<J>(truth);
     auto sham_candidate=std::make_unique<J>(shuffled);
     double truth_dev_ce=truth.evaluate_bytes(dev).ce();
     double sham_dev_ce=shuffled.evaluate_bytes(dev).ce();
     size_t truth_accepted=0,sham_accepted=0;
     size_t truth_rejected=0,sham_rejected=0;
     // Source-derived authentic/corrupted contexts, not synthetic sense IDs.
     // All four arms receive identical causal wave states and byte updates.
     for(size_t epoch=0;epoch<12;++epoch){
      std::shuffle(order.begin(),order.end(),rng);
      for(size_t idx:order){
       const auto& row=source_rows[idx];
       true_candidate->teach_role(cached[idx],row,0.035);
       auto negative_control=row;negative_control.label=random_labels[idx];
       sham_candidate->teach_role(cached[idx],negative_control,0.035);
       frozen.teach_role(cached[idx],row,0.035,false);
      }
      if((epoch+1)%2!=0)continue;
      auto consolidate=[&](J& current,std::unique_ptr<J>& proposal,
                           double& best,size_t& accepted,size_t& rejected){
       bool kept=false;
       for(double alpha:{1.0,0.25,0.0625,0.015625,0.00390625}){
        auto trial=std::make_unique<J>(current);
        trial->blend_projected_semantics_from(*proposal,train,alpha);
        if(trial->shared_displacement_norm(current)<=1e-12)continue;
        const double ce=trial->evaluate_bytes(dev).ce();
        if(std::isfinite(ce)&&ce<=best+1e-12){
         current.adopt_readout_from(*trial);
         best=ce;
         ++accepted;
         kept=true;
         break;
        }
       }
       if(!kept){
        auto head_only=std::make_unique<J>(current);
        head_only->blend_projected_semantics_from(*proposal,train,0.0);
        require(std::abs(head_only->evaluate_bytes(dev).ce()-best)<1e-12,
                "rejected shared candidate modified language head");
        current.adopt_readout_from(*head_only);
        ++rejected;
       }
       proposal=std::make_unique<J>(current);
      };
      consolidate(truth,true_candidate,truth_dev_ce,
                  truth_accepted,truth_rejected);
      consolidate(shuffled,sham_candidate,sham_dev_ce,
                  sham_accepted,sham_rejected);
     }
     const auto truth_roles=truth.evaluate_roles(test_rows);
     const auto shuffled_roles=shuffled.evaluate_roles(test_rows);
     const auto frozen_roles=frozen.evaluate_roles(test_rows);
     const double byte_ce=weighted_byte_ce(control,early,late);
     const double true_ce=weighted_byte_ce(truth,early,late);
     const double sham_ce=weighted_byte_ce(shuffled,early,late);
     const double frozen_ce=weighted_byte_ce(frozen,early,late);
     require(truth_roles.count==test_rows.size()&&
             shuffled_roles.count==test_rows.size()&&
             frozen_roles.count==test_rows.size()&&
             std::isfinite(truth_roles.ce())&&
             std::isfinite(shuffled_roles.ce())&&
             std::isfinite(frozen_roles.ce()),
             "invalid heldout contextual order classification");
     require(truth_dev_ce<=control.evaluate_bytes(dev).ce()+1e-10&&
             sham_dev_ce<=control.evaluate_bytes(dev).ce()+1e-10,
             "projected joint model worsened independent Alice DEV CE");
     require(frozen_ce==byte_ce,"frozen semantic training altered byte CE");
     require(universe.event_count()==journal,"grounded classifier changed exact journal");
     require(std::filesystem::file_size(root)==ROOT_BYTES,
             "physical 8MiB universe was replaced");
     // Check V2 sidecar and persist real extracted contextual discrimination.
     truth.save();
     J restored(wave,root+"-true",init_seed);
     require(std::abs(restored.evaluate_roles(test_rows).ce()-truth_roles.ce())<1e-11&&
             std::abs(weighted_byte_ce(restored,early,late)-true_ce)<1e-11,
             "grounded contextual readout did not survive exact reload");
     std::cout<<std::setprecision(12)<<"GROUNDED_CONTEXT_SEED"
      <<" seed="<<order_seed
      <<" readout_init_seed="<<init_seed
      <<" train_pairs="<<source_rows.size()/2
      <<" test_pairs="<<test_rows.size()/2
      <<" heldout_byte_tokens="<<early.size()+late.size()-2
      <<" truth_shared_accepted="<<truth_accepted
      <<" truth_shared_rejected="<<truth_rejected
      <<" sham_shared_accepted="<<sham_accepted
      <<" sham_shared_rejected="<<sham_rejected
      <<" alice_dev_byte_ce="<<control.evaluate_bytes(dev).ce()
      <<" alice_dev_truth_ce="<<truth_dev_ce
      <<" alice_dev_sham_ce="<<sham_dev_ce
      <<" dickens_byte_ce="<<byte_ce
      <<" dickens_truth_ce="<<true_ce
      <<" dickens_sham_ce="<<sham_ce
      <<" dickens_frozen_ce="<<frozen_ce
      <<" dickens_truth_minus_byte="<<true_ce-byte_ce
      <<" dickens_sham_minus_byte="<<sham_ce-byte_ce
      <<" dickens_truth_minus_sham="<<true_ce-sham_ce
      <<" context_truth_ce="<<truth_roles.ce()
      <<" context_sham_ce="<<shuffled_roles.ce()
      <<" context_frozen_ce="<<frozen_roles.ce()
      <<" context_truth_acc="<<truth_roles.accuracy()
      <<" context_sham_acc="<<shuffled_roles.accuracy()
      <<" context_frozen_acc="<<frozen_roles.accuracy()
      <<" root_bytes="<<ROOT_BYTES
      <<" journal_events="<<journal
      <<"\n"<<std::flush;
    }
    cleanup(root);
   }catch(...){cleanup(root);throw;}
  }
  std::cout<<"GROUNDED_CONTEXT_AUDIT PASS source_derived=1 "
            "author_disjoint=1 test_not_used_for_selection=1\n";
  return 0;
 }catch(const std::exception& e){
  std::cerr<<"GROUNDED_CONTEXT_AUDIT FAIL "<<e.what()<<"\n";
  return 1;
 }
}
