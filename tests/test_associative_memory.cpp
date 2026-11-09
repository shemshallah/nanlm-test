#include "nanlm/episodic_memory.hpp"
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

using namespace nanlm;
static void require(bool yes,const char* reason){
 if(!yes)throw std::runtime_error(reason);
}
static uint32_t step(uint32_t& seed){
 seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;return seed;
}
static std::string key(unsigned record){
 uint32_t seed=0xA516CE2Du+record*747796405u;
 std::string cue="Archive research ";
 for(size_t i=0;i<16;++i)
  cue.push_back(char('a'+step(seed)%26));
 cue+=" key: ";
 return cue;
}
int main(){
 const std::string path=(std::filesystem::temp_directory_path()/
  ("nanlm-associative-"+std::to_string(::getpid())+".episodes")).string();
 auto cleanup=[&](){
  std::error_code e;std::filesystem::remove(path,e);
  std::filesystem::remove(path+".tmp",e);
 };
 try {
  constexpr unsigned RECORDS=128;
  ExactEpisodicMemory bank(path);
  std::ostringstream contents;
  for(unsigned i=0;i<RECORDS;++i)
   contents<<key(i)<<"\t"<<"source-backed-value-"<<i<<"\n";
  std::istringstream source(contents.str());
  require(bank.import_tsv(source,"synthetic-fixed-seed")==RECORDS,
          "failed to load lexical retrieval fixture");
  TriadCueRouter::Mapping groups{};
  for(size_t i=0;i<groups.size();++i)
   groups[i]=static_cast<uint8_t>(i/3);
  std::array<double,255> synthetic_wave{};
  for(size_t i=0;i<synthetic_wave.size();++i)
   synthetic_wave[i]=std::sin(0.13*double(i));
  size_t trials=0,baseline_top1=0,baseline_top3=0;
  size_t edit_top1=0,edit_top3=0,triad_top1=0,triad_top3=0;
  double baseline_micros=0,edit_micros=0,triad_micros=0;
  for(unsigned i=0;i<RECORDS;++i){
   const std::string exact=key(i);
   const size_t offset=std::string("Archive research ").size();
   std::array<std::string,4> corrupt{
      exact,exact,exact,exact
   };
   auto change=[](char& ch){ch=ch=='z'?'y':'z';};
   change(corrupt[0][offset+2]); // substitution
   corrupt[1].insert(offset+4,1,'x'); // insertion
   corrupt[2].erase(offset+6,1); // deletion
   change(corrupt[3][offset+1]); // two substitutions
   change(corrupt[3][offset+9]);
   for(const auto& typo:corrupt){
    ++trials;
    const auto t0=std::chrono::steady_clock::now();
    const auto phase=bank.suggest(
      reinterpret_cast<const uint8_t*>(typo.data()),typo.size(),3,
      &synthetic_wave);
    const auto t1=std::chrono::steady_clock::now();
    const auto lexical=bank.suggest_robust(
      reinterpret_cast<const uint8_t*>(typo.data()),typo.size(),3,2,
      groups,nullptr,false);
    const auto t2=std::chrono::steady_clock::now();
    const auto hybrid=bank.suggest_robust(
      reinterpret_cast<const uint8_t*>(typo.data()),typo.size(),3,2,
      groups,&synthetic_wave,true);
    const auto t3=std::chrono::steady_clock::now();
    const auto elapsed=[](auto first,auto second){
     return std::chrono::duration<double,std::micro>(
       second-first).count();
    };
    baseline_micros+=elapsed(t0,t1);
    edit_micros+=elapsed(t1,t2);
    triad_micros+=elapsed(t2,t3);
    auto rank=[&](const auto& records){
     size_t found=0;
     for(size_t j=0;j<records.size();++j)
      if(records[j].cue==exact)return j+1;
     return found;
    };
    const size_t a=rank(phase),b=rank(lexical),c=rank(hybrid);
    baseline_top1+=(a==1);baseline_top3+=(a>0&&a<=3);
    edit_top1+=(b==1);edit_top3+=(b>0&&b<=3);
    triad_top1+=(c==1);triad_top3+=(c>0&&c<=3);
    // No approximate cue may trigger byte-exact copy, even if its
    // candidate rank is excellent.
    const auto copy=bank.probe(
       reinterpret_cast<const uint8_t*>(typo.data()),typo.size(),false);
    require(copy.record==ExactEpisodicMemory::NONE,
            "noisy associative suggestion bypassed byte verification");
    for(const auto& s:hybrid)
     require(!s.exact_verified&&s.hot_triads<=8&&
             std::isfinite(s.triad_similarity),
             "approximate result marked verified or has invalid hotspots");
   }
  }
  require(trials==512,"unexpected number of typo challenge trials");
  require(triad_top3>=edit_top3-10,
          "triad hints severely degraded lexical shortlist recall");
  require(triad_top3>baseline_top3,
          "hybrid router did not recover more typo cues than phase-only");
  // Each metric measures only source-cue rank, NOT correctness of the
  // source's values, natural-language reasoning or general semantics.
  const auto out=[&](const char* name,size_t a,size_t b,double micros){
   std::cout<<"ASSOCIATIVE_BENCH mode="<<name
            <<" trials="<<trials
            <<" top1="<<a
            <<" top3="<<b
            <<" top1_rate="<<double(a)/trials
            <<" top3_rate="<<double(b)/trials
            <<" mean_microseconds="<<micros/trials<<"\n";
  };
  out("phase",baseline_top1,baseline_top3,baseline_micros);
  out("edit_only",edit_top1,edit_top3,edit_micros);
  out("edit_triad_hotspot",triad_top1,triad_top3,triad_micros);
  cleanup();
  std::cout<<"PASS: fixed-seed 128-source 512-typo phase/edit/triad "
              "retrieval ranks and zero unsupported exact-copy\n";
  return 0;
 }catch(const std::exception& e){
  std::cerr<<"FAIL: "<<e.what()<<"\n";cleanup();return 1;
 }
}
