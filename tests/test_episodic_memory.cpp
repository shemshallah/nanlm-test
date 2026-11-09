#include "nanlm/unified_wave.hpp"
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>

using namespace nanlm;
static void require(bool good,const char* reason){
 if(!good)throw std::runtime_error(reason);
}
int main(){
 const std::string root=(std::filesystem::temp_directory_path()/
  ("nanlm-exact-memory-"+std::to_string(::getpid()))).string();
 const std::string cue="Beacon archive code: ";
 // Longer than 64 bytes: continuation MUST NOT depend on the cue
 // remaining in the 64-byte attention window.
 const std::string value="JUPITER-42-"+
   std::string(196,'Q')+"-THIRD-NEBULA";
 const std::string long_gap(1200,'x');
 const std::string source="trusted-test-fixture-v1";
 auto cleanup=[&](){
  for(const char* suffix:{"",".episodes",".episodes.tmp",
                                 ".so3",".overlay",".overlay.tmp"}){
   std::error_code ec;std::filesystem::remove(root+suffix,ec);
  }
 };
 try{
  double persisted=0;
  std::array<uint8_t,128> ingress{};
  {
   Universe u(root);
   std::memcpy(ingress.data(),u.root().h[255].data,128);
   UnifiedWaveLanguage mind(u);
   require(mind.episodic_records()==0,"blank checkpoint has records");
   mind.memorize(cue,value,source);
   require(mind.episodic_records()==1,"record not inserted");
   const auto* record=mind.exact_recall(cue);
   require(record&&record->value==value&&record->source==source,
           "exact source attribution lost");
   require(std::filesystem::file_size(root)==ROOT_BYTES,
           "memory expanded canonical 8 MiB root");
   const auto checkpoint_bytes=std::filesystem::file_size(root+".episodes");
   {
    UnifiedWaveLanguage::State state{};
    for(unsigned char c:long_gap+cue)mind.observe(state,c);
    require(state.length==UnifiedWaveLanguage::WINDOW,
            "long gap did not exceed attention window");
    require(state.recall_record!=ExactEpisodicMemory::NONE,
            "cue did not activate exact retrieval");
    const auto full=mind.predict(state);
    const auto ablated=mind.predict(
      state,UnifiedWaveLanguage::ABLATE_EPISODIC);
    require(full.best==static_cast<uint8_t>(value[0])&&
            full.p[full.best]>0.99999,"retrieval did not control copy");
    require(ablated.p[static_cast<uint8_t>(value[0])]<0.999,
            "no-memory ablation silently used exact value");
    for(size_t i=0;i<value.size();++i){
     const auto next=mind.predict(state);
     require(next.best==static_cast<uint8_t>(value[i]),
             "copy did not survive 64-byte window truncation");
     mind.observe(state,static_cast<uint8_t>(value[i]));
     if(i==100)
      require(state.recall_record!=ExactEpisodicMemory::NONE,
              "retrieval did not maintain multi-byte stream cursor");
    }
    require(state.recall_record==ExactEpisodicMemory::NONE,
            "completed episode cursor did not retire");
   }
   require(mind.generate(long_gap+cue,value.size(),11,1.,1.)==value,
           "exact generation was randomized or truncated");
   require(std::filesystem::file_size(root+".episodes")==checkpoint_bytes,
           "generation mutated episodic checkpoint");
   const auto scale=mind.wave_memory_scale();
   require(std::fabs(scale-1.0)<1e-12,"legacy wave gate changed");
   UnifiedWaveLanguage::State unrelated{};
   for(unsigned char ch:std::string("standing wave probe "))
    mind.observe(unrelated,ch);
   mind.set_wave_memory_scale(0.);
   require(mind.wave_memory_scale()==0.,"wave gate zero cannot persist");
   const auto zero=mind.predict(unrelated);
   mind.set_wave_memory_scale(2.);
   const auto doubled=mind.predict(unrelated);
   require(std::fabs(doubled.p['t']-zero.p['t'])>1e-12,
           "direct wave readout remains obligatory or inert");
   // The memory engine still performs exact retrieval at zero wave gain.
   mind.set_wave_memory_scale(0.);
   require(mind.generate(long_gap+cue,value.size(),19)==value,
           "wave readout cannot be disabled during memory recall");
   bool rejected=false;
   try{mind.memorize("bad","Z",source);}
   catch(const std::invalid_argument&){rejected=true;}
   require(rejected&&mind.episodic_records()==1,
           "invalid cues changed committed memory");
   persisted=mind.wave_memory_scale();
   u.sync();
   require(std::memcmp(ingress.data(),u.root().h[255].data,128)==0,
           "network ingress overwritten by episodic memory");
  }
  {
   Universe u(root);
   UnifiedWaveLanguage mind(u);
   require(mind.episodic_records()==1&&mind.exact_recall(cue)!=nullptr,
           "memory failed checkpoint reopen");
   require(mind.wave_memory_scale()==persisted,"wave gate not persisted");
   require(mind.generate(long_gap+cue,value.size(),997)==value,
           "reopened episodic memory lost exact fact");
   // TSV imports are source-tagged and checkpted in one transaction.
   std::istringstream pairs("Mission label: \tAurora\n"
                            "Mission ID: \tPX-8837\n");
   require(mind.memorize_tsv(pairs,"unit-tsv-source")==2,
           "TSV episode import failed");
   require(mind.exact_recall("Mission ID: ")->value=="PX-8837",
           "TSV memory exact content mismatch");
   mind.memorize("Mission ID: ","PX-9999","independent-source");
   require(mind.contested_episodic("Mission ID: ")&&
           mind.exact_recall("Mission ID: ")==nullptr,
           "independent contradictory claim was overwritten");
   UnifiedWaveLanguage::State contested{};
   for(unsigned char c:std::string("Mission ID: "))mind.observe(contested,c);
   require(contested.recall_record==ExactEpisodicMemory::NONE,
           "contested fact was incorrectly forced into automatic recall");
   const auto normal=mind.predict(contested);
   const auto no_memory=mind.predict(
      contested,UnifiedWaveLanguage::ABLATE_EPISODIC);
   for(size_t i=0;i<UnifiedWaveLanguage::N;++i)
    require(std::fabs(normal.p[i]-no_memory.p[i])<1e-12,
            "contested memory changed predictive distribution");

  }
  {
   std::fstream damage(root+".episodes",std::ios::binary|
                        std::ios::in|std::ios::out);
   require(bool(damage),"cannot open checkpoint for integrity probe");
   damage.seekg(55);
   const int original=damage.get();
   require(original!=std::char_traits<char>::eof(),
           "checkpoint too short for corruption test");
   damage.seekp(55);
   damage.put(static_cast<char>(original^1));
   damage.flush();
  }
  bool corrupt_rejected=false;
  try{ExactEpisodicMemory bad(root+".episodes");(void)bad;}
  catch(const std::exception&){corrupt_rejected=true;}
  require(corrupt_rejected,"corrupt episodic checkpoint was accepted");
  cleanup();
  std::cout<<"PASS: exact source-grounded episodic recall across 1200-byte gaps "
              "and 200-byte values, restart, integrity, zero-wave gate, "
              "no memory/ingress contamination\n";
  return 0;
 }catch(const std::exception& e){
  std::cerr<<"FAIL: "<<e.what()<<"\n";cleanup();return 1;
 }
}
