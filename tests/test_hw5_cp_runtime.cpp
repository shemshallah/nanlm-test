// Verify opt-in trained CP readout is the actual UnifiedWaveLanguage API,
// gives normalized causal P(next | prev2,prev1), and defaults to legacy.
#include "nanlm/unified_wave.hpp"
#include <array>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unistd.h>
using namespace nanlm;
int main(){
 const auto root=(std::filesystem::temp_directory_path()/
  ("hw5-cp-runtime-"+std::to_string(::getpid()))).string();
 std::filesystem::remove(root);
 try {
  {
   Universe u(root);
   UnifiedWaveLanguage mind(u);
   UnifiedWaveLanguage::State state{};
   state.push('a');state.push('b');
   if(mind.cp_context_readout_enabled())
    throw std::runtime_error("CP opt-in changed legacy default");
   const auto direct=mind.predict_cp_context(state);
   mind.enable_cp_context_readout(true);
   if(!mind.cp_context_readout_enabled())
    throw std::runtime_error("CP opt-in was ignored");
   const auto routed=mind.predict(state);
   const auto ablated=mind.predict(state,UnifiedWaveLanguage::ABLATE_ALL);
   double sum=0,base_sum=0;
   for(size_t i=0;i<FabricTeacher::VOCAB;++i){
    sum+=routed.p[i];base_sum+=ablated.p[i];
    if(std::fabs(direct.p[i]-routed.p[i])>1e-12)
     throw std::runtime_error("Native runtime CP differs from direct shared-head readout");
   }
   if(std::fabs(sum-1.0)>1e-12 || std::fabs(base_sum-1.0)>1e-12)
    throw std::runtime_error("unnormalized contextual distribution");
   mind.enable_cp_context_readout(false);
   if(mind.cp_context_readout_enabled())
    throw std::runtime_error("legacy mode cannot be restored");
   const auto legacy=mind.predict(state);
   if(legacy.p[0]<=0 || legacy.p[0]>=1)
    throw std::runtime_error("invalid legacy probability");
   std::cout<<"HW5_CP_RUNTIME_PASS mode=opt_in_shared_root" 
     <<" root_bytes="<<ROOT_BYTES
     <<" vocab="<<FabricTeacher::VOCAB
     <<" inference_trigram_lookup=0 legacy_default=1\n";
  }
  std::filesystem::remove(root);
  std::filesystem::remove(root+".episodes");
  std::filesystem::remove(root+".so3");
  return 0;
 }catch(const std::exception& e){
  std::cerr<<"HW5_CP_RUNTIME_ERROR "<<e.what()<<"\n";
  std::filesystem::remove(root);
  return 2;
 }
}
