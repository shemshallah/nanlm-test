#include "nanlm/unified_wave.hpp"
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
using namespace nanlm;
static void check(bool yes,const char* why){if(!yes)throw std::runtime_error(why);}
static void set_group(Universe& u,uint8_t holder,float value){
 std::memcpy(u.root().h[holder].data+UnifiedWaveLanguage::HYPER_SHARED_OFFSET,&value,4);
}
int main(){
 const std::string root=(std::filesystem::temp_directory_path()/
  ("nanlm-adaptive-triad-"+std::to_string(::getpid()))).string();
 try{
  double target=0;
  {
   Universe u(root);
   UnifiedWaveLanguage mind(u);
   check(UnifiedWaveLanguage::GROUPS==85,"missing 85 voting groups");
   check(mind.head_triad(0)==0&&mind.head_triad(1)==0&&
         mind.head_triad(4)==1,"initial triads not initialized");
   check(u.root().h[255].data[UnifiedWaveLanguage::TRIAD_MAGIC_OFFSET]==0,
         "gateway was assigned to a cognitive triad");
   set_group(u,2,0.7f);
   set_group(u,5,-0.7f);
   u.write('x','y',PRECEDES,false,2.f,1);
   u.write('y',4,PRECEDES,false,2.f,2);
   u.write('x',4,PRECEDES,false,1.f,3);
   mind.refresh_spectrum();
   UnifiedWaveLanguage::State state{};
   mind.observe(state,'x');mind.observe(state,'y');
   const double before=mind.predict(state).p[4];
   check(mind.trial_swap_triads(1,4,state,4),
         "beneficial three-head vote swap not committed");
   check(mind.head_triad(1)==1&&mind.head_triad(4)==0,
         "swapped memberships wrong");
   // A topology edit changes the recurrent field; replay before scoring.
   state.reset();mind.observe(state,'x');mind.observe(state,'y');
   const double after=mind.predict(state).p[4];
   check(after>before+1e-7,"accepted topology swap did not improve exact prediction");
   check(!mind.trial_swap_triads(1,4,state,4),
         "worsening inverse swap should roll back");
   check(mind.head_triad(1)==1&&mind.head_triad(4)==0,
         "rejected swap damaged physical triads");
   std::array<unsigned,UnifiedWaveLanguage::GROUPS> counts{};
   for(size_t h=0;h<UnifiedWaveLanguage::N;++h)
    ++counts[mind.head_triad(static_cast<uint8_t>(h))];
   for(auto count:counts)check(count==3,"triad cardinality not conserved");
   target=after;
   u.sync();
  }
  {
   Universe u(root);
   UnifiedWaveLanguage mind(u);
   check(mind.head_triad(1)==1&&mind.head_triad(4)==0,
         "adaptive triad map did not persist");
   UnifiedWaveLanguage::State state{};
   mind.observe(state,'x');mind.observe(state,'y');
   check(std::fabs(mind.predict(state).p[4]-target)<1e-7,
         "triad voting inference changed after restart");
   check(u.root().h[255].data[UnifiedWaveLanguage::TRIAD_MAGIC_OFFSET]==0,
         "network ingress overwritten by triad map");
  }
  for(const auto& ext:{"",".so3",".overlay",".overlay.tmp"})
   std::filesystem::remove(root+ext);
  std::cout<<"PASS: 85 balanced adaptive triads, exact-loss voting swap, rollback, reload\n";
  return 0;
 }catch(const std::exception& e){
  std::cerr<<"FAIL: "<<e.what()<<"\n";
  for(const auto& ext:{"",".so3",".overlay",".overlay.tmp"})
   std::filesystem::remove(root+ext);
  return 1;
 }
}
