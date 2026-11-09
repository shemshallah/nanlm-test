#include "nanlm/unified_wave.hpp"
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
using namespace nanlm;
static void require(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
static float fourth_gain(const Universe& u,uint8_t h){
 const size_t offset=h==0?UnifiedWaveLanguage::FOURTH_HEAD0_GAIN_OFFSET:
                    UnifiedWaveLanguage::FOURTH_GAIN_OFFSET;
 float gain=0;std::memcpy(&gain,u.root().h[h].data+offset,sizeof(gain));
 return gain;
}
int main(){
 const auto file=(std::filesystem::temp_directory_path()/
  ("nanlm-rank-four-"+std::to_string(::getpid()))).string();
 try{
  double stored=0;
  {
   Universe u(file);
   UnifiedWaveLanguage m(u);
   u.write('a','b',PRECEDES,false,2.f,1);
   u.write('b','c',PRECEDES,false,2.f,2);
   u.write('c','d',PRECEDES,false,2.f,3);
   u.write('a','d',PRECEDES,false,1.f,4);
   m.refresh_spectrum();
   UnifiedWaveLanguage::State short_state{},long_state{};
   m.observe(short_state,'a');m.observe(short_state,'b');
   require(std::fabs(m.predict(short_state).p['d']-
          m.predict(short_state,UnifiedWaveLanguage::ABLATE_FOURTH_ORDER).p['d'])<1e-12,
          "rank-four readout acts before third context byte");
   for(unsigned char c:std::string("abc"))m.observe(long_state,c);
   require(u.hyperedge_count()==0,"fourth-order feature allocated sparse triples");
   require(fourth_gain(u,'d')==0,"new rank-four gain not zero initialized");
   const auto initial=m.predict(long_state).p['d'];
   std::istringstream train("abcd\nabcd\nabcd\nabcd\nabcd\n");
   m.train(train);
   require(std::fabs(fourth_gain(u,'d'))>1e-6,
           "rank-four gradient did not learn its mapped head coefficient");
   long_state.reset();
   for(unsigned char c:std::string("abc"))m.observe(long_state,c);
   const auto full=m.predict(long_state);
   const auto no_fourth=m.predict(long_state,UnifiedWaveLanguage::ABLATE_FOURTH_ORDER);
   require(std::fabs(full.p['d']-no_fourth.p['d'])>1e-6,
           "fourth-order ablation remains inert after training");
   require(full.p['d']>0&&std::isfinite(initial),"nonfinite conditional prediction");
   require(u.root().h[255].data[UnifiedWaveLanguage::FOURTH_GAIN_OFFSET]==0,
           "reserved ingress mutated as fourth-order head");
   stored=full.p['d'];
   u.sync();
  }
  {
   Universe u(file); UnifiedWaveLanguage m(u);
   UnifiedWaveLanguage::State s{};
   for(unsigned char c:std::string("abc"))m.observe(s,c);
   require(std::fabs(m.predict(s).p['d']-stored)<1e-7,
           "fourth-order mapped state changed across restart");
  }
  for(const auto& suffix:{"",".so3",".overlay",".overlay.tmp"})
   std::filesystem::remove(file+suffix);
  std::cout<<"PASS: rank-four factorized temporal coupling, native gradient, ablation, reload\n";
  return 0;
 }catch(const std::exception& e){
  std::cerr<<"FAIL: "<<e.what()<<"\n";
  for(const auto& suffix:{"",".so3",".overlay",".overlay.tmp"})
   std::filesystem::remove(file+suffix);
  return 1;
 }
}
