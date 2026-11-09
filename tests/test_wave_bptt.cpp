#include "nanlm/unified_wave.hpp"
#include <array>
#include <cmath>
#include <cstring>
#include <deque>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
using namespace nanlm;
static void require(bool yes,const char* why){if(!yes)throw std::runtime_error(why);}
static double loss(UnifiedWaveLanguage& m,const UnifiedWaveLanguage::State& initial,
                   const std::deque<uint8_t>& tokens,uint8_t label){
 auto state=initial;
 for(uint8_t byte:tokens)m.observe(state,byte);
 return -std::log(std::max(m.predict(state).p[label],1e-300));
}
int main(){
 const std::string path=(std::filesystem::temp_directory_path()/
  ("nanlm-bptt-"+std::to_string(::getpid()))).string();
 try{
  double saved=0;
  {
   Universe u(path);
   UnifiedWaveLanguage m(u);
   require(UnifiedWaveLanguage::BPTT_STEPS>=6,"truncation below six transitions");
   require(m.recurrent_input_gain('a')==1.0,"new weights must start neutral");
   const std::string input="abcdefg";
   for(size_t i=1;i<input.size();++i)
    u.write(static_cast<uint8_t>(input[i-1]),static_cast<uint8_t>(input[i]),
            PRECEDES,false,1.5f,i);
   m.refresh_spectrum();
   UnifiedWaveLanguage::State state{};
   std::deque<UnifiedWaveLanguage::State> states{state};
   std::deque<uint8_t> symbols{};
   for(unsigned char byte:input){
    m.observe(state,byte);
    symbols.push_back(byte);states.push_back(state);
   }
   const uint8_t label='h';
   const auto grad=m.bptt_gradient(states,symbols,label);
   require(grad.steps==7&&std::isfinite(grad.norm)&&grad.norm>1e-7,
           "missing nonzero BPTT gradient");
   // A remote initial symbol receives gradient THROUGH six later
   // transitions, not through a single-token immediate update.
   require(std::fabs(grad.input['a'])>1e-7,
           "no long-range wave gradient reached first input register");
   for(const uint8_t head:{uint8_t('a'),uint8_t('c'),uint8_t('f')}){
    const double old=m.recurrent_input_gain(head),eps=0.002;
    m.set_recurrent_input_gain(head,old+eps);
    const double plus=loss(m,states.front(),symbols,label);
    m.set_recurrent_input_gain(head,old-eps);
    const double minus=loss(m,states.front(),symbols,label);
    m.set_recurrent_input_gain(head,old);
    const double finite=(plus-minus)/(2*eps);
    const double analytical=grad.input[head];
    const double scale=std::max({1.0,std::fabs(finite),std::fabs(analytical)});
    std::cout<<"BPTT_GRADIENT head="<<int(head)<<" analytical="<<analytical
             <<" finite_difference="<<finite<<" error="<<std::fabs(finite-analytical)<<"\n";
    require(std::fabs(finite-analytical)/scale<0.004,
            "reverse-mode gradient failed finite difference check");
   }
   const double prior=grad.loss;
   require(m.learn_bptt(state,states,symbols,label),
           "exact-loss BPTT step failed to accept on causal seven-step tape");
   require(m.bptt_accepted()==1&&m.bptt_attempted()==1,
           "native BPTT counters wrong");
   const auto after=-std::log(std::max(m.predict(state).p[label],1e-300));
   require(after+1e-9<prior,"accepted BPTT failed true softmax objective");
   require(std::fabs(m.recurrent_input_gain('a')-1.0)>1e-7,
           "remote input weight not trained");
   require(u.root().h[255].data[UnifiedWaveLanguage::BPTT_MAGIC_OFFSET]==0,
           "reserved head 255 acquired BPTT parameters");
   saved=m.recurrent_input_gain('a');
   u.sync();
  }
  {
   Universe u(path);
   UnifiedWaveLanguage m(u);
   require(std::fabs(m.recurrent_input_gain('a')-saved)<1e-7,
           "recurrent gain not persisted across root reboot");
   require(u.root().h[255].data[UnifiedWaveLanguage::BPTT_MAGIC_OFFSET]==0,
           "gateway mutated after reload");
  }
  for(const auto ext:{"",".so3",".overlay",".overlay.tmp"})
   std::filesystem::remove(path+ext);
  std::cout<<"PASS: 7-step reverse-mode T/T* wave BPTT finite-difference, acceptance and reload\n";
  return 0;
 }catch(const std::exception& e){
  std::cerr<<"FAIL: "<<e.what()<<"\n";
  for(const auto ext:{"",".so3",".overlay",".overlay.tmp"})
   std::filesystem::remove(path+ext);
  return 1;
 }
}
