#include "nanlm/unified_wave.hpp"
#include <array>
#include <cmath>
#include <deque>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
using namespace nanlm;
static void check(bool ok,const char* msg){if(!ok)throw std::runtime_error(msg);}
int main(){
 const std::string root=(std::filesystem::temp_directory_path()/
  ("nanlm-sequence-bptt-"+std::to_string(::getpid()))).string();
 try{
  {
   Universe u(root);
   UnifiedWaveLanguage model(u);
   const std::string text="abcdefg";
   for(size_t i=1;i<text.size();++i)
    u.write(uint8_t(text[i-1]),uint8_t(text[i]),PRECEDES,false,1.5f,i);
   u.write('a','g',PRECEDES,true,0.5f,9);
   model.refresh_spectrum();
   UnifiedWaveLanguage::State state{};
   std::deque<UnifiedWaveLanguage::State> states{state};
   std::deque<uint8_t> inputs;
   for(const unsigned char x:text){
    model.observe(state,x);
    inputs.push_back(x);
    states.push_back(state);
   }
   const auto label=uint8_t('h');
   const auto grad=model.bptt_gradient(states,inputs,label,true);
   check(grad.steps==7&&grad.norm>1e-8&&std::isfinite(grad.loss),
         "sequence loss and reverse recurrence failed");
   for(const uint8_t head:{uint8_t('a'),uint8_t('c'),uint8_t('f')}){
    const double original=model.recurrent_input_gain(head);
    constexpr double epsilon=0.002;
    const auto score=[&](double value){
     model.set_recurrent_input_gain(head,value);
     std::deque<UnifiedWaveLanguage::State> replay{states.front()};
     for(uint8_t input:inputs){
      auto next=replay.back();
      model.observe(next,input);
      replay.push_back(next);
     }
     return model.bptt_window_loss(replay,inputs,label,true);
    };
    const double plus=score(original+epsilon);
    const double minus=score(original-epsilon);
    model.set_recurrent_input_gain(head,original);
    const double numerical=(plus-minus)/(2*epsilon);
    const double analytic=grad.input[head];
    const double error=std::fabs(numerical-analytic);
    std::cout<<"SEQUENCE_GRADCHECK head="<<unsigned(head)
             <<" analytic="<<analytic<<" finite="<<numerical
             <<" error="<<error<<"\n";
    check(error<0.003*std::max({1.0,std::fabs(numerical),std::fabs(analytic)}),
          "multi-target recurrence gradient disagrees with finite difference");
   }
   const double old=grad.loss;
   check(model.learn_bptt(state,states,inputs,label),
         "sequence objective failed to accept improving update");
   check(model.bptt_window_loss(states,inputs,label,true)<old-1e-10,
         "sequence learner committed non-improving objective");
   check(model.bptt_attempted()==1&&model.bptt_accepted()==1,
         "BPTT counters not updated");
   u.sync();
  }
  {
   Universe u(root);
   UnifiedWaveLanguage m(u);
   check(m.recurrent_input_gain('a')>=0.6,
         "sequence learned weight was not persisted");
  }
  for(const auto& suffix:{"",".overlay",".overlay.tmp",".so3"})
   std::filesystem::remove(root+suffix);
  std::cout<<"PASS: eight-position multi-target sequence BPTT finite-difference and acceptance\n";
  return 0;
 }catch(const std::exception& e){
  std::cerr<<"FAIL: "<<e.what()<<"\n";
  for(const auto& suffix:{"",".overlay",".overlay.tmp",".so3"})
   std::filesystem::remove(root+suffix);
  return 1;
 }
}
