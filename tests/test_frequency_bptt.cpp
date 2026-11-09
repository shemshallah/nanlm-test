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
static void check(bool yes,const char* what){if(!yes)throw std::runtime_error(what);}
int main(){
 const std::string root=(std::filesystem::temp_directory_path()/
   ("nanlm-frequency-bptt-"+std::to_string(::getpid()))).string();
 try{
  std::array<uint8_t,128> reserved{};
  double persisted=0;
  {
   Universe u(root); UnifiedWaveLanguage mind(u);
   std::memcpy(reserved.data(),u.root().h[255].data,128);
   const std::string sample="abcdefg";
   for(size_t j=1;j<sample.size();++j)
    u.write(uint8_t(sample[j-1]),uint8_t(sample[j]),
            PRECEDES,false,1.75f,j);
   mind.refresh_spectrum();
   UnifiedWaveLanguage::State state{};
   std::deque<UnifiedWaveLanguage::State> history{state};
   std::deque<uint8_t> tokens{};
   for(unsigned char c:sample){
    mind.observe(state,c);
    tokens.push_back(c);history.push_back(state);
   }
   const uint8_t label='h';
   const auto gradient=mind.bptt_gradient(history,tokens,label);
   check(gradient.steps==7&&gradient.norm>1e-8,
         "seven-step joint input and frequency adjoint missing");
   const auto evaluate=[&](uint8_t head,double frequency){
    mind.set_recurrent_frequency_shift(head,frequency);
    UnifiedWaveLanguage::State replay=history.front();
    for(uint8_t byte:tokens)mind.observe(replay,byte);
    return -std::log(std::max(mind.predict(replay).p[label],1e-300));
   };
   size_t informative=0;
   for(const auto c:{uint8_t('a'),uint8_t('c'),uint8_t('f')}){
    const auto original=mind.recurrent_frequency_shift(c);
    const auto epsilon=0.0005;
    const double above=evaluate(c,original+epsilon);
    const double below=evaluate(c,original-epsilon);
    mind.set_recurrent_frequency_shift(c,original);
    const double numeric=(above-below)/(2*epsilon);
    const double analytic=gradient.frequency[c];
    std::cout<<"FREQUENCY_GRADCHECK head="<<int(c)
             <<" analytic="<<analytic<<" finite="<<numeric
             <<" discrepancy="<<std::fabs(analytic-numeric)<<"\n";
    check(std::fabs(analytic-numeric)<std::max(1e-7,0.02*std::fabs(analytic)),
          "frequency backprop disagrees with finite differences");
    informative+=std::fabs(analytic)>1e-6;
   }
   check(informative>=2,"frequency gradients did not flow through recurrence");
   const double initial=gradient.loss;
   check(mind.learn_bptt(state,history,tokens,label),
         "joint phase and input gradient did not improve true softmax CE");
   check(-std::log(std::max(mind.predict(state).p[label],1e-300))<initial-1e-10,
         "joint optimizer accepted an invalid update");
   persisted=mind.recurrent_frequency_shift('c');
   check(std::fabs(persisted)>1e-8,"frequency register never updated");
   check(std::memcmp(reserved.data(),u.root().h[255].data,128)==0,
         "reserved ingress register changed during frequency BPTT");
   u.sync();
  }
  {
   Universe u(root); UnifiedWaveLanguage mind(u);
   check(std::fabs(mind.recurrent_frequency_shift('c')-persisted)<1e-8,
         "frequency parameter did not persist through mapped root");
   check(std::memcmp(reserved.data(),u.root().h[255].data,128)==0,
         "network gateway not persistent");
  }
  for(const auto suffix:{"",".overlay",".overlay.tmp",".so3"})
   std::filesystem::remove(root+suffix);
  std::cout<<"PASS: seven-step differentiable standing-wave frequency gradient and persistence\n";
  return 0;
 }catch(const std::exception& ex){
  std::cerr<<"FAIL: "<<ex.what()<<"\n";
  for(const auto suffix:{"",".overlay",".overlay.tmp",".so3"})
   std::filesystem::remove(root+suffix);
  return 1;
 }
}
