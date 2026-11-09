#include "nanlm/unified_wave.hpp"
#include <cmath>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

using namespace nanlm;
int main(){
 const auto root=(std::filesystem::temp_directory_path()/
  ("nanlm-unified-"+std::to_string(::getpid()))).string();
 const std::string train=
  "wind moves through the pines.\n"
  "sun moves across the clouds.\n"
  "wind moves through the fields.\n";
 const std::string unseen="clouds move across the mountain.\n";
 try{
  double final_ce=0;
  std::string sample;
  {
   Universe u(root);
   UnifiedWaveLanguage mind(u);
   const auto pre=mind.spectral();
   if(pre.edge_mass!=0||pre.eigenvalue!=0||pre.residual!=0)
    throw std::runtime_error("blank spectral graph incorrectly nonzero");
   // Every head must contribute a causal activation, but 255 is reserved.
   UnifiedWaveLanguage::State state{};
   mind.observe(state,static_cast<uint8_t>('w'));
   if(state.length!=1||state.wave['w']<=0||state.wave[254]!=0||
      state.inverse_wave['w']<=0)
    throw std::runtime_error("standing wave excitation broken");
   const auto first=mind.predict(state);
   double total=0;
   for(double p:first.p)total+=p;
   if(std::fabs(total-1)>1e-9)
    throw std::runtime_error("unified probability normalization broken");
   std::istringstream data(train);
   const auto fit=mind.train(data);
   if(fit.tokens!=train.size()-1||!std::isfinite(fit.ce()))
    throw std::runtime_error("unified training failed");
   const auto learned=mind.spectral();
   if(learned.edge_mass<=0||!std::isfinite(learned.eigenvalue)||
      !std::isfinite(learned.residual))
    throw std::runtime_error("mirror eigenmode was not established");
   for(size_t mode=0;mode<UnifiedWaveLanguage::MODES;++mode){
    if(!std::isfinite(learned.eigenvalues[mode])||
       !std::isfinite(learned.residuals[mode]))
     throw std::runtime_error("spectral modes invalid");
   }
   if(u.root().h['w'].data[UnifiedWaveLanguage::HEAD_GAIN_MAGIC_OFFSET]!=
      UnifiedWaveLanguage::HEAD_GAIN_MAGIC)
    throw std::runtime_error("palindrome expert gates not initialized");
   {
    UnifiedWaveLanguage::State temporal{};
    mind.observe(temporal,static_cast<uint8_t>('w'));
    mind.observe(temporal,static_cast<uint8_t>('i'));
    double diff=0;
    for(size_t i=0;i<UnifiedWaveLanguage::N;++i)
     diff+=std::fabs(temporal.wave[i]-temporal.inverse_wave[i]);
    if(!(diff>0))throw std::runtime_error("forward and inverse waves did not separate");
   }
   if(u.query('w','i',PRECEDES).support<=0||
      u.query('i','w',FOLLOWS).support<=0)
    throw std::runtime_error("inverse temporal pair was not written");
   std::istringstream held(unseen);
   const auto evaluated=mind.evaluate(held);
   final_ce=evaluated.ce();
   if(evaluated.tokens!=unseen.size()-1||!std::isfinite(final_ce)||
      final_ce<=0)
    throw std::runtime_error("unified heldout score invalid");
   sample=mind.generate("wind ",80,19);
   if(sample.size()!=80||!valid_utf8(sample)||
      sample!=mind.generate("wind ",80,19))
    throw std::runtime_error("unified generation not deterministic valid UTF-8");
   if(u.root().h[255].data[3]!=0)
    throw std::runtime_error("gateway head modified as cognitive head");
   u.sync();
  }
  {
   Universe u(root);
   UnifiedWaveLanguage mind(u);
   const auto restored=mind.generate("wind ",80,19);
   if(restored!=sample)
    throw std::runtime_error("unified generation drift after root reload");
   std::istringstream held(unseen);
   if(std::fabs(mind.evaluate(held).ce()-final_ce)>1e-7)
    throw std::runtime_error("unified predictive loss drift after reload");
  }
  for(const std::string suffix:{"",".so3",".overlay",".overlay.tmp"})
   std::filesystem::remove(root+suffix);
  std::cout<<"PASS: one 255-head universe, mirror inverse temporal wave, spectral eigenmode, attention, valid UTF-8 generation, restart\n";
  return 0;
 }catch(const std::exception& e){
  std::cerr<<"FAIL: "<<e.what()<<"\n";
  for(const std::string suffix:{"",".so3",".overlay",".overlay.tmp"})
   std::filesystem::remove(root+suffix);
  return 1;
 }
}
