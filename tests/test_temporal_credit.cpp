#include "nanlm/unified_wave.hpp"
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

using namespace nanlm;
static void check(bool yes,const char* why){if(!yes)throw std::runtime_error(why);}
static float gain(const Universe& u,uint8_t head){
 float result=0;
 std::memcpy(&result,u.root().h[head].data+UnifiedWaveLanguage::LONG_GAIN_OFFSET,4);
 return result;
}
static void set_gain(Universe& u,uint8_t head,float value){
 std::memcpy(u.root().h[head].data+
             UnifiedWaveLanguage::LONG_GAIN_OFFSET,&value,4);
}
int main(){
 const auto path=(std::filesystem::temp_directory_path()/
  ("nanlm-temporal-credit-"+std::to_string(::getpid()))).string();
 try{
  double saved=0;
  const std::string context="A"+std::string(40,'b');
  {
   Universe u(path);
   UnifiedWaveLanguage m(u);
   check(!m.bptt_enabled(),"default training unexpectedly enables reverse tape");
   check(UnifiedWaveLanguage::WINDOW>=48,"long history was not extended");
   u.write('A','Z',PRECEDES,false,4.f,1);
   UnifiedWaveLanguage::State short_state{},long_state{};
   for(unsigned char b:context){
    m.observe(long_state,b);
    if(short_state.length<16)m.observe(short_state,b);
   }
   const auto short_full=m.predict(short_state);
   const auto short_ablated=m.predict(short_state,UnifiedWaveLanguage::ABLATE_LONG_CONTEXT);
   check(std::fabs(short_full.p['Z']-short_ablated.p['Z'])<1e-13,
         "short-context legacy output depends on long credit");
   check(std::fabs(gain(u,'Z'))<1e-12,"fresh long credit must be zero");
   std::string train;
   for(int i=0;i<6;++i)train+=context+"Z\n";
   std::istringstream source(train);
   const auto fit=m.train(source);
   check(fit.tokens>200&&std::isfinite(fit.ce()),"long-context training failed");
   check(m.bptt_attempted()==0&&m.bptt_accepted()==0,
         "default local/geometric credit entered reverse-mode tape");
   check(m.temporal_credit_accepted()>0,
         "existing temporal credit never accepted an update");
   check(std::fabs(gain(u,'Z'))>1e-6,
         "distant native pair evidence did not change mapped credit gain");
   UnifiedWaveLanguage::State probe{};
   for(unsigned char b:context)m.observe(probe,b);
   const auto full=m.predict(probe);
   const auto ablated=m.predict(probe,UnifiedWaveLanguage::ABLATE_LONG_CONTEXT);
   std::istringstream eval_full(context+"Z"+context+"Z");
   std::istringstream eval_without(context+"Z"+context+"Z");
   const auto exact_full=m.evaluate(eval_full);
   const auto exact_ablated=m.evaluate(
      eval_without,UnifiedWaveLanguage::ABLATE_LONG_CONTEXT);
   check(exact_full.tokens==exact_ablated.tokens&&
         std::isfinite(exact_full.ce())&&std::isfinite(exact_ablated.ce()),
         "credit-on/off paired evaluation invalid");
   check(std::fabs(exact_full.ce()-exact_ablated.ce())>1e-9,
         "existing temporal readout has no measurable effect");
   check(std::fabs(full.p['Z']-ablated.p['Z'])>1e-8,
         "long-context contribution is inert on distant evidence");
   check(u.root().h[255].data[UnifiedWaveLanguage::LONG_GAIN_OFFSET]==0,
         "reserved ingress head was mutated");
   // Geometric sign consistency: explicit PRECEDES support raises its
   // readout's relative log-odds; explicit refutation lowers them.
   // Compare SAME state/root with the SAME other experts enabled to
   // isolate only the existing geometric eligibility readout.
   u.write('A','W',PRECEDES,false,4.f,3);
   u.write('A','Y',PRECEDES,true,4.f,3);
   set_gain(u,'W',0.75f);
   set_gain(u,'Y',0.75f);
   const auto signed_full=m.predict(probe);
   const auto signed_off=m.predict(
      probe,UnifiedWaveLanguage::ABLATE_LONG_CONTEXT);
   const auto odds_delta=[&](uint8_t candidate){
    return (std::log(signed_full.p[candidate])-
            std::log(signed_full.p['X']))-
           (std::log(signed_off.p[candidate])-
            std::log(signed_off.p['X']));
   };
   check(odds_delta('W')>1e-5,
         "positive directed geometric credit lost support sign");
   check(odds_delta('Y')< -1e-5,
         "negative directed geometric credit lost refutation sign");
   check(u.root().h[255].data[UnifiedWaveLanguage::LONG_GAIN_OFFSET]==0,
         "sign-consensus probe modified network ingress");
   saved=m.predict(probe).p['Z'];
   u.sync();
  }
  {
   Universe u(path);
   UnifiedWaveLanguage m(u);
   UnifiedWaveLanguage::State probe{};
   for(unsigned char b:context)m.observe(probe,b);
   check(std::fabs(m.predict(probe).p['Z']-saved)<1e-7,
         "48-step causal credit changed after root reload");
  }
  for(const auto& ext:{"",".so3",".overlay",".overlay.tmp"})
   std::filesystem::remove(path+ext);
  std::cout<<"PASS: 48-step causal eligibility credit, bounded head-gain update, ablation, reload\n";
  return 0;
 }catch(const std::exception& e){
  std::cerr<<"FAIL: "<<e.what()<<"\n";
  for(const auto& ext:{"",".so3",".overlay",".overlay.tmp"})
   std::filesystem::remove(path+ext);
  return 1;
 }
}
