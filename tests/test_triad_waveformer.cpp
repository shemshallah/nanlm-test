#include "nanlm/unified_wave.hpp"
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
using namespace nanlm;
static void check(bool x,const char* why){if(!x)throw std::runtime_error(why);}
int main(){
 const std::string root=(std::filesystem::temp_directory_path()/
  ("nanlm-triad-waveformer-"+std::to_string(::getpid()))).string();
 try{
  constexpr size_t N=TriadWaveFormer::HEADS;
  check(N==255&&TriadWaveFormer::GROUPS==85&&TriadWaveFormer::FIELD==64,
        "triad/global shape mismatch");
  double reloaded=0;
  std::string generated;
  TriadWaveFormer::Parameters trained{};
  {
   Universe u(root);
   std::array<uint8_t,128> ingress{};
   std::memcpy(ingress.data(),u.root().h[255].data,ingress.size());
   UnifiedWaveLanguage mind(u);
   check(u.root().h[0].data[UnifiedWaveLanguage::TRIAD_KERNEL_MARKER_OFFSET]==
      UnifiedWaveLanguage::TRIAD_KERNEL_MAGIC,"shared kernel not persisted");
   check(std::memcmp(ingress.data(),u.root().h[255].data,128)==0,
      "network ingress modified by triad kernel");
   auto old=mind.triad_parameters();
   UnifiedWaveLanguage::State state{};
   for(unsigned char c:std::string("a sailor returned to the harbor "))
    mind.observe(state,c);
   check(state.workspace.active_count==8,"sparse routing not active");
   for(double x:state.workspace.field)check(std::isfinite(x),"nonfinite global state");
   std::array<bool,85> seen{};
   for(size_t i=0;i<state.workspace.active_count;++i){
    const size_t g=state.workspace.active[i];
    check(g<85&&!seen[g],"duplicate active triad");
    seen[g]=true;
   }
   double mass=0;
   for(double r:state.workspace.routing)mass+=r;
   check(std::fabs(mass-1)<1e-10,"routing probability not normalized");
   const auto p=mind.predict(state);
   const auto no_tri=mind.predict(state,UnifiedWaveLanguage::ABLATE_TRIAD);
   const auto no_field=mind.predict(state,UnifiedWaveLanguage::ABLATE_FIELD);
   check(std::fabs(p.p['s']-no_tri.p['s'])>1e-10,
         "triad readout disconnected");
   check(std::isfinite(no_field.p['s']),"global ablation not finite");
   // Exact log-odds sensitivity of the persisted head gain: no hidden
   // production ensemble and no change in physical mirror evidence.
   auto changed=old;
   changed.head_gain['s']+=0.025f;
   mind.set_triad_parameters(changed);
   const auto shifted=mind.predict(state);
   check(std::fabs(p.p['s']-shifted.p['s'])>1e-10,
         "persistent trainable readout inert");
   mind.set_triad_parameters(old);
   std::istringstream corpus(
      "sailor sails to harbor. the sailor arrived home. "
      "harbor lights guide sailors and ships. ");
   auto result=mind.train(corpus);
   check(result.tokens>30&&std::isfinite(result.ce()),"unified train failed");
   check(mind.triad_accepted()>0,"head gains did not learn from CE gradient");
   trained=mind.triad_parameters();
   check(trained.head_gain!=old.head_gain,"trained parameters unchanged");
   UnifiedWaveLanguage::State fresh{};
   for(unsigned char c:std::string("sailor sails to harbor"))
    mind.observe(fresh,c);
   reloaded=mind.predict(fresh).p['.'];
   generated=mind.generate("sailor ",24,47);
   check(valid_utf8(generated),"generation not valid UTF-8");
   check(std::filesystem::file_size(root)==8388608,"root size changed");
   check(std::memcmp(ingress.data(),u.root().h[255].data,128)==0,
         "network ingress modified during training");
   u.sync();
  }
  {
   Universe u(root);
   UnifiedWaveLanguage mind(u);
   check(mind.triad_parameters().head_gain==trained.head_gain,
         "triad learned gains did not survive restart");
   UnifiedWaveLanguage::State fresh{};
   for(unsigned char c:std::string("sailor sails to harbor"))
    mind.observe(fresh,c);
   check(std::fabs(reloaded-mind.predict(fresh).p['.'])<1e-11,
         "reloaded global-field prediction drift");
   check(generated==mind.generate("sailor ",24,47),
         "reloaded UTF-8 generation drift");
  }
  for(const auto& ext:{"",".so3",".overlay",".overlay.tmp"})
   std::filesystem::remove(root+ext);
  std::cout<<"PASS: one-root 85-triad sparse attention, 64D recurrent field, gradient gains, persistence and generation\n";
  return 0;
 }catch(const std::exception& e){
  std::cerr<<"FAIL: "<<e.what()<<"\n";
  for(const auto& ext:{"",".so3",".overlay",".overlay.tmp"})
   std::filesystem::remove(root+ext);
  return 1;
 }
}
