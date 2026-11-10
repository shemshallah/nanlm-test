#include "nanlm/unified_wave.hpp"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <array>
#include <vector>
#include <filesystem>
#include <iostream>
#include <iomanip>
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
  // Probe individual readout families to diagnose root reload invariance.
  const std::array<unsigned,12> ablations{{
   0u, UnifiedWaveLanguage::ABLATE_LONG_CONTEXT,
   UnifiedWaveLanguage::ABLATE_WAVE,
   UnifiedWaveLanguage::ABLATE_ATTENTION,
   UnifiedWaveLanguage::ABLATE_SPECTRAL,
   UnifiedWaveLanguage::ABLATE_CYCLE,
   UnifiedWaveLanguage::ABLATE_INVERSE_EVIDENCE,
   UnifiedWaveLanguage::ABLATE_HYPEREDGE,
   UnifiedWaveLanguage::ABLATE_BPTT_READOUT,
   UnifiedWaveLanguage::ABLATE_TRIAD,
   UnifiedWaveLanguage::ABLATE_EPISODIC,
   UnifiedWaveLanguage::ABLATE_ALL
  }};
  std::array<double,12> before_ablations{};
  std::array<double,4> before_modes{};
  std::array<float,3> before_registers{};
  size_t before_events=0,before_episodes=0;
  std::vector<uint8_t> head_bytes,pair_bytes;
  std::array<double,UnifiedWaveLanguage::N> before_wave{},before_inverse{};
  std::array<double,TriadWaveFormer::GROUPS> before_routing{};
  auto first_differences=[](const std::vector<uint8_t>& saved,
                            const void* now,size_t size,const char* label){
   const auto* bytes=reinterpret_cast<const uint8_t*>(now);
   size_t changes=0;
   for(size_t i=0;i<size;++i)if(saved.at(i)!=bytes[i]){
    if(changes<10)
     std::cerr<<"TEMPORAL_RELOAD_BYTE "<<label
              <<" offset="<<i<<" before="<<unsigned(saved[i])
              <<" after="<<unsigned(bytes[i])<<"\n";
    ++changes;
   }
   std::cerr<<"TEMPORAL_RELOAD_RAW "<<label<<" changed_bytes="<<changes<<"\n";
  };
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
   // Adding directed evidence changes the spectral geometry; rebuild
   // the same derived operator before the in-process measurement so
   // restart/reload compares identical model states.
   m.refresh_spectrum();
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
   for(size_t i=0;i<ablations.size();++i)
    before_ablations[i]=m.predict(probe,ablations[i]).p['Z'];
   before_modes=m.spectral().eigenvalues;
   before_registers={gain(u,'Z'),gain(u,'W'),gain(u,'Y')};
   before_events=u.event_count();before_episodes=m.episodic_records();
   before_wave=probe.wave;before_inverse=probe.inverse_wave;
   before_routing=probe.workspace.routing;
   const auto* heads=reinterpret_cast<const uint8_t*>(&u.root().h);
   head_bytes.assign(heads,heads+sizeof(u.root().h));
   const auto* pairs=reinterpret_cast<const uint8_t*>(&u.root().p);
   pair_bytes.assign(pairs,pairs+sizeof(u.root().p));
   u.sync();
  }
  {
   Universe u(path);
   UnifiedWaveLanguage m(u);
   UnifiedWaveLanguage::State probe{};
   for(unsigned char b:context)m.observe(probe,b);
   const double restored=m.predict(probe).p['Z'];
   std::cerr<<"TEMPORAL_RELOAD_DIAGNOSTIC saved="<<std::setprecision(17)<<saved
            <<" restored="<<restored<<" difference="<<restored-saved<<"\n";
   std::cerr<<"TEMPORAL_RELOAD_COMPONENTS"
            <<" events_before="<<before_events
            <<" events_after="<<u.event_count()
            <<" episodes_before="<<before_episodes
            <<" episodes_after="<<m.episodic_records()
            <<" gainZ_before="<<before_registers[0]
            <<" gainZ_after="<<gain(u,'Z')
            <<" gainW_before="<<before_registers[1]
            <<" gainW_after="<<gain(u,'W')
            <<" gainY_before="<<before_registers[2]
            <<" gainY_after="<<gain(u,'Y')<<"\n";
   for(size_t i=0;i<ablations.size();++i)
    std::cerr<<"TEMPORAL_RELOAD_ABLATION mask="<<ablations[i]
             <<" before="<<before_ablations[i]
             <<" after="<<m.predict(probe,ablations[i]).p['Z']<<"\n";
   first_differences(head_bytes,&u.root().h,sizeof(u.root().h),"heads");
   first_differences(pair_bytes,&u.root().p,sizeof(u.root().p),"pairs");
   double wave_diff=0,inverse_diff=0,routing_diff=0;
   size_t wave_changed=0,inverse_changed=0,routing_changed=0;
   for(size_t i=0;i<before_wave.size();++i){
    const double w=std::abs(before_wave[i]-probe.wave[i]);
    const double inv=std::abs(before_inverse[i]-probe.inverse_wave[i]);
    wave_diff=std::max(wave_diff,w);inverse_diff=std::max(inverse_diff,inv);
    wave_changed+=w>1e-14;inverse_changed+=inv>1e-14;
   }
   for(size_t i=0;i<before_routing.size();++i){
    const double difference=std::abs(before_routing[i]-probe.workspace.routing[i]);
    routing_diff=std::max(routing_diff,difference);
    routing_changed+=difference>1e-14;
   }
   std::cerr<<"TEMPORAL_RELOAD_STATE wave_max="<<wave_diff
            <<" wave_changed="<<wave_changed
            <<" inverse_max="<<inverse_diff
            <<" inverse_changed="<<inverse_changed
            <<" routing_max="<<routing_diff
            <<" routing_changed="<<routing_changed<<"\n";
   for(size_t i=0;i<before_modes.size();++i)
    std::cerr<<"TEMPORAL_RELOAD_SPECTRAL mode="<<i
             <<" before="<<before_modes[i]
             <<" after="<<m.spectral().eigenvalues[i]<<"\n";
   check(std::fabs(restored-saved)<1e-7,
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
