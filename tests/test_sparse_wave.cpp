#include "nanlm/unified_wave.hpp"
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
using namespace nanlm;
static void check(bool ok,const char* what){if(!ok)throw std::runtime_error(what);}
static double edge(Universe& u,uint8_t a,uint8_t b){
 if(a==b)return 0.0;
 const auto e=u.query(a,b,PRECEDES),inv=u.query(b,a,FOLLOWS);
 const double p=0.5*(e.support+inv.support);
 const double n=0.5*(e.refute+inv.refute);
 return std::clamp((p-n)/(1.0+p+n),-1.0,1.0)*
        std::cos(u.covariant_phase(a,b));
}
int main(){
 const auto root=(std::filesystem::temp_directory_path()/
  ("nanlm-sparse-mirror-"+std::to_string(::getpid()))).string();
 try{
  Universe u(root);
  UnifiedWaveLanguage m(u);
  check(m.active_mirror_arcs()==0,"empty graph has scattering arcs");
  u.write('a','b',PRECEDES,false,3.0f,1);
  u.write('b','c',PRECEDES,false,1.1f,2);
  u.write('d','b',PRECEDES,false,2.2f,3);
  u.write('c','a',PRECEDES,true,0.7f,4);
  u.set_phase('a','b',0.36);
  u.set_phase('d','b',-0.18);
  m.refresh_spectrum();
  check(m.active_mirror_arcs()==4,"sparse cache duplicated or omitted arcs");
  UnifiedWaveLanguage::State s{};
  for(unsigned char c:std::string("abcda"))m.observe(s,c);
  auto before=s;
  m.observe(s,'b');
  const auto& eig=m.spectral();
  for(size_t i=0;i<UnifiedWaveLanguage::N;++i){
   double f=0,inv=0,fi=1,fo=1,modal=0,mod=0;
   for(size_t j=0;j<UnifiedWaveLanguage::N;++j){
    const double e=edge(u,static_cast<uint8_t>(j),static_cast<uint8_t>(i));
    const double b=edge(u,static_cast<uint8_t>(i),static_cast<uint8_t>(j));
    f+=e*before.wave[j];inv+=b*before.inverse_wave[j];
    fi+=std::fabs(e);fo+=std::fabs(b);
   }
   for(size_t k=0;k<UnifiedWaveLanguage::MODES;++k){
    mod+=eig.vectors[k][i]*eig.vectors[k]['b']/(1.0+k);
    modal+=eig.eigenvalues[k]*eig.vectors[k][i]*eig.vectors[k][i];
   }
   const double osc=2.0*0.80*std::cos(0.53+0.1*std::tanh(modal));
   const double input=(i=='b'?1.0:0.0);
   const auto clamp=[](double x){return std::clamp(x,-3.0,3.0);};
   const double expected=clamp(osc*before.wave[i]-0.64*before.old_wave[i]+
                         0.08*f/fi+0.05*mod+input);
   const double inverse=clamp(osc*before.inverse_wave[i]-0.64*before.inverse_old_wave[i]+
                         0.08*inv/fo+0.05*mod+input);
   check(std::fabs(expected-s.wave[i])<1e-11,
         "sparse forward mirror recurrence diverged from dense oracle");
   check(std::fabs(inverse-s.inverse_wave[i])<1e-11,
         "sparse inverse mirror recurrence diverged from dense oracle");
  }
  // Observe latency diagnostic only; CI machines have variable clocks.
  const auto clock_start=std::chrono::steady_clock::now();
  for(int step=0;step<512;++step)
   m.observe(s,static_cast<uint8_t>('a'+step%4));
  const auto elapsed=std::chrono::duration_cast<std::chrono::microseconds>(
      std::chrono::steady_clock::now()-clock_start).count();
  std::cout<<"SPARSE_MIRROR_BENCHMARK steps=512 elapsed_us="<<elapsed
           <<" arcs="<<m.active_mirror_arcs()<<" root_bytes="<<ROOT_BYTES<<"\n";
  for(const auto ext:{"",".overlay",".overlay.tmp",".so3"})
   std::filesystem::remove(root+ext);
  std::cout<<"PASS: cached sparse T and T* equal dense native mirror recurrence\n";
  return 0;
 }catch(const std::exception& e){
  std::cerr<<"FAIL: "<<e.what()<<"\n";
  for(const auto ext:{"",".overlay",".overlay.tmp",".so3"})
   std::filesystem::remove(root+ext);
  return 1;
 }
}
