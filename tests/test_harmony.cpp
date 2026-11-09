#include "nanlm/harmony.hpp"
#include "nanlm/wordnet.hpp"
#include <iostream>
#include <filesystem>
using namespace nanlm;
static void check(bool ok,const char* msg){if(!ok)throw std::runtime_error(msg);}
int main(){
 double err=0;
 for(unsigned p=0;p<65536;++p){err=std::max(err,std::abs(nan_sin(static_cast<uint16_t>(p))/32767.-std::sin(TAU*p/65536.)));}
 check(err<3.5e-5,"Q15 total error");
 unsigned cordic_error=0;
 for(unsigned p=0;p<65536;++p){
  auto q=nan_atan2(int32_t(std::llround(1000000*std::sin(TAU*p/65536.))),int32_t(std::llround(1000000*std::cos(TAU*p/65536.))));
  int d=int(q)-int(p);if(d>32768)d-=65536;if(d< -32768)d+=65536;
  cordic_error=std::max(cordic_error,unsigned(std::abs(d)));
 }
 check(cordic_error<=4,"CORDIC angle error");
 check(nan_atan2(INT32_MIN,INT32_MIN)>32768,"CORDIC extreme inputs");
 for(unsigned n=0;n<16;++n){
  for(unsigned a=0;a<2;++a)for(unsigned b=0;b<2;++b)check((nan_bool(n,a,b)&1)==((n>>(2*a+b))&1),"truth table");
  check(nan_bool(n,0xcccccccc,0xaaaaaaaa)==n*0x11111111u,"Boolean translation");
  uint8_t bytes[8];nan_pack(bytes,n,0x12345678);uint32_t payload=0;
  check(nan_unpack(bytes,&payload)==int(n)&&payload==0x12345678,"carrier roundtrip");
  std::reverse(bytes,bytes+8);check(nan_unpack(bytes,&payload)==int(((n&3)<<2)|(n>>2))&&payload==0x78563412,"mirror carrier");
  bytes[0]=0;check(nan_unpack(bytes,&payload)==-1,"invalid head rejected");
 }
 HermitianView h;
 for(unsigned a=0;a<256;++a)for(unsigned b=a+1;b<256;++b){h.set(a,b,static_cast<uint16_t>(a*1009+b*31),50000);check(std::abs(h.get(a,b)-std::conj(h.get(b,a)))<1e-12,"Hermitian mirror");}
 bool rejected=false;try{h.set(0,0,1,10);}catch(const std::invalid_argument&){rejected=true;}check(rejected,"real diagonal");
 std::array<std::complex<double>,256> x{},original{};
 for(unsigned i=0;i<256;++i)x[i]=original[i]={std::sin(i*.13),std::cos(i*.37)};
 row_fft(x);double fft_err=0;
 for(unsigned k=0;k<256;++k){std::complex<double> ref{};for(unsigned j=0;j<256;++j)ref+=original[j]*std::polar(1.,-TAU*j*k/256.);fft_err=std::max(fft_err,std::abs(x[k]-ref));}
 check(fft_err<.015,"FFT versus DFT");row_fft(x,true);
 for(unsigned i=0;i<256;++i)check(std::abs(x[i]-original[i])<.0003,"FFT roundtrip");
 const std::string root="/tmp/nan-harmony-"+std::to_string(getpid());
 std::string generated;std::array<double,255> prediction{};
 {
  Universe u(root);Harmony m(u);const double before=m.loss('a','b');
  for(unsigned i=0;i<500;++i){m.teach('a','b');m.teach('b','a');}
  check(m.loss('a','b')<before*.1,"learning improves prediction");
  generated=m.generate('a',128,11);prediction=m.predict('a');
  check(generated==m.generate('a',128,11),"deterministic generation");
  for(unsigned char b:generated)check(b!=255,"gateway excluded");
  double sum=0;for(double p:prediction){check(p>0&&std::isfinite(p),"positive probabilities");sum+=p;}check(std::abs(sum-1)<1e-12,"normalized");
  auto wave=m.wave('a');double trace=0,purity=0;
  for(unsigned a=0;a<256;++a){trace+=wave.born(a);for(unsigned b=0;b<256;++b)purity+=std::norm(wave.density(a,b));}
  check(std::abs(trace-1)<1e-12&&std::abs(purity-1)<1e-10,"density trace and pure-state purity");
  check(wave.born(255)==0,"gateway has no Born mass");
  for(unsigned a=0;a<255;++a)check(std::abs(wave.born(a)-prediction[a])<1e-12,"Born probability matches learner");
  auto code=m.program();uint32_t payload;check(nan_unpack(code.data(),&payload)>=0&&payload==0x00120003,"self-modified executable carrier");
  WordNetIndex w;w.import_noun_data("tests/fixtures/data.noun",u);w.save(root+".wordnet.tsv");
  auto packet=u.nannet_packet(4,0,address(1,2),1);check(packet.hw==4&&packet.slot==0,"WordNet route");
  u.sync();
 }
 {
  Universe u(root);Harmony m(u);check(prediction==m.predict('a'),"restart probabilities");check(generated==m.generate('a',128,11),"restart generation");
  WordNetIndex w;w.load(root+".wordnet.tsv");check(w.concepts()==5,"exact WordNet preserved");
 }
 for(const auto& suffix:{"",".overlay",".so3",".wordnet.tsv"})std::filesystem::remove(root+suffix);
 std::cout<<"PASS harmony: all phases, 16 truth tables, carriers, Hermitian, FFT, learning, generation, restart, WordNet; trig_error="<<err<<" fft_error="<<fft_err<<"\n";
}
