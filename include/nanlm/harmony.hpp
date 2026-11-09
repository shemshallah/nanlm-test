#pragma once
#include "nanlm/phase.hpp"
#include <random>
namespace nanlm {
// Bayesian mixture of online experts. The class-program changes by evidence;
// no generated text is silently fed back as factual training data.
class Harmony {
 Universe& u_;
 bool structured_;
 static constexpr unsigned N=255,K=19;
 using Dist=std::array<double,N>;
 template<class T>T read(unsigned h,unsigned off)const{T v{};std::memcpy(&v,u_.root().h[h].data+off,sizeof v);return v;}
 template<class T>void put(unsigned h,unsigned off,T v){std::memcpy(u_.root().h[h].data+off,&v,sizeof v);}
 double count(unsigned a,unsigned b)const{
  if(a==b)return read<uint32_t>(a,24);
  return u_.query(static_cast<uint8_t>(a),static_cast<uint8_t>(b),PRECEDES).support;
 }
 std::array<Dist,K> experts(uint8_t a)const{
  if(a==255)throw std::invalid_argument("255 is the nannet gateway");
  std::array<Dist,K> e{};
  double total=0,row=0;
  for(unsigned j=0;j<N;++j){e[1][j]=double(read<uint64_t>(j,88))+.5;total+=e[1][j];row+=count(a,j);}
  const double re=read<double>(255,32),im=read<double>(255,40);
  const double mag=std::hypot(re,im);
  const uint16_t rotor=binary_angle(std::atan2(im,re));
  const double concentration=std::min(4.,mag/(1.+read<double>(255,80))*4.);
  for(unsigned j=0;j<N;++j){
   e[1][j]/=total;
   e[0][j]=(count(a,j)+8.*e[1][j])/(row+8.);
   // Learned circular translation; no assumption that bytes encode semantics.
   const auto delta=static_cast<uint16_t>((int(j)-int(a))*256);
   e[2][j]=std::exp(concentration*nan_cos(static_cast<uint16_t>(delta-rotor))/32767.);
   for(unsigned n=0;n<16;++n)
    e[n+3][j]=std::exp(.25*std::popcount(nan_bool(n,a,j)&255u));
  }
  for(unsigned k=2;k<K;++k){double z=0;for(double p:e[k])z+=p;for(auto& p:e[k])p/=z;}
  return e;
 }
 unsigned bank(uint8_t a)const{double n=0;for(unsigned j=0;j<N;++j)n+=count(a,j);return n==0?32:0;}
 std::array<double,K> weights(unsigned base=0)const{
  std::array<double,K>w{};double m=-1e300,z=0;
  for(unsigned k=0;k<(structured_?K:2);++k)m=std::max(m,read<double>(base+k,80));
  for(unsigned k=0;k<(structured_?K:2);++k){w[k]=std::exp(read<double>(base+k,80)-m);z+=w[k];}
  for(auto& p:w)p/=z;
  return w;
 }
 public:
 explicit Harmony(Universe& u,bool structured=true):u_(u),structured_(structured){
  // Free head fields; independent of teacher's vectors, gauges and shared rotor.
  const auto magic=read<uint32_t>(255,68);
  if(magic!=0&&magic!=0x48415231)throw std::runtime_error("unknown harmony state");
  if(!magic)put<uint32_t>(255,68,0x48415231);
 }
 Dist predict(uint8_t a)const{
  const auto e=experts(a);const auto w=weights(bank(a));Dist p{};
  for(unsigned j=0;j<N;++j)for(unsigned k=0;k<K;++k)p[j]+=w[k]*e[k][j];
  return p;
 }
 double loss(uint8_t a,uint8_t b)const{if(b==255)throw std::invalid_argument("gateway target");return -std::log(predict(a)[b]);}
 double teach(uint8_t a,uint8_t b){
  if(b==255)throw std::invalid_argument("gateway target");
  auto e=experts(a);const unsigned base=bank(a);auto w=weights(base);double prob=0;
  for(unsigned k=0;k<K;++k)prob+=w[k]*e[k][b];
  double best=-1e300;unsigned program=0;
  for(unsigned k=0;k<K;++k){
   double logw=read<double>(base+k,80)+std::log(e[k][b])-std::log(prob);
   put<double>(base+k,80,logw);
   if(k>=3&&logw>best){best=logw;program=k-3;}
  }
  // Actual valid NaN16 BOOL r1,r2,r3 carrier, selected by accumulated log loss.
  uint8_t code[8];nan_pack(code,program,0x00120003);
  std::memcpy(u_.root().h[255].data+56,code,8);
  auto incoming=read<uint64_t>(b,88);
  if(incoming==UINT64_MAX)throw std::overflow_error("incoming count");
  put<uint64_t>(b,88,incoming+1);
  const uint16_t delta=static_cast<uint16_t>((int(b)-int(a))*256);
  put<double>(255,32,read<double>(255,32)+nan_cos(delta)/32767.);
  put<double>(255,40,read<double>(255,40)+nan_sin(delta)/32767.);
  put<double>(255,80,read<double>(255,80)+1.);
  if(a==b){auto c=read<uint32_t>(a,24);if(c==UINT32_MAX)throw std::overflow_error("self count");put<uint32_t>(a,24,c+1);}
  else{
   u_.write(a,b,PRECEDES,false,1,static_cast<uint64_t>(read<double>(255,80)));
   // Quantized connection in existing overlay: maintains its legacy file format.
   u_.set_phase(a,b,double(delta)*TAU/65536.);
  }
  return -std::log(prob);
 }
 std::string generate(uint8_t start,size_t length,uint64_t seed)const{
  if(start==255)throw std::invalid_argument("gateway seed");
  std::mt19937_64 rng(seed);std::string result;result.reserve(length);
  for(size_t i=0;i<length;++i){auto p=predict(start);double x=double(rng()>>11)*0x1.0p-53;unsigned j=0;
   while(j<N-1&&x>=p[j]){x-=p[j];++j;}start=static_cast<uint8_t>(j);result.push_back(static_cast<char>(j));}
  return result;
 }
 WaveState wave(uint8_t a)const{
  WaveState state;auto p=predict(a);
  for(unsigned b=0;b<N;++b){
   auto z=phasor(binary_angle(a==b?0:u_.covariant_phase(a,static_cast<uint8_t>(b))));
   state.amplitude[b]=std::sqrt(p[b])*z/std::abs(z);
  }
  return state;
 }
 std::array<double,K> expert_weights(bool cold=false)const{return weights(cold?32:0);}
 std::array<uint8_t,8> program()const{std::array<uint8_t,8> p{};std::memcpy(p.data(),u_.root().h[255].data+56,8);return p;}
};
}
