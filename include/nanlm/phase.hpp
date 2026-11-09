#pragma once
#include "nanlm/nan16.h"
#include "nanlm/universe.hpp"
#include <complex>
namespace nanlm {
inline uint16_t binary_angle(double radians){
 if(!std::isfinite(radians))throw std::invalid_argument("nonfinite phase");
 return static_cast<uint16_t>(static_cast<int32_t>(std::llround(std::remainder(radians,TAU)*65536/TAU)));
}
inline std::complex<double> phasor(uint16_t p){return {nan_cos(p)/32767.,nan_sin(p)/32767.};}
struct HermitianView {
 std::array<uint16_t,PAIRS> phase{},weight{};
 std::array<int16_t,HEADS> diagonal{};
 void set(uint8_t a,uint8_t b,uint16_t p,uint16_t w){
  if(a==b){
   if((p!=0&&p!=0x8000)||w>32767)throw std::invalid_argument("diagonal must be real Q15");
   diagonal[a]=static_cast<int16_t>(p?-int(w):int(w));return;
  }
  const auto k=pair_index(a,b);phase[k]=a<b?p:static_cast<uint16_t>(-p);weight[k]=w;
 }
 std::complex<double> get(uint8_t a,uint8_t b)const{
  if(a==b)return {diagonal[a]/32767.,0};
  const auto k=pair_index(a,b);
  return (weight[k]/65535.)*phasor(a<b?phase[k]:static_cast<uint16_t>(-phase[k]));
 }
};
// A pure density operator is PSD by construction: rho=psi psi*, trace=1.
// Generic HermitianView is NOT necessarily PSD and is not called a density matrix.
struct WaveState {
 std::array<std::complex<double>,256> amplitude{};
 std::complex<double> density(unsigned a,unsigned b)const{return amplitude.at(a)*std::conj(amplitude.at(b));}
 double born(unsigned a)const{return std::norm(amplitude.at(a));}
 double interference(unsigned a,unsigned b)const{return std::norm(amplitude.at(a)+amplitude.at(b));}
};
// Natural order, or existing rev8 input. Phase-only storage cannot hold butterflies.
inline void row_fft(std::array<std::complex<double>,256>& row,bool inverse=false,bool bit_reversed=false){
 if(!bit_reversed)for(unsigned i=0;i<256;++i){unsigned j=reverse8(static_cast<uint8_t>(i));if(i<j)std::swap(row[i],row[j]);}
 for(unsigned m=1;m<256;m*=2)for(unsigned j=0;j<m;++j){
  const auto p=static_cast<uint16_t>((inverse?1:-1)*static_cast<int>(j*(0x8000u/m)));
  const auto w=phasor(p);
  for(unsigned k=j;k<256;k+=2*m){auto a=row[k],b=w*row[k+m];row[k]=a+b;row[k+m]=a-b;}
 }
 if(inverse)for(auto& z:row)z/=256.;
}
}
