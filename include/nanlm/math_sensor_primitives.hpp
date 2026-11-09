#pragma once
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
namespace nanlm::math_sensor {
inline constexpr std::size_t DIM=256,HEADS=255,TRIADS=85;
inline constexpr double PI=3.1415926535897932384626433832795;
using Hypervector=std::array<std::uint64_t,4>;
using TriadMap=std::array<std::array<std::uint8_t,3>,TRIADS>;
inline std::uint64_t mix(std::uint64_t x){
 x+=0x9e3779b97f4a7c15ULL;
 x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;
 x=(x^(x>>27))*0x94d049bb133111ebULL;
 return x^(x>>31);
}
inline Hypervector basis(std::uint64_t label){
 Hypervector v{};
 const auto seed=0x4e614e4c4d485730ULL^(label*0xd6e8feb86659fd93ULL);
 for(std::size_t i=0;i<4;++i)v[i]=mix(seed+i*0x9e3779b97f4a7c15ULL);
 return v;
}
inline Hypervector bind(const Hypervector& a,const Hypervector& b){
 Hypervector v{};for(std::size_t i=0;i<4;++i)v[i]=a[i]^b[i];return v;
}
inline std::size_t distance(const Hypervector& a,const Hypervector& b){
 std::size_t n=0;for(std::size_t i=0;i<4;++i)n+=std::popcount(a[i]^b[i]);return n;
}
inline double similarity(const Hypervector& a,const Hypervector& b){
 return 1.-double(distance(a,b))/128.;
}
inline double wrap(double a){return std::atan2(std::sin(a),std::cos(a));}
inline double connection(std::uint8_t a,std::uint8_t b){
 if(a==b)return 0;
 const auto lo=std::min(a,b),hi=std::max(a,b);
 const auto key=(std::uint64_t(lo)<<8)|hi;
 const double f=double(mix(0x434f4e4e454354ULL^key)>>11)/double(1ULL<<53);
 return a<b?(2*f-1)*PI:-(2*f-1)*PI;
}
inline double curvature(std::uint8_t a,std::uint8_t b,std::uint8_t c){
 if(a==b||b==c||a==c)throw std::invalid_argument("degenerate triangle");
 return wrap(connection(a,b)+connection(b,c)+connection(c,a));
}
inline TriadMap default_map(){
 TriadMap m{};for(std::size_t g=0;g<TRIADS;++g)
  for(std::size_t k=0;k<3;++k)m[g][k]=std::uint8_t(3*g+k);
 return m;
}
} // namespace nanlm::math_sensor
