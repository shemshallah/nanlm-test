#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string_view>

namespace nanlm {

// Read-only approximate cue locator over the *current* 85 NaNLM triads.
// Never supplies source fact values, never starts exact-copy cursors.
// This uses the actual head->triad assignment provided by UnifiedWaveLanguage;
// dynamic reassignment does not invalidate a second persistent index.
struct TriadCueRouter {
 static constexpr size_t GROUPS=85, HEADS=255, HOT=8, MAX_CUE=64;
 using Mapping=std::array<uint8_t,HEADS>;
 using Sketch=std::array<double,GROUPS>;
 static Sketch encode(std::string_view bytes,const Mapping& group_of){
  Sketch z{};
  for(size_t i=0;i<bytes.size();++i){
   const uint8_t head=static_cast<uint8_t>(bytes[i]);
   if(head>=HEADS)continue; // UTF-8 0xff excluded from valid cues.
   const uint8_t group=group_of[head];
   if(group>=GROUPS)throw std::invalid_argument("invalid live triad map");
   const double lag=double(bytes.size()-i-1);
   const double wave=std::exp(-0.035*lag)*
      (std::cos(0.23*lag)+0.35*std::sin(0.41*lag));
   z[group]+=wave;
  }
  double length=0;
  for(double x:z)length+=x*x;
  if(length>1e-20){
   length=std::sqrt(length);
   for(double& x:z)x/=length;
  }
  return z;
 }
 static double cosine(const Sketch& a,const Sketch& b){
  double sum=0;for(size_t i=0;i<GROUPS;++i)sum+=a[i]*b[i];
  return std::clamp(sum,-1.0,1.0);
 }
 static size_t hotspots(const Sketch& a,const Sketch& b){
  std::array<size_t,GROUPS> indices_a{},indices_b{};
  std::iota(indices_a.begin(),indices_a.end(),size_t(0));
  std::iota(indices_b.begin(),indices_b.end(),size_t(0));
  const auto by_a=[&](size_t i,size_t j){
   const auto ai=std::fabs(a[i]),aj=std::fabs(a[j]);
   return ai==aj?i<j:ai>aj;
  };
  const auto by_b=[&](size_t i,size_t j){
   const auto ai=std::fabs(b[i]),aj=std::fabs(b[j]);
   return ai==aj?i<j:ai>aj;
  };
  std::partial_sort(indices_a.begin(),indices_a.begin()+HOT,
                    indices_a.end(),by_a);
  std::partial_sort(indices_b.begin(),indices_b.begin()+HOT,
                    indices_b.end(),by_b);
  size_t shared=0;
  for(size_t i=0;i<HOT;++i)
   for(size_t j=0;j<HOT;++j)
    shared+=indices_a[i]==indices_b[j]?1u:0u;
  return shared;
 }
 // Byte-wise Levenshtein for UTF-8 cue candidates; at most 64 characters
 // in the byte representation. No semantic/factual verification is inferred.
 static size_t edit_distance(std::string_view a,std::string_view b,
                             size_t limit=3){
  if(a.size()>MAX_CUE||b.size()>MAX_CUE||limit>MAX_CUE)
   throw std::invalid_argument("bounded cue distance out of range");
  const size_t delta=a.size()>b.size()?a.size()-b.size():b.size()-a.size();
  if(delta>limit)return limit+1;
  std::array<size_t,MAX_CUE+1> previous{},current{};
  for(size_t j=0;j<=b.size();++j)previous[j]=j;
  for(size_t i=1;i<=a.size();++i){
   current[0]=i;
   size_t min_row=current[0];
   for(size_t j=1;j<=b.size();++j){
    const size_t cost=a[i-1]==b[j-1]?0u:1u;
    current[j]=std::min({previous[j]+1,current[j-1]+1,
                         previous[j-1]+cost});
    min_row=std::min(min_row,current[j]);
   }
   if(min_row>limit)return limit+1;
   previous=current;
  }
  return previous[b.size()];
 }
};
} // namespace nanlm
