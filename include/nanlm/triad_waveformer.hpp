#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace nanlm {

// A single, shared-weight causal computation over 255 head registers.
// No separate expert language models, output heads, or memory universes.
class TriadWaveFormer {
public:
 static constexpr size_t HEADS=255, GROUPS=85, LANES=3;
 static constexpr size_t DIM=6, FIELD=64, CONTEXT=64, ACTIVE=8;
 static constexpr size_t PROJECTION_WEIGHTS=3*DIM*DIM;
 struct Parameters {
  std::array<float,PROJECTION_WEIGHTS> qkv{};
  std::array<float,HEADS> head_gain{};
  std::array<float,GROUPS> adapter{};
 };
 struct Workspace {
  std::array<double,FIELD> field{};
  std::array<double,GROUPS> triad{},routing{};
  std::array<uint8_t,ACTIVE> active{};
  size_t active_count=0;
 };
 static Parameters defaults(){
  Parameters p{};
  for(size_t matrix=0;matrix<3;++matrix)
   for(size_t r=0;r<DIM;++r)
    for(size_t c=0;c<DIM;++c){
     const double diagonal=r==c?(matrix==2?0.85:0.90):0.0;
     const double cross=0.035*std::sin(double(1+r+7*c+23*matrix));
     p.qkv[matrix*DIM*DIM+r*DIM+c]=float(diagonal+cross);
    }
  p.head_gain.fill(0.06f);
  p.adapter.fill(0.20f);
  return p;
 }
 static std::array<double,DIM> project(
     const Parameters& p,size_t which,const std::array<double,DIM>& v){
  std::array<double,DIM> out{};
  for(size_t i=0;i<DIM;++i)
   for(size_t j=0;j<DIM;++j)
    out[i]+=p.qkv[which*DIM*DIM+i*DIM+j]*v[j];
  return out;
 }
 static double dot(const std::array<double,DIM>& x,
                   const std::array<double,DIM>& y){
  double z=0;for(size_t d=0;d<DIM;++d)z+=x[d]*y[d];return z;
 }
 static double logit(const Workspace& s,const Parameters& p,
                     uint8_t head,size_t group){
  if(head>=HEADS||group>=GROUPS)throw std::out_of_range("ingress/triad");
  const double f=s.field[(size_t(head)*7)%FIELD]-
                 s.field[(size_t(head)*7+17)%FIELD];
  return p.head_gain[head]*std::clamp(
    s.triad[group]+0.28*f+0.2*s.routing[group],-3.0,3.0);
 }
 // Every head contributes to cheap triad scoring. Only selected triads
 // execute the three causal attention lanes. Mirror() receives ordered
 // addresses; the inverse lane must reverse these arguments, never look
 // at unobserved bytes. The 64D field has four distinct time constants.
 template<class GroupOf,class Embed,class Mirror,class Phase>
 static void advance(Workspace& out,const Workspace& prior,
       const uint8_t* history,size_t length,
       const std::array<double,HEADS>& forward,
       const std::array<double,HEADS>& inverse,
       const std::array<std::array<double,HEADS>,4>& modes,
       const Parameters& p,GroupOf group_of,Embed embedding,
       Mirror mirror,Phase phase,size_t k=ACTIVE){
  if(k==0||k>ACTIVE)throw std::invalid_argument("active triads must be 1..8");
  out=Workspace{};
  if(length==0)return;
  std::array<std::array<uint8_t,LANES>,GROUPS> member{};
  std::array<size_t,GROUPS> count{};
  for(size_t h=0;h<HEADS;++h){
   const size_t g=group_of(static_cast<uint8_t>(h));
   if(g>=GROUPS||count[g]>=LANES)
    throw std::runtime_error("broken triad cardinality");
   member[g][count[g]++]=static_cast<uint8_t>(h);
  }
  for(size_t g=0;g<GROUPS;++g)
   if(count[g]!=LANES)throw std::runtime_error("incomplete triad");
  const uint8_t last=history[length-1];
  const auto input=embedding(last);
  std::array<double,GROUPS> scores{};
  std::array<size_t,GROUPS> rank{};
  for(size_t g=0;g<GROUPS;++g){
   rank[g]=g;
   double wave=0,sim=0,modal=0;
   for(size_t j=0;j<LANES;++j){
    const uint8_t h=member[g][j];
    wave+=0.5*(forward[h]+inverse[h]);
    sim+=dot(input,embedding(h))/DIM;
    modal+=modes[0][h]*modes[0][last];
   }
   // The inactive triads retain a low-cost recurrent footprint.
   out.triad[g]=std::tanh(0.10*wave/3.0+0.015*sim);
   scores[g]=0.36*sim+0.16*wave+0.05*modal+
             0.19*prior.field[g%FIELD]+
             0.10*prior.triad[g];
  }
  std::sort(rank.begin(),rank.end(),[&](size_t a,size_t b){
   return scores[a]==scores[b]?a<b:scores[a]>scores[b];
  });
  out.active_count=k;
  double max_score=-std::numeric_limits<double>::infinity();
  for(size_t j=0;j<k;++j)
   max_score=std::max(max_score,scores[rank[j]]);
  double route_total=0;
  for(size_t j=0;j<k;++j){
   const size_t g=rank[j];
   out.active[j]=static_cast<uint8_t>(g);
   out.routing[g]=std::exp(std::clamp(scores[g]-max_score,-50.0,0.0));
   route_total+=out.routing[g];
  }
  for(size_t j=0;j<k;++j)
   out.routing[rank[j]]/=route_total;
  const size_t start=length>CONTEXT?length-CONTEXT:0;
  for(size_t slot=0;slot<k;++slot){
   const size_t g=rank[slot];
   std::array<double,LANES> lanes{};
   for(size_t lane=0;lane<LANES;++lane){
    const uint8_t query_head=member[g][lane];
    const auto q=project(p,0,embedding(query_head));
    std::array<double,CONTEXT> attention{};
    double maximum=-std::numeric_limits<double>::infinity();
    for(size_t pos=start;pos<length;++pos){
     const uint8_t src=history[pos];
     const auto key=project(p,1,embedding(src));
     const double forward_evidence=mirror(src,query_head);
     const double inverse_evidence=mirror(query_head,src);
     const double directed=lane==0?forward_evidence:
         (lane==1?inverse_evidence:forward_evidence*inverse_evidence);
     const double angle=phase(src,query_head);
     const double lag=double(length-pos);
     const double score=dot(q,key)/std::sqrt(double(DIM))+
        0.28*directed+0.09*std::cos(angle+0.22*lag);
     attention[pos-start]=score;
     maximum=std::max(maximum,score);
    }
    double total=0;
    for(size_t pos=start;pos<length;++pos){
     double& w=attention[pos-start];
     w=std::exp(std::clamp(w-maximum,-50.0,0.0));
     total+=w;
    }
    double pooled=0;
    for(size_t pos=start;pos<length;++pos){
     const uint8_t src=history[pos];
     const auto value=project(p,2,embedding(src));
     const double current=lane==1?inverse[src]:forward[src];
     pooled+=(attention[pos-start]/total)*
       (dot(q,value)/DIM+0.19*current+
        0.10*mirror(lane==1?query_head:src,
                    lane==1?src:query_head));
    }
    lanes[lane]=std::tanh(pooled);
   }
   const double composition=lanes[2]+0.35*lanes[0]*lanes[1];
   out.triad[g]=std::tanh(out.triad[g]+
       p.adapter[g]*(lanes[0]+lanes[1]+composition));
  }
  static constexpr std::array<double,4> decay{{0.53,0.84,0.962,0.994}};
  for(size_t i=0;i<FIELD;++i){
   double routed=0;
   for(size_t slot=0;slot<k;++slot){
    const size_t g=rank[slot];
    routed+=out.routing[g]*out.triad[g]*
      std::cos(0.11*double((g+1)*(i%16+1)));
   }
   const double global_input=0.21*input[i%DIM]+0.27*routed+
      0.05*(forward[last]+inverse[last])+0.05*modes[i%4][last];
   const double gate=1.0/(1.0+std::exp(-std::clamp(
       0.20+0.35*global_input+0.10*prior.field[i],-20.0,20.0)));
   const double d=decay[i/16];
   out.field[i]=std::clamp(d*prior.field[i]+
       (1.0-d)*gate*std::tanh(global_input+0.18*prior.field[i]),
       -4.0,4.0);
  }
 }
};
} // namespace nanlm
