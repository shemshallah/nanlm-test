#include "nanlm/harmony.hpp"
#include <iostream>
using namespace nanlm;
int main(){
 std::cout<<"seed,model,trained_ce,unseen_ce,unseen_accuracy\n";
 for(uint32_t seed:{11u,29u,47u})for(unsigned arm=0;arm<3;++arm){
  auto path="/tmp/harmony-audit-"+std::to_string(getpid())+"-"+std::to_string(seed)+"-"+std::to_string(arm);
  {
   Universe u(path);Harmony model(u,arm!=0);std::vector<uint8_t> sources;uint32_t rng=seed;
   double re=0,im=0;unsigned observations=0;
   for(unsigned a=16;a<176;++a)if((a-16)%4)sources.push_back(a);
   for(unsigned epoch=0;epoch<8;++epoch){
    for(size_t i=sources.size();i>1;--i){rng=1664525u*rng+1013904223u;std::swap(sources[i-1],sources[rng%i]);}
    for(auto a:sources){
     auto b=static_cast<uint8_t>((a+19+(arm==2?79:0))%255);model.teach(a,b);
     re+=std::cos(TAU*(int(b)-int(a))/256.);im+=std::sin(TAU*(int(b)-int(a))/256.);++observations;
    }
   }
   double seen=0,unseen=0,matched=0;unsigned correct=0;
   for(unsigned a=16;a<176;++a){
    auto label=static_cast<uint8_t>((a+19)%255);auto p=model.predict(a);double ce=-std::log(p[label]);
    if((a-16)%4)seen+=ce;
    else{
     unseen+=ce;correct+=unsigned(std::max_element(p.begin(),p.end())-p.begin())==label;
     double z=0,truth=0;const double phi=std::atan2(im,re),gain=4*std::hypot(re,im)/(observations+1.);
     for(unsigned j=0;j<255;++j){double v=std::exp(gain*std::cos(TAU*(int(j)-int(a))/256.-phi));z+=v;if(j==label)truth=v;}
     matched+=std::log(z/truth);
    }
   }
   if(arm==1&&(correct!=40||std::abs(unseen-matched)/40>.02))throw std::runtime_error("transfer / matched offset parity failed");
   if(arm==2&&unseen/40<=std::log(255.))throw std::runtime_error("corrupted labels must fail true rule");
   std::cout<<seed<<","<<(arm==0?"count_backoff":arm==1?"harmony":"corrupted_labels")<<","<<seen/120<<","<<unseen/40<<","<<correct/40.<<"\n";
   if(arm==1)std::cout<<seed<<",matched_offset,,"<<matched/40<<",1\n";
  }
  for(auto suffix:{"",".overlay",".so3"})std::filesystem::remove(path+suffix);
 }
 std::cout<<"PASS: Harmony source-disjoint transfer, matched offset parity, corrupted-label control\n";
}
