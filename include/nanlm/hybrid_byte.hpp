#pragma once
#include "nanlm/recurrent_byte.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace nanlm {

// Shared-inference composition, not a duplicate model or duplicate Root.
// Keep both pathways independently testable: mixture 0 == exact count model,
// mixture 1 == exact neural state model. All state resets at reserved byte 255.
class HybridByteInference {
public:
 struct State {
  ByteModel::Context suffix{};
  RecurrentByteModel::State recurrent{};
 };
 struct Metrics {
  uint64_t tokens=0,correct=0;
  double nll=0;
  double ce()const{return tokens?nll/tokens:0.0;}
  double accuracy()const{return tokens?double(correct)/tokens:0.0;}
 };
private:
 const ByteModel& count_;
 const RecurrentByteModel& neural_;
 double neural_weight_;
 unsigned count_order_;
 static uint64_t rng(uint64_t& x) {
  x+=0x9e3779b97f4a7c15ull;
  uint64_t z=x;
  z=(z^(z>>30))*0xbf58476d1ce4e5b9ull;
  z=(z^(z>>27))*0x94d049bb133111ebull;
  return z^(z>>31);
 }
 static double uniform(uint64_t& state){
  return double(rng(state)>>11)*(1.0/9007199254740992.0);
 }
public:
 HybridByteInference(const ByteModel& count,const RecurrentByteModel& neural,double mix,
                     unsigned count_order=ByteModel::ORDER):
  count_(count),neural_(neural),neural_weight_(mix),count_order_(count_order){
  if(count_order>ByteModel::ORDER||!std::isfinite(mix)||mix<0||mix>1)
   throw std::invalid_argument("hybrid mixture must lie in [0,1]");
 }
 void observe(State& s,uint8_t token)const{
  s.suffix.push(token);
  neural_.observe(s.recurrent,token);
 }
 RecurrentByteModel::Scores predict(const State& state)const{
  RecurrentByteModel::Scores p{};
  if(neural_weight_==0){
   p=count_.predict(state.suffix,count_order_).probability;
  }else if(neural_weight_==1){
   p=neural_.predict(state.recurrent);
  }else{
   const auto counts=count_.predict(state.suffix,count_order_);
   const auto recurrent=neural_.predict(state.recurrent);
   for(size_t i=0;i<ByteModel::VOCAB;++i)
    p[i]=(1-neural_weight_)*counts.probability[i]+
         neural_weight_*recurrent[i];
  }
  return p;
 }
 Metrics evaluate(std::istream& input)const{
  State s{};
  Metrics m{};
  char next;
  while(input.get(next)){
   const uint8_t symbol=static_cast<uint8_t>(static_cast<unsigned char>(next));
   if(symbol==255){s=State{};continue;}
   // Aligned evaluation: condition on a previously observed symbol
   // for all arms, rather than giving the count model an extra BOS score.
   if(s.recurrent.started){
    const auto p=predict(s);
    m.nll-=std::log(std::max(p[symbol],1e-300));
    if(std::distance(p.begin(),std::max_element(p.begin(),p.end()))==symbol)
     ++m.correct;
    ++m.tokens;
   }
   observe(s,symbol);
  }
  return m;
 }
 std::string generate(const std::string& prompt,size_t length,uint64_t seed,
                      double temperature=0.85,double top_p=0.92)const{
  if(count_.tokens()==0||neural_.updates()==0)
   throw std::runtime_error("both hybrid pathways require training");
  if(length>1000000||!std::isfinite(temperature)||temperature<=0||
     temperature>5||!std::isfinite(top_p)||top_p<=0||top_p>1)
   throw std::invalid_argument("invalid hybrid sampling");
  State state{};
  for(unsigned char b:prompt)observe(state,b);
  uint64_t random=seed;
  std::string generated;generated.reserve(length);
  for(size_t i=0;i<length;++i){
   const auto probs=predict(state);
   std::vector<std::pair<double,uint8_t>> ranked;
   ranked.reserve(ByteModel::VOCAB);
   for(size_t c=0;c<ByteModel::VOCAB;++c)
    ranked.emplace_back(std::pow(probs[c],1.0/temperature),uint8_t(c));
   std::sort(ranked.begin(),ranked.end(),[](const auto& a,const auto& b){
    return a.first==b.first?a.second<b.second:a.first>b.first;
   });
   double normalizer=0;
   for(const auto& a:ranked)normalizer+=a.first;
   size_t take=0;double sum=0;
   while(take<ranked.size()&&(take==0||sum/normalizer<top_p))
    sum+=ranked[take++].first;
   const double sample=uniform(random)*sum;
   double cdf=0;
   uint8_t selected=ranked[take-1].second;
   for(size_t j=0;j<take;++j){
    cdf+=ranked[j].first;
    if(sample<cdf){selected=ranked[j].second;break;}
   }
   generated.push_back(static_cast<char>(selected));
   observe(state,selected);
  }
  return generated;
 }
};
} // namespace nanlm
