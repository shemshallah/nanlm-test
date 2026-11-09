#pragma once
#include "nanlm/byte_model.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace nanlm {

// A small, trainable recurrent neural language model, as an EXPERIMENTAL
// component of the same HW0:0 logical space. No second Root or 256-head map.
// It is NOT a GRU or Transformer; it uses a bounded 8-step online
// truncated backpropagation-through-time (TBPTT) approximation.
// Its value must be established with held-out, same-budget comparisons.
class RecurrentByteModel {
public:
 static constexpr size_t V=ByteModel::VOCAB;
 static constexpr size_t H=32;
 static constexpr size_t BPTT_WINDOW=8;
 using Hidden=std::array<float,H>;
 using Scores=std::array<double,V>;
 struct State {
  Hidden h{};
  bool started=false;
 };
 struct Metrics {
  uint64_t tokens=0,correct=0;
  double nll=0;
  double ce()const{return tokens?nll/tokens:0.0;}
  double accuracy()const{return tokens?double(correct)/tokens:0.0;}
 };
private:
 std::string path_;
 std::array<Hidden,V> input_{};
 std::array<Hidden,V> output_{};
 std::array<Hidden,H> recurrent_{};
 std::array<float,V> bias_{};
 uint64_t updates_=0;
 float lr_=0.015f;

 static uint64_t step(uint64_t& x) {
  x+=0x9e3779b97f4a7c15ull;
  uint64_t z=x;
  z=(z^(z>>30))*0xbf58476d1ce4e5b9ull;
  z=(z^(z>>27))*0x94d049bb133111ebull;
  return z^(z>>31);
 }
 static uint64_t hash(const uint8_t* b,size_t n) {
  uint64_t h=14695981039346656037ull;
  for(size_t i=0;i<n;++i){h^=b[i];h*=1099511628211ull;}
  return h;
 }
 static void write64(std::vector<uint8_t>& b,uint64_t n){
  for(unsigned i=0;i<8;++i)b.push_back(uint8_t(n>>(i*8)));
 }
 static uint64_t read64(const std::vector<uint8_t>& b,size_t& offset){
  if(b.size()-offset<8)throw std::runtime_error("truncated recurrent overlay");
  uint64_t n=0;
  for(unsigned i=0;i<8;++i)n|=uint64_t(b[offset++])<<(i*8);
  return n;
 }
 static double rng01(uint64_t& s){return double(step(s)>>11)*(1.0/9007199254740992.0);}
 static double signed_rng(uint64_t& s){return 2.0*rng01(s)-1.0;}
 static void copy_out(std::vector<uint8_t>& data,const void* p,size_t n){
  const auto* q=static_cast<const uint8_t*>(p);data.insert(data.end(),q,q+n);
 }
 static void copy_in(const std::vector<uint8_t>& data,size_t& pos,void* p,size_t n){
  if(data.size()-pos<n)throw std::runtime_error("truncated recurrent parameters");
  std::memcpy(p,data.data()+pos,n);pos+=n;
 }
 static double stable_nll(const Scores& probs,uint8_t label){
  return -std::log(std::max(probs[label],1e-300));
 }
 void initialize(){
  uint64_t s=0x4e414e4c4d313235ull;
  for(auto& embedding:input_)
   for(auto& x:embedding)x=float(0.14*signed_rng(s));
  for(auto& embedding:output_)
   for(auto& x:embedding)x=float(0.08*signed_rng(s));
  for(size_t i=0;i<H;++i)
   for(size_t j=0;j<H;++j)
    recurrent_[i][j]=float((i==j?0.55:0.0)+0.02*signed_rng(s));
 }
 void load(){
  std::ifstream in(path_,std::ios::binary);
  if(!in){
   if(std::filesystem::exists(path_))throw std::runtime_error("cannot open recurrent overlay");
   initialize();return;
  }
  in.seekg(0,std::ios::end);
  const auto n=in.tellg();
  if(n<32||n>static_cast<std::streamoff>(1024*1024))
   throw std::runtime_error("invalid recurrent overlay size");
  in.seekg(0);
  std::vector<uint8_t> bytes(static_cast<size_t>(n));
  if(!in.read(reinterpret_cast<char*>(bytes.data()),n))
   throw std::runtime_error("cannot read recurrent overlay");
  if(std::memcmp(bytes.data(),"NRNN0001",8)!=0)
   throw std::runtime_error("invalid recurrent magic");
  size_t pos=8;
  const uint64_t version=read64(bytes,pos);
  updates_=read64(bytes,pos);
  const uint64_t digest=read64(bytes,pos);
  if(version!=1||digest!=hash(bytes.data()+pos,bytes.size()-pos))
   throw std::runtime_error("recurrent checksum/version invalid");
  copy_in(bytes,pos,input_.data(),sizeof(input_));
  copy_in(bytes,pos,output_.data(),sizeof(output_));
  copy_in(bytes,pos,recurrent_.data(),sizeof(recurrent_));
  copy_in(bytes,pos,bias_.data(),sizeof(bias_));
  if(pos!=bytes.size())throw std::runtime_error("recurrent trailing bytes");
  for(const auto& row:input_)for(float x:row)if(!std::isfinite(x))throw std::runtime_error("input NaN");
  for(const auto& row:output_)for(float x:row)if(!std::isfinite(x))throw std::runtime_error("output NaN");
  for(const auto& row:recurrent_)for(float x:row)if(!std::isfinite(x))throw std::runtime_error("recurrent NaN");
  for(float x:bias_)if(!std::isfinite(x))throw std::runtime_error("bias NaN");
 }
public:
 explicit RecurrentByteModel(Universe& root,std::string path,float lr=0.015f):
  path_(std::move(path)+".rnn-lm"),lr_(lr){
  (void)root;
  if(!std::isfinite(lr)||lr<=0||lr>0.1f)
   throw std::invalid_argument("recurrent learning rate must be in (0, 0.1]");
  load();
 }
 uint64_t updates()const{return updates_;}
 void observe(State& state,uint8_t symbol)const{
  if(symbol==255){state=State{};return;}
  Hidden next{};
  for(size_t i=0;i<H;++i){
   double sum=input_[symbol][i];
   for(size_t j=0;j<H;++j)sum+=recurrent_[i][j]*state.h[j];
   next[i]=static_cast<float>(std::tanh(sum));
  }
  state.h=next;state.started=true;
 }
 Scores predict(const State& state)const{
  Scores probs{};
  double largest=-std::numeric_limits<double>::infinity();
  for(size_t c=0;c<V;++c){
   double score=bias_[c];
   for(size_t i=0;i<H;++i)score+=output_[c][i]*state.h[i];
   probs[c]=score;largest=std::max(largest,score);
  }
  double denominator=0;
  for(double& p:probs){p=std::exp(p-largest);denominator+=p;}
  for(double& p:probs)p/=denominator;
  return probs;
 }
 Metrics evaluate(std::istream& stream)const{
  Metrics result{};
  State s{};
  char ch;
  while(stream.get(ch)){
   const auto c=static_cast<uint8_t>(static_cast<unsigned char>(ch));
   if(c==255){s=State{};continue;}
   if(s.started){
    const Scores p=predict(s);
    result.nll+=stable_nll(p,c);
    const auto best=std::max_element(p.begin(),p.end())-p.begin();
    if(size_t(best)==c)++result.correct;
    ++result.tokens;
   }
   observe(s,c);
  }
  return result;
 }
 Metrics train(std::istream& stream){
  Metrics result{};
  State state{};
  struct Snapshot {
   uint8_t symbol=0;
   Hidden old{};
   Hidden now{};
  };
  // An online truncated eight-step BPTT window. The historical forward
  // activations were computed under the previous online weights; this is
  // an approximation, but the reverse Jacobian is differentiable and
  // explicitly traverses all states in the window, not only its endpoint.
  std::deque<Snapshot> history;
  char ch;
  while(stream.get(ch)){
   const uint8_t label=static_cast<uint8_t>(static_cast<unsigned char>(ch));
   if(label==255){
    state=State{};
    history.clear();
    continue;
   }
   if(state.started){
    const Scores p=predict(state);
    result.nll+=stable_nll(p,label);
    const auto best=std::max_element(p.begin(),p.end())-p.begin();
    if(size_t(best)==label)++result.correct;
    ++result.tokens;
    Hidden upstream{};
    for(size_t c=0;c<V;++c){
     const double delta=p[c]-(c==label?1.0:0.0);
     for(size_t i=0;i<H;++i)
      upstream[i]+=static_cast<float>(delta*output_[c][i]);
    }
    // Output parameters are updated after reading their old Jacobian.
    for(size_t c=0;c<V;++c){
     const double delta=p[c]-(c==label?1.0:0.0);
     for(size_t i=0;i<H;++i)
      output_[c][i]-=float(lr_*delta*state.h[i]);
     bias_[c]-=float(lr_*delta);
    }
    // Work backward through the original unrolled recurrent states.
    for(auto it=history.rbegin();it!=history.rend();++it){
     Hidden local{};
     Hidden propagated{};
     for(size_t i=0;i<H;++i)
      local[i]=static_cast<float>(std::clamp(
       double(upstream[i])*(1-double(it->now[i])*it->now[i]),-2.0,2.0));
     // Use pre-update recurrence for the backward flow.
     for(size_t j=0;j<H;++j){
      double derivative=0;
      for(size_t i=0;i<H;++i)
       derivative+=recurrent_[i][j]*local[i];
      propagated[j]=static_cast<float>(derivative);
     }
     for(size_t i=0;i<H;++i){
      input_[it->symbol][i]-=lr_*local[i];
      for(size_t j=0;j<H;++j)
       recurrent_[i][j]-=lr_*local[i]*it->old[j];
     }
     upstream=propagated;
    }
    ++updates_;
   }
   const State before=state;
   observe(state,label);
   history.push_back(Snapshot{label,before.h,state.h});
   if(history.size()>BPTT_WINDOW)history.pop_front();
  }
  return result;
 }
 std::string generate(const std::string& prompt,size_t length,uint64_t seed,
                      double temperature=0.9,double top_p=0.9)const{
  if(updates_==0)throw std::runtime_error("recurrent model not trained");
  if(length>1000000||!std::isfinite(temperature)||temperature<=0||
     temperature>5||!std::isfinite(top_p)||top_p<=0||top_p>1)
   throw std::invalid_argument("bad recurrent generation arguments");
  State state{};
  for(unsigned char c:prompt)observe(state,c);
  std::string result;
  result.reserve(length);
  uint64_t random=seed;
  for(size_t n=0;n<length;++n){
   const auto p=predict(state);
   std::vector<std::pair<double,uint8_t>> candidates;
   for(size_t i=0;i<V;++i)candidates.emplace_back(std::pow(p[i],1.0/temperature),uint8_t(i));
   std::sort(candidates.begin(),candidates.end(),[](const auto& a,const auto& b){
    return a.first==b.first?a.second<b.second:a.first>b.first;
   });
   double total=0;for(const auto& item:candidates)total+=item.first;
   size_t take=0;double mass=0;
   while(take<candidates.size()&&(take==0||mass/total<top_p))
    mass+=candidates[take++].first;
   const double target=rng01(random)*mass;
   double cum=0;uint8_t chosen=candidates[take-1].second;
   for(size_t i=0;i<take;++i) {
    cum+=candidates[i].first;
    if(target<cum){chosen=candidates[i].second;break;}
   }
   result.push_back(static_cast<char>(chosen));
   observe(state,chosen);
  }
  return result;
 }
 void save()const{
  std::vector<uint8_t> payload;
  copy_out(payload,input_.data(),sizeof(input_));
  copy_out(payload,output_.data(),sizeof(output_));
  copy_out(payload,recurrent_.data(),sizeof(recurrent_));
  copy_out(payload,bias_.data(),sizeof(bias_));
  std::vector<uint8_t> bytes;
  for(char c:std::string("NRNN0001"))bytes.push_back(uint8_t(c));
  write64(bytes,1);
  write64(bytes,updates_);
  write64(bytes,hash(payload.data(),payload.size()));
  bytes.insert(bytes.end(),payload.begin(),payload.end());
  const std::string tmp=path_+".tmp";
  {
   std::ofstream file(tmp,std::ios::binary|std::ios::trunc);
   if(!file||!file.write(reinterpret_cast<const char*>(bytes.data()),
                         static_cast<std::streamsize>(bytes.size())))
    throw std::runtime_error("cannot write recurrent overlay");
   file.flush();
   if(!file)throw std::runtime_error("recurrent overlay flush failed");
  }
  std::filesystem::rename(tmp,path_);
 }
};
} // namespace nanlm
