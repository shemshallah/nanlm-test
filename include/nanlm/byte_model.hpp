#pragma once
#include "nanlm/teacher.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace nanlm {

// An autoregressive, finite-context probabilistic learner attached to ONE HW0:0.
// Its sparse variable-order statistics are a persistent overlay, not another
// 8 MiB Root, additional 256-head universe, or a trained neural language model.
// All distributions are proper, non-zero probabilities over byte values 0..254.
// Byte 255 is only a stream boundary / reserved Nannet ingress.
class ByteModel {
public:
 static constexpr unsigned VOCAB=FabricTeacher::VOCAB;
 static constexpr size_t ORDER=5;
 static constexpr size_t MAX_NODES=500000;
 struct Context {
  std::array<uint8_t,ORDER> bytes{};
  uint8_t size=0;
  void push(uint8_t byte) {
   if(byte==255){size=0;return;}
   if(size<ORDER){bytes[size++]=byte;return;}
   for(size_t i=1;i<ORDER;++i)bytes[i-1]=bytes[i];
   bytes[ORDER-1]=byte;
  }
 };
 struct Distribution {
  std::array<double,VOCAB> probability{};
  uint8_t best=0;
 };
 struct Metrics {
  uint64_t bytes=0,correct=0,novel_contexts=0;
  double sum_nll=0;
  double mean_ce()const{return bytes?sum_nll/static_cast<double>(bytes):0.0;}
  double accuracy()const{return bytes?double(correct)/double(bytes):0.0;}
 };
private:
 struct Node {
  uint32_t total=0;
  std::vector<std::pair<uint8_t,uint32_t>> next; // sorted by symbol
 };
 std::string path_;
 std::unordered_map<uint64_t,Node> nodes_;
 uint64_t tokens_=0;
 double smoothing_=0.35;
 double strength_=4.0;
 FabricTeacher* teacher_=nullptr; // existing resident teacher, never owned
 double teacher_mix_=0.0;

 static uint64_t make_key(const Context& ctx,size_t order) {
  if(order>ctx.size||order>ORDER)throw std::invalid_argument("bad context order");
  uint64_t key=static_cast<uint64_t>(order)<<56;
  for(size_t i=0;i<order;++i)
   key|=static_cast<uint64_t>(ctx.bytes[ctx.size-order+i])<<(i*8);
  return key;
 }
 const Node* find(uint64_t key)const {
  const auto it=nodes_.find(key);
  return it==nodes_.end()?nullptr:&it->second;
 }
 static void increment(Node& n,uint8_t label) {
  if(n.total==std::numeric_limits<uint32_t>::max())
   throw std::overflow_error("byte model counter saturated");
  const auto it=std::lower_bound(n.next.begin(),n.next.end(),label,
    [](const auto& p,uint8_t b){return p.first<b;});
  if(it==n.next.end()||it->first!=label)
   n.next.insert(it,std::make_pair(label,1u));
  else {
   if(it->second==std::numeric_limits<uint32_t>::max())
    throw std::overflow_error("byte model symbol counter saturated");
   ++it->second;
  }
  ++n.total;
 }
 static uint64_t checksum(const uint8_t* p,size_t n) {
  uint64_t h=14695981039346656037ull;
  for(size_t i=0;i<n;++i){h^=p[i];h*=1099511628211ull;}
  return h;
 }
 static void write32(std::vector<uint8_t>& out,uint32_t value) {
  for(int i=0;i<4;++i)out.push_back(static_cast<uint8_t>(value>>(8*i)));
 }
 static void write64(std::vector<uint8_t>& out,uint64_t value) {
  for(int i=0;i<8;++i)out.push_back(static_cast<uint8_t>(value>>(8*i)));
 }
 static uint32_t read32(const std::vector<uint8_t>& b,size_t& pos){
  if(b.size()-pos<4)throw std::runtime_error("truncated byte overlay");
  uint32_t v=0;
  for(unsigned i=0;i<4;++i)v|=uint32_t(b[pos++])<<(8*i);
  return v;
 }
 static uint64_t read64(const std::vector<uint8_t>& b,size_t& pos){
  if(b.size()-pos<8)throw std::runtime_error("truncated byte overlay");
  uint64_t v=0;
  for(unsigned i=0;i<8;++i)v|=uint64_t(b[pos++])<<(8*i);
  return v;
 }
 static uint64_t random64(uint64_t& x) {
  x+=0x9e3779b97f4a7c15ull;
  uint64_t z=x;
  z=(z^(z>>30))*0xbf58476d1ce4e5b9ull;
  z=(z^(z>>27))*0x94d049bb133111ebull;
  return z^(z>>31);
 }
 static double random01(uint64_t& state){
  return static_cast<double>(random64(state)>>11)*(1.0/9007199254740992.0);
 }
 void load(){
  std::ifstream in(path_,std::ios::binary);
  if(!in) {
   if(std::filesystem::exists(path_))
    throw std::runtime_error("cannot read byte overlay");
   return;
  }
  in.seekg(0,std::ios::end);
  const auto size=in.tellg();
  if(size<40||size>static_cast<std::streamoff>(128u*1024u*1024u))
   throw std::runtime_error("invalid byte overlay size");
  in.seekg(0);
  std::vector<uint8_t> b(static_cast<size_t>(size));
  if(!in.read(reinterpret_cast<char*>(b.data()),size))
   throw std::runtime_error("truncated byte overlay");
  if(std::memcmp(b.data(),"NBLM0001",8)!=0)
   throw std::runtime_error("invalid byte overlay magic");
  size_t pos=8;
  if(read32(b,pos)!=1||read32(b,pos)!=ORDER)
   throw std::runtime_error("unsupported byte overlay version");
  const uint64_t n_tokens=read64(b,pos);
  const uint64_t n_nodes=read64(b,pos);
  const uint64_t expected=read64(b,pos);
  if(n_nodes>MAX_NODES||checksum(b.data()+pos,b.size()-pos)!=expected)
   throw std::runtime_error("byte overlay checksum or size mismatch");
  for(uint64_t i=0;i<n_nodes;++i){
   const uint64_t key=read64(b,pos);
   const uint32_t total=read32(b,pos);
   const uint32_t count=read32(b,pos);
   const uint8_t order=static_cast<uint8_t>(key>>56);
   if(order>ORDER||count>VOCAB||count==0||total==0)
    throw std::runtime_error("invalid byte context record");
   if(order<ORDER && ((key&0x00ffffffffffffffull)>>(8*order))!=0)
    throw std::runtime_error("noncanonical context key");
   Node node;node.total=total;
   uint64_t sum=0;
   for(uint32_t j=0;j<count;++j){
    if(pos>=b.size())throw std::runtime_error("truncated byte counts");
    const uint8_t symbol=b[pos++];
    const uint32_t count_symbol=read32(b,pos);
    if(symbol==255||count_symbol==0||
       (!node.next.empty()&&symbol<=node.next.back().first))
     throw std::runtime_error("invalid byte count ordering");
    sum+=count_symbol;
    node.next.emplace_back(symbol,count_symbol);
   }
   if(sum!=total||!nodes_.emplace(key,std::move(node)).second)
    throw std::runtime_error("invalid or duplicate byte context");
  }
  if(pos!=b.size())throw std::runtime_error("byte overlay trailing data");
  if((n_tokens==0)!=(nodes_.empty()))
   throw std::runtime_error("byte overlay token count mismatch");
  const Node* base=find(0);
  if(n_tokens>0&&(!base||base->total!=n_tokens))
   throw std::runtime_error("byte overlay unigram count mismatch");
  tokens_=n_tokens;
 }
public:
 ByteModel(Universe& universe,std::string root_path,double smoothing=0.35,
           double strength=4.0,FabricTeacher* teacher=nullptr,double teacher_mix=0.0):
  path_(std::move(root_path)+".byte-lm"),
  smoothing_(smoothing),strength_(strength),teacher_(teacher),
  teacher_mix_(teacher_mix) {
  if(!std::isfinite(smoothing)||smoothing<=0||!std::isfinite(strength)||
     strength<=0||!std::isfinite(teacher_mix)||teacher_mix<0||teacher_mix>1||
     (teacher_mix>0&&!teacher))
   throw std::invalid_argument("invalid byte model hyperparameters");
  (void)universe; // Explicitly bind the overlay's lifecycle to the caller's sole HW0 universe.
  load();
 }
 size_t contexts()const{return nodes_.size();}
 uint64_t tokens()const{return tokens_;}
 void learn(const Context& ctx,uint8_t next) {
  if(next==255)throw std::invalid_argument("255 is reserved as stream boundary");
  if(tokens_==std::numeric_limits<uint32_t>::max())
   throw std::overflow_error("byte model training limit reached");
  for(size_t k=0;k<=ctx.size;++k) {
   const uint64_t key=make_key(ctx,k);
   auto it=nodes_.find(key);
   if(it==nodes_.end()){
    if(nodes_.size()>=MAX_NODES)continue; // bounded sparse overlay
    it=nodes_.emplace(key,Node{}).first;
   }
   increment(it->second,next);
  }
  ++tokens_;
 }
 bool seen_context(const Context& ctx)const{
  return find(make_key(ctx,ctx.size))!=nullptr;
 }
 Distribution predict(const Context& ctx,unsigned max_order=ORDER)const {
  if(max_order>ORDER)throw std::invalid_argument("unsupported context order");
  Distribution out{};
  const Node* base=find(0);
  if(!base){
   out.probability.fill(1.0/static_cast<double>(VOCAB));
  }else{
   const double denom=double(base->total)+smoothing_;
   out.probability.fill((smoothing_/VOCAB)/denom);
   for(const auto& [sym,count]:base->next)
    out.probability[sym]+=double(count)/denom;
  }
  if(teacher_mix_>0&&ctx.size>0) {
   const auto geometric=teacher_->predict(ctx.bytes[ctx.size-1]);
   for(unsigned i=0;i<VOCAB;++i)
    out.probability[i]=(1-teacher_mix_)*out.probability[i]+
                        teacher_mix_*geometric.probability[i];
  }
  for(size_t k=1;k<=std::min<size_t>(ctx.size,max_order);++k){
   const Node* node=find(make_key(ctx,k));
   if(!node)continue;
   // Variable-order hierarchical interpolation: rare contexts retain a
   // strong suffix prior instead of memorizing a single observation.
   const double n=double(node->total);
   const double smooth=strength_+1.5*node->next.size();
   const double weight=n/(n+smooth);
   for(double& p:out.probability)p*=1.0-weight;
   for(const auto& [sym,count]:node->next)
    out.probability[sym]+=weight*(double(count)/n);
  }
  double best=-1;
  for(unsigned i=0;i<VOCAB;++i) {
   if(!std::isfinite(out.probability[i])||out.probability[i]<=0)
    throw std::runtime_error("nonfinite or zero byte probability");
   if(out.probability[i]>best){best=out.probability[i];out.best=static_cast<uint8_t>(i);}
  }
  return out;
 }
 Metrics evaluate(std::istream& input,unsigned max_order=ORDER)const {
  Metrics m{};
  Context ctx{};
  char ch;
  while(input.get(ch)){
   const uint8_t symbol=static_cast<uint8_t>(static_cast<unsigned char>(ch));
   if(symbol==255){ctx=Context{};continue;}
   if(ctx.size>0&&!seen_context(ctx))++m.novel_contexts;
   const auto distribution=predict(ctx,max_order);
   m.sum_nll-=std::log(distribution.probability[symbol]);
   if(distribution.best==symbol)++m.correct;
   ++m.bytes;
   ctx.push(symbol);
  }
  return m;
 }
 uint64_t train(std::istream& input,FabricTeacher* teacher=nullptr,
                uint64_t teacher_stride=0,uint64_t initial_step=0) {
  if(teacher_stride&& !teacher)
   throw std::invalid_argument("teacher stride requires a resident teacher");
  uint64_t trained=0;
  Context ctx{};
  char ch;
  while(input.get(ch)){
   const uint8_t symbol=static_cast<uint8_t>(static_cast<unsigned char>(ch));
   if(symbol==255){ctx=Context{};continue;}
   if(teacher_stride&&ctx.size>0&&((initial_step+trained)%teacher_stride==0)){
    teacher->teach(ctx.bytes[ctx.size-1],symbol,initial_step+trained+1);
   }
   learn(ctx,symbol);
   ctx.push(symbol);
   ++trained;
  }
  return trained;
 }
 std::string generate(const std::string& prompt,size_t length,uint64_t seed,
                      double temperature=0.85,double top_p=0.92)const {
  if(tokens_==0)throw std::runtime_error("train a corpus before generating");
  if(!std::isfinite(temperature)||temperature<=0||temperature>5||
     !std::isfinite(top_p)||top_p<=0||top_p>1||length>1000000)
   throw std::invalid_argument("invalid generation parameters");
  Context ctx{};
  for(unsigned char c:prompt)ctx.push(c);
  const Node* base=find(0);
  if(!base)throw std::runtime_error("unigram state missing");
  std::string result;result.reserve(length);
  uint64_t state=seed;
  for(size_t step=0;step<length;++step) {
   const auto distribution=predict(ctx);
   std::vector<std::pair<double,uint8_t>> candidates;
   candidates.reserve(base->next.size());
   // Do not generate symbols absent from the training corpus. Evaluation
   // still assigns nonzero probability to all 255 ordinary byte values.
   for(const auto& [byte,count]:base->next) {
    (void)count;
    candidates.emplace_back(std::pow(distribution.probability[byte],1.0/temperature),byte);
   }
   std::sort(candidates.begin(),candidates.end(),[](const auto& a,const auto& b){
    return a.first==b.first?a.second<b.second:a.first>b.first;
   });
   double mass=0;
   for(const auto& [value,byte]:candidates){(void)byte;mass+=value;}
   if(!(mass>0)||!std::isfinite(mass))
    throw std::runtime_error("invalid generation distribution");
   double kept=0;
   size_t n=0;
   while(n<candidates.size()&&(kept/mass<top_p||n==0))
    kept+=candidates[n++].first;
   const double sample=random01(state)*kept;
   double cursor=0;
   uint8_t chosen=candidates[n-1].second;
   for(size_t i=0;i<n;++i){
    cursor+=candidates[i].first;
    if(sample<cursor){chosen=candidates[i].second;break;}
   }
   result.push_back(static_cast<char>(chosen));
   ctx.push(chosen);
  }
  return result;
 }
 void save()const{
  if(nodes_.empty())return;
  std::vector<uint64_t> keys;keys.reserve(nodes_.size());
  for(const auto& kv:nodes_)keys.push_back(kv.first);
  std::sort(keys.begin(),keys.end());
  std::vector<uint8_t> payload;
  for(uint64_t key:keys){
   const auto& n=nodes_.at(key);
   write64(payload,key);
   write32(payload,n.total);
   write32(payload,static_cast<uint32_t>(n.next.size()));
   for(const auto& [byte,count]:n.next){
    payload.push_back(byte);
    write32(payload,count);
   }
  }
  std::vector<uint8_t> out;
  for(const char c:std::string("NBLM0001"))out.push_back(static_cast<uint8_t>(c));
  write32(out,1);write32(out,ORDER);
  write64(out,tokens_);write64(out,nodes_.size());
  write64(out,checksum(payload.data(),payload.size()));
  out.insert(out.end(),payload.begin(),payload.end());
  const std::string temp=path_+".tmp";
  {
   std::ofstream file(temp,std::ios::binary|std::ios::trunc);
   if(!file||!file.write(reinterpret_cast<const char*>(out.data()),
                         static_cast<std::streamsize>(out.size())))
    throw std::runtime_error("cannot save byte overlay");
   file.flush();
   if(!file)throw std::runtime_error("byte overlay flush failed");
  }
  std::filesystem::rename(temp,path_);
  // This is a checksum-verified atomic replacement, NOT a jointly atomic
  // transaction with the mmap Root, quaternion and temporal evidence journals.
 }
};

} // namespace nanlm
