#pragma once
#include "nanlm/byte_model.hpp"
#include "nanlm/recurrent_byte.hpp"
#include "nanlm/harmony.hpp"
#include "nanlm/wordnet.hpp"
#include <iomanip>
#include <sstream>
namespace nanlm {
// One causal state, one normalized distribution, one leaf contract.
class UnifiedModel {
 public:
 static constexpr unsigned V=255,K=6,REGIMES=3;
 using Distribution=std::array<double,V>;
 inline static constexpr const char* names[K]={"context","recurrent","boolean_phase","geometry","spectral","wordnet"};
 struct State {
  ByteModel::Context context{};
  RecurrentByteModel::State recurrent{},previous{};
  std::string word;
  std::vector<uint32_t> senses;
  bool pinned=false;
 };
 struct Prediction {
  std::array<Distribution,K> expert{};
  std::array<double,K> weights{};
  Distribution probability{};
  uint8_t best=0;
 };
 struct Metrics {
  uint64_t bytes=0,correct=0;
  double nll=0;
  double ce()const{return bytes?nll/bytes:0;}
 };
 private:
 Universe& u_;
 std::string path_;
 ByteModel context_;
 RecurrentByteModel recurrent_;
 Harmony harmony_;
 FabricTeacher geometry_;
 WordNetIndex knowledge_;
 std::array<std::array<double,K>,REGIMES> gates_{};
 uint64_t steps_=0;
 inline static constexpr const char* suffixes[6]={"",".overlay",".so3",".byte-lm",".rnn-lm",".wordnet.tsv"};
 static uint64_t hash_bytes(const std::string& s){uint64_t h=14695981039346656037ull;for(unsigned char c:s){h^=c;h*=1099511628211ull;}return h;}
 static uint64_t file_hash(const std::string& path){
  if(!std::filesystem::exists(path))return 0;
  std::ifstream in(path,std::ios::binary);if(!in)throw std::runtime_error("cannot read checkpoint component");
  uint64_t h=14695981039346656037ull;char buf[65536];
  while(in){in.read(buf,sizeof buf);for(std::streamsize i=0;i<in.gcount();++i){h^=static_cast<uint8_t>(buf[i]);h*=1099511628211ull;}}
  if(!in.eof())throw std::runtime_error("checkpoint component read failed");
  return h;
 }
 unsigned regime(const State& s)const{return s.context.size==0?0:context_.seen_context(s.context)?2:1;}
 static uint8_t previous_byte(const State& s){return s.context.size?s.context.bytes[s.context.size-1]:0;}
 void select(State& s,const std::string& lemma)const{
  s.senses=knowledge_.lookup(lemma);
  // Exact IDs, never collision-prone physical anchors, define the retained senses.
  std::sort(s.senses.begin(),s.senses.end());s.senses.erase(std::unique(s.senses.begin(),s.senses.end()),s.senses.end());
  if(s.senses.size()>64)s.senses.resize(64);
 }
 void load_gate(){
  for(auto& row:gates_)row.fill(1./K);
  std::ifstream f(path_+".unified",std::ios::binary);if(!f)return;
  if(std::filesystem::file_size(path_+".unified")>4096)throw std::runtime_error("oversized unified manifest");
  const std::string text((std::istreambuf_iterator<char>(f)),{});
  const auto marker=text.rfind("CHECK ");if(marker==std::string::npos)throw std::runtime_error("unified checksum absent");
  const auto body=text.substr(0,marker);
  std::istringstream checksum(text.substr(marker+6));uint64_t expected=0;checksum>>expected;checksum>>std::ws;
  if(!checksum.eof()||expected!=hash_bytes(body))throw std::runtime_error("unified checksum mismatch");
  std::istringstream in(body);std::string magic;in>>magic>>steps_;
  if(magic!="NUNIFIED1")throw std::runtime_error("unsupported unified format");
  for(auto& row:gates_){double sum=0;for(auto& w:row){in>>w;if(!std::isfinite(w)||w<=0||w>1)throw std::runtime_error("invalid mixture gate");sum+=w;}
   if(std::abs(sum-1)>1e-10)throw std::runtime_error("unnormalized mixture gate");}
  for(const char* suffix:suffixes){uint64_t digest=0;in>>digest;if(!in||digest!=file_hash(path_+suffix))throw std::runtime_error("unified component changed: restore a complete checkpoint or explicitly adopt-state");}
  in>>std::ws;if(!in.eof())throw std::runtime_error("unified trailing fields");
 }
 public:
 explicit UnifiedModel(Universe& universe,const std::string& path,bool adopt=false):
  u_(universe),path_(path),context_(universe,path),recurrent_(universe,path),harmony_(universe),geometry_(universe){
  if(adopt)for(auto& row:gates_)row.fill(1./K);else load_gate();
  if(std::filesystem::exists(path_+".wordnet.tsv"))knowledge_.load(path_+".wordnet.tsv");
 }
 uint64_t steps()const{return steps_;}
 uint64_t tokens()const{return context_.tokens();}
 size_t contexts()const{return context_.contexts();}
 void observe(State& s,uint8_t b)const{
  if(b==255){s=State{};return;}
  s.previous=s.recurrent;recurrent_.observe(s.recurrent,b);s.context.push(b);
  if((b>='a'&&b<='z')||(b>='A'&&b<='Z')||b=='_'){
   if(s.word.size()<128)s.word.push_back(static_cast<char>(b));
  }else if(!s.word.empty()){
   if(!s.pinned){auto ids=knowledge_.lookup(s.word);if(!ids.empty())select(s,s.word);}
   s.word.clear();
  }
 }
 State prime(const std::string& prompt,const std::string& lemma="")const{
  State s;for(unsigned char b:prompt)observe(s,b);
  if(!lemma.empty()){select(s,lemma);s.pinned=true;}
  else if(!s.word.empty()&&!knowledge_.lookup(s.word).empty())select(s,s.word);
  return s;
 }
 Prediction predict(const State& s,unsigned enabled=(1u<<K)-1)const{
  if((enabled&((1u<<K)-1))==0)throw std::invalid_argument("at least one expert required");
  Prediction out;out.expert[0]=context_.predict(s.context).probability;out.expert[1]=recurrent_.predict(s.recurrent);
  if(s.context.size){out.expert[2]=harmony_.predict(previous_byte(s));out.expert[3]=geometry_.predict(previous_byte(s)).probability;}
  else{out.expert[2].fill(1./V);out.expert[3].fill(1./V);}
  // Same leaf-target order before/after transform; spectral index is an explicit
  // learned byte prior, not a claim that frequency bins have lexical semantics.
  std::array<std::complex<double>,256> wave{};
  for(unsigned j=0;j<V;++j){const auto p=binary_angle(u_.covariant_phase(previous_byte(s),static_cast<uint8_t>(j)));auto z=phasor(p);wave[j]=std::sqrt(out.expert[0][j])*z/std::abs(z);}
  row_fft(wave);double mass=0;
  for(unsigned j=0;j<V;++j){out.expert[4][j]=std::norm(wave[j])+1e-9;mass+=out.expert[4][j];}
  for(auto& p:out.expert[4])p/=mass;
  out.expert[5].fill(.5);mass=V*.5;
  for(auto id:s.senses){const auto* c=knowledge_.get_concept(id);if(!c)throw std::runtime_error("missing exact synset");
   const std::string text=c->lemma+" "+c->gloss;int last=-1;
   for(unsigned char b:text){if(b==255){last=-1;continue;}const double w=(last==previous_byte(s)?4.:1.);out.expert[5][b]+=w;mass+=w;last=b;}}
  for(auto& p:out.expert[5])p/=mass;
  double total=0;
  for(unsigned k=0;k<K;++k){out.weights[k]=(enabled&(1u<<k))?gates_[regime(s)][k]:0;total+=out.weights[k];}
  for(auto& w:out.weights)w/=total;
  double sum=0;
  for(unsigned j=0;j<V;++j){for(unsigned k=0;k<K;++k)out.probability[j]+=out.weights[k]*out.expert[k][j];sum+=out.probability[j];}
  for(auto& p:out.probability){p/=sum;if(!std::isfinite(p)||p<=0)throw std::runtime_error("invalid unified probability");}
  out.best=static_cast<uint8_t>(std::max_element(out.probability.begin(),out.probability.end())-out.probability.begin());return out;
 }
 // Every expert sees the same pre-observation history. One factual pair event,
 // even though both geometric and count objectives learn from that observation.
 double learn(State& s,uint8_t label){
  if(label==255){observe(s,label);return 0;}
  auto p=predict(s);const double loss=-std::log(p.probability[label]);
  auto& gate=gates_[regime(s)];
  for(unsigned k=0;k<K;++k)gate[k]=.999*p.weights[k]*p.expert[k][label]/p.probability[label]+.001/K;
  double sum=0;for(auto w:gate)sum+=w;for(auto& w:gate)w/=sum;
  context_.learn(s.context,label);
  if(s.context.size){
   const auto prev=previous_byte(s);
   recurrent_.teach_step(s.recurrent,s.previous,prev,label);
   harmony_.teach(prev,label);
   geometry_.teach(prev,label,steps_+1,false);
  }
  ++steps_;observe(s,label);return loss;
 }
 Metrics train(std::istream& input){
  State state;Metrics m;char c;
  while(input.get(c)){const auto b=static_cast<uint8_t>(c);if(b==255){observe(state,b);continue;}m.nll+=learn(state,b);++m.bytes;}
  if(!input.eof())throw std::runtime_error("training input error");
  return m;
 }
 Metrics evaluate(std::istream& input,unsigned enabled=(1u<<K)-1)const{
  State state;Metrics m;char c;
  while(input.get(c)){const auto b=static_cast<uint8_t>(c);if(b==255){observe(state,b);continue;}auto p=predict(state,enabled);m.nll-=std::log(p.probability[b]);m.correct+=p.best==b;++m.bytes;observe(state,b);}
  if(!input.eof())throw std::runtime_error("evaluation input error");
  return m;
 }
 WaveState wave(const State& s)const{
  auto p=predict(s);WaveState result;
  for(unsigned j=0;j<V;++j){auto z=phasor(binary_angle(u_.covariant_phase(previous_byte(s),static_cast<uint8_t>(j))));result.amplitude[j]=std::sqrt(p.probability[j])*z/std::abs(z);}return result;
 }
 std::string generate(const std::string& prompt,size_t length,uint64_t seed,double temperature=.85,double top_p=.92,const std::string& lemma="")const{
  if(context_.tokens()==0)throw std::runtime_error("train or import context counts before generation");
  if(length>1000000||!std::isfinite(temperature)||temperature<=0||temperature>5||!std::isfinite(top_p)||top_p<=0||top_p>1)throw std::invalid_argument("invalid sampling parameters");
  State s=prime(prompt,lemma);std::mt19937_64 random(seed);std::string result;result.reserve(length);
  for(size_t i=0;i<length;++i){const auto p=predict(s);std::array<std::pair<double,uint8_t>,V> choices{};
   double maxlog=-1e300;for(double q:p.probability)maxlog=std::max(maxlog,std::log(q)/temperature);
   double total=0;for(unsigned j=0;j<V;++j){double v=std::exp(std::log(p.probability[j])/temperature-maxlog);choices[j]={v,static_cast<uint8_t>(j)};total+=v;}
   std::sort(choices.begin(),choices.end(),[](auto a,auto b){return a.first==b.first?a.second<b.second:a.first>b.first;});
   size_t count=0;double kept=0;while(count<V&&(count==0||kept<total*top_p))kept+=choices[count++].first;
   double sample=double(random()>>11)*0x1.0p-53*kept;uint8_t b=choices[count-1].second;
   for(size_t j=0;j<count;++j){if(sample<choices[j].first){b=choices[j].second;break;}sample-=choices[j].first;}
   result.push_back(static_cast<char>(b));observe(s,b);
  }return result;
 }
 const Leaf& leaf(const State& s,uint8_t candidate)const{
  if(candidate==255)throw std::invalid_argument("gateway is not a generation leaf");
  return u_.leaf(address(previous_byte(s),candidate));
 }
 std::vector<Concept> facts(const State& s)const{std::vector<Concept> v;for(auto id:s.senses)v.push_back(*knowledge_.get_concept(id));return v;}
 void import_wordnet(const std::string& file){knowledge_.import_noun_data(file,u_);knowledge_.save(path_+".wordnet.tsv");}
 void save()const{
  context_.save();recurrent_.save();u_.sync();
  std::ostringstream out;out<<"NUNIFIED1 "<<steps_<<"\n"<<std::setprecision(17);
  for(const auto& row:gates_){for(double w:row)out<<w<<" ";out<<"\n";}
  for(const char* suffix:suffixes)out<<file_hash(path_+suffix)<<"\n";
  const std::string body=out.str();std::ofstream f(path_+".unified.tmp",std::ios::binary|std::ios::trunc);
  f<<body<<"CHECK "<<hash_bytes(body)<<"\n";f.close();if(!f)throw std::runtime_error("unified manifest save failed");
  std::filesystem::rename(path_+".unified.tmp",path_+".unified");
 }
};
}
