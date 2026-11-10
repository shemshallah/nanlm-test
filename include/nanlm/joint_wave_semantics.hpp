#pragma once
// EXPERIMENTAL: differentiable semantic + next-byte *readout* over the actual
// UnifiedWaveLanguage recurrent state. The oscillator, teacher, exact WordNet
// claims and NaN16 root are not altered by these gradients. This is NOT
// end-to-end reverse-mode credit through the recurrent wave dynamics.
#include "nanlm/unified_wave.hpp"
#include "nanlm/contextual_senses.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace nanlm {
class JointWaveSemanticLearner {
 public:
 static constexpr size_t D=24;
 static constexpr size_t N=UnifiedWaveLanguage::N;
 using V=std::array<double,D>;
 using X=std::array<double,N>;
 using State=UnifiedWaveLanguage::State;
 using Role=ContextualRoleLearner::Example;
 using Sense=ContextualSenseLearner::Example;
 struct Metrics {
  size_t count=0,correct=0;
  double loss=0;
  double ce() const {return count?loss/double(count):0.;}
  double accuracy() const {return count?double(correct)/double(count):0.;}
 };
 struct BytePrediction {std::array<double,N> p{}; uint8_t best=0;};
 struct SensePrediction {uint32_t synset=0;double confidence=0;};
 private:
 UnifiedWaveLanguage& wave_;
 std::string filename_;
 std::array<V,N> projection_{},byte_heads_{};
 V bias_{},orientation_{};
 std::map<std::string,V> role_heads_;
 std::map<std::string,double> role_bias_;
 std::map<uint32_t,V> sense_heads_;
 std::map<uint32_t,double> sense_bias_;
 uint64_t byte_updates_=0,role_updates_=0,sense_updates_=0;
 static uint64_t fnv(const std::string& s) {
  uint64_t h=14695981039346656037ull;
  for(unsigned char c:s){h^=c;h*=1099511628211ull;}
  return h;
 }
 static std::string lower(std::string s) {
  for(char& c:s)if(c>='A'&&c<='Z')c=char(c-'A'+'a');
  return s;
 }
 static void key(const std::string& s) {
  if(s.empty()||s.size()>128)throw std::invalid_argument("invalid joint identity");
  for(unsigned char c:s)if(!((c>='a'&&c<='z')||(c>='0'&&c<='9')||c=='_'))
   throw std::invalid_argument("joint identity must be ASCII word");
 }
 static V init(const std::string& tag,double scale) {
  V v{};
  const uint64_t seed=fnv(tag);
  for(size_t d=0;d<D;++d)
   v[d]=scale*std::sin(double((seed%100003u)+uint64_t(17*d+1))*0.0131);
  return v;
 }
 static X input(const State& s) {
  if(!s.length)throw std::invalid_argument("empty wave context");
  X x{};
  for(size_t i=0;i<N;++i)
   x[i]=std::tanh(0.25*s.wave[i]+0.25*s.inverse_wave[i]);
  return x;
 }
 struct Activation {X x{};V h{};};
 Activation encode(const State& s,double direction=0) const {
  Activation a;
  a.x=input(s);
  for(size_t d=0;d<D;++d)
   a.h[d]=bias_[d]+orientation_[d]*direction;
  for(size_t i=0;i<N;++i)
   if(a.x[i]!=0)for(size_t d=0;d<D;++d)
    a.h[d]+=projection_[i][d]*a.x[i];
  for(double& v:a.h)v=std::tanh(v);
  return a;
 }
 static int role_orientation(const Role& row) {
  std::string a=lower(row.subject),b=lower(row.object);
  key(a);key(b);key(lower(row.relation));
  if(a==b)throw std::invalid_argument("identical role entities");
  std::vector<std::string> tokens;
  std::string word;
  if(row.sentence.empty()||row.sentence.size()>4096)
   throw std::invalid_argument("bad role sentence");
  for(size_t i=0;i<=row.sentence.size();++i) {
   const unsigned char c=i<row.sentence.size()?static_cast<unsigned char>(row.sentence[i]):0;
   if((c>='a'&&c<='z')||(c>='A'&&c<='Z')||(c>='0'&&c<='9')||c=='_') {
    if(word.size()>=128)throw std::invalid_argument("oversize role token");
    word.push_back(c>='A'&&c<='Z'?char(c-'A'+'a'):char(c));
   } else if(!word.empty()){tokens.push_back(word);word.clear();}
  }
  if(tokens.size()<3||tokens.size()>128)throw std::invalid_argument("role token count");
  size_t ia=0,ib=0,na=0,nb=0;
  for(size_t i=0;i<tokens.size();++i) {
   if(tokens[i]==a){ia=i;++na;}
   if(tokens[i]==b){ib=i;++nb;}
  }
  if(na!=1||nb!=1)throw std::invalid_argument("role entities must occur exactly once");
  return ia<ib?1:-1;
 }
 static void check_target(const Sense& row) {
  key(lower(row.target));
  size_t hits=0;
  std::string w;
  for(size_t i=0;i<=row.sentence.size();++i) {
   const unsigned char c=i<row.sentence.size()?static_cast<unsigned char>(row.sentence[i]):0;
   if((c>='a'&&c<='z')||(c>='A'&&c<='Z')||(c>='0'&&c<='9')||c=='_') {
    if(w.size()>=128)throw std::invalid_argument("oversize sense token");
    w.push_back(c>='A'&&c<='Z'?char(c-'A'+'a'):char(c));
   } else if(!w.empty()){hits+=(w==lower(row.target));w.clear();}
  }
  if(row.sentence.empty()||row.sentence.size()>4096||hits!=1)
   throw std::invalid_argument("sense target must occur exactly once");
 }
 static std::vector<uint32_t> candidates(const WordNetIndex& wn,const std::string& target) {
  auto ids=wn.lookup(lower(target));
  std::sort(ids.begin(),ids.end());
  ids.erase(std::unique(ids.begin(),ids.end()),ids.end());
  if(ids.size()<2||ids.size()>64)throw std::invalid_argument("ambiguous WordNet target needs 2..64 synsets");
  return ids;
 }
 static void lr_valid(double lr) {
  if(!(lr>0&&lr<=0.1)||!std::isfinite(lr))
   throw std::invalid_argument("joint learning rate outside (0,.1]");
 }
 static double sigmoid(double v) {
  return v>=0?1.0/(1.0+std::exp(-std::min(v,60.))):
      std::exp(std::max(v,-60.))/(1.0+std::exp(std::max(v,-60.)));
 }
 static double dot(const V& a,const V& b){
  double v=0;for(size_t d=0;d<D;++d)v+=a[d]*b[d];return v;
 }
 void adjust(const Activation& a,const V& dh,double lr,double direction) {
  for(size_t d=0;d<D;++d) {
   const double delta=std::clamp(dh[d]*(1-a.h[d]*a.h[d]),-5.,5.);
   bias_[d]-=lr*delta;
   orientation_[d]-=lr*delta*direction;
   for(size_t i=0;i<N;++i)if(a.x[i]!=0)
    projection_[i][d]-=lr*delta*a.x[i];
  }
 }
 static double safe_log(double v){return std::log(std::max(v,1e-300));}
 static void vector_out(std::ostream& out,const V& v) {
  for(double x:v)out<<" "<<x;
  out<<"\n";
 }
 static void vector_in(std::istream& in,V& v) {
  for(double& x:v)
   if(!(in>>x)||!std::isfinite(x)||std::abs(x)>100.)
    throw std::runtime_error("invalid joint parameter");
 }
 void load() {
  std::ifstream f(filename_,std::ios::binary);
  if(!f){
   if(std::filesystem::exists(filename_))throw std::runtime_error("unreadable joint checkpoint");
   return;
  }
  const auto size=std::filesystem::file_size(filename_);
  if(size<40||size>8u*1024u*1024u)throw std::runtime_error("invalid joint checkpoint size");
  const std::string raw((std::istreambuf_iterator<char>(f)),{});
  const size_t pos=raw.rfind("CHECK ");
  if(pos==std::string::npos||pos==0||raw[pos-1]!='\n')
   throw std::runtime_error("joint checksum missing");
  const std::string body=raw.substr(0,pos);
  std::istringstream hash(raw.substr(pos+6));
  uint64_t expected=0;
  if(!(hash>>expected)||fnv(body)!=expected)
   throw std::runtime_error("joint checksum mismatch");
  hash>>std::ws;
  if(!hash.eof())throw std::runtime_error("joint checksum trailing data");
  std::istringstream in(body);
  std::string tag;
  if(!(in>>tag>>byte_updates_>>role_updates_>>sense_updates_)||tag!="NJOINTWAVE1")
   throw std::runtime_error("unsupported joint version");
  if(!(in>>tag)||tag!="BIAS")throw std::runtime_error("missing joint bias");
  vector_in(in,bias_);
  if(!(in>>tag)||tag!="ORDER")throw std::runtime_error("missing joint order");
  vector_in(in,orientation_);
  for(size_t i=0;i<N;++i) {
   size_t idx=N;
   if(!(in>>tag>>idx)||tag!="P"||idx!=i)throw std::runtime_error("bad joint projection");
   vector_in(in,projection_[i]);
  }
  for(size_t i=0;i<N;++i) {
   size_t idx=N;
   if(!(in>>tag>>idx)||tag!="B"||idx!=i)throw std::runtime_error("bad byte head");
   vector_in(in,byte_heads_[i]);
  }
  size_t count=0;
  if(!(in>>tag>>count)||tag!="ROLES"||count>1000)
   throw std::runtime_error("bad joint role count");
  for(size_t i=0;i<count;++i) {
   std::string rel;double b=0;V v{};
   if(!(in>>tag>>rel>>b)||tag!="R"||!std::isfinite(b)||std::abs(b)>100)
    throw std::runtime_error("bad joint relation");
   key(rel);vector_in(in,v);
   if(!role_heads_.emplace(rel,v).second)throw std::runtime_error("duplicate joint relation");
   role_bias_[rel]=b;
  }
  if(!(in>>tag>>count)||tag!="SENSES"||count>100000)
   throw std::runtime_error("bad joint sense count");
  for(size_t i=0;i<count;++i) {
   uint32_t id=0;double b=0;V v{};
   if(!(in>>tag>>id>>b)||tag!="S"||!std::isfinite(b)||std::abs(b)>100)
    throw std::runtime_error("bad joint sense");
   vector_in(in,v);
   if(!sense_heads_.emplace(id,v).second)throw std::runtime_error("duplicate joint sense");
   sense_bias_[id]=b;
  }
  in>>std::ws;
  if(!in.eof())throw std::runtime_error("unexpected joint checkpoint fields");
 }
 public:
 explicit JointWaveSemanticLearner(UnifiedWaveLanguage& wave,const std::string& root):
  wave_(wave),filename_(root+".joint-wave"){
  for(size_t i=0;i<N;++i)projection_[i]=init("wave:"+std::to_string(i),0.008);
  orientation_=init("orientation",0.07);
  load();
 }
 // Copying a learner retains its existing Universe reference. Commit a
 // candidate's complete trainable readout only after a heldout DEV gate;
 // no NaN16 root or exact symbolic claim is changed by this operation.
 void adopt_readout_from(JointWaveSemanticLearner& candidate){
  if(&wave_!=&candidate.wave_||filename_!=candidate.filename_)
   throw std::invalid_argument("cannot adopt parameters from unrelated root");
  using std::swap;
  swap(projection_,candidate.projection_);
  swap(byte_heads_,candidate.byte_heads_);
  swap(bias_,candidate.bias_);
  swap(orientation_,candidate.orientation_);
  swap(role_heads_,candidate.role_heads_);
  swap(role_bias_,candidate.role_bias_);
  swap(sense_heads_,candidate.sense_heads_);
  swap(sense_bias_,candidate.sense_bias_);
  swap(byte_updates_,candidate.byte_updates_);
  swap(role_updates_,candidate.role_updates_);
  swap(sense_updates_,candidate.sense_updates_);
 }
 // Controlled semantic consolidation: keep supervised semantic heads and
 // role-only orientation, but line-search the genuinely shared projection
 // and bias toward the trained proposal. alpha=0 is a semantic-head-only
 // ablation, not language transfer. Byte readout weights never change.
 void blend_semantics_from(const JointWaveSemanticLearner& candidate,double alpha){
  if(&wave_!=&candidate.wave_||filename_!=candidate.filename_||
     byte_updates_!=candidate.byte_updates_)
   throw std::invalid_argument("cannot blend unrelated semantic state");
  if(!std::isfinite(alpha)||alpha<0||alpha>1)
   throw std::invalid_argument("shared semantic blend must be 0..1");
  for(size_t i=0;i<N;++i)
   for(size_t d=0;d<D;++d)
    projection_[i][d]+=alpha*(candidate.projection_[i][d]-projection_[i][d]);
  for(size_t d=0;d<D;++d)
   bias_[d]+=alpha*(candidate.bias_[d]-bias_[d]);
  // This orientation term appears in role inference, never next-byte
  // prediction; semantic-head updates therefore cannot harm byte CE.
  orientation_=candidate.orientation_;
  role_heads_=candidate.role_heads_;
  role_bias_=candidate.role_bias_;
  sense_heads_=candidate.sense_heads_;
  sense_bias_=candidate.sense_bias_;
  role_updates_=candidate.role_updates_;
  sense_updates_=candidate.sense_updates_;
 }
 uint64_t byte_updates()const{return byte_updates_;}
 uint64_t role_updates()const{return role_updates_;}
 uint64_t sense_updates()const{return sense_updates_;}
 double projection_norm()const {
  double total=0;
  for(const auto& v:projection_)for(double x:v)total+=x*x;
  return std::sqrt(total);
 }
 State state_for(const std::string& text)const {
  if(text.empty()||text.size()>4096)throw std::invalid_argument("joint sentence length");
  State s{};
  for(unsigned char c:text) {
   if(c==255)throw std::invalid_argument("reserved stream delimiter");
   wave_.observe(s,c);
  }
  return s;
 }
 BytePrediction predict_byte(const State& s)const{
  const auto base=wave_.predict(s);
  BytePrediction out{};
  if(!s.length){out.p=base.p;out.best=base.best;return out;}
  const auto a=encode(s);
  std::array<double,N> logits{};
  double max_logit=-std::numeric_limits<double>::infinity();
  for(size_t c=0;c<N;++c){
   logits[c]=safe_log(base.p[c])+dot(byte_heads_[c],a.h);
   max_logit=std::max(logits[c],max_logit);
  }
  double total=0;
  for(size_t c=0;c<N;++c){out.p[c]=std::exp(logits[c]-max_logit);total+=out.p[c];}
  for(size_t c=0;c<N;++c){
   out.p[c]/=total;
   if(out.p[c]>out.p[out.best])out.best=static_cast<uint8_t>(c);
  }
  return out;
 }
 double teach_byte(const State& s,uint8_t label,double lr) {
  lr_valid(lr);
  if(label>=N)throw std::invalid_argument("byte reserved for transport");
  const auto a=encode(s);
  const auto p=predict_byte(s);
  V dh{};
  for(size_t c=0;c<N;++c){
   const double delta=p.p[c]-double(c==label);
   for(size_t d=0;d<D;++d){
    dh[d]+=delta*byte_heads_[c][d];
    byte_heads_[c][d]-=lr*delta*a.h[d];
   }
  }
  adjust(a,dh,lr,0);
  ++byte_updates_;
  return -safe_log(p.p[label]);
 }
 double role_probability(const State& s,const Role& row)const {
  const std::string rel=lower(row.relation);
  const int direction=role_orientation(row);
  const auto it=role_heads_.find(rel);
  if(it==role_heads_.end())throw std::invalid_argument("untrained joint role");
  const auto a=encode(s,direction);
  return sigmoid(dot(it->second,a.h)+role_bias_.at(rel));
 }
 double teach_role(const State& s,const Role& row,double lr,bool update_shared=true) {
  lr_valid(lr);
  if(row.label!=0&&row.label!=1)throw std::invalid_argument("nonbinary role");
  const std::string rel=lower(row.relation);
  const int direction=role_orientation(row);
  if(!role_heads_.count(rel)) {
   if(role_heads_.size()>=1000)throw std::runtime_error("joint relation capacity");
   role_heads_.emplace(rel,init("role:"+rel,0.20));
   role_bias_[rel]=0;
  }
  auto& head=role_heads_.at(rel);
  const auto a=encode(s,direction);
  const double p=sigmoid(dot(head,a.h)+role_bias_.at(rel));
  const double delta=p-double(row.label);
  V dh{};
  for(size_t d=0;d<D;++d){
   dh[d]=delta*head[d];
   head[d]-=lr*delta*a.h[d];
  }
  role_bias_[rel]-=lr*delta;
  if(update_shared)adjust(a,dh,lr,direction);
  else for(size_t d=0;d<D;++d)
   orientation_[d]-=lr*std::clamp(dh[d]*(1-a.h[d]*a.h[d]),-5.,5.)*direction;
  ++role_updates_;
  return -(row.label?safe_log(p):safe_log(1-p));
 }
 std::optional<SensePrediction> predict_sense(const State& s,const Sense& row,
                                               const WordNetIndex& wn)const {
  check_target(row);
  const auto ids=candidates(wn,row.target);
  for(uint32_t id:ids)if(!sense_heads_.count(id))return std::nullopt;
  const auto a=encode(s);
  size_t best=0;
  double max_logit=-std::numeric_limits<double>::infinity();
  std::vector<double> logits;
  for(uint32_t id:ids) {
   const double logit=dot(sense_heads_.at(id),a.h)+sense_bias_.at(id);
   logits.push_back(logit);
   max_logit=std::max(logit,max_logit);
  }
  double total=0;
  for(double& p:logits){p=std::exp(p-max_logit);total+=p;}
  for(size_t i=1;i<logits.size();++i)if(logits[i]>logits[best])best=i;
  return SensePrediction{ids[best],logits[best]/total};
 }
 double teach_sense(const State& s,const Sense& row,const WordNetIndex& wn,double lr,bool update_shared=true){
  lr_valid(lr);check_target(row);
  const auto ids=candidates(wn,row.target);
  if(std::find(ids.begin(),ids.end(),row.gold)==ids.end())
   throw std::invalid_argument("gold synset is not WordNet candidate");
  const auto a=encode(s);
  std::vector<double> logits;
  double max_logit=-std::numeric_limits<double>::infinity();
  for(uint32_t id:ids){
   if(!sense_heads_.count(id)){
    if(sense_heads_.size()>=100000)throw std::runtime_error("joint sense capacity");
    sense_heads_[id]=init("sense:"+std::to_string(id),0.20);
    sense_bias_[id]=0;
   }
   const double v=dot(sense_heads_.at(id),a.h)+sense_bias_.at(id);
   logits.push_back(v);max_logit=std::max(v,max_logit);
  }
  double total=0;
  for(double& p:logits){p=std::exp(p-max_logit);total+=p;}
  V dh{};
  double gold_prob=0;
  for(size_t i=0;i<ids.size();++i) {
   const uint32_t id=ids[i];
   const double p=logits[i]/total;
   if(id==row.gold)gold_prob=p;
   const double delta=p-double(id==row.gold);
   auto& head=sense_heads_.at(id);
   for(size_t d=0;d<D;++d){
    dh[d]+=delta*head[d];
    head[d]-=lr*delta*a.h[d];
   }
   sense_bias_[id]-=lr*delta;
  }
  if(update_shared)adjust(a,dh,lr,0);
  ++sense_updates_;
  return -safe_log(gold_prob);
 }
 Metrics evaluate_bytes(const std::string& stream)const {
  State s{};Metrics m{};
  for(unsigned char ch:stream) {
   if(ch==255){s.reset();continue;}
   if(s.length){
    auto p=predict_byte(s);
    m.loss-=safe_log(p.p[ch]);
    m.correct+=(p.best==ch);
    ++m.count;
   }
   wave_.observe(s,ch);
  }
  return m;
 }
 Metrics evaluate_roles(const std::vector<Role>& rows)const {
  Metrics m{};
  for(const auto& row:rows) {
   const auto s=state_for(row.sentence);
   const double p=role_probability(s,row);
   m.loss-=row.label?safe_log(p):safe_log(1-p);
   m.correct+=(int(p>=0.5)==row.label);
   ++m.count;
  }
  return m;
 }
 Metrics evaluate_senses(const std::vector<Sense>& rows,const WordNetIndex& wn)const{
  Metrics m{};
  for(const auto& row:rows){
   const auto s=state_for(row.sentence);
   auto ids=candidates(wn,row.target);
   if(std::find(ids.begin(),ids.end(),row.gold)==ids.end())
    throw std::invalid_argument("gold sense absent");
   const auto a=encode(s);
   double maxlog=-std::numeric_limits<double>::infinity();
   std::vector<double> logits;
   for(uint32_t id:ids){
    const double v=dot(sense_heads_.at(id),a.h)+sense_bias_.at(id);
    logits.push_back(v);maxlog=std::max(maxlog,v);
   }
   double denom=0;
   for(double& v:logits){v=std::exp(v-maxlog);denom+=v;}
   const size_t gold=static_cast<size_t>(
     std::find(ids.begin(),ids.end(),row.gold)-ids.begin());
   const size_t best=static_cast<size_t>(
     std::max_element(logits.begin(),logits.end())-logits.begin());
   m.loss-=safe_log(logits[gold]/denom);
   m.correct+=best==gold;
   ++m.count;
  }
  return m;
 }
 // A semantic loss on the terminal state, evaluated without updating
 // weights. Used to test the semantic-to-recurrence gradient against a
 // finite-difference reference before attempting any physical root write.
 double sense_loss(const State& s,const Sense& row,const WordNetIndex& wn)const{
  check_target(row);
  const auto ids=candidates(wn,row.target);
  const auto a=encode(s);
  double largest=-std::numeric_limits<double>::infinity();
  std::vector<double> logits;
  for(uint32_t id:ids) {
   if(!sense_heads_.count(id))throw std::invalid_argument("untrained synset");
   const double v=dot(sense_heads_.at(id),a.h)+sense_bias_.at(id);
   logits.push_back(v);largest=std::max(largest,v);
  }
  double total=0,gold=0;
  for(size_t i=0;i<ids.size();++i){
   const double v=std::exp(logits[i]-largest);
   total+=v;
   if(ids[i]==row.gold)gold=v;
  }
  if(gold==0)throw std::invalid_argument("invalid supervised sense");
  return -safe_log(gold/total);
 }
 UnifiedWaveLanguage::WaveAdjoint sense_wave_adjoint(
     const State& s,const Sense& row,const WordNetIndex& wn)const{
  check_target(row);
  const auto ids=candidates(wn,row.target);
  const auto a=encode(s);
  std::vector<double> scores;
  double largest=-std::numeric_limits<double>::infinity();
  for(uint32_t id:ids){
   if(!sense_heads_.count(id))throw std::invalid_argument("untrained synset");
   const double v=dot(sense_heads_.at(id),a.h)+sense_bias_.at(id);
   scores.push_back(v);largest=std::max(largest,v);
  }
  double normalizer=0;
  for(double& v:scores){v=std::exp(v-largest);normalizer+=v;}
  V dh{};
  bool has_gold=false;
  for(size_t k=0;k<ids.size();++k){
   has_gold|=(ids[k]==row.gold);
   const double error=scores[k]/normalizer-double(ids[k]==row.gold);
   for(size_t d=0;d<D;++d)
    dh[d]+=error*sense_heads_.at(ids[k])[d]*(1.0-a.h[d]*a.h[d]);
  }
  if(!has_gold)throw std::invalid_argument("gold absent from synset candidates");
  UnifiedWaveLanguage::WaveAdjoint gradient{};
  for(size_t i=0;i<N;++i){
   double dx=0;
   for(size_t d=0;d<D;++d)dx+=dh[d]*projection_[i][d];
   // d tanh(.25*wave+.25*inverse) / d wave == /d inverse
   const double source=0.25*(1.0-a.x[i]*a.x[i])*dx;
   gradient.x[i]=source;
   gradient.y[i]=source;
  }
  return gradient;
 }
 double role_loss(const State& s,const Role& row)const{
  const double p=role_probability(s,row);
  if(row.label!=0&&row.label!=1)throw std::invalid_argument("bad role gold");
  return -(row.label?safe_log(p):safe_log(1-p));
 }
 UnifiedWaveLanguage::WaveAdjoint role_wave_adjoint(
     const State& s,const Role& row)const {
  if(row.label!=0&&row.label!=1)throw std::invalid_argument("bad role gold");
  const std::string relation=lower(row.relation);
  const int direction=role_orientation(row);
  const auto it=role_heads_.find(relation);
  if(it==role_heads_.end())throw std::invalid_argument("untrained relation");
  const auto a=encode(s,direction);
  const double error=sigmoid(dot(it->second,a.h)+role_bias_.at(relation))-
                     double(row.label);
  V dh{};
  for(size_t d=0;d<D;++d)
   dh[d]=error*it->second[d]*(1.0-a.h[d]*a.h[d]);
  UnifiedWaveLanguage::WaveAdjoint gradient{};
  for(size_t i=0;i<N;++i){
   double dx=0;
   for(size_t d=0;d<D;++d)dx+=dh[d]*projection_[i][d];
   const double source=0.25*(1.0-a.x[i]*a.x[i])*dx;
   gradient.x[i]=source;gradient.y[i]=source;
  }
  return gradient;
 }
 // Opt-in semantic-to-recurrent step. Use the *actual mapped root* and
 // strict, independent byte-development CE as an acceptance gate. We do
 // NOT accept an optimizer step merely because a surrogate gradient exists.
 // A failed proposal restores the exact pre-step register bytes and markers.
 template<class Loss>
 bool guarded_recurrent_step(Universe& root,const std::string& sentence,
                             const UnifiedWaveLanguage::WaveAdjoint& terminal,
                             Loss score,const std::string& byte_dev,
                             double allowed_byte_regression=0.) {
  if(root.path()+".joint-wave"!=filename_)
   throw std::invalid_argument("joint checkpoint/root mismatch");
  if(sentence.size()<8||sentence.size()>4096||byte_dev.size()<2||
     byte_dev.size()>100000||!std::isfinite(allowed_byte_regression)||
     allowed_byte_regression<0||allowed_byte_regression>0.1)
   throw std::invalid_argument("invalid guarded recurrent acceptance data");
  const double before_loss=score(state_for(sentence));
  const auto before_dev=evaluate_bytes(byte_dev);
  if(!std::isfinite(before_loss)||before_dev.count<1||
     !std::isfinite(before_dev.ce()))
   throw std::runtime_error("invalid guarded recurrent baseline");
  const std::string suffix=sentence.substr(sentence.size()-8);
  std::deque<State> states{state_for(sentence.substr(0,sentence.size()-8))};
  std::deque<uint8_t> symbols;
  for(unsigned char c:suffix){
   auto s=states.back();
   wave_.observe(s,c);
   states.push_back(std::move(s));
   symbols.push_back(c);
  }
  const auto gradient=wave_.semantic_wave_gradient(states,symbols,terminal);
  if(gradient.norm<1e-12)return false;
  struct Snapshot {
   uint8_t head=0;
   double gain=0;
   std::array<uint8_t,5> bytes{};
  };
  std::vector<Snapshot> saved;
  for(uint8_t symbol:symbols){
   if(std::any_of(saved.begin(),saved.end(),[symbol](const Snapshot& x){
       return x.head==symbol;}))continue;
   if(std::fabs(gradient.input[symbol])<1e-12)continue;
   Snapshot x{};
   x.head=symbol;
   x.gain=wave_.recurrent_input_gain(symbol);
   auto& data=root.root().h[symbol].data;
   std::copy_n(data+UnifiedWaveLanguage::BPTT_INPUT_OFFSET,4,x.bytes.begin());
   x.bytes[4]=data[UnifiedWaveLanguage::BPTT_MAGIC_OFFSET];
   saved.push_back(x);
  }
  if(saved.empty())return false;
  const auto rollback=[&](){
   for(const auto& x:saved){
    auto& data=root.root().h[x.head].data;
    std::copy_n(x.bytes.begin(),4,data+UnifiedWaveLanguage::BPTT_INPUT_OFFSET);
    data[UnifiedWaveLanguage::BPTT_MAGIC_OFFSET]=x.bytes[4];
   }
  };
  const auto journal_before=root.event_count();
  try{
   for(size_t retry=0;retry<10;++retry){
    rollback();
    const double rate=std::ldexp(0.08,-static_cast<int>(retry));
    bool changed=false;
    for(const auto& x:saved){
     const double proposal=std::clamp(
      x.gain-rate*gradient.input[x.head],0.6,1.4);
     if(std::abs(proposal-x.gain)<1e-8)continue;
     wave_.set_recurrent_input_gain(x.head,proposal);
     changed=true;
    }
    if(!changed)continue;
    const double after_loss=score(state_for(sentence));
    const double after_byte=evaluate_bytes(byte_dev).ce();
    if(std::isfinite(after_loss)&&std::isfinite(after_byte)&&
       after_loss+1e-10<before_loss&&
       after_byte<=before_dev.ce()+allowed_byte_regression+1e-12&&
       root.event_count()==journal_before){
     root.sync();
     return true;
    }
   }
  }catch(...){rollback();throw;}
  rollback();
  if(root.event_count()!=journal_before)
   throw std::logic_error("unexpected exact journal mutation");
  return false;
 }
 bool guarded_sense_recurrent_step(Universe& root,const Sense& row,
                                   const WordNetIndex& wn,
                                   const std::string& byte_dev,
                                   double allowed_byte_regression=0.) {
  const auto terminal=sense_wave_adjoint(state_for(row.sentence),row,wn);
  return guarded_recurrent_step(root,row.sentence,terminal,
   [&](const State& s){return sense_loss(s,row,wn);},
   byte_dev,allowed_byte_regression);
 }
 bool guarded_role_recurrent_step(Universe& root,const Role& row,
                                  const std::string& byte_dev,
                                  double allowed_byte_regression=0.) {
  const auto terminal=role_wave_adjoint(state_for(row.sentence),row);
  return guarded_recurrent_step(root,row.sentence,terminal,
   [&](const State& s){return role_loss(s,row);},
   byte_dev,allowed_byte_regression);
 }
 void save()const {
  std::ostringstream out;
  out<<std::setprecision(17);
  out<<"NJOINTWAVE1 "<<byte_updates_<<" "<<role_updates_<<" "<<sense_updates_<<"\n";
  out<<"BIAS";vector_out(out,bias_);
  out<<"ORDER";vector_out(out,orientation_);
  for(size_t i=0;i<N;++i){out<<"P "<<i;vector_out(out,projection_[i]);}
  for(size_t i=0;i<N;++i){out<<"B "<<i;vector_out(out,byte_heads_[i]);}
  out<<"ROLES "<<role_heads_.size()<<"\n";
  for(const auto& [name,v]:role_heads_) {
   out<<"R "<<name<<" "<<role_bias_.at(name);
   vector_out(out,v);
  }
  out<<"SENSES "<<sense_heads_.size()<<"\n";
  for(const auto& [id,v]:sense_heads_) {
   out<<"S "<<id<<" "<<sense_bias_.at(id);
   vector_out(out,v);
  }
  const std::string body=out.str(),tmp=filename_+".tmp";
  {
   std::ofstream file(tmp,std::ios::binary|std::ios::trunc);
   if(!file)throw std::runtime_error("joint checkpoint create failed");
   file<<body<<"CHECK "<<fnv(body)<<"\n";
   file.flush();
   if(!file)throw std::runtime_error("joint checkpoint write failed");
  }
  std::filesystem::rename(tmp,filename_);
 }
};
} // namespace nanlm
