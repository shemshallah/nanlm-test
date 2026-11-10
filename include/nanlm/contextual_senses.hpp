#pragma once
// Experimental exact-WordNet-synset supervision sharing the native contextual
// lexicon/position encoder with ContextualRoleLearner. This is a neural WSD
// probe, not symbolic proof, next-byte credit, or unsupervised language learning.
#include "nanlm/contextual_roles.hpp"
#include "nanlm/wordnet.hpp"
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <vector>

namespace nanlm {
class ContextualSenseLearner : public ContextualRoleLearner {
 public:
 struct Example {
  std::string sentence,target;
  uint32_t gold=0;
 };
 struct Prediction {
  uint32_t synset=0;
  double confidence=0;
 };
 struct Metrics {
  size_t rows=0,correct=0;
  double nll=0;
  double ce()const{return rows?nll/double(rows):0;}
  double accuracy()const{return rows?double(correct)/double(rows):0;}
 };
 private:
 std::string sense_file_;
 std::map<uint32_t,Vector> sense_heads_;
 std::map<uint32_t,double> sense_bias_;
 uint64_t sense_updates_=0;
 static std::string norm(std::string s) {
  for(char& c:s)if(c>='A'&&c<='Z')c=char(c-'A'+'a');
  return s;
 }
 static std::vector<uint32_t> candidates(const WordNetIndex& wn,
                                          const std::string& target) {
  auto ids=wn.lookup(norm(target));
  std::sort(ids.begin(),ids.end());
  ids.erase(std::unique(ids.begin(),ids.end()),ids.end());
  if(ids.size()<2||ids.size()>64)
   throw std::invalid_argument("sense supervision requires 2..64 exact WordNet candidate IDs");
  return ids;
 }
 static Feature context(const std::string& sentence,const std::string& target){
  const auto tokens=tokenize(sentence);
  const auto t=norm(target);key(t);
  size_t pos=tokens.size(),hits=0;
  for(size_t i=0;i<tokens.size();++i)
   if(tokens[i]==t){pos=i;++hits;}
  if(hits!=1)throw std::invalid_argument("word-sense target must occur exactly once");
  Feature f;
  f.orientation=0;
  for(size_t i=0;i<tokens.size();++i)
   if(i!=pos)f.words.emplace_back(tokens[i],i<pos?0u:3u);
  if(f.words.empty())throw std::invalid_argument("word-sense context is empty");
  f.normalization=1.0/std::sqrt(double(f.words.size()));
  return f;
 }
 Vector encode(const Feature& f,bool create) {
  Vector h=bias_;
  for(const auto& [word_key,slot]:f.words) {
   const auto it=words_.find(word_key);
   const Vector* e=nullptr;
   if(create)e=&word(word_key);
   else if(it!=words_.end())e=&it->second;
   if(e)for(size_t d=0;d<D;++d)
    h[d]+=f.normalization*position_[slot][d]*(*e)[d];
  }
  for(double& x:h)x=std::tanh(x);
  return h;
 }
 struct Output {
  Feature f;
  Vector h{};
  std::vector<uint32_t> ids;
  std::vector<double> prob;
 };
 Output forward_sense(const Example& example,const WordNetIndex& wn,bool create){
  Output o;
  o.ids=candidates(wn,example.target);
  o.f=context(example.sentence,example.target);
  o.h=encode(o.f,create);
  double max_logit=-std::numeric_limits<double>::infinity();
  for(uint32_t id:o.ids){
   auto it=sense_heads_.find(id);
   if(create&&it==sense_heads_.end()){
    sense_bias_.emplace(id,0.);
    it=sense_heads_.emplace(id,initialized("sense:"+std::to_string(id),0.30)).first;
   }
   if(it==sense_heads_.end())
    throw std::invalid_argument("untrained candidate synset");
   double logit=sense_bias_.at(id);
   for(size_t d=0;d<D;++d)logit+=it->second[d]*o.h[d];
   o.prob.push_back(logit);
   max_logit=std::max(max_logit,logit);
  }
  double total=0;
  for(double& p:o.prob){p=std::exp(p-max_logit);total+=p;}
  for(double& p:o.prob)p/=total;
  return o;
 }
 static void numeric(std::istream& s,Vector& v){
  for(double& x:v)if(!(s>>x)||!std::isfinite(x)||std::abs(x)>1e6)
   throw std::runtime_error("invalid synset parameters");
 }
 void load_senses(){
  std::ifstream in(sense_file_,std::ios::binary);
  if(!in){
   if(std::filesystem::exists(sense_file_))
    throw std::runtime_error("unreadable synset checkpoint");
   return;
  }
  const auto length=std::filesystem::file_size(sense_file_);
  if(length<35||length>8u*1024u*1024u)
   throw std::runtime_error("invalid synset checkpoint size");
  const std::string raw((std::istreambuf_iterator<char>(in)),{});
  const auto index=raw.rfind("CHECK ");
  if(index==std::string::npos||index==0||raw[index-1]!='\n')
   throw std::runtime_error("synset checksum missing");
  const auto body=raw.substr(0,index);
  std::istringstream hashline(raw.substr(index+6));
  uint64_t hashvalue=0;
  if(!(hashline>>hashvalue)||hashvalue!=fnv(body))
   throw std::runtime_error("synset checksum mismatch");
  hashline>>std::ws;
  if(!hashline.eof())throw std::runtime_error("extra synset checksum bytes");
  std::istringstream input(body);
  std::string tag;
  uint64_t base_steps=0;
  size_t count=0;
  if(!(input>>tag>>base_steps>>sense_updates_>>count)||
      tag!="NCTXSENSE1"||base_steps!=updates()||count>100000)
   throw std::runtime_error("synset checkpoint not aligned to contextual state");
  for(size_t i=0;i<count;++i){
   uint32_t id=0;double b=0;Vector v{};
   if(!(input>>tag>>id>>b)||tag!="S"||!std::isfinite(b)||std::abs(b)>1e6)
    throw std::runtime_error("invalid stored sense record");
   numeric(input,v);
   if(!sense_heads_.emplace(id,v).second)
    throw std::runtime_error("duplicate synset output head");
   sense_bias_[id]=b;
  }
  input>>std::ws;
  if(!input.eof())throw std::runtime_error("extra synset checkpoint rows");
 }
 public:
 explicit ContextualSenseLearner(Universe& u):
  ContextualRoleLearner(u),sense_file_(u.path()+".context-senses"){
  load_senses();
 }
 uint64_t sense_updates()const{return sense_updates_;}
 size_t trained_senses()const{return sense_heads_.size();}
 static Example parse_sense(const std::string& line){
  const auto a=line.find('\t');
  const auto b=a==std::string::npos?a:line.find('\t',a+1);
  if(a==std::string::npos||b==std::string::npos||
     line.find('\t',b+1)!=std::string::npos)
   throw std::invalid_argument("sense data requires sentence, target, exact synset ID");
  const auto id=line.substr(b+1);
  if(id.empty()||id.find_first_not_of("0123456789")!=std::string::npos)
   throw std::invalid_argument("bad gold synset ID");
  const auto raw=std::stoull(id);
  if(raw>UINT32_MAX)throw std::invalid_argument("gold synset overflow");
  return Example{line.substr(0,a),line.substr(a+1,b-a-1),static_cast<uint32_t>(raw)};
 }
 static std::vector<Example> read_senses(std::istream& input){
  std::vector<Example> result;
  std::string line;
  while(std::getline(input,line)){
   if(line.empty()||line.front()=='#')continue;
   if(result.size()>=100000)throw std::invalid_argument("too many sense examples");
   result.push_back(parse_sense(line));
  }
  if(!input.eof()||result.empty())throw std::runtime_error("missing or truncated sense corpus");
  return result;
 }
 std::optional<Prediction> infer(const std::string& sentence,
                                 const std::string& target,
                                 const WordNetIndex& wn) {
  const auto ids=candidates(wn,target);
  for(uint32_t id:ids)if(!sense_heads_.count(id))return std::nullopt;
  const auto o=forward_sense({sentence,target,0},wn,false);
  const size_t best=static_cast<size_t>(
    std::max_element(o.prob.begin(),o.prob.end())-o.prob.begin());
  return Prediction{o.ids[best],o.prob[best]};
 }
 Metrics evaluate_senses(const std::vector<Example>& rows,const WordNetIndex& wn){
  Metrics m;
  for(const auto& row:rows){
   const auto o=forward_sense(row,wn,false);
   const auto found=std::find(o.ids.begin(),o.ids.end(),row.gold);
   if(found==o.ids.end())throw std::invalid_argument("gold is not a candidate sense");
   const auto gold=static_cast<size_t>(found-o.ids.begin());
   const auto best=static_cast<size_t>(
     std::max_element(o.prob.begin(),o.prob.end())-o.prob.begin());
   m.nll-=std::log(std::max(o.prob[gold],1e-12));
   m.correct+=best==gold;
   ++m.rows;
  }
  return m;
 }
 void teach_sense(const Example& row,const WordNetIndex& wn,double lr){
  if(!std::isfinite(lr)||lr<=0||lr>0.25)
   throw std::invalid_argument("sense learning rate must be in (0,.25]");
  auto o=forward_sense(row,wn,true);
  if(std::find(o.ids.begin(),o.ids.end(),row.gold)==o.ids.end())
   throw std::invalid_argument("gold synset absent from WordNet candidates");
  Vector dh{};
  for(size_t i=0;i<o.ids.size();++i){
   const uint32_t id=o.ids[i];
   Vector& v=sense_heads_.at(id);
   const double delta=o.prob[i]-double(id==row.gold);
   for(size_t d=0;d<D;++d){
    dh[d]+=delta*v[d]*(1-o.h[d]*o.h[d]);
    v[d]-=lr*delta*o.h[d];
   }
   sense_bias_[id]-=lr*delta;
  }
  for(const auto& [token,slot]:o.f.words){
   Vector& e=words_.at(token);
   for(size_t d=0;d<D;++d){
    const double old_gate=position_[slot][d],old_e=e[d];
    e[d]-=lr*dh[d]*o.f.normalization*old_gate;
    position_[slot][d]-=lr*dh[d]*o.f.normalization*old_e;
   }
  }
  for(size_t d=0;d<D;++d)bias_[d]-=lr*dh[d];
  ++steps_;
  ++sense_updates_;
 }
 void train_senses(const std::vector<Example>& rows,const WordNetIndex& wn,
                   size_t epochs,double lr,uint64_t seed=20261109){
  if(rows.empty()||epochs==0||epochs>256)
   throw std::invalid_argument("invalid sense training schedule");
  std::mt19937_64 rng(seed);
  std::vector<size_t> order(rows.size());
  std::iota(order.begin(),order.end(),0);
  for(size_t epoch=0;epoch<epochs;++epoch){
   std::shuffle(order.begin(),order.end(),rng);
   for(size_t i:order)teach_sense(rows[i],wn,lr);
  }
 }
 void save_senses(){
  // Each sidecar is integrity checked; this two-file experimental write is
  // NOT an atomic checkpoint transaction across the entire NaNLM root.
  ContextualRoleLearner::save();
  std::ostringstream out;
  out<<std::setprecision(17);
  out<<"NCTXSENSE1 "<<updates()<<" "<<sense_updates_<<" "<<sense_heads_.size()<<"\n";
  for(const auto& [id,vec]:sense_heads_){
   out<<"S "<<id<<" "<<sense_bias_.at(id);
   for(double x:vec)out<<" "<<x;
   out<<"\n";
  }
  const auto body=out.str(),tmp=sense_file_+".tmp";
  {
   std::ofstream file(tmp,std::ios::binary|std::ios::trunc);
   if(!file)throw std::runtime_error("cannot write synset checkpoint");
   file<<body<<"CHECK "<<fnv(body)<<"\n";
   file.flush();
   if(!file)throw std::runtime_error("synset checkpoint flush failed");
  }
  std::filesystem::rename(tmp,sense_file_);
 }
};
} // namespace nanlm
