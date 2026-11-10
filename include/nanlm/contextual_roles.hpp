#pragma once
// Experimental supervised contextual-role learner. All parameters are updated
// by native sentence-level cross entropy, not by table-count interpolation.
// Bound to an existing NaNLM Universe root; exact concepts remain in WordNet.
#include "nanlm/universe.hpp"
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
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace nanlm {
class ContextualRoleLearner {
 friend class ContextualSenseLearner;
 public:
 static constexpr size_t D=24;
 using Vector=std::array<double,D>;
 struct Example {
  std::string sentence,subject,object,relation;
  int label=0;
 };
 struct Metrics {
  size_t queries=0,correct=0;
  double nll=0;
  double ce()const{return queries?nll/double(queries):0;}
  double accuracy()const{return queries?double(correct)/double(queries):0;}
 };
 private:
 Universe& root_;
 std::string filename_;
 std::map<std::string,Vector> words_,relations_;
 std::map<std::string,double> relation_bias_;
 std::array<Vector,4> position_{};
 Vector bias_{},order_{};
 uint64_t steps_=0;
 static uint64_t fnv(const std::string& s) {
  uint64_t h=14695981039346656037ull;
  for(unsigned char c:s){h^=c;h*=1099511628211ull;}
  return h;
 }
 static uint64_t next(uint64_t& state) {
  state+=0x9e3779b97f4a7c15ull;
  uint64_t z=state;
  z=(z^(z>>30))*0xbf58476d1ce4e5b9ull;
  z=(z^(z>>27))*0x94d049bb133111ebull;
  return z^(z>>31);
 }
 static double rand_symmetric(uint64_t& s) {
  return 2.0*double(next(s)>>11)*(1.0/9007199254740992.0)-1.0;
 }
 static std::string lower(std::string s) {
  for(char& c:s)if(c>='A'&&c<='Z')c=char(c-'A'+'a');
  return s;
 }
 static void key(const std::string& s) {
  if(s.empty()||s.size()>128)throw std::invalid_argument("invalid contextual identity length");
  for(unsigned char c:s)
   if(!((c>='a'&&c<='z')||(c>='0'&&c<='9')||c=='_'))
    throw std::invalid_argument("contextual identity must be one ASCII token");
 }
 static std::vector<std::string> tokenize(const std::string& s) {
  if(s.empty()||s.size()>4096)throw std::invalid_argument("sentence length must be 1..4096");
  std::vector<std::string> result;
  std::string w;
  for(unsigned char c:s) {
   if((c>='A'&&c<='Z')||(c>='a'&&c<='z')||
      (c>='0'&&c<='9')||c=='_') {
    if(w.size()>=128)throw std::invalid_argument("oversized token");
    w.push_back(c>='A'&&c<='Z'?char(c-'A'+'a'):char(c));
   }else if(!w.empty()){
    result.push_back(w);w.clear();
   }
  }
  if(!w.empty())result.push_back(w);
  if(result.size()>128||result.size()<3)throw std::invalid_argument("invalid sentence token count");
  return result;
 }
 static Vector initialized(const std::string& tag,double scale) {
  uint64_t seed=fnv(tag);
  Vector v{};
  for(double& x:v)x=scale*rand_symmetric(seed);
  return v;
 }
 Vector& word(const std::string& s) {
  auto it=words_.find(s);
  if(it!=words_.end())return it->second;
  if(words_.size()>=50000)throw std::runtime_error("vocabulary bound exceeded");
  return words_.emplace(s,initialized("token:"+s,0.35)).first->second;
 }
 Vector& relation(const std::string& s) {
  auto it=relations_.find(s);
  if(it!=relations_.end())return it->second;
  if(relations_.size()>=1000)throw std::runtime_error("relation count bound exceeded");
  relation_bias_.emplace(s,0.0);
  return relations_.emplace(s,initialized("relation:"+s,0.30)).first->second;
 }
 struct Feature {
  std::vector<std::pair<std::string,size_t>> words;
  double orientation=1;
  double normalization=1;
 };
 static Feature features(const Example& row) {
  const std::string a=lower(row.subject),b=lower(row.object);
  key(a);key(b);key(lower(row.relation));
  if(a==b)throw std::invalid_argument("subject and object must differ");
  const auto tokens=tokenize(row.sentence);
  size_t sa=tokens.size(),sb=tokens.size(),ac=0,bc=0;
  for(size_t i=0;i<tokens.size();++i) {
   if(tokens[i]==a){sa=i;++ac;}
   if(tokens[i]==b){sb=i;++bc;}
  }
  if(ac!=1||bc!=1)throw std::invalid_argument("both exact entities must occur once");
  Feature f;
  f.orientation=sa<sb?1.0:-1.0;
  for(size_t i=0;i<tokens.size();++i) {
   if(i==sa||i==sb)continue;
   const size_t slot=size_t(i>sa)+2u*size_t(i>sb);
   f.words.emplace_back(tokens[i],slot);
  }
  if(f.words.empty())throw std::invalid_argument("no lexical context");
  f.normalization=1.0/std::sqrt(double(f.words.size()));
  return f;
 }
 struct Activation {
  Feature feat;
  Vector h{};
  double probability=0.5;
 };
 static double sigmoid(double x) {
  if(x>=0){const double z=std::exp(-std::min(x,60.0));return 1.0/(1.0+z);}
  const double z=std::exp(std::max(x,-60.0));return z/(1.0+z);
 }
 Activation forward(const Example& row,bool create) {
  Activation a;
  a.feat=features(row);
  const auto r=relations_.find(lower(row.relation));
  if(r==relations_.end()&&!create)throw std::invalid_argument("untrained relation");
  const Vector& rel=create?relation(lower(row.relation)):r->second;
  for(size_t d=0;d<D;++d)a.h[d]=bias_[d]+order_[d]*a.feat.orientation;
  for(const auto& [token,slot]:a.feat.words) {
   const auto it=words_.find(token);
   const Vector* e=nullptr;
   if(create)e=&word(token);
   else if(it!=words_.end())e=&it->second;
   if(!e)continue;
   for(size_t d=0;d<D;++d)
    a.h[d]+=a.feat.normalization*position_[slot][d]*(*e)[d];
  }
  double score=relation_bias_.at(lower(row.relation));
  for(size_t d=0;d<D;++d) {
   a.h[d]=std::tanh(a.h[d]);
   score+=rel[d]*a.h[d];
  }
  a.probability=sigmoid(score);
  return a;
 }
 static void vector_out(std::ostream& out,const Vector& v) {
  for(double x:v)out<<" "<<x;
  out<<"\n";
 }
 static void vector_in(std::istream& in,Vector& v) {
  for(double& x:v) {
   if(!(in>>x)||!std::isfinite(x)||std::abs(x)>1e6)
    throw std::runtime_error("invalid contextual parameter");
  }
 }
 void load() {
  std::ifstream file(filename_,std::ios::binary);
  if(!file) {
   if(std::filesystem::exists(filename_))throw std::runtime_error("cannot read contextual checkpoint");
   return;
  }
  const auto len=std::filesystem::file_size(filename_);
  if(len<40||len>32u*1024u*1024u)throw std::runtime_error("invalid contextual checkpoint length");
  const std::string contents((std::istreambuf_iterator<char>(file)),{});
  const auto at=contents.rfind("CHECK ");
  if(at==std::string::npos||at==0||contents[at-1]!='\n')
   throw std::runtime_error("contextual checkpoint checksum absent");
  const std::string body=contents.substr(0,at);
  std::istringstream check(contents.substr(at+6));
  uint64_t digest=0;
  if(!(check>>digest)||digest!=fnv(body))
   throw std::runtime_error("contextual checkpoint checksum mismatch");
  check>>std::ws;
  if(!check.eof())throw std::runtime_error("contextual checkpoint trailing checksum");
  std::istringstream in(body);
  std::string tag;
  if(!(in>>tag>>steps_)||tag!="NCTXROLE1")
   throw std::runtime_error("unsupported contextual checkpoint");
  if(!(in>>tag)||tag!="B")throw std::runtime_error("contextual bias missing");
  vector_in(in,bias_);
  if(!(in>>tag)||tag!="O")throw std::runtime_error("contextual orientation missing");
  vector_in(in,order_);
  for(size_t i=0;i<4;++i) {
   size_t slot=4;
   if(!(in>>tag>>slot)||tag!="G"||slot!=i)throw std::runtime_error("contextual gate missing");
   vector_in(in,position_[i]);
  }
  size_t count=0;
  if(!(in>>tag>>count)||tag!="WORDS"||count>50000)
   throw std::runtime_error("contextual vocabulary count invalid");
  for(size_t i=0;i<count;++i){
   std::string s;Vector v{};
   if(!(in>>tag>>s)||tag!="T")throw std::runtime_error("contextual token missing");
   key(s);vector_in(in,v);
   if(!words_.emplace(s,v).second)throw std::runtime_error("duplicate contextual token");
  }
  if(!(in>>tag>>count)||tag!="RELATIONS"||count>1000)
   throw std::runtime_error("contextual relation count invalid");
  for(size_t i=0;i<count;++i) {
   std::string s;double b=0;Vector v{};
   if(!(in>>tag>>s>>b)||tag!="R"||!std::isfinite(b)||std::abs(b)>1e6)
    throw std::runtime_error("invalid contextual relation");
   key(s);vector_in(in,v);
   if(!relations_.emplace(s,v).second)throw std::runtime_error("duplicate contextual relation");
   relation_bias_[s]=b;
  }
  in>>std::ws;
  if(!in.eof())throw std::runtime_error("contextual checkpoint extra fields");
 }
 public:
 explicit ContextualRoleLearner(Universe& root):
  root_(root),filename_(root.path()+".context-role") {
  for(size_t s=0;s<4;++s)
   for(size_t d=0;d<D;++d)
    position_[s][d]=1.0+0.1*std::sin(double(11*s+3*d+1));
  order_=initialized("order",0.15);
  load();
 }
 uint64_t updates()const{return steps_;}
 size_t tokens()const{return words_.size();}
 size_t relations()const{return relations_.size();}
 static Example parse(const std::string& line) {
  std::vector<std::string> fields;
  size_t begin=0;
  for(size_t i=0;i<=line.size();++i)
   if(i==line.size()||line[i]=='\t'){
    fields.push_back(line.substr(begin,i-begin));begin=i+1;
   }
  if(fields.size()!=5)throw std::invalid_argument("context rows require sentence, subject, object, relation, 0|1");
  if(fields[4]!="0"&&fields[4]!="1")throw std::invalid_argument("bad contextual gold label");
  return Example{fields[0],fields[1],fields[2],fields[3],fields[4]=="1"?1:0};
 }
 static std::vector<Example> read(std::istream& in){
  std::vector<Example> result;
  std::string line;
  while(std::getline(in,line)){
   if(line.empty()||line[0]=='#')continue;
   if(result.size()>=100000)throw std::invalid_argument("context corpus too large");
   result.push_back(parse(line));
  }
  if(!in.eof()||result.empty())throw std::runtime_error("context corpus empty or truncated");
  return result;
 }
 std::optional<double> predict(const Example& row) {
  if(relations_.find(lower(row.relation))==relations_.end())return std::nullopt;
  return forward(row,false).probability;
 }
 Metrics evaluate(const std::vector<Example>& rows) {
  Metrics m;
  for(const auto& row:rows) {
   if(row.label!=0&&row.label!=1)throw std::invalid_argument("label must be binary");
   const auto p=predict(row);
   if(!p)throw std::invalid_argument("held-out relation has no train support");
   const double q=std::clamp(*p,1e-12,1.0-1e-12);
   m.nll-=row.label?std::log(q):std::log(1-q);
   m.correct+=int(q>=0.5)==row.label;
   ++m.queries;
  }
  return m;
 }
 void teach(const Example& row,double lr) {
  if(row.label!=0&&row.label!=1)throw std::invalid_argument("label must be binary");
  if(!std::isfinite(lr)||lr<=0||lr>0.25)throw std::invalid_argument("learning rate outside (0,.25]");
  const auto a=forward(row,true);
  const std::string rel=lower(row.relation);
  Vector& output=relations_.at(rel);
  const double delta=a.probability-double(row.label);
  Vector dh{};
  for(size_t d=0;d<D;++d) {
   dh[d]=delta*output[d]*(1-a.h[d]*a.h[d]);
   output[d]-=lr*delta*a.h[d];
  }
  relation_bias_.at(rel)-=lr*delta;
  for(const auto& [token,slot]:a.feat.words) {
   Vector& e=words_.at(token);
   for(size_t d=0;d<D;++d) {
    const double old_e=e[d],old_gate=position_[slot][d];
    e[d]-=lr*dh[d]*a.feat.normalization*old_gate;
    position_[slot][d]-=lr*dh[d]*a.feat.normalization*old_e;
   }
  }
  for(size_t d=0;d<D;++d) {
   bias_[d]-=lr*dh[d];
   order_[d]-=lr*dh[d]*a.feat.orientation;
  }
  ++steps_;
 }
 void train(const std::vector<Example>& rows,size_t epochs,double lr,uint64_t seed=20261109) {
  if(epochs==0||epochs>256||rows.empty())throw std::invalid_argument("bad contextual epoch count");
  std::mt19937_64 rng(seed);
  std::vector<size_t> order(rows.size());
  for(size_t i=0;i<order.size();++i)order[i]=i;
  for(size_t epoch=0;epoch<epochs;++epoch) {
   std::shuffle(order.begin(),order.end(),rng);
   for(size_t j:order)teach(rows[j],lr);
  }
 }
 void save() {
  std::ostringstream out;
  out<<std::setprecision(17);
  out<<"NCTXROLE1 "<<steps_<<"\n";
  out<<"B";vector_out(out,bias_);
  out<<"O";vector_out(out,order_);
  for(size_t i=0;i<4;++i){out<<"G "<<i;vector_out(out,position_[i]);}
  out<<"WORDS "<<words_.size()<<"\n";
  for(const auto& [name,embedding]:words_){out<<"T "<<name;vector_out(out,embedding);}
  out<<"RELATIONS "<<relations_.size()<<"\n";
  for(const auto& [name,embedding]:relations_){
   out<<"R "<<name<<" "<<relation_bias_.at(name);vector_out(out,embedding);
  }
  const auto body=out.str(),tmp=filename_+".tmp";
  {
   std::ofstream file(tmp,std::ios::binary|std::ios::trunc);
   if(!file)throw std::runtime_error("cannot create contextual checkpoint");
   file<<body<<"CHECK "<<fnv(body)<<"\n";
   file.flush();
   if(!file)throw std::runtime_error("cannot flush contextual checkpoint");
  }
  std::filesystem::rename(tmp,filename_);
  root_.sync();
 }
};
} // namespace nanlm
