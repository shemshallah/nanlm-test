#pragma once
// Native, bounded semantic grounding within the existing NaNLM Universe.
// Exact synset/predicate/source identities stay in a sparse sidecar; the NaN16
// projection is lossy and must never be mistaken for exact identity.
#include "nanlm/wordnet.hpp"
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace nanlm {
class SemanticBridge {
 public:
 struct Claim {
  bool isa=false, positive=false;
  uint32_t subject=0, object=0;
  std::string action, source;
 };
 struct Answer {
  uint8_t state=0; // 0 unknown, 1 support, 2 refute, 3 contested
  size_t depth=0;
  std::vector<uint32_t> path;
  std::vector<std::string> sources;
 };
 private:
 Universe& u_;
 const WordNetIndex& wn_;
 std::string ledger_;
 std::vector<Claim> facts_;
 std::unordered_map<uint32_t,std::vector<uint32_t>> parents_;
 static std::string lower(std::string s) {
  for(auto& c:s)c=static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return s;
 }
 static uint64_t hash(const std::string& s) {
  uint64_t h=14695981039346656037ULL;
  for(unsigned char c:s){h^=c;h*=1099511628211ULL;}
  return h;
 }
 static void validate_field(const std::string& s) {
  if(s.empty()||s.size()>256||s.find_first_of("\r\n\t")!=std::string::npos)
   throw std::invalid_argument("bad semantic field");
 }
 static std::string action(std::string s) {
  s=lower(s);validate_field(s);
  for(unsigned char c:s)
   if(!std::isalnum(c)&&c!='_'&&c!='-')
    throw std::invalid_argument("action is not a canonical single word");
  return s;
 }
 static std::vector<std::string> split(const std::string& line) {
  std::vector<std::string> fields;
  std::istringstream in(line);std::string p;
  while(std::getline(in,p,'\t'))fields.push_back(p);
  return fields;
 }
 static bool article(const std::string& t) {
  return t=="a"||t=="an"||t=="the";
 }
 uint32_t resolve(std::string lemma)const {
  lemma=lower(lemma);
  if(lemma.rfind("n:",0)==0) {
   const std::string d=lemma.substr(2);
   if(d.empty()||d.find_first_not_of("0123456789")!=std::string::npos)
    throw std::invalid_argument("bad explicit synset ID");
   const auto raw=std::stoull(d);
   if(raw>UINT32_MAX||!wn_.get_concept(static_cast<uint32_t>(raw)))
    throw std::invalid_argument("unknown synset ID");
   return static_cast<uint32_t>(raw);
  }
  const auto ids=wn_.lookup(lemma);
  if(ids.size()!=1)
   throw std::invalid_argument("missing or ambiguous lemma, use n:ID: "+lemma);
  return ids.front();
 }
 void insert(const Claim& f,bool project) {
  if(!wn_.get_concept(f.subject)||(f.isa&&!wn_.get_concept(f.object)))
   throw std::invalid_argument("fact requires exact known synset");
  validate_field(f.source);
  if(!f.isa)action(f.action);
  for(const auto& x:facts_)
   if(x.isa==f.isa&&x.positive==f.positive&&x.subject==f.subject
      &&x.object==f.object&&x.action==f.action&&x.source==f.source)return;
  if(project) {
   const auto* s=wn_.get_concept(f.subject);
   uint8_t a=static_cast<uint8_t>(source(s->leaf)%255u),b=0;
   if(f.isa) {
    const auto* o=wn_.get_concept(f.object);
    b=static_cast<uint8_t>(target(o->leaf)%255u);
   }else b=static_cast<uint8_t>(hash(f.action)%255u);
   if(a==b)b=static_cast<uint8_t>((unsigned(b)+1u)%255u);
   u_.write(a,b,f.isa?ISA:SUPPORTS,!f.positive,1.0f,1,0,
     hash(f.source+":"+std::to_string(f.subject)+":"+
          (f.isa?std::to_string(f.object):f.action)));
  }
  facts_.push_back(f);
  if(f.isa&&f.positive)parents_[f.subject].push_back(f.object);
 }
 std::unordered_map<uint32_t,std::vector<uint32_t>>
 ancestor_paths(uint32_t start,size_t max_hops=16)const {
  std::unordered_map<uint32_t,std::vector<uint32_t>> paths;
  paths[start]={start};std::deque<uint32_t> q{start};
  while(!q.empty()) {
   uint32_t here=q.front();q.pop_front();
   if(paths.at(here).size()>max_hops)continue;
   auto it=parents_.find(here);
   if(it==parents_.end())continue;
   for(uint32_t next:it->second) {
    if(paths.count(next))continue;
    auto p=paths.at(here);p.push_back(next);
    paths[next]=std::move(p);q.push_back(next);
   }
  }
  return paths;
 }
 void load() {
  std::ifstream in(ledger_);
  if(!in)return;
  std::string line;
  if(!std::getline(in,line)||line!="NANLM-SEM1")
   throw std::runtime_error("invalid semantic ledger header");
  size_t lines=0;
  while(std::getline(in,line)) {
   if(line.empty())continue;
   auto t=split(line);
   if(t.size()!=5||(t[0]!="ISA"&&t[0]!="ABLE")||
      (t[3]!="+"&&t[3]!="-")||++lines>20000000)
    throw std::runtime_error("invalid semantic ledger row");
   const auto parse=[](const std::string& s) {
    if(s.empty()||s.find_first_not_of("0123456789")!=std::string::npos)
     throw std::runtime_error("invalid semantic synset");
    const auto n=std::stoull(s);
    if(n>UINT32_MAX)throw std::runtime_error("semantic synset overflow");
    return static_cast<uint32_t>(n);
   };
   Claim f{};f.isa=t[0]=="ISA";f.subject=parse(t[1]);
   f.positive=t[3]=="+";f.source=t[4];
   if(f.isa)f.object=parse(t[2]);else f.action=action(t[2]);
   insert(f,false);
  }
  if(!in.eof())throw std::runtime_error("truncated semantic ledger");
 }
 public:
 SemanticBridge(Universe& u,const WordNetIndex& wn):
  u_(u),wn_(wn),ledger_(u.path()+".semantic.tsv") {
  for(const auto& edge:wn_.edges())
   if(edge.type==ISA)parents_[edge.src].push_back(edge.dst);
  load();
 }
 size_t claims()const{return facts_.size();}
 void assert_isa(const std::string& a,const std::string& b,bool yes,
                 const std::string& provenance) {
  Claim f{};f.isa=true;f.subject=resolve(a);f.object=resolve(b);
  f.positive=yes;f.source=provenance;insert(f,true);
 }
 void assert_ability(const std::string& a,const std::string& b,bool yes,
                     const std::string& provenance) {
  Claim f{};f.subject=resolve(a);f.action=action(b);
  f.positive=yes;f.source=provenance;insert(f,true);
 }
 void ingest(std::string sentence,const std::string& provenance) {
  sentence=lower(sentence);
  while(!sentence.empty()&&std::isspace(static_cast<unsigned char>(sentence.back())))
   sentence.pop_back();
  if(!sentence.empty()&&sentence.back()=='.')sentence.pop_back();
  std::istringstream in(sentence);std::vector<std::string> t;std::string word;
  while(in>>word)t.push_back(word);
  if(!t.empty()&&article(t.front()))t.erase(t.begin());
  if(t.size()==3&&t[1]=="can")return assert_ability(t[0],t[2],true,provenance);
  if(t.size()==3&&t[1]=="cannot")return assert_ability(t[0],t[2],false,provenance);
  if(t.size()==4&&t[1]=="is"&&article(t[2]))
   return assert_isa(t[0],t[3],true,provenance);
  if(t.size()==5&&t[1]=="is"&&t[2]=="not"&&article(t[3]))
   return assert_isa(t[0],t[4],false,provenance);
  throw std::invalid_argument("unsupported controlled semantic sentence");
 }
 Answer query_isa(const std::string& subject,const std::string& object)const {
  const uint32_t a=resolve(subject),b=resolve(object);
  Answer ans{};auto paths=ancestor_paths(a);
  if(auto it=paths.find(b);it!=paths.end()) {
   ans.state|=1;ans.depth=it->second.size()-1;ans.path=it->second;
   ans.sources.push_back("WordNet_or_supervised_ISA_path");
  }
  for(const auto& f:facts_)
   if(f.isa&&!f.positive&&f.subject==a&&f.object==b) {
    ans.state|=2;ans.sources.push_back(f.source);
   }
  return ans;
 }
 Answer query_ability(const std::string& subject,const std::string& verb)const {
  const uint32_t a=resolve(subject);
  const auto verb_key=action(verb);
  auto paths=ancestor_paths(a);
  Answer ans{};size_t best=std::numeric_limits<size_t>::max();
  for(const auto& f:facts_) {
   if(f.isa||f.action!=verb_key)continue;
   auto it=paths.find(f.subject);
   if(it==paths.end())continue;
   const auto dist=it->second.size()-1;
   if(dist>best)continue;
   if(dist<best) {
    best=dist;ans={};ans.depth=dist;ans.path=it->second;
   }
   ans.state|=f.positive?1u:2u;
   ans.sources.push_back(f.source);
  }
  return ans;
 }
 void save()const {
  const auto tmp=ledger_+".tmp";
  {
   std::ofstream out(tmp,std::ios::trunc);
   if(!out)throw std::runtime_error("semantic ledger open failed");
   out<<"NANLM-SEM1\n";
   for(const auto& f:facts_)
    out<<(f.isa?"ISA":"ABLE")<<"\t"<<f.subject<<"\t"
       <<(f.isa?std::to_string(f.object):f.action)<<"\t"
       <<(f.positive?"+":"-")<<"\t"<<f.source<<"\n";
   out.flush();
   if(!out)throw std::runtime_error("semantic ledger flush failed");
  }
  std::filesystem::rename(tmp,ledger_);
  u_.sync();
 }
};
} // namespace nanlm
