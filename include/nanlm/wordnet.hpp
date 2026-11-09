#pragma once
#include "nanlm/universe.hpp"
#include <cctype>
#include <deque>
#include <iomanip>
#include <sstream>
#include <unordered_set>
namespace nanlm {
// Synset IDs remain exact keys in a sparse overlay. The 256 heads are processors,
// NOT 256 concept buckets. A leaf can anchor arbitrarily many disambiguated synsets.
struct Concept {uint32_t offset=0;uint16_t leaf=0;std::string lemma,gloss;};
struct ConceptEdge {
 uint32_t src=0,dst=0;
 uint8_t type=0;
 uint64_t source=0;
};
class WordNetIndex {
 std::unordered_map<uint32_t,Concept> concepts_;
 std::vector<ConceptEdge> edges_;
 std::unordered_map<std::string,std::vector<uint32_t>> lemma_;
 std::unordered_map<uint16_t,std::vector<uint32_t>> leaf_;
 std::unordered_map<uint32_t,std::vector<size_t>> outgoing_;
 static uint64_t fnv(const std::string& s){
  uint64_t x=14695981039346656037ULL;
  for(unsigned char c:s){x^=c;x*=1099511628211ULL;}
  return x;
 }
 static std::string lower(std::string s){
  for(auto& c:s)c=static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return s;
 }
 static bool pointer_type(const std::string& p,uint8_t& type){
  if(p=="@"||p=="@i"){type=ISA;return true;}
  if(p=="~"||p=="~i"){type=HAS_SUBTYPE;return true;}
  if(p=="#p"||p=="#m"||p=="#s"){type=PART_OF;return true;}
  if(p=="%p"||p=="%m"||p=="%s"){type=HAS_PART;return true;}
  if(p=="!"){type=CONTRADICTS;return true;}
  if(p=="&"){type=ASSOCIATED;return true;}
  return false;
 }
 public:
 size_t concepts()const{return concepts_.size();}
 size_t relations()const{return edges_.size();}
 const Concept* concept(uint32_t id)const{
  auto it=concepts_.find(id);return it==concepts_.end()?nullptr:&it->second;
 }
 const std::vector<uint32_t>& concepts_at(uint16_t address)const{
  static const std::vector<uint32_t> empty;
  auto it=leaf_.find(address);return it==leaf_.end()?empty:it->second;
 }
 std::vector<uint32_t> lookup(const std::string& lemma)const{
  auto it=lemma_.find(lower(lemma));
  return it==lemma_.end()?std::vector<uint32_t>{}:it->second;
 }
 size_t import_noun_data(const std::string& filename,Universe& u,uint64_t asof=1){
  std::ifstream input(filename);
  if(!input)throw std::runtime_error("WordNet data.noun not found");
  concepts_.clear();edges_.clear();lemma_.clear();leaf_.clear();outgoing_.clear();
  struct Pending {uint32_t a,b;uint8_t type;};
  std::vector<Pending> pending;
  std::string line;
  while(std::getline(input,line)){
   if(line.empty()||!std::isdigit(static_cast<unsigned char>(line[0])))continue;
   const auto bar=line.find('|');
   const std::string left=line.substr(0,bar);
   const std::string gloss=bar==std::string::npos?"":line.substr(bar+1);
   std::istringstream ss(left);
   std::string offset,lex,pos,wcount;
   if(!(ss>>offset>>lex>>pos>>wcount)||pos!="n")continue;
   uint32_t id=0;size_t words=0;
   try {id=static_cast<uint32_t>(std::stoul(offset));words=std::stoul(wcount,nullptr,16);}
   catch(...){continue;}
   if(words==0||words>1000)continue;
   std::vector<std::string> lemmas;
   for(size_t i=0;i<words;++i){
    std::string word,lexid;
    if(!(ss>>word>>lexid))throw std::runtime_error("invalid WordNet lexeme");
    lemmas.push_back(word);
   }
   size_t count=0;
   if(!(ss>>count)||count>100000)throw std::runtime_error("invalid pointer count");
   for(size_t i=0;i<count;++i){
    std::string symbol,other,part,source_target;
    if(!(ss>>symbol>>other>>part>>source_target))throw std::runtime_error("invalid WordNet pointer");
    uint8_t type=0;
    if(part!="n"||!pointer_type(symbol,type))continue;
    uint32_t dest=0;
    try{dest=static_cast<uint32_t>(std::stoul(other));}catch(...){continue;}
    pending.push_back(Pending{id,dest,type});
   }
   const uint16_t addr=static_cast<uint16_t>(fnv("n:"+offset)&0xffff);
   concepts_[id]=Concept{id,addr,lemmas[0],gloss};
   leaf_[addr].push_back(id);
   for(const auto& name:lemmas)lemma_[lower(name)].push_back(id);
  }
  for(const auto& p:pending){
   auto ia=concepts_.find(p.a),ib=concepts_.find(p.b);
   if(ia==concepts_.end()||ib==concepts_.end())continue;
   edges_.push_back(ConceptEdge{p.a,p.b,p.type,fnv(std::to_string(p.a)+":"+std::to_string(p.b))});
   outgoing_[p.a].push_back(edges_.size()-1);
   uint8_t a=source(ia->second.leaf),b=target(ib->second.leaf);
   if(a==b)b=static_cast<uint8_t>(b+1); // colliding projection, exact endpoints stay in overlay
   u.write(a,b,p.type,false,1.0f,asof,0,edges_.back().source);
  }
  return edges_.size();
 }
 std::vector<uint32_t> path(uint32_t start,uint32_t goal,uint8_t type=ISA,size_t max_hops=16)const{
  if(!concepts_.count(start)||!concepts_.count(goal))return {};
  std::deque<uint32_t> q{start};
  std::unordered_map<uint32_t,uint32_t> prev{{start,start}};
  std::unordered_map<uint32_t,size_t> distance{{start,0}};
  while(!q.empty()){
   auto cur=q.front();q.pop_front();
   if(cur==goal){
    std::vector<uint32_t> result{goal};
    while(result.back()!=start)result.push_back(prev.at(result.back()));
    std::reverse(result.begin(),result.end());return result;
   }
   if(distance.at(cur)>=max_hops)continue;
   auto it=outgoing_.find(cur);
   if(it==outgoing_.end())continue;
   for(size_t j:it->second){
    const auto& e=edges_[j];
    if(e.type!=type||prev.count(e.dst))continue;
    prev[e.dst]=cur;distance[e.dst]=distance.at(cur)+1;q.push_back(e.dst);
   }
  }
  return {};
 }
 // Text export is deterministic and inspectable, unlike an implicit lookup table.
 void save(const std::string& filename)const{
  std::ofstream f(filename,std::ios::trunc);
  if(!f)throw std::runtime_error("wordnet export");
  std::vector<uint32_t> ids;
  for(const auto& [id,_]:concepts_)ids.push_back(id);
  std::sort(ids.begin(),ids.end());
  f<<"NANLM-WN1\t"<<ids.size()<<"\t"<<edges_.size()<<"\n";
  for(uint32_t id:ids){
   const auto& c=concepts_.at(id);
   std::string gloss=c.gloss;
   for(char& x:gloss)if(x=='\t'||x=='\n'||x=='\r')x=' ';
   f<<"N\t"<<id<<"\t"<<c.leaf<<"\t"<<c.lemma<<"\t"<<gloss<<"\n";
  }
  for(const auto& e:edges_)f<<"E\t"<<e.src<<"\t"<<e.dst<<"\t"<<int(e.type)<<"\n";
 }
};
} // namespace nanlm
