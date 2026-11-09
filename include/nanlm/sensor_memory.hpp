#pragma once
// Experimental sensor-derived memory: 255 addressable head branches,
// 85 triadic aggregators, one global field, source-aware typed hyperedges.
// Lexical hashes route information, but DO NOT constitute learned semantics.
#include "nanlm/math_sensor_primitives.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <queue>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace nanlm::math_sensor {

inline std::uint64_t fingerprint(const std::string& s) {
 std::uint64_t h=14695981039346656037ULL;
 for(unsigned char c:s){h^=c;h*=1099511628211ULL;}
 return mix(h);
}
inline Hypervector role_vector(const std::string& text,std::uint64_t role) {
 return basis(mix(fingerprint(text)^role));
}
inline Hypervector claim_vector(const std::string& subject,const std::string& predicate,
                                const std::string& object) {
 return nanlm::math_sensor::bind(nanlm::math_sensor::bind(role_vector(subject,0x5355424a454354ULL),
                  role_vector(predicate,0x50524544494354ULL)),
             role_vector(object,0x4f424a454354ULL));
}
struct Assertion {
 std::string source,subject,predicate,object;
 bool refutes=false;
 double confidence=1.;
};
struct StatementKey {
 std::string subject,predicate,object;
 bool operator<(const StatementKey& rhs) const {
  return std::tie(subject,predicate,object)<std::tie(rhs.subject,rhs.predicate,rhs.object);
 }
};
struct SupportSet {
 std::set<std::string> supporting_sources,refuting_sources;
 unsigned state()const {return unsigned(!supporting_sources.empty())|
                               (unsigned(!refuting_sources.empty())<<1);}
};
struct FeatureLeaf {
 Hypervector vector{};
 std::array<double,9> tensor{};
 double support_charge=0,refute_charge=0;
 double curvature=0,activation=0;
 std::uint64_t last_tick=0;
 std::set<std::string> sources;
};
struct TriadAggregate {
 std::array<double,9> tensor{};
 double support=0,refute=0,activation=0,curvature=0;
};
struct GlobalAggregate {
 std::array<double,9> tensor{};
 double support=0,refute=0,activation=0;
 std::size_t unique_statements=0,unique_sources=0;
 std::uint64_t tick=0;
};
struct ReasoningResult {
 SupportSet evidence{};
 bool inferred=false;
 std::vector<std::string> proof_nodes;
 std::set<std::string> proof_sources;
 unsigned state()const{return evidence.state();}
};

class SensorMemory {
public:
 static constexpr std::size_t HEAD_COUNT=255,TRIAD_COUNT=85;
 using Branch=std::map<StatementKey,FeatureLeaf>;
 using CurvatureSampler=std::function<double(std::uint8_t,std::uint8_t,std::uint8_t)>;
private:
 std::array<Branch,HEAD_COUNT> head_branches_{};
 std::array<TriadAggregate,TRIAD_COUNT> triads_{};
 std::array<double,HEAD_COUNT> head_state_{};
 GlobalAggregate global_{};
 TriadMap map_=default_map();
 std::array<std::uint8_t,HEAD_COUNT> reverse_group_{};
 std::array<std::uint8_t,HEAD_COUNT> reverse_position_{};
 std::map<StatementKey,SupportSet> graph_{};
 std::vector<Assertion> journal_{};
 std::set<std::tuple<std::string,std::string,std::string,std::string,bool>> seen_{};
 std::set<std::string> sources_{};
 CurvatureSampler curvature_;
 std::size_t leaf_limit_,statement_limit_;
 double leak_=0.97;
 std::uint64_t clock_=0;
 static void check_text(const std::string& s) {
  if(s.empty()||s.size()>192||s.find_first_of("\t\r\n\0",0,4)!=std::string::npos)
   throw std::invalid_argument("empty, overlength or unsafe TSV field");
 }
 static void check_assertion(const Assertion& a) {
  check_text(a.source);check_text(a.subject);check_text(a.predicate);check_text(a.object);
  if(!std::isfinite(a.confidence)||a.confidence<=0||a.confidence>1)
   throw std::invalid_argument("confidence must be finite in (0,1]");
 }
 void map_positions() {
  std::array<bool,HEAD_COUNT> used{};
  for(std::size_t g=0;g<TRIAD_COUNT;++g)
   for(std::size_t k=0;k<3;++k){
    const auto h=map_[g][k];
    if(h>=HEAD_COUNT||used[h])throw std::invalid_argument("unbalanced triad mapping");
    used[h]=true;
    reverse_group_[h]=static_cast<std::uint8_t>(g);
    reverse_position_[h]=static_cast<std::uint8_t>(k);
   }
  if(reverse_group_[0]!=0)throw std::invalid_argument("head zero must remain group zero");
 }
 static std::array<std::uint8_t,6> route(const Hypervector& x) {
  std::vector<std::pair<std::size_t,std::uint8_t>> ranked;
  ranked.reserve(HEAD_COUNT);
  for(std::size_t h=0;h<HEAD_COUNT;++h)
   ranked.push_back({distance(x,basis(0x100000000000ULL+h)),static_cast<std::uint8_t>(h)});
  std::partial_sort(ranked.begin(),ranked.begin()+6,ranked.end());
  std::array<std::uint8_t,6> heads{};
  for(std::size_t k=0;k<heads.size();++k)heads[k]=ranked[k].second;
  return heads;
 }
 static std::array<double,3> features(const Hypervector& claim,std::uint8_t head,
                                      double tick_phase) {
  const double v=similarity(claim,basis(0x100000000000ULL+head));
  const double cs=std::cos(tick_phase),sn=std::sin(tick_phase);
  const double depth=1.-double(distance(claim,basis(0x200000000000ULL+head)))/256.;
  return {cs*v-sn*depth,sn*v+cs*depth,depth};
 }
 void recalc_global() {
  global_.tensor.fill(0);
  global_.support=global_.refute=global_.activation=0;
  for(const auto& t:triads_) {
   global_.support+=t.support;global_.refute+=t.refute;
   global_.activation+=t.activation;
   for(std::size_t i=0;i<9;++i)global_.tensor[i]+=t.tensor[i];
  }
  global_.support/=TRIAD_COUNT;global_.refute/=TRIAD_COUNT;
  global_.activation/=TRIAD_COUNT;
  for(auto& x:global_.tensor)x/=TRIAD_COUNT;
  global_.unique_statements=graph_.size();
  global_.unique_sources=sources_.size();
  global_.tick=clock_;
 }
 void evict(Branch& b) {
  if(b.size()<leaf_limit_)return;
  auto victim=b.begin();
  for(auto it=b.begin();it!=b.end();++it)
   if(it->second.last_tick<victim->second.last_tick)victim=it;
  b.erase(victim); // graph and journal retain exact source-bearing claims
 }
public:
 explicit SensorMemory(std::size_t leaves_per_head=128,
                       std::size_t statements=100000,double leak=0.97)
  :curvature_([](std::uint8_t a,std::uint8_t b,std::uint8_t c){return curvature(a,b,c);}),
   leaf_limit_(leaves_per_head),statement_limit_(statements),leak_(leak) {
  if(!leaf_limit_||!statement_limit_||!std::isfinite(leak)||leak<0||leak>=1)
   throw std::invalid_argument("invalid sensor bounds");
  map_positions();
 }
 void set_mapping(const TriadMap& map) {
  if(!journal_.empty())throw std::logic_error("remap before ingestion; or rebuild and replay");
  const auto old=map_;
  map_=map;
  try {map_positions();}
  catch(...) {map_=old;map_positions();throw;}
 }
 template<class NativeModel> void import_native_mapping(const NativeModel& model) {
  TriadMap map{};std::array<std::uint8_t,TRIAD_COUNT> next{};
  for(std::size_t h=0;h<HEAD_COUNT;++h){
   const auto g=model.head_triad(static_cast<std::uint8_t>(h));
   if(g>=TRIAD_COUNT||next[g]>=3)throw std::invalid_argument("bad native membership");
   map[g][next[g]++]=static_cast<std::uint8_t>(h);
  }
  set_mapping(map);
 }
 template<class NativeUniverse> void import_native_geometry(const NativeUniverse& universe) {
  curvature_=[&universe](std::uint8_t a,std::uint8_t b,std::uint8_t c){
   return universe.triangle(a,b,c);
  };
 }
 void tick() {
  ++clock_;
  for(auto& x:head_state_)x*=leak_;
  for(auto& t:triads_){
   t.support*=leak_;t.refute*=leak_;t.activation*=leak_;
   for(auto& x:t.tensor)x*=leak_;
  }
  recalc_global(); // exact graph and source provenance never decay on idle ticks
 }
 bool ingest(const Assertion& a) {
  check_assertion(a);
  auto id=std::make_tuple(a.source,a.subject,a.predicate,a.object,a.refutes);
  if(seen_.contains(id))return false;
  if(journal_.size()>=statement_limit_)throw std::length_error("journal budget exceeded");
  if(clock_==std::numeric_limits<std::uint64_t>::max())throw std::overflow_error("clock overflow");
  tick();
  seen_.insert(id);journal_.push_back(a);sources_.insert(a.source);
  const StatementKey key{a.subject,a.predicate,a.object};
  auto& e=graph_[key];
  (a.refutes?e.refuting_sources:e.supporting_sources).insert(a.source);
  const auto v=claim_vector(a.subject,a.predicate,a.object);
  const auto heads=route(v);
  std::array<bool,TRIAD_COUNT> touched{};
  const double angle=2.*PI*double(clock_%257)/257.;
  for(auto h:heads){
   auto& b=head_branches_[h];
   if(!b.contains(key))evict(b);
   auto& leaf=b[key];
   if(leaf.last_tick==0)leaf.vector=v;
   leaf.last_tick=clock_;leaf.sources.insert(a.source);
   const auto xyz=features(v,h,angle);
   const double strength=std::abs(xyz[0])+std::abs(xyz[1])+std::abs(xyz[2]);
   leaf.activation+=strength*a.confidence;
   head_state_[h]+=strength*a.confidence;
   if(a.refutes)leaf.refute_charge+=a.confidence;
   else leaf.support_charge+=a.confidence;
   for(std::size_t i=0;i<3;++i)
    for(std::size_t j=0;j<3;++j)
     leaf.tensor[3*i+j]+=a.confidence*xyz[i]*xyz[j];
   const auto g=reverse_group_[h];
   auto& t=triads_[g];touched[g]=true;
   t.activation+=strength*a.confidence;
   if(a.refutes)t.refute+=a.confidence;
   else t.support+=a.confidence;
   for(std::size_t i=0;i<9;++i)t.tensor[i]+=a.confidence*xyz[i/3]*xyz[i%3];
  }
  for(std::size_t g=0;g<TRIAD_COUNT;++g)if(touched[g]){
   const auto& heads3=map_[g];
   triads_[g].curvature=curvature_(heads3[0],heads3[1],heads3[2]);
  }
  recalc_global();
  return true;
 }
 SupportSet direct(const std::string& subject,const std::string& predicate,
                   const std::string& object)const {
  const auto it=graph_.find(StatementKey{subject,predicate,object});
  return it==graph_.end()?SupportSet{}:it->second;
 }
 // Restricted proof search: only positive ISA edges are transitive.
 // Refutations remain direct evidence; no arbitrary explosive entailment.
 ReasoningResult reason(const std::string& subject,const std::string& predicate,
                        const std::string& object,std::size_t max_hops=5)const {
  ReasoningResult out;out.evidence=direct(subject,predicate,object);
  if(predicate!="ISA"||max_hops<2||subject==object)return out;
  struct Entry {std::string node;std::vector<std::string> path;std::set<std::string> sources;};
  std::queue<Entry> pending;
  pending.push({subject,{subject},{}});
  std::map<std::string,std::size_t> visited{{subject,0}};
  std::size_t steps=0;
  while(!pending.empty()&&steps++<4096){
   auto current=std::move(pending.front());pending.pop();
   const auto depth=current.path.size()-1;
   if(depth>=max_hops)continue;
   for(const auto& [key,ev]:graph_) {
    if(key.predicate!="ISA"||key.subject!=current.node||ev.supporting_sources.empty())continue;
    if(std::find(current.path.begin(),current.path.end(),key.object)!=current.path.end())continue;
    auto path=current.path;path.push_back(key.object);
    auto proof=current.sources;proof.insert(*ev.supporting_sources.begin());
    if(key.object==object&&path.size()>=3){
     out.inferred=true;out.proof_nodes=std::move(path);out.proof_sources=std::move(proof);
     out.evidence.supporting_sources.insert(out.proof_sources.begin(),out.proof_sources.end());
     return out;
    }
    const auto it=visited.find(key.object);
    if(it==visited.end()||it->second>depth+1){
     visited[key.object]=depth+1;
     pending.push({key.object,std::move(path),std::move(proof)});
    }
   }
  }
  return out;
 }
 void save(const std::filesystem::path& path)const {
  const auto tmp=std::filesystem::path(path.string()+".tmp");
  std::ofstream file(tmp,std::ios::binary|std::ios::trunc);
  if(!file)throw std::runtime_error("cannot open sensor snapshot");
  file<<"NANLM_SENSOR_V1\n";
  for(const auto& a:journal_)
   file<<a.source<<'\t'<<(a.refutes?'-':'+')<<'\t'<<a.subject<<'\t'
       <<a.predicate<<'\t'<<a.object<<'\t'<<std::hexfloat<<a.confidence<<std::defaultfloat<<'\n';
  file.close();if(!file)throw std::runtime_error("sensor snapshot incomplete");
  std::filesystem::rename(tmp,path);
 }
 void load(const std::filesystem::path& path) {
  if(!journal_.empty())throw std::logic_error("load into a fresh sensor");
  std::ifstream file(path,std::ios::binary);
  if(!file)throw std::runtime_error("cannot read sensor snapshot");
  std::string line;
  if(!std::getline(file,line)||line!="NANLM_SENSOR_V1")
   throw std::runtime_error("unknown sensor snapshot schema");
  while(std::getline(file,line)){
   std::istringstream in(line);std::vector<std::string> fields;std::string token;
   while(std::getline(in,token,'\t'))fields.push_back(token);
   if(fields.size()!=6||(fields[1]!="+"&&fields[1]!="-"))
    throw std::runtime_error("malformed sensor record");
   std::size_t pos=0;
   const double conf=std::stod(fields[5],&pos);
   if(pos!=fields[5].size())throw std::runtime_error("bad confidence field");
   ingest({fields[0],fields[2],fields[3],fields[4],fields[1]=="-",conf});
  }
  if(!file.eof())throw std::runtime_error("incomplete sensor snapshot read");
 }
 std::size_t ingest_tsv(std::istream& text) {
  std::string line;std::size_t accepted=0;
  while(std::getline(text,line)){
   if(line.empty()||line[0]=='#')continue;
   std::istringstream input(line);
   std::vector<std::string> fields;std::string field;
   while(std::getline(input,field,'\t'))fields.push_back(field);
   if(fields.size()!=5||(fields[1]!="+"&&fields[1]!="-"))
    throw std::runtime_error("expected source TAB +/- TAB subject TAB RELATION TAB object");
   accepted+=ingest({fields[0],fields[2],fields[3],fields[4],fields[1]=="-",1.});
  }
  return accepted;
 }
 double head_activity(std::uint8_t h)const{return head_state_.at(h);}
 const Branch& branch(std::uint8_t h)const {return head_branches_.at(h);}
 const TriadAggregate& triad(std::size_t g)const {return triads_.at(g);}
 const GlobalAggregate& global()const {return global_;}
 std::size_t journal_size()const {return journal_.size();}
 std::uint64_t clock()const {return clock_;}
};
} // namespace nanlm::math_sensor
