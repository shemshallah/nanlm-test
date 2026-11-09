#pragma once
#include "nanlm/utf8.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace nanlm {

// Explicit, source-attributed EXACT byte memory. This is a bounded persisted
// retrieval index attached to the SAME model/root, not a second predictor.
// No evaluation/generation path writes records; only explicit import does.
// A matching user-specified cue enables a deterministic byte-copy continuation.
class ExactEpisodicMemory {
public:
 static constexpr size_t MAX_KEY=64, MAX_VALUE=4096, MAX_SOURCE=128;
 static constexpr size_t MAX_RECORDS=4096, MAX_FILE_BYTES=24u*1024u*1024u;
 static constexpr uint32_t NONE=std::numeric_limits<uint32_t>::max();
 static constexpr size_t PHASE_BANDS=4;
 using PhaseCode=std::array<double,2*PHASE_BANDS>;
 struct Record {
  std::string cue,value,source;
  // Derived from the cue; never trusted as evidence and not serialized.
  PhaseCode phase{};
 };
 struct Lookup {
  uint32_t record=NONE;
  size_t routed=0,exact_compares=0;
  double resonance=0;
 };
 struct Suggestion {
  std::string cue,source;
  double resonance=0;
  bool exact_verified=false;
 };
private:
 std::string path_;
 std::vector<Record> entries_;
 // Derived addressing, never serialized; rebuilding it cannot modify
 // the 8MiB root or its source-attributed episodic records.
 std::array<std::unordered_map<uint64_t,std::vector<size_t>>,MAX_KEY+1>
     suffix_index_{};
 std::unordered_map<std::string,std::vector<size_t>> cue_index_;
 std::array<bool,MAX_KEY+1> lengths_{};
 static uint64_t reversed_hash(std::string_view bytes){
  uint64_t h=14695981039346656037ull;
  for(auto it=bytes.rbegin();it!=bytes.rend();++it){
   h^=static_cast<uint8_t>(*it);
   h*=1099511628211ull;
  }
  return h;
 }
 void rebuild_index(){
  cue_index_.clear();lengths_.fill(false);
  for(auto& m:suffix_index_)m.clear();
  for(size_t i=0;i<entries_.size();++i){
   const auto& r=entries_[i];
   const size_t n=r.cue.size();
   lengths_[n]=true;
   suffix_index_[n][reversed_hash(r.cue)].push_back(i);
   cue_index_[r.cue].push_back(i);
  }
 }
 Lookup probe_hash(const uint8_t* history,size_t length)const{
  Lookup out{};
  if(!length||history[length-1]!=static_cast<uint8_t>(' '))return out;
  const size_t limit=std::min(length,MAX_KEY);
  std::array<uint64_t,MAX_KEY+1> hashes{};
  uint64_t h=14695981039346656037ull;
  for(size_t n=1;n<=limit;++n){
   h^=history[length-n];h*=1099511628211ull;
   hashes[n]=h;
  }
  for(size_t n=limit;n>=6;--n){
   if(!lengths_[n])continue;
   auto it=suffix_index_[n].find(hashes[n]);
   if(it==suffix_index_[n].end())continue;
   out.routed+=it->second.size();
   for(size_t i:it->second){
    const std::string& cue=entries_[i].cue;
    ++out.exact_compares;
    bool match=true;
    for(size_t j=0;j<n;++j)
     if(history[length-n+j]!=static_cast<uint8_t>(cue[j])){
      match=false;break;
     }
    if(!match)continue; // 64-bit hash is NOT a proof.
    const Record* candidate=exact(cue);
    if(candidate)
     out.record=static_cast<uint32_t>(candidate-entries_.data());
    return out; // Preserve conflict abstention.
   }
  }
  return out;
 }
 static PhaseCode phase_code(std::string_view bytes){
  // Damped fixed-frequency two-channel oscillators approximate a cue's
  // temporal pattern; only byte equality can authorize copying.
  constexpr std::array<double,PHASE_BANDS> frequency{
   0.238,0.427,0.731,1.149
  };
  constexpr std::array<double,PHASE_BANDS> damping{
   0.912,0.956,0.977,0.992
  };
  PhaseCode result{};
  for(size_t i=0;i<bytes.size();++i){
   const double lag=double(bytes.size()-1-i);
   const double excitation=
      (double(static_cast<uint8_t>(bytes[i]))-127.5)/127.5;
   for(size_t band=0;band<PHASE_BANDS;++band){
    const double amplitude=excitation*std::pow(damping[band],lag);
    result[2*band]+=amplitude*std::cos(frequency[band]*lag);
    result[2*band+1]+=amplitude*std::sin(frequency[band]*lag);
   }
  }
  double magnitude=0;
  for(double x:result)magnitude+=x*x;
  magnitude=std::sqrt(magnitude);
  if(magnitude>1e-14)for(double& x:result)x/=magnitude;
  return result;
 }
 static double similarity(const PhaseCode& a,const PhaseCode& b){
  double score=0;
  for(size_t i=0;i<a.size();++i)score+=a[i]*b[i];
  return score;
 }
 static constexpr std::array<char,8> MAGIC{{'N','A','N','E','P','0','1','\0'}};
 static uint64_t hash(const Record& r){
  uint64_t h=14695981039346656037ull;
  const auto part=[&](const std::string& s){
   for(unsigned char b:s){h^=b;h*=1099511628211ull;}
   h^=0xffu;h*=1099511628211ull;
  };
  part(r.cue);part(r.value);part(r.source);
  return h;
 }
 static void validate(const Record& r){
  if(r.cue.size()<6||r.cue.size()>MAX_KEY||
     r.cue.back()!=' '||r.value.empty()||r.value.size()>MAX_VALUE||
     r.source.empty()||r.source.size()>MAX_SOURCE||
     !valid_utf8(r.cue)||!valid_utf8(r.value)||!valid_utf8(r.source))
   throw std::invalid_argument("episodic record requires UTF-8 cue ending in a space, nonempty UTF-8 value and source");
  if(r.cue.find('\n')!=std::string::npos||
     r.cue.find('\r')!=std::string::npos||
     r.cue.find('\t')!=std::string::npos||
     r.value.find('\n')!=std::string::npos||
     r.value.find('\r')!=std::string::npos||
     r.source.find('\n')!=std::string::npos||
     r.source.find('\r')!=std::string::npos)
   throw std::invalid_argument("episodic record contains forbidden control separator");
 }
 static void write32(std::ostream& out,uint32_t v){
  for(unsigned i=0;i<4;++i)out.put(static_cast<char>((v>>(8*i))&255u));
 }
 static void write64(std::ostream& out,uint64_t v){
  for(unsigned i=0;i<8;++i)out.put(static_cast<char>((v>>(8*i))&255u));
 }
 static uint64_t read_integer(std::istream& in,unsigned bytes){
  uint64_t v=0;
  for(unsigned i=0;i<bytes;++i){
   const int c=in.get();
   if(c==std::char_traits<char>::eof())
    throw std::runtime_error("truncated episodic memory");
   v|=uint64_t(static_cast<unsigned>(c))<<(8*i);
  }
  return v;
 }
 static std::string read_string(std::istream& in,size_t size){
  std::string value(size,'\0');
  if(size&&!in.read(value.data(),static_cast<std::streamsize>(size)))
   throw std::runtime_error("truncated episodic record");
  return value;
 }
 static void upsert(std::vector<Record>& data,Record r){
  validate(r);
  r.phase=phase_code(r.cue);
  auto it=std::find_if(data.begin(),data.end(),
                       [&](const Record& x){
                         return x.cue==r.cue&&x.source==r.source;
                       });
  if(it!=data.end())*it=std::move(r);
  else {
   if(data.size()>=MAX_RECORDS)throw std::runtime_error("episodic index full");
   data.push_back(std::move(r));
  }
 }
 void save(const std::vector<Record>& next)const{
  const std::string temporary=path_+".tmp";
  try{
   std::ofstream out(temporary,std::ios::binary|std::ios::trunc);
   if(!out)throw std::runtime_error("cannot create episodic checkpoint");
   out.write(MAGIC.data(),static_cast<std::streamsize>(MAGIC.size()));
   write32(out,static_cast<uint32_t>(next.size()));
   for(const Record& r:next){
    validate(r);
    write32(out,static_cast<uint32_t>(r.cue.size()));
    write32(out,static_cast<uint32_t>(r.value.size()));
    write32(out,static_cast<uint32_t>(r.source.size()));
    write64(out,hash(r));
    out.write(r.cue.data(),static_cast<std::streamsize>(r.cue.size()));
    out.write(r.value.data(),static_cast<std::streamsize>(r.value.size()));
    out.write(r.source.data(),static_cast<std::streamsize>(r.source.size()));
   }
   out.flush();
   if(!out)throw std::runtime_error("episodic checkpoint write failed");
   out.close();
   if(!out)throw std::runtime_error("episodic checkpoint close failed");
   if(std::filesystem::file_size(temporary)>MAX_FILE_BYTES)
    throw std::runtime_error("episodic index exceeds storage cap");
   std::filesystem::rename(temporary,path_);
  }catch(...){
   std::error_code ec;
   std::filesystem::remove(temporary,ec);
   throw;
  }
 }
public:
 explicit ExactEpisodicMemory(std::string path):path_(std::move(path)){
  if(!std::filesystem::exists(path_))return;
  if(std::filesystem::file_size(path_)>MAX_FILE_BYTES)
   throw std::runtime_error("episodic checkpoint exceeds storage cap");
  std::ifstream in(path_,std::ios::binary);
  if(!in)throw std::runtime_error("cannot open episodic checkpoint");
  std::array<char,8> magic{};
  if(!in.read(magic.data(),static_cast<std::streamsize>(magic.size()))||
     magic!=MAGIC)throw std::runtime_error("invalid episodic checkpoint format");
  const auto count=read_integer(in,4);
  if(count>MAX_RECORDS)throw std::runtime_error("episodic checkpoint record limit");
  for(size_t i=0;i<count;++i){
   const auto keylen=read_integer(in,4);
   const auto vallen=read_integer(in,4);
   const auto srclen=read_integer(in,4);
   const auto digest=read_integer(in,8);
   if(keylen>MAX_KEY||vallen>MAX_VALUE||srclen>MAX_SOURCE)
    throw std::runtime_error("oversize episodic record");
   Record r{read_string(in,static_cast<size_t>(keylen)),
            read_string(in,static_cast<size_t>(vallen)),
            read_string(in,static_cast<size_t>(srclen))};
   validate(r);
   if(hash(r)!=digest)throw std::runtime_error("episodic checkpoint digest mismatch");
   if(std::any_of(entries_.begin(),entries_.end(),
       [&](const Record& e){return e.cue==r.cue&&e.source==r.source;}))
    throw std::runtime_error("duplicate episodic cue");
   r.phase=phase_code(r.cue);
   entries_.push_back(std::move(r));
  }
  if(in.peek()!=std::char_traits<char>::eof())
   throw std::runtime_error("unexpected episodic checkpoint trailing data");
  rebuild_index();
 }
 size_t size()const{return entries_.size();}
 // Multiple independent sources may make the SAME claim. Divergent
 // values are contested: never auto-complete one value as authoritative.
 const Record* exact(std::string_view cue)const{
  const auto it=cue_index_.find(std::string(cue));
  if(it==cue_index_.end())return nullptr;
  const auto& first=entries_[it->second.front()];
  for(size_t index:it->second)
   if(entries_[index].value!=first.value)return nullptr;
  return &first;
 }
 bool contested(std::string_view cue)const{
  const auto it=cue_index_.find(std::string(cue));
  if(it==cue_index_.end())return false;
  const auto& first=entries_[it->second.front()];
  for(size_t index:it->second)
   if(entries_[index].value!=first.value)return true;
  return false;
 }
 void remember(const std::string& cue,const std::string& value,
               const std::string& source){
  auto next=entries_;
  upsert(next,{cue,value,source});
  save(next);
  entries_=std::move(next);
  rebuild_index();
 }
 size_t import_tsv(std::istream& input,const std::string& source){
  auto next=entries_;
  std::string line;
  size_t loaded=0;
  while(std::getline(input,line)){
   if(!line.empty()&&line.back()=='\r')line.pop_back();
   if(line.empty())continue;
   const auto split=line.find('\t');
   if(split==std::string::npos)
    throw std::invalid_argument("episodic TSV line has no tab separator");
   upsert(next,{line.substr(0,split),line.substr(split+1),source});
   ++loaded;
  }
  if(!input.eof()&&input.fail())throw std::runtime_error("episodic input read failed");
  if(loaded){save(next);entries_=std::move(next);rebuild_index();}
  return loaded;
 }
 // Explicit source-document indexing: each sufficiently long UTF-8 line
 // becomes a literal continuation keyed by its opening phrase, not a
 // semantic fact inferred by the network. Full source/line attribution
 // prevents distinct quotations from overwriting one another.
 size_t import_text_lines(std::istream& input,const std::string& source){
  auto next=entries_;
  std::string line;
  size_t accepted=0,line_no=0;
  while(std::getline(input,line)){
   ++line_no;
   if(!line.empty()&&line.back()=='\r')line.pop_back();
   if(!valid_utf8(line))
    throw std::invalid_argument("document contains invalid UTF-8");
   if(line.size()<20)continue;
   const auto ending=line.find(' ',16);
   if(ending==std::string::npos||ending+1>MAX_KEY||
      ending+1>=line.size())continue;
   Record r{line.substr(0,ending+1),line.substr(ending+1),
            source+"#L"+std::to_string(line_no)};
   upsert(next,std::move(r));
   ++accepted;
  }
  if(!input.eof()&&input.fail())
   throw std::runtime_error("document indexing input read failed");
  if(accepted){save(next);entries_=std::move(next);rebuild_index();}
  return accepted;
 }
 // Optional phase-resonant ranking. Candidate ranking is advisory:
 // *every selected record is checked byte-for-byte* and only the longest
 // verified exact cue can activate copying. No approximate record may
 // enter the exact-copy path. Different sources stay contested.
 Lookup probe(const uint8_t* history,size_t length,
              bool resonate=false,
              const std::array<double,255>* live_wave=nullptr)const{
  Lookup found{};
  if(length==0||history[length-1]!=static_cast<uint8_t>(' '))
   return found;
  if(!resonate)return probe_hash(history,length);
  std::vector<size_t> order;
  order.reserve(entries_.size());
  std::vector<double> scores(entries_.size(),0.0);
  std::array<PhaseCode,MAX_KEY+1> recent{};
  std::array<bool,MAX_KEY+1> ready{};
  for(size_t i=0;i<entries_.size();++i){
   const auto& r=entries_[i];
   const size_t n=r.cue.size();
   if(n>length)continue;
   order.push_back(i);
   if(!resonate)continue;
   if(!ready[n]){
    const auto* begin=reinterpret_cast<const char*>(history+length-n);
    recent[n]=phase_code(std::string_view(begin,n));
    ready[n]=true;
   }
   scores[i]=similarity(recent[n],r.phase);
   // Feed native 255-head standing-wave activations into candidate priority.
   // Small, bounded modulation cannot turn a nonmatching cue into evidence.
   if(live_wave){
    double imprint=0;
    for(size_t j=0;j<n;++j){
     const uint8_t head=static_cast<uint8_t>(r.cue[j]);
     if(head<255)
      imprint+=(*live_wave)[head]*std::cos(0.238*double(n-j));
    }
    scores[i]+=0.005*std::tanh(imprint/double(n));
   }
  }
  std::sort(order.begin(),order.end(),[&](size_t a,size_t b){
   const size_t na=entries_[a].cue.size(),nb=entries_[b].cue.size();
   if(na!=nb)return na>nb; // Longest exact suffix always wins.
   if(resonate&&scores[a]!=scores[b])return scores[a]>scores[b];
   return a<b;
  });
  found.routed=order.size();
  for(size_t i:order){
   const auto& cue=entries_[i].cue;
   ++found.exact_compares;
   bool match=true;
   for(size_t j=0;j<cue.size();++j){
    if(history[length-cue.size()+j]!=static_cast<uint8_t>(cue[j])){
     match=false;break;
    }
   }
   if(!match)continue;
   if(contested(cue))return found; // Never fall back past conflict.
   const Record* unique=exact(cue);
   if(unique){
    found.record=static_cast<uint32_t>(
        static_cast<size_t>(unique-entries_.data()));
    found.resonance=resonate?scores[i]:0.0;
   }
   return found;
  }
  return found;
 }
 // A nonbinding associative search for operator review: approximate
 // cues and source labels only; NEVER returns the fact/value, never copies,
 // and never claims a phase hit is an exact memory match.
 std::vector<Suggestion> suggest(const uint8_t* history,size_t length,
                        size_t count=3,
                        const std::array<double,255>* live_wave=nullptr)const{
  if(!history||length==0||count==0||count>16)return {};
  std::array<PhaseCode,MAX_KEY+1> recent{};
  std::array<bool,MAX_KEY+1> ready{};
  std::vector<Suggestion> scored;
  scored.reserve(entries_.size());
  for(const auto& r:entries_){
   const size_t n=r.cue.size();
   if(n>length)continue;
   if(!ready[n]){
    const auto* start=reinterpret_cast<const char*>(history+length-n);
    recent[n]=phase_code(std::string_view(start,n));ready[n]=true;
   }
   double score=similarity(recent[n],r.phase);
   if(live_wave){
    double imprint=0;
    for(size_t i=0;i<n;++i){
     const uint8_t h=static_cast<uint8_t>(r.cue[i]);
     if(h<255)imprint+=(*live_wave)[h]*
         std::cos(0.238*double(n-i));
    }
    score+=0.005*std::tanh(imprint/double(n));
   }
   bool verified=true;
   for(size_t i=0;i<n;++i)
    if(history[length-n+i]!=static_cast<uint8_t>(r.cue[i])){
     verified=false;break;
    }
   // Even a byte-exact contested key is NOT unambiguously verified.
   verified=verified&&!contested(r.cue);
   scored.push_back({r.cue,r.source,score,verified});
  }
  std::sort(scored.begin(),scored.end(),
            [](const Suggestion& a,const Suggestion& b){
    if(a.exact_verified!=b.exact_verified)
     return a.exact_verified;
    if(a.resonance!=b.resonance)
     return a.resonance>b.resonance;
    if(a.cue!=b.cue)return a.cue<b.cue;
    return a.source<b.source;
  });
  if(scored.size()>count)scored.resize(count);
  return scored;
 }
 // A caller streams one symbol at a time. Once a cue matches, the active
 // record ID/offset survives truncation of the 64-byte attention window.
 // A wrong byte aborts the copy instead of silently rewriting a fact.
 void advance(const uint8_t* history,size_t length,uint8_t symbol,
              uint32_t& record,size_t& offset,bool resonate=false,
              const std::array<double,255>* live_wave=nullptr)const{
  if(record<entries_.size()){
   const auto& value=entries_[record].value;
   if(offset<value.size()&&static_cast<uint8_t>(value[offset])==symbol){
    ++offset;
    if(offset==value.size()){record=NONE;offset=0;}
    return;
   }
  }
  record=NONE;offset=0;
  if(symbol!=static_cast<uint8_t>(' '))return;
  const auto matched=probe(history,length,resonate,live_wave);
  record=matched.record;
 }
 bool next(uint32_t record,size_t offset,uint8_t& symbol)const{
  if(record>=entries_.size()||offset>=entries_[record].value.size())
   return false;
  symbol=static_cast<uint8_t>(entries_[record].value[offset]);
  return true;
 }
};

} // namespace nanlm
