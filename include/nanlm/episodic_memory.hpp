#pragma once
#include "nanlm/utf8.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
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
 struct Record { std::string cue,value,source; };
private:
 std::string path_;
 std::vector<Record> entries_;
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
  auto it=std::find_if(data.begin(),data.end(),
                       [&](const Record& x){return x.cue==r.cue;});
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
       [&](const Record& e){return e.cue==r.cue;}))
    throw std::runtime_error("duplicate episodic cue");
   entries_.push_back(std::move(r));
  }
  if(in.peek()!=std::char_traits<char>::eof())
   throw std::runtime_error("unexpected episodic checkpoint trailing data");
 }
 size_t size()const{return entries_.size();}
 const Record* exact(std::string_view cue)const{
  for(const auto& r:entries_)if(r.cue==cue)return &r;
  return nullptr;
 }
 void remember(const std::string& cue,const std::string& value,
               const std::string& source){
  auto next=entries_;
  upsert(next,{cue,value,source});
  save(next);
  entries_=std::move(next);
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
  if(loaded){save(next);entries_=std::move(next);}
  return loaded;
 }
 // A caller streams one symbol at a time. Once a cue matches, the active
 // record ID/offset survives truncation of the 64-byte attention window.
 // A wrong byte aborts the copy instead of silently rewriting a fact.
 void advance(const uint8_t* history,size_t length,uint8_t symbol,
              uint32_t& record,size_t& offset)const{
  if(record<entries_.size()){
   const std::string& value=entries_[record].value;
   if(offset<value.size()&&static_cast<uint8_t>(value[offset])==symbol){
    ++offset;
    if(offset==value.size()){record=NONE;offset=0;}
    return;
   }
  }
  record=NONE;offset=0;
  if(length==0)return;
  size_t best=0;
  for(size_t i=0;i<entries_.size();++i){
   const std::string& cue=entries_[i].cue;
   if(cue.size()>best&&cue.size()<=length){
    bool match=true;
    for(size_t j=0;j<cue.size();++j)
     if(history[length-cue.size()+j]!=static_cast<uint8_t>(cue[j]))
      {match=false;break;}
    if(match){best=cue.size();record=static_cast<uint32_t>(i);}
   }
  }
 }
 bool next(uint32_t record,size_t offset,uint8_t& symbol)const{
  if(record>=entries_.size()||offset>=entries_[record].value.size())
   return false;
  symbol=static_cast<uint8_t>(entries_[record].value[offset]);
  return true;
 }
};

} // namespace nanlm
