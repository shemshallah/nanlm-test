#pragma once
// Branch-local mathematical measurement log. A bounded 8 MiB sidecar;
// never writes the native 8 MiB HW0 root or its NaN16 evidence overlay.
#include "nanlm/sensor_memory.hpp"
#include "nanlm/universe.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nanlm::math_sensor {

struct MatrixMeasurement {
 std::uint8_t id=255;
 std::array<std::uint8_t,3> heads{255,255,255};
 std::array<float,3> vector{};
 std::array<float,9> tensor{};
 float activity=0,support=0,refute=0,curvature=0,score=0,prior=0;
};
struct MirrorMeasurement {
 std::uint8_t group_a=255,group_b=255,from=255,to=255;
 std::uint16_t forward=0,reverse=0;
 std::uint32_t physical_offset=0;
 float tensor_cos=0,phase_lock=0,oriented_flux=0;
 std::uint16_t shared_sources=0;
};
struct MatrixTimeSlice {
 std::uint64_t nanclock=0;
 std::uint32_t source_count=0,fact_count=0;
 // Sparse rows, indexed by original register/head/triad IDs, not hot slots.
 std::vector<MatrixMeasurement> heads,triads,hot;
 std::vector<MirrorMeasurement> mirrors;
 MatrixMeasurement hot_field{},global_field{};
 const MatrixMeasurement* head(std::uint8_t h)const {
  for(const auto& sample:heads)if(sample.id==h)return &sample;
  return nullptr;
 }
 const MatrixMeasurement* triad(std::uint8_t g)const {
  for(const auto& sample:triads)if(sample.id==g)return &sample;
  return nullptr;
 }
};
inline float sensor_float(double x){
 if(!std::isfinite(x)||std::abs(x)>double(std::numeric_limits<float>::max()))
  throw std::overflow_error("sensor measurement exceeds finite binary32 range");
 return static_cast<float>(x);
}
template<class A,class B> inline void copy_measure(A& dst,const B& src) {
 for(std::size_t i=0;i<3;++i)dst.vector[i]=sensor_float(src.vector[i]);
 for(std::size_t i=0;i<9;++i)dst.tensor[i]=sensor_float(src.tensor[i]);
}
inline MatrixTimeSlice measure_matrix(const SensorMemory& sensor) {
 MatrixTimeSlice slice{};
 slice.nanclock=sensor.clock();
 if(sensor.global().unique_sources>std::numeric_limits<std::uint32_t>::max()||
    sensor.global().unique_statements>std::numeric_limits<std::uint32_t>::max())
  throw std::overflow_error("sensor graph counters exceed time slice format");
 slice.source_count=static_cast<std::uint32_t>(sensor.global().unique_sources);
 slice.fact_count=static_cast<std::uint32_t>(sensor.global().unique_statements);
 for(std::size_t i=0;i<255;++i){
  const auto h=static_cast<std::uint8_t>(i);
  if(sensor.head_activity(h)==0&&sensor.head_support(h)==0&&
     sensor.head_refute(h)==0)continue;
  MatrixMeasurement v{};v.id=h;v.heads[0]=h;
  for(std::size_t k=0;k<3;++k)v.vector[k]=sensor_float(sensor.head_vector(h)[k]);
  for(std::size_t k=0;k<9;++k)v.tensor[k]=sensor_float(sensor.head_tensor(h)[k]);
  v.activity=sensor_float(sensor.head_activity(h));
  v.support=sensor_float(sensor.head_support(h));
  v.refute=sensor_float(sensor.head_refute(h));
  slice.heads.push_back(v);
 }
 for(std::size_t g=0;g<85;++g){
  const auto& t=sensor.triad(g);
  if(t.activation==0&&t.support==0&&t.refute==0)continue;
  MatrixMeasurement v{};v.id=static_cast<std::uint8_t>(g);
  v.heads=sensor.triad_heads(static_cast<std::uint8_t>(g));
  copy_measure(v,t);
  v.activity=sensor_float(t.activation);
  v.support=sensor_float(t.support);v.refute=sensor_float(t.refute);
  v.curvature=sensor_float(t.curvature);
  slice.triads.push_back(v);
 }
 for(std::size_t k=0;k<sensor.hot_count();++k){
  const auto& h=sensor.hot_triad(k);
  MatrixMeasurement v{};v.id=h.group;v.heads=h.heads;
  copy_measure(v,h);
  v.activity=sensor_float(h.activation);v.score=sensor_float(h.score);
  v.prior=sensor_float(h.previous_activation);
  v.support=sensor_float(h.support);v.refute=sensor_float(h.refute);
  v.curvature=sensor_float(h.curvature);
  slice.hot.push_back(v);
 }
 const auto find_hot=[&](std::uint8_t g)->const HotTriad& {
  const int slot=sensor.hot_slot(g);
  if(slot<0)throw std::logic_error("hot link endpoint is not an active triad");
  return sensor.hot_triad(static_cast<std::size_t>(slot));
 };
 for(const auto& edge:sensor.hot_links()){
  const auto& a=find_hot(edge.a);
  const auto& b=find_hot(edge.b);
  const std::uint8_t x=a.heads[0],y=b.heads[0];
  if(x==y||x>=255||y>=255)throw std::logic_error("invalid mirror measurement anchors");
  MirrorMeasurement m{};
  m.group_a=edge.a;m.group_b=edge.b;m.from=x;m.to=y;
  m.forward=address(x,y);m.reverse=mirror(m.forward);
  m.physical_offset=static_cast<std::uint32_t>(offsetof(Root,p)+
                      pair_index(x,y)*sizeof(Pair));
  m.tensor_cos=sensor_float(edge.tensor_cosine);
  m.phase_lock=sensor_float(edge.phase_lock);
  // Antisymmetric oriented area measured between the two triad vectors.
  // Reversing the NaN16 mirror leg reverses the sign; it does not create
  // a second independent physical pair cell.
  m.oriented_flux=sensor_float(a.vector[0]*b.vector[1]-a.vector[1]*b.vector[0]);
  m.shared_sources=static_cast<std::uint16_t>(
      std::min<std::size_t>(edge.shared_sources,65535));
  slice.mirrors.push_back(m);
 }
 copy_measure(slice.hot_field,sensor.hot_field());
 slice.hot_field.id=255;
 slice.hot_field.activity=sensor_float(sensor.hot_field().activation);
 slice.hot_field.support=sensor_float(sensor.hot_field().support);
 slice.hot_field.refute=sensor_float(sensor.hot_field().refute);
 copy_measure(slice.global_field,sensor.global());
 slice.global_field.id=255;
 slice.global_field.activity=sensor_float(sensor.global().activation);
 slice.global_field.support=sensor_float(sensor.global().support);
 slice.global_field.refute=sensor_float(sensor.global().refute);
 return slice;
}

// Explicit little-endian wire format (no C++ struct padding or raw NaN floats).
class SliceWriter {
 std::vector<std::uint8_t> bytes_;
public:
 void u8(std::uint8_t v){bytes_.push_back(v);}
 void u16(std::uint16_t v){for(int i=0;i<2;++i)u8(std::uint8_t(v>>(i*8)));}
 void u32(std::uint32_t v){for(int i=0;i<4;++i)u8(std::uint8_t(v>>(i*8)));}
 void u64(std::uint64_t v){for(int i=0;i<8;++i)u8(std::uint8_t(v>>(i*8)));}
 void f32(float v){
  if(!std::isfinite(v))throw std::invalid_argument("nonfinite sensor frame value");
  u32(std::bit_cast<std::uint32_t>(v));
 }
 void sample(const MatrixMeasurement& x){
  u8(x.id);for(auto h:x.heads)u8(h);
  for(float f:x.vector)f32(f);
  for(float f:x.tensor)f32(f);
  for(float f:{x.activity,x.support,x.refute,x.curvature,x.score,x.prior})f32(f);
 }
 void link(const MirrorMeasurement& x){
  for(auto v:{x.group_a,x.group_b,x.from,x.to})u8(v);
  u16(x.forward);u16(x.reverse);u32(x.physical_offset);
  f32(x.tensor_cos);f32(x.phase_lock);f32(x.oriented_flux);
  u16(x.shared_sources);
 }
 const std::vector<std::uint8_t>& bytes()const{return bytes_;}
};
class SliceReader {
 const std::vector<std::uint8_t>& data_;
 std::size_t cursor_=0;
public:
 explicit SliceReader(const std::vector<std::uint8_t>& data):data_(data){}
 std::uint8_t u8(){if(cursor_>=data_.size())throw std::runtime_error("truncated slice");return data_[cursor_++];}
 std::uint16_t u16(){std::uint16_t v=0;for(int i=0;i<2;++i)v|=std::uint16_t(u8())<<(i*8);return v;}
 std::uint32_t u32(){std::uint32_t v=0;for(int i=0;i<4;++i)v|=std::uint32_t(u8())<<(i*8);return v;}
 std::uint64_t u64(){std::uint64_t v=0;for(int i=0;i<8;++i)v|=std::uint64_t(u8())<<(i*8);return v;}
 float f32(){float v=std::bit_cast<float>(u32());if(!std::isfinite(v))throw std::runtime_error("nonfinite stored slice");return v;}
 MatrixMeasurement sample(){
  MatrixMeasurement x{};x.id=u8();for(auto& h:x.heads)h=u8();
  for(auto& f:x.vector)f=f32();
  for(auto& f:x.tensor)f=f32();
  x.activity=f32();x.support=f32();x.refute=f32();
  x.curvature=f32();x.score=f32();x.prior=f32();
  return x;
 }
 MirrorMeasurement link(){
  MirrorMeasurement x{};
  x.group_a=u8();x.group_b=u8();x.from=u8();x.to=u8();
  x.forward=u16();x.reverse=u16();x.physical_offset=u32();
  x.tensor_cos=f32();x.phase_lock=f32();x.oriented_flux=f32();
  x.shared_sources=u16();
  return x;
 }
 void end()const{if(cursor_!=data_.size())throw std::runtime_error("trailing time-slice bytes");}
};
inline std::vector<std::uint8_t> encode_matrix(const MatrixTimeSlice& slice) {
 if(slice.heads.size()>255||slice.triads.size()>85||slice.hot.size()>12||
    slice.mirrors.size()>66)
  throw std::invalid_argument("slice exceeds spatial topology capacity");
 SliceWriter w;
 for(char c:std::string("MATS"))w.u8(static_cast<std::uint8_t>(c));
 w.u16(1);w.u16(0);
 w.u64(slice.nanclock);w.u32(slice.source_count);w.u32(slice.fact_count);
 for(std::size_t n:{slice.heads.size(),slice.triads.size(),slice.hot.size(),slice.mirrors.size()})
  w.u16(static_cast<std::uint16_t>(n));
 for(const auto& x:slice.heads)w.sample(x);
 for(const auto& x:slice.triads)w.sample(x);
 for(const auto& x:slice.hot)w.sample(x);
 for(const auto& x:slice.mirrors)w.link(x);
 w.sample(slice.hot_field);w.sample(slice.global_field);
 return w.bytes();
}
inline MatrixTimeSlice decode_matrix(const std::vector<std::uint8_t>& bytes) {
 SliceReader r(bytes);
 if(r.u8()!='M'||r.u8()!='A'||r.u8()!='T'||r.u8()!='S'||
    r.u16()!=1||r.u16()!=0)throw std::runtime_error("wrong matrix time-slice schema");
 MatrixTimeSlice f{};
 f.nanclock=r.u64();f.source_count=r.u32();f.fact_count=r.u32();
 const auto hn=r.u16(),tn=r.u16(),kn=r.u16(),ln=r.u16();
 if(hn>255||tn>85||kn>12||ln>66)throw std::runtime_error("invalid matrix slice dimensions");
 std::array<bool,255> hs{};std::array<bool,85> ts{},ks{};
 for(int i=0;i<hn;++i){
  auto x=r.sample();
  if(x.id>=255||hs[x.id]||x.heads[0]!=x.id||x.heads[1]!=255||x.heads[2]!=255)
   throw std::runtime_error("invalid sparse head row");
  hs[x.id]=true;f.heads.push_back(x);
 }
 for(int i=0;i<tn;++i){
  auto x=r.sample();
  if(x.id>=85||ts[x.id]||x.heads[0]>=255||x.heads[1]>=255||x.heads[2]>=255||
     x.heads[0]==x.heads[1]||x.heads[0]==x.heads[2]||x.heads[1]==x.heads[2])
   throw std::runtime_error("invalid sparse triad row");
  ts[x.id]=true;f.triads.push_back(x);
 }
 for(int i=0;i<kn;++i){
  auto x=r.sample();
  if(x.id>=85||ks[x.id]||x.heads[0]>=255||x.heads[1]>=255||x.heads[2]>=255||
     x.heads[0]==x.heads[1]||x.heads[0]==x.heads[2]||x.heads[1]==x.heads[2])
   throw std::runtime_error("invalid hot triad row");
  ks[x.id]=true;f.hot.push_back(x);
 }
 for(int i=0;i<ln;++i){
  auto x=r.link();
  if(x.group_a>=85||x.group_b>=85||x.group_a==x.group_b||
     !ks[x.group_a]||!ks[x.group_b]||x.from>=255||x.to>=255||x.from==x.to||
     x.forward!=address(x.from,x.to)||x.reverse!=mirror(x.forward)||
     x.physical_offset!=offsetof(Root,p)+pair_index(x.from,x.to)*sizeof(Pair))
   throw std::runtime_error("NaN16 mirror pair address mismatch");
  f.mirrors.push_back(x);
 }
 f.hot_field=r.sample();f.global_field=r.sample();
 if(f.hot_field.id!=255||f.global_field.id!=255)
  throw std::runtime_error("reserved global field marker missing");
 r.end();
 return f;
}
inline std::uint32_t checksum(const std::vector<std::uint8_t>& data,
                             std::uint64_t seq,std::uint64_t tick) {
 // FNV1a32 detects common truncation/torn writes, not adversarial tampering.
 std::uint32_t x=2166136261u;
 for(std::uint8_t b:data){x^=b;x*=16777619u;}
 for(int k=0;k<8;++k){x^=std::uint8_t(seq>>(8*k));x*=16777619u;}
 for(int k=0;k<8;++k){x^=std::uint8_t(tick>>(8*k));x*=16777619u;}
 return x;
}
class TemporalMatrixLog {
public:
 static constexpr std::uint32_t DEFAULT_BYTES=8388608;
 static constexpr std::uint32_t HEADER_BYTES=64,RECORD_HEADER=32,ALIGN=64;
 struct Entry {
  std::uint64_t seq=0;
  std::uint32_t offset=0;
  MatrixTimeSlice frame{};
 };
private:
 std::filesystem::path path_;
 std::uint32_t capacity_;
 std::uint64_t next_seq_=1;
 std::uint32_t next_offset_=HEADER_BYTES;
 static std::uint32_t align(std::uint32_t v){
  return (v+ALIGN-1u)&~(ALIGN-1u);
 }
 static std::vector<std::uint8_t> header(std::uint32_t cap){
  SliceWriter w;
  for(char c:std::string("NLSLICE1"))w.u8(static_cast<std::uint8_t>(c));
  w.u32(1);w.u32(cap);w.u32(ALIGN);w.u32(HEADER_BYTES);
  auto out=w.bytes();out.resize(HEADER_BYTES,0);return out;
 }
 void create()const {
  std::ofstream file(path_,std::ios::binary|std::ios::trunc);
  if(!file)throw std::runtime_error("cannot allocate NaN sensor time-slice file");
  const auto h=header(capacity_);
  file.write(reinterpret_cast<const char*>(h.data()),static_cast<std::streamsize>(h.size()));
  file.seekp(static_cast<std::streamoff>(capacity_-1));
  file.put('\0');file.close();
  if(!file)throw std::runtime_error("failed to initialize bounded sensor trace");
 }
 std::vector<std::uint8_t> mapped()const {
  std::ifstream file(path_,std::ios::binary|std::ios::ate);
  if(!file||file.tellg()!=static_cast<std::streamoff>(capacity_))
   throw std::runtime_error("invalid or truncated sensor trace file length");
  file.seekg(0);
  std::vector<std::uint8_t> data(capacity_);
  file.read(reinterpret_cast<char*>(data.data()),static_cast<std::streamsize>(data.size()));
  if(!file)throw std::runtime_error("sensor trace read failed");
  const auto expected=header(capacity_);
  if(!std::equal(expected.begin(),expected.end(),data.begin()))
   throw std::runtime_error("bad sensor trace schema or capacity");
  return data;
 }
 static std::uint32_t read32(const std::vector<std::uint8_t>& data,std::size_t at){
  std::uint32_t v=0;for(int i=0;i<4;++i)v|=std::uint32_t(data[at+i])<<(8*i);
  return v;
 }
 static std::uint64_t read64(const std::vector<std::uint8_t>& data,std::size_t at){
  std::uint64_t v=0;for(int i=0;i<8;++i)v|=std::uint64_t(data[at+i])<<(8*i);
  return v;
 }
 static std::vector<Entry> scan(const std::vector<std::uint8_t>& data) {
  std::vector<Entry> result;
  for(std::uint32_t offset=HEADER_BYTES;offset+RECORD_HEADER<=data.size();offset+=ALIGN){
   if(data[offset]!='N'||data[offset+1]!='S'||data[offset+2]!='F'||
      data[offset+3]!='1')continue;
   if(data[offset+4]!=1||data[offset+5]!=0||data[offset+6]!=0||data[offset+7]!=0)
    continue;
   const std::uint64_t seq=read64(data,offset+8),tick=read64(data,offset+16);
   const std::uint32_t size=read32(data,offset+24),crc=read32(data,offset+28);
   if(seq==0||size<32||size>65536||offset+RECORD_HEADER+size>data.size())continue;
   std::vector<std::uint8_t> bytes(data.begin()+offset+RECORD_HEADER,
                                    data.begin()+offset+RECORD_HEADER+size);
   if(checksum(bytes,seq,tick)!=crc)continue;
   try {
    auto frame=decode_matrix(bytes);
    if(frame.nanclock!=tick)continue;
    result.push_back({seq,offset,std::move(frame)});
   }catch(const std::exception&){continue;}
  }
  std::sort(result.begin(),result.end(),[](const Entry& a,const Entry& b){
   if(a.seq!=b.seq)return a.seq<b.seq;
   return a.offset<b.offset;
  });
  return result;
 }
public:
 explicit TemporalMatrixLog(std::filesystem::path path,std::uint32_t capacity=DEFAULT_BYTES)
  :path_(std::move(path)),capacity_(capacity) {
  if(capacity_<65536||capacity_>67108864||capacity_%ALIGN!=0)
   throw std::invalid_argument("sensor trace capacity must be 64-byte aligned, 64 KiB..64 MiB");
  if(!std::filesystem::exists(path_))create();
  const auto entries=scan(mapped());
  if(!entries.empty()){
   const auto& last=entries.back();
   next_seq_=last.seq+1;
   if(!next_seq_)throw std::overflow_error("time-slice sequence exhausted");
   const auto size=encode_matrix(last.frame).size();
   next_offset_=align(static_cast<std::uint32_t>(last.offset+RECORD_HEADER+size));
   if(next_offset_>=capacity_)next_offset_=HEADER_BYTES;
  }
 }
 std::uint32_t capacity()const{return capacity_;}
 std::uint64_t next_sequence()const{return next_seq_;}
 std::vector<Entry> entries()const{return scan(mapped());}
 std::uint64_t append(const MatrixTimeSlice& frame) {
  const auto payload=encode_matrix(frame);
  if(payload.size()+RECORD_HEADER>65536||payload.size()+RECORD_HEADER+HEADER_BYTES>capacity_)
   throw std::length_error("matrix frame exceeds bounded trace record");
  std::uint32_t offset=next_offset_;
  if(offset+RECORD_HEADER+payload.size()>capacity_)offset=HEADER_BYTES;
  const std::uint64_t seq=next_seq_;
  SliceWriter record;
  for(char c:std::string("NSF1"))record.u8(static_cast<std::uint8_t>(c));
  record.u32(1);record.u64(seq);record.u64(frame.nanclock);
  record.u32(static_cast<std::uint32_t>(payload.size()));
  record.u32(checksum(payload,seq,frame.nanclock));
  auto data=record.bytes();
  if(data.size()!=RECORD_HEADER)throw std::logic_error("invalid frame header");
  data.insert(data.end(),payload.begin(),payload.end());
  std::fstream file(path_,std::ios::in|std::ios::out|std::ios::binary);
  if(!file)throw std::runtime_error("sensor trace not writable");
  file.seekp(offset);
  file.write(reinterpret_cast<const char*>(data.data()),static_cast<std::streamsize>(data.size()));
  file.flush();
  if(!file)throw std::runtime_error("sensor trace write incomplete");
  next_seq_=seq+1;
  if(!next_seq_)throw std::overflow_error("sensor frame index overflow");
  next_offset_=align(static_cast<std::uint32_t>(offset+data.size()));
  if(next_offset_>=capacity_)next_offset_=HEADER_BYTES;
  return seq;
 }
 std::uint64_t capture(const SensorMemory& sensor){return append(measure_matrix(sensor));}
 Entry latest()const {
  auto v=entries();
  if(v.empty())throw std::runtime_error("sensor trace contains no valid measurements");
  return v.back();
 }
 std::vector<Entry> recent(std::size_t max_count)const {
  auto v=entries();
  if(v.size()>max_count)v.erase(v.begin(),v.end()-static_cast<std::ptrdiff_t>(max_count));
  return v;
 }
};
} // namespace nanlm::math_sensor
