#pragma once
// Separate fixed 8 MiB experimental chart per mathematical sensor family.
// This journal is NOT native NaN16 chart state and is NOT transactional with
// the authoritative source/evidence journal.
#include "nanlm/sensor_family_maps.hpp"
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace nanlm::math_sensor {
class FamilyChartLog {
public:
 static constexpr std::uint32_t CHART_BYTES=8388608;
 static constexpr std::uint32_t SLOT_BYTES=16384;
 static constexpr std::uint32_t SLOT_COUNT=CHART_BYTES/SLOT_BYTES-1;
 static constexpr std::uint32_t RECORD_HEADER=32;
 struct Entry {
  std::uint64_t sequence=0;
  FamilyFrame frame{};
 };
private:
 std::filesystem::path path_;
 SensorFamily family_;
 std::uint64_t next_=1;
 static std::vector<std::uint8_t> magic_header(SensorFamily family) {
  SliceWriter w;
  for(char x:std::string("NANFAM01"))w.u8(static_cast<std::uint8_t>(x));
  w.u8(static_cast<std::uint8_t>(family));w.u8(1);
  w.u32(CHART_BYTES);w.u32(SLOT_BYTES);w.u32(SLOT_COUNT);
  auto bytes=w.bytes();bytes.resize(SLOT_BYTES,0);
  return bytes;
 }
 std::optional<Entry> read_slot(std::ifstream& file,std::uint32_t slot)const {
  if(slot>=SLOT_COUNT)throw std::out_of_range("family chart slot");
  const std::uint64_t offset=std::uint64_t(slot+1)*SLOT_BYTES;
  file.clear();
  file.seekg(static_cast<std::streamoff>(offset));
  std::vector<std::uint8_t> h(RECORD_HEADER);
  file.read(reinterpret_cast<char*>(h.data()),RECORD_HEADER);
  if(!file)return std::nullopt;
  try {
   SliceReader r(h);
   if(r.u8()!='F'||r.u8()!='R'||r.u8()!='E'||r.u8()!='C')return std::nullopt;
   const auto seq=r.u64(),tick=r.u64();
   const auto length=r.u32(),stored_check=r.u32();
   const auto family=r.u8(),version=r.u8();
   if(!seq||family!=static_cast<std::uint8_t>(family_)||version!=1||
      length> SLOT_BYTES-RECORD_HEADER)return std::nullopt;
   // Reserved header bytes must be zero.
   if(r.u16()!=0||r.u16()!=0)return std::nullopt;
   std::vector<std::uint8_t> payload(length);
   file.read(reinterpret_cast<char*>(payload.data()),static_cast<std::streamsize>(length));
   if(!file)return std::nullopt;
   if(checksum(payload,seq,tick)!=stored_check)return std::nullopt;
   auto frame=decode_family(payload);
   if(frame.family!=family_||frame.nanclock!=tick)return std::nullopt;
   return Entry{seq,std::move(frame)};
  }catch(const std::exception&){return std::nullopt;}
 }
 void verify()const {
  if(!std::filesystem::is_regular_file(path_)||
     std::filesystem::file_size(path_)!=CHART_BYTES)
   throw std::runtime_error("family chart capacity mismatch");
  std::ifstream f(path_,std::ios::binary);
  if(!f)throw std::runtime_error("family chart inaccessible");
  std::vector<std::uint8_t> head(SLOT_BYTES);
  f.read(reinterpret_cast<char*>(head.data()),SLOT_BYTES);
  if(!f||head!=magic_header(family_))
   throw std::runtime_error("family chart header/schema mismatch");
 }
public:
 FamilyChartLog(std::filesystem::path path,SensorFamily family):
  path_(std::move(path)),family_(family) {
  if(static_cast<std::uint8_t>(family)<1||static_cast<std::uint8_t>(family)>8)
   throw std::invalid_argument("invalid family id");
  if(!std::filesystem::exists(path_)) {
   std::ofstream f(path_,std::ios::binary|std::ios::trunc);
   if(!f)throw std::runtime_error("cannot allocate family chart");
   const auto head=magic_header(family_);
   f.write(reinterpret_cast<const char*>(head.data()),SLOT_BYTES);
   f.seekp(CHART_BYTES-1);f.put('\0');f.flush();
   if(!f)throw std::runtime_error("family chart allocation failed");
  }
  verify();
  std::ifstream f(path_,std::ios::binary);
  for(std::uint32_t s=0;s<SLOT_COUNT;++s) {
   const auto e=read_slot(f,s);
   if(e&&e->sequence>=next_)next_=e->sequence+1;
  }
 }
 std::uint64_t append(const FamilyFrame& frame) {
  if(frame.family!=family_)throw std::invalid_argument("family chart projection mismatch");
  const auto payload=encode_family(frame);
  if(payload.size()>SLOT_BYTES-RECORD_HEADER)
   throw std::overflow_error("sensor chart slot payload overflow");
  const auto seq=next_;
  SliceWriter w;
  for(char c:std::string("FREC"))w.u8(static_cast<std::uint8_t>(c));
  w.u64(seq);w.u64(frame.nanclock);
  w.u32(static_cast<std::uint32_t>(payload.size()));
  w.u32(checksum(payload,seq,frame.nanclock));
  w.u8(static_cast<std::uint8_t>(family_));w.u8(1);w.u16(0);w.u16(0);
  const auto& header=w.bytes();
  if(header.size()!=RECORD_HEADER)throw std::logic_error("wrong family record header width");
  std::fstream f(path_,std::ios::in|std::ios::out|std::ios::binary);
  if(!f)throw std::runtime_error("family chart cannot open for append");
  const auto slot=(seq-1)%SLOT_COUNT;
  f.seekp(static_cast<std::streamoff>((slot+1)*SLOT_BYTES));
  f.write(reinterpret_cast<const char*>(header.data()),static_cast<std::streamsize>(header.size()));
  f.write(reinterpret_cast<const char*>(payload.data()),static_cast<std::streamsize>(payload.size()));
  f.flush();
  if(!f)throw std::runtime_error("family chart write failed");
  ++next_;
  return seq;
 }
 std::optional<Entry> latest()const {
  std::ifstream f(path_,std::ios::binary);
  if(!f)throw std::runtime_error("family chart cannot read");
  std::optional<Entry> best;
  for(std::uint32_t s=0;s<SLOT_COUNT;++s) {
   auto e=read_slot(f,s);
   if(e&&(!best||e->sequence>best->sequence))best=std::move(e);
  }
  return best;
 }
 std::uint64_t next_sequence()const{return next_;}
};
} // namespace nanlm::math_sensor
