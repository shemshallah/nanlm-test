#pragma once
#include "nanlm/sensor_memory.hpp"
#include <array>
#include <cmath>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <type_traits>
namespace nanlm::math_sensor {
// Hot entries are indexed by original triad ID; inactive entries use 255.
struct HotTriadPacket {
 std::array<float,9> tensor{};
 float score=0,activation=0,previous_activation=0;
 float support=0,refute=0,curvature=0;
 std::uint16_t source_count=0;
 std::uint8_t group=255;
 std::array<std::uint8_t,3> heads{};
};
struct HotLinkPacket {
 float tensor_cosine=0,phase_lock=0;
 std::uint16_t shared_sources=0;
 std::uint8_t a=255,b=255;
};
struct NumericProofPacket {
 std::array<std::uint64_t,3> query{};
 std::array<std::uint64_t,6> proof_nodes{};
 std::array<std::uint64_t,6> proof_sources{};
 std::array<float,255> head_activity{};
 std::array<std::array<float,9>,85> triad_tensors{};
 std::array<HotTriadPacket,SensorMemory::HOT_MAX> hot_triads{};
 std::array<HotLinkPacket,SensorMemory::HOT_LINK_MAX> hot_links{};
 std::array<float,9> hot_field_tensor{};
 float hot_field_support=0,hot_field_refute=0;
 std::uint8_t hot_count=0,hot_link_count=0;
 std::array<float,9> global_tensor{};
 float global_support=0,global_refute=0;
 std::uint8_t truth=0,inferred=0,path_length=0,source_length=0;
};
static_assert(std::is_trivially_copyable_v<NumericProofPacket>);
static_assert(std::is_standard_layout_v<NumericProofPacket>);
class BootstrapCatalog {
 std::map<std::uint64_t,std::string> lookup_;
public:
 std::uint64_t register_label(const std::string& label){
  if(label.empty())throw std::invalid_argument("blank catalog entry");
  const auto key=fingerprint(label);
  auto [it,inserted]=lookup_.emplace(key,label);
  if(!inserted&&it->second!=label)throw std::runtime_error("symbol identifier collision");
  return key;
 }
 void register_assertion(const Assertion& a){
  register_label(a.source);register_label(a.subject);
  register_label(a.predicate);register_label(a.object);
 }
 const std::string& decode(std::uint64_t symbol)const{
  auto it=lookup_.find(symbol);
  if(it==lookup_.end())throw std::runtime_error("missing HW0 bootstrap symbol");
  return it->second;
 }
 std::size_t size()const{return lookup_.size();}
};
inline NumericProofPacket encode_proof(const SensorMemory& sensor,
                                       const std::string& subject,
                                       const std::string& predicate,
                                       const std::string& object){
 const auto r=sensor.reason(subject,predicate,object);
 NumericProofPacket out{};
 out.query={fingerprint(subject),fingerprint(predicate),fingerprint(object)};
 out.truth=static_cast<std::uint8_t>(r.state());
 out.inferred=static_cast<std::uint8_t>(r.inferred);
 if(r.proof_nodes.size()>out.proof_nodes.size()||r.proof_sources.size()>out.proof_sources.size())
  throw std::length_error("proof exceeds numeric packet bounds");
 out.path_length=std::uint8_t(r.proof_nodes.size());
 out.source_length=std::uint8_t(r.proof_sources.size());
 for(std::size_t i=0;i<r.proof_nodes.size();++i)
  out.proof_nodes[i]=fingerprint(r.proof_nodes[i]);
 std::size_t idx=0;
 for(const auto& source:r.proof_sources)out.proof_sources[idx++]=fingerprint(source);
 for(std::size_t i=0;i<255;++i)
  out.head_activity[i]=float(sensor.head_activity(std::uint8_t(i)));
 for(std::size_t g=0;g<85;++g)
  for(std::size_t k=0;k<9;++k)
   out.triad_tensors[g][k]=float(sensor.triad(g).tensor[k]);
 out.hot_count=static_cast<std::uint8_t>(sensor.hot_count());
 out.hot_link_count=static_cast<std::uint8_t>(sensor.hot_links().size());
 for(std::size_t slot=0;slot<sensor.hot_count();++slot) {
  const auto& h=sensor.hot_triad(slot);
  auto& o=out.hot_triads[slot];
  o.group=h.group;o.heads=h.heads;
  o.score=float(h.score);o.activation=float(h.activation);
  o.previous_activation=float(h.previous_activation);
  o.support=float(h.support);o.refute=float(h.refute);
  o.curvature=float(h.curvature);
  o.source_count=static_cast<std::uint16_t>(std::min<std::size_t>(h.source_count,65535));
  for(std::size_t k=0;k<9;++k)o.tensor[k]=float(h.tensor[k]);
 }
 for(std::size_t slot=0;slot<sensor.hot_links().size();++slot) {
  const auto& e=sensor.hot_links()[slot];
  auto& o=out.hot_links[slot];
  o.a=e.a;o.b=e.b;
  o.tensor_cosine=float(e.tensor_cosine);
  o.phase_lock=float(e.phase_lock);
  o.shared_sources=static_cast<std::uint16_t>(std::min<std::size_t>(e.shared_sources,65535));
 }
 for(std::size_t k=0;k<9;++k)
  out.hot_field_tensor[k]=float(sensor.hot_field().tensor[k]);
 out.hot_field_support=float(sensor.hot_field().support);
 out.hot_field_refute=float(sensor.hot_field().refute);
 for(std::size_t k=0;k<9;++k)
  out.global_tensor[k]=float(sensor.global().tensor[k]);
 out.global_support=float(sensor.global().support);
 out.global_refute=float(sensor.global().refute);
 return out;
}
} // namespace nanlm::math_sensor
