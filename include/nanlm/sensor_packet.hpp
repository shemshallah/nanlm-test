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
struct NumericProofPacket {
 std::array<std::uint64_t,3> query{};
 std::array<std::uint64_t,6> proof_nodes{};
 std::array<std::uint64_t,6> proof_sources{};
 std::array<float,255> head_activity{};
 std::array<std::array<float,9>,85> triad_tensors{};
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
 for(std::size_t k=0;k<9;++k)
  out.global_tensor[k]=float(sensor.global().tensor[k]);
 out.global_support=float(sensor.global().support);
 out.global_refute=float(sensor.global().refute);
 return out;
}
} // namespace nanlm::math_sensor
