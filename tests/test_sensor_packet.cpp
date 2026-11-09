#include "nanlm/sensor_packet.hpp"
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
using namespace nanlm::math_sensor;
static void require(bool x,const char* msg){if(!x)throw std::runtime_error(msg);}
int main(){
 try{
  SensorMemory m;BootstrapCatalog bootstrap;
  const Assertion a{"A","penguin","ISA","bird",false,1.};
  const Assertion b{"B","bird","ISA","animal",false,1.};
  const Assertion c{"C","penguin","ISA","animal",true,1.};
  for(const auto& e:{a,b,c}){m.ingest(e);bootstrap.register_assertion(e);}
  const auto packet=encode_proof(m,"penguin","ISA","animal");
  require(packet.truth==3&&packet.inferred==1,"lost numeric proof truth");
  require(packet.path_length==3&&packet.source_length==2,"wrong numeric proof cardinality");
  require(bootstrap.decode(packet.query[0])=="penguin","bootstrap decoded wrong source");
  require(bootstrap.decode(packet.query[1])=="ISA","bootstrap decoded wrong relation");
  require(bootstrap.decode(packet.query[2])=="animal","bootstrap decoded wrong object");
  require(bootstrap.decode(packet.proof_nodes[1])=="bird","proof word resolution failed");
  require(bootstrap.decode(packet.proof_sources[0])=="A"&&
          bootstrap.decode(packet.proof_sources[1])=="B","provenance resolution failed");
  double head_mass=0,triad_mass=0,global_mass=0;
  for(auto x:packet.head_activity)head_mass+=x;
  for(const auto& t:packet.triad_tensors)for(auto x:t)triad_mass+=std::abs(x);
  for(auto x:packet.global_tensor)global_mass+=std::abs(x);
  require(head_mass>0&&triad_mass>0&&global_mass>0,"numeric multiresolution signal missing");
  require(packet.global_support>0&&packet.global_refute>0,"numeric polarities were collapsed");
  require(packet.hot_count>0&&packet.hot_count<=SensorMemory::HOT_MAX,
          "numeric packet missing dynamic hot-triad resolution");
  require(packet.hot_link_count==packet.hot_count*(packet.hot_count-1)/2,
          "hot-triad numeric link projection incomplete");
  const auto hot=m.hot_triad(0);
  require(packet.hot_triads[0].group==hot.group&&
          packet.hot_triads[0].heads==hot.heads,
          "numeric hot-triad mapping lost parent head identities");
  require(std::abs(packet.hot_triads[0].tensor[0]-hot.tensor[0])<1e-5,
          "numeric hot-triad tensor lost");
  require(packet.hot_field_support>0&&packet.hot_field_refute>0,
          "numeric hotspot field lost signed evidence");
  double vector_norm=0,head_tensor_norm=0,triad_vector_norm=0;
  for(std::size_t i=0;i<255;++i){
   for(std::size_t k=0;k<3;++k){
    vector_norm+=std::abs(packet.head_vectors[i][k]);
    require(std::abs(packet.head_vectors[i][k]-sensor_float_packet(m.head_vector(std::uint8_t(i))[k]))<1e-5f,
            "head vector readout differs from sensor");
   }
   for(float t:packet.head_tensors[i])head_tensor_norm+=std::abs(t);
  }
  for(const auto& v:packet.triad_vectors)for(float x:v)triad_vector_norm+=std::abs(x);
  require(vector_norm>0&&head_tensor_norm>0&&triad_vector_norm>0,
          "numeric-only packet omitted vector or 255-point tensor levels");
  for(std::size_t k=0;k<3;++k){
   require(std::abs(packet.global_vector[k]-sensor_float_packet(m.global().vector[k]))<1e-5f,
           "global vector not mapped to bootstrap packet");
   require(std::abs(packet.hot_field_vector[k]-sensor_float_packet(m.hot_field().vector[k]))<1e-5f,
           "hot field vector not mapped to bootstrap packet");
   require(std::abs(packet.hot_triads[0].vector[k]-sensor_float_packet(hot.vector[k]))<1e-5f,
           "hot trihead signed vector omitted");
  }
  require(packet.hot_count==m.hot_field().active,
          "numeric hotspot count does not match sensor");
  bool missing=false;
  try{(void)bootstrap.decode(0xDEADBEEF);}catch(const std::runtime_error&){missing=true;}
  require(missing,"unknown symbol invented a text label");
  std::cout<<"PASS: plaintext-free numeric 255/85/1 proof packet, 3-hop structure, dual polarity, HW0 bootstrap symbol reconstruction\n";
  return 0;
 }catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<"\n";return 1;}
}
