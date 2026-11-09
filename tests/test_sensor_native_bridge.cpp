#include "nanlm/unified_wave.hpp"
#include "nanlm/sensor_memory.hpp"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
using namespace nanlm;
using nanlm::math_sensor::SensorMemory;
static void require(bool x,const char* msg){if(!x)throw std::runtime_error(msg);}
int main(){
 const auto path=(std::filesystem::temp_directory_path()/
  ("nanlm-native-sensor-"+std::to_string(::getpid()))).string();
 const auto cleanup=[&](){
  for(const char* suffix:{"",".so3",".overlay",".overlay.tmp"})
   std::filesystem::remove(path+suffix);
 };
 try{
  {
   Universe universe(path);
   UnifiedWaveLanguage model(universe);
   require(UnifiedWaveLanguage::N==255&&UnifiedWaveLanguage::GROUPS==85,
           "native mapping size differs from experimental sensor");
   const auto addr=universe.hypermatrix_address(0,1,2);
   require(addr.inverse[0]==mirror(addr.directed[0]),
           "native hypermatrix reverse addressing differs");
   require(addr.pair_offsets[0]==universe.leaf(addr.inverse[0]).physical_offset,
           "native mirror lanes do not share same root cell");
   std::vector<std::uint8_t> root_before(ROOT_BYTES);
   std::memcpy(root_before.data(),&universe.root(),ROOT_BYTES);
   const auto hyper_before=universe.hyperedge_count();
   SensorMemory sensor;
   sensor.import_native_mapping(model);
   sensor.import_native_geometry(universe);
   require(sensor.ingest({"first","owl","ISA","bird",false,1.}),
           "first-source intake failed");
   require(sensor.ingest({"second","bird","ISA","animal",false,1.}),
           "second-source intake failed");
   require(sensor.ingest({"third","owl","ISA","animal",true,1.}),
           "conflicting-source intake failed");
   const auto result=sensor.reason("owl","ISA","animal");
   require(result.state()==3&&result.inferred&&result.proof_sources.size()==2,
           "native sensor proof failed");
   require(std::memcmp(root_before.data(),&universe.root(),ROOT_BYTES)==0,
           "experimental sensor modified production NaN16 resident root");
   require(universe.hyperedge_count()==hyper_before,
           "experimental sidecar unexpectedly wrote native overlay");
   bool saw_curvature=false;
   for(std::size_t g=0;g<85;++g){
    if(sensor.triad(g).activation==0)continue;
    // Native unmodified model starts with consecutive triad memberships.
    const auto a=std::uint8_t(g*3),b=std::uint8_t(g*3+1),c=std::uint8_t(g*3+2);
    require(std::fabs(sensor.triad(g).curvature-universe.triangle(a,b,c))<1e-12,
            "native U(1) triangle sampling diverged");
    saw_curvature=true;
   }
   require(saw_curvature,"no active triad sampled native curvature");
   universe.sync();
  }
  require(std::filesystem::file_size(path)==ROOT_BYTES,
          "native sensor changed mapped root byte count");
  cleanup();
  std::cout<<"PASS: live UnifiedWaveLanguage membership, native triangle and physical hypermatrix ABI; no 8 MiB root or native overlay writes\n";
  return 0;
 }catch(const std::exception& e){
  cleanup();std::cerr<<"FAIL: "<<e.what()<<"\n";return 1;
 }
}
