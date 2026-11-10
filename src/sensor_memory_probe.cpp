#include "nanlm/sensor_memory.hpp"
#include <fstream>
#include <iostream>
using namespace nanlm::math_sensor;
int main(int argc,char** argv){
 if(argc!=5){std::cerr<<"usage: sensor-memory-probe ASSERTIONS.tsv SUBJECT RELATION OBJECT\n";return 2;}
 try{
  std::ifstream in(argv[1]);if(!in)throw std::runtime_error("unavailable TSV");
  SensorMemory sensor;const auto accepted=sensor.ingest_tsv(in);
  const auto r=sensor.reason(argv[2],argv[3],argv[4]);
  std::cout<<"SENSOR_MEMORY accepted="<<accepted<<" heads=255 triads=85"
           <<" sources="<<sensor.global().unique_sources
           <<" facts="<<sensor.global().unique_statements
           <<" state="<<r.state()<<" inferred="<<r.inferred<<" proof=";
  for(std::size_t i=0;i<r.proof_nodes.size();++i){if(i)std::cout<<" -> ";std::cout<<r.proof_nodes[i];}
  std::cout<<" provenance=";
  for(const auto& s:r.proof_sources)std::cout<<' '<<s;
  std::cout<<"\n";
  return 0;
 }catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<"\n";return 1;}
}
