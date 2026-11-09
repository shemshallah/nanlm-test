#include "nanlm/universe.hpp"
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

using namespace nanlm;
static void require(bool ok,const char* reason){if(!ok)throw std::runtime_error(reason);}
int main(){
 const std::string root=(std::filesystem::temp_directory_path()/
   ("nanlm-fabric-integrity-"+std::to_string(::getpid()))).string();
 try{
  {
   Universe u(root);
   const auto maximum=std::numeric_limits<float>::max();
   u.write(24,49,PRECEDES,false,maximum,1);
   const auto old=u.query(24,49,PRECEDES);
   const auto count=u.event_count();
   bool overflow=false;
   try{u.write(24,49,PRECEDES,false,maximum,2);}
   catch(const std::overflow_error&){overflow=true;}
   require(overflow,"pair overflow did not reject before mutation");
   require(u.event_count()==count&&u.query(24,49,PRECEDES).support==old.support,
           "rejected write altered the mapped fabric");
   require(u.query(49,24,FOLLOWS).support==old.support,
           "inverse evidence diverged after rejection");
   require(!u.trial(255,24,49,PRECEDES,false,0.1f,1,3),
           "ingress permitted local self modification");
   bool bad=false;
   try{u.add_hyper(10,20,30,PRECEDES,false,
                   std::numeric_limits<float>::quiet_NaN(),4);}
   catch(const std::invalid_argument&){bad=true;}
   require(bad&&!u.hyperedge_count(),"nonfinite hyperedge was accepted");
   u.add_hyper(10,20,30,PRECEDES,false,0.4f,4);
   u.sync();
  }
  {
   Universe u(root);
   require(u.event_count()==1&&u.hyperedge_count()==1,
           "indexed overlay did not survive reboot");
   require(u.hyper_query(10,20,30,PRECEDES).support>0.39f,
           "signed hyperedge missing after reboot");
  }
  // Malformed nonfinite persisted journal must never be projected into
  // mapped pair lanes and then presented as valid prediction evidence.
  {
   std::fstream io(root+".overlay",std::ios::binary|std::ios::in|std::ios::out);
   require(bool(io),"overlay missing for corruption regression");
   const auto offset=sizeof(OverlayHeader)+sizeof(double)*PAIRS+offsetof(Event,weight);
   io.seekp(static_cast<std::streamoff>(offset));
   const float invalid=std::numeric_limits<float>::quiet_NaN();
   io.write(reinterpret_cast<const char*>(&invalid),sizeof(invalid));
   require(bool(io),"unable to corrupt test overlay");
  }
  bool rejected=false;
  try{Universe invalid(root);}
  catch(const std::runtime_error&){rejected=true;}
  require(rejected,"invalid event was accepted from persisted overlay");
  for(const auto& suffix:{"",".so3",".overlay",".overlay.tmp"})
   std::filesystem::remove(root+suffix);
  std::cout<<"PASS: overflow-safe mirror writes, bounded hypergraph, invalid overlay rejection\n";
  return 0;
 }catch(const std::exception& e){
  std::cerr<<"FAIL: "<<e.what()<<"\n";
  for(const auto& suffix:{"",".so3",".overlay",".overlay.tmp"})
   std::filesystem::remove(root+suffix);
  return 1;
 }
}
