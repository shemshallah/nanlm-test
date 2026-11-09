#include "nanlm/sensor_memory.hpp"
#include <filesystem>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
using namespace nanlm::math_sensor;
static std::vector<std::string> fields(const std::string& s){
 std::istringstream in(s);std::string token;std::vector<std::string> out;
 while(std::getline(in,token,'\t'))out.push_back(token);
 return out;
}
int main(int argc,char** argv){
 if(argc!=2){std::cerr<<"usage: sensor-resident SNAPSHOT.sensor.tsv\n";return 2;}
 try{
  const std::filesystem::path path=argv[1];
  SensorMemory sensor;
  if(std::filesystem::exists(path))sensor.load(path);
  std::cout<<"READY heads=255 triads=85 sources="<<sensor.global().unique_sources
           <<" facts="<<sensor.global().unique_statements<<"\n";
  for(std::string command;std::getline(std::cin,command);){
   try{
    const auto f=fields(command);
    if(f.empty())continue;
    if(f[0]=="INGEST"){
     if(f.size()!=6||(f[2]!="+"&&f[2]!="-"))
      throw std::invalid_argument("INGEST requires SOURCE POLARITY SUBJECT RELATION OBJECT");
     const bool changed=sensor.ingest({f[1],f[3],f[4],f[5],f[2]=="-",1.});
     if(changed)sensor.save(path);
     std::cout<<"INGEST accepted="<<changed<<" events="<<sensor.journal_size()<<"\n";
    }else if(f[0]=="QUERY"){
     if(f.size()!=4)throw std::invalid_argument("QUERY requires SUBJECT RELATION OBJECT");
     const auto r=sensor.reason(f[1],f[2],f[3]);
     std::cout<<"QUERY state="<<r.state()<<" inferred="<<r.inferred<<" proof=";
     for(std::size_t k=0;k<r.proof_nodes.size();++k){
      if(k)std::cout<<"->";
      std::cout<<r.proof_nodes[k];
     }
     std::cout<<" sources=";
     for(const auto& s:r.proof_sources)std::cout<<s<<",";
     std::cout<<"\n";
    }else if(f[0]=="TICK"){
     if(f.size()!=1)throw std::invalid_argument("TICK takes no arguments");
     sensor.tick();
     std::cout<<"TICK now="<<sensor.clock()<<"\n";
    }else if(f[0]=="STATUS"){
     std::cout<<"STATUS tick="<<sensor.clock()<<" sources="<<sensor.global().unique_sources
              <<" facts="<<sensor.global().unique_statements
              <<" activity="<<sensor.global().activation<<"\n";
    }else if(f[0]=="SYNC"){
     sensor.save(path);std::cout<<"SYNC saved=1\n";
    }else if(f[0]=="QUIT"){
     sensor.save(path);std::cout<<"BYE saved=1\n";return 0;
    }else throw std::invalid_argument("unknown command");
   }catch(const std::exception& e){std::cout<<"ERROR "<<e.what()<<"\n";}
  }
  sensor.save(path);std::cout<<"BYE saved=1\n";
  return 0;
 }catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<"\n";return 1;}
}
