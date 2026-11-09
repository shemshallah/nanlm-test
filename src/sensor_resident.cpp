#include "nanlm/sensor_timeslices.hpp"
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
  TemporalMatrixLog temporal(path.string()+".slices");
  if(temporal.next_sequence()==1)temporal.capture(sensor);
  else {
   const auto latest=temporal.latest();
   if(sensor.clock()<latest.frame.nanclock)sensor.advance_to(latest.frame.nanclock);
   else if(sensor.clock()>latest.frame.nanclock)temporal.capture(sensor);
  }
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
     if(changed){
      sensor.save(path);
      temporal.capture(sensor);
     }
     std::cout<<"INGEST accepted="<<changed<<" events="<<sensor.journal_size()
              <<" matrix_seq="<<(temporal.next_sequence()-1)<<"\n";
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
    }else if(f[0]=="HOTMAP"){
     if(f.size()!=1)throw std::invalid_argument("HOTMAP takes no arguments");
     std::cout<<"HOTMAP count="<<sensor.hot_count()
              <<" budget="<<sensor.hot_budget()
              <<" links="<<sensor.hot_links().size()
              <<" tick="<<sensor.clock()<<"\n";
     for(std::size_t slot=0;slot<sensor.hot_count();++slot) {
      const auto& hot=sensor.hot_triad(slot);
      std::cout<<"HOT slot="<<slot<<" triad="<<int(hot.group)
               <<" heads="<<int(hot.heads[0])<<","<<int(hot.heads[1])
               <<","<<int(hot.heads[2])
               <<" score="<<hot.score<<" activity="<<hot.activation
               <<" prior="<<hot.previous_activation
               <<" support="<<hot.support<<" refute="<<hot.refute
               <<" curvature="<<hot.curvature
               <<" sources="<<hot.source_count<<"\n";
     }
     for(const auto& link:sensor.hot_links())
      std::cout<<"HOTEDGE a="<<int(link.a)<<" b="<<int(link.b)
               <<" tensor_cos="<<link.tensor_cosine
               <<" phase_lock="<<link.phase_lock
               <<" shared_sources="<<link.shared_sources<<"\n";
    }else if(f[0]=="MEASURE"){
     if(f.size()!=1)throw std::invalid_argument("MEASURE takes no arguments");
     const auto seq=temporal.capture(sensor);
     const auto latest=temporal.latest();
     std::cout<<"MEASURE seq="<<seq<<" nanclock="<<latest.frame.nanclock
              <<" heads="<<latest.frame.heads.size()
              <<" triads="<<latest.frame.triads.size()
              <<" hot="<<latest.frame.hot.size()
              <<" mirror_pairs="<<latest.frame.mirrors.size()
              <<" bytes="<<encode_matrix(latest.frame).size()<<"\n";
    }else if(f[0]=="MATRIX"){
     if(f.size()!=1)throw std::invalid_argument("MATRIX takes no arguments");
     const auto snapshot=temporal.latest();
     const auto& frame=snapshot.frame;
     std::cout<<"MATRIX seq="<<snapshot.seq<<" nanclock="<<frame.nanclock
              <<" heads="<<frame.heads.size()<<" triads="<<frame.triads.size()
              <<" hot="<<frame.hot.size()<<" mirror_pairs="<<frame.mirrors.size()
              <<" encoded_bytes="<<encode_matrix(frame).size()
              <<" sources="<<frame.source_count<<" facts="<<frame.fact_count<<"\n";
     const auto emit=[](const char* name,const MatrixMeasurement& x){
      std::cout<<name<<" vector="<<x.vector[0]<<","<<x.vector[1]<<","<<x.vector[2]
               <<" tensor=";
      for(std::size_t i=0;i<9;++i){if(i)std::cout<<",";std::cout<<x.tensor[i];}
      std::cout<<" support="<<x.support<<" refute="<<x.refute
               <<" activity="<<x.activity<<"\n";
     };
     emit("HOTFIELD",frame.hot_field);
     emit("GLOBAL",frame.global_field);
    }else if(f[0]=="SENSOR_VECTOR"||f[0]=="SENSOR_TENSOR"){
     const bool tensor=f[0]=="SENSOR_TENSOR";
     if(f.size()<2||f.size()>3)throw std::invalid_argument("vector/tensor command: LEVEL [INDEX]");
     const auto frame=temporal.latest().frame;
     const MatrixMeasurement* x=nullptr;
     if(f[1]=="GLOBAL"&&f.size()==2)x=&frame.global_field;
     else if(f[1]=="HOTFIELD"&&f.size()==2)x=&frame.hot_field;
     else if(f.size()==3){
      const unsigned id=static_cast<unsigned>(std::stoul(f[2]));
      if(id>=255)throw std::out_of_range("sensor matrix coordinate out of range");
      if(f[1]=="HEAD")x=frame.head(static_cast<std::uint8_t>(id));
      else if(f[1]=="TRIAD")x=frame.triad(static_cast<std::uint8_t>(id));
      else if(f[1]=="HOT")for(const auto& hot:frame.hot)
       if(hot.id==id)x=&hot;
     }
     if(!x)throw std::out_of_range("no active measurement at requested resolution/address");
     std::cout<<(tensor?"SENSOR_TENSOR":"SENSOR_VECTOR")
              <<" level="<<f[1]<<" id="<<int(x->id)<<" data=";
     if(tensor)for(std::size_t i=0;i<9;++i){
      if(i)std::cout<<",";std::cout<<x->tensor[i];
     }else for(std::size_t i=0;i<3;++i){
      if(i)std::cout<<",";std::cout<<x->vector[i];
     }
     std::cout<<"\n";
    }else if(f[0]=="TIMELINE"){
     if(f.size()!=2)throw std::invalid_argument("TIMELINE requires count");
     const unsigned n=static_cast<unsigned>(std::stoul(f[1]));
     if(n==0||n>32)throw std::invalid_argument("TIMELINE count must be 1..32");
     const auto records=temporal.recent(n);
     std::cout<<"TIMELINE count="<<records.size()<<"\n";
     for(const auto& x:records)
      std::cout<<"SLICE seq="<<x.seq<<" nanclock="<<x.frame.nanclock
               <<" h="<<x.frame.heads.size()<<" t="<<x.frame.triads.size()
               <<" hot="<<x.frame.hot.size()<<" edges="<<x.frame.mirrors.size()
               <<" global_vec_x="<<x.frame.global_field.vector[0]
               <<" global_tensor_00="<<x.frame.global_field.tensor[0]
               <<"\n";
    }else if(f[0]=="TICK"){
     if(f.size()!=1)throw std::invalid_argument("TICK takes no arguments");
     sensor.tick();
     const auto seq=temporal.capture(sensor);
     std::cout<<"TICK now="<<sensor.clock()<<" matrix_seq="<<seq<<"\n";
    }else if(f[0]=="STATUS"){
     std::cout<<"STATUS tick="<<sensor.clock()<<" sources="<<sensor.global().unique_sources
              <<" facts="<<sensor.global().unique_statements
              <<" activity="<<sensor.global().activation
              <<" hot="<<sensor.hot_count()
              <<" hot_links="<<sensor.hot_links().size()
              <<" matrix_seq="<<(temporal.next_sequence()-1)
              <<" trace_bytes="<<temporal.capacity()<<"\n";
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
