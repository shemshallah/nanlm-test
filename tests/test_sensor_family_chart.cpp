#include "nanlm/sensor_family_chart.hpp"
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>
using namespace nanlm::math_sensor;

static void check(bool value,const char* msg) {
 if(!value)throw std::runtime_error(msg);
}
int main() {
 const auto stem=(std::filesystem::temp_directory_path()/
  ("nanlm-eight-charts-"+std::to_string(::getpid()))).string();
 std::vector<std::string> paths;
 auto cleanup=[&]() {
  for(const auto& p:paths)std::filesystem::remove(p);
 };
 try {
  SensorMemory sensor(128,2000,0.8);
  sensor.ingest({"a","penguin","ISA","bird",false,1.});
  sensor.ingest({"b","bird","ISA","animal",false,1.});
  sensor.ingest({"c","penguin","ISA","animal",true,1.});
  const auto input=measure_matrix(sensor);
  const std::array<SensorFamily,8> families={
   SensorFamily::Vector,SensorFamily::Tensor,SensorFamily::Rotation,
   SensorFamily::Spectral,SensorFamily::Temporal,SensorFamily::Topology,
   SensorFamily::Information,SensorFamily::ComplexAmplitude
  };
  for(std::size_t i=0;i<families.size();++i) {
   const auto path=stem+"-"+std::to_string(i+1)+".chart";
   paths.push_back(path);
   FamilyProjector projector;
   const auto frame=projector.measure(families[i],input);
   {
    FamilyChartLog chart(path,families[i]);
    check(std::filesystem::file_size(path)==FamilyChartLog::CHART_BYTES,
          "not a blank 8 MiB independent sensor chart");
    check(chart.append(frame)==1,"first family journal sequence missing");
    const auto newest=chart.latest();
    check(newest&&newest->sequence==1&&
          newest->frame.family==families[i]&&
          newest->frame.nanclock==input.nanclock&&
          encode_family(newest->frame)==encode_family(frame),
          "family journal failed first numeric roundtrip");
   }
   {
    FamilyChartLog resumed(path,families[i]);
    check(resumed.next_sequence()==2,"family journal restart lost sequence");
    check(resumed.append(frame)==2,"family journal cannot append after restart");
   }
   bool rejected=false;
   try{
    const auto wrong=families[(i+1)%families.size()];
    FamilyChartLog invalid(path,wrong);
    (void)invalid;
   }catch(const std::runtime_error&){rejected=true;}
   check(rejected,"family chart accepted wrong mathematical sensor");
   std::cout<<"CHART family="<<i+1<<" bytes="<<std::filesystem::file_size(path)
            <<" record_bytes="<<encode_family(frame).size()<<"\n";
  }
  // Circular overwrite: fixed 511 slots after the 16 KiB header.
  const auto path=paths[0];
  FamilyProjector projector;
  auto frame=projector.measure(SensorFamily::Vector,input);
  std::uint64_t last=0;
  {
   FamilyChartLog chart(path,SensorFamily::Vector);
   for(std::uint32_t i=0;i<FamilyChartLog::SLOT_COUNT+5;++i) {
    frame.nanclock+=1;
    last=chart.append(frame);
   }
   check(last==2+FamilyChartLog::SLOT_COUNT+5,
         "fixed chart circular journal skipped/reused a sequence");
   check(chart.latest()&&chart.latest()->sequence==last,
         "circular chart cannot recover latest frame");
   check(std::filesystem::file_size(path)==FamilyChartLog::CHART_BYTES,
         "circular chart grew beyond its fixed allocation");
  }
  {
   FamilyChartLog chart(path,SensorFamily::Vector);
   check(chart.next_sequence()==last+1,
         "circular chart restart did not reconstruct next sequence");
   const auto slot=(last-1)%FamilyChartLog::SLOT_COUNT;
   std::fstream f(path,std::ios::in|std::ios::out|std::ios::binary);
   f.seekp(static_cast<std::streamoff>((slot+1)*FamilyChartLog::SLOT_BYTES+
                                     FamilyChartLog::RECORD_HEADER+12));
   f.put(static_cast<char>(0xff));f.flush();
   check(bool(f),"could not simulate a torn chart record");
   f.close();
   const auto latest=chart.latest();
   check(latest&&latest->sequence<last,
         "corrupted newest chart record passed checksum validation");
  }
  cleanup();
  std::cout<<"PASS: eight isolated blank 8 MiB sensor charts, family-specific numeric time slices, bounded wrap, restart, family mismatch and torn-record recovery\n";
 }catch(const std::exception& e) {
  cleanup();
  std::cerr<<"FAIL: "<<e.what()<<"\n";
  return 1;
 }
 return 0;
}
