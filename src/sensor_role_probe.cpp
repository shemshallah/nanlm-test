// Experimental numeric-only role instrumentation of V2 sensor journal.
// No NaN16 writes, no new persistent sidecar, no lexical labels in output.
#include "nanlm/sensor_memory.hpp"
#include <algorithm>
#include <array>
#include <charconv>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace nanlm::math_sensor;

static std::array<unsigned,6> heads_for(const std::string& word,std::uint64_t role) {
 const auto vector=role_vector(word,role);
 std::vector<std::pair<std::size_t,unsigned>> distances;
 distances.reserve(255);
 for(unsigned h=0;h<255;++h)
  distances.push_back({distance(vector,basis(0x100000000000ULL+h)),h});
 std::partial_sort(distances.begin(),distances.begin()+6,distances.end());
 std::array<unsigned,6> result{};
 for(std::size_t i=0;i<6;++i) result[i]=distances[i].second;
 std::sort(result.begin(),result.end());
 return result;
}
static std::vector<std::string> fields(const std::string& line) {
 std::vector<std::string> values;std::size_t pos=0;
 for(;;){
  const auto end=line.find('\t',pos);
  values.push_back(line.substr(pos,end==std::string::npos?end:end-pos));
  if(end==std::string::npos)break;
  pos=end+1;
 }
 return values;
}
static std::uint64_t decimal(const std::string& input) {
 std::uint64_t output=0;
 if(input.empty())throw std::runtime_error("empty event clock");
 const auto result=std::from_chars(input.data(),input.data()+input.size(),output);
 if(result.ec!=std::errc{}||result.ptr!=input.data()+input.size())
  throw std::runtime_error("invalid event clock");
 return output;
}
static void print_heads(const std::array<unsigned,6>& arr){
 for(std::size_t i=0;i<6;++i){if(i)std::cout<<',';std::cout<<arr[i];}
}
int main(int argc,char**argv){
 if(argc!=2){std::cerr<<"usage: sensor-role-probe SNAPSHOT.sensor.tsv\n";return 2;}
 try{
  const std::string path=argv[1];
  SensorMemory checked;
  checked.load(path); // Native full journal semantic/provenance validation.
  std::ifstream file(path,std::ios::binary);
  if(!file)throw std::runtime_error("journal not readable");
  std::string line;
  if(!std::getline(file,line)||line.rfind("NANLM_SENSOR_V2\t",0)!=0)
   throw std::runtime_error("requires V2 source journal");
  const std::uint64_t final_tick=decimal(line.substr(16));
  if(final_tick!=checked.clock())throw std::runtime_error("journal end-clock mismatch");
  std::uint64_t last=0;
  std::size_t count=0;
  while(std::getline(file,line)){
   auto v=fields(line);
   if(v.size()!=7)throw std::runtime_error("invalid V2 assertion fields");
   const auto tick=decimal(v[6]);
   if(tick<=last||tick>final_tick)throw std::runtime_error("nonmonotonic record tick");
   last=tick;++count;
   std::cout<<"NLRP1\t"<<tick<<'\t';
   print_heads(heads_for(v[2],0x5355424A454354ULL));
   std::cout<<'\t';
   print_heads(heads_for(v[3],0x50524544494354ULL));
   std::cout<<'\t';
   print_heads(heads_for(v[4],0x4F424A454354ULL));
   std::cout<<'\n';
  }
  if(!file.eof()||count!=checked.journal_size())
   throw std::runtime_error("incomplete numeric role trace");
  return 0;
 }catch(const std::exception&e){std::cerr<<"role-probe rejected: "<<e.what()<<"\n";return 1;}
}