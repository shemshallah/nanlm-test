#include "nanlm/sensor_timeslices.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
using namespace nanlm;
using namespace nanlm::math_sensor;
static void check(bool value,const char* message){
 if(!value)throw std::runtime_error(message);
}
static bool close(float a,float b,float tol=1e-6f){
 return std::abs(a-b)<=tol*std::max(1.0f,std::max(std::abs(a),std::abs(b)));
}
static void equal_vectors(const MatrixMeasurement& a,const MatrixMeasurement& b){
 for(std::size_t i=0;i<3;++i)check(close(a.vector[i],b.vector[i]),"vector time-slice reconstruction drift");
 for(std::size_t i=0;i<9;++i)check(close(a.tensor[i],b.tensor[i]),"tensor time-slice reconstruction drift");
 check(close(a.support,b.support)&&close(a.refute,b.refute),"signed sensor charges not reconstructed");
}
int main(){
 const auto dir=std::filesystem::temp_directory_path();
 const auto filename=(dir/("nanlm-matrix-slices-"+std::to_string(::getpid()))).string();
 const auto native=filename+".native";
 auto cleanup=[&](){
  for(const auto& path:{filename,native,native+".overlay",native+".so3",native+".overlay.tmp"})
   std::filesystem::remove(path);
 };
 try {
  SensorMemory sensor(128,2000,0.8);
  sensor.set_hot_budget(8);
  const auto blank=measure_matrix(sensor);
  check(blank.nanclock==0&&blank.heads.empty()&&blank.triads.empty()&&blank.hot.empty(),
        "blank sensor unexpectedly produces active vector or tensor rows");
  check(blank.global_field.id==255,"global sensor marker missing");
  const std::vector<Assertion> observations={
   {"source_1","penguin","ISA","bird",false,1.},
   {"source_2","bird","ISA","animal",false,1.},
   {"source_3","penguin","ISA","animal",true,1.},
   {"source_4","penguin","CAN_FLY","yes",true,1.},
   {"source_5","bird","HAS_PART","wing",false,1.}
  };
  for(const auto& o:observations)check(sensor.ingest(o),"duplicate source in fixture");
  auto snapshot=measure_matrix(sensor);
  check(snapshot.heads.size()>=6&&snapshot.triads.size()>0&&snapshot.hot.size()>0,
        "four-resolution sensor measurement lacks active vectors");
  check(snapshot.source_count==5&&snapshot.fact_count==5,"graph/provenance metadata missing");
  bool head_vec=false,head_ten=false,tri_vec=false,tri_ten=false;
  for(const auto& h:snapshot.heads){
   for(auto x:h.vector)head_vec|=std::abs(x)>1e-7f;
   for(auto x:h.tensor)head_ten|=std::abs(x)>1e-7f;
   for(int a=0;a<3;++a)for(int b=0;b<3;++b)
    check(close(h.tensor[3*a+b],h.tensor[3*b+a]),"head tensor is not symmetric");
   check(h.id==h.heads[0]&&h.heads[1]==255&&h.heads[2]==255,
         "head vector row lost fixed-point address");
  }
  for(const auto& t:snapshot.triads){
   for(auto x:t.vector)tri_vec|=std::abs(x)>1e-7f;
   for(auto x:t.tensor)tri_ten|=std::abs(x)>1e-7f;
   check(t.heads==sensor.triad_heads(t.id),"triad vector row lost adaptive heads");
  }
  check(head_vec&&head_ten&&tri_vec&&tri_ten,
        "vector/tensor sensor measurement is degenerate");
  // The full-field vector is the mean across all 85 groups; the hot field
  // is an independent salience-sensitive readout of only selected groups.
  std::array<double,3> sum{};
  for(const auto& t:snapshot.triads)
   for(std::size_t k=0;k<3;++k)sum[k]+=t.vector[k];
  for(std::size_t k=0;k<3;++k)
   check(close(snapshot.global_field.vector[k],float(sum[k]/85.)),
         "global vector not the full triad mean");
  const auto raw=encode_matrix(snapshot);
  auto recovered=decode_matrix(raw);
  check(encode_matrix(recovered)==raw,"explicit little-endian numerical codec not bijective");
  check(recovered.nanclock==snapshot.nanclock&&recovered.source_count==5,
        "nanclock metadata not reconstructible");
  check(recovered.hot.size()==snapshot.hot.size(),"hot layer dropped in time slice");
  for(std::size_t i=0;i<snapshot.heads.size();++i)
   equal_vectors(snapshot.heads[i],recovered.heads[i]);
  for(std::size_t i=0;i<snapshot.triads.size();++i)
   equal_vectors(snapshot.triads[i],recovered.triads[i]);
  for(std::size_t i=0;i<snapshot.hot.size();++i)
   equal_vectors(snapshot.hot[i],recovered.hot[i]);
  equal_vectors(snapshot.global_field,recovered.global_field);
  equal_vectors(snapshot.hot_field,recovered.hot_field);
  // Every sparse mirror-matrix edge resolves to EXACTLY the same native
  // physical pair cell under reverse16, with independent directed values.
  {
   Universe universe(native);
   for(const auto& edge:recovered.mirrors){
    check(edge.from<255&&edge.to<255&&edge.from!=edge.to,
          "mirror edge addressed the reserved ingress");
    check(edge.forward==address(edge.from,edge.to)&&edge.reverse==mirror(edge.forward),
          "mirror and inverse address mismatch");
    check(edge.physical_offset==universe.leaf(edge.forward).physical_offset&&
          edge.physical_offset==universe.leaf(edge.reverse).physical_offset,
          "forward and reverse sensor measure different canonical pair cells");
    check(edge.physical_offset==offsetof(Root,p)+pair_index(edge.from,edge.to)*sizeof(Pair),
          "canonical pair index uses incorrect matrix coordinate");
    check(std::isfinite(edge.oriented_flux),"nonfinite directed vector flux");
   }
   check(std::filesystem::file_size(native)==ROOT_BYTES,"sensor modified native root size");
  }
  // Enforce fixed-size bounded storage and monotonic sequence across wrap/reopen.
  std::uint64_t last_sequence=0;
  {
   TemporalMatrixLog trace(filename,65536);
   check(std::filesystem::file_size(filename)==65536,"trace is not fixed-size");
   check(trace.capture(sensor)==1,"new trace sequence must start at one");
   last_sequence=1;
   for(int i=0;i<85;++i){
    sensor.tick();
    last_sequence=trace.capture(sensor);
   }
   check(last_sequence==86,"periodic trace append lost NaNclock slices");
   check(std::filesystem::file_size(filename)==65536,"ring growth exceeded trace budget");
   const auto last=trace.latest();
   check(last.seq==last_sequence&&last.frame.nanclock==sensor.clock(),
         "latest time-slice index not reconstructed");
   check(!last.frame.heads.empty(),"rolling trace lost current vector state");
   const auto recent=trace.recent(32);
   check(recent.size()>1&&recent.size()<=32,"ring scan did not find recent slices");
   for(std::size_t i=1;i<recent.size();++i)
    check(recent[i].seq>recent[i-1].seq,"ring sequence not monotonic");
  }
  {
   TemporalMatrixLog trace(filename,65536);
   check(trace.next_sequence()==last_sequence+1,"ring restart lost monotonic index");
   const auto valid=trace.latest();
   check(valid.seq==last_sequence,"restart lost latest valid temporal matrix");
   sensor.tick();
   last_sequence=trace.capture(sensor);
   check(last_sequence==87,"resume did not append next expected frame");
   // Introduce a deliberately torn payload; recovery must not return it
   // as an apparently valid observation, but preserve an earlier valid one.
   const auto latest=trace.latest();
   std::fstream file(filename,std::ios::in|std::ios::out|std::ios::binary);
   file.seekp(latest.offset+TemporalMatrixLog::RECORD_HEADER+8);
   char corrupt=0xff;
   file.write(&corrupt,1);file.flush();
   check(bool(file),"corruption test write failed");
   file.close();
   const auto recovered_last=trace.latest();
   check(recovered_last.seq<last_sequence,"checksum accepted corrupted temporal frame");
  }
  // A new recording is no longer permitted to silently use a mismatched
  // capacity, even when both files happen to have valid initial records.
  bool rejected=false;
  try{TemporalMatrixLog different(filename,131072);(void)different;}
  catch(const std::runtime_error&){rejected=true;}
  check(rejected,"ring accepted incompatible storage dimensions");
  cleanup();
  std::cout<<"PASS: four-resolution vector/tensor measurements, NaN16 mirror matrix, time-slice binary roundtrip, 64 KiB bounded wrap, restart and torn-frame recovery\n";
  return 0;
 }catch(const std::exception& e){
  cleanup();
  std::cerr<<"FAIL: "<<e.what()<<"\n";
  return 1;
 }
}
