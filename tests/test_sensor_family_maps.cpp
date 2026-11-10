#include "nanlm/sensor_family_maps.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace nanlm::math_sensor;

static void check(bool value,const char* reason) {
 if(!value)throw std::runtime_error(reason);
}
static bool near(float a,double b,double tol=1e-5) {
 return std::abs(double(a)-b)<=tol*std::max(1.,std::abs(b));
}
int main() {
 try {
  SensorMemory corpus(128,2000,0.8);
  const std::vector<Assertion> statements={
   {"one","penguin","ISA","bird",false,1.},
   {"two","bird","ISA","animal",false,1.},
   {"three","penguin","ISA","animal",true,1.},
   {"four","penguin","CAN_FLY","yes",true,1.},
   {"five","bird","HAS_PART","wing",false,1.}
  };
  for(const auto& a:statements)check(corpus.ingest(a),"sensor corpus ingest failed");
  const auto original=measure_matrix(corpus);
  check(!original.heads.empty()&&!original.triads.empty()&&!original.hot.empty(),
        "source corpus has no measured hierarchy");
  const auto before=encode_matrix(original);
  const std::array<SensorFamily,8> families={
   SensorFamily::Vector,SensorFamily::Tensor,SensorFamily::Rotation,
   SensorFamily::Spectral,SensorFamily::Temporal,SensorFamily::Topology,
   SensorFamily::Information,SensorFamily::ComplexAmplitude
  };
  std::size_t total_bytes=0;
  for(const auto family:families) {
   FamilyProjector projector;
   const auto result=projector.measure(family,original);
   check(result.nanclock==original.nanclock&&result.sources==5&&result.facts==5,
         "family lost original time/provenance coordinates");
   check(result.points.size()==original.heads.size()&&
         result.triads.size()==original.triads.size()&&
         result.hot.size()==original.hot.size(),
         "family discarded a resolution of the hierarchy");
   check(result.global.id==255&&result.hot_field.id==255,
         "family field marker is not reserved");
   for(std::size_t i=0;i<result.hot.size();++i)
    check(result.hot[i].id==original.hot[i].id&&
          result.hot[i].heads==original.hot[i].heads,
          "hot group identity changed to ephemeral slot");
   const auto bytes=encode_family(result);
   check(bytes.size()<16384,"family frame exceeds planned 16 KiB slot");
   total_bytes+=bytes.size();
   const auto decoded=decode_family(bytes);
   check(encode_family(decoded)==bytes,"family endian codec not bijective");
   check(encode_matrix(original)==before,"family changed shared input state");
   if(family==SensorFamily::Vector) {
    for(const auto& x:result.points) {
     const double n=x.values[0];
     const double dir=double(x.values[1])*x.values[1]+
                      double(x.values[2])*x.values[2]+
                      double(x.values[3])*x.values[3];
     check(n>=0&&near(static_cast<float>(dir),n>0?1.:0.),
           "vector direction not normalized");
    }
   }
   if(family==SensorFamily::Tensor) {
    for(const auto& x:result.points)
     check(x.values[0]>=-1e-5f&&x.values[1]>=0&&x.values[3]>=0,
           "positive-semidefinite tensor invariants failed");
   }
   if(family==SensorFamily::Information) {
    bool found_conflict=false;
    for(const auto& x:result.points) {
     check(x.values[0]>=0&&x.values[0]<=1.00001f,
           "binary Shannon entropy outside range");
     if(x.support>0&&x.refute>0){
      found_conflict=true;
      check(x.values[0]>0,"coexisting source evidence erased by entropy sensor");
     }
    }
    check(found_conflict,"source conflict fixture not reaching a head");
   }
   if(family==SensorFamily::ComplexAmplitude) {
    for(const auto* rows:{&result.points,&result.triads,&result.hot}) {
     double total=0;
     for(const auto& r:*rows)total+=r.values[2];
     check(near(static_cast<float>(total),1.),"squared complex magnitudes not normalized");
    }
   }
   std::cout<<"FAMILY "<<int(static_cast<std::uint8_t>(family))
            <<" bytes="<<bytes.size()
            <<" heads="<<result.points.size()
            <<" triads="<<result.triads.size()
            <<" hot="<<result.hot.size()
            <<" mirrors="<<result.active_links<<"\n";
  }
  // Independent spectral and time-derivative temporal probes.
  FamilyProjector spectral, temporal,rotational;
  MatrixTimeSlice frame{};
  MatrixMeasurement p{};
  p.id=4;p.heads={4,255,255};p.vector={1,0,0};
  frame.heads.push_back(p);
  for(std::uint64_t t=1;t<=8;++t) {
   frame.nanclock=t;
   frame.heads[0].vector[0]=float(2-std::cos(6.2831853071795864769*double(t)/4.));
   auto s=spectral.measure(SensorFamily::Spectral,frame);
   if(t==8)check(s.points[0].values[2]>0.01f&&s.points[0].values[5]==8.f,
                 "periodic signal frequency bin not detected");
  }
  frame.nanclock=40;
  auto gap=spectral.measure(SensorFamily::Spectral,frame);
  check(gap.points[0].values[5]==1.f&&gap.points[0].values[4]==0.f,
        "spectral sensor mislabels irregular samples as uniform frequency");
  frame.nanclock=1;frame.heads[0].vector={1,0,0};
  (void)temporal.measure(SensorFamily::Temporal,frame);
  (void)rotational.measure(SensorFamily::Rotation,frame);
  frame.nanclock=4;frame.heads[0].vector={0,4,0};
  auto td=temporal.measure(SensorFamily::Temporal,frame);
  auto tr=rotational.measure(SensorFamily::Rotation,frame);
  check(near(td.points[0].values[1],1.)&&near(td.points[0].values[3],3.),
        "temporal derivative ignores actual NaNclock gap");
  check(near(tr.points[0].values[1],(3.14159265358979323846/2.)/3.),
        "phase velocity does not divide by NaNclock gap");
  // Malformed family schema and nonfinite binary32 payloads must fail.
  auto valid=encode_family(spectral.measure(SensorFamily::Spectral,frame));
  valid[4]=99;
  bool rejected=false;
  try{(void)decode_family(valid);}catch(const std::runtime_error&){rejected=true;}
  check(rejected,"out-of-range family code accepted");
  std::cout<<"FAMILY_TOTAL_SAMPLE_PAYLOAD_BYTES="<<total_bytes<<"\n";
  std::cout<<"PASS: eight math sensors with independent time-slice metrics, shared head/triad/hot/field coordinates, phase, spectrum, entropy, and normalized complex amplitudes\n";
 } catch(const std::exception& e) {
  std::cerr<<"FAIL: "<<e.what()<<"\n";
  return 1;
 }
 return 0;
}
