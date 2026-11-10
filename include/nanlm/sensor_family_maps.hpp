#pragma once
// Independent mathematical sensor channels over one event-aligned four-level
// observation hierarchy. Experimental host-side reference; no native writes.
#include "nanlm/sensor_timeslices.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <deque>
#include <map>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace nanlm::math_sensor {

enum class SensorFamily : std::uint8_t {
 Vector=1, Tensor=2, Rotation=3, Spectral=4, Temporal=5,
 Topology=6, Information=7, ComplexAmplitude=8
};
enum class SensorScale : std::uint8_t { Point=0, Triad=1, HotTriad=2, Global=3, HotField=4 };

struct FamilyRow {
 std::uint8_t id=255;
 std::array<std::uint8_t,3> heads{255,255,255};
 std::array<float,6> values{}; // meanings depend on family; see docs
 float support=0,refute=0;
};
struct FamilyFrame {
 SensorFamily family=SensorFamily::Vector;
 std::uint64_t nanclock=0;
 std::uint32_t sources=0,facts=0;
 std::vector<FamilyRow> points,triads,hot;
 FamilyRow global{},hot_field{};
 std::uint32_t active_links=0;
};

inline double safe_norm(const std::array<float,3>& v) {
 double sum=0;
 for(float x:v)sum+=double(x)*x;
 return std::sqrt(sum);
}
inline double finite_value(double x) {
 if(!std::isfinite(x))throw std::runtime_error("nonfinite derived family metric");
 return x;
}
inline double phase_angle(double x,double y) {
 return (x==0&&y==0)?0:std::atan2(y,x);
}
inline double wrap_angle(double x) { return std::atan2(std::sin(x),std::cos(x)); }
inline float as_sensor_float(double x) { return sensor_float(finite_value(x)); }

class FamilyProjector {
 struct History {
  std::uint64_t tick=0;
  double norm=0,phase=0,derivative=0;
  std::deque<double> consecutive_norms;
  bool has_previous=false;
 };
 // Stable original IDs and separate scales: hot triad ID is NOT a hot slot.
 std::map<std::pair<int,int>,History> history_;
 static int key(SensorScale level, std::uint8_t id) {
  return int(level)*256+id;
 }
 static double shannon(double a,double b) {
  if(a<0||b<0)throw std::invalid_argument("negative evidence activity");
  const double n=a+b;
  if(n<=0)return 0;
  const auto term=[n](double z){return z<=0?0:-(z/n)*std::log2(z/n);};
  return term(a)+term(b);
 }
 static double dft_power(const std::deque<double>& x,int bin) {
  const int n=static_cast<int>(x.size());
  if(n<2)return 0;
  double re=0,im=0,mean=std::accumulate(x.begin(),x.end(),0.)/n;
  constexpr double tau=6.2831853071795864769;
  for(int j=0;j<n;++j) {
   const double angle=tau*double(bin*j)/n;
   re+=(x[static_cast<std::size_t>(j)]-mean)*std::cos(angle);
   im-=(x[static_cast<std::size_t>(j)]-mean)*std::sin(angle);
  }
  return (re*re+im*im)/double(n*n);
 }
 static std::vector<MatrixMeasurement> full(const MatrixTimeSlice& f,SensorScale s) {
  if(s==SensorScale::Point)return f.heads;
  if(s==SensorScale::Triad)return f.triads;
  if(s==SensorScale::HotTriad)return f.hot;
  return s==SensorScale::Global?std::vector<MatrixMeasurement>{f.global_field}:
                                std::vector<MatrixMeasurement>{f.hot_field};
 }
public:
 // Each projection is independent: call a distinct projector per family;
 // the input event sequence and map IDs must be shared across projectors.
 FamilyFrame measure(SensorFamily family,const MatrixTimeSlice& frame) {
  FamilyFrame out{};
  out.family=family;out.nanclock=frame.nanclock;
  out.sources=frame.source_count;out.facts=frame.fact_count;
  out.active_links=static_cast<std::uint32_t>(frame.mirrors.size());
  for(int s=0;s<5;++s) {
   const auto scale=static_cast<SensorScale>(s);
   const auto rows=full(frame,scale);
   auto* output = scale==SensorScale::Point?&out.points:
                  scale==SensorScale::Triad?&out.triads:
                  scale==SensorScale::HotTriad?&out.hot:nullptr;
   double amplitude_denom=0;
   if(family==SensorFamily::ComplexAmplitude) {
    for(const auto& r:rows)
     amplitude_denom+=double(r.vector[0])*r.vector[0]+
                      double(r.vector[1])*r.vector[1];
   }
   std::array<int,85> degrees{};
   if(family==SensorFamily::Topology) {
    for(const auto& e:frame.mirrors) {
     if(e.group_a<85)++degrees[e.group_a];
     if(e.group_b<85)++degrees[e.group_b];
    }
   }
   for(const auto& r:rows) {
    FamilyRow x{};x.id=r.id;x.heads=r.heads;
    x.support=r.support;x.refute=r.refute;
    const double norm=safe_norm(r.vector);
    const double angle=phase_angle(r.vector[0],r.vector[1]);
    auto& h=history_[{int(family),key(scale,r.id)}];
    const bool advanced=h.has_previous&&frame.nanclock>h.tick;
    const double dt=advanced?double(frame.nanclock-h.tick):0;
    const double velocity=advanced?wrap_angle(angle-h.phase)/dt:0;
    if(family==SensorFamily::Vector) {
     x.values={as_sensor_float(norm),
               as_sensor_float(norm?double(r.vector[0])/norm:0),
               as_sensor_float(norm?double(r.vector[1])/norm:0),
               as_sensor_float(norm?double(r.vector[2])/norm:0),
               as_sensor_float(angle),as_sensor_float(r.activity)};
    } else if(family==SensorFamily::Tensor) {
     const auto& t=r.tensor;
     double frob2=0;for(float y:t)frob2+=double(y)*y;
     const double trace=double(t[0])+t[4]+t[8];
     const double det=double(t[0])*(double(t[4])*t[8]-double(t[5])*t[7])-
                      double(t[1])*(double(t[3])*t[8]-double(t[5])*t[6])+
                      double(t[2])*(double(t[3])*t[7]-double(t[4])*t[6]);
     const double anis=std::sqrt(std::max(0.,frob2-trace*trace/3.));
     x.values={as_sensor_float(trace),as_sensor_float(std::sqrt(frob2)),
               as_sensor_float(det),as_sensor_float(anis),t[0],t[1]};
    } else if(family==SensorFamily::Rotation) {
     x.values={as_sensor_float(angle),as_sensor_float(velocity),
               as_sensor_float(std::cos(angle)),as_sensor_float(std::sin(angle)),
               as_sensor_float(norm),as_sensor_float(advanced?1.:0.)};
    } else if(family==SensorFamily::Spectral) {
     // Frequencies are cycles per NaNclock tick ONLY for consecutive ticks.
     if(!h.has_previous||frame.nanclock!=h.tick+1)h.consecutive_norms.clear();
     if(!h.has_previous||frame.nanclock>h.tick) {
      h.consecutive_norms.push_back(norm);
      if(h.consecutive_norms.size()>8)h.consecutive_norms.pop_front();
     }
     const double p1=dft_power(h.consecutive_norms,1);
     const double p2=dft_power(h.consecutive_norms,2);
     const double p3=dft_power(h.consecutive_norms,3);
     const int dominant=(p1>=p2&&p1>=p3)?1:((p2>=p3)?2:3);
     const double dc=h.consecutive_norms.empty()?0:
       std::accumulate(h.consecutive_norms.begin(),h.consecutive_norms.end(),0.)/
       h.consecutive_norms.size();
     x.values={as_sensor_float(dc),as_sensor_float(p1),as_sensor_float(p2),
               as_sensor_float(p3),as_sensor_float(h.consecutive_norms.size()>=4?dominant:0),
               as_sensor_float(h.consecutive_norms.size())};
    } else if(family==SensorFamily::Temporal) {
     const double slope=advanced?(norm-h.norm)/dt:0;
     const double acceleration=advanced?(slope-h.derivative)/dt:0;
     x.values={as_sensor_float(norm),as_sensor_float(slope),
               as_sensor_float(acceleration),as_sensor_float(advanced?dt:0),
               as_sensor_float(h.has_previous?std::exp(-dt/8.):0),
               as_sensor_float(r.prior)};
     if(!h.has_previous||frame.nanclock>h.tick)h.derivative=slope;
    } else if(family==SensorFamily::Topology) {
     const int degree=r.id<85&&(scale==SensorScale::Triad||scale==SensorScale::HotTriad)?
       degrees[r.id]:0;
     x.values={as_sensor_float(degree),as_sensor_float(r.curvature),
               as_sensor_float(frame.mirrors.size()),as_sensor_float(norm),
               as_sensor_float(r.activity),0};
    } else if(family==SensorFamily::Information) {
     const double total=double(r.support)+r.refute;
     x.values={as_sensor_float(shannon(r.support,r.refute)),
               as_sensor_float(total>0?std::min(r.support,r.refute)/total:0),
               as_sensor_float(total),as_sensor_float(std::log1p(norm)),
               as_sensor_float(r.activity),as_sensor_float(frame.source_count)};
    } else if(family==SensorFamily::ComplexAmplitude) {
     const double den=std::sqrt(amplitude_denom);
     x.values={as_sensor_float(den>0?r.vector[0]/den:0),
               as_sensor_float(den>0?r.vector[1]/den:0),
               as_sensor_float(amplitude_denom>0?
                 (double(r.vector[0])*r.vector[0]+double(r.vector[1])*r.vector[1])/amplitude_denom:0),
               as_sensor_float(angle),as_sensor_float(amplitude_denom),
               as_sensor_float(amplitude_denom>0?1:0)};
    }
    if(!h.has_previous||frame.nanclock>h.tick) {
     h.tick=frame.nanclock;h.phase=angle;h.norm=norm;h.has_previous=true;
    }
    if(output)output->push_back(x);
    else if(scale==SensorScale::Global)out.global=x;
    else out.hot_field=x;
   }
  }
  return out;
 }
};

// Portable little-endian sparse numeric export. Family and scale are explicit;
// no single canonical "semantic" coordinate system is imposed on outputs.
inline std::vector<std::uint8_t> encode_family(const FamilyFrame& f) {
 if(f.points.size()>255||f.triads.size()>85||f.hot.size()>12)
  throw std::invalid_argument("family measurements exceed sensor topology");
 SliceWriter w;
 for(char c:std::string("FAM1"))w.u8(static_cast<std::uint8_t>(c));
 w.u8(static_cast<std::uint8_t>(f.family));w.u8(1);w.u16(0);
 w.u64(f.nanclock);w.u32(f.sources);w.u32(f.facts);w.u32(f.active_links);
 for(auto n:{f.points.size(),f.triads.size(),f.hot.size()})
  w.u16(static_cast<std::uint16_t>(n));
 const auto row=[&](const FamilyRow& r){
  w.u8(r.id);for(auto h:r.heads)w.u8(h);
  for(float x:r.values)w.f32(x);
  w.f32(r.support);w.f32(r.refute);
 };
 for(const auto& r:f.points)row(r);
 for(const auto& r:f.triads)row(r);
 for(const auto& r:f.hot)row(r);
 row(f.global);row(f.hot_field);
 return w.bytes();
}
inline FamilyFrame decode_family(const std::vector<std::uint8_t>& bytes) {
 SliceReader r(bytes);
 if(r.u8()!='F'||r.u8()!='A'||r.u8()!='M'||r.u8()!='1')
  throw std::runtime_error("invalid family magic");
 const auto tag=r.u8();
 if(tag<1||tag>8||r.u8()!=1||r.u16()!=0)
  throw std::runtime_error("invalid family schema");
 FamilyFrame f{};f.family=static_cast<SensorFamily>(tag);
 f.nanclock=r.u64();f.sources=r.u32();f.facts=r.u32();f.active_links=r.u32();
 const auto np=r.u16(),nt=r.u16(),nh=r.u16();
 if(np>255||nt>85||nh>12)throw std::runtime_error("invalid family row count");
 const auto row=[&](){
  FamilyRow x{};x.id=r.u8();
  for(auto& h:x.heads)h=r.u8();
  for(auto& v:x.values)v=r.f32();
  x.support=r.f32();x.refute=r.f32();
  return x;
 };
 std::array<bool,255> seen_p{};std::array<bool,85> seen_t{},seen_h{};
 for(int i=0;i<np;++i) {
  auto x=row();
  if(x.id>=255||seen_p[x.id]||x.heads[0]!=x.id)
   throw std::runtime_error("invalid family head index");
  seen_p[x.id]=true;f.points.push_back(x);
 }
 for(int i=0;i<nt;++i) {
  auto x=row();
  if(x.id>=85||seen_t[x.id])throw std::runtime_error("invalid family triad index");
  seen_t[x.id]=true;f.triads.push_back(x);
 }
 for(int i=0;i<nh;++i) {
  auto x=row();
  if(x.id>=85||seen_h[x.id])throw std::runtime_error("invalid family hot index");
  seen_h[x.id]=true;f.hot.push_back(x);
 }
 f.global=row();f.hot_field=row();
 if(f.global.id!=255||f.hot_field.id!=255)
  throw std::runtime_error("invalid family field marker");
 r.end();
 return f;
}
} // namespace nanlm::math_sensor
