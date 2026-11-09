#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace nanlm {
constexpr size_t SO3_PAIRS=32640;
constexpr double SO3_PI=3.1415926535897932384626433832795;
struct Vec3q {double x=0,y=0,z=0;};
inline Vec3q cross(Vec3q a,Vec3q b) {return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};}
inline double dot(Vec3q a,Vec3q b){return a.x*b.x+a.y*b.y+a.z*b.z;}
inline double length(Vec3q a){return std::sqrt(dot(a,a));}
inline Vec3q unit(Vec3q v){
 const double n=length(v);
 if(n<1e-15)return {1,0,0};
 return {v.x/n,v.y/n,v.z/n};
}
struct alignas(16) Quat {
 float w=1,x=0,y=0,z=0;
 static Quat identity(){return {};}
 Quat conjugate()const{return {w,-x,-y,-z};}
 Quat normalized()const{
  const double n=std::sqrt(double(w)*w+double(x)*x+double(y)*y+double(z)*z);
  if(!std::isfinite(n)||n<1e-30)throw std::runtime_error("invalid quaternion");
  return {float(w/n),float(x/n),float(y/n),float(z/n)};
 }
 Quat operator*(Quat b)const{
  return Quat{
   float(double(w)*b.w-double(x)*b.x-double(y)*b.y-double(z)*b.z),
   float(double(w)*b.x+double(x)*b.w+double(y)*b.z-double(z)*b.y),
   float(double(w)*b.y-double(x)*b.z+double(y)*b.w+double(z)*b.x),
   float(double(w)*b.z+double(x)*b.y-double(y)*b.x+double(z)*b.w)
  }.normalized();
 }
 Vec3q rotate(Vec3q v)const{
  const Vec3q q{double(x),double(y),double(z)};
  const Vec3q c=cross(q,v);
  const Vec3q t{2*c.x,2*c.y,2*c.z};
  const Vec3q k=cross(q,t);
  return {v.x+double(w)*t.x+k.x,v.y+double(w)*t.y+k.y,v.z+double(w)*t.z+k.z};
 }
 static Quat axis_angle(Vec3q axis,double angle){
  if(!std::isfinite(angle))throw std::invalid_argument("nonfinite angle");
  const Vec3q a=unit(axis);
  const double s=std::sin(angle/2);
  return Quat{float(std::cos(angle/2)),float(s*a.x),float(s*a.y),float(s*a.z)}.normalized();
 }
 static Quat from_to(Vec3q src,Vec3q dst){
  if(length(src)<1e-12||length(dst)<1e-12)return identity();
  const Vec3q a=unit(src),b=unit(dst);
  const double c=std::clamp(dot(a,b),-1.0,1.0);
  if(c>1-1e-10)return identity();
  if(c<-1+1e-10){
   Vec3q n=cross(a,std::fabs(a.x)<0.8?Vec3q{1,0,0}:Vec3q{0,1,0});
   return axis_angle(n,SO3_PI);
  }
  const Vec3q axis=cross(a,b);
  const double w=std::sqrt(2*(1+c));
  return Quat{float(w/2),float(axis.x/w),float(axis.y/w),float(axis.z/w)}.normalized();
 }
 double angle()const{
  return 2*std::acos(std::clamp(std::fabs(double(normalized().w)),0.0,1.0));
 }
 double distance(Quat other)const{return (conjugate()*other).angle();}
 static Quat slerp(Quat a,Quat b,double rate){
  if(rate<=0)return a;
  if(rate>=1)return b;
  double cosine=double(a.w)*b.w+double(a.x)*b.x+double(a.y)*b.y+double(a.z)*b.z;
  if(cosine<0){b={-b.w,-b.x,-b.y,-b.z};cosine=-cosine;}
  cosine=std::clamp(cosine,0.,1.);
  double a_weight=1-rate,b_weight=rate;
  if(cosine<0.9995){
   const double theta=std::acos(cosine),sin_theta=std::sin(theta);
   a_weight=std::sin((1-rate)*theta)/sin_theta;
   b_weight=std::sin(rate*theta)/sin_theta;
  }
  return Quat{float(a_weight*a.w+b_weight*b.w),float(a_weight*a.x+b_weight*b.x),
              float(a_weight*a.y+b_weight*b.y),float(a_weight*a.z+b_weight*b.z)}.normalized();
 }
};
static_assert(sizeof(Quat)==16);
struct alignas(64) QuaternionHeader {
 char magic[8]={'N','A','N','S','O','3','0','1'};
 uint32_t version=1;
 uint32_t cells=SO3_PAIRS;
 uint8_t reserved[48]{};
};
static_assert(sizeof(QuaternionHeader)==64);
struct QuaternionMap {
 QuaternionHeader header;
 Quat cells[SO3_PAIRS];
};
static_assert(sizeof(QuaternionMap)==64+16*SO3_PAIRS);
class QuaternionField {
 std::string path_;
 int fd_=-1;
 QuaternionMap* map_=nullptr;
 static size_t index(uint8_t a,uint8_t b){
  if(a==b)throw std::invalid_argument("diagonal quaternion is identity");
  if(a>b)std::swap(a,b);
  return size_t(a)*(511u-a)/2u +b-a-1u;
 }
 public:
 explicit QuaternionField(std::string path):path_(std::move(path)){
  fd_=::open(path_.c_str(),O_CREAT|O_RDWR,0600);
  if(fd_<0)throw std::runtime_error("cannot open quaternion field");
  struct stat st{};
  if(::fstat(fd_,&st)!=0)throw std::runtime_error("quaternion stat failed");
  const bool fresh=st.st_size==0;
  if(fresh&&::ftruncate(fd_,sizeof(QuaternionMap)))throw std::runtime_error("quaternion allocation failed");
  if(!fresh&&st.st_size!=static_cast<off_t>(sizeof(QuaternionMap)))throw std::runtime_error("wrong quaternion file length");
  void* p=::mmap(nullptr,sizeof(QuaternionMap),PROT_READ|PROT_WRITE,MAP_SHARED,fd_,0);
  if(p==MAP_FAILED)throw std::runtime_error("quaternion mmap failed");
  map_=static_cast<QuaternionMap*>(p);
  if(fresh){
   map_->header=QuaternionHeader{};
   for(auto& q:map_->cells)q=Quat::identity();
  }else if(std::memcmp(map_->header.magic,"NANSO301",8)!=0 ||
           map_->header.version!=1 || map_->header.cells!=SO3_PAIRS)
   throw std::runtime_error("wrong quaternion format");
 }
 QuaternionField(const QuaternionField&)=delete;
 QuaternionField& operator=(const QuaternionField&)=delete;
 ~QuaternionField(){if(map_)::munmap(map_,sizeof(QuaternionMap));if(fd_>=0)::close(fd_);}
 static constexpr size_t mapped_bytes(){return sizeof(QuaternionMap);}
 Quat get(uint8_t a,uint8_t b)const{
  if(a==b)return Quat::identity();
  const Quat q=map_->cells[index(a,b)];
  return a<b?q:q.conjugate();
 }
 void set(uint8_t a,uint8_t b,Quat q){
  if(a==b)throw std::invalid_argument("self loop is identity");
  const Quat normalized=q.normalized();
  map_->cells[index(a,b)]=a<b?normalized:normalized.conjugate();
 }
 Quat triangle(uint8_t a,uint8_t b,uint8_t c)const{
  if(a==b||b==c||a==c)throw std::invalid_argument("degenerate quaternion triangle");
  return get(a,b)*get(b,c)*get(c,a);
 }
 double curvature(uint8_t a,uint8_t b,uint8_t c)const{return triangle(a,b,c).angle();}
 // An active change of local coordinate frames, not an information update:
 // q_ab -> G_a q_ab G_b^-1. Triangle holonomy conjugates at its origin.
 void gauge(const std::array<Quat,256>& g){
  for(unsigned a=0;a<256;++a)for(unsigned b=a+1;b<256;++b)
   set(static_cast<uint8_t>(a),static_cast<uint8_t>(b),
       g[a]*get(static_cast<uint8_t>(a),static_cast<uint8_t>(b))*g[b].conjugate());
 }
 void sync()const{if(::msync(map_,sizeof(QuaternionMap),MS_SYNC))throw std::runtime_error("quaternion sync failed");}
};
} // namespace nanlm
