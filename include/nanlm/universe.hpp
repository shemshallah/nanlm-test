#pragma once
#include "nanlm/quaternion.hpp"
#include <array>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace nanlm {
constexpr size_t HEADS=256, PAIRS=32640, LEAVES=65536, ROOT_BYTES=8388608;
constexpr double TAU=6.283185307179586476925286766559;
constexpr uint64_t LATEST=std::numeric_limits<uint64_t>::max();

// A NaN16 address is an OPAQUE 16-bit payload, never evaluated as a floating NaN.
// Physical reverse16 is exactly logical matrix transposition.
inline uint8_t reverse8(uint8_t v) {
 v=static_cast<uint8_t>(((v&0x55u)<<1)|((v&0xaau)>>1));
 v=static_cast<uint8_t>(((v&0x33u)<<2)|((v&0xccu)>>2));
 return static_cast<uint8_t>((v<<4)|(v>>4));
}
inline uint16_t address(uint8_t a,uint8_t b){return static_cast<uint16_t>((a<<8)|reverse8(b));}
inline uint16_t mirror(uint16_t x){return address(reverse8(static_cast<uint8_t>(x)),static_cast<uint8_t>(x>>8));}
inline uint8_t source(uint16_t x){return static_cast<uint8_t>(x>>8);}
inline uint8_t target(uint16_t x){return reverse8(static_cast<uint8_t>(x));}
inline size_t pair_index(uint8_t a,uint8_t b){
 if(a==b)throw std::invalid_argument("diagonal is a head");
 if(a>b)std::swap(a,b);
 return static_cast<size_t>(a)*(511u-a)/2u +b-a-1u;
}
struct alignas(64) Head {uint8_t data[128]{};};
struct alignas(64) Pair {float f[32]{},r[32]{};};
struct Root {Head h[HEADS];Pair p[PAIRS];};
static_assert(sizeof(Head)==128 && sizeof(Pair)==256 && sizeof(Root)==ROOT_BYTES);
struct Leaf {
 uint32_t physical_offset; // offset into exact 8MiB root
 uint16_t raw, inverse;
 uint8_t from,to, direction,kind;
};
static_assert(sizeof(Leaf)==12);
enum Relation:uint8_t {
 ISA=0,HAS_SUBTYPE=1,PART_OF=2,HAS_PART=3,CAUSES=4,CAUSED_BY=5,
 PRECEDES=6,FOLLOWS=7,LOCATED_IN=8,CONTAINS=9,SUPPORTS=10,SUPPORTED_BY=11,
 ASSERTS=12,ASSERTED_BY=13,CONTRADICTS=14,ASSOCIATED=15
};
inline uint8_t inverse_type(uint8_t type){
 if(type>=16)throw std::out_of_range("relation type");
 return type<14?static_cast<uint8_t>(type^1u):type;
}
// Each event has both support/refute polarity and effective-time interval.
// A single provenance-bearing event projects into two inverse directed cache lanes.
struct Event {
 uint64_t since=0,until=0,provenance=0;
 float weight=0;
 uint8_t from=0,to=0,type=0,refutes=0;
 uint32_t reserved=0;
};
struct Hyperedge {
 uint64_t id=0,created=0,provenance=0;
 float strength=0;
 uint8_t a=0,b=0,c=0,relation=0,refutes=0,arity=3;
 uint16_t reserved=0;
};
struct Evidence {float support=0,refute=0;
 uint8_t state()const{return static_cast<uint8_t>((support>0.00001f?1:0)|(refute>0.00001f?2:0));}
};
struct Vec3 {double x=0,y=0,z=0;};
inline Vec3 rotate_x(Vec3 v,double a){const double c=std::cos(a),s=std::sin(a);return {v.x,c*v.y-s*v.z,s*v.y+c*v.z};}
inline Vec3 rotate_y(Vec3 v,double a){const double c=std::cos(a),s=std::sin(a);return {c*v.x+s*v.z,v.y,-s*v.x+c*v.z};}
inline Vec3 rotate_z(Vec3 v,double a){const double c=std::cos(a),s=std::sin(a);return {c*v.x-s*v.y,s*v.x+c*v.y,v.z};}
inline double wrap(double v){return std::atan2(std::sin(v),std::cos(v));}
struct OverlayHeader {
 char magic[8]={'N','M','O','V','L','0','1','\0'};
 uint32_t version=1,events=0,hyperedges=0,phases=PAIRS;
 uint64_t reserved=0;
};
class Universe {
 std::string filename_;
 int fd_=-1;
 Root* root_=nullptr;
 QuaternionField quaternions_;
 std::array<Leaf,LEAVES> leaves_{};
 std::array<double,PAIRS> phase_{};
 std::vector<Event> events_;
 std::vector<Hyperedge> hyper_;
 // Sparse residual of the register-coupled factorized hypermatrix.
 // The full 255^3 tensor is not allocated: head registers and mirror Pair
 // cells supply its factorized representation.
 std::unordered_map<uint64_t,size_t> hyper_index_;
 std::unordered_map<uint32_t,std::vector<size_t>> event_index_;
 static uint32_t key(uint8_t a,uint8_t b,uint8_t t){return (static_cast<uint32_t>(a)<<16)|(static_cast<uint32_t>(b)<<8)|t;}
 static constexpr char identity_[9]="HW0NAN01";
 float& lane(uint8_t a,uint8_t b,uint8_t t,bool refutes) {
  if(a==b)throw std::invalid_argument("no relation slot on fixed head");
  Pair& p=root_->p[pair_index(a,b)];
  return (a<b?p.f:p.r)[t+(refutes?16:0)];
 }
 float lane_value(uint8_t a,uint8_t b,uint8_t t,bool refutes)const {
  if(a==b)return 0;
  const Pair& p=root_->p[pair_index(a,b)];
  return (a<b?p.f:p.r)[t+(refutes?16:0)];
 }
 void project(const Event& e){
  lane(e.from,e.to,e.type,e.refutes!=0)+=e.weight;
  lane(e.to,e.from,inverse_type(e.type),e.refutes!=0)+=e.weight;
 }
 void index_event(size_t i){
  const auto& e=events_[i];
  event_index_[key(e.from,e.to,e.type)].push_back(i);
  event_index_[key(e.to,e.from,inverse_type(e.type))].push_back(i);
 }
 void load_overlay() {
  std::ifstream in(filename_+".overlay",std::ios::binary);
  if(!in)return;
  OverlayHeader h{};
  if(!in.read(reinterpret_cast<char*>(&h),sizeof(h)) ||
   std::memcmp(h.magic,"NMOVL01",7)!=0 || h.version!=1 || h.phases!=PAIRS ||
   h.events>20000000u || h.hyperedges>20000000u)throw std::runtime_error("invalid overlay header");
  in.read(reinterpret_cast<char*>(phase_.data()),sizeof(phase_));
  events_.resize(h.events);hyper_.resize(h.hyperedges);
  if(h.events)in.read(reinterpret_cast<char*>(events_.data()),sizeof(Event)*events_.size());
  if(h.hyperedges)in.read(reinterpret_cast<char*>(hyper_.data()),sizeof(Hyperedge)*hyper_.size());
  if(!in)throw std::runtime_error("truncated overlay");
  for(size_t i=0;i<hyper_.size();++i){
   const auto& x=hyper_[i];
   if(x.a==x.b||x.b==x.c||x.a==x.c||x.relation>=16||
      x.refutes>1||x.arity!=3||!std::isfinite(x.strength)||
      x.strength<=0||x.id!=hyper_id(x.a,x.b,x.c,x.relation,x.refutes!=0)||
      !hyper_index_.emplace(x.id,i).second)
    throw std::runtime_error("invalid hypermatrix overlay");
  }
  for(size_t i=0;i<events_.size();++i){
   const auto& e=events_[i];
   if(e.type>=16||e.from==e.to||e.refutes>1||!std::isfinite(e.weight)||
       e.weight<=0||(e.until&&e.until<=e.since))
    throw std::runtime_error("invalid event");
   index_event(i);
  }
 }
 // Experimental full C(256,2) byte-pair plane is an explicit, isolated
 // root format. Its two directional residuals occupy slot 31 of each Pair.
 // Never zero those persisted floats when there is no graph journal. Normal
 // graph/event roots keep the existing journal-derived cache behavior.
 void rebuild_cache() {
  const bool pair_plane=std::memcmp(root_->h[255].data+112,"PAIRC256",8)==0;
  if(pair_plane){
   if(!events_.empty())throw std::runtime_error(
    "full-pair language root cannot also replay a relation journal");
   return;
  }
  for(auto& cell:root_->p)cell=Pair{};
  for(const auto& e:events_)project(e);
 }
 void build_leaves() {
  for(size_t i=0;i<LEAVES;++i){
   const auto raw=static_cast<uint16_t>(i);
   const uint8_t a=source(raw),b=target(raw);
   const bool fixed=a==b;
   const size_t offset=fixed?(offsetof(Root,h)+a*sizeof(Head)):
     (offsetof(Root,p)+pair_index(a,b)*sizeof(Pair));
   leaves_[i]=Leaf{static_cast<uint32_t>(offset),raw,mirror(raw),
    a,b,static_cast<uint8_t>(a>b),static_cast<uint8_t>(fixed)};
  }
 }
 public:
 explicit Universe(const std::string& path):filename_(path),quaternions_(path+".so3"){
  fd_=::open(path.c_str(),O_CREAT|O_RDWR,0600);
  if(fd_<0)throw std::runtime_error("cannot open root");
  struct stat st{};
  if(::fstat(fd_,&st))throw std::runtime_error("cannot stat root");
  const bool fresh=st.st_size==0;
  if(fresh && ::ftruncate(fd_,ROOT_BYTES))throw std::runtime_error("cannot allocate 8MiB");
  if(!fresh && st.st_size!=static_cast<off_t>(ROOT_BYTES))throw std::runtime_error("wrong root size");
  void* p=::mmap(nullptr,ROOT_BYTES,PROT_READ|PROT_WRITE,MAP_SHARED,fd_,0);
  if(p==MAP_FAILED)throw std::runtime_error("mmap failed");
  root_=static_cast<Root*>(p);
  if(fresh){
   for(auto& head:root_->h)head=Head{};
   for(auto& cell:root_->p)cell=Pair{};
   for(unsigned i=0;i<HEADS;++i){root_->h[i].data[0]=i>>6;root_->h[i].data[1]=(i>>4)&3;root_->h[i].data[2]=i&15;}
   std::memcpy(root_->h[255].data+120,identity_,8);
  }else if(std::memcmp(root_->h[255].data+120,identity_,8)!=0)
   throw std::runtime_error("root not NMOVL01 format: migrate explicitly");
  build_leaves();
  load_overlay();
  if(!fresh)rebuild_cache();
  // Populate page tables now. This does not guarantee OS mlock/physical residency.
  volatile uint8_t touch=0;
  const volatile uint8_t* data=reinterpret_cast<const volatile uint8_t*>(root_);
  for(size_t i=0;i<ROOT_BYTES;i+=4096)touch=static_cast<uint8_t>(touch^data[i]);
  (void)touch;
 }
 Universe(const Universe&)=delete;
 Universe& operator=(const Universe&)=delete;
 ~Universe(){if(root_){::msync(root_,ROOT_BYTES,MS_SYNC);::munmap(root_,ROOT_BYTES);}if(fd_>=0)::close(fd_);}
 const std::string& path()const{return filename_;}
 const Root& root()const{return *root_;}
 Root& root(){return *root_;}
 QuaternionField& quaternion_field(){return quaternions_;}
 const QuaternionField& quaternion_field()const{return quaternions_;}
 const Leaf& leaf(uint16_t raw)const{return leaves_[raw];}
 // A rank-three hypermatrix coordinate has three directed mirror
 // addresses but NO additional physical cells: the pair legs use the
 // same canonical NaN16 cells and their reverse orientations. The head
 // offsets point into the same 256 palindromic 128-byte registers.
 struct HyperMatrixAddress {
  std::array<uint16_t,3> directed{},inverse{};
  std::array<uint32_t,3> pair_offsets{},head_offsets{};
 };
 HyperMatrixAddress hypermatrix_address(uint8_t a,uint8_t b,uint8_t c)const{
  if(a==b||b==c||a==c)throw std::invalid_argument("degenerate hypermatrix coordinate");
  HyperMatrixAddress out{};
  out.directed={address(a,b),address(b,c),address(c,a)};
  out.head_offsets={
   static_cast<uint32_t>(offsetof(Root,h)+size_t(a)*sizeof(Head)),
   static_cast<uint32_t>(offsetof(Root,h)+size_t(b)*sizeof(Head)),
   static_cast<uint32_t>(offsetof(Root,h)+size_t(c)*sizeof(Head))
  };
  for(size_t i=0;i<3;++i){
   out.inverse[i]=mirror(out.directed[i]);
   out.pair_offsets[i]=leaf(out.directed[i]).physical_offset;
  }
  return out;
 }
 size_t event_count()const{return events_.size();}
 size_t hyperedge_count()const{return hyper_.size();}
 size_t resident_bytes_nominal()const{return ROOT_BYTES+sizeof(leaves_)+sizeof(phase_)+quaternions_.mapped_bytes()+
   events_.capacity()*sizeof(Event)+hyper_.capacity()*sizeof(Hyperedge);}
 void observe(uint8_t byte) {
  // Noncommuting 3-plane Givens stages, with lossless byte identity retained by the head address.
  const double angle=TAU*byte/256.0;
  Vec3 v=rotate_z(rotate_y(rotate_x({1,0,0},angle),angle/2),angle/3);
  auto& h=root_->h[byte];
  // The special head 255 is nannet ingress/egress, never modified by ordinary intake.
  if(byte==255)return;
  float previous=0;
  std::memcpy(&previous,h.data+8,sizeof(float));
  const float updated=static_cast<float>(0.95*previous+0.05*v.x);
  std::memcpy(h.data+8,&updated,sizeof(float));
  uint32_t count=0;std::memcpy(&count,h.data+16,4);
  ++count;std::memcpy(h.data+16,&count,4);
 }
 Vec3 transformed(uint8_t byte)const{
  const double angle=TAU*byte/256.0;
  const auto& h=root_->h[byte];
  float state=0;std::memcpy(&state,h.data+8,4);
  return rotate_z(rotate_y(rotate_x({1,0,0},angle+state),angle/2),angle/3);
 }
 void write(uint8_t a,uint8_t b,uint8_t type,bool refutes,float weight,
            uint64_t since,uint64_t until=0,uint64_t provenance=0){
  if(std::memcmp(root_->h[255].data+112,"PAIRC256",8)==0)
   throw std::runtime_error("cannot write graph evidence into full-pair language plane");
  if(a==b||type>=16||weight<=0||!std::isfinite(weight)|| (until && until<=since))
   throw std::invalid_argument("invalid typed relation");
  // Validate both physical mirror lanes before appending journal state.
  const float forward=lane_value(a,b,type,refutes);
  const float backward=lane_value(b,a,inverse_type(type),refutes);
  if(!std::isfinite(forward+weight)||!std::isfinite(backward+weight))
   throw std::overflow_error("native mirror-pair evidence overflow");
  const Event e{since,until,provenance,weight,a,b,type,static_cast<uint8_t>(refutes),0};
  events_.push_back(e);
  index_event(events_.size()-1);
  project(e);
 }
 Evidence query(uint8_t a,uint8_t b,uint8_t type,uint64_t at=LATEST)const{
  if(type>=16)throw std::invalid_argument("invalid type");
  if(a==b)return {};
  if(at==LATEST)return {lane_value(a,b,type,false),lane_value(a,b,type,true)};
  Evidence out;
  auto it=event_index_.find(key(a,b,type));
  if(it==event_index_.end())return out;
  for(size_t index:it->second){
   const auto& e=events_[index];
   if(at<e.since || (e.until!=0 && at>=e.until))continue;
   if(e.refutes)out.refute+=e.weight;else out.support+=e.weight;
  }
  return out;
 }
 // Self-modifying data, not host instructions: stage a pair in scratch,
 // accept iff it moves the desired score closer to the supervised target.
 bool trial(uint8_t head,uint8_t a,uint8_t b,uint8_t type,bool refutes,
            float step,float target_score,uint64_t time,uint64_t provenance=0){
  if(head==255||a==b||type>=16||step<=0||!std::isfinite(step))return false;
  const Pair scratch=root_->p[pair_index(a,b)];
  const float old=(a<b?scratch.f:scratch.r)[type+(refutes?16:0)];
  const float proposed=old+step;
  if(std::fabs(target_score-proposed)>=std::fabs(target_score-old))return false;
  write(a,b,type,refutes,step,time,0,provenance);
  uint32_t accepted=0;std::memcpy(&accepted,root_->h[head].data+20,4);
  ++accepted;std::memcpy(root_->h[head].data+20,&accepted,4);
  return true;
 }
 void set_phase(uint8_t a,uint8_t b,double radians){
  if(a==b||!std::isfinite(radians))throw std::invalid_argument("phase");
  phase_[pair_index(a,b)]=(a<b?1.0:-1.0)*radians;
 }
 double phase(uint8_t a,uint8_t b)const{
  return a==b?0.0:(a<b?1.0:-1.0)*phase_[pair_index(a,b)];
 }
 double gauge_potential(uint8_t head)const{
  double out=0;std::memcpy(&out,root_->h[head].data+72,8);return out;
 }
 double covariant_phase(uint8_t a,uint8_t b)const{
  return phase(a,b)-gauge_potential(a)+gauge_potential(b);
 }
 void set_covariant_phase(uint8_t a,uint8_t b,double angle){
  set_phase(a,b,angle+gauge_potential(a)-gauge_potential(b));
 }
 void gauge(const std::array<double,HEADS>& shifts){
  for(unsigned a=0;a<HEADS;++a){
   if(!std::isfinite(shifts[a]))throw std::invalid_argument("nonfinite gauge");
  }
  for(unsigned a=0;a<HEADS;++a)for(unsigned b=a+1;b<HEADS;++b)
   phase_[pair_index(a,b)]+=shifts[a]-shifts[b];
  for(unsigned a=0;a<HEADS;++a){
   double new_potential=gauge_potential(static_cast<uint8_t>(a))+shifts[a];
   std::memcpy(root_->h[a].data+72,&new_potential,8);
  }
 }
 double triangle(uint8_t a,uint8_t b,uint8_t c)const{
  if(a==b||b==c||a==c)throw std::invalid_argument("degenerate triangle");
  return phase(a,b)+phase(b,c)+phase(c,a);
 }
 // Hodge projection onto K_256 vertex potentials in the zero-mean gauge.
 // L = 256 I - 11^T: g_i = div(connection)_i / 256.
 std::array<double,HEADS> hodge_potentials()const{
  std::array<double,HEADS> g{};
  for(unsigned a=0;a<HEADS;++a)for(unsigned b=a+1;b<HEADS;++b){
   const double theta=covariant_phase(static_cast<uint8_t>(a),static_cast<uint8_t>(b));
   g[a]+=theta;g[b]-=theta;
  }
  for(double& x:g)x/=HEADS;
  return g;
 }
 // Divergence-free residual of the edge field, retaining all triangle curls.
 double hodge_cycle(uint8_t a,uint8_t b,const std::array<double,HEADS>& g)const{
  return covariant_phase(a,b)-g[a]+g[b];
 }
 // All C(256,3) Abelian triangle holonomies are generated by just
 // C(255,2)=32,385 triangles rooted at vertex 0.
 double anchored_triangle(uint8_t a,uint8_t b,uint8_t c)const{
  if(a==0||b==0||c==0||a==b||b==c||a==c)
   throw std::invalid_argument("requires three distinct non-anchor vertices");
  return triangle(0,a,b)+triangle(0,b,c)+triangle(0,c,a);
 }
 uint64_t hyper_id(uint8_t a,uint8_t b,uint8_t c,uint8_t type,bool refutes=false)const {
  // Opaque binary64 quiet-NaN-pattern integer, NOT a floating point NaN value.
  return 0x7ff8000000000000ULL | (static_cast<uint64_t>(type&15u)<<33) |
   (static_cast<uint64_t>(refutes?1:0)<<32) |
   (static_cast<uint64_t>(a)<<16) | (static_cast<uint64_t>(b)<<8) |c;
 }
 uint64_t add_hyper(uint8_t a,uint8_t b,uint8_t c,uint8_t type,bool refutes,
                   float strength,uint64_t when,uint64_t provenance=0){
  if(a==b||b==c||a==c||type>=16||strength<=0||!std::isfinite(strength))
   throw std::invalid_argument("hyperedge");
  const uint64_t id=hyper_id(a,b,c,type,refutes);
  const auto found=hyper_index_.find(id);
  if(found!=hyper_index_.end()){
   auto& x=hyper_[found->second];
   if(!std::isfinite(x.strength+strength))
    throw std::overflow_error("hypermatrix strength overflow");
   x.strength+=strength;x.created=when;x.provenance=provenance;
   return id;
  }
  if(hyper_.size()>=20000000u)
   throw std::length_error("sparse hypermatrix overlay capacity reached");
  const size_t position=hyper_.size();
  hyper_.push_back(Hyperedge{id,when,provenance,strength,a,b,c,type,
                             static_cast<uint8_t>(refutes),3,0});
  try{hyper_index_.emplace(id,position);}
  catch(...){hyper_.pop_back();throw;}
  return id;
 }
 // Constant-time access to sparse residuals of the logical H[a,b,c,type].
 Evidence hyper_query(uint8_t a,uint8_t b,uint8_t c,uint8_t type)const{
  if(type>=16)throw std::invalid_argument("hyperedge relation type");
  if(a==b||b==c||a==c)return {};
  Evidence out{};
  const auto positive=hyper_index_.find(hyper_id(a,b,c,type,false));
  const auto negative=hyper_index_.find(hyper_id(a,b,c,type,true));
  if(positive!=hyper_index_.end())out.support=hyper_[positive->second].strength;
  if(negative!=hyper_index_.end())out.refute=hyper_[negative->second].strength;
  return out;
 }
 std::vector<Hyperedge> routes(uint8_t vertex)const{
  std::vector<Hyperedge> result;
  for(const auto& x:hyper_)if(x.a==vertex||x.b==vertex||x.c==vertex)result.push_back(x);
  return result;
 }
 struct RoutePacket {uint8_t local=255,hw=4,slot=0;uint16_t ingress=0;uint64_t payload=0;};
 RoutePacket nannet_packet(uint8_t remote_hw,uint8_t remote_slot,uint16_t leaf_address,uint64_t id)const{
  return RoutePacket{255,remote_hw,remote_slot,leaf_address,id};
 }
 void sync()const{
  quaternions_.sync();
  if(::msync(root_,ROOT_BYTES,MS_SYNC)!=0)throw std::runtime_error("msync");
  const auto temp=filename_+".overlay.tmp",dest=filename_+".overlay";
  std::ofstream out(temp,std::ios::binary|std::ios::trunc);
  if(!out)throw std::runtime_error("cannot write overlay");
  if(events_.size()>UINT32_MAX||hyper_.size()>UINT32_MAX)throw std::runtime_error("overlay too large");
  OverlayHeader hdr;
  hdr.events=static_cast<uint32_t>(events_.size());
  hdr.hyperedges=static_cast<uint32_t>(hyper_.size());
  out.write(reinterpret_cast<const char*>(&hdr),sizeof(hdr));
  out.write(reinterpret_cast<const char*>(phase_.data()),sizeof(phase_));
  if(!events_.empty())out.write(reinterpret_cast<const char*>(events_.data()),sizeof(Event)*events_.size());
  if(!hyper_.empty())out.write(reinterpret_cast<const char*>(hyper_.data()),sizeof(Hyperedge)*hyper_.size());
  out.close();
  if(!out)throw std::runtime_error("overlay write failed");
  std::filesystem::rename(temp,dest);
 }
};
} // namespace nanlm
