#include "nanlm/universe.hpp"
#include <array>
#include <cmath>
#include <filesystem>
#include <iostream>
using namespace nanlm;
static void require(bool yes,const char* why){if(!yes)throw std::runtime_error(why);}
int main(){
 const std::string path=(std::filesystem::temp_directory_path()/("nanlm-so3-"+std::to_string(::getpid()))).string();
 std::filesystem::remove(path);std::filesystem::remove(path+".overlay");std::filesystem::remove(path+".so3");
 double before=0;
 {
  Universe universe(path);
  auto& field=universe.quaternion_field();
  require(std::filesystem::file_size(path+".so3")==522304,"510 KiB+64 B mmap");
  for(unsigned a=0;a<HEADS;++a)for(unsigned b=a+1;b<HEADS;++b){
   const auto raw=address(static_cast<uint8_t>(a),static_cast<uint8_t>(b));
   require(universe.leaf(raw).physical_offset==universe.leaf(mirror(raw)).physical_offset,"mirror topology");
   require(field.get(static_cast<uint8_t>(a),static_cast<uint8_t>(b)).angle()<1e-7,"fresh flat field");
  }
  const Quat x=Quat::axis_angle({1,0,0},SO3_PI/2);
  const Quat y=Quat::axis_angle({0,1,0},SO3_PI/2);
  require((x*y).distance(y*x)>0.1,"noncommuting three-dimensional rotations");
  const Vec3q rotated=x.rotate({0,1,0});
  require(std::fabs(rotated.z-1)<1e-6,"quaternion rotates vectors");
  const Quat arc=Quat::from_to({1,0,0},{0,1,0});
  require(std::fabs(arc.rotate({1,0,0}).y-1)<1e-6,"shortest-arc rotation");
  field.set(1,2,x);field.set(2,3,y);
  const Quat composite=field.get(1,2)*field.get(2,3);
  field.set(1,3,composite);
  require(field.curvature(1,2,3)<1e-5,"composable triangle holonomy");
  field.set(1,3,Quat::identity());
  before=field.curvature(1,2,3);
  require(before>0.1,"nonflat triangle curvature retained");
  require((field.get(1,2)*field.get(2,1)).angle()<1e-5,"inverse direction");
  require((field.get(2,3)*field.get(3,2)).angle()<1e-5,"inverse orientation");
  std::array<Quat,HEADS> gauge{};
  for(unsigned i=0;i<HEADS;++i)
   gauge[i]=Quat::axis_angle({1.0,double(i%17+1),0.3},0.33*std::sin(double(i)*0.27));
  field.gauge(gauge);
  require(std::fabs(field.curvature(1,2,3)-before)<2e-5,"holonomy conjugacy invariant");
  require((field.get(1,2)*field.get(2,1)).angle()<1e-5,"gauge maintains inverse");
  require(std::fabs(field.get(2,1).distance(field.get(1,2).conjugate()))<1e-6,"reverse transport exactly inverse");
  universe.sync();
 }
 {
  Universe universe(path);
  require(std::fabs(universe.quaternion_field().curvature(1,2,3)-before)<2e-5,
          "quaternion curvature persisted across process");
  require(universe.resident_bytes_nominal()>=ROOT_BYTES+522304,"resident budget includes field");
 }
 std::filesystem::remove(path);std::filesystem::remove(path+".overlay");std::filesystem::remove(path+".so3");
 std::cout<<"PASS SO3: 32640 mirrored quaternion connections, noncommutation, reverse inverse, triangle gauge invariance and restart\n";
}