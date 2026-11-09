#include "nanlm/multiloss.hpp"
#include <cmath>
#include <filesystem>
#include <iostream>
using namespace nanlm;
static void require(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
int main(){
 const std::string path=(std::filesystem::temp_directory_path()/("nanlm-multi-"+std::to_string(::getpid()))).string();
 std::filesystem::remove(path);std::filesystem::remove(path+".overlay");std::filesystem::remove(path+".so3");
 double expected=0;
 {
  Universe u(path);
  FabricTeacher teacher(u);
  JointFieldTrainer joint(u,teacher);
  u.write(1,2,ISA,false,2,10,20,1001);
  u.write(2,3,ISA,false,2,10,20,1002);
  u.quaternion_field().set(1,2,Quat::axis_angle({1,0,0},1.0));
  u.quaternion_field().set(2,3,Quat::axis_angle({0,1,0},1.1));
  auto early=joint.teach(1,2,3,ISA,5);
  require(!early.eligible&&!early.accepted,"future evidence cannot teach earlier period");
  const double initial=u.quaternion_field().curvature(1,2,3);
  auto yes=joint.teach(1,2,3,ISA,15);
  require(yes.eligible&&yes.accepted,"composable triple updates shared SO3 connection");
  require(yes.after.total+1e-10<yes.before.total,"all four terms evaluated before commit");
  require(u.quaternion_field().curvature(1,2,3)<initial,"trusted composition sharpens triangle");
  require(joint.teach(1,2,3,ISA,25).eligible==false,"expired evidence not presumed current");
  u.write(1,3,ISA,false,1,10,20,1003);
  u.write(1,3,ISA,true,1,10,20,1004);
  require(u.query(1,3,ISA,15).state()==3,"support and refutation preserved");
  auto contested=joint.teach(1,2,3,ISA,15);
  require(contested.eligible,"a contested conclusion is still queried, not erased");
  require(u.query(1,3,ISA,15).state()==3,"training cannot collapse contradiction");
  u.write(1,2,ISA,true,1,10,20,1005);
  require(!joint.teach(1,2,3,ISA,15).eligible,"contradicted antecedent blocked");
  expected=u.quaternion_field().curvature(1,2,3);
  u.sync();
 }
 {
  Universe u(path);FabricTeacher teacher(u);
  JointFieldTrainer joint(u,teacher);
  require(std::fabs(u.quaternion_field().curvature(1,2,3)-expected)<1e-5,"retained curvature survives restart");
  require(u.query(1,3,ISA,15).state()==3,"retained contradictory evidence");
  require(!joint.teach(1,2,3,ISA,15).eligible,"retained contradictory antecedent blocks extrapolation");
 }
 std::filesystem::remove(path);std::filesystem::remove(path+".overlay");std::filesystem::remove(path+".so3");
 std::cout<<"PASS JOINT: predictive + SO3 triangle + effective-time contradiction + retention objectives, scratch-accepted and restored\n";
}