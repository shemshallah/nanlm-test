#include "nanlm/teacher.hpp"
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
using namespace nanlm;
static void expect(bool p,const char* msg){if(!p)throw std::runtime_error(msg);}
int main(){
 const std::string filename=(std::filesystem::temp_directory_path()/
    ("nanlm-teacher-"+std::to_string(::getpid()))).string();
 std::filesystem::remove(filename);std::filesystem::remove(filename+".overlay");
 double expected=0;
 {
  Universe u(filename);
  FabricTeacher teach(u);
  expect(u.root().h[255].data[3]==0,"router head not overwritten");
  auto before=teach.loss(23,40);
  unsigned accepts=0;
  for(unsigned t=1;t<=100;++t)if(teach.teach(23,40,t).accepted)++accepts;
  expect(accepts>90,"teacher must accept real local objective improvements");
  expected=teach.loss(23,40);
  expect(expected<before-0.25,"learning improves conditional log loss");
  expect(u.query(23,40,PRECEDES).support>0,"head writes paired evidence");
  expect(u.query(40,23,FOLLOWS).support>0,"head teaches inverse lane");
  expect(u.query(40,23,PRECEDES).support==0,"inverse not conflated with forward");
  const double cv=u.covariant_phase(23,40);
  std::array<double,HEADS> shifts{};
  for(unsigned i=0;i<HEADS;++i)shifts[i]=5*std::sin(i*0.27);
  const double pred=teach.loss(23,40);
  u.gauge(shifts);
  expect(std::fabs(cv-u.covariant_phase(23,40))<1e-12,"gauge connection invariant");
  expect(std::fabs(pred-teach.loss(23,40))<1e-10,"model prediction gauge invariant");
  expect(u.query(23,40,PRECEDES,1).state()==1,"time-indexed signal");
  expect(u.query(23,40,PRECEDES,0).state()==0,"signal not in past");
  u.sync();
 }
 {
  Universe u(filename);FabricTeacher teach(u);
  expect(std::fabs(teach.loss(23,40)-expected)<1e-9,"learned head and fabric survive restart");
  expect(u.query(23,40,PRECEDES).support>0,"relation survives restart");
  const auto start=std::chrono::steady_clock::now();
  volatile double checksum=0;
  for(int n=0;n<1000;++n){
   auto x=teach.predict(static_cast<uint8_t>(n%255));
   checksum=checksum+x.probability[static_cast<unsigned>(n%255)];
  }
  const auto us=std::chrono::duration_cast<std::chrono::microseconds>(
   std::chrono::steady_clock::now()-start).count();
  expect(std::isfinite(checksum),"prediction timings checksum");
  std::cout<<"PASS: self-modifying 256 heads teach sparse directed fabric; learned predictions survive restart\n";
  std::cout<<"PASS: held-state gauge invariance; scratch log-loss acceptance; inverse direction and temporal evidence\n";
  std::cout<<"predict_1000_us="<<us<<" per_prediction_us="<<double(us)/1000<<" before/after_loss_verified\n";
 }
 std::filesystem::remove(filename);std::filesystem::remove(filename+".overlay");
}