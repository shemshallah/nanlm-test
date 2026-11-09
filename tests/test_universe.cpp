#include "nanlm/universe.hpp"
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <unordered_set>
using namespace nanlm;
static void check(bool ok,const char* msg){if(!ok)throw std::runtime_error(msg);}
static std::string path(){
 return (std::filesystem::temp_directory_path()/("nanlm-test-"+std::to_string(::getpid()))).string();
}
int main(){
 const auto root=path();
 const double before=1.1-0.2+0.8;
 std::filesystem::remove(root);std::filesystem::remove(root+".overlay");
 size_t hits[PAIRS]{};size_t fixed=0;
 std::uint64_t leaf_checks=0;
 {
 Universe u(root);
 check(std::filesystem::file_size(root)==ROOT_BYTES,"exact native root size");
 check(u.resident_bytes_nominal()<12*1024*1024,"bounded topology footprint");
 for(size_t i=0;i<LEAVES;++i){
  const auto& leaf=u.leaf(static_cast<uint16_t>(i));
  check(leaf.raw==i && leaf.inverse==mirror(static_cast<uint16_t>(i)),"leaf inverse");
  check(mirror(leaf.inverse)==i,"involution");
  check(address(leaf.from,leaf.to)==i,"address mapping");
  check(u.leaf(leaf.inverse).physical_offset==leaf.physical_offset,"mirror shared cell");
  if(leaf.kind){++fixed;check(leaf.inverse==i,"palindromic leaf");}
  else{
   check(leaf.inverse!=i,"nontrivial mirror");
   ++hits[pair_index(leaf.from,leaf.to)];
  }
  ++leaf_checks;
 }
 check(fixed==256,"exact palindromic count");
 for(const auto k:hits)check(k==2,"each pair exactly twice");
 for(unsigned i=0;i<HEADS;++i){
  const auto& h=u.root().h[i];
  check(h.data[0]==(i>>6) && h.data[1]==((i>>4)&3) && h.data[2]==(i&15),"4x4x16 head type");
 }
 for(unsigned t=0;t<16;++t)check(inverse_type(inverse_type(t))==t,"inverse relation law");
 check(u.query(1,2,ISA).state()==0,"unknown not falsely asserted");
 u.write(1,2,ISA,false,1.0f,10,0,1001);
 check(u.query(1,2,ISA).state()==1,"support");
 check(u.query(2,1,HAS_SUBTYPE).state()==1,"inverse support");
 check(u.query(1,2,HAS_SUBTYPE).state()==0,"different type still unknown");
 u.write(1,2,ISA,true,0.9f,20,30,1002);
 check(u.query(1,2,ISA).state()==3,"Belnap BOTH retained");
 check(u.query(2,1,HAS_SUBTYPE).state()==3,"inverse contradictory evidence retained");
 check(u.query(1,2,ISA,5).state()==0,"before time");
 check(u.query(1,2,ISA,15).state()==1,"past support");
 check(u.query(1,2,ISA,25).state()==3,"temporal overlap");
 check(u.query(1,2,ISA,35).state()==1,"expired refutation");
 check(u.query(1,2,CAUSES).state()==0,"no logical explosion");
 check(!u.trial(255,3,4,CAUSES,false,1,3,30),"reserved router refuses self modification");
 check(!u.trial(7,3,4,CAUSES,false,1,-5,30),"scratch rejects worsening update");
 check(u.query(3,4,CAUSES).state()==0,"rejected trial did not change root");
 check(u.trial(7,3,4,CAUSES,false,0.5f,1,30),"scratch accepts improving update");
 check(u.query(4,3,CAUSED_BY).state()==1,"trial commits inverse relation");
 u.set_phase(1,2,1.1);u.set_phase(2,3,-0.2);u.set_phase(3,1,0.8);
 check(std::fabs(u.triangle(1,2,3)-before)<1e-12,"triangle circulation");
 std::array<double,HEADS> g{};
 for(size_t i=0;i<HEADS;++i)g[i]=30*std::sin(i*0.137);
 u.gauge(g);
 check(std::fabs(u.triangle(1,2,3)-before)<1e-11,"large vertex gauge invariance");
 Vec3 a=u.transformed(41);
 check(std::fabs((a.x*a.x+a.y*a.y+a.z*a.z)-1)<1e-12,"orthogonal trig transform");
 u.observe(41);check(u.root().h[41].data[16]==1,"learned register update");
 const auto packet=u.nannet_packet(4,0,address(5,9),123);
 check(packet.local==255&&packet.hw==4&&packet.slot==0&&packet.ingress==address(5,9),"nannet reserved route");
 const auto id=u.add_hyper(1,2,3,ISA,false,0.8f,35,777);
 check((id&0x7ff8000000000000ULL)==0x7ff8000000000000ULL,"opaque qNaN-like hyperedge id");
 check(u.routes(2).size()==1 && u.routes(8).empty(),"hypergraph vertex routing");
 check(u.hyperedge_count()==1,"hyperedge overlay");
 u.sync();
 }
 {
 Universe u(root);
 check(u.event_count()==3,"durable event journal");
 check(u.hyperedge_count()==1,"durable hypergraph");
 check(u.query(1,2,ISA,25).state()==3,"temporal four-state after restore");
 check(u.query(1,2,ISA,35).state()==1,"temporal expiration after restore");
 check(u.query(3,4,CAUSES).state()==1,"scratch decision persisted");
 check(std::fabs(u.triangle(1,2,3)-before)<1e-11,"phase overlay persisted");
 check(u.root().h[41].data[16]==1,"register persisted");
 const auto start=std::chrono::steady_clock::now();
 uint64_t checksum=0;
 for(unsigned i=0;i<1000000;++i){
  const auto& leaf=u.leaf(static_cast<uint16_t>(i));
  checksum+=leaf.physical_offset;
 }
 auto us=std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-start).count();
 check(checksum>0 && leaf_checks==65536,"hot path checksum");
 std::cout<<"PASS: 65536 leaves, 32640 pairs, 256 palindrome heads, 4x4x16 gates\n";
 std::cout<<"PASS: temporal contradiction / inverse / gauge / trial / hyperedge / reload\n";
 std::cout<<"root_bytes="<<ROOT_BYTES<<" nominal_bytes="<<u.resident_bytes_nominal()<<"\n";
 std::cout<<"leaf_lookups=1000000 elapsed_us="<<us<<" checksum="<<checksum<<"\n";
 }
 std::filesystem::remove(root);std::filesystem::remove(root+".overlay");
 return 0;
}