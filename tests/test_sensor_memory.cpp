#include "nanlm/sensor_memory.hpp"
#include <cmath>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
using namespace nanlm::math_sensor;
static void check(bool yes,const char* message) {if(!yes)throw std::runtime_error(message);}
int main() {
 const auto path=std::filesystem::temp_directory_path()/"nanlm-sensor-memory-unit.tsv";
 try {
  SensorMemory m(2);
  std::stringstream sources;
  sources<<"sourceA\t+\tpenguin\tISA\tbird\n"
         <<"sourceB\t+\tbird\tISA\tanimal\n"
         <<"sourceC\t-\tpenguin\tISA\tanimal\n"
         <<"sourceD\t+\tpenguin\tCAN_FLY\tyes\n"
         <<"sourceE\t-\tpenguin\tCAN_FLY\tyes\n";
  check(m.ingest_tsv(sources)==5,"multisource intake wrong");
  check(m.global().unique_sources==5&&m.global().unique_statements==4,"global integration counts");
  const auto result=m.reason("penguin","ISA","animal");
  check(result.state()==3,"independent refutation and transitive support not combined");
  check(result.inferred&&result.proof_nodes.size()==3,"two source derivation unavailable");
  check(result.proof_sources.contains("sourceA")&&result.proof_sources.contains("sourceB"),
        "proof did not preserve cross-source dependencies");
  check(result.evidence.refuting_sources.contains("sourceC"),"refuting source lost");
  check(m.direct("penguin","CAN_FLY","yes").state()==3,"four-valued contested edge lost");
  check(m.reason("penguin","CAN_FLY","yes").state()==3,"unrelated edge reasoning corrupted");
  check(m.reason("penguin","LOCATED_IN","Moon").state()==0,"contradiction explosion");
  check(m.reason("animal","ISA","penguin").state()==0,"wrong orientation");
  check(!m.ingest({"sourceA","penguin","ISA","bird",false,1.}),
        "replayed evidence double-counted");
  check(m.journal_size()==5,"idempotency journaling broken");
  bool has_leaf=false,has_tensor=false,has_triad=false;
  for(int h=0;h<255;++h){
   check(m.branch(static_cast<std::uint8_t>(h)).size()<=2,"leaf budget exceeded");
   for(const auto& [key,leaf]:m.branch(static_cast<std::uint8_t>(h))){
    has_leaf=true;
    for(auto x:leaf.tensor)if(std::abs(x)>1e-14)has_tensor=true;
    check(leaf.support_charge>=0&&leaf.refute_charge>=0,"signed channels conflated");
   }
  }
  for(int g=0;g<85;++g)if(m.triad(g).activation>0)has_triad=true;
  check(has_leaf&&has_tensor&&has_triad&&m.global().activation>0,"three-level tensor sensing inactive");
  const auto before=m.global();m.tick();
  check(m.direct("penguin","CAN_FLY","yes").state()==3,"tick decayed exact facts");
  check(m.global().activation<before.activation,"idle clock did not decay ephemerals");
  m.save(path);
  SensorMemory replay(2);replay.load(path);
  check(replay.journal_size()==5,"persistence lost journal");
  const auto r=replay.reason("penguin","ISA","animal");
  check(r.state()==3&&r.inferred&&r.proof_sources.size()==2,"replay changed proof");
  check(replay.global().unique_sources==5,"replay lost global sources");
  SensorMemory bad(2);
  auto invalid=default_map();invalid[0][1]=255;
  bool rejected=false;
  try {bad.set_mapping(invalid);}catch(const std::invalid_argument&){rejected=true;}
  check(rejected,"ingress head mistakenly included");
  struct Native {std::size_t head_triad(std::uint8_t h)const {
   return h==1?1:h==4?0:h/3;
  }} native;
  bad.import_native_mapping(native);
  bad.ingest({"doc1","Jupiter","ISA","planet",false,1.});
  check(bad.global().unique_statements==1,"native remapped intake failure");
  std::filesystem::remove(path);
  std::cout<<"PASS: 255 branch/85 triad/global sensor tensors, multisource ISA proof, four-valued conflicts, idempotence, bounded leaves, persistence and clock\n";
  return 0;
 }catch(const std::exception& e){
  std::filesystem::remove(path);
  std::cerr<<"FAIL: "<<e.what()<<"\n";
  return 1;
 }
}
