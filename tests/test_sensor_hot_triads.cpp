#include "nanlm/sensor_packet.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
using namespace nanlm::math_sensor;
static void require(bool condition,const char* why) {
 if(!condition)throw std::runtime_error(why);
}
int main() {
 try {
  SensorMemory sensor(128,1000,0.5);
  require(sensor.hot_count()==0,"blank sensor has spurious hot triheads");
  sensor.set_hot_budget(4);
  require(sensor.hot_budget()==4,"hot budget not applied");
  const Assertion first{"S1","penguin","ISA","bird",false,1.};
  const Assertion second{"S2","penguin","ISA","bird",true,1.};
  const Assertion third{"S3","bird","ISA","animal",false,1.};
  require(sensor.ingest(first)&&sensor.ingest(second)&&sensor.ingest(third),
          "source corpus intake failed");
  require(sensor.hot_count()>0&&sensor.hot_count()<=4,
          "top-K selection did not enforce budget");
  require(sensor.hot_field().active==sensor.hot_count(),
          "hot field not synchronized with active triheads");
  require(sensor.hot_links().size()==sensor.hot_count()*(sensor.hot_count()-1)/2,
          "hot-hypergraph pair overlay has incorrect cardinality");
  std::set<int> selected;
  for(std::size_t i=0;i<sensor.hot_count();++i) {
   const auto& h=sensor.hot_triad(i);
   require(selected.insert(h.group).second,"duplicate hot triad");
   require(sensor.hot_slot(h.group)==static_cast<int>(i),
           "hot slot cannot resolve parent triad");
   require(h.group<85,"invalid triad in hot mapping");
   for(std::size_t j=0;j<3;++j)
    require(h.heads[j]==3*h.group+j,"hot head membership lost");
   require(h.score>0&&h.activation>0,"inactive triad selected as hot");
   require(h.source_count>=1,"source-aware hot mapping missing");
   for(std::size_t j=0;j<9;++j)
    require(std::abs(h.tensor[j]-sensor.triad(h.group).tensor[j])<1e-12,
            "hot tensor copy diverges from parent");
   if(i)require(sensor.hot_triad(i-1).score>=h.score,
               "hot triheads not deterministically ranked");
  }
  for(const auto& l:sensor.hot_links()) {
   require(selected.contains(l.a)&&selected.contains(l.b),
           "hot overlay has stale/cold endpoints");
   require(l.a!=l.b&&std::isfinite(l.tensor_cosine)&&
           l.tensor_cosine>=-1.00001&&l.tensor_cosine<=1.00001,
           "invalid hot pair tensor alignment");
   require(std::isfinite(l.phase_lock)&&l.phase_lock>=-1.00001&&
           l.phase_lock<=1.00001,"invalid hot pair phase coherence");
  }
  // One source can refute another without a signed charge collapsing.
  const auto r=sensor.reason("penguin","ISA","animal");
  require(r.state()==1&&r.inferred&&r.proof_sources.size()==2,
          "refutation of a lower-level claim incorrectly destroys proof");
  const auto packet=encode_proof(sensor,"penguin","ISA","bird");
  require(packet.hot_count==sensor.hot_count(),"numeric hot packet size");
  require(packet.hot_link_count==sensor.hot_links().size(),
          "numeric hot link count");
  require(packet.hot_triads[0].group==sensor.hot_triad(0).group,
          "numeric hot packet lost triad mapping");
  require(packet.hot_triads[0].heads==sensor.hot_triad(0).heads,
          "numeric hot packet lost 3 underlying head identities");
  require(packet.hot_field_support>0&&packet.hot_field_refute>0,
          "hot field conflated support and refutation");
  const auto old_activity=sensor.hot_field().activation;
  sensor.tick();
  require(sensor.hot_field().activation<old_activity,
          "nanclock did not cool hot field");
  if(sensor.hot_count())
   require(sensor.hot_triad(0).previous_activation>=sensor.hot_triad(0).activation,
           "hot triad temporal mirror does not expose prior tick");
  for(int i=0;i<24;++i)sensor.tick();
  require(sensor.hot_count()==0,"cold triads remained hot without new evidence");
  require(sensor.global().unique_sources==3&&sensor.global().unique_statements==2,
          "cold expiry corrupted exact global memory");
  require(sensor.direct("penguin","ISA","bird").state()==3,
          "hot expiry erased conflicting exact evidence");
  require(sensor.reason("penguin","ISA","animal").inferred,
          "hot expiry erased cross-source proof");
  require(sensor.hot_links().empty()&&sensor.hot_field().active==0,
          "expired hot graph was not cleared");
  require(sensor.ingest({"S4","pangolin","ISA","mammal",false,1.}),
          "hot reactivation ingest failed");
  require(sensor.hot_count()>0,"new evidence did not reactivate hot triheads");
  // Hot maps must honor dynamically assigned native three-head groups,
  // rather than presuming consecutive head addresses.
  SensorMemory remapped;
  auto mapping=default_map();
  std::swap(mapping[0][1],mapping[1][1]);
  remapped.set_mapping(mapping);
  remapped.ingest({"R1","whale","ISA","mammal",false,1.});
  require(remapped.hot_count()>0,"adaptively mapped hot view absent");
  for(std::size_t i=0;i<remapped.hot_count();++i) {
   const auto& h=remapped.hot_triad(i);
   require(h.heads==mapping[h.group],
           "hot-triad mapping ignored adaptive trihead membership");
  }
  bool threw=false;
  try{sensor.set_hot_budget(13);}catch(const std::invalid_argument&){threw=true;}
  require(threw,"out-of-bounds hot capacity was allowed");
  std::cout<<"PASS: dynamic 255/85/hot/global triad map, signed sources, numeric overlay, temporal cooling, reactivation and cold evidence preservation\n";
  return 0;
 }catch(const std::exception& e) {
  std::cerr<<"FAIL: "<<e.what()<<"\n";return 1;
 }
}
