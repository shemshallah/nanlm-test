#include "nanlm/semantic.hpp"
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unistd.h>
using namespace nanlm;

static void check(bool b,const char* reason){
 if(!b)throw std::runtime_error(reason);
}
int main(){
 const std::string root=(std::filesystem::temp_directory_path()/
                         ("nanlm-sem-"+std::to_string(::getpid()))).string();
 for(const std::string suffix:{"",".overlay",".semantic.tsv",".wordnet.tsv",".so3"})
  std::filesystem::remove(root+suffix);
 {
  Universe u(root);WordNetIndex wn;
  wn.import_noun_data("tests/fixtures/data.noun",u);
  wn.save(root+".wordnet.tsv");
  SemanticBridge m(u,wn);
  check(m.claims()==0,"fresh facts");
  check(m.query_isa("artifact","entity").state==1,"3-hop lexical entailment");
  check(m.query_isa("artifact","entity").depth==3,"exact proof length");
  check(m.query_isa("entity","artifact").state==0,"inverse is not ISA");
  check(m.query_isa("artifact","artifact").state==1,"reflexive identity");
  m.ingest("Entity can exist.","source-A");
  check(m.query_ability("artifact","exist").state==1,
        "semantic ability transfers through three unseen edges");
  check(m.query_ability("artifact","exist").depth==3,"inherited proof depth");
  m.ingest("Object cannot exist.","source-B");
  check(m.query_ability("artifact","exist").state==2,
        "nearby negative exception overrides distant positive");
  m.ingest("Artifact can exist.","source-C");
  check(m.query_ability("artifact","exist").state==1,
        "more specific positive overrides inherited negative");
  m.ingest("Artifact cannot exist.","source-D");
  check(m.query_ability("artifact","exist").state==3,
        "same-specificity support/refutation both preserved");
  check(m.query_ability("artifact","unknown").state==0,"unknown abstains");
  check(m.query_ability("fiction","exist").state==0,
        "contradiction relation not confused with ISA");
  m.ingest("An artifact is not an entity.","source-E");
  check(m.query_isa("artifact","entity").state==3,
        "direct counterclaim coexists with grounded ISA proof");
  check(m.query_isa("fiction","entity").state==0,"no explosion");
  check(m.query_ability("existence","exist").state==1,
        "exact WordNet alias points to same synset");
  const auto n=m.claims();
  m.ingest("Artifact cannot exist.","source-D");
  check(m.claims()==n,"same source fact is idempotent");
  try{
   m.ingest("Artifact imagines a world.","bad");
   throw std::runtime_error("invalid grammar accepted");
  }catch(const std::invalid_argument&){}
  try{
   m.ingest("unknown_lemma can fly.","bad");
   throw std::runtime_error("unknown lemma accepted");
  }catch(const std::invalid_argument&){}
  check(u.event_count()==wn.relations()+n,
        "one physical NaN16 event per new exact assertion");
  m.save();
 }
 {
  Universe u(root);WordNetIndex wn;wn.load(root+".wordnet.tsv");
  const auto prior_events=u.event_count();
  SemanticBridge m(u,wn);
  check(m.claims()==5,"five persisted exact facts");
  check(m.query_ability("artifact","exist").state==3,"signed persistent evidence");
  check(m.query_isa("artifact","entity").state==3,"persistent contradiction");
  check(u.event_count()==prior_events,"reload must not project duplicate NaN16 events");
 }
 for(const std::string suffix:{"",".overlay",".semantic.tsv",".wordnet.tsv",".so3"})
  std::filesystem::remove(root+suffix);
 std::cout<<"PASS: grounded synsets, 3-hop entailment, signed specificity, "
          <<"contestation, exact provenance, abstention, persistence, one NaN16 root\n";
}
