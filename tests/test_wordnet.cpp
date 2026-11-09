#include "nanlm/wordnet.hpp"
#include <iostream>
#include <filesystem>
using namespace nanlm;
int main(){
 const std::string root=(std::filesystem::temp_directory_path()/("nanlm-wn-"+std::to_string(::getpid()))).string();
 std::filesystem::remove(root);std::filesystem::remove(root+".overlay");
 {
  Universe u(root);WordNetIndex w;
  w.import_noun_data("tests/fixtures/data.noun",u);
  if(w.concepts()!=5||w.relations()!=4)throw std::runtime_error("fixture import");
  auto entities=w.lookup("entity"),artifacts=w.lookup("artifact");
  if(entities.size()!=1||artifacts.size()!=1)throw std::runtime_error("word index");
  auto path=w.path(artifacts[0],entities[0]);
  if(path.size()!=4)throw std::runtime_error("exact three-hop inference");
  if(w.path(entities[0],artifacts[0]).size()!=0)throw std::runtime_error("incorrect inverse forward chain");
  auto concept=w.concept(artifacts[0]);
  if(!concept||w.concepts_at(concept->leaf).empty())throw std::runtime_error("leaf addressability");
  if(u.event_count()!=4)throw std::runtime_error("physical relation projection");
  w.save(root+".wordnet.tsv");u.sync();
 }
 {
  Universe u(root);
  if(u.event_count()!=4)throw std::runtime_error("persisted WordNet projections");
 }
 std::filesystem::remove(root);std::filesystem::remove(root+".overlay");
 std::filesystem::remove(root+".wordnet.tsv");
 std::cout<<"PASS: WordNet-format parsed, exact synset IDs, 3-hop concept route, persisted geometry\n";
 return 0;
}
