#include "nanlm/unified.hpp"
#include <iostream>
using namespace nanlm;
void check(bool ok,const char* what){if(!ok)throw std::runtime_error(what);}
void clean(const std::string& root){for(auto s:{"",".overlay",".so3",".byte-lm",".rnn-lm",".wordnet.tsv",".unified",".unified.tmp"})std::filesystem::remove(root+s);}
int main(){
 const std::string root="/tmp/nanlm-unified-"+std::to_string(getpid());clean(root);
 std::string generated;UnifiedModel::Distribution saved{};
 {
  Universe u(root);UnifiedModel m(u,root);
  const std::string text="the little fox walks through the quiet forest. the little bird sings near the river. ";
  auto initial=m.predict(m.prime("the little "));const double before=-std::log(initial.probability['f']);
  for(unsigned i=0;i<8;++i){std::istringstream input(text);m.train(input);}
  auto state=m.prime("the little ");auto p=m.predict(state);
  check(-std::log(p.probability['f'])<before,"unified model learns");
  for(unsigned k=0;k<UnifiedModel::K;++k){double z=0;for(double v:p.expert[k]){check(v>0&&std::isfinite(v),"finite expert");z+=v;}check(std::abs(z-1)<1e-9,"expert normalization");
   auto isolated=m.predict(state,1u<<k);for(unsigned b=0;b<255;++b)check(std::abs(isolated.probability[b]-p.expert[k][b])<1e-10,"single-layer causal ablation");}
  double z=0;auto wave=m.wave(state);
  for(unsigned b=0;b<255;++b){double q=0;for(unsigned k=0;k<UnifiedModel::K;++k)q+=p.weights[k]*p.expert[k][b];check(std::abs(q-p.probability[b])<1e-12,"one mixture");check(std::abs(wave.born(b)-q)<1e-12,"density uses final probabilities");z+=q;
   auto leaf=m.leaf(state,b);check(leaf.raw==address(' ',b)&&leaf.inverse==mirror(leaf.raw),"leaf identity across layers");}
  check(std::abs(z-1)<1e-12&&wave.born(255)==0,"gateway excluded from probability");
  // Every observed pair appears once despite multiple gradient objectives.
  auto s=m.prime("\x01");const auto count=u.query(1,2,PRECEDES).support;m.learn(s,2);check(u.query(1,2,PRECEDES).support==count+1,"no duplicated evidence");
  m.observe(s,255);check(s.context.size==0&&!s.recurrent.started&&s.senses.empty(),"boundary resets all causal layers");
  m.import_wordnet("tests/fixtures/data.noun");auto retrieved=m.prime("an ","entity");check(m.facts(retrieved).size()==1,"exact WordNet retrieval");
  auto a=m.predict(retrieved),b=m.predict(m.prime("an "));check(a.expert[5]!=b.expert[5]&&a.probability!=b.probability,"knowledge affects final generation distribution");
  const auto events=u.event_count();std::istringstream held("the little bird walks near the forest. ");auto metrics=m.evaluate(held);check(metrics.bytes>0&&std::isfinite(metrics.ce())&&u.event_count()==events,"causal read-only evaluation");
  saved=m.predict(m.prime("the little ")).probability;generated=m.generate("the little ",100,11);check(generated==m.generate("the little ",100,11),"deterministic sampling");
  check(generated.find(char(255))==std::string::npos,"reserved gateway never generated");m.save();
  std::cout<<"unified heldout_ce="<<metrics.ce()<<" sample="<<std::quoted(generated.substr(0,70))<<"\n";
 }
 {
  Universe u(root);UnifiedModel m(u,root);check(saved==m.predict(m.prime("the little ")).probability,"checkpoint distribution equality");check(generated==m.generate("the little ",100,11),"checkpoint generation equality");
 }
 {
  std::ofstream f(root+".wordnet.tsv",std::ios::app);f<<"\n";
 }
 bool rejected=false;try{Universe u(root);UnifiedModel m(u,root);}catch(const std::runtime_error&){rejected=true;}check(rejected,"mixed checkpoint rejected");
 clean(root);std::cout<<"PASS: unified causal experts, transforms, leaf identity, knowledge, generation, restart and integrity\n";
}
