#include "nanlm/unified_wave.hpp"
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

using namespace nanlm;
static void require(bool ok,const char* what){
 if(!ok)throw std::runtime_error(what);
}
int main(){
 const auto root=(std::filesystem::temp_directory_path()/
  ("nanlm-hypermatrix-"+std::to_string(::getpid()))).string();
 try{
  double exact_probability=0;
  {
   Universe u(root);
   UnifiedWaveLanguage mind(u);
   require(UnifiedWaveLanguage::GROUPS==85,"not 85 coupled triads");
   require(u.root().h[255].data[UnifiedWaveLanguage::HYPER_LOCAL_MAGIC_OFFSET]==0,
           "network gateway was initialized as a language head");
   require(u.root().h[0].data[UnifiedWaveLanguage::HYPER_SHARED_MAGIC_OFFSET]==
           UnifiedWaveLanguage::HYPER_FIELD_MAGIC,"field register absent");
   require(u.root().h[2].data[UnifiedWaveLanguage::HYPER_SHARED_MAGIC_OFFSET]==
           UnifiedWaveLanguage::HYPER_GROUP_MAGIC,"triad register absent");
   require(u.root().h['z'].data[UnifiedWaveLanguage::HYPER_LOCAL_MAGIC_OFFSET]==
           UnifiedWaveLanguage::HYPER_LOCAL_MAGIC,"individual register absent");
   UnifiedWaveLanguage::State context{};
   mind.observe(context,'x');
   mind.observe(context,'y');
   require(u.hyper_query('x','y','z',PRECEDES).state()==0,"unexpected exact triple");
   const auto empty=mind.predict(context);
   const auto empty_ablated=mind.predict(context,UnifiedWaveLanguage::ABLATE_HYPEREDGE);
   require(std::fabs(empty.p['z']-empty_ablated.p['z'])<1e-12,
           "untrained factorized tensor should be zero");

   // The pair bank IS the factorized hypermatrix: this triple is never
   // inserted into the sparse overlay, yet its contextual logit changes.
   u.write('x','y',PRECEDES,false,2.f,1);
   u.write('y','z',PRECEDES,false,2.f,2);
   u.write('x','z',PRECEDES,false,1.f,3);
   require(u.hyper_query('x','y','z',PRECEDES).state()==0,
           "fabric factor accidentally constructed an exact hyperedge");
   // Refresh the derived eigensystem after direct writes so the pre-
   // and post-restart predictors use identical persistent graph state.
   mind.refresh_spectrum();
   // Also replay the causal wave itself: it was originally excited while
   // pair evidence was empty, before the graph mutation.
   context.reset();
   mind.observe(context,'x');
   mind.observe(context,'y');
   const auto coords=u.hypermatrix_address('x','y','z');
   const std::array<uint8_t,3> vertices{'x','y','z'};
   for(size_t i=0;i<3;++i){
    require(coords.directed[i]==address(vertices[i],vertices[(i+1)%3]),
            "hypermatrix leg is not a NaN16 directed address");
    require(coords.inverse[i]==mirror(coords.directed[i]),
            "hypermatrix direction lacks inverse address");
    require(coords.pair_offsets[i]==u.leaf(coords.inverse[i]).physical_offset,
            "hypermatrix mirror pair is not the same physical cell");
    require(coords.head_offsets[i]==offsetof(Root,h)+size_t(vertices[i])*sizeof(Head),
            "hypermatrix register is not physically mapped");
   }
   const auto factor=mind.predict(context);
   const auto without=mind.predict(context,UnifiedWaveLanguage::ABLATE_HYPEREDGE);
   require(std::fabs(factor.p['z']-without.p['z'])>1e-7,
           "factorized third-order mirror coupling ablates to zero");
   // Perturb one EXISTING trained head vector while keeping all pair
   // evidence fixed. Difference-of-log-odds isolates the hypermatrix
   // readout even though attention also consumes this same vector.
   auto impact=[&](const UnifiedWaveLanguage::Prediction& full,
                   const UnifiedWaveLanguage::Prediction& ablated){
    return std::log(full.p['z']/full.p['q'])-
           std::log(ablated.p['z']/ablated.p['q']);
   };
   float initial=0;
   std::memcpy(&initial,u.root().h['x'].data+32,sizeof(initial));
   const float perturbed=initial+2.f;
   std::memcpy(u.root().h['x'].data+32,&perturbed,sizeof(perturbed));
   const auto changed=mind.predict(context);
   const auto changed_ablated=mind.predict(context,UnifiedWaveLanguage::ABLATE_HYPEREDGE);
   require(std::fabs(impact(changed,changed_ablated)-
                     impact(factor,without))>1e-6,
           "head-vector rank-three tensor does not affect hyperedge output");
   std::memcpy(u.root().h['x'].data+32,&initial,sizeof(initial));
   // Sparse residuals are ordered, typed, signed and indexed.
   u.add_hyper('x','y','z',PRECEDES,false,0.5f,4);
   u.add_hyper('x','y','z',PRECEDES,true,0.2f,5);
   const auto evidence=u.hyper_query('x','y','z',PRECEDES);
   require(evidence.support>0.49f&&evidence.refute>0.19f,
           "signed indexed hyperedges lost their independent lanes");
   require(u.hyper_query('y','x','z',PRECEDES).state()==0,
           "wrong triple orientation aliased");
   exact_probability=mind.predict(context).p['z'];
   require(std::fabs(exact_probability-factor.p['z'])>1e-7,
           "exact residual made no predictive difference");
   u.sync();
  }
  {
   Universe u(root);
   UnifiedWaveLanguage mind(u);
   UnifiedWaveLanguage::State context{};
   mind.observe(context,'x');mind.observe(context,'y');
   require(u.hyperedge_count()==2,"sparse hyperedge count not persisted");
   require(std::fabs(mind.predict(context).p['z']-exact_probability)<1e-7,
           "hypermatrix prediction not restored after mapped-root reload");
   // All three voter levels must update in-place when trained.
   float old_local=0,old_group=0,old_field=0;
   auto read=[&](uint8_t h,size_t offset){
    float v=0;std::memcpy(&v,u.root().h[h].data+offset,4);return v;
   };
   old_local=read('z',UnifiedWaveLanguage::HYPER_LOCAL_OFFSET);
   old_group=read('z',UnifiedWaveLanguage::HYPER_SHARED_OFFSET);
   old_field=read(0,UnifiedWaveLanguage::HYPER_SHARED_OFFSET);
   std::istringstream corpus("xyz xyz xyz xyz xyz xyz xyz xyz");
   mind.train(corpus);
   const auto learnt=u.hyper_query('x','y','z',PRECEDES);
   require(learnt.support>0.5f,"training did not write hyperedge residual");
   const auto local=read('z',UnifiedWaveLanguage::HYPER_LOCAL_OFFSET);
   const auto group=read('z',UnifiedWaveLanguage::HYPER_SHARED_OFFSET);
   const auto field=read(0,UnifiedWaveLanguage::HYPER_SHARED_OFFSET);
   require(std::isfinite(local)&&std::isfinite(group)&&std::isfinite(field),
           "nonfinite consensus mutation");
   require(local!=old_local&&group!=old_group&&field!=old_field,
           "individual, three-head, and field votes did not all learn");
   require(u.root().h[255].data[UnifiedWaveLanguage::HYPER_SHARED_MAGIC_OFFSET]==0,
           "ingress head overwritten by consensus");
  }
  for(const std::string suffix:{"",".so3",".overlay",".overlay.tmp"})
   std::filesystem::remove(root+suffix);
  std::cout<<"PASS: native register-coupled factorized hypermatrix, indexed signed residuals, individual triad field learning, reload\n";
  return 0;
 }catch(const std::exception& e){
  std::cerr<<"FAIL: "<<e.what()<<"\n";
  for(const std::string suffix:{"",".so3",".overlay",".overlay.tmp"})
   std::filesystem::remove(root+suffix);
  return 1;
 }
}
