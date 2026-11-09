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
   const auto factor=mind.predict(context);
   const auto without=mind.predict(context,UnifiedWaveLanguage::ABLATE_HYPEREDGE);
   require(std::fabs(factor.p['z']-without.p['z'])>1e-7,
           "factorized third-order mirror coupling ablates to zero");
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
