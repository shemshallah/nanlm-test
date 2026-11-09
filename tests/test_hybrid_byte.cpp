#include "nanlm/hybrid_byte.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
using namespace nanlm;
int main(){
 const auto root=(std::filesystem::temp_directory_path()/
  ("nanlm-hybrid-"+std::to_string(::getpid()))).string();
 const std::string train="echoes in the hills.\nechoes in the sky.\nechoes in the air.\n";
 const std::string held="echoes in the garden.\n";
 try {
  {
   Universe u(root);
   ByteModel count(u,root);
   RecurrentByteModel rnn(u,root);
   std::istringstream text(train);
   count.train(text);count.save();
   for(unsigned i=0;i<8;++i){
    std::istringstream input(train);
    rnn.train(input);
   }
   rnn.save();
   HybridByteInference statistical(count,rnn,0.0);
   HybridByteInference neural(count,rnn,1.0);
   HybridByteInference mixture(count,rnn,0.3);
   HybridByteInference::State state{};
   for(unsigned char c:std::string("echoes in "))mixture.observe(state,c);
   const auto pc=count.predict(state.suffix).probability;
   const auto pr=rnn.predict(state.recurrent);
   const auto ph=mixture.predict(state);
   const auto p0=statistical.predict(state);
   const auto p1=neural.predict(state);
   for(size_t j=0;j<ByteModel::VOCAB;++j){
    if(std::fabs(p0[j]-pc[j])>1e-14||std::fabs(p1[j]-pr[j])>1e-14||
       std::fabs(ph[j]-(0.7*pc[j]+0.3*pr[j]))>1e-12)
     throw std::runtime_error("hybrid pathway incorrect");
   }
   std::istringstream input(held);
   const auto metrics=mixture.evaluate(input);
   if(metrics.tokens!=held.size()-1||!std::isfinite(metrics.ce()))
    throw std::runtime_error("hybrid held-out evaluation invalid");
   const auto a=mixture.generate("echoes in ",128,19);
   if(a.size()!=128||a!=mixture.generate("echoes in ",128,19))
    throw std::runtime_error("hybrid generation not deterministic");
  }
  for(const auto& suffix:{"",".so3",".overlay",".byte-lm",".rnn-lm"})
   std::filesystem::remove(root+suffix);
  std::cout<<"PASS: neural-statistical hybrid mixture, causal held-out evaluation, free generation\n";
  return 0;
 }catch(const std::exception& e){
  std::cerr<<"FAIL: "<<e.what()<<"\n";
  for(const auto& suffix:{"",".so3",".overlay",".byte-lm",".rnn-lm"})
   std::filesystem::remove(root+suffix);
  return 1;
 }
}
