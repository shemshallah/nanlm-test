#include "nanlm/byte_model.hpp"
#include <cassert>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>

using namespace nanlm;
int main(){
 const auto root=(std::filesystem::temp_directory_path()/
  ("nanlm-byte-"+std::to_string(::getpid()))).string();
 const std::string train=
  "the tiny engine learns that the river moves around the stone.\n"
  "the tiny engine learns that the wind moves around the hill.\n"
  "the tiny engine learns that the light moves across the field.\n";
 const std::string holdout=
  "the tiny engine learns that the wind moves around the stone.\n";
 try {
  {
   Universe u(root);
   ByteModel m(u,root);
   std::istringstream input(train);
   const auto trained=m.train(input);
   if(trained!=train.size()||m.tokens()!=train.size()||m.contexts()<30)
    throw std::runtime_error("did not train expected byte sequence");
   const auto before=m.predict(ByteModel::Context{});
   double mass=0;
   for(double p:before.probability)mass+=p;
   if(std::fabs(mass-1)>1e-9)throw std::runtime_error("mass normalization failed");
   std::istringstream held(holdout);
   const auto result=m.evaluate(held);
   if(result.bytes!=holdout.size()||!std::isfinite(result.mean_ce())||
      result.mean_ce()<=0||result.mean_ce()>5.55)
    throw std::runtime_error("held-out CE is invalid or worse than uniform");
   const auto generated=m.generate("the tiny ",160,11);
   if(generated.size()!=160||generated.find(char(255))!=std::string::npos)
    throw std::runtime_error("generation invalid length or reserved byte");
   if(m.generate("the tiny ",160,11)!=generated)
    throw std::runtime_error("generation is nondeterministic");
   const auto first=m.generate("the tiny ",160,11);
   m.save();
   u.sync();
   std::cout<<"heldout_CE="<<result.mean_ce()<<" accuracy="<<result.accuracy()
            <<" context_nodes="<<m.contexts()<<" samples="<<trained<<"\n";
   (void)first;
  }
  {
   Universe u(root);
   ByteModel m(u,root);
   if(m.tokens()!=train.size())
    throw std::runtime_error("training token count lost after reload");
   const std::string before=m.generate("the tiny ",160,11);
   const std::string after=m.generate("the tiny ",160,11);
   if(before!=after)throw std::runtime_error("generation drifted after reload");
   std::istringstream held(holdout);
   const auto result=m.evaluate(held);
   if(!std::isfinite(result.mean_ce()))
    throw std::runtime_error("reloaded evaluation broken");
   std::istringstream additional("the tiny engine learns.\n");
   m.train(additional);
   if(m.tokens()!=train.size()+std::string("the tiny engine learns.\n").size())
    throw std::runtime_error("incremental training count wrong");
   m.save();
  }
  {
   Universe u(root);
   ByteModel m(u,root);
   if(m.tokens()!=train.size()+std::string("the tiny engine learns.\n").size())
    throw std::runtime_error("incremental training not persisted");
  }
  for(const std::string suffix:{"",".so3",".overlay",".byte-lm",".byte-lm.tmp"})
   std::filesystem::remove(root+suffix);
  std::cout<<"PASS: variable-order causal autoregression, held-out CE, "
           <<"persistence, incremental learning and free-running deterministic sampling\n";
  return 0;
 }catch(const std::exception& e){
  std::cerr<<"FAIL: "<<e.what()<<"\n";
  for(const std::string suffix:{"",".so3",".overlay",".byte-lm",".byte-lm.tmp"})
   std::filesystem::remove(root+suffix);
  return 1;
 }
}
