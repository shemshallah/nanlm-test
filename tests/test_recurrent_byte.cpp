#include "nanlm/recurrent_byte.hpp"
#include <cmath>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

using namespace nanlm;
int main(){
 const auto root=(std::filesystem::temp_directory_path()/
  ("nanlm-rnn-"+std::to_string(::getpid()))).string();
 const std::string text="abacabada abacabada abacabada abacabada\n";
 try{
  double first=0,after=0;
  std::string sample;
  {
   Universe u(root);
   RecurrentByteModel rnn(u,root);
   std::istringstream baseline(text);
   first=rnn.evaluate(baseline).ce();
   for(unsigned epoch=0;epoch<45;++epoch){
    std::istringstream input(text);
    const auto m=rnn.train(input);
    if(!std::isfinite(m.ce()))throw std::runtime_error("nonfinite training loss");
   }
   std::istringstream probe(text);
   after=rnn.evaluate(probe).ce();
   if(!(after+0.12<first))
    throw std::runtime_error("RNN did not learn repeated bytes");
   sample=rnn.generate("aba",100,11);
   if(sample.size()!=100||sample.find(char(255))!=std::string::npos)
    throw std::runtime_error("invalid free-running recurrent sample");
   rnn.save();u.sync();
  }
  {
   Universe u(root);
   RecurrentByteModel rnn(u,root);
   if(rnn.updates()==0||sample!=rnn.generate("aba",100,11))
    throw std::runtime_error("recurrent reload changed generation");
   std::istringstream probe(text);
   if(std::fabs(after-rnn.evaluate(probe).ce())>1e-7)
    throw std::runtime_error("recurrent reload changed CE");
  }
  for(const std::string suffix:{"",".so3",".overlay",".rnn-lm",".rnn-lm.tmp"})
   std::filesystem::remove(root+suffix);
  std::cout<<"RNN baseline_CE="<<first<<" fitted_CE="<<after<<"\n";
  std::cout<<"PASS: trainable 32D recurrent neural predictor, deterministic sampling, checkpoint reload\n";
  return 0;
 }catch(const std::exception& e){
  std::cerr<<"FAIL: "<<e.what()<<"\n";
  for(const std::string suffix:{"",".so3",".overlay",".rnn-lm",".rnn-lm.tmp"})
   std::filesystem::remove(root+suffix);
  return 1;
 }
}
