#include "nanlm/universe.hpp"
#include "nanlm/unified_wave.hpp"
#include "nanlm/wordnet.hpp"
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace nanlm;

// Production HW0:0 CLI: exactly one language engine over one root.
// Older independent n-gram/RNN probes are built as nanlm-diagnostics.
int main(int argc,char** argv){
 try{
  if(argc<3){
   std::cerr<<"usage: nanlm ROOT {init|inspect|train TEXT [EPOCHS]|eval TEXT|generate PROMPT BYTES [SEED] [TEMPERATURE] [TOP_P]|spectrum|query A B TYPE [TIME]|write A B TYPE REFUTES WEIGHT TIME|import-wordnet DATA.NOUN}\n";
   return 2;
  }
  const std::string path=argv[1],operation=argv[2];
  Universe universe(path);
  auto number=[&](int index)->int{
   if(index>=argc)throw std::invalid_argument("missing integer argument");
   return std::stoi(argv[index]);
  };
  auto head=[&](int index)->uint8_t{
   const int n=number(index);
   if(n<0||n>255)throw std::out_of_range("head must be 0..255");
   return static_cast<uint8_t>(n);
  };
  auto relation=[&](int index)->uint8_t{
   const int n=number(index);
   if(n<0||n>=16)throw std::out_of_range("relation must be 0..15");
   return static_cast<uint8_t>(n);
  };
  if(operation=="init"||operation=="inspect"){
   std::cout<<"HW0:0 root_bytes="<<ROOT_BYTES
            <<" heads=256 cognitive_heads=255 gateway=255"
            <<" mirror_pairs="<<PAIRS<<" leaves="<<LEAVES
            <<" hyperedges="<<universe.hyperedge_count()
            <<" events="<<universe.event_count()<<"\n";
  }else if(operation=="train"){
   if(argc<4)throw std::invalid_argument("training file required");
   const unsigned epochs=argc>4?static_cast<unsigned>(std::stoul(argv[4])):1u;
   if(epochs==0||epochs>1000)throw std::invalid_argument("epochs must be 1..1000");
   UnifiedWaveLanguage mind(universe);
   UnifiedWaveLanguage::Metrics last{};
   for(unsigned e=0;e<epochs;++e){
    std::ifstream file(argv[3],std::ios::binary);
    if(!file)throw std::runtime_error("training file unavailable");
    last=mind.train(file,1u);
   }
   universe.sync();
   std::cout<<std::setprecision(9)<<"UNIFIED_TRAIN bytes="<<mind.trained()
            <<" online_ce="<<last.ce()<<" online_accuracy="<<last.accuracy()
            <<" eigenvalue="<<mind.spectral().eigenvalue
            <<" eigen_residual="<<mind.spectral().residual
            <<" mirror_mass="<<mind.spectral().edge_mass<<"\n";
  }else if(operation=="eval"){
   if(argc<4)throw std::invalid_argument("evaluation file required");
   std::ifstream file(argv[3],std::ios::binary);
   if(!file)throw std::runtime_error("evaluation file unavailable");
   UnifiedWaveLanguage mind(universe);
   const auto result=mind.evaluate(file);
   std::cout<<std::setprecision(9)<<"UNIFIED_EVAL bytes="<<result.tokens
            <<" ce_nats="<<result.ce()
            <<" accuracy="<<result.accuracy()<<"\n";
  }else if(operation=="generate"){
   if(argc<5)throw std::invalid_argument("prompt and requested byte count required");
   UnifiedWaveLanguage mind(universe);
   const std::string prompt=argv[3];
   const size_t bytes=std::stoull(argv[4]);
   const uint64_t seed=argc>5?std::stoull(argv[5]):11ull;
   const double temperature=argc>6?std::stod(argv[6]):0.85;
   const double top_p=argc>7?std::stod(argv[7]):0.92;
   std::cout<<prompt<<mind.generate(prompt,bytes,seed,temperature,top_p);
   std::cout.flush();
  }else if(operation=="spectrum"){
   UnifiedWaveLanguage mind(universe);
   const auto& s=mind.spectral();
   std::cout<<std::setprecision(9)<<"UNIFIED_SPECTRUM eigenvalue="
            <<s.eigenvalue<<" residual="<<s.residual
            <<" mirror_mass="<<s.edge_mass<<" heads=255\n";
  }else if(operation=="query"){
   if(argc<6)throw std::invalid_argument("query A B TYPE [TIME]");
   const auto evidence=universe.query(head(3),head(4),relation(5),
                           argc>6?std::stoull(argv[6]):LATEST);
   std::cout<<"state="<<int(evidence.state())<<" support="
            <<evidence.support<<" refute="<<evidence.refute<<"\n";
  }else if(operation=="write"){
   if(argc<9)throw std::invalid_argument("write A B TYPE REFUTES WEIGHT TIME");
   universe.write(head(3),head(4),relation(5),number(6)!=0,
                  std::stof(argv[7]),std::stoull(argv[8]));
   universe.sync();
   std::cout<<"stored inverse-addressed temporal pair\n";
  }else if(operation=="import-wordnet"){
   if(argc<4)throw std::invalid_argument("WordNet noun data required");
   WordNetIndex index;
   index.import_noun_data(argv[3],universe);
   index.save(path+".wordnet.tsv");
   universe.sync();
   std::cout<<"synsets="<<index.concepts()<<" relations="<<index.relations()<<"\n";
  }else{
   throw std::invalid_argument("unknown operation");
  }
  return 0;
 }catch(const std::exception& error){
  std::cerr<<"ERROR: "<<error.what()<<"\n";
  return 1;
 }
}
