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
   std::cerr<<"usage: nanlm ROOT {init|inspect|train TEXT [EPOCHS] [BPTT_0_OR_1]|eval TEXT|audit TEXT|calibrate DEV_TEXT|generate PROMPT BYTES [SEED] [TEMPERATURE] [TOP_P]|spectrum|query A B TYPE [TIME]|write A B TYPE REFUTES WEIGHT TIME|import-wordnet DATA.NOUN}\n";
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
   if(argc>5){
    const int flag=number(5);
    if(flag!=0&&flag!=1)
     throw std::invalid_argument("training BPTT flag must be 0 or 1");
    mind.set_bptt_enabled(flag==1);
   }
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
            <<" mirror_mass="<<mind.spectral().edge_mass
            <<" bptt_enabled="<<int(mind.bptt_enabled())
            <<" bptt_attempted="<<mind.bptt_attempted()
            <<" bptt_accepted="<<mind.bptt_accepted()<<"\n";
  }else if(operation=="eval"){
   if(argc<4)throw std::invalid_argument("evaluation file required");
   std::ifstream file(argv[3],std::ios::binary);
   if(!file)throw std::runtime_error("evaluation file unavailable");
   UnifiedWaveLanguage mind(universe);
   const auto result=mind.evaluate(file);
   std::cout<<std::setprecision(9)<<"UNIFIED_EVAL bytes="<<result.tokens
            <<" ce_nats="<<result.ce()
            <<" accuracy="<<result.accuracy()<<"\n";
  }else if(operation=="audit"){
   if(argc<4)throw std::invalid_argument("audit file required");
   UnifiedWaveLanguage mind(universe);
   struct Ablation {const char* name;unsigned disabled;};
   const Ablation runs[]={
    {"full",0},
    {"no_wave",UnifiedWaveLanguage::ABLATE_WAVE},
    {"no_attention",UnifiedWaveLanguage::ABLATE_ATTENTION},
    {"no_spectral",UnifiedWaveLanguage::ABLATE_SPECTRAL},
    {"no_cycle",UnifiedWaveLanguage::ABLATE_CYCLE},
    {"no_inverse_evidence",UnifiedWaveLanguage::ABLATE_INVERSE_EVIDENCE},
    {"no_hyperedge",UnifiedWaveLanguage::ABLATE_HYPEREDGE},
    {"no_long_context",UnifiedWaveLanguage::ABLATE_LONG_CONTEXT},
    {"no_tbptt_readout",UnifiedWaveLanguage::ABLATE_BPTT_READOUT},
    {"teacher_only",UnifiedWaveLanguage::ABLATE_ALL}
   };
   for(const auto& run:runs){
    std::ifstream file(argv[3],std::ios::binary);
    if(!file)throw std::runtime_error("audit file unavailable");
    const auto result=mind.evaluate(file,run.disabled);
    std::cout<<std::setprecision(9)<<"UNIFIED_AUDIT mode="<<run.name
             <<" tokens="<<result.tokens<<" ce_nats="<<result.ce()
             <<" accuracy="<<result.accuracy()<<"\n";
   }
  }else if(operation=="calibrate"){
   if(argc!=4)throw std::invalid_argument("calibrate requires separate development text");
   UnifiedWaveLanguage mind(universe);
   const auto measure=[&](){
    std::ifstream file(argv[3],std::ios::binary);
    if(!file)throw std::runtime_error("development file unavailable");
    const auto result=mind.evaluate(file);
    if(result.tokens==0||!std::isfinite(result.ce()))
     throw std::runtime_error("invalid development loss");
    return result;
   };
   const auto before=measure();
   auto chosen=mind.global_scales();
   const double choices[]={0.,0.5,1.,1.5,2.};
   for(size_t mode=0;mode<chosen.size();++mode){
    double best_loss=measure().ce();
    double best_scale=chosen[mode];
    for(double candidate:choices){
     if(candidate==chosen[mode])continue;
     auto trial=chosen;
     trial[mode]=candidate;
     mind.set_global_scales(trial);
     const double loss=measure().ce();
     if(loss+1e-6<best_loss){best_loss=loss;best_scale=candidate;}
    }
    chosen[mode]=best_scale;
    mind.set_global_scales(chosen);
   }
   // Fit CP-tensor and inverse-wave factors only to development text.
   // Signed proposals allow anti-alignment without deleting geometry.
   auto tensor=mind.tensor_scales();
   const double tensor_choices[]={-2.,-1.5,-1.,-0.5,0.,0.5,1.,1.5,2.};
   for(size_t mode=0;mode<tensor.size();++mode){
    double best_loss=measure().ce();
    double best_scale=tensor[mode];
    for(double candidate:tensor_choices){
     if(candidate==tensor[mode])continue;
     auto trial=tensor;
     trial[mode]=candidate;
     mind.set_tensor_scales(trial);
     const double loss=measure().ce();
     if(loss+1e-6<best_loss){best_loss=loss;best_scale=candidate;}
    }
    tensor[mode]=best_scale;
    mind.set_tensor_scales(tensor);
   }
   const auto after=measure();
   if(after.ce()>before.ce()+1e-7)
    throw std::runtime_error("development calibration worsened loss");
   universe.sync();
   std::cout<<std::setprecision(9)<<"UNIFIED_CALIBRATE dev_tokens="<<after.tokens
            <<" ce_before="<<before.ce()<<" ce_after="<<after.ce()
            <<" wave_scale="<<chosen[0]<<" attention_scale="<<chosen[1]
            <<" spectral_scale="<<chosen[2]<<" cycle_scale="<<chosen[3]
            <<" cp_scale="<<tensor[0]
            <<" inverse_wave_tensor_scale="<<tensor[1]<<"\n";
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
