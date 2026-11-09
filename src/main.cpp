#include "nanlm/universe.hpp"
#include "nanlm/wordnet.hpp"
#include "nanlm/teacher.hpp"
#include "nanlm/byte_model.hpp"
#include "nanlm/recurrent_byte.hpp"
#include "nanlm/hybrid_byte.hpp"
#include <memory>
#include <cmath>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <string>
using namespace nanlm;
int main(int argc,char**argv){
 try{
  if(argc<3){
   std::cerr<<"usage: nanlm ROOT {init|inspect|resident|teach-stream FILE|eval-stream FILE|train-text FILE [EPOCHS] [TEACHER_STRIDE]|eval-text FILE|generate PROMPT N [SEED] [TEMP] [TOP_P]|ingest TEXT|write A B TYPE POLARITY WEIGHT TIME|query A B TYPE [TIME]|phase A B ANGLE|triangle A B C|hyper A B C TYPE|route HW SLOT RAW|import-wordnet DATA.NOUN}\n";
   return 2;
  }
  const std::string path=argv[1],op=argv[2];
  Universe u(path);
  auto value=[&](int i)->int{if(i>=argc)throw std::invalid_argument("missing argument");return std::stoi(argv[i]);};
  auto vert=[&](int i)->uint8_t{int x=value(i);if(x<0||x>255)throw std::out_of_range("head must be 0..255");return static_cast<uint8_t>(x);};
  if(op=="resident"){
   FabricTeacher teacher(u);
   std::cout<<"READY HW0:0 mapped_root=8388608 memory_lifetime=process"
            <<" next_byte_vocab=255 gateway=255"<<std::endl;
   std::string line;
   uint64_t steps=0;
   while(std::getline(std::cin,line)){
    std::istringstream s(line);std::string cmd;s>>cmd;
    if(cmd=="exit"||cmd=="quit"){u.sync();std::cout<<"BYE saved"<<std::endl;break;}
    if(cmd=="status"){std::cout<<"READY steps="<<steps<<" events="<<u.event_count()<<std::endl;continue;}
    if(cmd=="sync"){u.sync();std::cout<<"SAVED"<<std::endl;continue;}
    if(cmd=="train"){
     int a=-1,b=-1;s>>a>>b;
     if(a<0||a>254||b<0||b>254){std::cout<<"ERROR byte range 0..254"<<std::endl;continue;}
     const auto z=teacher.teach(static_cast<uint8_t>(a),static_cast<uint8_t>(b),++steps);
     std::cout<<std::setprecision(9)<<"TRAIN accepted="<<z.accepted
              <<" loss_before="<<z.before<<" loss_scratch="<<z.after<<std::endl;
     continue;
    }
    if(cmd=="predict"){
     int a=-1;s>>a;
     if(a<0||a>254){std::cout<<"ERROR byte range 0..254"<<std::endl;continue;}
     auto pred=teacher.predict(static_cast<uint8_t>(a));
     std::cout<<"PREDICT head="<<a<<" next="<<int(pred.choice)
              <<" probability="<<pred.probability[pred.choice]<<std::endl;
     continue;
    }
    std::cout<<"ERROR commands: status, train A B, predict A, sync, exit"<<std::endl;
   }
  }else if(op=="eval-hybrid"||op=="generate-hybrid"){
   ByteModel counts(u,path);
   RecurrentByteModel rnn(u,path);
   if(op=="eval-hybrid"){
    if(argc<4)throw std::invalid_argument("need held-out corpus");
    const double weight=argc>4?std::stod(argv[4]):0.25;
    HybridByteInference hybrid(counts,rnn,weight);
    std::ifstream validation(argv[3],std::ios::binary);
    if(!validation)throw std::runtime_error("held-out corpus not found");
    const auto m=hybrid.evaluate(validation);
    std::cout<<std::setprecision(9)<<"HYBRID_EVAL weight="<<weight
             <<" tokens="<<m.tokens<<" ce_nats="<<m.ce()
             <<" accuracy="<<m.accuracy()<<"\n";
   }else{
    if(argc<5)throw std::invalid_argument("need prompt and output length");
    const auto length=std::stoull(argv[4]);
    const auto seed=argc>5?std::stoull(argv[5]):11ull;
    const double weight=argc>6?std::stod(argv[6]):0.25;
    HybridByteInference hybrid(counts,rnn,weight);
    std::cout<<argv[3]<<hybrid.generate(argv[3],length,seed);
    std::cout.flush();
   }
  }else if(op=="benchmark-text"){
   if(argc<4)throw std::invalid_argument("need independent held-out corpus");
   ByteModel count(u,path);
   RecurrentByteModel rnn(u,path);
   std::cout<<std::setprecision(9);
   for(unsigned order=0;order<=ByteModel::ORDER;++order){
    std::ifstream validation(argv[3],std::ios::binary);
    if(!validation)throw std::runtime_error("validation file not found");
    const auto m=count.evaluate(validation,order);
    std::cout<<"BENCH model=ngram order="<<order<<" tokens="<<m.bytes
             <<" ce_nats="<<m.mean_ce()<<" accuracy="<<m.accuracy()
             <<" unseen_contexts="<<m.novel_contexts<<"\n";
   }
   std::ifstream validation(argv[3],std::ios::binary);
   if(!validation)throw std::runtime_error("validation file not found");
   const auto n=rnn.evaluate(validation);
   std::cout<<"BENCH model=rnn32 order=recurrent tokens="<<n.tokens
            <<" ce_nats="<<n.ce()<<" accuracy="<<n.accuracy()
            <<" trained_updates="<<rnn.updates()<<"\n";
  }else if(op=="train-rnn"||op=="eval-rnn"||op=="generate-rnn"){
   RecurrentByteModel rnn(u,path);
   if(op=="train-rnn"){
    if(argc<4)throw std::invalid_argument("need corpus file");
    const unsigned epochs=argc>4?static_cast<unsigned>(std::stoul(argv[4])):1u;
    if(epochs==0||epochs>1000)throw std::invalid_argument("epochs must be 1..1000");
    RecurrentByteModel::Metrics last{};
    for(unsigned epoch=0;epoch<epochs;++epoch){
     std::ifstream file(argv[3],std::ios::binary);
     if(!file)throw std::runtime_error("training corpus not found");
     last=rnn.train(file);
    }
    rnn.save();u.sync();
    std::cout<<std::setprecision(9)<<"RNN_TRAIN updates="<<rnn.updates()
             <<" last_epoch_ce="<<last.ce()<<" accuracy="<<last.accuracy()<<"\n";
   }else if(op=="eval-rnn"){
    if(argc<4)throw std::invalid_argument("need evaluation corpus");
    std::ifstream file(argv[3],std::ios::binary);
    if(!file)throw std::runtime_error("evaluation corpus not found");
    auto result=rnn.evaluate(file);
    std::cout<<std::setprecision(9)<<"RNN_EVAL tokens="<<result.tokens
             <<" ce_nats="<<result.ce()<<" accuracy="<<result.accuracy()<<"\n";
   }else{
    if(argc<5)throw std::invalid_argument("need prompt and byte count");
    const auto length=std::stoull(argv[4]);
    const auto seed=argc>5?std::stoull(argv[5]):11ull;
    const auto temperature=argc>6?std::stod(argv[6]):0.9;
    const auto top_p=argc>7?std::stod(argv[7]):0.9;
    std::cout<<argv[3]<<rnn.generate(argv[3],length,seed,temperature,top_p);
    std::cout.flush();
   }
  }else if(op=="train-text"||op=="eval-text"||op=="generate"){
   ByteModel model(u,path);
   if(op=="train-text"){
    if(argc<4)throw std::invalid_argument("need corpus file");
    const unsigned epochs=argc>4?static_cast<unsigned>(std::stoul(argv[4])):1u;
    const uint64_t stride=argc>5?std::stoull(argv[5]):0ull;
    if(epochs<1||epochs>1000)throw std::invalid_argument("epochs must be 1..1000");
    std::unique_ptr<FabricTeacher> teacher;
    if(stride)teacher=std::make_unique<FabricTeacher>(u);
    uint64_t count=0;
    for(unsigned epoch=0;epoch<epochs;++epoch){
     std::ifstream corpus(argv[3],std::ios::binary);
     if(!corpus)throw std::runtime_error("corpus not found");
     count+=model.train(corpus,teacher.get(),stride,count);
    }
    model.save();u.sync();
    std::cout<<"TRAINED bytes="<<count<<" cumulative="<<model.tokens()
             <<" contexts="<<model.contexts()<<" order="<<ByteModel::ORDER
             <<" teacher_stride="<<stride<<" root_bytes="<<ROOT_BYTES<<"\n";
   }else if(op=="eval-text"){
    if(argc<4)throw std::invalid_argument("need evaluation file");
    std::ifstream corpus(argv[3],std::ios::binary);
    if(!corpus)throw std::runtime_error("evaluation corpus not found");
    const auto m=model.evaluate(corpus);
    std::cout<<std::setprecision(9)<<"EVAL bytes="<<m.bytes
             <<" ce_nats="<<m.mean_ce()<<" bits_per_byte="
             <<m.mean_ce()/std::log(2.0)<<" accuracy="<<m.accuracy()
             <<" novel_contexts="<<m.novel_contexts<<"\n";
   }else{
    if(argc<5)throw std::invalid_argument("need quoted prompt and byte count");
    const size_t length=std::stoull(argv[4]);
    const uint64_t seed=argc>5?std::stoull(argv[5]):11ull;
    const double temperature=argc>6?std::stod(argv[6]):0.85;
    const double top_p=argc>7?std::stod(argv[7]):0.92;
    std::cout<<argv[3]<<model.generate(argv[3],length,seed,temperature,top_p);
    std::cout.flush();
   }
  }else if(op=="teach-stream"||op=="eval-stream"){
   if(argc<4)throw std::invalid_argument("need corpus file");
   std::ifstream corpus(argv[3],std::ios::binary);
   if(!corpus)throw std::runtime_error("corpus not found");
   FabricTeacher teacher(u);
   int last=-1;
   uint64_t step=0,accepted=0;
   double ce=0,after=0;
   char ch=0;
   while(corpus.get(ch)){
    const unsigned value=static_cast<unsigned char>(ch);
    if(value==255){last=-1;continue;} // reserved for nannet; valid UTF-8 never contains FF
    if(last>=0){
     if(op=="teach-stream"){
      auto z=teacher.teach(static_cast<uint8_t>(last),static_cast<uint8_t>(value),step+1);
      ce+=z.before;after+=z.after;accepted+=z.accepted?1:0;
     }else ce+=teacher.loss(static_cast<uint8_t>(last),static_cast<uint8_t>(value));
     ++step;
    }
    last=static_cast<int>(value);
   }
   if(op=="teach-stream")u.sync();
   std::cout<<std::setprecision(9)<<"pairs="<<step<<" accepted="<<accepted
            <<" mean_ce_before="<<(step?ce/step:0)
            <<" mean_scratch_ce="<<(step?after/step:0)
            <<" updated="<<(op=="teach-stream")<<"\n";
  }else if(op=="init"||op=="inspect"){
   std::cout<<"HW0:0 ready root=8388608 cells=32896 mirror_pairs=32640 palindromes=256 leaves=65536\n";
   std::cout<<"events="<<u.event_count()<<" hyperedges="<<u.hyperedge_count()<<" nominal_ram="<<u.resident_bytes_nominal()<<"\n";
  }else if(op=="ingest"){
   if(argc<4)throw std::invalid_argument("need text");
   std::string s=argv[3];
   for(unsigned char c:s)u.observe(c);
   u.sync();
   std::cout<<"transformed "<<s.size()<<" bytes into head states; original symbols preserved by caller\n";
  }else if(op=="write"){
   if(argc<9)throw std::invalid_argument("need from to type polarity weight timestamp");
   const int type=value(5);
   if(type<0||type>=16)throw std::out_of_range("relation type");
   u.write(vert(3),vert(4),static_cast<uint8_t>(type),value(6)!=0,
           std::stof(argv[7]),std::stoull(argv[8]));
   u.sync();std::cout<<"stored source relation and its typed inverse\n";
  }else if(op=="query"){
   if(argc<6)throw std::invalid_argument("need from to type");
   int type=value(5);if(type<0||type>=16)throw std::out_of_range("relation type");
   auto ev=u.query(vert(3),vert(4),static_cast<uint8_t>(type),argc>6?std::stoull(argv[6]):LATEST);
   std::cout<<"state="<<int(ev.state())<<" support="<<ev.support<<" refute="<<ev.refute<<"\n";
  }else if(op=="phase"){
   if(argc<6)throw std::invalid_argument("need a b radians");
   u.set_phase(vert(3),vert(4),std::stod(argv[5]));u.sync();
  }else if(op=="triangle"){
   if(argc<6)throw std::invalid_argument("need a b c");
   std::cout<<"oriented_triangle="<<u.triangle(vert(3),vert(4),vert(5))<<"\n";
  }else if(op=="hyper"){
   if(argc<7)throw std::invalid_argument("need a b c type");
   std::cout<<"id="<<std::hex<<u.add_hyper(vert(3),vert(4),vert(5),static_cast<uint8_t>(value(6)),false,1,1)<<"\n";
   u.sync();
  }else if(op=="route"){
   if(argc<6)throw std::invalid_argument("need hw slot raw16");
   auto packet=u.nannet_packet(vert(3),vert(4),static_cast<uint16_t>(std::stoul(argv[5],nullptr,0)),0);
   std::cout<<"packet intent HW0:0/255 -> HW"<<int(packet.hw)<<":"<<int(packet.slot)<<" raw="<<packet.ingress<<" (NOT TRANSMITTED: adapter required)\n";
  }else if(op=="import-wordnet"){
   if(argc<4)throw std::invalid_argument("need path to WordNet 3.1 data.noun");
   WordNetIndex index;
   index.import_noun_data(argv[3],u);
   index.save(path+".wordnet.tsv");
   u.sync();
   std::cout<<"synsets="<<index.concepts()<<" relations="<<index.relations()<<" concept_index="<<path<<".wordnet.tsv\n";
  }else throw std::invalid_argument("unknown command");
  return 0;
 }catch(const std::exception& ex){std::cerr<<"ERROR: "<<ex.what()<<"\n";return 1;}
}