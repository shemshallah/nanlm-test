#include "nanlm/universe.hpp"
#include "nanlm/wordnet.hpp"
#include "nanlm/teacher.hpp"
#include "nanlm/harmony.hpp"
#include "nanlm/unified.hpp"
#include "nanlm/byte_model.hpp"
#include "nanlm/recurrent_byte.hpp"
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
   std::cerr<<"usage: nanlm ROOT {init|inspect|resident|teach-stream FILE|eval-stream FILE|train-text FILE [EPOCHS] [TEACHER_STRIDE]|eval-text FILE|generate PROMPT N [SEED] [TEMP] [TOP_P] [LEMMA]|explain PROMPT [LEMMA]|spectrum PROMPT|live|adopt-state|ingest TEXT|write A B TYPE POLARITY WEIGHT TIME|query A B TYPE [TIME]|phase A B ANGLE|triangle A B C|hyper A B C TYPE|route HW SLOT RAW|import-wordnet DATA.NOUN}\n";
   return 2;
  }
  const std::string path=argv[1],op=argv[2];
  Universe u(path);
  auto value=[&](int i)->int{if(i>=argc)throw std::invalid_argument("missing argument");return std::stoi(argv[i]);};
  auto vert=[&](int i)->uint8_t{int x=value(i);if(x<0||x>255)throw std::out_of_range("head must be 0..255");return static_cast<uint8_t>(x);};
  if(op=="train-text"||op=="eval-text"||op=="generate"||op=="explain"||op=="spectrum"||op=="adopt-state"||op=="live"){
   UnifiedModel model(u,path,op=="adopt-state");
   if(op=="adopt-state"){model.save();std::cout<<"ADOPTED existing components; mixture gates reset\n";}
   else if(op=="train-text"){
    if(argc<4||argc>5)throw std::invalid_argument("need corpus file and optional epochs; legacy stride is train-count only");
    unsigned epochs=argc==5?static_cast<unsigned>(std::stoul(argv[4])):1;
    if(epochs==0||epochs>1000)throw std::invalid_argument("epochs 1..1000");
    UnifiedModel::Metrics metrics;
    for(unsigned i=0;i<epochs;++i){std::ifstream input(argv[3],std::ios::binary);if(!input)throw std::runtime_error("corpus not found");metrics=model.train(input);}
    model.save();std::cout<<"TRAINED unified_steps="<<model.steps()<<" tokens="<<model.tokens()<<" contexts="<<model.contexts()<<" causal_ce="<<metrics.ce()<<"\n";
   }else if(op=="eval-text"){
    if(argc!=4)throw std::invalid_argument("need evaluation corpus");
    std::ifstream input(argv[3],std::ios::binary);if(!input)throw std::runtime_error("corpus not found");auto metrics=model.evaluate(input);
    std::cout<<std::setprecision(10)<<"EVAL bytes="<<metrics.bytes<<" ce_nats="<<metrics.ce()<<" accuracy="<<(metrics.bytes?double(metrics.correct)/metrics.bytes:0)<<"\n";
   }else if(op=="generate"){
    if(argc<5||argc>9)throw std::invalid_argument("generate PROMPT N [SEED] [TEMP] [TOP_P] [WORDNET_LEMMA]");
    auto output=model.generate(argv[3],std::stoull(argv[4]),argc>5?std::stoull(argv[5]):11,argc>6?std::stod(argv[6]):.85,argc>7?std::stod(argv[7]):.92,argc>8?argv[8]:"");
    std::cout<<argv[3]<<output;
   }else if(op=="explain"||op=="spectrum"){
    if(argc<4||argc>5)throw std::invalid_argument("need prompt and optional WordNet lemma");
    auto state=model.prime(argv[3],argc==5?argv[4]:"");auto prediction=model.predict(state);
    if(op=="spectrum"){auto wave=model.wave(state);row_fft(wave.amplitude);for(unsigned i=0;i<256;++i)std::cout<<i<<","<<std::norm(wave.amplitude[i])/256.<<"\n";}
    else{
     for(unsigned k=0;k<UnifiedModel::K;++k)std::cout<<"WEIGHT "<<UnifiedModel::names[k]<<" "<<prediction.weights[k]<<"\n";
     for(const auto& fact:model.facts(state)){auto packet=u.nannet_packet(4,0,fact.leaf,fact.offset);std::cout<<"FACT HW4:0::0:HW0 gateway="<<int(packet.local)<<" synset=n:"<<packet.payload<<" leaf="<<packet.ingress<<" gloss="<<std::quoted(fact.gloss)<<"\n";}
     std::cout<<"byte,leaf,mirror,offset,phase16,quaternion_angle,probability,context,recurrent,boolean_phase,geometry,spectral,wordnet\n";
     for(unsigned j=0;j<255;++j){const auto& leaf=model.leaf(state,static_cast<uint8_t>(j));std::cout<<j<<","<<leaf.raw<<","<<leaf.inverse<<","<<leaf.physical_offset<<","<<binary_angle(u.covariant_phase(leaf.from,leaf.to))<<","<<u.quaternion_field().get(leaf.from,leaf.to).angle()<<","<<prediction.probability[j];for(unsigned k=0;k<UnifiedModel::K;++k)std::cout<<","<<prediction.expert[k][j];std::cout<<"\n";}
    }
   }else{
    std::cout<<"READY unified gateway=255 commands: train TEXT; generate N SEED PROMPT; predict PROMPT; save; exit"<<std::endl;
    std::string line;
    while(std::getline(std::cin,line)){std::istringstream in(line);std::string command;in>>command;
     if(command=="exit"){model.save();break;}
     if(command=="save"){model.save();std::cout<<"SAVED"<<std::endl;continue;}
     if(command=="train"){std::string text;std::getline(in>>std::ws,text);std::istringstream data(text);auto metrics=model.train(data);std::cout<<"TRAIN bytes="<<metrics.bytes<<" ce="<<metrics.ce()<<std::endl;continue;}
     if(command=="predict"){std::string prompt;std::getline(in>>std::ws,prompt);auto pred=model.predict(model.prime(prompt));std::cout<<"PREDICT byte="<<int(pred.best)<<" probability="<<pred.probability[pred.best]<<std::endl;continue;}
     if(command=="generate"){size_t n=0;uint64_t seed=11;in>>n>>seed;if(!in)throw std::invalid_argument("generate N SEED PROMPT");std::string prompt;std::getline(in>>std::ws,prompt);std::cout<<prompt<<model.generate(prompt,n,seed)<<std::endl;continue;}
     std::cout<<"ERROR unknown command"<<std::endl;
    }
   }
  }else if(op=="harmony-train"||op=="harmony-eval"){
   if(argc!=4)throw std::invalid_argument("need corpus file");
   std::ifstream input(argv[3],std::ios::binary);if(!input)throw std::runtime_error("corpus not found");
   Harmony model(u);int last=-1;char ch;size_t pairs=0;double loss=0;
   while(input.get(ch)){auto b=static_cast<uint8_t>(ch);if(b==255){last=-1;continue;}
    if(last>=0){loss+=op=="harmony-train"?model.teach(static_cast<uint8_t>(last),b):model.loss(static_cast<uint8_t>(last),b);++pairs;}last=b;
   }
   if(op=="harmony-train")u.sync();
   std::cout<<"pairs="<<pairs<<" mean_ce="<<(pairs?loss/pairs:0)<<" weights=";
   for(double w:model.expert_weights())std::cout<<w<<",";
   std::cout<<"\n";
  }else if(op=="harmony-generate"){
   if(argc!=6)throw std::invalid_argument("need start-byte length seed");
   int length=value(4);if(length<0||length>1000000)throw std::invalid_argument("length 0..1000000");
   Harmony model(u);auto text=model.generate(vert(3),static_cast<size_t>(length),std::stoull(argv[5]));
   std::cout.write(text.data(),static_cast<std::streamsize>(text.size()));
  }else if(op=="harmony-spectrum"){
   if(argc!=4)throw std::invalid_argument("need source byte");
   Harmony model(u);auto wave=model.wave(vert(3));row_fft(wave.amplitude);
   for(unsigned k=0;k<256;++k)std::cout<<k<<","<<std::norm(wave.amplitude[k])/256.<<"\n";
  }else if(op=="nan-program"){
   Harmony model(u);auto code=model.program();uint32_t payload=0;int n=nan_unpack(code.data(),&payload);
   if(n<0)throw std::runtime_error("train before exporting a program");
   std::cout<<std::hex<<std::setfill('0')<<std::setw(4)<<nan_head(unsigned(n)>>2)<<"_"<<std::setw(8)<<payload<<"_"<<std::setw(2)<<unsigned(code[1])<<std::setw(2)<<unsigned(code[0])<<"\n";
  }else if(op=="wordnet-link"){
   if(argc!=4)throw std::invalid_argument("need lemma");
   WordNetIndex index;index.load(path+".wordnet.tsv");
   for(auto id:index.lookup(argv[3])){
    const auto* c=index.get_concept(id);const auto packet=u.nannet_packet(4,0,c->leaf,id);
    std::cout<<"HW4:0::0:HW0 gateway=255 synset=n:"<<id<<" leaf="<<packet.ingress<<" lemma="<<c->lemma<<" transport=local-intent\n";
   }
  }else if(op=="resident"){
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
  }else if(op=="train-count"||op=="eval-count"||op=="generate-count"){
   ByteModel model(u,path);
   if(op=="train-count"){
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
   }else if(op=="eval-count"){
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
   if(std::filesystem::exists(path+".unified")){
    UnifiedModel model(u,path);model.import_wordnet(argv[3]);model.save();std::cout<<"WordNet imported into unified checkpoint\n";return 0;
   }
   WordNetIndex index;
   index.import_noun_data(argv[3],u);
   index.save(path+".wordnet.tsv");
   u.sync();
   std::cout<<"synsets="<<index.concepts()<<" relations="<<index.relations()<<" concept_index="<<path<<".wordnet.tsv\n";
  }else throw std::invalid_argument("unknown command");
  return 0;
 }catch(const std::exception& ex){std::cerr<<"ERROR: "<<ex.what()<<"\n";return 1;}
}