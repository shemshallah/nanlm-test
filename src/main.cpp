#include "nanlm/universe.hpp"
#include "nanlm/wordnet.hpp"
#include "nanlm/teacher.hpp"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <string>
using namespace nanlm;
int main(int argc,char**argv){
 try{
  if(argc<3){
   std::cerr<<"usage: nanlm ROOT {init|inspect|resident|teach-stream FILE|eval-stream FILE|ingest TEXT|write A B TYPE POLARITY WEIGHT TIME|query A B TYPE [TIME]|phase A B ANGLE|triangle A B C|hyper A B C TYPE|route HW SLOT RAW|import-wordnet DATA.NOUN}\n";
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