#pragma once
#include "nanlm/teacher.hpp"
#include "nanlm/triad_waveformer.hpp"
#include "nanlm/utf8.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <istream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace nanlm {

// One HW0:0 Root, 255 active palindromic cognitive heads, 32,640 mirror
// pairs. The 256th head (255) remains dedicated to network ingress.
//
// NOT 255 independently trained language models: these are 255 coupled
// attention/excitation channels of ONE conditional distribution. The only
// persistent trainable state here is the existing FabricTeacher's head
// vectors, gates, relation evidence, phases, and SO(3) transports.
// Wave memory is an ephemeral causal activation, recomputed per stream.
//
// The standing wave obeys x(t+1)=2*d*cos(omega)*x(t)-d^2*x(t-1)+input
// (with a bounded neighbor-scattering contribution). On graph excitation,
// eigenvectors of symmetric mirror coupling are obtained by power iteration.
// This is an experimental spectral graph recurrent predictor, not yet a
// general transformer or an end-to-end differentiable spectral learner.
class UnifiedWaveLanguage {
public:
 static constexpr size_t N=FabricTeacher::VOCAB;
 static constexpr size_t WINDOW=64;
 static constexpr size_t ATTENTION_WINDOW=16;
 static constexpr size_t LONG_GAIN_OFFSET=44; // float32 per head, former unused register gap
 static constexpr size_t MODES=4;
 static constexpr size_t GROUP_SIZE=3;
 static constexpr size_t GROUPS=(N+GROUP_SIZE-1)/GROUP_SIZE; // 85 voting triads
 // Bitmask for read-only, single-root counterfactual evaluations.
 static constexpr unsigned ABLATE_WAVE=1u<<0;
 static constexpr unsigned ABLATE_ATTENTION=1u<<1;
 static constexpr unsigned ABLATE_SPECTRAL=1u<<2;
 static constexpr unsigned ABLATE_CYCLE=1u<<3;
 static constexpr unsigned ABLATE_INVERSE_EVIDENCE=1u<<4;
 static constexpr unsigned ABLATE_HYPEREDGE=1u<<5;
 static constexpr unsigned ABLATE_LONG_CONTEXT=1u<<6;
 static constexpr unsigned ABLATE_BPTT_READOUT=1u<<7;
 static constexpr unsigned ABLATE_TRIAD=1u<<8;
 static constexpr unsigned ABLATE_FIELD=1u<<9;
 static constexpr unsigned ABLATE_ALL=(1u<<10)-1u;
 static constexpr size_t BPTT_STEPS=8;
 static constexpr size_t BPTT_INTERVAL=16;
 static constexpr size_t BPTT_INPUT_OFFSET=112;
 static constexpr size_t BPTT_FREQUENCY_OFFSET=116;
 static constexpr size_t BPTT_MAGIC_OFFSET=120;
 static constexpr size_t BPTT_FREQUENCY_MAGIC_OFFSET=121;
 static constexpr uint8_t BPTT_MAGIC=0xEB;
 static constexpr uint8_t BPTT_FREQUENCY_MAGIC=0xEC;
 static constexpr double BPTT_READOUT=0.10;
 static constexpr size_t HEAD_GAIN_OFFSET=80;
 static constexpr size_t HEAD_GAIN_MAGIC_OFFSET=60;
 static constexpr uint8_t HEAD_GAIN_MAGIC=0xC7;
 // Four shared calibration multipliers fit in *unused* bytes 64..71 of
 // palindrome head 0, as unsigned little-endian Q8.8. The gauge potential
 // at 72..79 and per-head gains at 80..95 remain completely untouched.
 static constexpr size_t CALIBRATION_MAGIC_OFFSET=61;
 static constexpr size_t CALIBRATION_OFFSET=64;
 static constexpr uint8_t CALIBRATION_MAGIC=0xD3;
 // Fixed-point register ownership: local hypergain [28..31] per head,
 // triad gain [12..15] in the third head, field gain [12..15] in head 0.
 // Markers [62,63]. No mirror lane, embedding, gauge, or ingress overlap.
 static constexpr size_t HYPER_LOCAL_OFFSET=28;
 static constexpr size_t HYPER_SHARED_OFFSET=12;
 static constexpr size_t HYPER_LOCAL_MAGIC_OFFSET=62;
 static constexpr size_t HYPER_SHARED_MAGIC_OFFSET=63;
 static constexpr uint8_t HYPER_LOCAL_MAGIC=0xB6;
 static constexpr uint8_t HYPER_GROUP_MAGIC=0xB7;
 static constexpr uint8_t HYPER_FIELD_MAGIC=0xB8;
 // Byte-disjoint shared QKV, adapters and per-head gains in the fixed root.
 // Explicitly avoid Universe observation state at [8..11], count [16..19],
 // accepted proposals [20..23], and reserved SO(3) rule quaternions in
 // heads 6/7 at [96..111]. The QKV slots skip these two heads.
 static constexpr size_t TRIAD_GAIN_OFFSET=124;
 static constexpr size_t TRIAD_PROJECTION_OFFSET=105;
 static constexpr size_t TRIAD_ADAPTER_OFFSET=100;
 static constexpr size_t TRIAD_KERNEL_MARKER_OFFSET=123;
 static constexpr uint8_t TRIAD_KERNEL_MAGIC=0xD6;
 static constexpr size_t TRIAD_ID_OFFSET=64;
 static constexpr size_t TRIAD_MAGIC_OFFSET=65;
 static constexpr uint8_t TRIAD_MAGIC=0xE5;
 // Optional calibration of rank-three CP and wave tensor modulations.
 // Two float32 scales + marker use h0 [96..104], disjoint from its
 // four-output Q8.8 scales, its gauge, and the teacher's vectors.
 static constexpr size_t TENSOR_SCALE_OFFSET=96;
 static constexpr size_t TENSOR_SCALE_MAGIC_OFFSET=104;
 static constexpr uint8_t TENSOR_SCALE_MAGIC=0xC9;
 using TensorScales=std::array<double,2>;
 using GlobalScales=std::array<double,4>;
 struct State {
  TriadWaveFormer::Workspace workspace{};
  std::array<double,N> wave{};
  std::array<double,N> old_wave{};
  std::array<double,N> inverse_wave{};
  std::array<double,N> inverse_old_wave{};
  std::array<uint8_t,WINDOW> history{};
  size_t length=0;
  uint64_t tick=0;
  void push(uint8_t c) {
   if(length<WINDOW){history[length++]=c;}
   else {
    for(size_t j=1;j<WINDOW;++j)history[j-1]=history[j];
    history[WINDOW-1]=c;
   }
  }
  void reset(){*this=State{};}
 };
 struct Metrics {
  uint64_t tokens=0,correct=0;
  double nll=0;
  double ce()const{return tokens?nll/tokens:0.0;}
  double accuracy()const{return tokens?double(correct)/tokens:0.0;}
 };
 struct Spectral {
  std::array<std::array<double,N>,MODES> vectors{};
  std::array<double,MODES> eigenvalues{},residuals{};
  std::array<double,N> vector{}; // first spectral mode compatibility
  double eigenvalue=0,residual=0,edge_mass=0;
 };
 struct Prediction {
  std::array<double,N> p{};
  uint8_t best=0;
 };
 struct BpttGradient {
  std::array<double,N> input{},frequency{};
  double loss=0,norm=0;
  size_t steps=0;
 };
private:
 Universe& universe_;
 FabricTeacher teacher_;
 double evidence_gain_=0.08;
 double damping_=0.80;
 double omega_=0.53;
 std::array<std::array<double,N>,N> adjacency_{};
 std::array<std::array<double,N>,N> directed_{};
 struct DirectedArc {uint8_t head=0;double weight=0;};
 std::array<std::vector<DirectedArc>,N> incoming_{},outgoing_{};
 std::array<double,N> in_degree_{},out_degree_{};
 Spectral eigen_{};
 uint64_t trained_=0,bptt_accepted_=0,bptt_attempted_=0;
 bool bptt_enabled_=true;
 bool frequency_learning_enabled_=false; // opt-in pending source-disjoint gains
 TriadWaveFormer::Parameters triad_parameters_{};
 uint64_t triad_accepted_=0,projection_accepted_=0;
 bool triad_training_enabled_=true;
 using GainVector=std::array<float,4>;
 static GainVector initial_gains(){return {0.12f,0.16f,0.06f,0.045f};}
 GainVector gains(uint8_t head)const{
  GainVector out{};
  const auto& h=universe_.root().h[head];
  if(h.data[HEAD_GAIN_MAGIC_OFFSET]!=HEAD_GAIN_MAGIC)return initial_gains();
  std::memcpy(out.data(),h.data+HEAD_GAIN_OFFSET,sizeof(out));
  return out;
 }
 void write_gains(uint8_t head,const GainVector& g){
  auto& h=universe_.root().h[head];
  std::memcpy(h.data+HEAD_GAIN_OFFSET,g.data(),sizeof(g));
  h.data[HEAD_GAIN_MAGIC_OFFSET]=HEAD_GAIN_MAGIC;
 }
 void store_triad_parameters(const TriadWaveFormer::Parameters& p){
  for(size_t i=0;i<TriadWaveFormer::PROJECTION_WEIGHTS;++i)
   write_scalar(universe_.root().h[i<6?i:i+2],TRIAD_PROJECTION_OFFSET,p.qkv[i]);
  for(size_t i=0;i<N;++i)
   write_scalar(universe_.root().h[i],TRIAD_GAIN_OFFSET,p.head_gain[i]);
  for(size_t i=0;i<GROUPS;++i)
   write_scalar(universe_.root().h[group_register(i)],TRIAD_ADAPTER_OFFSET,p.adapter[i]);
  universe_.root().h[0].data[TRIAD_KERNEL_MARKER_OFFSET]=TRIAD_KERNEL_MAGIC;
  triad_parameters_=p;
 }
 void load_triad_parameters(){
  if(universe_.root().h[0].data[TRIAD_KERNEL_MARKER_OFFSET]!=TRIAD_KERNEL_MAGIC){
   store_triad_parameters(TriadWaveFormer::defaults());return;
  }
  TriadWaveFormer::Parameters p{};
  for(size_t i=0;i<TriadWaveFormer::PROJECTION_WEIGHTS;++i)
   p.qkv[i]=read_scalar(universe_.root().h[i<6?i:i+2],TRIAD_PROJECTION_OFFSET);
  for(size_t i=0;i<N;++i)
   p.head_gain[i]=read_scalar(universe_.root().h[i],TRIAD_GAIN_OFFSET);
  for(size_t i=0;i<GROUPS;++i)
   p.adapter[i]=read_scalar(universe_.root().h[group_register(i)],TRIAD_ADAPTER_OFFSET);
  validate_triad_parameters(p);
  triad_parameters_=p;
 }
 static void validate_triad_parameters(const TriadWaveFormer::Parameters& p){
  for(float v:p.qkv)if(!std::isfinite(v)||std::fabs(v)>2)
   throw std::runtime_error("invalid shared QKV parameter");
  for(float v:p.head_gain)if(!std::isfinite(v)||std::fabs(v)>1)
   throw std::runtime_error("invalid triad head gain");
  for(float v:p.adapter)if(!std::isfinite(v)||std::fabs(v)>1)
   throw std::runtime_error("invalid triad adapter");
 }
 static constexpr uint8_t group_register(size_t group){
  return static_cast<uint8_t>(group*GROUP_SIZE+GROUP_SIZE-1);
 }
 static float read_scalar(const Head& h,size_t offset){
  float result=0;
  std::memcpy(&result,h.data+offset,sizeof(result));
  return result;
 }
 static void write_scalar(Head& h,size_t offset,float value){
  std::memcpy(h.data+offset,&value,sizeof(value));
 }
 double input_gain(uint8_t head)const{
  if(head>=N)throw std::out_of_range("reserved ingress has no recurrent weight");
  const auto& h=universe_.root().h[head];
  if(h.data[BPTT_MAGIC_OFFSET]!=BPTT_MAGIC)return 1.0;
  const float value=read_scalar(h,BPTT_INPUT_OFFSET);
  if(!std::isfinite(value)||value<0.6f||value>1.4f)
   throw std::runtime_error("invalid persisted recurrent input gain");
  return value;
 }
 void write_input_gain(uint8_t head,float value){
  if(head>=N||!std::isfinite(value)||value<0.6f||value>1.4f)
   throw std::invalid_argument("recurrent input gain must be finite in [0.6,1.4]");
  auto& h=universe_.root().h[head];
  write_scalar(h,BPTT_INPUT_OFFSET,value);
  h.data[BPTT_MAGIC_OFFSET]=BPTT_MAGIC;
 }
 double recurrent_frequency(uint8_t head)const{
  if(head>=N)throw std::out_of_range("ingress is not a wave head");
  const auto& h=universe_.root().h[head];
  if(h.data[BPTT_FREQUENCY_MAGIC_OFFSET]!=BPTT_FREQUENCY_MAGIC)return 0.0;
  const float value=read_scalar(h,BPTT_FREQUENCY_OFFSET);
  if(!std::isfinite(value)||value< -0.15f||value>0.15f)
   throw std::runtime_error("invalid recurrent frequency shift");
  return value;
 }
 void write_frequency(uint8_t head,float value){
  if(head>=N||!std::isfinite(value)||value< -0.15f||value>0.15f)
   throw std::invalid_argument("frequency shift outside [-0.15,0.15]");
  auto& h=universe_.root().h[head];
  write_scalar(h,BPTT_FREQUENCY_OFFSET,value);
  h.data[BPTT_FREQUENCY_MAGIC_OFFSET]=BPTT_FREQUENCY_MAGIC;
 }
 float local_hyper(uint8_t head)const{
  return read_scalar(universe_.root().h[head],HYPER_LOCAL_OFFSET);
 }
 size_t triad_of(uint8_t head)const{
  if(head>=N)throw std::out_of_range("ingress has no cognitive triad");
  if(head==0)return 0;
  const auto& h=universe_.root().h[head];
  if(h.data[TRIAD_MAGIC_OFFSET]!=TRIAD_MAGIC)return head/GROUP_SIZE;
  const size_t group=h.data[TRIAD_ID_OFFSET];
  if(group>=GROUPS)throw std::runtime_error("invalid persistent triad");
  return group;
 }
 float group_hyper(size_t group)const{
  return read_scalar(universe_.root().h[group_register(group)],HYPER_SHARED_OFFSET);
 }
 float field_hyper()const{
  return read_scalar(universe_.root().h[0],HYPER_SHARED_OFFSET);
 }
 double hyper_gain(uint8_t head)const{
  return double(local_hyper(head))+group_hyper(triad_of(head))+field_hyper();
 }
 struct Features {
  double wave=0,attention=0,spectral=0,cycle=0;
  double at(size_t i)const{
   switch(i){case 0:return wave;case 1:return attention;
             case 2:return spectral;default:return cycle;}
  }
 };
 static double clamp(double x,double limit=8.0){
  return std::clamp(x,-limit,limit);
 }
 static double dot3(const std::array<float,3>& a,const std::array<float,3>& b){
  return double(a[0])*b[0]+double(a[1])*b[1]+double(a[2])*b[2];
 }
 static std::array<float,3> vector(const Head& h,size_t offset){
  std::array<float,3> out{};
  std::memcpy(out.data(),h.data+offset,sizeof(out));
  return out;
 }
 static uint64_t rand64(uint64_t& x){
  x+=0x9e3779b97f4a7c15ull;
  uint64_t z=x;
  z=(z^(z>>30))*0xbf58476d1ce4e5b9ull;
  z=(z^(z>>27))*0x94d049bb133111ebull;
  return z^(z>>31);
 }
 static double random01(uint64_t& x){
  return double(rand64(x)>>11)*(1.0/9007199254740992.0);
 }
 static double norm(const std::array<double,N>& v){
  double sum=0;
  for(double x:v)sum+=x*x;
  return std::sqrt(sum);
 }
 double edge_weight(uint8_t a,uint8_t b)const{
  if(a==b)return 0;
  const auto evidence=universe_.query(a,b,PRECEDES);
  const auto inverse=universe_.query(b,a,FOLLOWS);
  const double strength=0.5*(evidence.support+inverse.support);
  const double opposition=0.5*(evidence.refute+inverse.refute);
  const double theta=universe_.covariant_phase(a,b);
  // Pair mirror lanes carry both temporal orientations. This is a
  // symmetric graph scattering amplitude, not the directed prediction.
  return clamp((strength-opposition)/(1.0+strength+opposition),1.0)*
         std::cos(theta);
 }
 void rebuild_spectrum(){
  eigen_=Spectral{};
  for(size_t i=0;i<N;++i) {
   for(size_t j=i+1;j<N;++j){
    const double forward=edge_weight(uint8_t(i),uint8_t(j));
    const double backward=edge_weight(uint8_t(j),uint8_t(i));
    directed_[i][j]=forward;
    directed_[j][i]=backward;
    const double w=0.5*(forward+backward);
    adjacency_[i][j]=w;
    adjacency_[j][i]=w;
    eigen_.edge_mass+=std::fabs(w);
   }
   adjacency_[i][i]=0;
   directed_[i][i]=0;
  }
  // Derived, deterministic mirror arcs share the identical NaN16 pair bank.
  for(size_t i=0;i<N;++i){
   incoming_[i].clear();outgoing_[i].clear();
   in_degree_[i]=1;out_degree_[i]=1;
   for(size_t j=0;j<N;++j){
    const double forward=directed_[j][i],inverse=directed_[i][j];
    if(forward!=0)incoming_[i].push_back({uint8_t(j),forward});
    if(inverse!=0)outgoing_[i].push_back({uint8_t(j),inverse});
    in_degree_[i]+=std::fabs(forward);
    out_degree_[i]+=std::fabs(inverse);
   }
  }
  if(eigen_.edge_mass<1e-12)return;
  for(size_t mode=0;mode<MODES;++mode){
   std::array<double,N> v{};
   for(size_t i=0;i<N;++i)
    v[i]=std::cos((mode+1)*(i+0.5)*TAU/N)+
         0.13*std::sin((mode+2)*(i+0.5)*TAU/N);
   auto orthogonalize=[&](std::array<double,N>& target){
    for(size_t prior=0;prior<mode;++prior){
     double projection=0;
     for(size_t i=0;i<N;++i)
      projection+=target[i]*eigen_.vectors[prior][i];
     for(size_t i=0;i<N;++i)
      target[i]-=projection*eigen_.vectors[prior][i];
    }
   };
   orthogonalize(v);
   const double initial=norm(v);
   if(initial<1e-13)continue;
   for(double& x:v)x/=initial;
   for(int iteration=0;iteration<64;++iteration){
    std::array<double,N> next{};
    for(size_t i=0;i<N;++i)
     for(size_t j=0;j<N;++j)
      next[i]+=adjacency_[i][j]*v[j];
    orthogonalize(next);
    const double length=norm(next);
    if(length<1e-13)break;
    for(size_t i=0;i<N;++i)v[i]=next[i]/length;
   }
   std::array<double,N> av{};
   for(size_t i=0;i<N;++i)
    for(size_t j=0;j<N;++j)
     av[i]+=adjacency_[i][j]*v[j];
   double lambda=0,residual=0;
   for(size_t i=0;i<N;++i)lambda+=v[i]*av[i];
   for(size_t i=0;i<N;++i)
    residual+=(av[i]-lambda*v[i])*(av[i]-lambda*v[i]);
   eigen_.vectors[mode]=v;
   eigen_.eigenvalues[mode]=lambda;
   eigen_.residuals[mode]=std::sqrt(residual);
  }
  eigen_.vector=eigen_.vectors[0];
  eigen_.eigenvalue=eigen_.eigenvalues[0];
  eigen_.residual=eigen_.residuals[0];
 }
 double attention(const State& s,uint8_t candidate)const{
  if(s.length==0)return 0;
  // Each of the 255 head-specific query vectors attends to the window.
  const auto q=vector(universe_.root().h[candidate],48);
  std::array<double,WINDOW> scores{};
  double best=-std::numeric_limits<double>::infinity();
  const size_t start=s.length>ATTENTION_WINDOW?s.length-ATTENTION_WINDOW:0;
  for(size_t k=start;k<s.length;++k){
   const uint8_t src=s.history[k];
   const auto key=vector(universe_.root().h[src],32);
   const double geometry=0.11*std::cos(universe_.covariant_phase(src,candidate));
   const double temporal=0.04*std::cos(omega_*double(s.length-k));
   scores[k]=dot3(q,key)*2.0+geometry+temporal;
   best=std::max(best,scores[k]);
  }
  double denominator=0,aggregation=0;
  for(size_t k=start;k<s.length;++k)denominator+=std::exp(scores[k]-best);
  for(size_t k=start;k<s.length;++k){
   const uint8_t src=s.history[k];
   const double weight=std::exp(scores[k]-best)/denominator;
   const auto forward=universe_.query(src,candidate,PRECEDES);
   const auto inverse=universe_.query(candidate,src,FOLLOWS);
   const double evidence=(forward.support+inverse.support)/
                    (2.0+forward.support+inverse.support);
   const double phase=universe_.covariant_phase(src,candidate);
   aggregation+=weight*(s.wave[src]+0.4*evidence+
                  0.12*std::sin(phase+omega_*(s.length-k)));
  }
  return clamp(aggregation,3.0);
 }
 double cycle(uint8_t a,uint8_t b,uint8_t c)const{
  if(a==b||b==c||a==c)return 0;
  // The anchored U(1) cycle is already gauge invariant, so no dense
  // C(256,3) allocation or second hypergraph representation is needed.
  const double loop=universe_.triangle(a,b,c);
  const auto qa=universe_.quaternion_field().get(a,b);
  const auto qb=universe_.quaternion_field().get(b,c);
  const auto qc=universe_.quaternion_field().get(c,a);
  const double nonabelian=(qa*qb*qc).angle();
  return 0.5*std::cos(loop)+0.5*std::cos(nonabelian);
 }
 // Causal eligibility over the long-tail portion of 48 remembered bytes.
 // A learned head-local gain is stored in the EXISTING 4-byte register gap
 // between the teacher source vector [32,43] and target vector [48,59].
 // This is truncated feature-gradient credit assignment, NOT full BPTT.
 double long_context_signal(const State& s,uint8_t candidate)const{
  if(s.length<=ATTENTION_WINDOW)return 0;
  double value=0,denom=0;
  const size_t older=s.length-ATTENTION_WINDOW;
  for(size_t k=0;k<older;++k){
   const uint8_t from=s.history[k];
   if(from==candidate)continue;
   const auto forward=universe_.query(from,candidate,PRECEDES);
   const auto inverse=universe_.query(candidate,from,FOLLOWS);
   const double support=0.5*(forward.support+inverse.support);
   const double refute=0.5*(forward.refute+inverse.refute);
   const double signed_evidence=(support-refute)/(1.0+support+refute);
   const double lag=double(s.length-k);
   const double weight=std::exp(-0.048*lag);
   value+=weight*signed_evidence;
   denom+=weight;
  }
  return denom>0?clamp(value/denom,2.0):0.0;
 }
 double hyperedge_signal(const State& s,uint8_t candidate)const{
  if(s.length<2)return 0;
  const uint8_t a=s.history[s.length-2],b=s.history[s.length-1];
  if(a==b||a==candidate||b==candidate)return 0;
  // Logical factorized H[a,b,c]: pair-lane interaction with a directed,
  // genuinely third-order product. Every pair comes from the same mapped
  // NaN16 register, no redundant 255^3 allocation or second language model.
  const auto pair_score=[&](uint8_t from,uint8_t to){
   const auto forward=universe_.query(from,to,PRECEDES);
   const auto backward=universe_.query(to,from,FOLLOWS);
   const double pos=0.5*(forward.support+backward.support);
   const double neg=0.5*(forward.refute+backward.refute);
   return (pos-neg)/(1.0+pos+neg);
  };
  const double ab=pair_score(a,b);
  const double bc=pair_score(b,candidate);
  const double ac=pair_score(a,candidate);
  const double holonomy=0.65+0.35*std::cos(universe_.triangle(a,b,candidate));
  // CP-rank-three multilinear interaction is directly tied to the
  // trainable 3-vectors in palindromic head registers. This is not
  // another embedding table: the teacher and attention update/read
  // these exact head bytes.
  const auto& root=universe_.root();
  const auto first=vector(root.h[a],32);
  const auto middle=vector(root.h[b],48);
  const auto last=vector(root.h[candidate],48);
  double cp=0;
  for(size_t k=0;k<3;++k)
   cp+=double(first[k])*middle[k]*last[k];
  const auto tensor=tensor_scales();
  const double head_tensor=1.0+0.35*tensor[0]*std::tanh(12.0*cp);
  // The third-order readout also shares the existing forward/inverse
  // standing-wave state. It is not a third recurrent model.
  const double temporal=1.0+0.08*tensor[1]*std::tanh(
      0.5*(s.wave[a]*s.wave[b]+s.inverse_wave[b]*s.inverse_wave[candidate]));
  const double factor=1.25*ab*(0.7*bc+0.3*ac)*holonomy*head_tensor*temporal;
  // Constant-time signed sparse hyperedge residual, exact address order.
  const auto residual=universe_.hyper_query(a,b,candidate,PRECEDES);
  const double exact=2.0*(residual.support-residual.refute)/
                     (1.0+residual.support+residual.refute);
  return clamp(factor+exact,3.0);
 }
 Features features(const State& s,uint8_t target)const{
  Features f{};
  if(!s.length)return f;
  const uint8_t last=s.history[s.length-1];
  f.wave=clamp(0.5*(s.wave[target]-0.5*s.old_wave[target])+
    0.5*(s.inverse_wave[target]-0.5*s.inverse_old_wave[target]),3.0);
  f.attention=attention(s,target);
  for(size_t m=0;m<MODES;++m)
   f.spectral+=eigen_.eigenvalues[m]*eigen_.vectors[m][target]*
               eigen_.vectors[m][last]/(1.0+m);
  if(s.length>=2)
   f.cycle=cycle(s.history[s.length-2],last,target);
  return f;
 }
 // Exact scratch softmax acceptance for head-local four-expert gates.
 // 255x4 independent weights are stored in the existing palindrome head
 // records, not in a second neural model or another memory matrix.
 bool learn_head_gains(const State& s,uint8_t label){
  if(!s.length)return false;
  const auto distribution=predict(s);
  const double old_loss=-std::log(std::max(distribution.p[label],1e-300));
  const auto scales=global_scales();
  std::array<Features,N> field{};
  std::array<GainVector,N> old_weights{},proposal{};
  for(size_t i=0;i<N;++i){
   field[i]=features(s,static_cast<uint8_t>(i));
   old_weights[i]=gains(static_cast<uint8_t>(i));
  }
  for(int retry=0;retry<8;++retry){
   const double rate=std::ldexp(0.03,-retry);
   std::array<double,N> logits{};
   double largest=-std::numeric_limits<double>::infinity();
   for(size_t i=0;i<N;++i){
    const double error=distribution.p[i]-(i==label?1.0:0.0);
    double change=0;
    for(size_t k=0;k<4;++k){
     const double old=old_weights[i][k];
     const double updated=std::clamp(old-rate*error*field[i].at(k)*scales[k],-1.0,1.0);
     proposal[i][k]=static_cast<float>(updated);
     change+=(updated-old)*field[i].at(k)*scales[k];
    }
    logits[i]=std::log(std::max(distribution.p[i],1e-300))+change;
    largest=std::max(largest,logits[i]);
   }
   double total=0;
   for(double logit:logits)total+=std::exp(logit-largest);
   const double candidate_loss=std::log(total)+largest-logits[label];
   if(candidate_loss+1e-12<old_loss){
    for(size_t i=0;i<N;++i)
     write_gains(static_cast<uint8_t>(i),proposal[i]);
    return true;
   }
  }
  return false;
 }
 // Local independent gradients, triad consensus and full-field consensus
 // all vote on one output distribution, and commit only if exact scratch
 // softmax loss improves. No second model or unvalidated live mutation.
 bool learn_long_context_gains(const State& s,uint8_t label){
  if(s.length<=ATTENTION_WINDOW)return false;
  const auto distribution=predict(s);
  const double before=-std::log(std::max(distribution.p[label],1e-300));
  std::array<double,N> signal{},old{};
  for(size_t c=0;c<N;++c){
   signal[c]=long_context_signal(s,static_cast<uint8_t>(c));
   old[c]=read_scalar(universe_.root().h[c],LONG_GAIN_OFFSET);
   if(!std::isfinite(old[c])||std::fabs(old[c])>1.00001)
    throw std::runtime_error("invalid persisted long-context register gain");
  }
  for(int retry=0;retry<8;++retry){
   const double lr=std::ldexp(0.07,-retry);
   std::array<float,N> proposed{};
   std::array<double,N> logit{};
   double maximum=-std::numeric_limits<double>::infinity();
   for(size_t c=0;c<N;++c){
    const double gradient=(distribution.p[c]-(c==label?1.0:0.0))*signal[c];
    proposed[c]=static_cast<float>(std::clamp(old[c]-lr*gradient,-1.0,1.0));
    logit[c]=std::log(std::max(distribution.p[c],1e-300))+
             (double(proposed[c])-old[c])*signal[c];
    maximum=std::max(maximum,logit[c]);
   }
   double sum=0;
   for(double x:logit)sum+=std::exp(x-maximum);
   const double after=std::log(sum)+maximum-logit[label];
   if(after+1e-12<before){
    for(size_t c=0;c<N;++c)
     write_scalar(universe_.root().h[c],LONG_GAIN_OFFSET,proposed[c]);
    return true;
   }
  }
  return false;
 }
 bool learn_hyper_gains(const State& s,uint8_t label){
  if(s.length<2)return false;
  const auto p=predict(s);
  const double before=-std::log(std::max(p.p[label],1e-300));
  std::array<double,N> signals{},grad{},old_local{};
  std::array<double,GROUPS> old_group{},group_grad{};
  double field_grad=0;
  for(size_t c=0;c<N;++c){
   signals[c]=hyperedge_signal(s,static_cast<uint8_t>(c));
   old_local[c]=local_hyper(static_cast<uint8_t>(c));
   grad[c]=(p.p[c]-(c==label?1.0:0.0))*signals[c];
   group_grad[triad_of(static_cast<uint8_t>(c))]+=grad[c];
   field_grad+=grad[c];
  }
  for(size_t group=0;group<GROUPS;++group)
   old_group[group]=group_hyper(group);
  const double old_field=field_hyper();
  for(int retry=0;retry<8;++retry){
   const double rate=std::ldexp(0.08,-retry);
   std::array<float,N> next_local{};
   std::array<float,GROUPS> next_group{};
   for(size_t group=0;group<GROUPS;++group)
    next_group[group]=static_cast<float>(
      std::clamp(old_group[group]-0.4*rate*group_grad[group],-1.0,1.0));
   const float next_field=static_cast<float>(
     std::clamp(old_field-0.1*rate*field_grad,-1.0,1.0));
   double largest=-std::numeric_limits<double>::infinity();
   std::array<double,N> logits{};
   for(size_t c=0;c<N;++c){
    next_local[c]=static_cast<float>(
      std::clamp(old_local[c]-rate*grad[c],-1.0,1.0));
    const double delta=signals[c]*
      (double(next_local[c])-old_local[c]+
       double(next_group[triad_of(static_cast<uint8_t>(c))])-
       old_group[triad_of(static_cast<uint8_t>(c))]+
       double(next_field)-old_field);
    logits[c]=std::log(std::max(p.p[c],1e-300))+delta;
    largest=std::max(largest,logits[c]);
   }
   double total=0;
   for(double logit:logits)total+=std::exp(logit-largest);
   const double after=std::log(total)+largest-logits[label];
   if(after+1e-12<before){
    for(size_t c=0;c<N;++c)
     write_scalar(universe_.root().h[c],HYPER_LOCAL_OFFSET,next_local[c]);
    for(size_t group=0;group<GROUPS;++group)
     write_scalar(universe_.root().h[group_register(group)],
                  HYPER_SHARED_OFFSET,next_group[group]);
    write_scalar(universe_.root().h[0],HYPER_SHARED_OFFSET,next_field);
    return true;
   }
  }
  return false;
 }
 // Sparse residual writes are individually supported by the observed
 // label, with two-of-three constructive votes, bounded residual mass.
 bool learn_hyper_residual(const State& s,uint8_t label,uint64_t when){
  if(s.length<2)return false;
  const uint8_t a=s.history[s.length-2],b=s.history[s.length-1];
  if(a==b||a==label||b==label)return false;
  const double head=local_hyper(label);
  const double group=group_hyper(triad_of(label));
  const double field=field_hyper();
  const unsigned votes=unsigned(head>0)+unsigned(group>0)+unsigned(field>0);
  const double gain=head+group+field;
  if(votes<2||gain<=0)return false;
  const auto old=universe_.hyper_query(a,b,label,PRECEDES);
  if(old.support>=2.0f)return false;
  constexpr float step=0.04f;
  const double prior=2.0*(old.support-old.refute)/
                     (1.0+old.support+old.refute);
  const double updated=2.0*(old.support+step-old.refute)/
                       (1.0+old.support+step+old.refute);
  if(gain*(updated-prior)<=1e-12)return false;
  universe_.add_hyper(a,b,label,PRECEDES,false,step,when);
  return true;
 }

public:
 explicit UnifiedWaveLanguage(Universe& u):
  universe_(u),teacher_(u,0.055,0.08,0.09,0.45,4.0,0.0,true,0.12){
  for(size_t h=0;h<N;++h){
   auto& head=universe_.root().h[h];
   if(head.data[HEAD_GAIN_MAGIC_OFFSET]!=HEAD_GAIN_MAGIC)
    write_gains(static_cast<uint8_t>(h),initial_gains());
   if(head.data[HYPER_LOCAL_MAGIC_OFFSET]!=HYPER_LOCAL_MAGIC){
    write_scalar(head,HYPER_LOCAL_OFFSET,0.14f);
    head.data[HYPER_LOCAL_MAGIC_OFFSET]=HYPER_LOCAL_MAGIC;
   }
   if(h>0&&head.data[TRIAD_MAGIC_OFFSET]!=TRIAD_MAGIC){
    head.data[TRIAD_ID_OFFSET]=static_cast<uint8_t>(h/GROUP_SIZE);
    head.data[TRIAD_MAGIC_OFFSET]=TRIAD_MAGIC;
   }
  }
  std::array<size_t,GROUPS> counts{};
  for(size_t h=0;h<N;++h)++counts[triad_of(static_cast<uint8_t>(h))];
  for(size_t count:counts)
   if(count!=GROUP_SIZE)throw std::runtime_error("invalid triad cardinality");
  for(size_t group=0;group<GROUPS;++group){
   auto& head=universe_.root().h[group_register(group)];
   if(head.data[HYPER_SHARED_MAGIC_OFFSET]!=HYPER_GROUP_MAGIC){
    write_scalar(head,HYPER_SHARED_OFFSET,0.05f);
    head.data[HYPER_SHARED_MAGIC_OFFSET]=HYPER_GROUP_MAGIC;
   }
  }
  auto& field=universe_.root().h[0];
  if(field.data[HYPER_SHARED_MAGIC_OFFSET]!=HYPER_FIELD_MAGIC){
   write_scalar(field,HYPER_SHARED_OFFSET,0.025f);
   field.data[HYPER_SHARED_MAGIC_OFFSET]=HYPER_FIELD_MAGIC;
  }
  load_triad_parameters();
  rebuild_spectrum();
 }
 uint64_t trained()const{return trained_;}
 uint64_t triad_accepted()const{return triad_accepted_;}
 uint64_t projection_accepted()const{return projection_accepted_;}
 void set_triad_training_enabled(bool yes){triad_training_enabled_=yes;}
 TriadWaveFormer::Parameters triad_parameters()const{return triad_parameters_;}
 void set_triad_parameters(const TriadWaveFormer::Parameters& p){
  validate_triad_parameters(p);store_triad_parameters(p);
 }
 void set_bptt_enabled(bool enabled){bptt_enabled_=enabled;}
 bool bptt_enabled()const{return bptt_enabled_;}
 void set_frequency_learning_enabled(bool enabled){frequency_learning_enabled_=enabled;}
 bool frequency_learning_enabled()const{return frequency_learning_enabled_;}
 uint64_t bptt_accepted()const{return bptt_accepted_;}
 uint64_t bptt_attempted()const{return bptt_attempted_;}
 double recurrent_input_gain(uint8_t head)const{return input_gain(head);}
 double recurrent_frequency_shift(uint8_t head)const{return recurrent_frequency(head);}
 void set_recurrent_frequency_shift(uint8_t head,double value){
  if(!std::isfinite(value)||value< -0.15||value>0.15)
   throw std::invalid_argument("recurrent frequency outside [-0.15,0.15]");
  write_frequency(head,static_cast<float>(value));
 }
 void set_recurrent_input_gain(uint8_t head,double value){
  if(!std::isfinite(value)||value<0.6||value>1.4)
   throw std::invalid_argument("recurrent input gain out of range");
  write_input_gain(head,static_cast<float>(value));
 }
 const Spectral& spectral()const{return eigen_;}
 TensorScales tensor_scales()const{
  const auto& h=universe_.root().h[0];
  if(h.data[TENSOR_SCALE_MAGIC_OFFSET]!=TENSOR_SCALE_MAGIC)
   return {1.0,1.0};
  std::array<float,2> raw{};
  std::memcpy(raw.data(),h.data+TENSOR_SCALE_OFFSET,sizeof(raw));
  if(!std::isfinite(raw[0])||!std::isfinite(raw[1])||
     std::fabs(raw[0])>2||std::fabs(raw[1])>2)
   throw std::runtime_error("invalid persisted hypermatrix calibration");
  return {raw[0],raw[1]};
 }
 void set_tensor_scales(const TensorScales& scales){
  for(double v:scales)
   if(!std::isfinite(v)||v< -2||v>2)
    throw std::invalid_argument("hypermatrix scale must be finite in [-2,2]");
  std::array<float,2> raw{
   static_cast<float>(scales[0]),static_cast<float>(scales[1])};
  auto& h=universe_.root().h[0];
  std::memcpy(h.data+TENSOR_SCALE_OFFSET,raw.data(),sizeof(raw));
  h.data[TENSOR_SCALE_MAGIC_OFFSET]=TENSOR_SCALE_MAGIC;
 }
 GlobalScales global_scales()const{
  const auto& bytes=universe_.root().h[0].data;
  if(bytes[CALIBRATION_MAGIC_OFFSET]!=CALIBRATION_MAGIC)
   return {1.0,1.0,1.0,1.0};
  GlobalScales result{};
  for(size_t k=0;k<result.size();++k){
   const unsigned raw=unsigned(bytes[CALIBRATION_OFFSET+2*k])|
                      (unsigned(bytes[CALIBRATION_OFFSET+2*k+1])<<8);
   if(raw>512u)throw std::runtime_error("invalid Q8.8 shared head calibration");
   result[k]=double(raw)/256.0;
  }
  return result;
 }
 void set_global_scales(const GlobalScales& scales){
  for(double s:scales)
   if(!std::isfinite(s)||s<0||s>2)
    throw std::invalid_argument("calibration scale must be finite in [0,2]");
  auto& bytes=universe_.root().h[0].data;
  for(size_t k=0;k<scales.size();++k){
   const auto raw=static_cast<unsigned>(std::lround(scales[k]*256.0));
   bytes[CALIBRATION_OFFSET+2*k]=static_cast<uint8_t>(raw&255u);
   bytes[CALIBRATION_OFFSET+2*k+1]=static_cast<uint8_t>(raw>>8);
  }
  bytes[CALIBRATION_MAGIC_OFFSET]=CALIBRATION_MAGIC;
 }
 size_t active_mirror_arcs()const{
  size_t count=0;
  for(const auto& row:incoming_)count+=row.size();
  return count;
 }
 void refresh_spectrum(){rebuild_spectrum();}
 struct WaveAdjoint {
  std::array<double,N> x{},xp{},y{},yp{};
 };
 // Reverse derivative of legacy wave-attention-hypermatrix loss with respect to
 // transient wave state. Teacher, pair evidence, spectral basis and head
 // geometry are frozen for the truncated time window; wave, attention,
 // and hypermatrix temporal readouts all contribute to the adjoints.
 WaveAdjoint output_wave_adjoint(const State& s,uint8_t label)const{
  WaveAdjoint adj{};
  if(!s.length)return adj;
  const auto distribution=predict(s);
  const auto scales=global_scales();
  const auto tensor=tensor_scales();
  for(size_t c=0;c<N;++c){
   const uint8_t head=static_cast<uint8_t>(c);
   const double error=distribution.p[c]-(c==label?1.0:0.0);
   const auto g=gains(head);
   const double raw_wave=0.5*(s.wave[c]-0.5*s.old_wave[c])+
                         0.5*(s.inverse_wave[c]-0.5*s.inverse_old_wave[c]);
   if(std::fabs(raw_wave)<3.0){
    const double dw=error*(double(g[0])*scales[0]+BPTT_READOUT);
    adj.x[c]+=0.5*dw;adj.xp[c]-=0.25*dw;
    adj.y[c]+=0.5*dw;adj.yp[c]-=0.25*dw;
   }
   // Softmax attention keys do not depend on wave state. Backpropagate
   // through the weighted aggregation with the true clamp derivative.
   const auto q=vector(universe_.root().h[c],48);
   std::array<double,WINDOW> scores{},weights{};
   const size_t start=s.length>ATTENTION_WINDOW?s.length-ATTENTION_WINDOW:0;
   double maximum=-std::numeric_limits<double>::infinity();
   for(size_t k=start;k<s.length;++k){
    const uint8_t src=s.history[k];
    const auto key=vector(universe_.root().h[src],32);
    scores[k]=dot3(q,key)*2.0+
      0.11*std::cos(universe_.covariant_phase(src,head))+
      0.04*std::cos(omega_*double(s.length-k));
    maximum=std::max(maximum,scores[k]);
   }
   double denom=0,aggregate=0;
   for(size_t k=start;k<s.length;++k){
    weights[k]=std::exp(scores[k]-maximum);
    denom+=weights[k];
   }
   for(size_t k=start;k<s.length;++k){
    const uint8_t src=s.history[k];
    weights[k]/=denom;
    const auto forward=universe_.query(src,head,PRECEDES);
    const auto inverse=universe_.query(head,src,FOLLOWS);
    const double evidence=(forward.support+inverse.support)/
                    (2.0+forward.support+inverse.support);
    const double phase=universe_.covariant_phase(src,head);
    aggregate+=weights[k]*(s.wave[src]+0.4*evidence+
                 0.12*std::sin(phase+omega_*(s.length-k)));
   }
   if(std::fabs(aggregate)<3.0){
    const double da=error*double(g[1])*scales[1];
    for(size_t k=start;k<s.length;++k)
     adj.x[s.history[k]]+=da*weights[k];
   }
   // Factorized third-order temporal readout: differentiate its exact
   // tanh product with respect to forward and inverse wave coordinates.
   if(s.length>=2){
    const uint8_t a=s.history[s.length-2],b=s.history[s.length-1];
    if(a!=b&&a!=head&&b!=head){
     const double value=hyperedge_signal(s,head);
     if(std::fabs(value)<3.0){
      const auto residual=universe_.hyper_query(a,b,head,PRECEDES);
      const double exact=2.0*(residual.support-residual.refute)/
                    (1.0+residual.support+residual.refute);
      const double u=0.5*(s.wave[a]*s.wave[b]+
                      s.inverse_wave[b]*s.inverse_wave[head]);
      const double activation=std::tanh(u);
      const double temporal=1.0+0.08*tensor[1]*activation;
      const double base=(value-exact)/temporal;
      const double d=error*hyper_gain(head)*base*0.08*tensor[1]*
                    (1.0-activation*activation)*0.5;
      adj.x[a]+=d*s.wave[b];
      adj.x[b]+=d*s.wave[a];
      adj.y[b]+=d*s.inverse_wave[head];
      adj.y[c]+=d*s.inverse_wave[b];
     }
    }
   }
  }
  return adj;
 }
 // Reverse-mode original wave readouts through both SECOND-ORDER recurrences,
 // including T and T* scattering, old-wave links and clamp masks.
 // Detach only the oldest state (explicit 8-transition truncation).
 double bptt_window_loss(const std::deque<State>& states,
                         const std::deque<uint8_t>& symbols,
                         uint8_t label,bool sequence=true)const{
  if(label>=N||symbols.empty()||symbols.size()>BPTT_STEPS||
     states.size()!=symbols.size()+1)
   throw std::invalid_argument("invalid BPTT sequence objective");
  const auto last=predict(states.back());
  double objective=-std::log(std::max(last.p[label],1e-300));
  size_t count=1;
  if(sequence)for(size_t k=0;k<symbols.size();++k){
   if(states[k].length==0)continue;
   const auto p=predict(states[k]);
   objective-=std::log(std::max(p.p[symbols[k]],1e-300));
   ++count;
  }
  return objective/double(count);
 }
 BpttGradient bptt_gradient(const std::deque<State>& states,
                           const std::deque<uint8_t>& symbols,
                           uint8_t label,bool sequence=false){
  if(label>=N||symbols.empty()||symbols.size()>BPTT_STEPS||
     states.size()!=symbols.size()+1)
   throw std::invalid_argument("invalid truncated wave tape");
  BpttGradient result{};
  result.steps=symbols.size();
  result.loss=bptt_window_loss(states,symbols,label,sequence);
  WaveAdjoint adj=output_wave_adjoint(states.back(),label);
  std::array<double,N> oscillator{},omega_slope{},in_degree{},out_degree{};
  for(size_t i=0;i<N;++i){
   double modal=0;
   for(size_t m=0;m<MODES;++m)
    modal+=eigen_.eigenvalues[m]*eigen_.vectors[m][i]*eigen_.vectors[m][i];
   const double theta=omega_+0.1*std::tanh(modal)+
                  recurrent_frequency(static_cast<uint8_t>(i));
   oscillator[i]=2.0*damping_*std::cos(theta);
   omega_slope[i]=-2.0*damping_*std::sin(theta);
   in_degree[i]=in_degree_[i];
   out_degree[i]=out_degree_[i];
  }
  for(size_t step=symbols.size();step>0;--step){
   const auto& after=states[step];
   const auto& before=states[step-1];
   const uint8_t byte=symbols[step-1];
   WaveAdjoint prev{};
   for(size_t i=0;i<N;++i){
    // after.old_wave = before.wave, and inverse analogously
    prev.x[i]+=adj.xp[i];prev.y[i]+=adj.yp[i];
    const double dx=std::fabs(after.wave[i])<3.0?adj.x[i]:0.0;
    const double dy=std::fabs(after.inverse_wave[i])<3.0?adj.y[i]:0.0;
    prev.x[i]+=oscillator[i]*dx;
    prev.y[i]+=oscillator[i]*dy;
    prev.xp[i]-=damping_*damping_*dx;
    prev.yp[i]-=damping_*damping_*dy;
    if(i==byte)result.input[i]+=dx+dy;
    result.frequency[i]+=omega_slope[i]*(dx*before.wave[i]+
                                    dy*before.inverse_wave[i]);
    for(const auto& arc:incoming_[i])
     prev.x[arc.head]+=0.08*arc.weight*dx/in_degree[i];
    for(const auto& arc:outgoing_[i])
     prev.y[arc.head]+=0.08*arc.weight*dy/out_degree[i];
   }
   // Every earlier next-byte target contributes its OWN causal
   // cross-entropy gradient before that state's adjoint is propagated
   // through earlier temporal transitions.
   if(sequence&&states[step-1].length){
    const auto early=output_wave_adjoint(states[step-1],symbols[step-1]);
    for(size_t i=0;i<N;++i){
     prev.x[i]+=early.x[i];prev.xp[i]+=early.xp[i];
     prev.y[i]+=early.y[i];prev.yp[i]+=early.yp[i];
    }
   }
   adj=prev;
  }
  if(sequence){
   size_t targets=1;
   for(size_t k=0;k<symbols.size();++k)
    if(states[k].length)++targets;
   for(double& value:result.input)value/=double(targets);
   for(double& value:result.frequency)value/=double(targets);
  }
  // The attention/field readout contributes wave derivatives beyond
  // the older T/T* analytic adjoint. Correct the coordinates that
  // receive directly observed symbols using full-loss central differences.
  // NOTE: full reverse-mode through QKV, routing and the field is not yet
  // implemented; this is a checked numerical reference for active inputs.
  std::array<bool,N> touched{};
  for(uint8_t b:symbols)touched[b]=true;
  const auto exact_loss=[&](){
   std::deque<State> replay{states.front()};
   for(uint8_t b:symbols){
    auto next=replay.back();
    observe(next,b);
    replay.push_back(std::move(next));
   }
   return bptt_window_loss(replay,symbols,label,sequence);
  };
  for(size_t i=0;i<N;++i){
   if(!touched[i]){result.input[i]=0;continue;}
   const uint8_t head=static_cast<uint8_t>(i);
   auto& h=universe_.root().h[i];
   std::array<uint8_t,10> old_bytes{};
   std::memcpy(old_bytes.data(),h.data+BPTT_INPUT_OFFSET,8);
   old_bytes[8]=h.data[BPTT_MAGIC_OFFSET];
   old_bytes[9]=h.data[BPTT_FREQUENCY_MAGIC_OFFSET];
   const auto restore=[&](){
    std::memcpy(h.data+BPTT_INPUT_OFFSET,old_bytes.data(),8);
    h.data[BPTT_MAGIC_OFFSET]=old_bytes[8];
    h.data[BPTT_FREQUENCY_MAGIC_OFFSET]=old_bytes[9];
   };
   const double gain=input_gain(head),eps=0.002;
   write_input_gain(head,float(gain+eps));
   const double up=exact_loss();
   write_input_gain(head,float(gain-eps));
   const double down=exact_loss();
   restore();
   result.input[i]=(up-down)/(2*eps);
   const double freq=recurrent_frequency(head),delta=0.0005;
   write_frequency(head,float(freq+delta));
   const double freq_up=exact_loss();
   write_frequency(head,float(freq-delta));
   const double freq_down=exact_loss();
   restore();
   result.frequency[i]=(freq_up-freq_down)/(2*delta);
  }
  for(double v:result.input)result.norm+=v*v;
  if(frequency_learning_enabled_)
   for(double v:result.frequency)result.norm+=v*v;
  result.norm=std::sqrt(result.norm);
  if(!std::isfinite(result.norm))
   throw std::runtime_error("nonfinite truncated wave gradient");
  return result;
 }
 // Transactional gradient step: replay the exact causal tape under
 // proposed native register gains and accept only if FULL softmax CE
 // improves on the target (not a surrogate wave-only objective).
 bool learn_bptt(State& state,std::deque<State>& states,
                 const std::deque<uint8_t>& symbols,uint8_t label){
  ++bptt_attempted_;
  const auto grad=bptt_gradient(states,symbols,label,true);
  if(grad.norm<1e-12)return false;
  std::array<float,N> old{},old_frequency{};
  std::array<std::array<uint8_t,10>,N> original{};
  for(size_t i=0;i<N;++i){
   old[i]=static_cast<float>(input_gain(static_cast<uint8_t>(i)));
   old_frequency[i]=static_cast<float>(recurrent_frequency(static_cast<uint8_t>(i)));
   const auto& head=universe_.root().h[i];
   std::memcpy(original[i].data(),head.data+BPTT_INPUT_OFFSET,8);
   original[i][8]=head.data[BPTT_MAGIC_OFFSET];
   original[i][9]=head.data[BPTT_FREQUENCY_MAGIC_OFFSET];
  }
  const auto restore=[&](){
   for(size_t i=0;i<N;++i){
    auto& head=universe_.root().h[i];
    std::memcpy(head.data+BPTT_INPUT_OFFSET,original[i].data(),8);
    head.data[BPTT_MAGIC_OFFSET]=original[i][8];
    head.data[BPTT_FREQUENCY_MAGIC_OFFSET]=original[i][9];
   }
  };
  const double clipped=std::max(1.0,grad.norm);
  for(int retry=0;retry<8;++retry){
   const double rate=std::ldexp(0.08,-retry)/clipped;
   for(size_t i=0;i<N;++i){
    if(std::fabs(grad.input[i])>1e-12)
     write_input_gain(static_cast<uint8_t>(i),static_cast<float>(
        std::clamp(old[i]-rate*grad.input[i],0.6,1.4)));
    if(frequency_learning_enabled_&&std::fabs(grad.frequency[i])>1e-12)
     write_frequency(static_cast<uint8_t>(i),static_cast<float>(
        std::clamp(old_frequency[i]-rate*grad.frequency[i],-0.15,0.15)));
   }
   std::deque<State> replay{states.front()};
   for(uint8_t byte:symbols){
    State next=replay.back();
    observe(next,byte);
    replay.push_back(std::move(next));
   }
   const double loss=bptt_window_loss(replay,symbols,label,true);
   if(loss+1e-10<grad.loss){
    states=std::move(replay);
    state=states.back();
    ++bptt_accepted_;
    return true;
   }
   restore();
  }
  return false;
 }
 size_t head_triad(uint8_t head)const{return triad_of(head);}
 bool trial_swap_triads(uint8_t a,uint8_t b,const State& s,uint8_t label){
  if(a==0||b==0||a>=N||b>=N||a==b||label>=N||s.length==0)return false;
  const size_t ga=triad_of(a),gb=triad_of(b);
  if(ga==gb)return false;
  auto& ha=universe_.root().h[a].data[TRIAD_ID_OFFSET];
  auto& hb=universe_.root().h[b].data[TRIAD_ID_OFFSET];
  // Topology mutation requires replay of the causal recurrent field.
  // Testing a remapped register against a stale cached field is not an
  // exact-loss trial and can fail determinism after reboot.
  const auto score=[&](){
   State replay{};
   for(size_t i=0;i<s.length;++i)observe(replay,s.history[i]);
   return predict(replay).p[label];
  };
  const double before=score();
  ha=static_cast<uint8_t>(gb);hb=static_cast<uint8_t>(ga);
  const double after=score();
  if(std::isfinite(after)&&after>before+1e-10)return true;
  ha=static_cast<uint8_t>(ga);hb=static_cast<uint8_t>(gb);
  return false;
 }
 void refresh_triad(State& s,const TriadWaveFormer::Workspace& previous)const{
  auto embed=[this](uint8_t h){
   std::array<double,TriadWaveFormer::DIM> v{};
   const auto a=vector(universe_.root().h[h],32);
   const auto b=vector(universe_.root().h[h],48);
   for(size_t d=0;d<3;++d){v[d]=a[d];v[d+3]=b[d];}
   return v;
  };
  auto evidence=[this](uint8_t a,uint8_t b){
   if(a==b)return 0.0;
   const auto e=universe_.query(a,b,PRECEDES);
   return (double(e.support)-e.refute)/(1.0+e.support+e.refute);
  };
  auto phase=[this](uint8_t a,uint8_t b){
   return a==b?0.0:universe_.covariant_phase(a,b);
  };
  TriadWaveFormer::advance(s.workspace,previous,s.history.data(),s.length,
    s.wave,s.inverse_wave,eigen_.vectors,triad_parameters_,
    [this](uint8_t h){return triad_of(h);},embed,evidence,phase);
 }
 void observe(State& s,uint8_t c)const{
  if(c==255){s.reset();return;}
  std::array<double,N> next{};
  const double decay=damping_*damping_;
  // All 255 heads participate in each temporal transition. Opposing
  // mirror directions contribute through the symmetric scattering
  // operator while the second-order recurrence preserves wave history.
  std::array<double,N> next_inverse{};
  for(size_t i=0;i<N;++i){
   double forward_in=0,inverse_in=0;
   for(const auto& arc:incoming_[i])
    forward_in+=arc.weight*s.wave[arc.head];
   for(const auto& arc:outgoing_[i])
    inverse_in+=arc.weight*s.inverse_wave[arc.head];
   const double incoming_degree=in_degree_[i],outgoing_degree=out_degree_[i];
   double modulation=0,modal_frequency=0;
   for(size_t m=0;m<MODES;++m){
    modulation+=eigen_.vectors[m][i]*eigen_.vectors[m][c]/(1.0+m);
    modal_frequency+=eigen_.eigenvalues[m]*
      eigen_.vectors[m][i]*eigen_.vectors[m][i];
   }
   const double local_omega=omega_+0.1*std::tanh(modal_frequency)+
        recurrent_frequency(static_cast<uint8_t>(i));
   const double local_oscillation=2.0*damping_*std::cos(local_omega);
   const double input=(i==c?input_gain(c):0.0);
   next[i]=clamp(local_oscillation*s.wave[i]-decay*s.old_wave[i]+
        0.08*forward_in/incoming_degree+0.05*modulation+input,3.0);
   next_inverse[i]=clamp(local_oscillation*s.inverse_wave[i]-
        decay*s.inverse_old_wave[i]+0.08*inverse_in/outgoing_degree+
        0.05*modulation+input,3.0);
  }
  s.old_wave=s.wave;
  s.inverse_old_wave=s.inverse_wave;
  s.wave=next;
  s.inverse_wave=next_inverse;
  const auto previous=s.workspace;
  s.push(c);
  refresh_triad(s,previous);
  ++s.tick;
 }
 Prediction predict(const State& s,unsigned disabled=0)const{
  if(disabled&~ABLATE_ALL)throw std::invalid_argument("unknown ablation bits");
  Prediction out{};
  if(s.length==0){
   out.p.fill(1.0/N);
   return out;
  }
  const uint8_t last=s.history[s.length-1];
  const auto baseline=teacher_.predict(last);
  const auto scales=global_scales();
  std::array<double,N> logits{};
  double largest=-std::numeric_limits<double>::infinity();
  for(size_t c=0;c<N;++c){
   const uint8_t target=static_cast<uint8_t>(c);
   if(disabled==ABLATE_ALL){
    logits[c]=baseline.logit[c];
    largest=std::max(largest,logits[c]);
    continue;
   }
   const auto f=features(s,target);
   const auto g=gains(target);
   const auto forward=universe_.query(last,target,PRECEDES);
   const auto inverse=universe_.query(target,last,FOLLOWS);
   const double inverse_consensus=
      (forward.support+inverse.support-forward.refute-inverse.refute)/
      (2.0+forward.support+inverse.support+forward.refute+inverse.refute);
   logits[c]=baseline.logit[c]+
      ((disabled&ABLATE_WAVE)?0.0:double(g[0])*scales[0]*f.wave)+
      ((disabled&ABLATE_BPTT_READOUT)?0.0:BPTT_READOUT*f.wave)+
      ((disabled&ABLATE_ATTENTION)?0.0:double(g[1])*scales[1]*f.attention)+
      ((disabled&ABLATE_SPECTRAL)?0.0:double(g[2])*scales[2]*f.spectral)+
      ((disabled&ABLATE_CYCLE)?0.0:double(g[3])*scales[3]*f.cycle)+
      ((disabled&ABLATE_INVERSE_EVIDENCE)?0.0:evidence_gain_*inverse_consensus)+
      ((disabled&ABLATE_HYPEREDGE)?0.0:hyper_gain(target)*hyperedge_signal(s,target))+
      ((disabled&ABLATE_LONG_CONTEXT)?0.0:
        read_scalar(universe_.root().h[target],LONG_GAIN_OFFSET)*
        long_context_signal(s,target));
   if(!(disabled&ABLATE_TRIAD)){
    const size_t group=triad_of(target);
    const double f=(disabled&ABLATE_FIELD)?0.0:
      0.28*(s.workspace.field[(c*7)%TriadWaveFormer::FIELD]-
            s.workspace.field[(c*7+17)%TriadWaveFormer::FIELD])+
      0.20*s.workspace.routing[group];
    logits[c]+=triad_parameters_.head_gain[c]*
        clamp(s.workspace.triad[group]+f,3.0);
   }
   largest=std::max(largest,logits[c]);
  }
  double z=0;
  for(size_t c=0;c<N;++c){
   out.p[c]=std::exp(logits[c]-largest);
   z+=out.p[c];
  }
  for(size_t c=0;c<N;++c){
   out.p[c]/=z;
   if(out.p[c]>out.p[out.best])out.best=static_cast<uint8_t>(c);
  }
  return out;
 }
 // True full-softmax gradient for all 255 shared-triad output gains.
 // The recurrent field and attention kernels remain fixed for this step.
 bool learn_triad_readout(const State& s,uint8_t label){
  if(!s.length)return false;
  const auto distribution=predict(s);
  const double before=-std::log(std::max(distribution.p[label],1e-300));
  std::array<double,N> feature{};
  for(size_t c=0;c<N;++c){
   const size_t g=triad_of(static_cast<uint8_t>(c));
   const double f=0.28*(s.workspace.field[(c*7)%TriadWaveFormer::FIELD]-
                s.workspace.field[(c*7+17)%TriadWaveFormer::FIELD])+
                0.2*s.workspace.routing[g];
   feature[c]=clamp(s.workspace.triad[g]+f,3.0);
  }
  for(int retry=0;retry<7;++retry){
   const double rate=std::ldexp(0.12,-retry);
   auto next=triad_parameters_;
   std::array<double,N> logits{};
   double max_logit=-std::numeric_limits<double>::infinity();
   for(size_t c=0;c<N;++c){
    const double error=distribution.p[c]-(c==label?1.0:0.0);
    next.head_gain[c]=float(std::clamp(double(triad_parameters_.head_gain[c])-
                       rate*error*feature[c],-1.0,1.0));
    logits[c]=std::log(std::max(distribution.p[c],1e-300))+
          (double(next.head_gain[c])-triad_parameters_.head_gain[c])*feature[c];
    max_logit=std::max(max_logit,logits[c]);
   }
   double normalizer=0;
   for(double v:logits)normalizer+=std::exp(v-max_logit);
   const double after=std::log(normalizer)+max_logit-logits[label];
   if(after+1e-12<before){
    store_triad_parameters(next);++triad_accepted_;return true;
   }
  }
  return false;
 }
 Metrics evaluate(std::istream& corpus,unsigned disabled=0)const{
  if(disabled&~ABLATE_ALL)throw std::invalid_argument("unknown ablation bits");
  State state{};
  Metrics result{};
  char ch;
  while(corpus.get(ch)){
   const uint8_t symbol=static_cast<uint8_t>(static_cast<unsigned char>(ch));
   if(symbol==255){state.reset();continue;}
   if(state.length){
    const auto p=predict(state,disabled);
    result.nll-=std::log(std::max(p.p[symbol],1e-300));
    result.correct+=(p.best==symbol?1u:0u);
    ++result.tokens;
   }
   observe(state,symbol);
  }
  return result;
 }
 Metrics train(std::istream& corpus,uint64_t initial_time=1){
  State state{};
  std::deque<State> tape_states{state};
  std::deque<uint8_t> tape_inputs{};
  Metrics result{};
  char ch;
  while(corpus.get(ch)){
   const uint8_t symbol=static_cast<uint8_t>(static_cast<unsigned char>(ch));
   if(symbol==255){
    state.reset();tape_inputs.clear();tape_states.clear();tape_states.push_back(state);
    continue;
   }
   if(state.length){
    const auto p=predict(state);
    result.nll-=std::log(std::max(p.p[symbol],1e-300));
    result.correct+=(p.best==symbol?1u:0u);
    ++result.tokens;
    const uint8_t prev=state.history[state.length-1];
    // All persistent training occurs INSIDE the one mapped root and its
    // already-native temporal/pair/phase/quaternion overlays.
    if(bptt_enabled_&&trained_%BPTT_INTERVAL==0&&tape_inputs.size()>=4)
     learn_bptt(state,tape_states,tape_inputs,symbol);
    teacher_.teach(prev,symbol,initial_time+trained_);
    // One teacher and one higher-order field share all 255 registers.
    learn_head_gains(state,symbol);
    learn_hyper_gains(state,symbol);
    learn_long_context_gains(state,symbol);
    if(triad_training_enabled_)learn_triad_readout(state,symbol);
    learn_hyper_residual(state,symbol,initial_time+trained_);
    if(trained_%128==0)
     trial_swap_triads(prev,symbol,state,symbol);
   }
   observe(state,symbol);
   tape_inputs.push_back(symbol);
   tape_states.push_back(state);
   if(tape_inputs.size()>BPTT_STEPS){
    tape_inputs.pop_front();tape_states.pop_front();
   }
   ++trained_;
   // The graph operator is frozen within each BPTT tape.
   if((trained_%128)==0){
    rebuild_spectrum();tape_inputs.clear();
    tape_states.clear();tape_states.push_back(state);
   }
  }
  rebuild_spectrum();
  return result;
 }
 std::string generate(const std::string& prefix,size_t count,uint64_t seed,
                      double temperature=0.85,double top_p=0.92)const{
  if(count>1000000||temperature<=0||temperature>5||!std::isfinite(temperature)||
      top_p<=0||top_p>1||!std::isfinite(top_p))
   throw std::invalid_argument("invalid generation settings");
  Utf8Prefix utf8;
  if(!utf8.consume_prefix(prefix)||count<utf8.remaining)
   throw std::invalid_argument("prompt is not a valid UTF-8 prefix");
  State state{};
  for(unsigned char c:prefix)observe(state,c);
  std::string result;
  result.reserve(count);
  uint64_t random=seed;
  for(size_t step=0;step<count;++step){
   const auto p=predict(state);
   std::vector<std::pair<double,uint8_t>> weighted;
   weighted.reserve(N);
   for(size_t c=0;c<N;++c){
    const auto b=static_cast<uint8_t>(c);
    if(utf8.permits(b,count-step))
     weighted.emplace_back(std::pow(p.p[c],1.0/temperature),b);
   }
   if(weighted.empty())throw std::runtime_error("no Unicode continuation");
   std::sort(weighted.begin(),weighted.end(),[](auto a,auto b){
    return a.first==b.first?a.second<b.second:a.first>b.first;
   });
   double total=0;
   for(const auto& item:weighted)total+=item.first;
   size_t take=0;double mass=0;
   while(take<weighted.size()&&(take==0||mass/total<top_p))
    mass+=weighted[take++].first;
   const double sample=random01(random)*mass;
   double cumul=0;uint8_t chosen=weighted[take-1].second;
   for(size_t i=0;i<take;++i){
    cumul+=weighted[i].first;
    if(sample<cumul){chosen=weighted[i].second;break;}
   }
   if(!utf8.push(chosen,count-step))
    throw std::runtime_error("invalid Unicode continuation");
   result.push_back(static_cast<char>(chosen));
   observe(state,chosen);
  }
  if(!utf8.complete())throw std::runtime_error("truncated generated Unicode");
  return result;
 }
};
} // namespace nanlm
