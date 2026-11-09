#pragma once
#include "nanlm/teacher.hpp"
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
 static constexpr size_t WINDOW=16;
 static constexpr size_t MODES=4;
 // Bitmask for read-only, single-root counterfactual evaluations.
 static constexpr unsigned ABLATE_WAVE=1u<<0;
 static constexpr unsigned ABLATE_ATTENTION=1u<<1;
 static constexpr unsigned ABLATE_SPECTRAL=1u<<2;
 static constexpr unsigned ABLATE_CYCLE=1u<<3;
 static constexpr unsigned ABLATE_INVERSE_EVIDENCE=1u<<4;
 static constexpr unsigned ABLATE_HYPEREDGE=1u<<5;
 static constexpr unsigned ABLATE_ALL=(1u<<6)-1u;
 static constexpr size_t HEAD_GAIN_OFFSET=80;
 static constexpr size_t HEAD_GAIN_MAGIC_OFFSET=60;
 static constexpr uint8_t HEAD_GAIN_MAGIC=0xC7;
 struct State {
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
private:
 Universe& universe_;
 FabricTeacher teacher_;
 double evidence_gain_=0.08;
 double damping_=0.80;
 double omega_=0.53;
 std::array<std::array<double,N>,N> adjacency_{};
 std::array<std::array<double,N>,N> directed_{};
 Spectral eigen_{};
 uint64_t trained_=0;
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
  for(size_t k=0;k<s.length;++k){
   const uint8_t src=s.history[k];
   const auto key=vector(universe_.root().h[src],32);
   const double geometry=0.11*std::cos(universe_.covariant_phase(src,candidate));
   const double temporal=0.04*std::cos(omega_*double(s.length-k));
   scores[k]=dot3(q,key)*2.0+geometry+temporal;
   best=std::max(best,scores[k]);
  }
  double denominator=0,aggregation=0;
  for(size_t k=0;k<s.length;++k)denominator+=std::exp(scores[k]-best);
  for(size_t k=0;k<s.length;++k){
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
 double hyperedge_signal(const State& s,uint8_t candidate)const{
  if(s.length<2)return 0;
  const uint8_t a=s.history[s.length-2],b=s.history[s.length-1];
  if(a==b||a==candidate||b==candidate)return 0;
  // Query the existing sparse native hyperedge store: exact typed triples,
  // not an additional trainable tensor or lookup-based language model.
  const auto links=universe_.routes(candidate);
  double value=0;
  for(const auto& edge:links){
   if(edge.a==a&&edge.b==b&&edge.c==candidate&&edge.relation==PRECEDES)
    value+=(edge.refutes?-1.0:1.0)*edge.strength;
  }
  return clamp(value,3.0);
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
     const double updated=std::clamp(old-rate*error*field[i].at(k),-1.0,1.0);
     proposal[i][k]=static_cast<float>(updated);
     change+=(updated-old)*field[i].at(k);
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

public:
 explicit UnifiedWaveLanguage(Universe& u):
  universe_(u),teacher_(u,0.055,0.08,0.09,0.45,4.0,0.0,true,0.12){
  for(size_t h=0;h<N;++h){
   if(universe_.root().h[h].data[HEAD_GAIN_MAGIC_OFFSET]!=HEAD_GAIN_MAGIC)
    write_gains(static_cast<uint8_t>(h),initial_gains());
  }
  rebuild_spectrum();
 }
 uint64_t trained()const{return trained_;}
 const Spectral& spectral()const{return eigen_;}
 void refresh_spectrum(){rebuild_spectrum();}
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
   double incoming_degree=1,outgoing_degree=1;
   for(size_t j=0;j<N;++j){
    // Temporal forward operator T and its exact transpose T* operate
    // on the same physical mirror-pair fabric, not duplicated weights.
    forward_in+=directed_[j][i]*s.wave[j];
    inverse_in+=directed_[i][j]*s.inverse_wave[j];
    incoming_degree+=std::fabs(directed_[j][i]);
    outgoing_degree+=std::fabs(directed_[i][j]);
   }
   double modulation=0,modal_frequency=0;
   for(size_t m=0;m<MODES;++m){
    modulation+=eigen_.vectors[m][i]*eigen_.vectors[m][c]/(1.0+m);
    modal_frequency+=eigen_.eigenvalues[m]*
      eigen_.vectors[m][i]*eigen_.vectors[m][i];
   }
   const double local_omega=omega_+0.1*std::tanh(modal_frequency);
   const double local_oscillation=2.0*damping_*std::cos(local_omega);
   const double input=(i==c?1.0:0.0);
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
  s.push(c);
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
      ((disabled&ABLATE_WAVE)?0.0:double(g[0])*f.wave)+
      ((disabled&ABLATE_ATTENTION)?0.0:double(g[1])*f.attention)+
      ((disabled&ABLATE_SPECTRAL)?0.0:double(g[2])*f.spectral)+
      ((disabled&ABLATE_CYCLE)?0.0:double(g[3])*f.cycle)+
      ((disabled&ABLATE_INVERSE_EVIDENCE)?0.0:evidence_gain_*inverse_consensus)+
      ((disabled&ABLATE_HYPEREDGE)?0.0:0.03*hyperedge_signal(s,target));
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
  Metrics result{};
  char ch;
  while(corpus.get(ch)){
   const uint8_t symbol=static_cast<uint8_t>(static_cast<unsigned char>(ch));
   if(symbol==255){state.reset();continue;}
   if(state.length){
    const auto p=predict(state);
    result.nll-=std::log(std::max(p.p[symbol],1e-300));
    result.correct+=(p.best==symbol?1u:0u);
    ++result.tokens;
    const uint8_t prev=state.history[state.length-1];
    // All persistent training occurs INSIDE the one mapped root and its
    // already-native temporal/pair/phase/quaternion overlays.
    const auto accepted=teacher_.teach(prev,symbol,initial_time+trained_);
    // Prediction gradients now update the four expert gates held directly
    // in each of the 255 palindromic head registers.
    learn_head_gains(state,symbol);
    if(accepted.accepted&&state.length>=2&&(trained_%8==0)){
     const uint8_t first=state.history[state.length-2];
     if(first!=prev&&first!=symbol&&prev!=symbol)
      universe_.add_hyper(first,prev,symbol,PRECEDES,false,
                          0.02f,initial_time+trained_);
    }
   }
   observe(state,symbol);
   ++trained_;
   // Recalculate spectral graph after significant new edge evidence.
   if((trained_%128)==0)rebuild_spectrum();
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
