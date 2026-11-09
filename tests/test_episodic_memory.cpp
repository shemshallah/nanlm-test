#include "nanlm/unified_wave.hpp"
#include <cmath>
#include <chrono>
#include <iomanip>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>

using namespace nanlm;
static void require(bool good,const char* reason){
 if(!good)throw std::runtime_error(reason);
}
int main(){
 const std::string root=(std::filesystem::temp_directory_path()/
  ("nanlm-exact-memory-"+std::to_string(::getpid()))).string();
 const std::string cue="Beacon archive code: ";
 // Longer than 64 bytes: continuation MUST NOT depend on the cue
 // remaining in the 64-byte attention window.
 const std::string value="JUPITER-42-"+
   std::string(196,'Q')+"-THIRD-NEBULA";
 const std::string long_gap(1200,'x');
 const std::string source="trusted-test-fixture-v1";
 auto cleanup=[&](){
  for(const char* suffix:{"",".episodes",".episodes.tmp",
                                 ".episodes.lock",".so3",".overlay",".overlay.tmp"}){
   std::error_code ec;std::filesystem::remove(root+suffix,ec);
  }
 };
 try{
  double persisted=0;
  std::array<uint8_t,128> ingress{};
  {
   Universe u(root);
   std::memcpy(ingress.data(),u.root().h[255].data,128);
   UnifiedWaveLanguage mind(u);
   require(mind.episodic_records()==0,"blank checkpoint has records");
   // Old writers must not silently overwrite newer checkpoint content,
   // even when both instances opened before the first insert.
   {
    const std::string leasepath=root+"-lease.episodes";
    ExactEpisodicMemory writer_a(leasepath),writer_b(leasepath);
    writer_a.remember("Primary record: ","FIRST","writer-a");
    bool stale_rejected=false;
    try{
     writer_b.remember("Secondary record: ","SECOND","writer-b");
    }catch(const std::runtime_error&){
     stale_rejected=true;
    }
    require(stale_rejected,
            "stale writer overwrote another source checkpoint");
    ExactEpisodicMemory authoritative(leasepath);
    require(authoritative.exact("Primary record: ")!=nullptr&&
            authoritative.exact("Secondary record: ")==nullptr,
            "stale update changed authoritative source records");
    std::error_code ignored;
    std::filesystem::remove(leasepath,ignored);
    std::filesystem::remove(leasepath+".tmp",ignored);
    std::filesystem::remove(leasepath+".lock",ignored);
   }

   mind.memorize(cue,value,source);
   require(mind.episodic_records()==1,"record not inserted");
   const auto* record=mind.exact_recall(cue);
   require(record&&record->value==value&&record->source==source,
           "exact source attribution lost");
   // Adversarial cardinality probe: identical-length distractor cues,
   // with the exact answer deliberately inserted LAST. Phase routing may
   // lower equality comparisons, but its CPU cost is measured separately.
   {
    const std::string probe_path=root+"-routing.episodes";
    ExactEpisodicMemory many(probe_path);
    std::ostringstream distractors;
    for(unsigned i=0;i<180;++i){
     distractors<<"Beacon archive X"<<std::setfill('0')<<std::setw(4)
                <<i<<" \tDECOY-"<<i<<"\n";
    }
    std::istringstream source(distractors.str());
    require(many.import_tsv(source,"routing-control")==180,
            "distractor fixture did not import");
    many.remember(cue,value,"late-source");
    std::vector<uint8_t> exact_history(cue.begin(),cue.end());
    const auto no_rank=many.probe(exact_history.data(),
                                  exact_history.size(),false);
    const auto ranked=many.probe(exact_history.data(),
                                exact_history.size(),true);
    require(no_rank.record==ranked.record&&
            ranked.record!=ExactEpisodicMemory::NONE,
            "resonance changed byte-verified result under distractors");
    require(no_rank.exact_compares==1&&
            ranked.exact_compares>=1,
            "hashed exact lookup regressed or resonance skipped verification");
    const std::string near="Beacon archive coda: ";
    std::vector<uint8_t> wrong(near.begin(),near.end());
    require(many.probe(wrong.data(),wrong.size(),true).record==
            ExactEpisodicMemory::NONE,
            "high-scoring near-miss bypassed exact verification");
    // Time the routing algorithm itself, not the entire language engine.
    size_t verification_sum=0;
    const auto measure=[&](bool resonate){
     constexpr unsigned TRIALS=50;
     const auto start=std::chrono::steady_clock::now();
     for(unsigned j=0;j<TRIALS;++j){
      const auto q=many.probe(exact_history.data(),
                              exact_history.size(),resonate);
      verification_sum+=q.exact_compares;
     }
     const auto elapsed=std::chrono::duration<double,std::micro>(
        std::chrono::steady_clock::now()-start).count();
     return elapsed/double(TRIALS);
    };
    const double unranked_us=measure(false);
    const double ranked_us=measure(true);
    require(verification_sum>0&&std::isfinite(unranked_us)&&
            std::isfinite(ranked_us),
            "invalid routing timing");
    std::cout<<"EPISODIC_DISTRACTOR_AUDIT candidates="<<ranked.routed
             <<" hash_exact_compares="<<no_rank.exact_compares
             <<" resonance_exact_compares="<<ranked.exact_compares
             <<" hash_us="<<unranked_us
             <<" resonance_us="<<ranked_us<<"\n";
    // Dynamic 85-group map and 8 hottest triads: noisy/recovered
    // cue is a SUGGESTION ONLY; it never activates the copy cursor.
    TriadCueRouter::Mapping groups{};
    for(size_t h=0;h<groups.size();++h)
     groups[h]=static_cast<uint8_t>(h/3);
    const auto fuzzy=[&](const std::string& q,size_t edits=2){
     std::vector<uint8_t> x(q.begin(),q.end());
     return many.suggest_robust(x.data(),x.size(),5,edits,groups);
    };
    for(const auto& noisy:{"Beacon archive coda: ",
                           "Beacon archive cod: ",
                           "Beacon archive coode: "}){
     const auto results=fuzzy(noisy);
     require(!results.empty()&&results[0].cue==cue&&
             results[0].edit_distance==1&&
             !results[0].exact_verified&&
             results[0].hot_triads<=8,
             "one-byte typo failed 85-triad approximate suggestion");
    }
    const auto intact=fuzzy(cue);
    require(!intact.empty()&&intact[0].exact_verified&&
            intact[0].edit_distance==0&&
            intact[0].hot_triads==TriadCueRouter::HOT,
            "exact triad suggestion not marked verified");
    // Changing the dynamic head->group assignment changes the
    // triad sketch but not verification, source or model storage.
    const auto old_hot=TriadCueRouter::encode(cue,groups);
    std::swap(groups[uint8_t('B')],groups[uint8_t('J')]);
    const auto new_hot=TriadCueRouter::encode(cue,groups);
    double movement=0;
    for(size_t i=0;i<old_hot.size();++i)
     movement+=std::fabs(old_hot[i]-new_hot[i]);
    require(movement>1e-4,"live triad remapping is not reflected");
    const auto rerouted=fuzzy(cue);
    require(!rerouted.empty()&&rerouted[0].exact_verified,
            "head reassignment changed exact verification");
    const auto unrelated=fuzzy("Shadows quietly drift across the shore. ");
    require(unrelated.empty(),"far-away phrase suggested unsupported fact");
    std::cout<<"TRIAD_ASSOCIATIVE_AUDIT noisy_tests=3"
             <<" hot_triads="<<int(TriadCueRouter::HOT)
             <<" mapping_change="<<movement
             <<" copy_from_fuzzy=0\n";
    // Near-miss suggestions are explicitly unverified; source labels and
    // candidate cues can be inspected without exposing any fact value.
    const auto suggestions=many.suggest(wrong.data(),wrong.size(),5);
    require(!suggestions.empty(),"phase search produced no candidate hints");
    for(const auto& suggestion:suggestions)
     require(!suggestion.exact_verified,
             "near-miss phase result falsely marked exact");

    std::error_code ignored;std::filesystem::remove(probe_path,ignored);
   }
   require(std::filesystem::file_size(root)==ROOT_BYTES,
           "memory expanded canonical 8 MiB root");
   // Resonance is an optional hint, not an admission criterion. Every
   // phase-ranked proposal must pass the SAME strict byte check.
   UnifiedWaveLanguage::State phase_probe{};
   for(unsigned char c:long_gap+cue)mind.observe(phase_probe,c);
   const auto raw=mind.probe_memory(phase_probe,false);
   const auto phase=mind.probe_memory(phase_probe,true);
   require(raw.record!=ExactEpisodicMemory::NONE&&
           phase.record==raw.record&&phase.exact_compares>0&&
           std::isfinite(phase.resonance),
           "phase candidate ranking changed exact cue identity");
   mind.set_resonance_retrieval_enabled(true);
   UnifiedWaveLanguage::State phase_on{};
   for(unsigned char c:long_gap+cue)mind.observe(phase_on,c);
   require(phase_on.recall_record==raw.record,
           "phase-ranked exact recall lost ground truth");
   const auto hints=mind.suggest_memory(phase_on,3);
   require(!hints.empty()&&hints[0].exact_verified,
           "verified cue was not marked verified in optional suggestions");
   mind.set_resonance_retrieval_enabled(false);
   UnifiedWaveLanguage::State near_miss{};
   for(unsigned char c:std::string("Beacon archive codes: "))
    mind.observe(near_miss,c);
   require(near_miss.recall_record==ExactEpisodicMemory::NONE&&
           mind.probe_memory(near_miss,true).record==
             ExactEpisodicMemory::NONE,
           "phase-similar wrong cue hallucinated an exact record");
   std::cout<<"EPISODIC_RESONANCE_AUDIT candidates="<<phase.routed
            <<" verified_comparisons="<<phase.exact_compares
            <<" exact_match="<<int(phase.record==raw.record)
            <<" resonance="<<phase.resonance<<"\n";
   const auto checkpoint_bytes=std::filesystem::file_size(root+".episodes");
   {
    UnifiedWaveLanguage::State state{};
    for(unsigned char c:long_gap+cue)mind.observe(state,c);
    require(state.length==UnifiedWaveLanguage::WINDOW,
            "long gap did not exceed attention window");
    require(state.recall_record!=ExactEpisodicMemory::NONE,
            "cue did not activate exact retrieval");
    const auto full=mind.predict(state);
    const auto ablated=mind.predict(
      state,UnifiedWaveLanguage::ABLATE_EPISODIC);
    require(full.best==static_cast<uint8_t>(value[0])&&
            full.p[full.best]>0.99999,"retrieval did not control copy");
    require(ablated.p[static_cast<uint8_t>(value[0])]<0.999,
            "no-memory ablation silently used exact value");
    for(size_t i=0;i<value.size();++i){
     const auto next=mind.predict(state);
     require(next.best==static_cast<uint8_t>(value[i]),
             "copy did not survive 64-byte window truncation");
     mind.observe(state,static_cast<uint8_t>(value[i]));
     if(i==100)
      require(state.recall_record!=ExactEpisodicMemory::NONE,
              "retrieval did not maintain multi-byte stream cursor");
    }
    require(state.recall_record==ExactEpisodicMemory::NONE,
            "completed episode cursor did not retire");
   }
   require(mind.generate(long_gap+cue,value.size(),11,1.,1.)==value,
           "exact generation was randomized or truncated");
   // This is an indexed-memory challenge, NOT unseen-language competence.
   // The same fixed stream is evaluated with and without the source store.
   std::istringstream with_stream(long_gap+cue+value);
   std::istringstream without_stream(long_gap+cue+value);
   const auto with_memory=mind.evaluate(with_stream);
   const auto without_memory=mind.evaluate(
      without_stream,UnifiedWaveLanguage::ABLATE_EPISODIC);
   require(with_memory.tokens==without_memory.tokens&&
           without_memory.nll-with_memory.nll>50.0,
           "exact retrieval did not improve long-gap source-backed CE");
   std::cout<<"EPISODIC_GAP_AUDIT gap_bytes="<<long_gap.size()
            <<" value_bytes="<<value.size()
            <<" remembered_ce="<<with_memory.ce()
            <<" no_memory_ce="<<without_memory.ce()<<"\n";
   UnifiedWaveLanguage::State wrong_cue{};
   for(unsigned char c:std::string("Beacon archive coda: "))
    mind.observe(wrong_cue,c);
   require(wrong_cue.recall_record==ExactEpisodicMemory::NONE,
           "wrong cue produced a false exact-memory hit");
   require(std::filesystem::file_size(root+".episodes")==checkpoint_bytes,
           "generation mutated episodic checkpoint");
   const auto scale=mind.wave_memory_scale();
   require(std::fabs(scale-1.0)<1e-12,"legacy wave gate changed");
   UnifiedWaveLanguage::State unrelated{};
   for(unsigned char ch:std::string("standing wave probe "))
    mind.observe(unrelated,ch);
   mind.set_wave_memory_scale(0.);
   require(mind.wave_memory_scale()==0.,"wave gate zero cannot persist");
   const auto zero=mind.predict(unrelated);
   const auto no_wave=mind.predict(
      unrelated,UnifiedWaveLanguage::ABLATE_WAVE);
   for(size_t i=0;i<UnifiedWaveLanguage::N;++i)
    require(std::fabs(zero.p[i]-no_wave.p[i])<1e-12,
            "zero wave gate does not equal no direct-wave ablation");
   mind.set_wave_memory_scale(2.);
   const auto doubled=mind.predict(unrelated);
   require(std::fabs(doubled.p['t']-zero.p['t'])>1e-12,
           "direct wave readout remains obligatory or inert");
   // The memory engine still performs exact retrieval at zero wave gain.
   mind.set_wave_memory_scale(0.);
   require(mind.generate(long_gap+cue,value.size(),19)==value,
           "wave readout cannot be disabled during memory recall");
   mind.set_wave_execution_enabled(false);
   require(!mind.wave_execution_enabled()&&!mind.bptt_enabled(),
           "wave compute-off mode still runs wave gradients");
   UnifiedWaveLanguage::State no_wave_state{};
   for(unsigned char c:long_gap+cue)mind.observe(no_wave_state,c);
   for(double x:no_wave_state.wave)
    require(x==0.0,"standing wave executed with compute gate off");
   for(double x:no_wave_state.inverse_wave)
    require(x==0.0,"inverse wave executed with compute gate off");
   require(no_wave_state.recall_record!=ExactEpisodicMemory::NONE,
           "zero-wave compute prevented exact episodic trigger");
   require(mind.generate(long_gap+cue,value.size(),21)==value,
           "exact recall requires wave recurrence computation");
   std::istringstream on_text("the silent harbor receives a distant ship");
   std::istringstream off_text("the silent harbor receives a distant ship");
   const auto off_score=mind.evaluate(off_text);
   require(off_score.tokens>20&&std::isfinite(off_score.ce()),
           "no-wave compute produces invalid language probabilities");
   mind.set_wave_execution_enabled(true);
   const auto on_score=mind.evaluate(on_text);
   require(on_score.tokens==off_score.tokens&&std::isfinite(on_score.ce()),
           "wave compute ablation changed evaluation token count");
   std::cout<<"EPISODIC_WAVE_COMPUTE_AUDIT enabled_ce="<<on_score.ce()
            <<" disabled_ce="<<off_score.ce()<<"\n";
   bool rejected=false;
   try{mind.memorize("bad","Z",source);}
   catch(const std::invalid_argument&){rejected=true;}
   require(rejected&&mind.episodic_records()==1,
           "invalid cues changed committed memory");
   persisted=mind.wave_memory_scale();
   u.sync();
   require(std::memcmp(ingress.data(),u.root().h[255].data,128)==0,
           "network ingress overwritten by episodic memory");
  }
  {
   Universe u(root);
   UnifiedWaveLanguage mind(u);
   require(mind.episodic_records()==1&&mind.exact_recall(cue)!=nullptr,
           "memory failed checkpoint reopen");
   require(mind.wave_memory_scale()==persisted,"wave gate not persisted");
   require(mind.generate(long_gap+cue,value.size(),997)==value,
           "reopened episodic memory lost exact fact");
   // TSV imports are source-tagged and checkpted in one transaction.
   std::istringstream pairs("Mission label: \tAurora\n"
                            "Mission ID: \tPX-8837\n");
   require(mind.memorize_tsv(pairs,"unit-tsv-source")==2,
           "TSV episode import failed");
   require(mind.exact_recall("Mission ID: ")->value=="PX-8837",
           "TSV memory exact content mismatch");
   std::istringstream authored(
      "The silver spacecraft settled softly by the east laboratory.\n"
      "A quiet forest catches sunlight beyond the ancient village.\n");
   require(mind.memorize_lines(authored,"fixture-book")==2,
           "explicit source-line indexing rejected long quotations");
   const std::string line=
      "The silver spacecraft settled softly by the east laboratory.";
   const size_t cut=line.find(' ',16)+1;
   const std::string quote_key=line.substr(0,cut);
   const std::string quote_tail=line.substr(cut);
   const auto* quotation=mind.exact_recall(quote_key);
   require(quotation&&quotation->value==quote_tail&&
           quotation->source=="fixture-book#L1",
           "indexed quote does not preserve exact source line");
   require(mind.generate(long_gap+quote_key,quote_tail.size(),37)==quote_tail,
           "indexed quotation did not replay through source-backed memory");

   mind.memorize("Mission ID: ","PX-9999","independent-source");
   require(mind.contested_episodic("Mission ID: ")&&
           mind.exact_recall("Mission ID: ")==nullptr,
           "independent contradictory claim was overwritten");
   UnifiedWaveLanguage::State contested_hint{};
   for(unsigned char ch:std::string("Mission ID: "))
    mind.observe(contested_hint,ch);
   const auto hints_conflict=mind.suggest_triad_memory(contested_hint,4,0);
   require(hints_conflict.size()>=2,
           "contesting sources not both visible to suggestion operator");
   for(const auto& hint:hints_conflict)
    if(hint.cue=="Mission ID: ")
     require(!hint.exact_verified,
             "triad router legitimized contested evidence");

   UnifiedWaveLanguage::State contested{};
   for(unsigned char c:std::string("Mission ID: "))mind.observe(contested,c);
   require(contested.recall_record==ExactEpisodicMemory::NONE,
           "contested fact was incorrectly forced into automatic recall");
   mind.set_resonance_retrieval_enabled(true);
   UnifiedWaveLanguage::State contested_phase{};
   for(unsigned char c:std::string("Mission ID: "))
    mind.observe(contested_phase,c);
   require(contested_phase.recall_record==ExactEpisodicMemory::NONE&&
           mind.probe_memory(contested_phase,true).record==
               ExactEpisodicMemory::NONE,
           "resonant routing bypassed contradictory-source abstention");
   mind.set_resonance_retrieval_enabled(false);
   const auto normal=mind.predict(contested);
   const auto no_memory=mind.predict(
      contested,UnifiedWaveLanguage::ABLATE_EPISODIC);
   for(size_t i=0;i<UnifiedWaveLanguage::N;++i)
    require(std::fabs(normal.p[i]-no_memory.p[i])<1e-12,
            "contested memory changed predictive distribution");

  }
  {
   std::fstream damage(root+".episodes",std::ios::binary|
                        std::ios::in|std::ios::out);
   require(bool(damage),"cannot open checkpoint for integrity probe");
   damage.seekg(55);
   const int original=damage.get();
   require(original!=std::char_traits<char>::eof(),
           "checkpoint too short for corruption test");
   damage.seekp(55);
   damage.put(static_cast<char>(original^1));
   damage.flush();
  }
  bool corrupt_rejected=false;
  try{ExactEpisodicMemory bad(root+".episodes");(void)bad;}
  catch(const std::exception&){corrupt_rejected=true;}
  require(corrupt_rejected,"corrupt episodic checkpoint was accepted");
  cleanup();
  std::cout<<"PASS: exact source-grounded episodic recall across 1200-byte gaps "
              "and 200-byte values, restart, integrity, zero-wave gate, "
              "no memory/ingress contamination\n";
  return 0;
 }catch(const std::exception& e){
  std::cerr<<"FAIL: "<<e.what()<<"\n";cleanup();return 1;
 }
}
