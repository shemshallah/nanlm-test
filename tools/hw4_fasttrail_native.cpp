// HW4 FastTRAIL: one immutable corpus scan, aggregate rank-three CE gradient,
// plus sampled forward/inverse TRAIL eligibility on contiguous fixed-size windows.
// One shared, already-existing 8 MiB NaNLM student; no BPTT or second model.
// The two directions compete under the same full-corpus CE acceptance gate.
// IMPORTANT: sampled TRAIL credit is a biased truncated eligibility estimate,
// not a full-corpus TRAIL pass or an exact gradient of cross-entropy.
#include "nanlm/unified_wave.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

using namespace nanlm;
namespace {
constexpr size_t V = FabricTeacher::VOCAB;
constexpr uint64_t WINDOW = 2048;
constexpr double GAMMA = 0.93;
constexpr double LAMBDA = 0.82;
constexpr double DECAY = GAMMA * LAMBDA;
using Vec = std::array<double, 3>;
using Weights = std::array<Vec, V>;
using Counts = std::array<std::array<uint64_t, V>, V>;
using Probs = std::array<std::array<double, V>, V>;

Vec read_vec(const Head& head, size_t offset) {
  std::array<float, 3> raw{};
  std::memcpy(raw.data(), head.data + offset, sizeof(raw));
  return {raw[0], raw[1], raw[2]};
}
void write_vec(Head& head, size_t offset, const Vec& value) {
  std::array<float, 3> raw{};
  for (size_t k = 0; k < 3; ++k)
    raw[k] = static_cast<float>(std::clamp(value[k], -2.0, 2.0));
  std::memcpy(head.data + offset, raw.data(), sizeof(raw));
}
double seconds(std::chrono::steady_clock::time_point from) {
  return std::chrono::duration<double>(
    std::chrono::steady_clock::now() - from).count();
}
double evaluate(const Counts& counts, const std::array<uint64_t, V>& rows,
                uint64_t transitions, const FabricTeacher& teacher,
                double& accuracy) {
  long double loss = 0;
  uint64_t correct = 0;
  for (size_t a = 0; a < V; ++a) {
    if (!rows[a]) continue;
    const auto p = teacher.predict(static_cast<uint8_t>(a));
    correct += counts[a][p.choice];
    for (size_t b = 0; b < V; ++b)
      if (counts[a][b])
        loss -= static_cast<long double>(counts[a][b]) *
                std::log(std::max(p.probability[b], 1e-300));
  }
  accuracy = static_cast<double>(correct) / static_cast<double>(transitions);
  return static_cast<double>(loss / static_cast<long double>(transitions));
}
struct TwoRail {
  Weights eligibility_plus{}, eligibility_minus{}, credit_plus{}, credit_minus{};
  void clear_traces() {
    eligibility_plus = {};
    eligibility_minus = {};
  }
  void advance(uint8_t previous, uint8_t current, const Probs& forward,
               const Probs& inverse, const Weights& src, const Weights& dst) {
    const auto& pf = forward[previous];
    const auto& pf_next = forward[current];
    const auto& pr = inverse[current];
    const auto& pr_next = inverse[previous];
    const auto& sv = src[previous];
    const auto& dv = dst[current];
    for (size_t c = 0; c < V; ++c) {
      const double gp = 4.0 * pf[c] * (1.0 - pf[c]);
      const double gm = 4.0 * pr[c] * (1.0 - pr[c]);
      const double delta_plus =
          (c == current ? 1.0 : 0.0) + GAMMA * pf_next[c] - pf[c];
      const double delta_minus =
          (c == previous ? 1.0 : 0.0) + GAMMA * pr_next[c] - pr[c];
      for (size_t k = 0; k < 3; ++k) {
        double& ep = eligibility_plus[c][k];
        double& em = eligibility_minus[c][k];
        ep = DECAY * ep + gp * sv[k];
        em = DECAY * em + gm * dv[k];
        credit_plus[c][k] += delta_plus * ep;
        credit_minus[c][k] += delta_minus * em;
      }
    }
  }
};
double maxabs(const Weights& a, const Weights& b) {
  double result = 0;
  for (size_t i = 0; i < V; ++i)
    for (size_t k = 0; k < 3; ++k)
      result = std::max({result, std::fabs(a[i][k]), std::fabs(b[i][k])});
  return result;
}
void save_vectors(Universe& u, const Weights& src, const Weights& dst) {
  for (size_t i = 0; i < V; ++i) {
    write_vec(u.root().h[i], 32, src[i]);
    write_vec(u.root().h[i], 48, dst[i]);
  }
}
struct Result {
  double loss = std::numeric_limits<double>::infinity();
  double accuracy = 0;
  double step = 0;
  bool accepted = false;
  Weights src{}, dst{};
};
Result search_direction(Universe& u, FabricTeacher& teacher,
                        const Counts& counts,
                        const std::array<uint64_t, V>& rows,
                        uint64_t transitions, const Weights& old_src,
                        const Weights& old_dst, const Weights& step_src,
                        const Weights& step_dst, double baseline,
                        double baseline_accuracy) {
  Result best{};
  best.loss = baseline;
  best.accuracy = baseline_accuracy;
  const double magnitude = maxabs(step_src, step_dst);
  if (!std::isfinite(magnitude) || magnitude <= 1e-17)
    return best;
  for (int attempt = 0; attempt < 18; ++attempt) {
    const double step = std::ldexp(0.16 / magnitude, -attempt);
    for (size_t i = 0; i < V; ++i) {
      Vec src = old_src[i], dst = old_dst[i];
      for (size_t k = 0; k < 3; ++k) {
        src[k] += step * step_src[i][k];
        dst[k] += step * step_dst[i][k];
      }
      write_vec(u.root().h[i], 32, src);
      write_vec(u.root().h[i], 48, dst);
    }
    double trial_accuracy = 0;
    const double trial_loss = evaluate(counts, rows, transitions,
                                       teacher, trial_accuracy);
    if (std::isfinite(trial_loss) && trial_loss + 1e-9 < best.loss) {
      best.loss = trial_loss;
      best.accuracy = trial_accuracy;
      best.step = step;
      best.accepted = true;
      for (size_t i = 0; i < V; ++i) {
        best.src[i] = read_vec(u.root().h[i], 32);
        best.dst[i] = read_vec(u.root().h[i], 48);
      }
    }
    save_vectors(u, old_src, old_dst);
  }
  return best;
}
} // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 4 || argc > 6)
      throw std::invalid_argument(
        "usage: hw4-fasttrail ROOT CORPUS EXPECTED_BYTES [WINDOW_STRIDE=64] [TRAIL_BLEND=0.25]");
    const auto started = std::chrono::steady_clock::now();
    const std::filesystem::path root = argv[1], corpus = argv[2];
    const uint64_t expected = std::stoull(argv[3]);
    const uint64_t stride = argc >= 5 ? std::stoull(argv[4]) : 64;
    const double blend = argc >= 6 ? std::stod(argv[5]) : 0.25;
    if (expected < 1024 || expected > (1ull << 30) ||
        stride == 0 || stride > 4096 || !std::isfinite(blend) ||
        blend < 0.0 || blend > 1.0)
      throw std::invalid_argument("invalid corpus bytes, TRAIL stride or blend");
    if (!std::filesystem::is_regular_file(corpus) ||
        std::filesystem::file_size(corpus) != expected ||
        !std::filesystem::is_regular_file(root) ||
        std::filesystem::file_size(root) != ROOT_BYTES ||
        std::filesystem::equivalent(root, corpus) ||
        root.filename() == "wordnet-c256.root" ||
        root.string().find("hw4-full-reference") != std::string::npos)
      throw std::runtime_error("immutable corpus/reference or student root validation failed");

    Universe u(root.string());
    UnifiedWaveLanguage mind(u);
    if (mind.bptt_enabled())
      throw std::runtime_error("unexpected enabled BPTT");
    FabricTeacher teacher(u);
    std::array<uint8_t, 128> ingress{};
    std::memcpy(ingress.data(), u.root().h[255].data, ingress.size());
    Weights old_src{}, old_dst{}, grad_src{}, grad_dst{}, td_src{}, td_dst{};
    for (size_t i = 0; i < V; ++i) {
      old_src[i] = read_vec(u.root().h[i], 32);
      old_dst[i] = read_vec(u.root().h[i], 48);
    }
    Probs forward{}, inverse{}, scores{};
    for (size_t i = 0; i < V; ++i) {
      const auto prediction = teacher.predict(static_cast<uint8_t>(i));
      for (size_t j = 0; j < V; ++j) {
        forward[i][j] = prediction.probability[j];
        scores[i][j] = prediction.logit[j];
      }
    }
    for (size_t b = 0; b < V; ++b) {
      double largest = -std::numeric_limits<double>::infinity();
      for (size_t a = 0; a < V; ++a)
        largest = std::max(largest, scores[a][b]);
      double sum = 0;
      for (size_t a = 0; a < V; ++a) {
        inverse[b][a] = std::exp(scores[a][b] - largest);
        sum += inverse[b][a];
      }
      if (!(sum > 0) || !std::isfinite(sum))
        throw std::runtime_error("invalid inverse softmax");
      for (double& p : inverse[b]) p /= sum;
    }

    const auto scan_start = std::chrono::steady_clock::now();
    Counts counts{};
    std::array<uint64_t, V> rows{};
    TwoRail trail{};
    std::ifstream input(corpus, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open corpus");
    std::array<char, 1 << 17> buffer{};
    int previous = -1;
    uint64_t bytes = 0, transitions = 0, trail_transitions = 0;
    uint64_t windows = 0;
    while (input) {
      input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
      const std::streamsize n = input.gcount();
      for (std::streamsize i = 0; i < n; ++i) {
        const uint8_t current = static_cast<uint8_t>(
            static_cast<unsigned char>(buffer[static_cast<size_t>(i)]));
        if (current >= V)
          throw std::runtime_error("0xFF reserved ingress byte in corpus");
        const uint64_t window = bytes / WINDOW;
        const uint64_t offset = bytes % WINDOW;
        const bool selected = blend > 0.0 && window % stride == 0;
        if (selected && offset == 0) {
          trail.clear_traces();
          ++windows;
        }
        if (previous >= 0) {
          ++counts[static_cast<size_t>(previous)][current];
          ++rows[static_cast<size_t>(previous)];
          ++transitions;
          if (selected && offset != 0) {
            trail.advance(static_cast<uint8_t>(previous), current,
                          forward, inverse, old_src, old_dst);
            ++trail_transitions;
          }
        }
        previous = current;
        ++bytes;
      }
    }
    if (input.bad() || bytes != expected || transitions + 1 != bytes ||
        (blend > 0.0 && trail_transitions == 0))
      throw std::runtime_error("incomplete scan or no sampled TRAIL transitions");
    const double scan_seconds = seconds(scan_start);
    double before_accuracy = 0;
    const double before = evaluate(counts, rows, transitions,
                                   teacher, before_accuracy);

    // Same CE surrogate gradient as the original fast native learner.
    // This only differentiates the rank-3 4*dot(src,dst) score contribution;
    // angular/transport terms remain frozen in this local optimizer.
    for (size_t a = 0; a < V; ++a) {
      if (!rows[a]) continue;
      for (size_t b = 0; b < V; ++b) {
        const double residual =
            4.0 * (forward[a][b] * static_cast<double>(rows[a]) -
                   static_cast<double>(counts[a][b])) /
            static_cast<double>(transitions);
        for (size_t k = 0; k < 3; ++k) {
          grad_src[a][k] -= residual * old_dst[b][k];
          grad_dst[b][k] -= residual * old_src[a][k];
        }
      }
    }
    const double fast_max = maxabs(grad_src, grad_dst);
    if (!std::isfinite(fast_max) || fast_max < 1e-16)
      throw std::runtime_error("invalid or degenerate fast gradient");
    for (size_t i = 0; i < V; ++i)
      for (size_t k = 0; k < 3; ++k) {
        grad_src[i][k] /= fast_max;
        grad_dst[i][k] /= fast_max;
        if (blend > 0.0) {
          td_src[i][k] = -trail.credit_minus[i][k] /
                         static_cast<double>(trail_transitions);
          td_dst[i][k] = trail.credit_plus[i][k] /
                         static_cast<double>(trail_transitions);
        }
      }
    const double rail_max = maxabs(td_src, td_dst);
    if (blend > 0.0 && (!std::isfinite(rail_max) || rail_max < 1e-17))
      throw std::runtime_error("invalid or degenerate sampled TRAIL credit");
    // Normalize independently; remove a conflicting component parallel
    // to the negative of the fast descent direction, preserving first-order
    // alignment with the approximate aggregate CE descent direction.
    double dot_ft = 0, norm_f = 0, norm_t = 0;
    for (size_t i = 0; i < V; ++i)
      for (size_t k = 0; k < 3; ++k) {
        if (blend > 0.0) {
          td_src[i][k] /= rail_max;
          td_dst[i][k] /= rail_max;
        }
        dot_ft += grad_src[i][k] * td_src[i][k] +
                  grad_dst[i][k] * td_dst[i][k];
        norm_f += grad_src[i][k] * grad_src[i][k] +
                  grad_dst[i][k] * grad_dst[i][k];
        norm_t += td_src[i][k] * td_src[i][k] +
                  td_dst[i][k] * td_dst[i][k];
      }
    const double cosine = norm_t > 0.0 ? dot_ft / std::sqrt(norm_f * norm_t) : 0.0;
    if (dot_ft < 0) {
      const double projection = dot_ft / norm_f;
      for (size_t i = 0; i < V; ++i)
        for (size_t k = 0; k < 3; ++k) {
          td_src[i][k] -= projection * grad_src[i][k];
          td_dst[i][k] -= projection * grad_dst[i][k];
        }
    }
    Weights mixed_src = grad_src, mixed_dst = grad_dst;
    for (size_t i = 0; i < V; ++i)
      for (size_t k = 0; k < 3; ++k) {
        mixed_src[i][k] += blend * td_src[i][k];
        mixed_dst[i][k] += blend * td_dst[i][k];
      }
    // Try both candidate directions, each with bounded backtracking.
    // Select the lower observed FULL-corpus CE rather than assume TRAIL
    // always helps. Rejected candidates are restored before persisting.
    Result fast = search_direction(u, teacher, counts, rows, transitions,
                                   old_src, old_dst, grad_src, grad_dst,
                                   before, before_accuracy);
    Result mixed = blend > 0 ?
        search_direction(u, teacher, counts, rows, transitions, old_src,
                         old_dst, mixed_src, mixed_dst, before,
                         before_accuracy) : Result{};
    const bool rail_won = mixed.accepted &&
                         (!fast.accepted || mixed.loss + 1e-12 < fast.loss);
    const Result& chosen = rail_won ? mixed : fast;
    if (chosen.accepted)
      save_vectors(u, chosen.src, chosen.dst);
    if (std::memcmp(ingress.data(), u.root().h[255].data,
                    ingress.size()) != 0)
      throw std::runtime_error("reserved ingress head mutated");
    u.sync();
    uint64_t nonzero = 0;
    for (const auto& line : counts)
      for (uint64_t count : line) nonzero += count > 0;

    std::cout << std::fixed << std::setprecision(10)
              << "HW4_FASTTRAIL"
              << " input_bytes=" << bytes
              << " transition_updates=" << transitions
              << " sampled_trail_transitions=" << trail_transitions
              << " sampled_windows=" << windows
              << " trail_window=" << WINDOW
              << " trail_stride=" << stride
              << " trail_fraction=" <<
                  (static_cast<double>(trail_transitions) /
                   static_cast<double>(transitions))
              << " trail_blend=" << blend
              << " rail_fast_cosine=" << cosine
              << " fast_candidate_ce=" << fast.loss
              << " mixed_candidate_ce=" << (mixed.accepted ? mixed.loss : before)
              << " initial_teacher_ce=" << before
              << " final_teacher_ce=" << (chosen.accepted ? chosen.loss : before)
              << " initial_teacher_acc=" << before_accuracy
              << " final_teacher_acc=" <<
                  (chosen.accepted ? chosen.accuracy : before_accuracy)
              << " accepted=" << int(chosen.accepted)
              << " rail_selected=" << int(rail_won)
              << " applied_step=" << (chosen.accepted ? chosen.step : 0)
              << " nonzero_byte_pairs=" << nonzero
              << " scan_seconds=" << scan_seconds
              << " total_seconds=" << seconds(started)
              << " bptt_attempted=0"
              << " root_bytes=" << ROOT_BYTES
              << " sample_mode=fixed_2048_byte_windows"
              << " objective=full_corpus_byte_pair_ce"
              << " source_role=immutable"
              << " model_role=one_nanlm_student\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "HW4_FASTTRAIL_ERROR " << error.what() << "\n";
    return 2;
  }
}
