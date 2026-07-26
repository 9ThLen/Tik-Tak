#include "analysis/downbeat.hpp"

#include <algorithm>
#include <cmath>

#include "dsp/chroma.hpp"

namespace tiktak::analysis {
namespace {

constexpr std::size_t kChroma = dsp::ChromaFilterbank::kBins;

// Standardises a cue in place: zero mean, unit spread. A cue that never varies
// comes back all zeros, which is the right answer — a constant carries no
// information about where the bar line is, whatever its magnitude.
void standardise(std::vector<double>& v) {
    if (v.empty()) return;

    double mean = 0.0;
    for (double x : v) mean += x;
    mean /= static_cast<double>(v.size());

    double variance = 0.0;
    for (double x : v) variance += (x - mean) * (x - mean);
    variance /= static_cast<double>(v.size());

    const double sd = std::sqrt(variance);
    if (!(sd > 0.0)) {
        std::fill(v.begin(), v.end(), 0.0);
        return;
    }
    for (double& x : v) x = (x - mean) / sd;
}

// First frame at or after `t`, in a sorted time array.
std::size_t frameAtOrAfter(const double* times, std::size_t n, double t) {
    const auto it = std::lower_bound(times, times + n, t);
    return static_cast<std::size_t>(it - times);
}

}  // namespace

bool DownbeatConfig::valid() const {
    if (meters.empty()) return false;
    for (const MeterCandidate& m : meters) {
        if (m.beats_per_bar < 2) return false;
        if (!(m.prior > 0.0)) return false;
    }
    if (!(low_weight >= 0.0) || !(accent_weight >= 0.0) || !(harmony_weight >= 0.0)) return false;
    if (low_weight + accent_weight + harmony_weight <= 0.0) return false;
    if (!(window_before >= 0.0) || !(window_after > 0.0)) return false;
    if (window_after > 1.0) return false;
    if (min_bars < 2) return false;
    return true;
}

std::vector<BeatFeature> beatFeatures(const BeatFeatureInput& input,
                                      const DownbeatConfig& config) {
    std::vector<BeatFeature> out;
    if (input.beats == nullptr || input.beat_count == 0) return out;
    if (input.frame_times == nullptr || input.frame_count == 0) return out;

    out.resize(input.beat_count);

    std::vector<float> chroma_now(kChroma, 0.0f);
    std::vector<float> chroma_prev(kChroma, 0.0f);
    bool have_prev = false;

    for (std::size_t i = 0; i < input.beat_count; ++i) {
        const double beat = input.beats[i];

        // The gap to the next beat sets the scale of everything below. The last
        // beat has no next one, so it borrows the previous gap; if there is
        // only one beat at all there is no meter to find anyway.
        double gap = 0.0;
        if (i + 1 < input.beat_count) {
            gap = input.beats[i + 1] - beat;
        } else if (i > 0) {
            gap = beat - input.beats[i - 1];
        }
        if (!(gap > 0.0)) gap = 0.5;

        BeatFeature& f = out[i];
        f.time_sec = beat;

        const double from = beat - config.window_before * gap;
        const double to = beat + config.window_after * gap;

        std::size_t k = frameAtOrAfter(input.frame_times, input.frame_count, from);
        for (; k < input.frame_count && input.frame_times[k] <= to; ++k) {
            // The peak, not the mean: an onset is an event, and averaging it
            // over a window mostly measures how wide the window is.
            if (input.odf_low != nullptr) f.low = std::max(f.low, input.odf_low[k]);
            if (input.odf_full != nullptr) f.accent = std::max(f.accent, input.odf_full[k]);
        }

        if (input.chroma == nullptr) continue;

        // Harmony is averaged over the whole beat, not peaked over a window:
        // a chord is a state that persists, so more of the beat is more
        // evidence, and the note that happens to be loudest is not the chord.
        std::fill(chroma_now.begin(), chroma_now.end(), 0.0f);
        std::size_t counted = 0;
        std::size_t c = frameAtOrAfter(input.frame_times, input.frame_count, beat);
        for (; c < input.frame_count && input.frame_times[c] < beat + gap; ++c) {
            const float* frame = input.chroma + c * kChroma;
            for (std::size_t b = 0; b < kChroma; ++b) chroma_now[b] += frame[b];
            ++counted;
        }

        if (counted == 0) continue;

        if (have_prev) {
            f.harmonic_change = dsp::chromaDistance(chroma_prev.data(), chroma_now.data());
        }
        std::swap(chroma_prev, chroma_now);
        have_prev = true;
    }

    return out;
}

DownbeatResult findDownbeats(const std::vector<BeatFeature>& features,
                             const DownbeatConfig& config) {
    DownbeatResult result;
    const std::size_t n = features.size();
    if (n == 0 || !config.valid()) return result;

    std::vector<double> low(n);
    std::vector<double> accent(n);
    std::vector<double> harmony(n);
    for (std::size_t i = 0; i < n; ++i) {
        low[i] = features[i].low;
        accent[i] = features[i].accent;
        harmony[i] = features[i].harmonic_change;
    }
    // The onset cues are standardised because their units are arbitrary; the
    // harmony cue is left alone because its units are not. See the weights in
    // DownbeatConfig.
    standardise(low);
    standardise(accent);

    std::vector<double> weight(n);
    for (std::size_t i = 0; i < n; ++i) {
        weight[i] = config.low_weight * low[i] + config.accent_weight * accent[i] +
                    config.harmony_weight * harmony[i];
    }
    // Standardised again so that a contrast is measured in standard deviations
    // of the combined cue. Without this the numbers would depend on how the
    // weights happen to add up and would not compare between pieces.
    standardise(weight);

    double best_score = 0.0;
    bool have_best = false;

    for (const MeterCandidate& meter : config.meters) {
        const auto m = static_cast<std::size_t>(meter.beats_per_bar);
        if (n < m * static_cast<std::size_t>(config.min_bars)) continue;

        MeterScore entry;
        entry.beats_per_bar = meter.beats_per_bar;

        std::vector<double> contrast(m, 0.0);
        bool scored = false;

        for (std::size_t p = 0; p < m; ++p) {
            double in_sum = 0.0;
            std::size_t in_count = 0;
            double out_sum = 0.0;
            std::size_t out_count = 0;
            for (std::size_t i = 0; i < n; ++i) {
                if (i % m == p) {
                    in_sum += weight[i];
                    ++in_count;
                } else {
                    out_sum += weight[i];
                    ++out_count;
                }
            }
            if (in_count == 0 || out_count == 0) continue;

            // Contrast, not total: with a sum, a two-beat bar would win every
            // time simply by claiming half the beats instead of a quarter.
            contrast[p] = in_sum / static_cast<double>(in_count) -
                          out_sum / static_cast<double>(out_count);
            scored = true;
        }
        if (!scored) continue;

        // Strictly greater, scanning upwards, so the earliest phase wins a tie
        // and the same audio always produces the same bar lines.
        std::size_t best_phase = 0;
        for (std::size_t p = 1; p < m; ++p) {
            if (contrast[p] > contrast[best_phase]) best_phase = p;
        }
        const double best_contrast = contrast[best_phase];
        entry.phase = static_cast<int>(best_phase);

        // The rival is the best *other* place the bar line could go. Kept
        // signed: when every other phase scores well below zero the answer is
        // unambiguous, and that deserves to show up as a large margin.
        double runner_up = best_contrast;
        bool have_rival = false;
        for (std::size_t p = 0; p < m; ++p) {
            if (p == best_phase) continue;
            if (!have_rival || contrast[p] > runner_up) {
                runner_up = contrast[p];
                have_rival = true;
            }
        }

        // A negative contrast says the chosen beats are quieter than the ones
        // around them. That is not a weak bar line, it is the wrong answer, and
        // scaling it by a prior would make the least likely meter look best.
        entry.score = std::max(best_contrast, 0.0) * meter.prior;
        result.candidates.push_back(entry);

        if (!have_best || entry.score > best_score) {
            have_best = true;
            best_score = entry.score;
            result.beats_per_bar = entry.beats_per_bar;
            result.phase = entry.phase;
            result.strength = std::max(best_contrast, 0.0);
            result.phase_margin = std::max(best_contrast - runner_up, 0.0);
        }
    }

    std::stable_sort(result.candidates.begin(), result.candidates.end(),
                     [](const MeterScore& a, const MeterScore& b) { return a.score > b.score; });

    if (!have_best) return result;

    // How much better the winning meter is than the best of the others.
    //
    // This has to be a separate number from the phase margin and cannot be
    // derived from it: within one meter every rival phase has already conceded
    // the bar length, so a piece can be entirely unambiguous about where its
    // three-beat bars start while four fits it very nearly as well. Measuring
    // only the first produced confidently wrong meters — a 4/4 track read as
    // three with a phase margin of 0.69, which is the observation that put this
    // here.
    //
    // Scores rather than raw contrasts, so the prior that picked the winner is
    // the same quantity being compared. With one meter in the running there is
    // no rival to lose to, and the winner keeps its whole score.
    result.meter_margin = best_score;
    for (const MeterScore& other : result.candidates) {
        if (other.beats_per_bar == result.beats_per_bar) continue;
        result.meter_margin = std::max(best_score - other.score, 0.0);
        break;  // sorted best first, so the first other meter is the rival
    }

    for (std::size_t i = static_cast<std::size_t>(result.phase); i < n;
         i += static_cast<std::size_t>(result.beats_per_bar)) {
        result.downbeats.push_back(features[i].time_sec);
    }
    return result;
}

}  // namespace tiktak::analysis
