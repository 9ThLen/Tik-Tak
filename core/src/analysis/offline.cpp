#include "analysis/offline.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

namespace tiktak::analysis {

// Bar lines need harmony, and harmony needs the front end to produce it. Rather
// than make the caller keep two flags consistent, asking for downbeats turns on
// what finding them requires.
OfflineConfig OfflineAnalyzer::prepare(OfflineConfig config) {
    if (config.find_downbeats) config.odf.chroma = true;
    return config;
}

OfflineAnalyzer::OfflineAnalyzer(const OfflineConfig& config)
    : config_(prepare(config)),
      odf_(config_.odf),
      tempo_(config_.tempo),
      tracker_(config_.tracker, config_.tempo) {}

void OfflineAnalyzer::feed(const float* samples, std::size_t n) {
    if (samples == nullptr || n == 0) return;

    odf_.process(samples, n, [this](const dsp::OdfFrame& frame) {
        // The full band drives tempo and phase. The low band and the pitch
        // class profile are the downbeat cues, and are only collected when bar
        // lines were asked for — the high band is the subdivision cue and still
        // has no consumer, so a long file stays cheap either way.
        odf_values_.push_back(static_cast<double>(frame.full));
        frame_times_.push_back(frame.timeSec);

        if (!config_.find_downbeats) return;

        odf_low_.push_back(static_cast<double>(frame.low));
        const float* chroma = odf_.chroma();
        if (chroma != nullptr) {
            chroma_.insert(chroma_.end(), chroma, chroma + dsp::ChromaFilterbank::kBins);
        }
    });
}

// Tracks each of the leading tempo candidates and keeps the grid that best
// combines the estimator's confidence in the tempo with how well the grid it
// produces actually fits the onsets.
//
// The two have to be multiplied rather than either used alone. Strength alone
// is what the tracker did before, and it cannot tell a plausible tempo from a
// plausible tempo the music is not actually at. Fit alone is worse still: the
// objective is a per-beat mean, so a grid at half speed raises it simply by
// visiting only the strongest onsets, which is the octave error the prior was
// added to prevent. See OfflineConfig::tempo_fit_weight.
BeatResult OfflineAnalyzer::trackBestHypothesis(double fps, double fallback_bpm) {
    const int wanted = std::max(1, config_.tempo_hypotheses);
    std::vector<TempoCandidate> candidates(static_cast<std::size_t>(wanted));
    const std::size_t found =
        tempo_.topCandidates(candidates.data(), candidates.size());

    BeatResult best;
    double best_score = 0.0;
    bool have_best = false;

    for (std::size_t i = 0; i < found; ++i) {
        const TempoCandidate& candidate = candidates[i];
        if (!(candidate.bpm > 0.0) || !(candidate.strength > 0.0)) continue;

        BeatResult grid = tracker_.track(odf_values_.data(), frame_times_.data(),
                                         odf_values_.size(), fps, candidate.bpm);
        if (grid.beats.empty()) continue;

        const double score = std::pow(candidate.strength, config_.tempo_fit_weight) *
                             grid.objective_per_beat;
        // Strictly greater, so a tie leaves the estimator's own ranking in
        // charge and the same audio always produces the same grid.
        if (!have_best || score > best_score) {
            have_best = true;
            best_score = score;
            best = std::move(grid);
        }
    }

    if (have_best) return best;
    // No candidate produced a grid — fall back to the single estimate, which
    // keeps the failure identical to what it was before this existed.
    return tracker_.track(odf_values_.data(), frame_times_.data(),
                          odf_values_.size(), fps, fallback_bpm);
}

OfflineResult OfflineAnalyzer::finish() {
    OfflineResult result;
    result.frame_count = odf_values_.size();

    const double fps = config_.odf.sampleRate / static_cast<double>(config_.odf.hopSize);

    // Estimate unconditionally, even when a hint will override it. The estimate
    // costs one transform and gives the caller something to say when the two
    // disagree, which is the difference between manual mode failing loudly and
    // failing silently.
    const TempoEstimate estimate = tempo_.estimate(odf_values_.data(), odf_values_.size(), fps);
    result.estimated_bpm = estimate.bpm;

    BeatResult tracked;
    if (config_.bpm_hint > 0.0) {
        // A hint is an instruction, not a hypothesis. Searching around it would
        // be overriding the caller.
        tracked = tracker_.track(odf_values_.data(), frame_times_.data(),
                                 odf_values_.size(), fps, config_.bpm_hint);
    } else {
        tracked = trackBestHypothesis(fps, estimate.bpm);
    }

    result.beats = tracked.beats;
    result.bpm = tracked.bpm;
    result.beat_objective_per_beat = tracked.objective_per_beat;
    // A hint is the caller's assertion, so it carries their certainty, not ours.
    result.tempo_confidence = config_.bpm_hint > 0.0 ? 1.0 : estimate.confidence;

    if (config_.find_downbeats && !result.beats.empty()) {
        BeatFeatureInput input;
        input.frame_times = frame_times_.data();
        input.odf_full = odf_values_.data();
        input.odf_low = odf_low_.empty() ? nullptr : odf_low_.data();
        input.chroma = chroma_.empty() ? nullptr : chroma_.data();
        input.frame_count = frame_times_.size();
        input.beats = result.beats.data();
        input.beat_count = result.beats.size();

        result.beat_features = beatFeatures(input, config_.downbeat);
        const DownbeatResult bars =
            findDownbeats(result.beat_features, config_.downbeat);
        result.downbeats = bars.downbeats;
        result.beats_per_bar = bars.beats_per_bar;
        result.downbeat_strength = bars.strength;
        result.downbeat_phase_margin = bars.phase_margin;
        result.downbeat_meter_margin = bars.meter_margin;
        result.downbeat_confident = bars.confident(config_.downbeat.min_phase_margin,
                                                   config_.downbeat.min_meter_margin);
    }
    return result;
}

void OfflineAnalyzer::reset() {
    odf_.reset();
    odf_values_.clear();
    frame_times_.clear();
    odf_low_.clear();
    chroma_.clear();
}

OfflineResult analyseOffline(const float* samples, std::size_t n, const OfflineConfig& config) {
    OfflineAnalyzer analyzer(config);
    analyzer.feed(samples, n);
    return analyzer.finish();
}

}  // namespace tiktak::analysis
