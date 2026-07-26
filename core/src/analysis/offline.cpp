#include "analysis/offline.hpp"

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

    const double bpm = config_.bpm_hint > 0.0 ? config_.bpm_hint : estimate.bpm;
    const BeatResult tracked = tracker_.track(odf_values_.data(), frame_times_.data(),
                                              odf_values_.size(), fps, bpm);

    result.beats = tracked.beats;
    result.bpm = tracked.bpm;
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

        const DownbeatResult bars =
            findDownbeats(beatFeatures(input, config_.downbeat), config_.downbeat);
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
