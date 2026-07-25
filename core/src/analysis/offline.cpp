#include "analysis/offline.hpp"

namespace tiktak::analysis {

OfflineAnalyzer::OfflineAnalyzer(const OfflineConfig& config)
    : config_(config),
      odf_(config.odf),
      tempo_(config.tempo),
      tracker_(config.tracker, config.tempo) {}

void OfflineAnalyzer::feed(const float* samples, std::size_t n) {
    if (samples == nullptr || n == 0) return;

    odf_.process(samples, n, [this](const dsp::OdfFrame& frame) {
        // The full band drives tempo and phase; low and high are kept for the
        // downbeat and subdivision work in a later phase and are not collected
        // here, so a long file stays cheap.
        odf_values_.push_back(static_cast<double>(frame.full));
        frame_times_.push_back(frame.timeSec);
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
    return result;
}

void OfflineAnalyzer::reset() {
    odf_.reset();
    odf_values_.clear();
    frame_times_.clear();
}

OfflineResult analyseOffline(const float* samples, std::size_t n, const OfflineConfig& config) {
    OfflineAnalyzer analyzer(config);
    analyzer.feed(samples, n);
    return analyzer.finish();
}

}  // namespace tiktak::analysis
