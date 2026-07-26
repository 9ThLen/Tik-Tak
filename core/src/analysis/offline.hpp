#pragma once

#include <cstddef>
#include <vector>

#include "analysis/downbeat.hpp"
#include "analysis/tempo.hpp"
#include "analysis/tracker.hpp"
#include "dsp/odf.hpp"

namespace tiktak::analysis {

struct OfflineConfig {
    dsp::OdfConfig odf;
    TempoConfig tempo;
    TrackerConfig tracker;
    DownbeatConfig downbeat;
    // Fix the tempo instead of estimating it (manual mode). <= 0 estimates.
    double bpm_hint = 0.0;

    // Also find the bar lines. Turning this on implies OdfConfig::chroma and
    // makes the analyser keep the low band and a pitch class profile per frame
    // as well — roughly fourteen extra numbers per hop, which for a five-minute
    // track is a few megabytes and worth it for a file that is analysed once.
    //
    // Off is a real option, not a debugging one: a caller that only wants a
    // click on every beat has no use for bar lines and should not pay for them.
    bool find_downbeats = true;
};

struct OfflineResult {
    std::vector<double> beats;   // beat times, seconds from the start of the audio
    // The tempo the beats were tracked at: `bpm_hint` when one was given,
    // otherwise `estimated_bpm`.
    double bpm = 0.0;
    double tempo_confidence = 0.0;
    // What the audio itself says, measured even when a hint overrode it — so
    // manual mode can tell the user their 120 sounds like 90 instead of
    // silently tracking badly.
    double estimated_bpm = 0.0;
    std::size_t frame_count = 0;

    // Bar lines, a subset of `beats`. Empty when they were not asked for, or
    // when the track was too short for any meter to repeat.
    std::vector<double> downbeats;
    int beats_per_bar = 0;
    // See DownbeatResult: how far the bar lines stand above the beats around
    // them, and how far ahead of the next best place to put them. Both in
    // standard deviations of the per-beat cue. A UI that accents downbeats
    // should stop doing so when `downbeat_margin` is small — putting the accent
    // in the wrong place is worse than putting it nowhere.
    double downbeat_strength = 0.0;
    double downbeat_margin = 0.0;
};

// Whole-file beat analysis: audio in, a beat grid out.
//
// Audio is fed in blocks and reduced to ODF frames as it arrives, so a long
// track never has to be held in memory at once — a five-minute file becomes a
// few hundred kilobytes of onset frames instead of fifty megabytes of samples.
// Only when finish() is called does the tempo estimate and the dynamic-
// programming tracker run over the whole collected function, which is what
// makes the offline path more accurate than the microphone path: it decides
// with the entire piece in view.
//
// Offline component: allocates while collecting and while analysing, so it must
// not be driven from an audio callback. Feed it from a file-reading thread.
class OfflineAnalyzer {
public:
    explicit OfflineAnalyzer(const OfflineConfig& config);

    const OfflineConfig& config() const { return config_; }

    // Appends `n` mono samples. Any number of samples per call.
    void feed(const float* samples, std::size_t n);

    // Runs tempo estimation and beat tracking over everything fed so far.
    // Repeatable; feeding more and calling again extends the analysis.
    OfflineResult finish();

    // Clears all collected frames and the ODF state, ready for another file.
    void reset();

    // The collected onset function and its frame times, for diagnostics and for
    // the parity harness.
    const std::vector<double>& odfValues() const { return odf_values_; }
    const std::vector<double>& frameTimes() const { return frame_times_; }

    // Alternative readings of the tempo from the last finish(), strongest
    // first. These are what the half/double toggle in the UI should offer: when
    // the runner-up is an octave away with a similar score, the estimate is a
    // coin toss and the user is better placed to break the tie than we are.
    std::size_t tempoCandidates(TempoCandidate* out, std::size_t count) const {
        return tempo_.topCandidates(out, count);
    }

private:
    static OfflineConfig prepare(OfflineConfig config);

    OfflineConfig config_;
    dsp::Odf odf_;
    TempoEstimator tempo_;
    BeatTracker tracker_;

    std::vector<double> odf_values_;
    std::vector<double> frame_times_;
    std::vector<double> odf_low_;
    std::vector<float> chroma_;  // frame-major, kBins per frame
};

// Convenience wrapper for callers that already hold the whole signal.
OfflineResult analyseOffline(const float* samples, std::size_t n, const OfflineConfig& config);

}  // namespace tiktak::analysis
