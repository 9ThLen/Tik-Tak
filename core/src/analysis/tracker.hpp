#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "analysis/tempo.hpp"

namespace tiktak::analysis {

struct TrackerConfig {
    // Weight of the tempo-consistency penalty. Higher keeps the beat grid rigid
    // through weak passages; lower lets it follow a rubato performer.
    double tightness = 100.0;
    // Trim beats at the very start and end that sit on no real onset — the DP
    // will happily extend its grid into silence to keep the sequence regular.
    bool trim = true;

    bool valid() const;
};

struct BeatResult {
    std::vector<double> beats;         // beat times, seconds
    std::vector<std::size_t> frames;   // ODF frame index of each beat
    double bpm = 0.0;                  // period the tracker was run at
    double tempo_confidence = 0.0;
};

// Offline beat tracking by dynamic programming (Ellis, 2007).
//
// This is the *offline* back-end: it sees the whole ODF before deciding
// anything, so it can pick the globally best beat sequence instead of
// committing frame by frame. That is why an imported track is analysed far
// more accurately than the microphone path ever will be, and why it is worth
// having two trackers over one shared front-end.
//
// The objective, maximised over all beat sequences b_1..b_N:
//
//     sum_i  odf(b_i)  +  tightness * sum_i  -( log( (b_i - b_{i-1}) / period ) )^2
//
// The first term wants beats on onsets; the second wants the gaps between them
// to stay near the estimated period. One forward pass with a backtrace solves
// it exactly.
//
// Offline component: allocates per call and is not real-time safe.
class BeatTracker {
public:
    explicit BeatTracker(const TrackerConfig& config, const TempoConfig& tempo_config);

    const TrackerConfig& config() const { return config_; }

    // Finds the beat sequence in `n` ODF frames.
    //
    // `times` comes from the ODF rather than being derived from `fps`, because
    // the ODF stamps each frame with its window centre; recomputing the times
    // here would reintroduce half a window of bias.
    //
    // `bpm` <= 0 estimates the tempo first; pass a positive value to fix it,
    // which is exactly what the app's manual mode does.
    BeatResult track(const double* odf, const double* times, std::size_t n, double fps,
                     double bpm = 0.0);

    // Per-frame "beatiness" from the last track(), for diagnostics.
    const std::vector<double>& localScore() const { return local_score_; }

private:
    void computeLocalScore(const double* odf, std::size_t n, double period);
    void forward(double period);
    std::size_t lastBeat() const;
    void trim();

    TrackerConfig config_;
    TempoEstimator tempo_;

    std::vector<double> local_score_;
    std::vector<double> cumulative_;
    std::vector<std::int64_t> backlink_;
    std::vector<std::size_t> frames_;
};

}  // namespace tiktak::analysis
