#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "dsp/fft.hpp"

namespace tiktak::tracking {

struct ActivationTempoConfig {
    double min_bpm = 40.0;
    double max_bpm = 220.0;
    int grid_size = 512;

    // The same log-normal prior the offline estimator uses, and centred at the
    // same place for the same measured reason — see analysis/tempo.hpp, which
    // carries the corpus sweep behind 140. Two estimators disagreeing about
    // which octave is a priori plausible would be a bug that only shows up as
    // the two of them fighting.
    double prior_centre_bpm = 140.0;
    double prior_width_octaves = 0.7;

    // How much history the autocorrelation sees. Longer is better and flattens
    // out: measured over BeatNet's activation on ballroom, the right octave is
    // found on 81.7% of recordings from fifteen seconds, 82.8% from twenty and
    // 85.1% from thirty, against 85.1% from the whole recording. Thirty seconds
    // is where the causal estimate stops differing from the acausal one, so
    // there is nothing to buy above it.
    double window_sec = 30.0;

    // No answer before this much has been heard. Five seconds reaches 64.2%
    // (ballroom) and 61.4% (GTZAN), which is around where pinning a period
    // stops paying for itself, so answering that early would mostly be
    // answering wrongly with confidence.
    double min_window_sec = 15.0;

    // The autocorrelation is recomputed at most this often. It is a 4096-point
    // transform over a buffer that changes by one frame at a time, so doing it
    // per frame would be fifty times the work for an answer that cannot move
    // fifty times a second.
    double update_interval_sec = 1.0;

    // Frames a second of the internal grid. Observations are binned onto it,
    // so the estimator does not care whether the front end hands it BeatNet's
    // fifty a second or the ODF's rate, nor whether that rate is exactly
    // uniform.
    double fps = 50.0;

    bool valid() const;
};

struct ActivationTempoEstimate {
    // 0 when there is not yet enough history to answer. Not the same as a
    // tempo of zero and not the same as 120 — a caller that cannot tell those
    // apart will hold a period it was never given.
    double bpm = 0.0;

    // 0..1: how strongly the activation actually repeats at `bpm`, as a
    // fraction of its own variance. The same quantity, on the same scale, as
    // analysis::TempoEstimate::confidence.
    double confidence = 0.0;

    // How far the runner-up at another metrical level sits below the winner,
    // as a fraction of the winner's posterior. Near zero means half and double
    // are tied and the answer is a coin toss; this is the number to gate on
    // when the question is "which octave", which is not the same question as
    // "is there a pulse at all".
    double octave_margin = 0.0;

    bool answered() const { return bpm > 0.0; }
};

// Tempo from a causal stream of beat activations, by autocorrelation.
//
// The particle filter estimates a period too, and this exists because it
// estimates it worse. Both are fed the same BeatNet activation; asked at the
// same instant which octave the music is in, the filter is right on 66.6% of
// ballroom and 58.5% of GTZAN after fifteen seconds, and an autocorrelation
// over the same fifteen seconds is right on 81.7% and 69.1%. The information
// is in the activation and the filter is discarding it, because a filter is a
// local, recursive thing and an octave is a global fact about a recording.
//
// What that is worth downstream, with the resulting period held for the rest
// of the recording (698 ballroom and 998 GTZAN recordings, against annotated
// beats):
//
//                          ballroom                    GTZAN
//                     F     CMLt   AMLt   right    F     CMLt   AMLt   right
//     free running  0.700  0.584  0.600         0.632  0.508  0.589
//     held from 20s 0.778  0.782  0.841  85.4%  0.697  0.637  0.780  72.0%
//     the answer    0.881  0.873  0.889         0.737  0.790  0.826
//
// Read the GTZAN row carefully, because it settles a question that had been
// answered by interpolation. Holding the *filter's* own estimate, right 60.5%
// of the time, loses to free running (F 0.588 against 0.632). Holding this
// one, right 72.0%, beats it (0.697). Break-even is therefore somewhere in the
// sixties, not the eighties as a straight line drawn between 60% and the
// oracle's 100% suggested. An estimator does not have to be nearly right to be
// worth holding; it has to be better than two thirds.
//
// What this deliberately does not do is decide anything. It reports a tempo, a
// strength and an octave margin, and the tracker decides what to do with them.
// Holding a period outright is measured above and is not what ships: both
// corpora are steady-tempo music and cannot tell a held period from a held
// metrical level, and only one of those survives a singer slowing down.
//
// Real-time safe: observe() allocates nothing, and the transform is sized in
// the constructor.
class ActivationTempo {
public:
    explicit ActivationTempo(const ActivationTempoConfig& config);

    const ActivationTempoConfig& config() const { return config_; }

    // Feeds one observation: how much this instant looks like a beat, 0 to 1,
    // at a time in the caller's clock. Times must not go backwards.
    //
    // Observations are binned onto the internal grid by taking the largest in
    // each bin rather than the latest or the mean. An activation is a spike a
    // few frames wide, and the mean over a bin mostly measures the bin.
    void observe(double time_sec, double activation);

    ActivationTempoEstimate estimate() const { return estimate_; }

    // Forgets everything heard. For a new song, not for a gap in one.
    void reset();

    // Seconds of history currently held, saturating at window_sec.
    double heard_sec() const;

    // The candidate grid, log-spaced from min_bpm to max_bpm. Constant.
    const std::vector<double>& bpmGrid() const { return grid_; }
    // Posterior from the last recomputation, peak-normalised to 1.
    const std::vector<double>& posterior() const { return posterior_; }

private:
    void recompute();

    ActivationTempoConfig config_;
    std::vector<double> grid_;      // candidate tempi, BPM
    std::vector<double> prior_;     // log-normal weight per candidate
    std::vector<double> ring_;      // activation history, oldest at head_
    std::vector<double> linear_;    // ring_ unwrapped, oldest first
    std::vector<double> acf_;
    std::vector<double> posterior_;
    std::vector<double> scratch_re_;
    std::vector<double> scratch_im_;
    std::unique_ptr<dsp::Fft64> fft_;

    std::size_t head_ = 0;          // next slot to write
    std::size_t filled_ = 0;        // slots written, saturating at ring_.size()
    long long current_bin_ = -1;    // bin index of the slot at head_ - 1
    double first_time_sec_ = 0.0;
    double last_update_sec_ = 0.0;
    bool started_ = false;
    ActivationTempoEstimate estimate_;
};

}  // namespace tiktak::tracking
