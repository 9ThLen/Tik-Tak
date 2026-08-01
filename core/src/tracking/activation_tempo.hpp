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

    // How much history the autocorrelation sees. Short, and deliberately much
    // shorter than this estimator alone would ask for.
    //
    // Taken by itself, longer is plainly better: over BeatNet's activation on
    // ballroom the right octave is found on 64.2% of recordings from five
    // seconds, 81.7% from fifteen, 85.1% from thirty and 85.1% from the whole
    // recording. On that evidence this was 30, and that was wrong.
    //
    // What the isolated measurement misses is that the estimate is consumed by
    // a tracker, and a stale estimate is worth less there than an inaccurate
    // one. Scored end to end through the anchor, over 698 ballroom and 999
    // GTZAN recordings, shorter wins on the corpus *and* on how fast a tempo
    // change is followed — both at once, which is why this is not a trade:
    //
    //     window   ballroom F   GTZAN F   worst lag over six tempo changes
    //      6 s       0.794       0.666            6.8 s
    //      8 s       0.785       0.665            7.2 s
    //     10 s       0.781       0.663           10.4 s
    //     30 s       0.737       0.648           23.1 s
    //
    // Six seconds of activation is a worse look at the recording and a better
    // answer about it.
    //
    // **Below six has since been tested, and six stays.** Swept down to 1.5 s,
    // the corpus means look like a clean peak at four:
    //
    //     window   ballroom F/CMLt   GTZAN F/CMLt   SMC F/CMLt
    //      8 s      0.785 / 0.694    0.665 / 0.561  0.246 / 0.116
    //      6 s      0.794 / 0.705    0.666 / 0.565  0.245 / 0.121
    //      5 s      0.796 / 0.707    0.667 / 0.568  0.243 / 0.112
    //      4 s      0.801 / 0.710    0.666 / 0.569  0.241 / 0.116
    //      3 s      0.796 / 0.705    0.660 / 0.563  0.236 / 0.109
    //      2 s      0.785 / 0.696    0.656 / 0.560  0.229 / 0.108
    //
    // That reading does not survive scoring both settings on the *same* clips
    // and taking a bootstrap interval over the paired differences, which is the
    // only way to resolve gaps this small:
    //
    //     6 s -> 4 s      difference   95% interval        better/worse
    //     ballroom F        +0.0063   [+0.0006, +0.0120]     314 / 229
    //     ballroom CMLt     +0.0053   [-0.0014, +0.0119]     291 / 231
    //     GTZAN F           -0.0002   [-0.0053, +0.0049]     400 / 409
    //     GTZAN CMLt        +0.0030   [-0.0024, +0.0085]     354 / 317
    //     SMC F             -0.0044   [-0.0147, +0.0058]      98 / 106
    //     SMC CMLt          -0.0087   [-0.0166, -0.0010]      56 /  81
    //
    // One interval clears zero for four seconds — F on the easiest corpus, and
    // its lower bound is +0.0006. One clears zero against it, and it is on SMC,
    // the only corpus here whose tempo actually moves (68% of it varies by more
    // than 4%, against Ballroom's 5%). GTZAN, which the unpaired means put
    // ahead at four, is 400 clips better and 409 worse.
    //
    // So the trend below six is exhausted, and the honest reason is not that
    // shorter is worse everywhere but that the differences stop being
    // measurable while the one that remains measurable points the other way.
    // Four seconds would buy two seconds off the wait for the first anchor,
    // since min_window_sec follows this down, and that is a real product gain —
    // but it is being bought with a CMLt loss on tempo-varying material rather
    // than for free, and the free version was what the corpus means appeared to
    // promise.
    double window_sec = 6.0;

    // No answer before this much has been heard, which is the whole window: a
    // partially filled ring is zero-padded, and padding is silence the
    // autocorrelation would read as evidence about the tempo.
    double min_window_sec = 6.0;

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
