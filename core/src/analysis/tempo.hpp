#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "dsp/fft.hpp"

namespace tiktak::analysis {

struct TempoConfig {
    double min_bpm = 40.0;
    double max_bpm = 220.0;
    // 140, not the 120 this carried before, and the difference was measured
    // rather than chosen. Taken alone a log-normal prior centred at c prefers
    // t/2 over t once t > c*sqrt(2) — the two are equidistant in log2 when
    // log2(t/c) = 1/2 — so the pull starts at 170 BPM for a centre of 120 and
    // at 198 for 140. The posterior is that prior times the autocorrelation, so
    // the crossover a real signal actually shows sits a little lower: on
    // synthesised clicks it moved from below 176 to between 176 and 185. Either
    // way it moved up, out of the range real music uses. That single fact
    // was the tracker's largest failure mode: over 698 ballroom recordings it
    // landed on exactly half the annotated tempo on 186 of them, and the 120
    // prior independently prefers the half on 184 of the same 698. Cause, not
    // correlation.
    //
    // Re-centring was validated on a corpus it was not tuned on. Ballroom is
    // fast dance music (median 124 BPM, p90 200); GTZAN is not (median 114):
    //
    //             ballroom (698)              GTZAN (999, held out)
    //     centre  F      CMLt   octave        F      CMLt   octave
    //     120     0.746  0.553  18.8%         0.769  0.628  14.8%
    //     140     0.763  0.579  17.0%         0.782  0.649  14.6%
    //     150     0.772  0.583  16.2%         0.779  0.619  17.5%
    //
    // 140 is the only point that improves both. 150 buys more on ballroom and
    // gives it back on GTZAN, which is what fitting one corpus looks like.
    //
    // Say plainly what this does not do. Counting ballroom tracks by the ratio
    // between the estimate and the annotation, the move is close to zero sum:
    //
    //     est/ref     120    140
    //     1/2         186    119     -67
    //     1x          423    439     +16
    //     2x           11     54     +43
    //
    // Two thirds of the half errors it removes come back as double errors on
    // slower material, and the net is sixteen recordings out of 698. The octave
    // failure survives at 17%. That is the ceiling of the instrument, not of the
    // tuning: a global prior can only choose where the half/double crossover
    // sits, never tell fast music from slow. Moving the rest needs evidence
    // about the metre, not a better prior.
    double prior_centre_bpm = 140.0;
    // Standard deviation of the prior in octaves. Wide enough not to fight real
    // music, narrow enough to break the octave tie.
    //
    // Left at 0.7. Narrowing to 0.6 was the best point on ballroom (octave
    // 15.6%, F 0.768) and lost on GTZAN (CMLt 0.633 against 0.628 for the
    // default but octave 15.1% against 14.6%), so it is corpus fitting and was
    // dropped. Widening is worse everywhere: at 1.5 CMLt falls to 0.390 and at
    // 3.0 to 0.142 — the prior is carrying real weight, it was merely aimed
    // wrong.
    double prior_width_octaves = 0.7;
    int grid_size = 512;

    // A candidate period can be scored by a comb: its own autocorrelation plus
    // that of its multiples, on the theory that a real beat period is supported
    // at every metrical level above it while a spurious peak is not.
    //
    // Off by default (1 = score each period by its own lag alone), because
    // measurement disagreed with that theory. Over 140 synthetic clips spanning
    // 60-196 BPM the comb was worse on every metric:
    //
    //     harmonics=1   F 0.900   CMLt 0.702   AMLt 0.991   non-metrical 0/140
    //     harmonics=3   F 0.889   CMLt 0.681   AMLt 0.977   non-metrical 2/140
    //     harmonics=4   F 0.880   CMLt 0.660   AMLt 0.970   non-metrical 4/140
    //
    // and it introduced the very error it was meant to remove: a candidate at
    // two-thirds of the true period collects full support from every third
    // multiple. Restricting the comb to powers of two does not rescue it.
    //
    // Re-measured on real annotated audio, as that asked for — 250 ballroom
    // recordings with their published beats, everything else at defaults:
    //
    //     harmonics=1   F 0.759   CMLt 0.578   AMLt 0.849   octave 17.6%
    //     harmonics=2   F 0.764   CMLt 0.585   AMLt 0.862   octave 18.0%
    //     harmonics=4   F 0.763   CMLt 0.588   AMLt 0.870   octave 18.8%
    //     harmonics=6   F 0.763   CMLt 0.587   AMLt 0.869   octave 18.4%
    //
    // The synthetic verdict does not replicate — the comb is not worse on real
    // audio — but neither does it earn its place: every metric moves by less
    // than a point, and the octave rate, the one thing it was built to fix,
    // moves the wrong way. Weight decay makes no difference either (0.5 and 2.0
    // both land within 0.005 of 1.0 on every column).
    //
    // Still a parameter rather than a deletion, but the open question is now
    // closed on both kinds of material: combing does not decide the octave.
    // What moved it was the prior's centre, above.
    int comb_harmonics = 1;
    // Weight of harmonic k is k^-comb_weight_decay. Higher metrical levels
    // carry real but weaker evidence, so they should count for less.
    double comb_weight_decay = 1.0;

    bool valid() const;
};

struct TempoEstimate {
    double bpm = 0.0;
    // 0..1: how strongly the onset function actually repeats at `bpm`, as a
    // fraction of its own variance. 0 means "no periodicity found", which is
    // not the same as "120 BPM" — the UI is meant to show the difference.
    double confidence = 0.0;
};

struct TempoCandidate {
    double bpm = 0.0;
    double strength = 0.0;
};

// Tempo estimation from an onset detection function.
//
// Autocorrelation of the ODF, combed over metrical multiples and weighted by a
// log-normal prior over tempo. The prior is not cosmetic: autocorrelation peaks
// just as happily at half and double the true period, and without it the
// estimate flips octaves between neighbouring windows on perfectly ordinary
// music. That is the single most common failure mode of beat trackers.
//
// The whole posterior is kept, not just its peak, because the caller needs to
// know when two tempi are nearly tied — that is the difference between
// "confident" and "guessing".
//
// Offline component: unlike the ODF and the scheduler this is *not* real-time
// safe. It sizes its transform to the input, so a length it has not seen before
// allocates.
class TempoEstimator {
public:
    explicit TempoEstimator(const TempoConfig& config);

    const TempoConfig& config() const { return config_; }

    // Estimates over `n` ODF frames sampled at `fps` frames per second.
    TempoEstimate estimate(const double* odf, std::size_t n, double fps);

    // The candidate grid, log-spaced from min_bpm to max_bpm. Constant.
    const std::vector<double>& bpmGrid() const { return grid_; }
    // Posterior from the last estimate(), peak-normalised to 1.
    const std::vector<double>& posterior() const { return posterior_; }
    // Unbiased autocorrelation from the last estimate(), one value per lag.
    const std::vector<double>& autocorrelation() const { return acf_; }

    // Distinct peaks of the last posterior, strongest first. Used to spot
    // octave ambiguity: if the runner-up sits at half or double the winner with
    // a similar score, the estimate is a coin toss and the caller should say so
    // rather than commit. Returns how many were written.
    std::size_t topCandidates(TempoCandidate* out, std::size_t count,
                              double min_separation_octaves = 0.2) const;

private:
    void computeAutocorrelation(const double* values, std::size_t n);
    void computeCombScore(double fps);

    TempoConfig config_;
    std::vector<double> grid_;       // candidate tempi, BPM
    std::vector<double> prior_;      // log-normal weight per candidate
    std::vector<double> acf_;
    std::vector<double> posterior_;
    std::vector<double> scratch_re_;
    std::vector<double> scratch_im_;
    std::unique_ptr<dsp::Fft64> fft_;
};

}  // namespace tiktak::analysis
