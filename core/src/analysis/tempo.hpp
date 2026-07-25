#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "dsp/fft.hpp"

namespace tiktak::analysis {

struct TempoConfig {
    double min_bpm = 40.0;
    double max_bpm = 220.0;
    double prior_centre_bpm = 120.0;
    // Standard deviation of the prior in octaves. Wide enough not to fight real
    // music, narrow enough to break the octave tie.
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
    // Kept as a parameter rather than deleted: comb scoring is standard in the
    // literature and this evidence is entirely synthetic. Re-measure on real
    // annotated audio before concluding it is useless in general.
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
