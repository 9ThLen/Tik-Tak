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

    // How many tempo hypotheses to actually track before choosing between
    // them. 1 keeps the old behaviour — take the estimator's first choice and
    // track it once.
    //
    // The estimator already produces ranked candidates and already applies a
    // log-normal prior to rank them, but a prior cannot see how well a grid
    // would *fit*: it only knows which tempo is a priori more plausible.
    // Measured against a Beat This! reference on 106 recordings, the right
    // tempo is nearly always somewhere in this list and simply not first —
    // an oracle picking the best of these candidates reaches CMLt 0.704
    // against 0.488 for the first choice, and 0.695 for handing the tracker
    // the reference tempo outright. The evidence to break the tie is the
    // objective the tracker maximises anyway, so it costs one extra run per
    // candidate and nothing else.
    //
    // Four rather than all eight: the tail of the candidate list is where the
    // implausible octaves live, and every extra hypothesis is another full
    // dynamic-programming pass over the file.
    int tempo_hypotheses = 4;

    // How much the estimator's ranking counts against how well the grid fits,
    // as the exponent on candidate strength in `strength^w * objective`.
    //
    // Zero would trust the fit alone, which is much worse than the current
    // behaviour rather than better: the objective is a mean over beats and a
    // half-speed grid raises it by sitting only on the loudest onsets, so fit
    // alone chases exactly the octave error the prior exists to prevent.
    //
    // 1.5 was chosen by holding out each batch of the evaluation set and
    // fitting on the rest; both batches picked the same value and both gained
    // on material they had not been fitted to, 8 and 10 points of CMLt. The
    // optimum is broad — anything from 0.75 to 2.0 lands within 3 points —
    // which is the reason to believe it at all.
    double tempo_fit_weight = 1.5;
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

    // How well the beat grid fits the onset function, per beat — see
    // BeatResult. Exposed so an evaluation can compare the grids produced at
    // different tempo hypotheses, which is what deciding between them needs
    // and what the single estimate above cannot express.
    double beat_objective_per_beat = 0.0;

    // Bar lines, a subset of `beats`. Empty when they were not asked for, the
    // track was too short for a repeated meter, or the scorer supplied too
    // little varying bar-level evidence to choose one.
    std::vector<double> downbeats;
    int beats_per_bar = 0;
    // See DownbeatResult, where all three are explained: how far the bar lines
    // stand above the beats around them, how settled the beat they start on is,
    // and how far ahead the winning meter is of the next one. All in the
    // built-in salience backend's units; see DownbeatResult.
    //
    // A UI that accents downbeats needs both margins to be convincing, not just
    // one — `downbeat_confident` below is that decision in a single place so no
    // caller has to remember it.
    double downbeat_strength = 0.0;
    double downbeat_phase_margin = 0.0;
    double downbeat_meter_margin = 0.0;

    // Whether the bar lines are worth accenting. False means count from the
    // first beat and accent nothing — putting the accent in the wrong place is
    // worse for a player than putting it nowhere.
    bool downbeat_confident = false;

    // The per-beat cues the bar-line decision was made from, kept rather than
    // discarded so the decision can be taken apart afterwards.
    //
    // Measured on real recordings, the metre comes back right and the phase
    // comes back wrong, which is a statement about these three numbers and
    // cannot be investigated from the verdict alone. Keeping them also means
    // cue weights can be swept outside the core — recombine these into a
    // salience and feed it back through the resolver — instead of needing a
    // rebuild per candidate weighting.
    //
    // Cheap: about 32 bytes a beat, so a twenty-minute track costs under a
    // hundred kilobytes against the tens of megabytes its audio occupied.
    std::vector<BeatFeature> beat_features;
};

// ------------------------------------------- what this scores on real songs
//
// Every offline number this project quoted before 2026-08-01 came from either
// thirty-second excerpts or agreement with another tracker on unannotated
// audio. The application does neither: it analyses whole songs and is right or
// wrong about them. Measured with the shipped defaults against human
// annotation, one script over all four corpora so the harness cannot differ
// between rows:
//
//     corpus      n    length    beat F   CMLt   AMLt   downbeat F   metre
//     gtzan      998    30 s      0.781  0.648  0.845     0.417      76.2%
//     ballroom   698    30 s      0.757  0.574  0.831     0.488      73.9%
//     harmonix   554   3.7 min    0.473  0.447  0.667     0.288      89.4%
//     smc        217    40 s      0.392  0.153  0.348       —          —
//
// **The Harmonix row is mostly not about this analyser, and reading it as such
// was the first mistake made with it.** That corpus is audio realigned onto the
// timeline of the Harmonix Set's own mel spectrograms, because the Set ships no
// audio. The check is a tracker that owes nothing to ours and nothing to the
// alignment: Beat This!, run out of fold, which predicts where a person taps
// rather than where the audio is loudest. On 160 of these recordings it scores
// **0.490** — a model that reaches about 0.9 on Harmonix in its own paper —
// with its beats within 70 ms of the annotation on 56.9% of them, median
// +22 ms, interquartile spread 109 ms. The same harness on GTZAN, which Beat
// This! holds out of training entirely, gives 0.865 and 97.5% within 70 ms, so
// the instrument is sound and the corpus is not.
//
// **And the corpus cannot be repaired, because the looseness is in its own
// reference.** Three measurements, none involving a tracker: the produced audio
// matches the official mel at lag zero (median +0 ms, interquartile spread
// 11 ms whole-file, 1 ms across ten-second windows); it carries no splices (the
// discontinuity rate at the reconstruction's frame grid equals the half-frame
// control); and the official mel's own onset rise, fitted against the official
// annotation, puts that annotation off by a median +8 ms with an interquartile
// spread of **115 ms** — which correlates with where Beat This! lands at
// r = +0.47, rising to +0.59 where the mel's onsets are clearest. Two
// independent instruments agree on both the centre and the spread. The
// alignment already reproduces its reference to the millisecond; the reference
// is what sits ±55 ms from the annotation, at a 46 ms mel hop that cannot
// resolve better.
//
// The cleanest single line of it: `final0` — the checkpoint trained on
// Harmonix, on these recordings, against these annotations — scores **0.501**
// here, against 0.503 for the fold checkpoints that never saw them, and 0.865
// on GTZAN which it never saw either. A model does not do worse on its own
// training data than on a corpus held out from it. It does that when the audio
// it is handed is not the audio it was trained on, and no processing of ours
// can put back a correspondence the Set does not distribute.
//
// So the row has to be split, and on the part that is verifiably aligned —
// the 57 of 160 where Beat This! clears 0.8, so the audio demonstrably carries
// the beats the annotation claims — this analyser looks entirely different:
//
//                        beat F   CMLt   AMLt   downbeat F   metre
//     all 554             0.473  0.447  0.667     0.288      89.4%
//     the aligned 57      0.765  0.702  0.851     0.493      86.0%
//     ballroom (for scale) 0.757  0.574  0.831     0.488      73.9%
//
// **Full-length pop is not harder than the excerpt corpora.** 0.765 with a 95%
// interval of [0.690, 0.834] sits on top of ballroom's 0.757 and GTZAN's 0.781,
// and its CMLt and AMLt are better than either. The same conclusion arrives
// from the other direction: cutting all 554 songs to their middle thirty
// seconds changes almost nothing (-0.019 F [-0.032, -0.006] paired), so length
// was never the variable. What differed was how much of the corpus was
// measurable at all.
//
// Two things follow, and the second is the one that costs.
//
// *The 0.473 is not a number about this code and must not be quoted as one.*
// Nor is the "44% of grids land on the beat against 94% on GTZAN" that was
// briefly written here: our grid follows the audio, the audio is displaced, and
// a displaced audio moves our grid and the flux together — which is exactly the
// r = +0.96 between the two that made the effect look real. Everything measured
// on this corpus needs the same split before it means anything.
//
// *What survives is a comparison against Beat This! on identical, verified
// audio.* On those 57 recordings it scores 0.961 [0.948, 0.974] against our
// 0.765. Two tenths of an F-measure, on material where nothing is in dispute
// about the ground truth, is the honest size of the gap between a hand-built
// onset function and a learned one — and it is the largest single number
// anywhere in this file.
//
// It is not one latency, which was the cheap explanation and was tested first.
// The best single timing shift is +30 ms on Harmonix, -30 on GTZAN, -20 on
// ballroom and 0 on SMC; held out — chosen on one corpus, spent on the others
// — every choice loses. Constants that disagree in sign are not a constant.
//
// **What octave error is left is one-sided.** Against a reference tempo taken
// as the median annotated interval — and an octave, unlike a phase, survives a
// misaligned recording, so this row does not need the split above:
//
//     corpus     annotated median   same    double   half
//     harmonix        119 BPM       75.3%   16.6%    2.3%
//     gtzan           114 BPM       73.0%   10.6%    8.7%
//     ballroom        125 BPM       66.2%    9.2%   17.6%
//
// Ballroom halves and pop doubles, seven to one. analysis/tempo.hpp records
// that re-centring the prior from 120 to 140 traded 67 halved recordings for
// 43 doubled ones on ballroom and called it close to zero sum; on full-length
// pop the trade has landed almost entirely on the doubling side. That is not
// an argument to move it back — the same file explains why a global prior can
// only choose where the crossover sits — but it does say which direction the
// remaining error points on the product's own material.
//
// **The bar phase fails on top of all of it, and fails at the half bar.**
// Taking only recordings where the grid is already right (beat F >= 0.8) and
// the metre is right, so nothing upstream is at fault, and asking how far our
// bar lines sit from the annotated ones. The filter also does the alignment
// split for free on the Harmonix row: a grid cannot reach 0.8 F against an
// annotation the audio does not match.
//
//     corpus       n     phase right   off by 1   off by 2   off by 3
//     harmonix    138       68.1%       13.0%      14.5%       4.3%
//     ballroom    336       75.0%        8.0%      12.8%       4.2%
//     gtzan       481       59.0%       11.9%      22.5%       6.7%
//
// Half a bar is the single largest failure on all three, which is the same
// shape analysis/downbeat.hpp measured on a learned activation and predicted
// from first principles: a wrong phase that repeats on exactly the period the
// evidence is accumulated over cannot be broken by accumulating more of it.
//
// **Which learned front end matters, and BeatNet is not it here.** The
// causal LiveTracker driven by BeatNet was run over the same 554 songs, block
// by block, to ask whether swapping the observation would do for this path
// what it did for the microphone path. Over the whole corpus it comes out
// slightly ahead (0.488 against 0.473, +0.014 [-0.002, +0.031] paired) — but
// on the 57 recordings whose audio is verified, the two are **0.765 and
// 0.765**. Identical. The apparent gain was the misaligned remainder, where a
// smaller model degrades more gracefully, and it is not a gain in beat
// placement.
//
// One thing does survive that comparison and is worth keeping: a causal
// tracker with no lookahead at all draws level with this one, which sees the
// whole file. Whatever the whole-file view is worth on ballroom and GTZAN, on
// full-length pop it is worth nothing measurable. That is an argument about
// where the remaining accuracy lives — in the evidence, not in the search —
// and it points at the 0.765 against 0.961 above rather than at BeatNet.
//
// **The front end that does close it is already in this repository, and this
// is what it is worth.** `dump_analysis --beat-this models/small0.onnx` runs
// the core's own ONNX session, resampler, feature extractor and peak picker —
// the shipping port, not a research approximation. Over all 998 GTZAN
// recordings, which Beat This! holds out of training entirely:
//
//     path                                beat F   CMLt   AMLt   downbeat F
//     spectral flux cues (ships)           0.781  0.648  0.845     0.417
//     Beat This! small0 through the core   0.882  0.785  0.887     0.772
//
// Paired: beat F +0.102 [+0.087, +0.117], CMLt +0.138 [+0.110, +0.166], AMLt
// +0.042 [+0.028, +0.056]. 606 recordings better against 179 worse on F.
//
// Two things that table is not. The downbeat column is not this resolver on a
// better salience — when the model supplies the grid it also supplies the bar
// lines from its own downbeat head, so 0.417 -> 0.772 is our resolver replaced
// rather than improved. And the metre column is unchanged at 76.2% in both
// rows because `beats_per_bar` is still reported from the cue analysis; the
// tool does not re-run the metre search on the model's grid. Both are honest
// gaps in the measurement, not results.
//
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

    // Runs the tracker at each leading tempo candidate and returns the grid
    // that wins on prior-weighted fit. See OfflineConfig::tempo_hypotheses.
    BeatResult trackBestHypothesis(double fps, double fallback_bpm);

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
