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
