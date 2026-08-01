#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <optional>

#include "dsp/odf.hpp"
#include "ml/beatnet.hpp"
#include "tracking/activation_tempo.hpp"
#include "tracking/particle.hpp"
#include "tracking/sync.hpp"

namespace tiktak::tracking {

struct LiveConfig {
    dsp::OdfConfig odf;
    ParticleFilterConfig filter;
    SyncConfig sync;  // manual mode only: finding the phase at a known tempo

    // Half-life of the peak the onset function is normalised against. The
    // filter's constants are calibrated in units where a beat is worth about
    // one, so the front-end has to deliver that whatever the room's level —
    // hence dividing by the loudest onset of the last few seconds rather than
    // by a standard deviation. A z-score would also be level-invariant, but its
    // scale moves with the *material*: dense music raises the running mean and
    // shrinks every peak towards it, so the same beat arrives as weaker
    // evidence purely because there is more going on around it.
    double onset_peak_tau_sec = 3.0;

    // How long our own click is treated as blinding the microphone, relative to
    // the moment it is heard.
    double gate_before_sec = 0.005;
    double gate_after_sec = 0.050;

    // Hysteresis on publishing beats. Above `lock_confidence` the tracker
    // follows the cloud; below `release_confidence` it stops handing out beats
    // altogether; in between it coasts at the last tempo it was sure of, which
    // is what a musician does when the band drops out for a bar.
    //
    // 0.25 / 0.02, not the 0.35 / 0.15 this carried before. The old band was
    // chosen without a public corpus behind it, and measured against one it was
    // withholding most of the track: driven by BeatNet it emitted 0.48 beats
    // for every annotated one on GTZAN and said nothing at all on 2.1% of
    // recordings. Both corpora, prior held at what it shipped:
    //
    //     lock/release    GTZAN F   CMLt   beats/ref   silent
    //     0.45 / 0.15       0.408  0.322      0.42       8.0%
    //     0.35 / 0.15       0.486  0.362      0.48       2.1%
    //     0.25 / 0.02       0.513  0.376      0.60       0.9%
    //     0.15 / 0.02       0.522  0.375      0.65       0.2%
    //
    // The sweep wanted to keep going, and it is stopped at 0.25 by a test
    // rather than by a preference. Below it, ADipInConfidenceDoesNotLetTwo-
    // BeatsOutTogether fails: at 0.20 the tracker publishes two beats 0.163 s
    // apart on a half-second pulse, which is the stutter that test was written
    // for after it had already shipped once. Bisected, and the boundary is the
    // lock alone: 0.25 passes and 0.20 does not, at every release from 0.10
    // down to 0.01. An F-measure has no column for "clicked twice", so the
    // invariant decides here and the corpus does not get a vote.
    //
    // Giving up that last 0.01 F costs nothing anyway. With the prior re-aimed
    // as well, 0.25 beats 0.15 on CMLt on both corpora, 0.428 against 0.413 and
    // 0.495 against 0.484. The safe point is also the better one.
    //
    // What the band cannot do is improve the beats it lets through: CMLt moves
    // 0.014 across the whole sweep at a fixed prior. The withheld beats were not
    // being kept from a tracker that knew where they were, so opening the gate
    // lets out more of the same. Where they land is the prior's job, and see
    // ParticleFilterConfig::prior_centre_bpm for it: that is the larger of the
    // two changes by some way.
    double lock_confidence = 0.25;
    double release_confidence = 0.02;

    // Tempo measured from the activation history, used to aim the filter's
    // prior at the metrical level the recording is actually in.
    ActivationTempoConfig activation_tempo;

    // On. Over 698 ballroom and 999 GTZAN recordings, against free running:
    //
    //                   ballroom            GTZAN
    //     free       0.700  0.584        0.632  0.508
    //     anchored   0.794  0.705        0.666  0.565
    //     hard pin   0.778  0.782        0.697  0.637
    //     the answer 0.881  0.873        0.737  0.790
    //
    // On ballroom that is better than holding the period outright, which is
    // the thing the corpora were expected to prefer and the thing that cannot
    // ship. It is not a compromise reached against them: at a six-second
    // window the anchor is both better on the corpus and quicker to follow a
    // tempo change than any longer-window setting, so nothing was traded away
    // to get it. See ActivationTempoConfig::window_sec.
    bool anchor_tempo = true;

    // How much room the anchored tempo is left, in octaves. A fifth of an
    // octave is about 15% either way, so half and double sit four widths out
    // and are outvoted, while a singer drifting a few percent is followed
    // rather than fought.
    //
    // 0.10 against 0.20, at the six-second window both were measured on:
    //
    //     width   ballroom F   GTZAN F   worst lag over six tempo changes
    //     0.10      0.794       0.666           6.8 s
    //     0.20      0.760       0.659           5.6 s
    //
    // A point and a bit of latency for three and a half points of F. Note how
    // much smaller the choice is than it was at a thirty-second window, where
    // the same widths spanned 23.1 seconds of worst-case lag: once the window
    // is short the width stops being the thing that matters, which is the
    // right way round, because the width is a belief and the window is
    // evidence.
    double anchor_width_octaves = 0.1;

    // How decided the estimator has to be before its answer is used, as the
    // gap to the best rival at another metrical level.
    //
    // Zero, meaning "use it whenever there is one", and that is a measured
    // result rather than an omission. Gating looked obviously right — a
    // half-certain octave is exactly what one would not want to hold — and on
    // 120 ballroom recordings it is worse at every setting tried: F 0.752 with
    // no gate, 0.738 at 0.15, 0.714 at 0.30. The reason it does not help is
    // that a tie in the estimator is not a coin toss downstream. Both rivals
    // are metrical relatives of each other, so anchoring the wrong one still
    // puts the filter on a grid the right beats fall on, whereas refusing to
    // anchor leaves it with the fixed prior, which is worse than either.
    //
    // Kept as a parameter rather than deleted because the question is a real
    // one and the answer may not survive a corpus that is not ballroom.
    double anchor_octave_margin = 0.0;

    // A stream time this far from where the sample count says it should be
    // means the device dropped or repeated a buffer.
    double discontinuity_tolerance_sec = 0.002;

    bool valid() const;
};

// ------------------------------- what this path is worth to a person using it
//
// An average F-measure cannot answer the question the product asks. Two
// recordings with the same CMLt are not the same experience: one takes four
// seconds to start clicking and then holds, the other starts at once and jumps
// an octave twice in the middle. So the live benchmark also scores each
// recording pass or fail — starts within 8 s, at least 80% of the beats it
// emits land within 70 ms, *finds* at least 80% of the beats there were, and
// never spends more than 4 s at the wrong metrical level — and reports the
// share that passes. 1914 recordings, shipped thresholds:
//
//     front end          ballroom*   GTZAN   SMC*   median acquire
//     spectral flux         5.7%     13.4%   1.4%       6 s
//     BeatNet activation   57.2%     41.1%   3.2%       5 s
//
// The third criterion is matched recall and not the ratio of the two beat
// counts, which is what it was first written as. A tracker can emit exactly as
// many beats as there were and put all of them somewhere else: precision 0.80
// with a count ratio of 0.80 bounds the beats actually found at 64%. The
// figures above are after that correction and are about a point lower for it.
//
// **Quote GTZAN.** The weights are `beatnet_model_1`, and BeatNet ships three
// models holding out GTZAN, Ballroom and Rock Corpus respectively — see
// docs/ml-models.md, which records the correspondence. Model 1 holds out
// GTZAN, so ballroom is train-on-test for it and its 57.4% is memorisation,
// exactly as final0's ballroom score is for Beat This!. SMC's membership is
// not documented either way and is marked with it. An earlier revision of this
// comment averaged all three into "34.4% usable" and that number should not be
// repeated: the honest headline is **42.6% on the one corpus this model has
// certainly not seen**.
//
// **Acquisition is not the problem.** It was, and the diagnosis chain in
// research/eval/README.md is about it; it is fixed. The median is five seconds
// and slow acquisition is a listed failure on only 39% of SMC and almost
// nothing elsewhere. What fails now is precision: 51% of GTZAN emits beats
// that are more than a fifth wrong.
//
// **Where the metrical level comes from.** Of the tracking seconds the filter
// spends at the wrong level, 85% of GTZAN are seconds where the anchor was
// also wrong; read the other way round, when the anchor is wrong the filter is
// wrong with it 92% of the time. Both conditionals are worth stating and only
// the second is the causal one. Three ways of improving the anchor from what
// is already computed were tried and all lost (below).
//
// That does not establish that the filter is blameless, and an earlier
// revision of this comment said so on the strength of the filter agreeing with
// a correct anchor 94% of the time. That figure cannot carry the claim.
// `anchorTempo` is applied on every submitted frame — fifty a second with
// BeatNet, see LiveTracker::submit — with a prior a tenth of an octave wide,
// so the agreement is largely enforced rather than observed; only the estimate
// behind it is refreshed once a second. And 15.4% of GTZAN's wrong seconds
// happen while the anchor is right. Separating the two needs the filter run
// *without* the anchor and against an oracle level; until that is done, no
// statement here apportions blame between the estimator and the filter.
//
// **The level is not where the recordings are lost.** Scored per *recording*
// rather than per second, with the grid read at half (both phases) or twice
// its rate and judged at whichever agrees best — an oracle correction applied
// to the whole recording, and therefore an upper bound on any control the
// player could be given, while also removing the wrong-level criterion
// outright rather than modelling one press:
//
//     usable        ballroom*   GTZAN   SMC*
//     as it stands     57.2%    41.1%   3.2%
//     any level        59.0%    46.0%   4.6%
//
// Five points on the corpus that counts, and of the GTZAN recordings that fail
// today only 8.3% become usable at another level. A ×2 control in the product
// would recover little.
//
// What is left on GTZAN once the level is forgiven: **too few beats found on
// 52.6%**, wrong beats on 43.6%, slow acquisition on 9.2%. The largest single
// failure is recall — the tracker is not putting beats where the beats are —
// and that is what the count-ratio version of the criterion was hiding.
//
// Note what this does *not* say. A grid at the right level but the wrong local
// tempo, or one that drifts, arrives as both of those failures too, so nothing
// here separates placement from tempo. It says the level is not the way out,
// and no more than that.
//
// **What the published comparison is, and is not.** BeatNet's paper reports
// 0.754 beat F on GTZAN and BeatNet+ 0.806, both full systems — activation
// plus their two-level cascade particle filter, not activations alone. The
// same BeatNet activation through this tracker gives 0.666. That is a
// published score against a local one, measured by different code on
// different framing, and it is a *lead*, not a decoder gap: nothing here has
// yet run their filter on our activations. The A/B that would make it a
// measurement is the next piece of work.
//
// Reproducing any of this needs research/, which is not in the public
// repository — see the .gitignore and the note in eval/README.md. That is why
// the numbers live here in full rather than as a file reference.
//
// Material without percussion is a separate case and probably a real one: on
// SMC this path is usable on 3.2% of recordings with 95% of failures being
// wrong beats. That is *consistent with* the front end being the limit there
// and does not demonstrate it — the same symptom follows from a decoder that
// cannot hold a sparse pulse, and only an A/B on identical activations tells
// them apart. It is the material BeatNet+ claims to address, which makes it
// worth the A/B rather than worth assuming.
//
// **Listening before answering has no headroom, and the ceiling says so.**
// The proposal is reasonable and recurs: not every song has an audible beat in
// its first bar, so buffer some seconds of microphone and orient on them
// before committing. Its ceiling can be measured without building any of it,
// by handing the tracker the tempo an offline analysis of the *whole* file
// found — strictly more than any buffer could recover:
//
//     seeded with the whole file's tempo   ballroom*   GTZAN   SMC*
//     no                                     57.4%     42.6%   3.2%
//     yes                                    56.6%     44.0%   2.8%
//
// A point and a half on GTZAN, and it moves in different directions on
// different corpora. The same experiment on spectral flux was already
// negative; this replicates it with the front end that works, which is what
// makes it worth believing rather than a property of the old evidence.
//
// The reason is `anchor_tempo`, and it is structural rather than a shortfall.
// The activation-tempo estimator re-aims the prior from a six-second window,
// and applies it every frame, so any tempo put into the cloud at the start is
// gone within six seconds of audio whether it was right or not. Seeding a
// tracker that continuously re-anchors cannot do anything by construction —
// which is worth knowing before building a buffer to do it more slowly.
//
// What the ceiling does not test is the *phase*: seedTempo concentrates the
// cloud on a tempo and says nothing about where the beat falls. That half is
// still open. But note which failures are actually left — wrong octave on 27%
// of ballroom and 38% of GTZAN, *after* being told the tempo, and slightly
// worse than without. The octave is not failing for want of knowing the tempo.
// It is failing because the tracker leaves the tempo it was given, which puts
// this back with everything else in this comment: the decoder.

// A live configuration for a capture rate, with the front-end sized in
// milliseconds rather than samples.
//
// The ODF's defaults are 2048/512 *samples*, which at 48 kHz is a 43 ms window
// every 11 ms and at 22 kHz is 93 ms every 23 ms — the same numbers describing
// a front-end twice as coarse. The filter is tuned against how much onset
// energy a beat is worth per frame, so the coarser front-end quietly halves the
// evidence per beat while the charge per predicted beat stays put, and the
// tracker that was steady at 48 kHz wanders at 22. Scaling both with the rate
// keeps the tracker's world the same whatever the device hands it.
LiveConfig liveConfigFor(double sample_rate);

// The microphone path, whole: audio in, beat predictions out.
//
// This is the online counterpart of analysis::OfflineAnalyzer and is composed
// here rather than in each shell for the same reason render::Metronome is —
// the shells differ in how they obtain a buffer and a clock, and must not
// differ in what happens between them.
//
// Two things it owns that a shell would otherwise have to reinvent:
//
// *Level normalisation.* The particle filter's observation gain is a constant,
// so what it multiplies must not depend on how loud the room is.
//
// *Own-click gating.* A metronome listening through a microphone hears its own
// click, and a click is the most onset-like sound there is. Left alone the
// tracker locks onto itself: confidence goes to one, the tempo stops responding
// to the music, and nothing about the output looks wrong. The click cannot be
// subtracted — the room's response to it is unknown — so instead the tracker
// declines to look during the window it occupies. That costs information but
// does not bias the filter, because the observation is zero-mean: a dropped
// frame changes no weights at all, while a frame merely ignored by a
// single-hypothesis tracker would still shift its estimate.
//
// Real-time safe: process() allocates nothing and reads no clock.
class LiveTracker {
public:
    explicit LiveTracker(const LiveConfig& config);

    // With a learned front end instead of spectral flux.
    //
    // `weights` must be valid() and must outlive the tracker; the shell owns
    // the bytes, because the core does no I/O. Everything else about the
    // tracker is the same object it was — the same filter, the same gating, the
    // same publishing thresholds — and only the evidence differs. That is not
    // a coincidence of the implementation, it is the point: it is what makes
    // the measured before and after comparable.
    //
    // Measured on 107 produced recordings against reference beats, the causal
    // tracker's accuracy and coverage improve materially without changing its
    // thresholds. See ml/beatnet.hpp.
    //
    // Not the default. Spectral flux costs a few hundred kFLOP a second and
    // this costs tens of MFLOP plus 1.6 MB of weights, and which of those a
    // given device should spend is a decision that needs measurements from that
    // device, not from a workstation.
    LiveTracker(const LiveConfig& config, const ml::BeatNetWeights& weights);

    const LiveConfig& config() const { return config_; }

    // True when the learned front end is the one feeding the filter.
    bool usingModel() const { return model_.has_value(); }

    // Feeds captured audio. `stream_time_sec` is the time of samples[0], in the
    // same clock the shell schedules output in.
    void process(double stream_time_sec, const float* samples, std::size_t n);

    // Feeds one already-computed observation instead of audio, at a time in
    // the same clock: how much this instant looks like a beat, 0 to 1.
    //
    // This is the seam a learned front end arrives through, and it is here
    // rather than in the research harness because that is where the front end
    // is going. The built-in onset function is spectral flux, and measured
    // against a reference on 106 produced recordings it does not concentrate
    // on the beat: the filter's own coincidence term sat at 0.226 where
    // perfect tracking of that same evidence could only have reached 0.39, so
    // the gate stayed shut on the material the product exists for. Fed a
    // causal model's activation instead — same filter, same recordings — that
    // term reaches 0.535 and the lock rate goes from 1% of tracks to 45%.
    //
    // Everything downstream is unchanged and deliberately so: gating, level
    // normalisation and the publishing hysteresis are the tracker's, and only
    // the evidence is swapped. Callers use one of process() or observe(), not
    // both — mixing them feeds the filter two clocks and two scales.
    void observe(double time_sec, double activation);

    // Tells the tracker when its own click will reach the microphone — that is
    // the moment the click is *heard*, output latency and room delay already
    // added by the caller. The core cannot compute it: only the shell knows
    // what the round trip measured.
    void gateClick(double heard_time_sec);

    BeatEstimate estimate(double now_sec) const { return filter_.estimate(now_sec); }

    // What the autocorrelation over the activation history currently makes of
    // the tempo, whether or not it is being used. Reported separately from
    // estimate() because it answers a different question — which metrical
    // level the recording is in, rather than where the next beat falls — and
    // because a bench that cannot see both cannot tell which of the two is
    // wrong when the beats are.
    ActivationTempoEstimate tempoFromActivation() const {
        return activation_tempo_.estimate();
    }

    // Hands out the next beat to play, once, when it comes within
    // `lookahead_sec` of now. True when `beat_sec` was written.
    //
    // A beat, once handed out, is never revised: by then the click is in a
    // buffer on its way to the device, and moving it would be a click that
    // stutters rather than a click that corrects. Refinements land on the beat
    // after it.
    bool takeBeat(double now_sec, double lookahead_sec, double* beat_sec);

    // Concentrates the cloud on a known tempo — an offline analysis of the same
    // song, or a tempo the user typed.
    void seedTempo(double bpm, double spread_octaves = 0.05);

    // Manual mode: the tempo is the user's and the room is asked only where the
    // beat falls. Zero goes back to tracking the tempo too.
    //
    // This is a different promise from auto mode, and the difference is worth
    // being explicit about, because it is what the mode is for:
    //
    // - Nothing is played until the room has been heard. The user sets a tempo
    //   and starts; the click waits, catches the first phrase, and falls in on
    //   it. That waiting is the feature — a metronome that starts on the beat
    //   the user's own count-in landed on needs no count-in of its own.
    //
    // - Once it has fallen in, it does not stop. In auto mode a room that goes
    //   quiet has taken the tempo with it, so the tracker coasts and eventually
    //   gives up; here the tempo was never the room's to take. The click keeps
    //   the user's BPM through a silent bar, a solo, a cough, indefinitely.
    //   That falls out of the filter rather than being special-cased: with the
    //   period pinned and the observation zero-mean, silence moves no weights
    //   and the grid simply continues.
    //
    // The tempo is taken as given even outside the configured BPM range — see
    // BeatParticleFilter::pinPeriod.
    void setManualTempo(double bpm);
    double manualTempo() const { return manual_bpm_; }

    // Manual mode, still waiting for something to synchronise to. What a shell
    // shows as "listening…", and the reason no beats are coming out.
    bool waiting() const { return manual_bpm_ > 0.0 && !acquired_; }

    // How concentrated the room's onsets are at one phase, 0..1. Manual mode
    // only, and useful mainly as the meter behind that "listening…".
    double syncStrength() const { return sync_.strength(); }

    void reset();

    struct Stats {
        std::size_t frames = 0;          // ODF frames produced
        std::size_t gated = 0;           // frames withheld, our own click
        std::size_t beats = 0;           // beats handed out
        std::size_t beats_late = 0;      // predicted beats already in the past
        std::size_t discontinuities = 0; // capture buffers that did not follow
        BeatParticleFilter::Stats filter;
    };

    Stats stats() const;

private:
    // The filter's beat window, widened if the ODF is too coarse to support the
    // configured one. See the definition.
    static ParticleFilterConfig resolveFilter(const LiveConfig& config);

    bool gatedAt(double frame_time_sec) const;

    // Feeds one already-normalised observation to the filter and, in manual
    // mode, to the phase correlator. What process() and observe() share once
    // each has produced a number the filter can use.
    void submit(double time_sec, double normalised);

    LiveConfig config_;
    dsp::Odf odf_;
    BeatParticleFilter filter_;
    PhaseSync sync_;
    ActivationTempo activation_tempo_;

    // Engaged only by the constructor that was handed weights. Held by value
    // rather than behind a pointer so that the audio path has no indirection
    // and no chance of a null to check.
    std::optional<ml::BeatNetActivation> model_;

    // Half the width of one evidence window, for gating. The two front ends
    // disagree about it — the ODF's frame is the configured one, the model's is
    // a fixed 64 ms — and a gate measured against the wrong one either lets the
    // click through or blinds the tracker either side of it.
    double evidence_half_sec_ = 0.0;

    double manual_bpm_ = 0.0;
    bool acquired_ = false;

    double origin_sec_ = 0.0;  // stream time of the ODF's sample zero
    std::size_t consumed_ = 0;
    bool started_ = false;

    double onset_peak_ = 0.0;

    // A handful of pending gates is plenty: they are consumed within a beat of
    // being added, and a shell that has queued eight of them is not running.
    static constexpr std::size_t kGates = 8;
    std::array<double, kGates> gate_start_{};
    std::array<double, kGates> gate_end_{};
    std::size_t gate_next_ = 0;

    bool locked_ = false;
    bool published_ = false;
    // Far enough back that the first beat of a stream is never mistaken for a
    // repeat of one that was never handed out.
    double last_beat_sec_ = -std::numeric_limits<double>::infinity();
    double held_period_sec_ = 0.5;

    Stats stats_;
};

}  // namespace tiktak::tracking
