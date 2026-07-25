#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace tiktak::tracking {

// Deterministic uniform generator, xorshift128+.
//
// Not std::mt19937: this runs in the audio callback (a Mersenne twister's state
// is 2.5 kB and its refill is a spike every 624 draws), and the test suite pins
// tracker behaviour to exact numbers, which needs a generator that produces the
// same stream on every platform and every standard library. Both hold here.
class Rng {
public:
    explicit Rng(std::uint64_t seed) { reseed(seed); }

    void reseed(std::uint64_t seed);

    std::uint64_t next();

    // Uniform in [0, 1).
    double uniform() { return static_cast<double>(next() >> 11) * (1.0 / 9007199254740992.0); }

    // Standard normal. Box-Muller returns a pair, and the pair is what the
    // predict step wants — one for the period, one for the phase — so neither
    // half is thrown away.
    void normalPair(double* a, double* b);

private:
    std::uint64_t s0_ = 0;
    std::uint64_t s1_ = 0;
};

struct ParticleFilterConfig {
    std::size_t particles = 512;

    // The tempo range and prior are the online half of the same belief the
    // offline estimator applies: autocorrelation and a particle cloud both peak
    // just as happily at half and double the true period, and the log-normal
    // prior over tempo is what breaks that tie.
    double min_bpm = 40.0;
    double max_bpm = 220.0;
    double prior_centre_bpm = 120.0;
    double prior_width_octaves = 0.7;

    // How fast the prior is applied, in nats per second at one width from the
    // centre. Unlike the offline estimator, which weighs the prior against the
    // whole autocorrelation once, an online filter has to keep applying it:
    // draw it only at initialisation and it is forgotten within a second, and
    // the tracker then has nothing to say that a beat every 500 ms is a more
    // musical reading of an accented pattern than a beat every second.
    //
    // This is the octave dial, and it cannot be set to please everyone: raise
    // it and a genuinely slow piece gets tracked at double, lower it and an
    // accented one gets tracked at half. The UI's x2 / ÷2 buttons exist
    // because no value of this is right for every song.
    double prior_rate = 0.8;

    // Random walk on the period, in octaves per square-root second, and on the
    // phase, as a fraction of the period per square-root second. Brownian
    // scaling (sqrt of elapsed time) is deliberate: it makes the motion model
    // independent of the ODF hop size, so changing the frame rate does not
    // silently retune the tracker.
    double period_drift_octaves = 0.012;
    double phase_drift = 0.006;

    // Width of the observation window around a particle's beat, as a fraction
    // of its period. Roughly "how late may an onset be and still count as on
    // the beat" — 0.05 of a 500 ms beat is 25 ms.
    double beat_window = 0.05;

    // Gain on the onset strength in the log-likelihood. Higher locks on faster
    // and is more easily fooled by a syncopated bar.
    //
    // This and beat_gain are calibrated in units where a beat's onset arrives
    // as about 1.0 and silence as 0 — see LiveConfig::onset_peak_tau_sec, which
    // is what makes that true whatever the room's level.
    double observation_gain = 9.0;

    // Exponent applied to the normalised onset before it is weighed as
    // evidence. This is what separates a beat from its own subdivisions, and it
    // is not a detail: nearly all music has hits between the beats, and with
    // evidence proportional to amplitude a hi-hat at a third the level buys a
    // third of a beat's worth of belief. Two hypotheses then sit within noise
    // of each other — the beat, and the subdivision that also covers every hit
    // — and the tracker flickers between them. Squaring makes that hi-hat worth
    // a ninth instead, which is the difference between "quieter" and "not a
    // beat".
    double onset_exponent = 2.0;

    // What a particle pays for each beat it predicts. Without this term nothing
    // at all opposes double tempo: a particle beating twice as fast is right on
    // every real onset and is never charged for the beats it places in the
    // gaps, because a frame with no onset moves no weight. This is the
    // point-process half of the likelihood — the integral term that says a
    // prediction which did not happen is evidence too.
    //
    // The charge is scaled by how much onset energy a beat is currently worth
    // (the running onset rate times the cloud's own period), which is what
    // keeps it comparable to the reward and makes silence free: with nothing to
    // be right about there is nothing to pay either, so a quiet passage does
    // not slowly drag the tempo down.
    double beat_gain = 3.0;

    // Time constant of the running onset rate the charge is scaled by. It has
    // to be short: while it is still decaying, a silence is charging particles
    // for beats nobody could have been right about, and the drag is not even —
    // a fast hypothesis crosses more often and pays more, so a long tail here
    // makes the tracker slow down every time the music stops for a bar.
    double charge_tau_sec = 0.75;

    // Time constant of the coincidence measure that reports confidence. Longer,
    // because confidence is a claim about the last few seconds and should not
    // flicker with one loud snare.
    double evidence_tau_sec = 2.0;

    // Spread added to the period of every resampled particle, in octaves.
    // Resampling copies survivors, and copies are not hypotheses: without
    // roughening a cloud that has agreed once can never change its mind, which
    // shows up as a tracker that locks onto the wrong octave in the first
    // second and stays there through a whole song.
    double roughening_octaves = 0.01;

    // Fraction of the cloud redrawn from the prior at each resample. Roughening
    // lets the cloud move; this lets it jump. A new song at a different tempo,
    // or a phone picked up mid-set, is not a small perturbation of the current
    // hypothesis, and without a trickle of fresh particles the filter can only
    // ever refine the answer it settled on in its first second.
    double regeneration = 0.06;

    // Resample when the effective sample size falls below this fraction of the
    // cloud.
    double resample_ratio = 0.5;

    // A gap longer than this means the stream stopped rather than the music
    // pausing — a device restart, a suspended app. Winding every particle
    // forward one period at a time across such a gap is both slow and
    // meaningless, so the cloud is re-anchored instead.
    double max_gap_sec = 1.0;

    std::uint64_t seed = 0x9E3779B97F4A7C15ull;

    bool valid() const;
};

struct BeatEstimate {
    double bpm = 0.0;
    // The predicted time of the next beat. A prediction, not a detection: the
    // click has to be written into a buffer before the beat is heard, so a
    // tracker that only announced beats it had already seen would be useless
    // however accurate it was.
    double next_beat_sec = 0.0;
    // 0..1. Two things have to be true for a tracker to be trusted, and this is
    // their product: the cloud agrees on where the beat is (its resultant
    // length in phase), *and* the onsets actually keep landing there (the share
    // of onset energy that falls on the predicted beat, above chance).
    //
    // Agreement alone is not enough and would be a lie: resampling makes a
    // cloud agree with itself within a second or two of white noise, so a
    // filter reporting only its own concentration reports near-certainty on
    // material that has no beat in it.
    double confidence = 0.0;
    // Spread of the cloud in tempo, octaves. Large means the period itself is
    // still undecided, which is a different failure from a lost phase.
    double tempo_spread_octaves = 0.0;
};

// Online beat tracking by particle filter. State per particle: a period and the
// time of its next beat.
//
// Why not the dynamic programming of the offline path: DP needs the whole
// signal before it decides anything. Online has to decide ahead of the beat, so
// the question changes from "where were the beats" to "where is the next one",
// and a filter that carries a distribution answers it while a single-hypothesis
// tracker cannot say how sure it is.
//
// The observation is deliberately zero-mean: an onset lands on the weight of a
// particle as `onset * (window(distance to its beat) - mean window)`, so an
// onset on a particle's beat raises its weight, an onset between its beats
// lowers it, and a frame with no onset at all leaves every weight untouched.
// Silence therefore costs nothing but diffusion — the cloud coasts through a
// quiet passage at its last tempo and spreads slowly instead of collapsing onto
// whatever noise arrives first. It is also what makes dropping frames safe,
// which the microphone path relies on when it gates out its own click.
//
// Real-time safe: every buffer is sized in the constructor and observe()
// allocates nothing.
class BeatParticleFilter {
public:
    explicit BeatParticleFilter(const ParticleFilterConfig& config);

    const ParticleFilterConfig& config() const { return config_; }

    // Re-draws the cloud from the prior and forgets the stream clock.
    void reset();

    // Concentrates the cloud around a known tempo, phases still unknown. This
    // is how the offline estimate hands over to the live tracker, and how
    // manual mode fixes the period and asks only for the phase.
    void seedTempo(double bpm, double spread_octaves = 0.05);

    // One ODF frame: `onset` is expected already normalised for level (see
    // tracking::LiveTracker), non-negative, order of magnitude 1.
    void observe(double time_sec, double onset);

    BeatEstimate estimate(double now_sec) const;

    struct Stats {
        std::size_t observations = 0;
        std::size_t resamples = 0;
        std::size_t reanchors = 0;   // gaps in the stream, cloud re-anchored
        std::size_t out_of_order = 0;  // frames older than the last one, ignored
    };

    Stats stats() const { return stats_; }

private:
    void drawFromPrior();
    void resample();

    ParticleFilterConfig config_;
    Rng rng_;

    std::vector<double> period_;
    std::vector<double> next_beat_;
    std::vector<double> weight_;
    std::vector<double> scratch_period_;
    std::vector<double> scratch_beat_;

    double min_period_ = 0.0;
    double max_period_ = 0.0;
    double window_mean_ = 0.0;  // mean of the beat window over uniform phase

    double last_time_sec_ = 0.0;
    bool started_ = false;

    double mean_period_ = 0.5;   // the cloud's period, for the beat charge
    double charge_ema_ = 0.0;    // running onset, fast, scales the beat charge
    double onset_ema_ = 0.0;     // running onset per frame
    double on_beat_ema_ = 0.0;   // running onset that landed on the prediction
    double coincidence_ = 0.0;   // the two above, as a share above chance

    Stats stats_;
};

}  // namespace tiktak::tracking
