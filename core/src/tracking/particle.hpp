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
    // 150, and the causal path wants it higher than the offline path's 140.
    // Same mechanism as there — a log-normal prior centred at c prefers t/2
    // over t above c*sqrt(2) — but a much larger effect here, because the
    // particle cloud carries the prior continuously rather than applying it
    // once to a posterior. This was the biggest single number found in the
    // live path. Gate already at 0.15 / 0.02, BeatNet driving the observation:
    //
    //              GTZAN (999)              Ballroom (698)
    //     centre   F      CMLt  beats/ref   F      CMLt  beats/ref
    //     120      0.522  0.375    0.65     0.587  0.427    0.66
    //     140      0.551  0.407    0.74     0.622  0.473    0.73
    //     150      0.560  0.413    0.79     0.636  0.484    0.77
    //     180      0.566  0.391    0.93     0.665  0.502    0.89
    //     200      0.565  0.357    1.01     0.667  0.482    0.98
    //
    // Chosen on CMLt rather than F, and the two disagree past 150 for a reason
    // worth stating: F keeps creeping up while CMLt falls and beats/ref climbs
    // through 1.0, which is the tracker drifting into double time. The extra
    // beats hit enough annotated ones to flatter F and are on the wrong
    // metrical level. 150 is GTZAN's CMLt peak and within 0.02 of Ballroom's,
    // so it is the one point that is not a corpus's preference.
    double prior_centre_bpm = 150.0;
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
    //
    // 3.0, not 2.0. The argument above had the direction right and the amount
    // short. Measured on GTZAN with BeatNet driving the observation, one knob
    // at a time from what shipped:
    //
    //     onset_exponent    F      CMLt
    //     1.0               0.412  0.270
    //     2.0               0.536  0.400
    //     3.0               0.610  0.479
    //     5.0               0.657  0.528
    //
    // At a third the level a hi-hat buys a ninth of a beat's belief at 2.0 and
    // a 27th at 3.0, which is the difference between "quieter" and "not a beat"
    // on material that actually has hi-hats.
    //
    // The corpus wanted 5.0 and a test refused it. Above 3.0,
    // LiveMetronome.ClicksOnTheBeatsItHearsInTheRoom fails — bisected, 3.0
    // passes and 3.5 does not — because sharpening the evidence that far starts
    // costing the period its discipline, and the click then walks off a clean
    // 120 BPM grid a few milliseconds further every beat. A 70 ms matching
    // tolerance hides that on a corpus. A user playing along with it cannot.
    double onset_exponent = 3.0;

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
    //
    // 1.5, not 3.0. The charge was set high enough to be doing the opposite job
    // as well — suppressing beats the tracker had correctly found, not only the
    // ones it invented. Measured with everything else at what shipped:
    //
    //     beat_gain   GTZAN F   CMLt   beats per annotated beat
    //     0.5           0.676  0.482            0.97
    //     1.0           0.671  0.504            0.92
    //     1.5           0.661  0.510            0.87
    //     2.0           0.635  0.500            0.83
    //     3.0           0.536  0.400            0.72
    //     6.0           0.292  0.111            0.62
    //
    // It still has to oppose double tempo, and 6.0 shows the cliff on the other
    // side is real and near.
    //
    // **Left at 3.0 all the same**, and the corpus does not get to decide this
    // one. Lowering it to 1.5 is worth another 0.043 F and 0.012 CMLt on GTZAN
    // on top of the exponent below, and it costs the tempo its discipline:
    // ClicksOnTheBeatsItHearsInTheRoom and FollowsAnEncodedClickTrackThrough-
    // TheMicrophonePath both fail, with the click drifting off a clean 120 BPM
    // grid by a further 4.3 ms every beat — a 0.9% tempo error that a 70 ms
    // matching tolerance hides on real music and a metronome cannot hide from a
    // user at all. This is the term that holds the period; halving it buys
    // recall by letting the period wander, which is the wrong trade for a
    // product whose whole job is to click in time.
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

    // How much audio the coincidence measure has to have seen before it says
    // anything. An exponential average a fraction of its time constant old is
    // mostly its first few samples, and the ratio of two such averages is
    // noise pretending to be a measurement: fed white noise, the first
    // reported second reached 0.64 — nearly twice the lock threshold — and
    // decayed to a steady 0.00 by the sixth. Confidence has no business
    // reporting before its own estimator has converged.
    double evidence_warmup_sec = 4.0;

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

    // The three factors confidence is the product of, reported separately so a
    // low number can be diagnosed instead of guessed at: which one is limiting
    // decides whether the fix is tempo-side, phase-side, or in the evidence —
    // three different pieces of work. Guessing got it wrong once already.
    double cluster_share = 0.0;      // weight of the winning tempo cluster
    double phase_agreement = 0.0;    // its resultant length in phase
    double onset_coincidence = 0.0;  // on-beat against off-beat onset contrast
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

    // Fixes the period and stops the filter arguing about it — manual mode,
    // where the tempo is the user's and the only question left is the phase.
    // The given period is taken as it stands, outside the configured range if
    // that is what was asked for: the range is a belief about what tempo music
    // is likely to be at, and it has no business overruling a number somebody
    // typed.
    //
    // Several of the filter's terms fall away here, and that is the point
    // rather than an oversight. The tempo prior is a claim about which period
    // is more musical, and with one period left it makes no claim. The per-beat
    // charge exists to stop the cloud running to double tempo, and every
    // particle now predicts beats at the same rate, so it can separate nothing
    // — it would only add noise to the phase, which is the one quantity still
    // being estimated. Both are dropped while pinned; the period itself is held
    // by the clamp, which has nowhere left to clamp to.
    void pinPeriod(double period_sec);
    void unpinPeriod();
    bool pinned() const { return pinned_; }

    // Puts the cloud's next beat on a known grid, periods untouched: this is
    // tracking::PhaseSync handing over the offset it correlated out.
    void seedPhase(double next_beat_sec);

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

    // Manual mode. While pinned the two bounds above are the same number, and
    // these hold the range to go back to.
    bool pinned_ = false;
    double free_min_period_ = 0.0;
    double free_max_period_ = 0.0;

    double last_time_sec_ = 0.0;
    bool started_ = false;

    double mean_period_ = 0.5;   // the cloud's period, for the beat charge
    double charge_ema_ = 0.0;    // running onset, fast, scales the beat charge
    double onset_ema_ = 0.0;     // running onset per frame
    double on_beat_ema_ = 0.0;   // running onset that landed on the prediction
    double window_ema_ = 0.0;    // running mass of the prediction window itself
    double evidence_age_sec_ = 0.0;  // how long the EMAs above have been fed
    double coincidence_ = 0.0;   // on-beat against off-beat onset, as a contrast

    Stats stats_;
};

}  // namespace tiktak::tracking
