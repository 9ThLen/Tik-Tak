#include "tracking/particle.hpp"

#include <algorithm>
#include <cmath>

namespace tiktak::tracking {
namespace {

constexpr double kLn2 = 0.69314718055994530942;
constexpr double kTwoPi = 6.283185307179586476925286766559;

std::uint64_t splitMix64(std::uint64_t& state) {
    state += 0x9E3779B97F4A7C15ull;
    std::uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

// Mean of the beat window over a uniformly distributed phase. Subtracting it is
// what makes the observation zero-mean, so integrating it once here decides the
// filter's most important property; it is done numerically because the window
// may change shape and an analytic form for each would not.
double windowMean(double sigma) {
    constexpr int kSteps = 4096;
    double sum = 0.0;
    for (int i = 0; i < kSteps; ++i) {
        const double u = 0.5 * (static_cast<double>(i) + 0.5) / kSteps;
        sum += std::exp(-0.5 * u * u / (sigma * sigma));
    }
    return sum / kSteps;
}

}  // namespace

void Rng::reseed(std::uint64_t seed) {
    std::uint64_t state = seed;
    s0_ = splitMix64(state);
    s1_ = splitMix64(state);
    if (s0_ == 0 && s1_ == 0) s1_ = 1;
}

std::uint64_t Rng::next() {
    std::uint64_t x = s0_;
    const std::uint64_t y = s1_;
    s0_ = y;
    x ^= x << 23;
    s1_ = x ^ y ^ (x >> 17) ^ (y >> 26);
    return s1_ + y;
}

void Rng::normalPair(double* a, double* b) {
    double u1 = uniform();
    if (u1 < 1e-300) u1 = 1e-300;  // log(0) is the one input Box-Muller cannot take
    const double r = std::sqrt(-2.0 * std::log(u1));
    const double theta = kTwoPi * uniform();
    *a = r * std::cos(theta);
    *b = r * std::sin(theta);
}

bool ParticleFilterConfig::valid() const {
    return particles >= 8 && min_bpm > 0.0 && max_bpm > min_bpm && prior_centre_bpm > 0.0 &&
           prior_width_octaves > 0.0 && period_drift_octaves >= 0.0 && phase_drift >= 0.0 &&
           beat_window > 0.0 && beat_window < 0.5 && observation_gain >= 0.0 &&
           onset_exponent > 0.0 && beat_gain >= 0.0 && charge_tau_sec > 0.0 && evidence_tau_sec > 0.0 &&
           roughening_octaves >= 0.0 &&
           regeneration >= 0.0 && regeneration < 1.0 && prior_rate >= 0.0 &&
           resample_ratio > 0.0 && resample_ratio <= 1.0 && max_gap_sec > 0.0;
}

BeatParticleFilter::BeatParticleFilter(const ParticleFilterConfig& config)
    : config_(config),
      rng_(config.seed),
      period_(config.particles),
      next_beat_(config.particles),
      weight_(config.particles),
      scratch_period_(config.particles),
      scratch_beat_(config.particles),
      min_period_(60.0 / config.max_bpm),
      max_period_(60.0 / config.min_bpm),
      window_mean_(windowMean(config.beat_window)) {
    drawFromPrior();
}

void BeatParticleFilter::drawFromPrior() {
    const double centre = std::log2(config_.prior_centre_bpm);
    const std::size_t n = period_.size();
    for (std::size_t i = 0; i < n; ++i) {
        double a = 0.0;
        double b = 0.0;
        rng_.normalPair(&a, &b);
        const double bpm = std::pow(2.0, centre + config_.prior_width_octaves * a);
        const double period = std::min(max_period_, std::max(min_period_, 60.0 / bpm));
        period_[i] = period;
        // Phase is uniform over the period: before the first onset arrives the
        // tracker knows nothing at all about where the beat sits.
        next_beat_[i] = last_time_sec_ + rng_.uniform() * period;
        weight_[i] = 1.0 / static_cast<double>(n);
    }
}

void BeatParticleFilter::reset() {
    started_ = false;
    last_time_sec_ = 0.0;
    // A pin survives a reset. Resetting forgets what was heard; the tempo the
    // user typed is not something that was heard.
    //
    // An anchor does not survive, for exactly the same reason read the other
    // way: it *was* heard, and it was heard in the audio being forgotten.
    anchor_bpm_ = 0.0;
    anchor_width_octaves_ = 0.0;
    mean_period_ = pinned_ ? min_period_ : 60.0 / config_.prior_centre_bpm;
    charge_ema_ = 0.0;
    onset_ema_ = 0.0;
    on_beat_ema_ = 0.0;
    window_ema_ = 0.0;
    evidence_age_sec_ = 0.0;
    coincidence_ = 0.0;
    stats_ = Stats{};
    rng_.reseed(config_.seed);
    drawFromPrior();
}

void BeatParticleFilter::seedTempo(double bpm, double spread_octaves) {
    if (!(bpm > 0.0)) return;
    const double centre = std::log2(bpm);
    const double spread = std::max(0.0, spread_octaves);
    const std::size_t n = period_.size();
    for (std::size_t i = 0; i < n; ++i) {
        double a = 0.0;
        double b = 0.0;
        rng_.normalPair(&a, &b);
        const double drawn = std::pow(2.0, centre + spread * a);
        period_[i] = std::min(max_period_, std::max(min_period_, 60.0 / drawn));
        next_beat_[i] = last_time_sec_ + rng_.uniform() * period_[i];
        weight_[i] = 1.0 / static_cast<double>(n);
    }
}

void BeatParticleFilter::pinPeriod(double period_sec) {
    if (!(period_sec > 0.0)) return;
    if (!pinned_) {
        free_min_period_ = min_period_;
        free_max_period_ = max_period_;
        pinned_ = true;
    }
    min_period_ = period_sec;
    max_period_ = period_sec;
    mean_period_ = period_sec;

    const std::size_t n = period_.size();
    for (std::size_t i = 0; i < n; ++i) {
        period_[i] = period_sec;
        // The phase is not known yet and pretending otherwise would be worse
        // than admitting it: the correlation that finds it has not run.
        next_beat_[i] = last_time_sec_ + rng_.uniform() * period_sec;
        weight_[i] = 1.0 / static_cast<double>(n);
    }
}

void BeatParticleFilter::unpinPeriod() {
    if (!pinned_) return;
    pinned_ = false;
    min_period_ = free_min_period_;
    max_period_ = free_max_period_;
    mean_period_ = 60.0 / config_.prior_centre_bpm;
    drawFromPrior();
}

void BeatParticleFilter::anchorTempo(double bpm, double width_octaves) {
    if (!(bpm > 0.0) || !(width_octaves > 0.0)) {
        anchor_bpm_ = 0.0;
        anchor_width_octaves_ = 0.0;
        return;
    }
    anchor_bpm_ = bpm;
    anchor_width_octaves_ = width_octaves;
}

void BeatParticleFilter::seedPhase(double next_beat_sec) {
    const std::size_t n = period_.size();
    for (std::size_t i = 0; i < n; ++i) {
        // Each particle lands on the grid that beat belongs to *at its own
        // period*, which pinned is the same grid for everybody and unpinned is
        // the most that can honestly be said: a beat time on its own does not
        // determine a grid without a period to repeat it at.
        const double period = period_[i];
        double beat = next_beat_sec;
        if (beat <= last_time_sec_) {
            beat += (std::floor((last_time_sec_ - beat) / period) + 1.0) * period;
        }
        next_beat_[i] = beat;
        weight_[i] = 1.0 / static_cast<double>(n);
    }
}

void BeatParticleFilter::observe(double time_sec, double onset) {
    const std::size_t n = period_.size();

    if (!started_) {
        started_ = true;
        last_time_sec_ = time_sec;
        for (std::size_t i = 0; i < n; ++i) next_beat_[i] = time_sec + rng_.uniform() * period_[i];
        return;
    }

    const double dt = time_sec - last_time_sec_;
    if (dt <= 0.0) {
        ++stats_.out_of_order;
        return;
    }
    if (dt > config_.max_gap_sec) {
        for (std::size_t i = 0; i < n; ++i) next_beat_[i] = time_sec + rng_.uniform() * period_[i];
        last_time_sec_ = time_sec;
        ++stats_.reanchors;
        return;
    }
    last_time_sec_ = time_sec;
    ++stats_.observations;

    const double walk = std::sqrt(dt);
    const double sigma = config_.beat_window;
    const double level = std::pow(std::max(0.0, onset), config_.onset_exponent);
    const double gain = config_.observation_gain * level;

    // What one predicted beat costs. The onset rate is per second, so
    // multiplying by the cloud's period converts it into "the onset energy a
    // beat is worth" — a number the reward above is measured in too. Note that
    // it is the *cloud's* period, not the particle's: a per-particle period
    // here would make the charge per second identical at every tempo and
    // discriminate nothing, which is the whole point of the term.
    //
    // Both this and the prior below are switched off while the period is
    // pinned: with one period left in the cloud neither can separate two
    // hypotheses, and a term that discriminates nothing still adds variance to
    // the phase. See pinPeriod().
    const double charge =
        pinned_ ? 0.0 : config_.beat_gain * (charge_ema_ / dt) * mean_period_;

    // The prior, applied at a rate rather than once. `prior_scale` is half the
    // inverse square of the width, so the term below is the log of a
    // log-normal density in octaves, per second of elapsed time.
    // An anchor moves that centre onto a measured tempo and narrows the width.
    // The term is otherwise unchanged, which is the point: holding a metrical
    // level is this prior aimed by evidence, not a second mechanism arguing
    // with it. Everything else stays on, so the filter can still be argued out
    // of the anchored octave — see anchorTempo().
    const bool anchored = anchor_bpm_ > 0.0;
    const double prior_centre =
        std::log2(anchored ? anchor_bpm_ : config_.prior_centre_bpm);
    const double prior_width =
        anchored ? anchor_width_octaves_ : config_.prior_width_octaves;
    const double prior_scale =
        pinned_ ? 0.0
                : config_.prior_rate * dt / (2.0 * prior_width * prior_width);

    double sum = 0.0;
    double predicted_on_beat = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double a = 0.0;
        double b = 0.0;
        rng_.normalPair(&a, &b);

        double period = period_[i] * std::exp(kLn2 * config_.period_drift_octaves * walk * a);
        period = std::min(max_period_, std::max(min_period_, period));
        double beat = next_beat_[i] + config_.phase_drift * period * walk * b;

        // Advance to the first beat strictly after this frame, in one step
        // rather than a loop: a loop would be unbounded work in the audio
        // callback exactly when the period is small and the gap large.
        double crossings = 0.0;
        if (beat <= time_sec) {
            crossings = std::floor((time_sec - beat) / period) + 1.0;
            beat += crossings * period;
        }

        period_[i] = period;
        next_beat_[i] = beat;

        const double distance = std::min(beat - time_sec, time_sec - (beat - period));
        const double u = distance / period;
        const double window = std::exp(-0.5 * u * u / (sigma * sigma));

        // Measured before the update, so it is a genuine prediction: how much
        // of this frame's onset the cloud said would be here.
        predicted_on_beat += weight_[i] * window;

        const double octaves = std::log2(60.0 / period) - prior_centre;
        const double w = weight_[i] * std::exp(gain * (window - window_mean_) -
                                               charge * crossings - prior_scale * octaves * octaves);
        weight_[i] = w;
        sum += w;
    }

    if (!(sum > 0.0) || !std::isfinite(sum)) {
        // Every hypothesis has been ruled out, which cannot be true — it means
        // arithmetic, not evidence. Flatten rather than propagate NaN.
        for (std::size_t i = 0; i < n; ++i) weight_[i] = 1.0 / static_cast<double>(n);
        return;
    }

    double sum_squares = 0.0;
    double log_period = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        weight_[i] /= sum;
        sum_squares += weight_[i] * weight_[i];
        log_period += weight_[i] * std::log(period_[i]);
    }
    mean_period_ = std::exp(log_period);

    const double charge_decay = std::exp(-dt / config_.charge_tau_sec);
    charge_ema_ = charge_decay * charge_ema_ + (1.0 - charge_decay) * level;

    const double decay = std::exp(-dt / config_.evidence_tau_sec);
    onset_ema_ = decay * onset_ema_ + (1.0 - decay) * level;
    on_beat_ema_ = decay * on_beat_ema_ + (1.0 - decay) * level * predicted_on_beat;
    window_ema_ = decay * window_ema_ + (1.0 - decay) * predicted_on_beat;
    evidence_age_sec_ += dt;
    if (evidence_age_sec_ >= config_.evidence_warmup_sec &&
        onset_ema_ > 1e-6 && window_ema_ > 1e-6 && window_ema_ < 1.0 - 1e-6) {
        // On-beat onset density against off-beat onset density, as a contrast.
        //
        // This used to be the share of onset energy landing on the prediction,
        // rescaled so chance read zero. That statistic answers "is most of the
        // energy on the beat?", and on a produced mix the answer is no for
        // every possible tracker: evaluated with the reference beats
        // themselves — tracking that cannot be improved on — its median over
        // 106 real recordings was 0.137 against a lock threshold of 0.35, so
        // the gate was unpassable by construction and the metronome stayed
        // silent on material the product exists for. A confidence gate is
        // entitled to measure the tracker; it is not entitled to fail the
        // material.
        //
        // The contrast asks the question the filter's own weights ask — is
        // there *more* onset where the beats are claimed than between them —
        // and normalising each side by its own time makes it indifferent to
        // how much off-beat activity the arrangement carries. Perfect
        // prediction on the same recordings scores a median of 0.39, the same
        // prediction pushed half a beat off scores 0.175, silence changes
        // nothing (the guard holds the last value), and a uniform level leaves
        // it at zero exactly as before.
        const double on_mean = on_beat_ema_ / window_ema_;
        const double off_mean = (onset_ema_ - on_beat_ema_) / (1.0 - window_ema_);
        const double total = on_mean + off_mean;
        if (total > 0.0) {
            coincidence_ = std::min(1.0, std::max(0.0, (on_mean - off_mean) / total));
        }
    }

    const double ess = 1.0 / sum_squares;
    if (ess < config_.resample_ratio * static_cast<double>(n)) resample();
}

void BeatParticleFilter::resample() {
    const std::size_t n = period_.size();
    const double step = 1.0 / static_cast<double>(n);

    // Systematic resampling: one uniform draw and a single sweep. Cheaper than
    // multinomial and lower variance, both of which matter more here than in a
    // batch filter because this runs in an audio callback.
    double target = rng_.uniform() * step;
    std::size_t source = 0;
    double cumulative = weight_[0];
    for (std::size_t i = 0; i < n; ++i) {
        while (target > cumulative && source + 1 < n) {
            ++source;
            cumulative += weight_[source];
        }
        scratch_period_[i] = period_[source];
        scratch_beat_[i] = next_beat_[source];
        target += step;
    }

    period_.swap(scratch_period_);
    next_beat_.swap(scratch_beat_);

    // Roughening. The sweep above produced duplicates, and duplicates are one
    // hypothesis wearing many hats; spreading them back out is what lets the
    // cloud still change its mind a minute later.
    const double spread = kLn2 * config_.roughening_octaves;
    for (std::size_t i = 0; i < n; i += 2) {
        double a = 0.0;
        double b = 0.0;
        rng_.normalPair(&a, &b);
        period_[i] = std::min(max_period_, std::max(min_period_, period_[i] * std::exp(spread * a)));
        if (i + 1 < n) {
            period_[i + 1] = std::min(max_period_,
                                      std::max(min_period_, period_[i + 1] * std::exp(spread * b)));
        }
    }

    // Regeneration: a few particles drawn afresh from the prior, so that a
    // tempo nowhere near the current belief still has somebody arguing for it.
    // They start with the same weight as everyone else and die within a beat
    // unless the audio agrees, which is what makes this cheap.
    const auto fresh = static_cast<std::size_t>(config_.regeneration * static_cast<double>(n));
    if (pinned_) {
        // Pinned, "somewhere else" is a different phase rather than a different
        // tempo, and the same argument applies: a band coming back in on a
        // different beat is not a small perturbation of the current phase, so
        // without a few particles on fresh ones the cloud could only ever creep
        // towards the truth at the rate the phase drift allows.
        for (std::size_t i = 0; i < fresh; ++i) {
            next_beat_[i] = last_time_sec_ + rng_.uniform() * min_period_;
        }
    } else {
        const double centre = std::log2(config_.prior_centre_bpm);
        for (std::size_t i = 0; i < fresh; ++i) {
            double a = 0.0;
            double b = 0.0;
            rng_.normalPair(&a, &b);
            const double bpm = std::pow(2.0, centre + config_.prior_width_octaves * a);
            period_[i] = std::min(max_period_, std::max(min_period_, 60.0 / bpm));
            next_beat_[i] = last_time_sec_ + rng_.uniform() * period_[i];
        }
    }

    for (std::size_t i = 0; i < n; ++i) weight_[i] = step;
    ++stats_.resamples;
}

BeatEstimate BeatParticleFilter::estimate(double now_sec) const {
    const std::size_t n = period_.size();

    // The answer is the dominant hypothesis, not the average of the cloud.
    // A cloud split between 100 and 200 BPM averages to 141, a tempo not one
    // particle argues for — the same mistake as taking the arithmetic mean of a
    // phase that straddles the wrap-around. So the tempo is read off the
    // heaviest region of log-period and only the particles inside it are
    // averaged; the rest, including the handful regenerated from the prior
    // every resample, contribute to how much of the cloud agrees and nothing
    // else.
    constexpr std::size_t kBins = 48;
    double bin_weight[kBins] = {};
    const double low = std::log2(min_period_);
    const double span = std::log2(max_period_) - low;
    const double scale = span > 0.0 ? static_cast<double>(kBins) / span : 0.0;

    for (std::size_t i = 0; i < n; ++i) {
        auto bin = static_cast<std::size_t>((std::log2(period_[i]) - low) * scale);
        if (bin >= kBins) bin = kBins - 1;
        bin_weight[bin] += weight_[i];
    }

    std::size_t peak = 0;
    for (std::size_t b = 1; b < kBins; ++b) {
        if (bin_weight[b] > bin_weight[peak]) peak = b;
    }
    const std::size_t first = peak > 0 ? peak - 1 : 0;
    const std::size_t last = std::min(kBins - 1, peak + 1);

    double share = 0.0;
    double log_sum = 0.0;
    double log_sq_sum = 0.0;
    double x = 0.0;
    double y = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        auto bin = static_cast<std::size_t>((std::log2(period_[i]) - low) * scale);
        if (bin >= kBins) bin = kBins - 1;
        if (bin < first || bin > last) continue;

        const double w = weight_[i];
        const double log_period = std::log(period_[i]);
        share += w;
        log_sum += w * log_period;
        log_sq_sum += w * log_period * log_period;

        // The phase mean has to be circular. Two particles predicting beats at
        // 0.99 and 0.01 of a period from now agree far more than their
        // arithmetic mean of 0.5 suggests, and on a cloud straddling the
        // wrap-around that mean is not merely imprecise: it points at the one
        // place no particle believes in.
        const double fraction = (next_beat_[i] - now_sec) / period_[i];
        const double phase = kTwoPi * (fraction - std::floor(fraction));
        x += w * std::cos(phase);
        y += w * std::sin(phase);
    }

    BeatEstimate out;
    if (share <= 0.0) {
        out.bpm = 60.0 / mean_period_;
        out.next_beat_sec = now_sec + mean_period_;
        return out;
    }

    const double mean_log = log_sum / share;
    const double period = std::exp(mean_log);
    out.bpm = 60.0 / period;

    const double variance = std::max(0.0, log_sq_sum / share - mean_log * mean_log);
    out.tempo_spread_octaves = std::sqrt(variance) / kLn2;

    const double agreement = std::sqrt(x * x + y * y) / share;

    double fraction = std::atan2(y, x) / kTwoPi;
    if (fraction < 0.0) fraction += 1.0;
    const double next_beat = now_sec + fraction * period;
    out.next_beat_sec = next_beat;

    // A particle outside the winning tempo band whose own beat grid passes
    // through the winner's next beat is agreeing about the one thing that gets
    // played, and counts for the winner rather than against it.
    //
    // This is not a courtesy. On produced music the cloud legitimately holds
    // the beat at more than one metrical level at once — measured over 106
    // real recordings the winning band held a median 33% of the cloud while
    // the phase resultant stood at 0.93, so the confidence was reporting the
    // cloud's structure as doubt about the beat. Multiple related hypotheses
    // are the filter behaving well; a half-tempo particle predicts a click on
    // this very beat, and only a particle whose grid *misses* the click is
    // evidence against it. The alignment test is in absolute time against the
    // winner's own window, so a subharmonic's sparser grid is not penalised
    // for its sparseness.
    double corroborating = 0.0;
    const double align_window = config_.beat_window * period;
    for (std::size_t i = 0; i < n; ++i) {
        auto bin = static_cast<std::size_t>((std::log2(period_[i]) - low) * scale);
        if (bin >= kBins) bin = kBins - 1;
        if (bin >= first && bin <= last) continue;   // already in the share

        const double along = next_beat - next_beat_[i];
        const double miss =
            std::fabs(along - std::round(along / period_[i]) * period_[i]);
        if (miss <= align_window) corroborating += weight_[i];
    }

    // Three things, and all of them have to hold: the beat that will be
    // played is believed across the cloud, the winning band agrees on its
    // phase, and the onsets keep arriving where the cloud says they will.
    out.confidence =
        std::min(1.0, (share + corroborating) * agreement * coincidence_);
    out.cluster_share = std::min(1.0, share + corroborating);
    out.phase_agreement = agreement;
    out.onset_coincidence = coincidence_;
    return out;
}

}  // namespace tiktak::tracking
