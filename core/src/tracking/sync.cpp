#include "tracking/sync.hpp"

#include <algorithm>
#include <cmath>

namespace tiktak::tracking {
namespace {

constexpr double kTwoPi = 6.283185307179586476925286766559;

}  // namespace

bool SyncConfig::valid() const {
    return tau_sec > 0.0 && onset_exponent > 0.0 && acquire_strength > 0.0 &&
           acquire_strength <= 1.0 && release_strength >= 0.0 &&
           release_strength < acquire_strength && min_rayleigh > 0.0 && hold_sec >= 0.0 &&
           max_drift > 0.0 && max_drift < 0.5 && max_gap_sec > 0.0;
}

void PhaseSync::forget() {
    x_ = 0.0;
    y_ = 0.0;
    mass_ = 0.0;
    mass_sq_ = 0.0;
    held_sec_ = 0.0;
    ready_ = false;
}

void PhaseSync::setPeriod(double period_sec) {
    if (period_sec == period_sec_) return;
    period_sec_ = period_sec > 0.0 ? period_sec : 0.0;
    forget();
}

void PhaseSync::reset() {
    forget();
    last_time_sec_ = 0.0;
    started_ = false;
}

void PhaseSync::observe(double time_sec, double onset) {
    if (!(period_sec_ > 0.0)) return;

    if (!started_) {
        started_ = true;
        last_time_sec_ = time_sec;
        return;
    }

    const double dt = time_sec - last_time_sec_;
    if (dt <= 0.0) return;
    if (dt > config_.max_gap_sec) {
        // The stream stopped rather than the music pausing. Decaying the
        // accumulator across the gap would be arithmetic on nothing; the phase
        // on the other side of a suspended app has no relation to the one
        // before it anyway.
        forget();
        last_time_sec_ = time_sec;
        return;
    }
    last_time_sec_ = time_sec;

    const double level = std::pow(std::max(0.0, onset), config_.onset_exponent);
    const double decay = std::exp(-dt / config_.tau_sec);

    // Where this frame sits within the period, as an angle. Reducing the time
    // before multiplying keeps the argument small however long the session has
    // been running.
    const double turns = time_sec / period_sec_;
    const double angle = kTwoPi * (turns - std::floor(turns));

    x_ = decay * x_ + level * std::cos(angle);
    y_ = decay * y_ + level * std::sin(angle);
    mass_ = decay * mass_ + level;
    mass_sq_ = decay * mass_sq_ + level * level;

    // Acquiring needs both: concentrated enough to be worth acting on, and
    // concentrated enough for the amount of evidence behind it to rule out
    // chance. Releasing needs only the first, so that hysteresis is a single
    // threshold and a passage that thins out does not switch the click off.
    const double resultant = strength();
    if (resultant >= config_.acquire_strength &&
        resultant * resultant * evidence() >= config_.min_rayleigh) {
        held_sec_ += dt;
        if (held_sec_ >= config_.hold_sec) ready_ = true;
    } else {
        held_sec_ = 0.0;
        if (resultant < config_.release_strength) ready_ = false;
    }
}

double PhaseSync::strength() const {
    if (!(mass_ > 1e-9)) return 0.0;
    return std::min(1.0, std::sqrt(x_ * x_ + y_ * y_) / mass_);
}

double PhaseSync::evidence() const {
    if (!(mass_sq_ > 1e-18)) return 0.0;
    // The participation ratio of the accumulator's weights: equal weights give
    // their count, and one dominant weight gives one however many there are.
    return mass_ * mass_ / mass_sq_;
}

double PhaseSync::nextBeat(double now_sec) const {
    if (!(period_sec_ > 0.0)) return now_sec;

    double phase = std::atan2(y_, x_) / kTwoPi;
    if (phase < 0.0) phase += 1.0;

    // Beats sit at `offset + k * period`. The one wanted is the first strictly
    // after now, which is one step past the last one at or before it.
    const double offset = phase * period_sec_;
    const double step = std::floor((now_sec - offset) / period_sec_) + 1.0;
    return offset + step * period_sec_;
}

}  // namespace tiktak::tracking
