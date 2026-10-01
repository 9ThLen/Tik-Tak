#include "render/click_canceller.hpp"

#include <algorithm>
#include <cmath>

namespace tiktak::render {

namespace {

// This many silent samples in what was played and the next sound is a new
// click. A click's own zero crossings are single samples at most.
constexpr std::size_t kQuietRun = 64;

// The largest block either call is expected to be handed. The ring keeps this
// much more than the path needs on both sides, so the two streams can run a
// buffer apart in either direction.
constexpr std::int64_t kLargestBlock = 65536;

// How much of the last clicks' predicted return goes into the running figure.
constexpr double kReturnedUpdate = 0.5;

// The room's usual level is not asked about until this many clicks have set it.
constexpr std::size_t kUsualClicks = 3;

// A room that stays quiet for this many clicks is no longer emptying: it is a
// quieter room, and its level becomes the usual one.
constexpr std::size_t kAloneRun = 40;

std::int64_t nextPowerOfTwo(std::int64_t value) {
    std::int64_t out = 1;
    while (out < value) out <<= 1;
    return out;
}

double decibels(double numerator, double denominator) {
    if (!(numerator > 0.0) || !(denominator > 0.0)) return 0.0;
    return 10.0 * std::log10(numerator / denominator);
}

}  // namespace

bool ClickCancellerConfig::valid() const {
    if (!(sample_rate > 0.0) || !(round_trip_sec >= 0.0)) return false;
    if (!(before_sec >= 0.0) || !(dense_sec >= 0.0) || !(after_sec >= dense_sec)) return false;
    if (!(tap_spacing_sec > 0.0)) return false;
    if (!(update > 0.0) || !(update <= 1.0) || !(ridge > 0.0)) return false;
    if (!(quietest > 0.0) || !(level_sec > 0.0)) return false;
    if (!(alone_drop > 0.0) || !(alone_memory > 0.0) || !(alone_memory <= 1.0)) return false;
    if (!(alone_left_ratio > 0.0)) return false;
    // The normal matrix is factored once, densely; this keeps that to a few
    // megabytes and well under a second.
    const double taps = (before_sec + dense_sec) * sample_rate +
                        (after_sec - dense_sec) / tap_spacing_sec;
    return taps <= 2048.0;
}

ClickCanceller::ClickCanceller(const ClickCancellerConfig& config, const ClickConfig& click)
    : config_(config) {
    if (!config.valid() || !click.valid()) return;
    if (std::fabs(click.sample_rate - config.sample_rate) > 1e-6) return;

    const double rate = config.sample_rate;
    before_ = std::llround(config.before_sec * rate);
    const std::int64_t dense = before_ + std::llround(config.dense_sec * rate);
    const std::int64_t last = before_ + std::llround(config.after_sec * rate);
    const std::int64_t spacing =
        std::max<std::int64_t>(1, std::llround(config.tap_spacing_sec * rate));
    for (std::int64_t lag = 0; lag <= dense; ++lag) lags_.push_back(lag);
    for (std::int64_t lag = dense + spacing; lag <= last; lag += spacing) lags_.push_back(lag);
    const std::size_t taps = lags_.size();
    offset_ = std::llround(config.round_trip_sec * rate) - before_;

    // The click, from the renderer that will play it, so that nothing here has
    // to agree with its formula by hand.
    ClickRenderer renderer(click);
    std::vector<float> tone(
        static_cast<std::size_t>(std::ceil(click.beat.length_sec * rate)) + 8, 0.0f);
    renderer.schedule(0.0, schedule::BeatKind::Beat);
    renderer.mix(0.0, tone.data(), tone.size());
    std::size_t length = tone.size();
    while (length > 0 && tone[length - 1] == 0.0f) --length;
    if (length == 0) return;   // a silent click: nothing to remove

    // Every click excites the path the same way, so the normal matrix is the
    // click's own autocorrelation at the differences between tap delays, known
    // before anything is played, and it is factored here, off the audio thread.
    std::vector<double> lagged(static_cast<std::size_t>(lags_.back()) + 1, 0.0);
    for (std::size_t lag = 0; lag < lagged.size() && lag < length; ++lag) {
        double sum = 0.0;
        for (std::size_t n = 0; n + lag < length; ++n) {
            sum += static_cast<double>(tone[n]) * static_cast<double>(tone[n + lag]);
        }
        lagged[lag] = sum;
    }
    if (!(lagged[0] > 0.0)) return;

    factor_.assign(taps * taps, 0.0);
    for (std::size_t i = 0; i < taps; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            double value = lagged[static_cast<std::size_t>(lags_[i] - lags_[j])];
            if (i == j) value += config.ridge * lagged[0];
            for (std::size_t k = 0; k < j; ++k) {
                value -= factor_[i * taps + k] * factor_[j * taps + k];
            }
            if (i == j) {
                if (!(value > 0.0)) return;   // not positive definite: stay off
                factor_[i * taps + i] = std::sqrt(value);
            } else {
                factor_[i * taps + j] = value / factor_[j * taps + j];
            }
        }
    }

    window_ = static_cast<std::int64_t>(length) + lags_.back();
    click_energy_ = lagged[0];
    level_step_ = 1.0 / std::max(1.0, config.level_sec * rate);
    span_.assign(taps, 0.0);
    mean_.assign(taps, 0.0);
    theta_.assign(taps, 0.0);

    const std::int64_t reach = std::max<std::int64_t>(0, offset_) + lags_.back();
    const std::int64_t size = nextPowerOfTwo(reach + 2 * kLargestBlock);
    ring_.assign(static_cast<std::size_t>(size), 0.0f);
    mask_ = size - 1;
    quiet_run_ = kQuietRun;
    enabled_ = true;
}

void ClickCanceller::reset() {
    if (!enabled_) return;
    std::fill(ring_.begin(), ring_.end(), 0.0f);
    std::fill(span_.begin(), span_.end(), 0.0);
    std::fill(mean_.begin(), mean_.end(), 0.0);
    std::fill(theta_.begin(), theta_.end(), 0.0);
    have_played_ = false;
    played_end_ = 0;
    quiet_run_ = kQuietRun;
    starts_head_ = 0;
    starts_count_ = 0;
    active_ = false;
    dirty_ = false;
    level_ = 0.0;
    level_seen_ = 0;
    under_click_ = 0.0;
    weight_sum_ = 0.0;
    returned_energy_ = 0.0;
    last_left_ = 0.0;
    usual_level_ = 0.0;
    usual_seen_ = 0;
    alone_run_ = 0;
    stats_ = Stats{};
}

bool ClickCanceller::alone() const {
    if (!enabled_ || stats_.clicks == 0 || usual_seen_ < kUsualClicks) return false;
    // Not known yet is not empty: before the level has settled, say nothing.
    if (static_cast<double>(level_seen_) * level_step_ < 2.0) return false;
    if (!(level_ < config_.alone_drop * usual_level_)) return false;
    return last_left_ < config_.alone_left_ratio * returned_energy_;
}

float ClickCanceller::played_at(std::int64_t index) const {
    if (index >= played_end_ || index < played_end_ - (mask_ + 1)) return 0.0f;
    return ring_[static_cast<std::size_t>(index & mask_)];
}

void ClickCanceller::played(std::int64_t start_sample, const float* click, std::size_t frames) {
    if (!enabled_ || click == nullptr || frames == 0) return;

    if (!have_played_) {
        played_end_ = start_sample;
        have_played_ = true;
    }
    if (start_sample != played_end_) {
        const std::int64_t gap = start_sample - played_end_;
        if (gap > 0 && gap <= mask_) {
            // The device skipped ahead: nothing was played in between.
            for (std::int64_t i = played_end_; i < start_sample; ++i) {
                ring_[static_cast<std::size_t>(i & mask_)] = 0.0f;
            }
            quiet_run_ = kQuietRun;
        } else {
            // Backwards, or further than the ring remembers. What was played
            // can no longer be lined up with what is heard, so the clicks in
            // flight are let go; the path itself is still good.
            std::fill(ring_.begin(), ring_.end(), 0.0f);
            stats_.skipped += starts_count_ + (active_ ? 1 : 0);
            starts_count_ = 0;
            active_ = false;
            quiet_run_ = kQuietRun;
        }
        played_end_ = start_sample;
    }

    for (std::size_t i = 0; i < frames; ++i) {
        const float value = click[i];
        const std::int64_t index = played_end_ + static_cast<std::int64_t>(i);
        ring_[static_cast<std::size_t>(index & mask_)] = value;
        if (value != 0.0f) {
            if (quiet_run_ >= kQuietRun) {
                if (starts_count_ < kPending) {
                    starts_[(starts_head_ + starts_count_) % kPending] = index;
                    ++starts_count_;
                } else {
                    ++stats_.skipped;
                }
            }
            quiet_run_ = 0;
        } else if (quiet_run_ < kQuietRun) {
            ++quiet_run_;
        }
    }
    played_end_ += static_cast<std::int64_t>(frames);
}

void ClickCanceller::heard(std::int64_t start_sample, const float* in, float* out,
                           std::size_t frames) {
    if (in == nullptr || out == nullptr || frames == 0) return;
    if (!enabled_) {
        if (out != in) std::copy(in, in + frames, out);
        return;
    }

    const std::size_t taps = lags_.size();
    for (std::size_t i = 0; i < frames; ++i) {
        const std::int64_t n = start_sample + static_cast<std::int64_t>(i);
        const double y = static_cast<double>(in[i]);

        // The next click's window opens when its first sample can reach us.
        while (!active_ && starts_count_ > 0) {
            const std::int64_t first = starts_[starts_head_] + offset_;
            const std::int64_t last = first + window_ - 1;
            if (n < first) break;
            starts_head_ = (starts_head_ + 1) % kPending;
            --starts_count_;
            if (n > last) {
                ++stats_.skipped;   // the capture stream jumped clean over it
                continue;
            }
            active_ = true;
            // Joined late, and part of the click has already gone by unheard.
            dirty_ = n > first;
            window_last_ = last;
            // A level followed for less than two of its own time constants is
            // not a level yet. Negative says so, and finish() falls back on the
            // window's own energy, which errs towards trusting the click less.
            under_click_ = static_cast<double>(level_seen_) * level_step_ >= 2.0 ? level_ : -1.0;
            std::fill(span_.begin(), span_.end(), 0.0);
            heard_energy_ = left_energy_ = predicted_energy_ = 0.0;
        }
        if (!active_) {
            // The room by itself, which is what a click will have to be found
            // under. Followed only between clicks, so that it is never the
            // click's own level.
            level_ += level_step_ * (y * y - level_);
            if (level_seen_ < (std::size_t{1} << 30)) ++level_seen_;
            out[i] = in[i];
            continue;
        }

        // The newest sample the first tap needs has to have been played by
        // now. If it has not, the round trip is shorter than a buffer and this
        // click cannot be predicted or learned from.
        const std::int64_t newest = n - offset_;
        if (newest >= played_end_) dirty_ = true;

        double predicted = 0.0;
        for (std::size_t j = 0; j < taps; ++j) {
            const double x = static_cast<double>(played_at(newest - lags_[j]));
            if (x == 0.0) continue;
            predicted += theta_[j] * x;
            span_[j] += y * x;
        }
        const double left = y - predicted;
        out[i] = static_cast<float>(left);
        heard_energy_ += y * y;
        left_energy_ += left * left;
        predicted_energy_ += predicted * predicted;

        if (n >= window_last_) {
            if (starts_count_ > 0 && starts_[starts_head_] + offset_ <= window_last_) {
                // The next click began inside this one's reach. Both are still
                // removed, but neither can be learned from: their correlations
                // are tangled together.
                window_last_ = starts_[starts_head_] + offset_ + window_ - 1;
                starts_head_ = (starts_head_ + 1) % kPending;
                --starts_count_;
                dirty_ = true;
            } else {
                finish();
            }
        }
    }
}

void ClickCanceller::finish() {
    active_ = false;
    last_left_ = left_energy_;
    stats_.predicted_db = decibels(predicted_energy_, heard_energy_);
    stats_.removed_db = decibels(heard_energy_, left_energy_);

    // What the room is usually like just before a click. A click heard in a
    // room that has just emptied does not move it, or the emptiness would
    // become the usual and stop being noticed; one that stays that quiet for
    // long enough is a quieter piece of music, and does.
    if (under_click_ >= 0.0) {
        const bool emptied = usual_seen_ >= kUsualClicks &&
                             under_click_ < config_.alone_drop * usual_level_;
        alone_run_ = emptied ? alone_run_ + 1 : 0;
        if (!emptied || alone_run_ > kAloneRun) {
            usual_level_ = usual_seen_ == 0
                               ? under_click_
                               : usual_level_ + config_.alone_memory * (under_click_ - usual_level_);
            ++usual_seen_;
        }
    }
    if (dirty_) {
        ++stats_.skipped;
        return;
    }

    // Averaging the correlation, not the path, is what lets the music cancel:
    // it enters each click's correlation with a different phase. Each click is
    // weighed by how quiet the room was under it, and the weights already
    // gathered fade by `update` a click. In a steady room that is a plain
    // exponential average and the first click is the whole estimate, while a
    // click in a pause takes the estimate over at once and keeps it when the
    // music returns.
    ++stats_.clicks;
    const double noise =
        under_click_ >= 0.0 ? under_click_ * static_cast<double>(window_) : heard_energy_;
    const double weight = 1.0 / (noise + config_.quietest * click_energy_);
    weight_sum_ = (1.0 - config_.update) * weight_sum_ + weight;
    const double gain = weight / weight_sum_;
    const std::size_t taps = lags_.size();
    for (std::size_t j = 0; j < taps; ++j) {
        mean_[j] += gain * (span_[j] - mean_[j]);
    }

    // Two triangular solves against the factor made in the constructor.
    for (std::size_t i = 0; i < taps; ++i) {
        double value = mean_[i];
        for (std::size_t k = 0; k < i; ++k) value -= factor_[i * taps + k] * theta_[k];
        theta_[i] = value / factor_[i * taps + i];
    }
    for (std::size_t i = taps; i-- > 0;) {
        double value = theta_[i];
        for (std::size_t k = i + 1; k < taps; ++k) value -= factor_[k * taps + i] * theta_[k];
        theta_[i] = value / factor_[i * taps + i];
    }

    // What the click comes back with, by the new path: the normal equations
    // give it without another pass over the click.
    double returned = 0.0;
    std::size_t strongest = 0;
    for (std::size_t j = 0; j < taps; ++j) {
        returned += theta_[j] * mean_[j];
        if (std::fabs(theta_[j]) > std::fabs(theta_[strongest])) strongest = j;
    }
    returned = std::max(0.0, returned);
    returned_energy_ = stats_.clicks == 1
                           ? returned
                           : returned_energy_ + kReturnedUpdate * (returned - returned_energy_);
    stats_.arrival_sec =
        static_cast<double>(lags_[strongest] - before_) / config_.sample_rate;
}

}  // namespace tiktak::render
