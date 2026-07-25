#include "schedule/scheduler.hpp"

#include <algorithm>
#include <cmath>

namespace tiktak::schedule {
namespace {

// If the host stalls — the app was suspended, the audio thread was starved —
// the grid can fall arbitrarily far behind. Rather than walking every missed
// step one at a time, jump forward once past this much lost time.
constexpr double kCatchUpThresholdSec = 1.0;

}  // namespace

bool SchedulerConfig::valid() const {
    if (!(bpm > 0.0) || !(bpm < 100000.0)) return false;
    if (beats_per_bar < 1) return false;
    if (subdivisions < 1) return false;
    if (!(lookahead_sec >= 0.0)) return false;
    for (double latency : latency_sec) {
        if (!(latency >= 0.0) || !(latency < 10.0)) return false;
    }
    return true;
}

Scheduler::Scheduler(const SchedulerConfig& config) : config_(config) {
    step_duration_ = 60.0 / (config_.bpm * config_.subdivisions);

    max_latency_ = 0.0;
    enabled_count_ = 0;
    for (int i = 0; i < kChannelCount; ++i) {
        if (!config_.channel_enabled[i]) continue;
        ++enabled_count_;
        max_latency_ = std::max(max_latency_, config_.latency_sec[i]);
    }
}

void Scheduler::start(double now_sec) {
    anchor_time_ = now_sec;
    anchor_step_ = 0;
    next_step_ = 0;
    last_emitted_step_ = -1;
    late_count_ = 0;
    running_ = true;
}

void Scheduler::stop() { running_ = false; }

double Scheduler::step_time(std::int64_t step) const {
    return anchor_time_ + static_cast<double>(step - anchor_step_) * step_duration_;
}

void Scheduler::set_tempo(double bpm) {
    if (!(bpm > 0.0)) return;

    // Pin the grid to the last committed event before changing its rate, so
    // that beat keeps the time it was already given.
    const std::int64_t pivot = last_emitted_step_ >= 0 ? last_emitted_step_ : next_step_;
    const double pivot_time = step_time(pivot);

    config_.bpm = bpm;
    step_duration_ = 60.0 / (bpm * config_.subdivisions);
    anchor_step_ = pivot;
    anchor_time_ = pivot_time;
}

void Scheduler::align_to(double beat_time_sec, double now_sec) {
    if (!running_) return;

    // Shift the whole grid to the nearest beat-aligned step, so the correction
    // is the smallest one that achieves the alignment rather than a jump of a
    // whole bar.
    const double beat_duration = step_duration_ * config_.subdivisions;
    const double offset = beat_time_sec - step_time(0);
    const double beats = std::round(offset / beat_duration);

    anchor_time_ = beat_time_sec - beats * beat_duration;
    anchor_step_ = 0;

    // The shift may have moved unemitted steps into the past, or behind a beat
    // already handed to a device. Skip forward past both. Those skips are not
    // counted as late: a deliberate re-phase is not an overload, and reporting
    // it as one would have the host widen its lookahead for no reason.
    (void)catch_up(now_sec);
    while (last_emitted_step_ >= 0 && step_time(next_step_) <= step_time(last_emitted_step_)) {
        ++next_step_;
    }
}

std::size_t Scheduler::catch_up(double now_sec) {
    const double behind = now_sec - max_latency_ - step_time(next_step_);
    if (behind <= kCatchUpThresholdSec) return 0;

    const auto skipped = static_cast<std::int64_t>(behind / step_duration_);
    next_step_ += skipped;
    return static_cast<std::size_t>(skipped) * enabled_count_;
}

Event Scheduler::make_event(std::int64_t step, Channel channel, double time_sec,
                            double beat_time_sec) const {
    const auto subdivisions = static_cast<std::int64_t>(config_.subdivisions);
    const auto beats_per_bar = static_cast<std::int64_t>(config_.beats_per_bar);

    const std::int64_t beat_index = step / subdivisions;
    const int subdivision = static_cast<int>(step % subdivisions);
    const int beat_in_bar = static_cast<int>(beat_index % beats_per_bar);

    Event event;
    event.time_sec = time_sec;
    event.beat_time_sec = beat_time_sec;
    event.channel = channel;
    event.step = step;
    event.bar = beat_index / beats_per_bar;
    event.beat_in_bar = beat_in_bar;
    event.subdivision = subdivision;
    event.kind = subdivision != 0    ? BeatKind::Subdivision
                 : beat_in_bar == 0  ? BeatKind::Downbeat
                                     : BeatKind::Beat;
    return event;
}

std::size_t Scheduler::pull(double now_sec, Event* out, std::size_t capacity,
                            std::size_t* dropped_late) {
    std::size_t late = 0;
    std::size_t written = 0;

    if (running_ && out != nullptr && capacity > 0 && enabled_count_ > 0) {
        late += catch_up(now_sec);
        const double horizon = now_sec + config_.lookahead_sec;

        while (true) {
            const double beat_time = step_time(next_step_);
            // A step becomes due when its *earliest* channel needs handing
            // over, which is the one with the largest latency to absorb.
            if (beat_time - max_latency_ > horizon) break;

            // Emit a step's channels together or not at all, so a caller with a
            // small buffer never sees half a beat.
            if (written + enabled_count_ > capacity) break;

            for (int i = 0; i < kChannelCount; ++i) {
                if (!config_.channel_enabled[i]) continue;

                const double time_sec = beat_time - config_.latency_sec[i];
                if (time_sec < now_sec) {
                    ++late;
                    continue;
                }
                out[written++] = make_event(next_step_, static_cast<Channel>(i), time_sec,
                                            beat_time);
            }

            last_emitted_step_ = next_step_;
            ++next_step_;
        }
    }

    late_count_ += late;
    if (dropped_late != nullptr) *dropped_late = late;
    return written;
}

}  // namespace tiktak::schedule
