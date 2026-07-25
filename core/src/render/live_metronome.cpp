#include "render/live_metronome.hpp"

namespace tiktak::render {

bool LiveMetronomeConfig::valid() const {
    return tracker.valid() && click.valid() && round_trip_sec >= 0.0 && lookahead_sec >= 0.0;
}

bool LiveMetronome::Stats::clean() const {
    return beats_late == 0 && clicks_late == 0 && clicks_overflowed == 0 && voices_stolen == 0 &&
           discontinuities == 0 && capture_discontinuities == 0;
}

LiveMetronome::LiveMetronome(const LiveMetronomeConfig& config)
    : config_(config), tracker_(config.tracker), click_(config.click) {}

void LiveMetronome::start() { running_ = true; }

void LiveMetronome::stop() { running_ = false; }

void LiveMetronome::silence() {
    running_ = false;
    click_.reset();
}

void LiveMetronome::capture(double stream_time_sec, const float* samples, std::size_t n) {
    tracker_.process(stream_time_sec, samples, n);
}

void LiveMetronome::process(double stream_time_sec, float* out, std::size_t frames) {
    if (out == nullptr) return;

    if (running_) {
        const double buffer_sec = static_cast<double>(frames) / config_.click.sample_rate;
        // Everything happens in the tracker's clock and is converted once, on
        // the way out. Asking for beats up to a buffer plus the lookahead ahead
        // of *now + round trip* is exactly the set whose clicks belong in this
        // buffer or the next.
        const double now = stream_time_sec + config_.round_trip_sec;
        const double horizon = buffer_sec + config_.lookahead_sec;

        double beat = 0.0;
        while (tracker_.takeBeat(now, horizon, &beat)) {
            if (click_.schedule(beat - config_.round_trip_sec, schedule::BeatKind::Beat)) {
                ++beats_;
            }
            // The click we just committed to will be heard by the microphone at
            // the beat itself: the round trip taken off the submission is added
            // back by the journey. So the window to ignore is the prediction,
            // unadjusted.
            tracker_.gateClick(beat);
        }
    }

    // Always mixed, even when stopped: a click already sounding rings out
    // rather than being cut, which is what stopping a metronome sounds like.
    click_.mix(stream_time_sec, out, frames);
}

LiveMetronome::Stats LiveMetronome::stats() const {
    const tracking::LiveTracker::Stats tracked = tracker_.stats();

    Stats out;
    out.beats = beats_;
    out.beats_late = tracked.beats_late;
    out.clicks_late = click_.dropped_late();
    out.clicks_overflowed = click_.dropped_overflow();
    out.voices_stolen = click_.stolen();
    out.discontinuities = click_.discontinuities();
    out.capture_discontinuities = tracked.discontinuities;
    out.gated = tracked.gated;
    return out;
}

}  // namespace tiktak::render
