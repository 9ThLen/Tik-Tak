#include "render/live_metronome.hpp"

#include <algorithm>
#include <cmath>

namespace tiktak::render {

namespace {

// Buffers longer than this are handled a chunk at a time when the canceller is
// on. Far longer than any device's buffer, so in practice it is one chunk.
constexpr std::size_t kChunk = 4096;

ClickCancellerConfig cancellerFor(const LiveMetronomeConfig& config) {
    ClickCancellerConfig out = config.canceller;
    out.sample_rate = config.click.sample_rate;
    out.round_trip_sec = config.round_trip_sec;
    return out;
}

ClickCanceller makeCanceller(const LiveMetronomeConfig& config) {
    if (!config.subtract_own_clicks) return ClickCanceller();
    return ClickCanceller(cancellerFor(config), config.click);
}

}  // namespace

bool LiveMetronomeConfig::valid() const {
    if (!(tracker.valid() && click.valid() && round_trip_sec >= 0.0 && lookahead_sec >= 0.0)) {
        return false;
    }
    if (!subtract_own_clicks) return true;
    if (!(alone_gate_sec >= 0.0)) return false;
    // One clock and one rate, or what was played cannot be laid over what was
    // heard.
    return cancellerFor(*this).valid() &&
           std::fabs(tracker.odf.sampleRate - click.sample_rate) < 1e-6;
}

bool LiveMetronome::Stats::clean() const {
    return beats_late == 0 && clicks_late == 0 && clicks_overflowed == 0 && voices_stolen == 0 &&
           discontinuities == 0 && capture_discontinuities == 0;
}

LiveMetronome::LiveMetronome(const LiveMetronomeConfig& config)
    : config_(config),
      tracker_(config.tracker),
      click_(config.click),
      canceller_(makeCanceller(config)),
      scratch_(config.subtract_own_clicks ? kChunk : 0, 0.0f) {}

LiveMetronome::LiveMetronome(const LiveMetronomeConfig& config,
                             const ml::BeatNetWeights* const* weights, std::size_t count)
    : config_(config),
      tracker_(config.tracker, weights, count),
      click_(config.click),
      canceller_(makeCanceller(config)),
      scratch_(config.subtract_own_clicks ? kChunk : 0, 0.0f) {}

void LiveMetronome::start() { running_ = true; }

void LiveMetronome::stop() { running_ = false; }

void LiveMetronome::silence() {
    running_ = false;
    click_.reset();
}

void LiveMetronome::capture(double stream_time_sec, const float* samples, std::size_t n) {
    if (!canceller_.enabled()) {
        tracker_.process(stream_time_sec, samples, n);
        if (heard_observer_ != nullptr && samples != nullptr) {
            heard_observer_(heard_context_, stream_time_sec, samples, n);
        }
        return;
    }
    if (samples == nullptr) return;

    // The tracker is handed the room with our own click taken out of it.
    const double rate = config_.click.sample_rate;
    for (std::size_t done = 0; done < n;) {
        const std::size_t count = std::min(n - done, scratch_.size());
        const double at = stream_time_sec + static_cast<double>(done) / rate;
        canceller_.heard(std::llround(at * rate), samples + done, scratch_.data(), count);
        tracker_.process(at, scratch_.data(), count);
        if (heard_observer_ != nullptr) {
            heard_observer_(heard_context_, at, scratch_.data(), count);
        }
        done += count;
    }
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
            // With the canceller on, each click goes out upright or inverted at
            // random. The path is found by averaging what comes back with the
            // click, and music that repeats itself on every beat (a drum
            // machine does, to the sample) would otherwise average in with it
            // and be taken for the click's own echo. The echo turns over with
            // the click; the music does not.
            bool inverted = false;
            if (canceller_.enabled()) {
                polarity_ ^= polarity_ << 13;
                polarity_ ^= polarity_ >> 17;
                polarity_ ^= polarity_ << 5;
                inverted = (polarity_ & 0x10000u) != 0;
            }
            if (click_.schedule(beat - config_.round_trip_sec, schedule::BeatKind::Beat,
                                inverted)) {
                ++beats_;
            }
            // The click we just committed to will be heard by the microphone at
            // the beat itself: the round trip taken off the submission is added
            // back by the journey. So the window to ignore is the prediction,
            // unadjusted — when there is anything to hear; see gate_own_clicks.
            if (config_.gate_own_clicks) {
                tracker_.gateClick(beat);
            } else if (config_.gate_when_alone && canceller_.alone()) {
                // Nothing but our own click is sounding: what subtraction
                // leaves of it must not be heard, nor its tail, and there is no
                // music here for the gate to cost.
                tracker_.gateSpan(beat - config_.tracker.gate_before_sec,
                                  beat + config_.alone_gate_sec);
                ++clicks_alone_;
            }
            if (observer_ != nullptr) observer_(observer_context_, beat);
        }
    }

    // Always mixed, even when stopped: a click already sounding rings out
    // rather than being cut, which is what stopping a metronome sounds like.
    if (!canceller_.enabled()) {
        click_.mix(stream_time_sec, out, frames);
        return;
    }

    // The canceller needs the click by itself, exactly as it left: rendered
    // into silence first, shown to it, and only then added to the output.
    const double rate = config_.click.sample_rate;
    for (std::size_t done = 0; done < frames;) {
        const std::size_t count = std::min(frames - done, scratch_.size());
        const double at = stream_time_sec + static_cast<double>(done) / rate;
        std::fill(scratch_.begin(), scratch_.begin() + static_cast<std::ptrdiff_t>(count), 0.0f);
        click_.mix(at, scratch_.data(), count);
        canceller_.played(std::llround(at * rate), scratch_.data(), count);
        for (std::size_t i = 0; i < count; ++i) out[done + i] += scratch_[i];
        done += count;
    }
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
    out.clicks_alone = clicks_alone_;
    return out;
}

}  // namespace tiktak::render
