#include "render/click.hpp"

#include <cmath>
#include <cstddef>

namespace tiktak::render {
namespace {

// A click ends when it has fallen this far, and it is simply stopped there
// rather than faded further. The step at the cut is a thousandth of the peak,
// tens of dB below anything a listener can hear over the click itself.
constexpr double kEndAmplitude = 1e-3;

constexpr double kTwoPi = 6.283185307179586476925286766559;

}  // namespace

bool ClickTone::valid() const {
    if (!(frequency_hz > 0.0)) return false;
    if (!(length_sec > 0.0) || !(length_sec < 10.0)) return false;
    if (!(gain >= 0.0) || !(gain <= 100.0)) return false;
    return true;
}

bool ClickConfig::valid() const {
    if (!(sample_rate > 0.0) || !(sample_rate < 1e7)) return false;
    if (!downbeat.valid() || !beat.valid() || !subdivision.valid()) return false;

    // Above Nyquist a tone is not a higher click, it is a lower one at a
    // mirrored frequency — a silent way to get a metronome that sounds wrong
    // only at low sample rates.
    const double nyquist = sample_rate * 0.5;
    if (!(downbeat.frequency_hz < nyquist)) return false;
    if (!(beat.frequency_hz < nyquist)) return false;
    if (!(subdivision.frequency_hz < nyquist)) return false;

    if (max_voices < 1 || max_voices > 1024) return false;
    if (max_pending < 1 || max_pending > 65536) return false;
    if (!(late_tolerance_sec >= 0.0) || !(late_tolerance_sec < 1.0)) return false;
    return true;
}

ClickRenderer::ClickRenderer(const ClickConfig& config) : config_(config) {
    sample_period_ = 1.0 / config_.sample_rate;
    voices_.resize(static_cast<std::size_t>(config_.max_voices));
    pending_.resize(static_cast<std::size_t>(config_.max_pending));
}

const ClickTone& ClickRenderer::toneFor(schedule::BeatKind kind) const {
    switch (kind) {
        case schedule::BeatKind::Downbeat: return config_.downbeat;
        case schedule::BeatKind::Subdivision: return config_.subdivision;
        case schedule::BeatKind::Beat: break;
    }
    return config_.beat;
}

bool ClickRenderer::schedule(double time_sec, schedule::BeatKind kind) {
    // A NaN time compares false against everything, so it would never be placed
    // and never expire — it would sit in the queue forever, one slot smaller
    // every time until the metronome stopped sounding.
    if (std::isnan(time_sec)) return false;
    if (pending_count_ >= pending_.size()) {
        ++dropped_overflow_;
        return false;
    }
    pending_[pending_count_].time_sec = time_sec;
    pending_[pending_count_].kind = kind;
    ++pending_count_;
    return true;
}

void ClickRenderer::startVoice(const ClickTone& tone) {
    std::size_t chosen = voices_.size();
    double quietest = 0.0;

    for (std::size_t i = 0; i < voices_.size(); ++i) {
        if (voices_[i].remaining == 0) {
            chosen = i;
            break;
        }
        // Steal the quietest rather than the oldest: with three tone lengths in
        // play the oldest voice is not reliably the least audible one, and the
        // whole point of stealing is to cut what will be missed least.
        const double level = voices_[i].envelope * voices_[i].gain;
        if (chosen == voices_.size() || level < quietest) {
            chosen = i;
            quietest = level;
        }
    }

    if (voices_[chosen].remaining != 0) ++stolen_;

    // Ceil, so a click is never zero samples long however short the tone or low
    // the sample rate.
    const double exact = tone.length_sec * config_.sample_rate;
    std::size_t samples = static_cast<std::size_t>(std::ceil(exact));
    if (samples == 0) samples = 1;

    const double w = kTwoPi * tone.frequency_hz * sample_period_;

    Voice& voice = voices_[chosen];
    voice.remaining = samples;
    // Starting the rotation at (1, 0) and reading the sine means the first
    // sample is exactly zero: a click that began at full amplitude would put a
    // step into the signal, which is heard as a thump under the click.
    voice.cos_v = 1.0;
    voice.sin_v = 0.0;
    voice.cos_w = std::cos(w);
    voice.sin_w = std::sin(w);
    voice.envelope = 1.0;
    voice.decay = std::pow(kEndAmplitude, 1.0 / static_cast<double>(samples));
    voice.gain = tone.gain;
}

void ClickRenderer::renderVoice(Voice& voice, float* out, std::size_t frames) {
    const std::size_t n = frames < voice.remaining ? frames : voice.remaining;

    double cos_v = voice.cos_v;
    double sin_v = voice.sin_v;
    const double cos_w = voice.cos_w;
    const double sin_w = voice.sin_w;
    double envelope = voice.envelope;
    const double decay = voice.decay;
    const double gain = voice.gain;

    for (std::size_t i = 0; i < n; ++i) {
        out[i] += static_cast<float>(sin_v * envelope * gain);

        // Rotate the unit vector by one sample's worth of angle. Two multiplies
        // more than a table lookup and no table; the magnitude drifts, but over
        // a click's few thousand samples the drift is far below the last bit of
        // a float.
        const double c = cos_v * cos_w - sin_v * sin_w;
        const double s = sin_v * cos_w + cos_v * sin_w;
        cos_v = c;
        sin_v = s;
        envelope *= decay;
    }

    voice.cos_v = cos_v;
    voice.sin_v = sin_v;
    voice.envelope = envelope;
    voice.remaining -= n;
}

void ClickRenderer::mix(double start_time_sec, float* out, std::size_t frames) {
    if (!out || frames == 0) return;

    if (have_next_start_ &&
        std::fabs(start_time_sec - next_start_time_) > sample_period_ * 0.5) {
        ++discontinuities_;
    }
    next_start_time_ = start_time_sec + static_cast<double>(frames) * sample_period_;
    have_next_start_ = true;

    // Anything too late to place is dropped before the buffer is walked, so the
    // placement loop only ever deals with clicks it can actually put somewhere.
    const double earliest = start_time_sec - config_.late_tolerance_sec;
    for (std::size_t i = 0; i < pending_count_;) {
        if (pending_[i].time_sec < earliest) {
            ++dropped_late_;
            pending_[i] = pending_[pending_count_ - 1];
            --pending_count_;
        } else {
            ++i;
        }
    }

    std::size_t cursor = 0;

    for (;;) {
        // Earliest click still queued. The queue is small — a lookahead's worth,
        // single digits at any musical tempo — so scanning it per click is
        // cheaper than keeping it sorted.
        std::size_t next = pending_count_;
        for (std::size_t i = 0; i < pending_count_; ++i) {
            if (next == pending_count_ || pending_[i].time_sec < pending_[next].time_sec) {
                next = i;
            }
        }
        if (next == pending_count_) break;

        const double offset_exact = (pending_[next].time_sec - start_time_sec) / sample_period_;
        std::size_t offset;
        if (offset_exact <= 0.0) {
            offset = 0;   // late within tolerance: nudged, not dropped
        } else {
            // Which buffer a click belongs to is decided by the sample it
            // rounds to, not by whether its time falls inside this one. A click
            // in the last half-sample of a buffer rounds to the first sample of
            // the next, and squeezing it into this one instead would put it a
            // whole sample early — the one place this design can lose the
            // accuracy it exists to provide.
            const double rounded = std::floor(offset_exact + 0.5);
            if (rounded >= static_cast<double>(frames)) break;
            offset = static_cast<std::size_t>(rounded);
        }
        // Clicks are started in time order, so a click nudged forward can never
        // land before one already placed.
        if (offset < cursor) offset = cursor;

        for (Voice& voice : voices_) {
            if (voice.remaining != 0) renderVoice(voice, out + cursor, offset - cursor);
        }
        cursor = offset;

        startVoice(toneFor(pending_[next].kind));
        pending_[next] = pending_[pending_count_ - 1];
        --pending_count_;
    }

    for (Voice& voice : voices_) {
        if (voice.remaining != 0) renderVoice(voice, out + cursor, frames - cursor);
    }
}

void ClickRenderer::reset() {
    for (Voice& voice : voices_) voice.remaining = 0;
    pending_count_ = 0;
    have_next_start_ = false;
    // The counters are diagnostics for a whole run, not for one start/stop, so
    // they survive: a metronome restarted between takes would otherwise hide
    // the drops that happened in the take before.
}

std::size_t ClickRenderer::active_voice_count() const {
    std::size_t n = 0;
    for (const Voice& voice : voices_) {
        if (voice.remaining != 0) ++n;
    }
    return n;
}

}  // namespace tiktak::render
