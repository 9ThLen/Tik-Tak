#include "render/metronome.hpp"

namespace tiktak::render {
namespace {

// Events pulled from the grid in one go. The grid hands out a lookahead's worth
// at a time — single digits at any musical tempo, even counting every channel —
// so this is generous, and it is a fixed array because the pull happens on the
// audio thread.
constexpr std::size_t kEventBatch = 64;

}  // namespace

bool MetronomeConfig::valid() const { return grid.valid() && click.valid(); }

bool Metronome::Stats::clean() const {
    return grid_late == 0 && clicks_late == 0 && clicks_overflowed == 0 &&
           voices_stolen == 0 && discontinuities == 0 && cues_dropped == 0;
}

Metronome::Metronome(const MetronomeConfig& config)
    : config_(config), scheduler_(config.grid), click_(config.click) {}

void Metronome::start(double stream_time_sec) {
    scheduler_.start(stream_time_sec);
    beats_ = 0;
    cues_dropped_ = 0;
}

void Metronome::stop() { scheduler_.stop(); }

void Metronome::silence() {
    scheduler_.stop();
    click_.reset();
}

void Metronome::set_tempo(double bpm) { scheduler_.set_tempo(bpm); }

void Metronome::align_to(double beat_time_sec, double now_sec) {
    scheduler_.align_to(beat_time_sec, now_sec);
}

void Metronome::process(double stream_time_sec, float* out, std::size_t frames,
                        schedule::Event* cues, std::size_t cue_capacity,
                        std::size_t* cue_count) {
    if (cue_count) *cue_count = 0;

    schedule::Event events[kEventBatch];
    std::size_t cues_written = 0;

    // Pulled once per buffer rather than once per beat: the grid decides how far
    // ahead it hands events out, and asking again inside the buffer would only
    // ever return the same answer.
    const std::size_t count = scheduler_.pull(stream_time_sec, events, kEventBatch);

    for (std::size_t i = 0; i < count; ++i) {
        if (events[i].channel == schedule::Channel::Audio) {
            click_.schedule(events[i].time_sec, events[i].kind);
            ++beats_;
            continue;
        }
        // Haptics and the display are the shell's to deliver; they are handed
        // back with their own compensated times rather than the beat's, because
        // the taptic engine and the next display frame are not the same delay.
        if (cues && cues_written < cue_capacity) {
            cues[cues_written++] = events[i];
        } else {
            ++cues_dropped_;
        }
    }

    if (cue_count) *cue_count = cues_written;

    click_.mix(stream_time_sec, out, frames);
}

Metronome::Stats Metronome::stats() const {
    Stats s;
    s.beats = beats_;
    s.grid_late = scheduler_.late_count();
    s.clicks_late = click_.dropped_late();
    s.clicks_overflowed = click_.dropped_overflow();
    s.voices_stolen = click_.stolen();
    s.discontinuities = click_.discontinuities();
    s.cues_dropped = cues_dropped_;
    return s;
}

}  // namespace tiktak::render
