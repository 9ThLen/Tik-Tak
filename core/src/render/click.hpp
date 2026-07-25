#pragma once

#include <cstddef>
#include <vector>

#include "schedule/scheduler.hpp"

namespace tiktak::render {

// One click's sound. Three of these make a metronome: the downbeat has to be
// distinguishable from the other beats without being a different instrument,
// and a subdivision has to sit under both without competing.
struct ClickTone {
    double frequency_hz = 1000.0;
    // How long the click lasts, defined as the time to fall 60 dB. Not a time
    // constant: this is the number you would measure with a stopwatch, and it
    // has to stay well under a beat or fast tempos turn into a drone.
    double length_sec = 0.06;
    double gain = 1.0;

    bool valid() const;
};

struct ClickConfig {
    double sample_rate = 48000.0;

    // Roughly G6 / C6 / G5. A fifth apart so the downbeat reads as "the one"
    // rather than as a different sound, and the subdivision an octave below the
    // beat so it stays underneath it.
    ClickTone downbeat{1568.0, 0.075, 1.0};
    ClickTone beat{1046.5, 0.060, 0.75};
    ClickTone subdivision{784.0, 0.035, 0.35};

    // Clicks decay, so several can overlap at fast subdivisions. Eight is far
    // more than a metronome needs; the pool exists so that a pathological
    // setting degrades by dropping the oldest click instead of allocating.
    int max_voices = 8;

    // How many scheduled-but-not-yet-sounded clicks can be held. The host hands
    // over a lookahead's worth at a time, which is single digits at any sane
    // tempo.
    int max_pending = 64;

    // A click whose time falls before the buffer being filled is played at the
    // start of that buffer if it is late by less than this, and dropped if it
    // is later. Both halves matter: rounding and a host that polls a hair late
    // produce sub-millisecond lateness that is inaudible when nudged and
    // audible as a hole when dropped, while a genuinely late click actively
    // misleads a player and is better missing.
    double late_tolerance_sec = 0.002;

    bool valid() const;
};

// Turns scheduled beats into audio.
//
// The scheduler decides *when*, this decides *what it sounds like*, and the two
// are separate because the shells differ in the first and must not differ in
// the second: a click that sounds different on iOS and on the desktop harness
// would make every timing measurement taken on the harness unusable.
//
// Placement is sample-accurate. The click is written at the sample nearest its
// scheduled time rather than at the start of the buffer it lands in — the
// difference is up to a whole buffer, which at 10 ms is grossly audible, while
// the residual rounding error is half a sample, 10 µs, which is not.
//
// Real-time safe: everything is sized in the constructor, nothing allocates,
// and no clock is read. Single-threaded — schedule() and mix() are both meant
// to be called from the audio callback.
class ClickRenderer {
public:
    explicit ClickRenderer(const ClickConfig& config);

    const ClickConfig& config() const { return config_; }

    // Queues a click at host time `time_sec`, in the same clock domain as the
    // times mix() is given. False if the queue is full, which is counted.
    bool schedule(double time_sec, schedule::BeatKind kind);

    // Adds into `out` — it does not clear it. Mixing rather than filling is the
    // contract because Phase 4 plays the click over a track, and a fill would
    // silently erase it; the name is at the call site so the caller cannot
    // forget which one this is.
    //
    // `start_time_sec` is the host time of the first sample of `out`.
    void mix(double start_time_sec, float* out, std::size_t frames);

    // Silences everything, sounding and queued. For stopping the metronome
    // without tearing down the renderer.
    void reset();

    std::size_t pending_count() const { return pending_count_; }
    std::size_t active_voice_count() const;

    // Clicks that arrived too late to place — see `late_tolerance_sec`.
    std::size_t dropped_late() const { return dropped_late_; }
    // Clicks refused because the queue was full: the host is handing over more
    // lookahead than the renderer was built for.
    std::size_t dropped_overflow() const { return dropped_overflow_; }
    // Clicks cut short because every voice was busy.
    std::size_t stolen() const { return stolen_; }
    // mix() calls whose start time did not continue the previous buffer. A
    // sounding click assumes buffers are contiguous, so a jump means the device
    // dropped or repeated a buffer — which is exactly the glitch the desktop
    // harness exists to measure, so it is counted rather than hidden.
    std::size_t discontinuities() const { return discontinuities_; }

private:
    struct Voice {
        // A voice is sounding exactly while `remaining` is non-zero. The length
        // is fixed when it starts rather than being tested for audibility every
        // sample, which keeps the inner loop branchless and makes "a click is
        // this many samples long" something a test can state exactly.
        std::size_t remaining = 0;
        double cos_v = 1.0;      // rotating unit vector: the oscillator
        double sin_v = 0.0;
        double cos_w = 1.0;      // rotation per sample
        double sin_w = 0.0;
        double envelope = 0.0;
        double decay = 1.0;      // envelope multiplier per sample
        double gain = 1.0;
    };

    struct Pending {
        double time_sec = 0.0;
        schedule::BeatKind kind = schedule::BeatKind::Beat;
    };

    const ClickTone& toneFor(schedule::BeatKind kind) const;
    void startVoice(const ClickTone& tone);
    void renderVoice(Voice& voice, float* out, std::size_t frames);

    ClickConfig config_;
    double sample_period_ = 0.0;

    std::vector<Voice> voices_;
    std::vector<Pending> pending_;
    std::size_t pending_count_ = 0;

    double next_start_time_ = 0.0;
    bool have_next_start_ = false;

    std::size_t dropped_late_ = 0;
    std::size_t dropped_overflow_ = 0;
    std::size_t stolen_ = 0;
    std::size_t discontinuities_ = 0;
};

}  // namespace tiktak::render
