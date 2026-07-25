#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace tiktak::schedule {

// Output channels. Each carries its own latency, because they do not arrive
// together: audio waits on the device buffer, haptics on the taptic engine,
// video on the next frame. Compensating them with one number makes the
// vibration drift audibly against the click.
enum class Channel : int {
    Audio = 0,
    Haptic = 1,
    Visual = 2,
};

inline constexpr int kChannelCount = 3;

enum class BeatKind : int {
    Downbeat = 0,      // first beat of the bar
    Beat = 1,          // any other beat
    Subdivision = 2,   // between beats
};

struct Event {
    // When to hand this to its device: the musical instant minus that channel's
    // latency. This is the number the caller schedules against, never "now".
    double time_sec = 0.0;
    // The musical instant itself, shared by every channel of the same step.
    double beat_time_sec = 0.0;
    Channel channel = Channel::Audio;
    BeatKind kind = BeatKind::Beat;
    std::int64_t step = 0;   // grid position, counting subdivisions
    std::int64_t bar = 0;
    int beat_in_bar = 0;
    int subdivision = 0;     // 0 on the beat itself
};

struct SchedulerConfig {
    double bpm = 120.0;
    int beats_per_bar = 4;
    int subdivisions = 1;     // 1 = beats only, 2 = eighths, 3 = triplets...
    // How far ahead to hand out events. Must comfortably exceed the interval at
    // which the host polls, or beats fall through the gap between polls.
    double lookahead_sec = 0.25;
    std::array<double, kChannelCount> latency_sec{{0.0, 0.0, 0.0}};
    std::array<bool, kChannelCount> channel_enabled{{true, true, true}};

    bool valid() const;
};

// Turns a tempo into precisely timed events, ahead of time.
//
// The rule that makes a metronome feel solid is that nothing is ever played on
// demand. Calling into the audio device at the moment of the beat inherits
// every scheduling delay between the decision and the speaker — tens of
// milliseconds, and different every time. Instead the grid is computed in
// advance and each event is handed over with a timestamp, so the device places
// it exactly.
//
// Everything here is one clock domain: the caller's monotonic host clock, in
// seconds. The core never reads a clock itself — `now_sec` is always passed in,
// which is what makes this testable without waiting in real time.
//
// Single-threaded and allocation-free after construction; safe to pull from an
// audio callback.
class Scheduler {
public:
    explicit Scheduler(const SchedulerConfig& config);

    const SchedulerConfig& config() const { return config_; }
    bool running() const { return running_; }

    // Starts the grid with step 0 at `now_sec`.
    void start(double now_sec);
    void stop();

    // Changes tempo without moving anything already handed out.
    //
    // The grid is re-anchored on the last emitted event rather than on "now",
    // so the beat that was already committed stays where it was and the new
    // tempo takes effect from there. Re-anchoring on now would shift a beat the
    // device is already holding, which is heard as a stumble.
    void set_tempo(double bpm);

    // Aligns the grid's phase so a beat lands on `beat_time_sec`, for the
    // manual mode where the user fixes the tempo and only the offset has to be
    // found. Events already handed out are left alone; if the shift would put
    // the next event in the past, the grid skips forward instead.
    void align_to(double beat_time_sec, double now_sec);

    // Fills `out` with every event due between `now_sec` and the lookahead
    // horizon, and advances past them. Returns how many were written.
    //
    // An event whose compensated time has already passed is *not* emitted: a
    // late click is worse than a missing one, since it actively misleads the
    // player. `dropped_late` counts those, so the host can widen its lookahead
    // or report the overload instead of silently limping.
    std::size_t pull(double now_sec, Event* out, std::size_t capacity,
                     std::size_t* dropped_late = nullptr);

    // Host time of a grid step. Exposed for tests and for hosts that want to
    // draw the grid ahead of the events.
    double step_time(std::int64_t step) const;

    // Total events dropped for lateness since start().
    std::size_t late_count() const { return late_count_; }

private:
    Event make_event(std::int64_t step, Channel channel, double time_sec,
                     double beat_time_sec) const;
    // Returns how many events were skipped, so every drop is accounted for
    // through exactly one path.
    std::size_t catch_up(double now_sec);

    SchedulerConfig config_;
    double step_duration_ = 0.0;   // seconds between grid steps
    double max_latency_ = 0.0;

    double anchor_time_ = 0.0;     // host time of `anchor_step_`
    std::int64_t anchor_step_ = 0;
    std::int64_t next_step_ = 0;   // next step to hand out
    std::int64_t last_emitted_step_ = -1;

    std::size_t enabled_count_ = 0;
    std::size_t late_count_ = 0;
    bool running_ = false;
};

}  // namespace tiktak::schedule
