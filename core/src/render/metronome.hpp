#pragma once

#include <cstddef>

#include "render/click.hpp"
#include "schedule/scheduler.hpp"

namespace tiktak::render {

struct MetronomeConfig {
    schedule::SchedulerConfig grid;
    ClickConfig click;

    bool valid() const;
};

// A metronome: the beat grid and the click, wired together.
//
// This composition is here rather than in each shell because it is the same
// composition everywhere — the desktop harness, iOS and Android differ in how
// they obtain a buffer and a clock, and must not differ in what happens between
// them. Written out in every shell it would be three copies that drift apart,
// and the drift would show up as a metronome that measures well on the harness
// and feels wrong on the phone.
//
// One clock domain, supplied by the caller: *stream time*, meaning the moment a
// sample is handed to the device, not the moment it is heard. That is the clock
// the scheduler's latency compensation is expressed in — a click for a beat at
// stream time t is written one output latency before t, so that after the
// device's delay it arrives on the beat.
//
// Real-time safe: nothing allocates after construction and no clock is read.
class Metronome {
public:
    explicit Metronome(const MetronomeConfig& config);

    const MetronomeConfig& config() const { return config_; }

    // Starts the grid with the first beat at `stream_time_sec`. Start it far
    // enough ahead that the first beat is not already in the past by the time
    // the next buffer is filled, or it is dropped for lateness.
    void start(double stream_time_sec);

    // Stops handing out beats. Clicks already sounding ring out rather than
    // being cut, which is what stopping a real metronome sounds like; use
    // silence() to cut them.
    void stop();
    void silence();

    bool running() const { return scheduler_.running(); }

    void set_tempo(double bpm);

    // Puts a beat on `beat_time_sec` without changing the tempo — manual mode,
    // where the player fixes the tempo and only the offset has to be found.
    void align_to(double beat_time_sec, double now_sec);

    // The audio callback, in full. `stream_time_sec` is the time of out[0].
    //
    // Mixes into `out` — it does not clear it. A device hands over a buffer of
    // whatever was there before, so a caller playing nothing else must zero it
    // first; a caller playing a backing track must not.
    //
    // Haptic and visual events, if those channels are enabled, are written to
    // `cues` for the shell to deliver — they cannot be rendered here because
    // they are not audio. Pass nullptr to discard them. `cue_count` receives
    // how many were written, and events beyond `cue_capacity` are counted in
    // `cues_dropped()` rather than silently lost.
    void process(double stream_time_sec, float* out, std::size_t frames,
                 schedule::Event* cues = nullptr, std::size_t cue_capacity = 0,
                 std::size_t* cue_count = nullptr);

    // What the harness reports and the shells log. Every one of these is a
    // number that should be zero on a healthy run, which is the point: a
    // metronome that is quietly dropping every tenth beat looks fine until
    // something counts.
    struct Stats {
        std::size_t beats = 0;              // audio clicks scheduled since start()
        std::size_t grid_late = 0;          // beats the grid gave up on as too late
        std::size_t clicks_late = 0;        // clicks that arrived past their buffer
        std::size_t clicks_overflowed = 0;  // clicks refused, queue full
        std::size_t voices_stolen = 0;      // clicks cut short, all voices busy
        std::size_t discontinuities = 0;    // buffers that did not follow the last
        std::size_t cues_dropped = 0;       // cue events the caller had no room for

        // True when nothing went wrong at all — the check a harness run ends on.
        bool clean() const;
    };

    Stats stats() const;

private:
    MetronomeConfig config_;
    schedule::Scheduler scheduler_;
    ClickRenderer click_;

    std::size_t beats_ = 0;
    std::size_t cues_dropped_ = 0;
};

}  // namespace tiktak::render
