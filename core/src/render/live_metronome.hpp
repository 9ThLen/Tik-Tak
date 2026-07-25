#pragma once

#include <cstddef>

#include "render/click.hpp"
#include "tracking/live.hpp"

namespace tiktak::render {

struct LiveMetronomeConfig {
    tracking::LiveConfig tracker;
    ClickConfig click;

    // The measured round trip: how long after a sample is submitted to the
    // output it comes back through the input. `tiktak measure` on the harness
    // reports exactly this number, and on the phone it is what the calibration
    // screen finds.
    //
    // One number, not two, and this is why: the tracker's clock is the capture
    // stream's, in which a sound in the room appears one input latency after it
    // happened, and a click submitted now is heard one output latency from now.
    // Only the sum ever appears in the arithmetic — see roundTripSec below.
    double round_trip_sec = 0.0;

    // How far ahead of a beat a click has to be handed over. The click renders
    // in whatever buffer contains its sample, so this only has to cover the
    // time between one output callback and the next.
    double lookahead_sec = 0.05;

    bool valid() const;
};

// The microphone metronome: listens to the room and clicks on its beat.
//
// Composed here rather than in each shell for the same reason render::Metronome
// is — what happens between getting a buffer and returning one has to be the
// same everywhere. The shells differ only in how they obtain the two streams,
// which on some platforms is one callback and on others two.
//
// The latency arithmetic is the whole substance of this class and it is worth
// stating plainly, because it is the part a shell would get wrong:
//
//   The tracker predicts a beat at time `b` in the capture stream's clock. A
//   room event at physical time T appears there at T + input latency, so the
//   beat is physically at b - input. For the click to be *heard* then, it must
//   be submitted one output latency earlier still: at b - input - output, which
//   is b - round trip. That is the only place either latency appears, and only
//   their sum does.
//
//   Our own click is then heard by the microphone at exactly `b` again — the
//   round trip cancels — so the window gated out of the tracker's input is the
//   predicted beat itself, with no arithmetic at all.
//
// Real-time safe: capture() and process() allocate nothing and read no clock.
class LiveMetronome {
public:
    explicit LiveMetronome(const LiveMetronomeConfig& config);

    const LiveMetronomeConfig& config() const { return config_; }

    void start();
    void stop();
    void silence();
    bool running() const { return running_; }

    // Captured audio. `stream_time_sec` is the time of samples[0].
    void capture(double stream_time_sec, const float* samples, std::size_t n);

    // The output callback. Mixes into `out` — it does not clear it.
    void process(double stream_time_sec, float* out, std::size_t frames);

    tracking::BeatEstimate estimate(double now_sec) const { return tracker_.estimate(now_sec); }

    // Hands the tracker a tempo to start from: an offline analysis of the same
    // song, or one the user typed.
    void seedTempo(double bpm, double spread_octaves = 0.05) {
        tracker_.seedTempo(bpm, spread_octaves);
    }

    struct Stats {
        std::size_t beats = 0;              // clicks scheduled on tracked beats
        std::size_t beats_late = 0;         // beats predicted too late to play
        std::size_t clicks_late = 0;
        std::size_t clicks_overflowed = 0;
        std::size_t voices_stolen = 0;
        std::size_t discontinuities = 0;    // output buffers out of sequence
        std::size_t capture_discontinuities = 0;
        std::size_t gated = 0;              // frames withheld, our own click

        bool clean() const;
    };

    Stats stats() const;

private:
    LiveMetronomeConfig config_;
    tracking::LiveTracker tracker_;
    ClickRenderer click_;

    bool running_ = false;
    std::size_t beats_ = 0;
};

}  // namespace tiktak::render
