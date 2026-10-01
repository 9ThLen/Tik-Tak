#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "render/click.hpp"

namespace tiktak::render {

struct ClickCancellerConfig {
    double sample_rate = 48000.0;

    // Where a played click is expected back in the capture stream: the round
    // trip the metronome was given.
    double round_trip_sec = 0.0;

    // The path from speaker to microphone, as it is modelled. Around the
    // expected return there is a tap on every sample, from `before_sec` ahead
    // of it to `dense_sec` behind: that is where the direct sound and the
    // device's own case arrive, where nearly all of the click's energy is, and
    // where a delay has to be right to a fraction of a sample. After that the
    // desk, the walls and the start of the room's reverberation, a tap every
    // `tap_spacing_sec` out to `after_sec`. Those are weaker and wanted less
    // exactly, and a tap per sample there would cost twelve times the unknowns
    // for nothing.
    //
    // How far out is what a room decides. With the click sent back through a
    // synthetic room (0.4 s reverberation, 10 dB under the direct sound),
    // 28 ms of path took 11 dB off the click, 80 ms took 23 and 150 ms took
    // 32: what is not modelled is simply left in.
    double before_sec = 0.002;
    double dense_sec = 0.004;
    double after_sec = 0.150;
    double tap_spacing_sec = 0.00025;

    // How far one click moves the path estimate, when the room is as loud
    // under it as under the clicks before. The music under a click is louder
    // than the click as often as not, and it only averages out across clicks:
    // its phase at the click's pitch is different on every beat, the path's is
    // the same. 0.05 is the last twenty clicks or so: a path between a device's
    // own speaker and microphone moves rarely, and under music every halving
    // of this is worth 3 dB of click removed. In a digital loop 0.15 left
    // enough of the click under the music to cost beat F; 0.05 did not.
    //
    // Clicks are not weighed equally, though. Each counts in inverse proportion
    // to how loud the room was before it, so one click heard in a pause
    // outweighs every click heard under the music, and the estimate is clean
    // within a beat or two of the music stopping, which is exactly when a
    // metronome that still hears itself would carry on alone.
    double update = 0.05;

    // The room is never taken to be quieter than this, as a fraction of one
    // click's own energy. It bounds how much a click in dead silence can
    // outweigh the rest, and so how long an estimate made in a pause holds
    // against the clicks under the music that follow: about seventy of them.
    double quietest = 1e-3;

    // Over how long the room's level is followed, between clicks. Short, so
    // that what a click is weighed by is the room just before it, when the
    // last click's own reverberation has died away.
    double level_sec = 0.05;

    // Ridge on the normal equations, as a fraction of the click's own energy.
    double ridge = 1e-4;

    // The room counts as empty when its level just before a click has fallen
    // to less than `alone_drop` of what it usually is just before one: 0.01 is
    // 20 dB down. Compared with its own history and not with the click,
    // because what matters is that the music has gone, however loud the click
    // is beside it; and with the moment before a click, because music that
    // falls silent between its beats is silent there every time and does not
    // look like a room emptying. `alone_memory` is how far one click moves
    // what is usual. See alone().
    double alone_drop = 0.01;
    double alone_memory = 0.1;

    // And it does not count as empty while what subtraction left under the last
    // click was more than this fraction of what the click returned with. That
    // much cannot be the click's own remainder: it is music sounding on the
    // beat, under the click, where the level between clicks never sees it.
    double alone_left_ratio = 0.25;

    bool valid() const;
};

// Takes the metronome's own click out of what the microphone heard.
//
// The gate withholds every frame around a played click, and when the metronome
// is right those frames hold the music's beat. This removes the click instead
// and leaves the music where it was. It can, because the click is the one
// sound in the room whose waveform and timing are known exactly: the path from
// speaker to microphone is estimated by least squares from the clicks already
// played, averaged across them, and the click as that path would return it is
// subtracted from each new one as it arrives.
//
// What it models is the direct sound, the first reflections and the start of
// the room's reverberation, `after_sec` of them. What comes later is left in,
// and under music the estimate is only as good as the music lets it be.
// Measured in a digital loop, never yet in a room: about 49 dB with nothing in
// the way, 32 dB through a synthetic room, and 10 to 25 dB under music louder
// than the click, less the louder it is. That is plenty while music plays and
// not enough once it stops, when the click is all there is to hear; alone() is
// for that.
//
// Real-time safe once constructed: nothing allocates, and both calls belong on
// the audio thread. It adds no latency: a click is predicted from the clicks
// before it, so the first one is heard whole.
//
// It is not cheap. At 48 kHz the path is some 870 taps, each visited for every
// sample a click can reach: about nine million steps a click, and a pair of
// triangular solves when it ends. A few per cent of a desktop core; on a phone
// it has not been measured.
class ClickCanceller {
public:
    ClickCanceller() = default;   // off: heard() copies, played() does nothing
    ClickCanceller(const ClickCancellerConfig& config, const ClickConfig& click);

    // False when it was built off, or for a click with nothing to remove.
    bool enabled() const { return enabled_; }

    // What was played: the click alone, as it went into the output buffer whose
    // first sample is `start_sample` on the stream's clock.
    void played(std::int64_t start_sample, const float* click, std::size_t frames);

    // What the microphone heard on the same clock, written to `out` with the
    // predicted click removed. `out` may be `in`.
    void heard(std::int64_t start_sample, const float* in, float* out, std::size_t frames);

    // True when the click has the room to itself: the room has gone quiet
    // against what it was before the clicks so far. What is left of a click
    // after subtraction is then the loudest thing in the room however little it
    // is, and a tracker that hears it will go on confirming its own beat for
    // ever. A click played into an empty room should be gated as well, tail
    // and all, which costs nothing there, because there is no music to lose.
    bool alone() const;

    void reset();

    struct Stats {
        std::size_t clicks = 0;     // clicks the path estimate has learned from
        std::size_t skipped = 0;    // clicks it could not use: overlapping, or
                                    // back before their reference was rendered
        // The last click's window. Both are only what they say in silence: with
        // music under the click they mostly measure the music.
        double predicted_db = 0.0;  // the predicted click against what was heard
        double removed_db = 0.0;    // what was heard against what was left
        // Where the path is strongest, against the round trip given: how far
        // out the round trip is, as far as the canceller can tell.
        double arrival_sec = 0.0;
    };
    Stats stats() const { return stats_; }

    std::size_t taps() const { return lags_.size(); }
    // The path as currently estimated, one gain per tap.
    const std::vector<double>& path() const { return theta_; }

private:
    static constexpr std::size_t kPending = 16;

    float played_at(std::int64_t index) const;
    void finish();

    ClickCancellerConfig config_;
    bool enabled_ = false;

    std::vector<std::int64_t> lags_;   // each tap's delay, in samples after the first
    std::int64_t before_ = 0;          // the expected return, as a lag
    std::int64_t offset_ = 0;          // capture index minus the first tap's reference index
    std::int64_t window_ = 0;          // samples one click can reach in the capture stream

    std::vector<double> factor_;    // Cholesky factor of the click's own normal matrix
    std::vector<double> span_;      // this click's correlation with what was heard
    std::vector<double> mean_;      // the same, averaged over clicks
    std::vector<double> theta_;     // the path

    double click_energy_ = 0.0;     // one click, as played
    double level_ = 0.0;            // the room's mean square between clicks
    double level_step_ = 0.0;
    std::size_t level_seen_ = 0;    // samples it has been followed over, capped
    double under_click_ = 0.0;      // the same, as this click's window opened
    double weight_sum_ = 0.0;
    double returned_energy_ = 0.0;  // what a click comes back with, as predicted
    double last_left_ = 0.0;        // what the last click's window held once it was taken out
    double usual_level_ = 0.0;      // the room just before a click, as it usually is
    std::size_t usual_seen_ = 0;
    std::size_t alone_run_ = 0;     // clicks in a row heard in an empty room

    std::vector<float> ring_;       // what was played, by stream sample
    std::int64_t mask_ = 0;
    std::int64_t played_end_ = 0;
    bool have_played_ = false;
    std::size_t quiet_run_ = 0;

    std::array<std::int64_t, kPending> starts_{};
    std::size_t starts_head_ = 0;
    std::size_t starts_count_ = 0;

    bool active_ = false;
    bool dirty_ = false;
    std::int64_t window_last_ = 0;
    double heard_energy_ = 0.0;
    double left_energy_ = 0.0;
    double predicted_energy_ = 0.0;

    Stats stats_;
};

}  // namespace tiktak::render
