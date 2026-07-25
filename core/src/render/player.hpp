#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "render/click.hpp"
#include "schedule/scheduler.hpp"

namespace tiktak::render {

struct PlayerConfig {
    double sample_rate = 48000.0;
    ClickConfig click;

    // Bars are bookkeeping over the analysed grid: beat `downbeat_offset` is
    // taken as a bar's first beat and every `beats_per_bar`-th after it. Until
    // Phase 7 finds real downbeats this is a convention, not a detection — the
    // offset exists so the user can shift which beat is "the one".
    int beats_per_bar = 4;
    int downbeat_offset = 0;

    // Count-in beats before the music: clicks at the local beat interval, with
    // the track silent, so a singer knows the tempo before the first note.
    int count_in_beats = 0;

    // How far ahead haptic and visual cues are handed out. The click needs no
    // lookahead — it is rendered here, in the same callback — but the shell
    // has to schedule the taptic engine and the next frame in advance.
    double cue_lookahead_sec = 0.25;

    // Per-channel output latency. The click needs none *relative to the
    // track*: both leave through the same device buffer, so placing the click
    // on the track sample of the beat aligns them by construction. The audio
    // entry here is the device's output latency, used only to compute when a
    // beat is *heard* so the other channels can be compensated against it.
    std::array<double, schedule::kChannelCount> latency_sec{{0.0, 0.0, 0.0}};
    std::array<bool, schedule::kChannelCount> channel_enabled{{true, false, false}};

    bool valid() const;
};

// Playback of an analysed track with the metronome riding its beat grid — the
// main scenario of the whole app: the singer's backing track, with the click
// exactly on the beats the offline analysis found.
//
// This composition lives in core for the same reason Metronome does: the shells
// differ in how they obtain a buffer and a clock, and must not differ in what
// happens between them. It is a separate class rather than a Metronome mode
// because the time source is inverted — a metronome generates its grid from a
// tempo, while the player follows a grid the analysis already fixed, and the
// track, not the clock, is the authority on where the beats are.
//
// One clock domain, supplied by the caller: stream time, the moment a sample is
// handed to the device. Track audio and clicks share that path, so a click
// written on the same sample as its beat arrives with it whatever the output
// latency is. Only the haptic and visual cues need latency arithmetic.
//
// process() is real-time safe. setTrack() and setGrid() allocate and must be
// called before start(), from a normal thread.
class TrackPlayer {
public:
    explicit TrackPlayer(const PlayerConfig& config);

    const PlayerConfig& config() const { return config_; }

    // The decoded track, mono at the configured sample rate. Not copied — five
    // minutes of audio is tens of megabytes, and the caller already holds it
    // for the analysis. The buffer must outlive playback.
    void setTrack(const float* samples, std::size_t frames);

    // The analysed beat grid, seconds from the start of the track, ascending.
    // Copied — it is a few kilobytes, and holding it means a caller can drop
    // its OfflineResult once the player is loaded.
    void setGrid(const double* beat_times, std::size_t count);

    // Loops bars [start_bar, end_bar): the track jumps from the end bar's
    // first beat back to the start bar's first, sample-exactly, which is what
    // makes practising one difficult phrase bearable. Set before start().
    // False when the bars do not exist in the grid or the range is empty.
    bool setLoop(std::int64_t start_bar, std::int64_t end_bar);
    void clearLoop();

    // Starts playback at `from_bar`'s first beat, count-in first if configured.
    // `stream_time_sec` is when the first count-in sample leaves. False when
    // there is nothing to play, or a count-in is asked of a grid too short to
    // yield a beat interval.
    bool start(double stream_time_sec, std::int64_t from_bar = 0);

    // Stops advancing; sounding clicks ring out. silence() also cuts them.
    void stop();
    void silence();

    bool running() const { return started_ && !ended_; }

    // Current position, seconds into the track. Meaningful while running.
    double positionSec() const;

    // The audio callback. Mixes into `out` — it does not clear it.
    // `stream_time_sec` is the time of out[0]. Haptic and visual cues are
    // written to `cues` exactly as Metronome hands them out; events beyond
    // `cue_capacity` are counted in stats rather than silently lost.
    void process(double stream_time_sec, float* out, std::size_t frames,
                 schedule::Event* cues = nullptr, std::size_t cue_capacity = 0,
                 std::size_t* cue_count = nullptr);

    struct Stats {
        std::size_t beats = 0;              // clicks scheduled since start()
        std::size_t loops = 0;              // times the loop wrapped
        std::size_t clicks_late = 0;        // clicks that arrived past their buffer
        std::size_t clicks_overflowed = 0;  // clicks refused, queue full
        std::size_t voices_stolen = 0;      // clicks cut short, all voices busy
        std::size_t discontinuities = 0;    // buffers that did not follow the last
        std::size_t cues_dropped = 0;       // cue events the caller had no room for

        bool clean() const;
    };

    Stats stats() const;

private:
    // A position on the playback timeline: seconds since start(), count-in
    // included, loop iterations unrolled. Beat `i` on iteration `shift` sits at
    // count_in + (grid[i] - grid[start]) + shift.
    struct BeatCursor {
        std::size_t beat = 0;        // next grid index to emit
        std::size_t count_in = 0;    // next count-in click, < count_in_beats
        double loop_shift_sec = 0.0; // accumulated loop iterations
    };

    schedule::BeatKind kindOf(std::size_t beat_index) const;
    std::int64_t barOf(std::size_t beat_index) const;
    int beatInBar(std::size_t beat_index) const;

    // Timeline second of the cursor's next event, or false when the grid is
    // exhausted. Advancing past the loop's end beat wraps the cursor.
    bool nextEvent(BeatCursor& cursor, double* when_sec, std::size_t* beat_index,
                   bool* is_count_in);

    void mixTrack(double stream_time_sec, float* out, std::size_t frames);

    PlayerConfig config_;
    ClickRenderer click_;

    const float* track_ = nullptr;
    std::size_t track_frames_ = 0;
    std::vector<double> grid_;

    bool loop_set_ = false;
    std::size_t loop_start_beat_ = 0;
    std::size_t loop_end_beat_ = 0;      // one past the last looped beat
    double loop_start_sec_ = 0.0;
    double loop_end_sec_ = 0.0;

    bool started_ = false;
    bool ended_ = false;
    double start_stream_sec_ = 0.0;      // stream time of timeline zero
    std::size_t start_beat_ = 0;
    double start_beat_sec_ = 0.0;        // grid time the track enters at
    std::int64_t track_entry_frame_ = 0; // that time, in track samples
    double count_in_interval_sec_ = 0.0;
    double count_in_sec_ = 0.0;

    BeatCursor click_cursor_;
    BeatCursor cue_cursor_;

    // Timeline frames handed to the device so far — what positionSec reads.
    std::int64_t timeline_end_frame_ = 0;

    std::size_t beats_ = 0;
    std::size_t loops_ = 0;
    std::size_t cues_dropped_ = 0;
};

}  // namespace tiktak::render
