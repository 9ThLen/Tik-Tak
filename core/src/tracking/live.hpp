#pragma once

#include <array>
#include <cstddef>

#include "dsp/odf.hpp"
#include "tracking/particle.hpp"
#include "tracking/sync.hpp"

namespace tiktak::tracking {

struct LiveConfig {
    dsp::OdfConfig odf;
    ParticleFilterConfig filter;
    SyncConfig sync;  // manual mode only: finding the phase at a known tempo

    // Half-life of the peak the onset function is normalised against. The
    // filter's constants are calibrated in units where a beat is worth about
    // one, so the front-end has to deliver that whatever the room's level —
    // hence dividing by the loudest onset of the last few seconds rather than
    // by a standard deviation. A z-score would also be level-invariant, but its
    // scale moves with the *material*: dense music raises the running mean and
    // shrinks every peak towards it, so the same beat arrives as weaker
    // evidence purely because there is more going on around it.
    double onset_peak_tau_sec = 3.0;

    // How long our own click is treated as blinding the microphone, relative to
    // the moment it is heard.
    double gate_before_sec = 0.005;
    double gate_after_sec = 0.050;

    // Hysteresis on publishing beats. Above `lock_confidence` the tracker
    // follows the cloud; below `release_confidence` it stops handing out beats
    // altogether; in between it coasts at the last tempo it was sure of, which
    // is what a musician does when the band drops out for a bar.
    double lock_confidence = 0.35;
    double release_confidence = 0.15;

    // A stream time this far from where the sample count says it should be
    // means the device dropped or repeated a buffer.
    double discontinuity_tolerance_sec = 0.002;

    bool valid() const;
};

// A live configuration for a capture rate, with the front-end sized in
// milliseconds rather than samples.
//
// The ODF's defaults are 2048/512 *samples*, which at 48 kHz is a 43 ms window
// every 11 ms and at 22 kHz is 93 ms every 23 ms — the same numbers describing
// a front-end twice as coarse. The filter is tuned against how much onset
// energy a beat is worth per frame, so the coarser front-end quietly halves the
// evidence per beat while the charge per predicted beat stays put, and the
// tracker that was steady at 48 kHz wanders at 22. Scaling both with the rate
// keeps the tracker's world the same whatever the device hands it.
LiveConfig liveConfigFor(double sample_rate);

// The microphone path, whole: audio in, beat predictions out.
//
// This is the online counterpart of analysis::OfflineAnalyzer and is composed
// here rather than in each shell for the same reason render::Metronome is —
// the shells differ in how they obtain a buffer and a clock, and must not
// differ in what happens between them.
//
// Two things it owns that a shell would otherwise have to reinvent:
//
// *Level normalisation.* The particle filter's observation gain is a constant,
// so what it multiplies must not depend on how loud the room is.
//
// *Own-click gating.* A metronome listening through a microphone hears its own
// click, and a click is the most onset-like sound there is. Left alone the
// tracker locks onto itself: confidence goes to one, the tempo stops responding
// to the music, and nothing about the output looks wrong. The click cannot be
// subtracted — the room's response to it is unknown — so instead the tracker
// declines to look during the window it occupies. That costs information but
// does not bias the filter, because the observation is zero-mean: a dropped
// frame changes no weights at all, while a frame merely ignored by a
// single-hypothesis tracker would still shift its estimate.
//
// Real-time safe: process() allocates nothing and reads no clock.
class LiveTracker {
public:
    explicit LiveTracker(const LiveConfig& config);

    const LiveConfig& config() const { return config_; }

    // Feeds captured audio. `stream_time_sec` is the time of samples[0], in the
    // same clock the shell schedules output in.
    void process(double stream_time_sec, const float* samples, std::size_t n);

    // Tells the tracker when its own click will reach the microphone — that is
    // the moment the click is *heard*, output latency and room delay already
    // added by the caller. The core cannot compute it: only the shell knows
    // what the round trip measured.
    void gateClick(double heard_time_sec);

    BeatEstimate estimate(double now_sec) const { return filter_.estimate(now_sec); }

    // Hands out the next beat to play, once, when it comes within
    // `lookahead_sec` of now. True when `beat_sec` was written.
    //
    // A beat, once handed out, is never revised: by then the click is in a
    // buffer on its way to the device, and moving it would be a click that
    // stutters rather than a click that corrects. Refinements land on the beat
    // after it.
    bool takeBeat(double now_sec, double lookahead_sec, double* beat_sec);

    // Concentrates the cloud on a known tempo — an offline analysis of the same
    // song, or a tempo the user typed.
    void seedTempo(double bpm, double spread_octaves = 0.05);

    // Manual mode: the tempo is the user's and the room is asked only where the
    // beat falls. Zero goes back to tracking the tempo too.
    //
    // This is a different promise from auto mode, and the difference is worth
    // being explicit about, because it is what the mode is for:
    //
    // - Nothing is played until the room has been heard. The user sets a tempo
    //   and starts; the click waits, catches the first phrase, and falls in on
    //   it. That waiting is the feature — a metronome that starts on the beat
    //   the user's own count-in landed on needs no count-in of its own.
    //
    // - Once it has fallen in, it does not stop. In auto mode a room that goes
    //   quiet has taken the tempo with it, so the tracker coasts and eventually
    //   gives up; here the tempo was never the room's to take. The click keeps
    //   the user's BPM through a silent bar, a solo, a cough, indefinitely.
    //   That falls out of the filter rather than being special-cased: with the
    //   period pinned and the observation zero-mean, silence moves no weights
    //   and the grid simply continues.
    //
    // The tempo is taken as given even outside the configured BPM range — see
    // BeatParticleFilter::pinPeriod.
    void setManualTempo(double bpm);
    double manualTempo() const { return manual_bpm_; }

    // Manual mode, still waiting for something to synchronise to. What a shell
    // shows as "listening…", and the reason no beats are coming out.
    bool waiting() const { return manual_bpm_ > 0.0 && !acquired_; }

    // How concentrated the room's onsets are at one phase, 0..1. Manual mode
    // only, and useful mainly as the meter behind that "listening…".
    double syncStrength() const { return sync_.strength(); }

    void reset();

    struct Stats {
        std::size_t frames = 0;          // ODF frames produced
        std::size_t gated = 0;           // frames withheld, our own click
        std::size_t beats = 0;           // beats handed out
        std::size_t beats_late = 0;      // predicted beats already in the past
        std::size_t discontinuities = 0; // capture buffers that did not follow
        BeatParticleFilter::Stats filter;
    };

    Stats stats() const;

private:
    // The filter's beat window, widened if the ODF is too coarse to support the
    // configured one. See the definition.
    static ParticleFilterConfig resolveFilter(const LiveConfig& config);

    bool gatedAt(double frame_time_sec) const;

    LiveConfig config_;
    dsp::Odf odf_;
    BeatParticleFilter filter_;
    PhaseSync sync_;

    double manual_bpm_ = 0.0;
    bool acquired_ = false;

    double origin_sec_ = 0.0;  // stream time of the ODF's sample zero
    std::size_t consumed_ = 0;
    bool started_ = false;

    double onset_peak_ = 0.0;

    // A handful of pending gates is plenty: they are consumed within a beat of
    // being added, and a shell that has queued eight of them is not running.
    static constexpr std::size_t kGates = 8;
    std::array<double, kGates> gate_start_{};
    std::array<double, kGates> gate_end_{};
    std::size_t gate_next_ = 0;

    bool locked_ = false;
    bool published_ = false;
    double last_beat_sec_ = 0.0;
    double held_period_sec_ = 0.5;

    Stats stats_;
};

}  // namespace tiktak::tracking
