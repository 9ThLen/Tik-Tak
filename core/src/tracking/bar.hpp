#pragma once

#include <cstddef>
#include <vector>

#include "analysis/downbeat.hpp"

namespace tiktak::tracking {

// Bar length and bar line from a causal downbeat activation.
//
// The offline path answers this over a whole recording. This answers it from a
// trailing window, on the beats the live tracker has already handed out, and it
// deliberately does *not* contain a second copy of the decision: the scoring,
// the hypothesis set, the priors and the margins are analysis::resolveMeter's,
// reached through the same seam a learned backend reaches it through. Two
// implementations of "where does the bar start" would diverge, and the
// divergence would look like a metronome that accents one beat on a file and a
// different one through the microphone.
//
// **Why a window and not a running decision.** A meter that does not change has
// exactly one degree of freedom beyond its length — the phase — so fixing the
// pair fixes every bar line in the piece. That is the argument the offline path
// already rests on, and it is what makes the causal version cheap: resolve over
// what has been heard, then *count* forward. Nothing has to be decided about a
// beat that has not happened yet, which is the only reason a bar line can be
// accented at the instant it falls rather than 70 ms after it.
//
// ------------------------------------------------------------- what it needs
//
// A downbeat activation, per frame. That means the learned front end: the
// built-in cue backend scores a beat from chroma distance and a low band that
// the live ODF does not compute, so with spectral flux there is no bar evidence
// at all and this reports no answer rather than a guess.
//
// ------------------------------------------------ the limitation worth naming
//
// A beat's salience is the peak of the activation in a window centred on it,
// and a metronome clicking through a speaker is gated out over very nearly the
// same window. So when the tracker can hear its own click, the bar evidence is
// mostly gone.
//
// That is not a bug to route around. The accent is *derived from* the bar
// decision, so feeding ungated frames here would let an accented click confirm
// the bar line it was placed by — a tighter self-confirmation loop than the one
// the click gate already exists to break for tempo, and a worse one, because
// nothing in the output would look wrong. The honest position is that the bar
// display works when the tracker is not listening to its own click, and that
// covers the case it was asked for: a user watching the beat on screen while a
// track plays.
class BarTracker {
public:
    struct Config {
        // Half-width of the window a beat's salience is the peak of. 0.07 is
        // the beat-tracking tolerance, and is the same number and the same
        // reduction the research backends use — see sample_at_beats in
        // research/eval/backends.py for why the peak and not the nearest frame
        // or the mean. A different value here would make the live answer
        // incomparable with every offline number already measured.
        double salience_window_sec = 0.07;

        // How many scored beats the resolver sees. Thirty-two is eight bars of
        // four: long enough for min_bars, short enough that a section change is
        // out of the window within about fifteen seconds at 120 BPM.
        std::size_t window_beats = 32;

        // Fewest scored beats before any answer is attempted. Below three bars
        // the resolver refuses anyway; this refuses earlier and for the same
        // reason, so that a two-beat window cannot produce a confident-looking
        // margin out of two numbers.
        std::size_t min_beats = 12;

        analysis::DownbeatConfig resolver;

        bool valid() const;
    };

    explicit BarTracker(const Config& config);

    const Config& config() const { return config_; }

    void reset();

    // One activation frame, in the caller's clock. Real-time safe: appends to
    // buffers sized in the constructor and allocates nothing.
    //
    // Frames withheld by the click gate must be withheld here too — see the
    // limitation above. The caller owns that decision because only the caller
    // knows about the gate.
    void observe(double time_sec, double downbeat);

    // A beat has been handed out at `beat_sec`, carrying the tracker's own
    // monotonically increasing index. Real-time safe.
    //
    // Called when the beat is *published*, which is before it sounds, so the
    // beat sits pending until its salience window has passed.
    void addBeat(double beat_sec, long long index);

    // Scores every pending beat whose window has closed by `now_sec` and, if
    // any did, resolves again.
    //
    // **Not real-time safe**: analysis::resolveMeter allocates. Call it from
    // the same place a beat is consumed, not from an audio callback. Returns
    // true when a new resolution was produced.
    bool update(double now_sec);

    // What is currently being shown: the last bar length actually decided, or
    // 0 before there was one. Not necessarily the last window's answer — see
    // result() for that, and update() for why they are kept apart.
    int beatsPerBar() const { return held_beats_per_bar_; }

    // Position of the beat with that index within its bar, 0 for a bar line.
    // -1 when nothing has been decided, and for an index older than the window
    // the decision was made from.
    int positionOf(long long index) const;

    // The last window's answer, whatever it was, including "nothing". This is
    // the diagnostic view: its margins are fresh, and they can say the evidence
    // has gone even while beatsPerBar() still reports what it last decided.
    const analysis::DownbeatResult& result() const { return result_; }

    bool confident() const {
        return result_.confident(config_.resolver.min_phase_margin,
                                 config_.resolver.min_meter_margin);
    }

    // How many beats the last resolution saw. Zero before the first one.
    std::size_t scoredBeats() const { return scored_; }

private:
    // Peak activation over [centre - w, centre + w], or 0 when the ring holds
    // no frame in it — 0 rather than dropping the beat, because the resolver
    // refuses a salience vector whose length does not match its beat list, and
    // it is right to.
    double peakAround(double centre_sec) const;

    Config config_;

    // Activation history, a ring sized once. Long enough to cover a beat's
    // whole window even when the beat was published before the frames that
    // surround it arrived.
    static constexpr std::size_t kFrames = 64;
    std::vector<double> frame_time_;
    std::vector<double> frame_value_;
    std::size_t frame_next_ = 0;
    std::size_t frames_ = 0;

    // Beats published but not yet scored, oldest first.
    static constexpr std::size_t kPending = 8;
    std::vector<double> pending_time_;
    std::vector<long long> pending_index_;
    std::size_t pending_ = 0;

    // The scored window, oldest first, at most window_beats entries.
    std::vector<double> beat_time_;
    std::vector<double> salience_;
    std::vector<long long> beat_index_;

    // The last window's answer, fresh each update.
    analysis::DownbeatResult result_;

    // What is being displayed: the last answer that actually decided something,
    // with its bar line in the tracker's own beat numbering.
    int held_beats_per_bar_ = 0;
    long long held_downbeat_index_ = 0;
    bool decided_ = false;
    std::size_t scored_ = 0;
};

}  // namespace tiktak::tracking
