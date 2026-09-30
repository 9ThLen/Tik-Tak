#pragma once

namespace tiktak::tracking {

// Holding the metrical level against a brief change of mind.
//
// The activation-tempo estimator re-decides the octave every second from six
// seconds of evidence, and on full-length songs its wrong-level episodes are
// single confident flips lasting eight or nine seconds at the median. This
// holds the level the estimator has been reporting until a proposal an octave
// away has persisted for `seconds`; inside the octave it follows freely, so a
// band drifting from 128 to 132 BPM is never a proposal and never held.
//
// The semantics are exactly the research seam's (dump_analysis
// --live-octave-debounce), because those are what was measured. On RWC, one
// seed per arm against a five-seed noise floor (SD 0.005 usable, 0.016
// episode-free; PREREGISTERED_quickfix_diagnostics.md, Q2):
//
//     hold     usable    episode-free   correct time
//      10 s    +0.013       +0.027         +0.002
//      20 s    +0.028       +0.047         -0.001
//      30 s    +0.030       +0.050         -0.005
//      60 s    +0.033       +0.055         -0.012
//     never    +0.038       +0.060         -0.034
//
// Twenty seconds takes most of what holding is worth before it starts to cost
// correct time. RWC is a development corpus and those are single seeds, so
// this is off by default until a registered run on Harmonix confirms it.
//
// Note the one fragility, kept on purpose because it is what was measured: an
// estimate back inside the held octave restarts the clock, so a proposal that
// flickers never accumulates its seconds.
class OctaveHold {
public:
    explicit OctaveHold(double seconds = 0.0) : seconds_(seconds) {}

    bool enabled() const { return seconds_ > 0.0; }
    double seconds() const { return seconds_; }

    // The tempo to anchor to, given what the estimator measured at `time_sec`.
    double resolve(double time_sec, double measured_bpm);

    // Forgets the held level: a new song, not a gap in one.
    void reset();

    double heldBpm() const { return committed_bpm_; }

    // Near a power of two, at the 8% the labels and the live benchmark use.
    // Plain rounding in log space would call every ratio in (1.41, 2.83) a
    // doubling, and a 3:2 tempo relation sits inside it. 0 means "the same
    // octave".
    static int octaveIndex(double measured_bpm, double committed_bpm);

private:
    double seconds_ = 0.0;
    double committed_bpm_ = 0.0;
    double disagreement_since_sec_ = -1.0;
};

}  // namespace tiktak::tracking
