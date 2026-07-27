#pragma once

#include <cstddef>

namespace tiktak::dsp {

// Finding a signal you already have, in a recording of it.
//
// This is latency calibration and nothing else: the app plays a click it
// generated, the microphone hears it some milliseconds later, and the delay
// between the two is the round trip every other piece of timing arithmetic is
// built on. Because the click is known exactly, the problem is not detection
// in general — it is alignment of a known waveform, which is the one case
// where correlating against a template is the right answer rather than a
// heuristic.
//
// What it replaces, and why. The harness used to take the first sample above
// thirty per cent of the recording window's own peak. That is exact down a
// loopback cable and falls apart in a room: at 0 dB SNR against the click, a
// noise burst anywhere in the window sets the peak and the first crossing
// lands on the noise instead. Measured on synthetic mixtures, the threshold
// rule's median error goes from 0.1 ms at 30 dB to 174 ms at 0 dB, while
// correlation stays exact throughout.
//
// What it does not solve. In a reverberant room the correlation peak sits on
// the energy centroid of the room's response rather than on the direct sound,
// so the estimate is biased late by an amount that depends on the room. A
// synthetic room put that bias near 15 ms, but the number is a property of
// the simulation and should not be quoted as a property of this function.
// Shortening the template does not fix it and makes matters worse. Anyone
// calibrating in a live room rather than over a cable should know the result
// is late by an unmeasured amount; a cable, where there is no room, remains
// the way to get a trustworthy number.
//
// Offline component: allocates nothing, but is O(window x template) and not
// for an audio callback.

struct MatchResult {
    // Offset into the window, in samples, where the template best aligns.
    // Negative when nothing correlated well enough to report.
    double offset_samples = -1.0;

    // Peak correlation relative to the template's own energy. A perfect,
    // unattenuated copy scores 1.0; anything quieter scores proportionally
    // less, so this is a level measurement rather than a quality one.
    double strength = 0.0;

    bool found() const { return offset_samples >= 0.0; }
};

// Correlates `window` against `template_` and reports where they line up best.
//
// `min_strength` refuses a match that correlates too weakly to be the signal
// rather than a coincidence — the recording where the speaker was muted has to
// come back as "not found" rather than as an arbitrary offset, because a
// confident wrong latency is worse for every downstream calculation than none.
//
// The returned offset is interpolated between samples: the correlation of a
// smooth signal has a smooth peak, and fitting a parabola through the peak and
// its neighbours locates it to a fraction of a sample. At 48 kHz one sample is
// 21 microseconds, so this matters not for precision but because it removes a
// half-sample quantisation bias from a number that is later subtracted from
// every beat time.
MatchResult findKnownSignal(const float* window, std::size_t window_frames,
                            const float* template_, std::size_t template_frames,
                            double min_strength = 0.2);

}  // namespace tiktak::dsp
