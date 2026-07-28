#pragma once

#include <cstddef>
#include <vector>

namespace tiktak::dsp {

// Rational resampling with an anti-aliasing filter.
//
// Written because both models want 22050 Hz and devices hand over 44100 or
// 48000. The obvious shortcut — linear interpolation between neighbouring
// samples — is not a resampler: decimating 48 kHz that way folds everything
// above 11 kHz back into the band, and cymbals become a rattle underneath the
// music exactly where a beat tracker is looking for its onsets. The BeatNet
// path uses interpolation deliberately, to match the reference every published
// number was measured through; this is for the paths that have no such
// obligation.
//
// Polyphase, so the cost is the filter length divided by the upsampling factor
// rather than the filter length: 48 kHz to 22.05 kHz is 147/320, a 6401-tap
// prototype, and about 44 multiplies per output sample.
//
// Whole-signal rather than streaming. Its callers analyse a file that is
// already in memory, and a streaming resampler has a state and an edge policy
// that nothing here would exercise.
class Resampler {
public:
    // Ratio is reduced internally, so 48000 -> 22050 becomes 147/320.
    Resampler(double fromRate, double toRate);

    // Output samples a signal of `count` inputs will produce.
    std::size_t outputLength(std::size_t count) const;

    std::vector<float> apply(const float* samples, std::size_t count) const;

    // The reduced ratio, exposed for tests and diagnostics.
    std::size_t up() const { return up_; }
    std::size_t down() const { return down_; }
    std::size_t taps() const { return filter_.size(); }

private:
    std::size_t up_ = 1;
    std::size_t down_ = 1;
    std::vector<double> filter_;   // prototype low-pass, already scaled by up_
    std::size_t half_ = 0;         // the prototype's group delay, in taps
};

}  // namespace tiktak::dsp
