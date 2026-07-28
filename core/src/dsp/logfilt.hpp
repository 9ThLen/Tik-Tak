#pragma once

#include <cstddef>
#include <vector>

namespace tiktak::dsp {

// Triangular filterbank on a logarithmic frequency scale, area-normalised.
//
// The sibling of MelFilterbank, and different from it in both axes on purpose,
// because the two serve different consumers:
//
// * *Placement.* Mel spacing is a psychoacoustic curve; this is plain
//   equal-temperament spacing — a fixed number of bands per octave off a
//   reference pitch — so a filter covers the same musical interval everywhere.
// * *Normalisation.* MelFilterbank normalises each triangle to unit peak,
//   because its output is logged and differenced against its own past and
//   should not let wide bands dominate the flux sum. This normalises to unit
//   area, because its output goes to a network whose weights were fitted to
//   area-normalised bands, and a band that arrives scaled by its own width is
//   a band the first layer has never seen.
//
// That second point is the reason this class exists at all rather than a flag
// on MelFilterbank: the convention here is not ours to choose. It reproduces
// madmom's LogarithmicFilterbank, down to which FFT bin each triangle starts
// on, because that is what BeatNet was trained through.
//
// Filters are stored sparsely, as in MelFilterbank: at 24 bands an octave the
// low triangles touch a single bin each and a dense matrix would be almost all
// zeros.
class LogFilterbank {
public:
    // `dftSize` is the true transform length; the spectrum passed to apply()
    // has dftSize/2 + 1 bins. `bandsPerOctave` counts filters per octave off
    // `refHz`, and `minHz`/`maxHz` bound the range.
    LogFilterbank(std::size_t dftSize, double sampleRate, std::size_t bandsPerOctave,
                  double minHz, double maxHz, double refHz);

    std::size_t bands() const { return bands_; }
    std::size_t spectrumSize() const { return dftSize_ / 2 + 1; }

    // Projects a magnitude spectrum of spectrumSize() values onto bands()
    // outputs. Allocates nothing.
    void apply(const float* magnitude, float* out) const;

private:
    std::size_t dftSize_;
    std::size_t bands_ = 0;
    std::vector<std::size_t> start_;   // first FFT bin of each filter
    std::vector<std::size_t> length_;  // bin count of each filter
    std::vector<std::size_t> offset_;  // where each filter starts in weights_
    std::vector<float> weights_;       // concatenated triangle weights
};

}  // namespace tiktak::dsp
