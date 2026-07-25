#pragma once

#include <cstddef>
#include <vector>

namespace tiktak::dsp {

double hzToMel(double hz);
double melToHz(double mel);

// Triangular mel filterbank over a real FFT magnitude spectrum.
//
// Filters are peak-normalised (each triangle reaches 1.0 at its centre) rather
// than area-normalised. The onset detector takes a log of the output and then
// differences it in time, so what matters is that a band's response is
// comparable with its own past — not that bands are comparable with each other
// in absolute energy. Peak normalisation keeps wide high-frequency bands from
// dominating the flux sum purely because they are wide.
//
// Filters are stored sparsely: most triangles touch only a handful of FFT bins,
// so a dense (bands x bins) matrix would be almost entirely zeros.
class MelFilterbank {
public:
    // `fftSize` is the transform length; the spectrum passed to apply() has
    // fftSize/2 + 1 bins. `minHz`/`maxHz` bound the filterbank; maxHz is clamped
    // to Nyquist.
    MelFilterbank(std::size_t fftSize, double sampleRate, std::size_t bands,
                  double minHz, double maxHz);

    std::size_t bands() const { return bands_; }
    std::size_t spectrumSize() const { return fftSize_ / 2 + 1; }

    // Centre frequency of each band, Hz. Monotonically increasing.
    const std::vector<double>& centreFrequencies() const { return centres_; }

    // Index of the first band whose centre is at or above `hz`, or bands() if
    // there is none. Used to split the ODF into low/high bands.
    std::size_t bandAtOrAbove(double hz) const;

    // Projects a magnitude spectrum of spectrumSize() values onto bands()
    // outputs. Allocates nothing.
    void apply(const float* magnitude, float* out) const;

private:
    std::size_t fftSize_;
    std::size_t bands_;
    std::vector<double> centres_;
    std::vector<std::size_t> start_;    // first FFT bin of each filter
    std::vector<std::size_t> length_;   // bin count of each filter
    std::vector<float> weights_;        // concatenated triangle weights
    std::vector<std::size_t> offset_;   // where each filter starts in weights_
};

}  // namespace tiktak::dsp
