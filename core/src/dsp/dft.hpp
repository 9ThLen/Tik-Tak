#pragma once

#include <cstddef>
#include <vector>

#include "dsp/fft.hpp"

namespace tiktak::dsp {

// Magnitude spectrum of a real signal at any transform length, including the
// lengths a radix-2 FFT cannot do.
//
// This exists for one caller and one reason. BeatNet's front end analyses
// 1411-sample frames — 64 ms at 22050 Hz — and 1411 is prime to nothing useful.
// The obvious way out is to zero-pad up to 2048, and it is the wrong way out:
// padding changes the bin frequencies, so the filterbank built on those bins
// integrates different bins with different weights, and the network is handed
// features that are not the features it was trained on. The trained weights are
// only worth having if the input is the input they were fitted to, so the
// transform bends and the model does not.
//
// Bluestein's algorithm gets there: a DFT of length N is rewritten as a
// convolution, and a convolution of length N is a power-of-two FFT of length
// M >= 2N-1 with zero padding — padding that is now an implementation detail of
// the convolution rather than a change to the spectrum. For N = 1411 that is
// M = 4096, so a frame costs two 4096-point complex transforms instead of one
// 2048-point real one. At 50 frames a second that is a few MFLOP/s.
//
// In double throughout, and not from caution: the chirp phases run to n^2 for
// n approaching N, and in float the angle for the last few hundred samples has
// lost most of its significant bits.
//
// Every buffer is allocated in the constructor; magnitude() allocates nothing.
class RealDft {
public:
    // `size` is the true transform length and may be anything >= 2.
    explicit RealDft(std::size_t size);

    std::size_t size() const { return size_; }
    std::size_t spectrumSize() const { return size_ / 2 + 1; }

    // Magnitude of the first size/2 + 1 bins of `input`, which holds size
    // real samples.
    void magnitude(const float* input, float* out);

private:
    std::size_t size_;
    std::size_t convolution_;   // M, the padded power of two
    Fft64 fft_;

    std::vector<double> chirp_re_, chirp_im_;       // exp(-i*pi*n^2/N), n < N
    std::vector<double> kernel_re_, kernel_im_;     // FFT of the conjugate chirp
    std::vector<double> work_re_, work_im_;
};

}  // namespace tiktak::dsp
