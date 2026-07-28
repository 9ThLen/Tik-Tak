#include "dsp/dft.hpp"

#include <cassert>
#include <cmath>

namespace tiktak::dsp {
namespace {

constexpr double kPi = 3.14159265358979323846;

// exp(-i*pi*n^2/N), computed from n^2 reduced modulo 2N.
//
// The reduction is the whole point. n^2 for n near 1411 is around two million,
// and handing that to sin/cos means the argument's own rounding error is
// already a sizeable fraction of a radian by the time the library reduces it.
// n^2 mod 2N stays under 2822, where the angle is exact to the last bit.
void chirpAt(std::size_t n, std::size_t size, double* re, double* im) {
    const std::size_t modulus = 2 * size;
    const std::size_t squared = (n % modulus) * (n % modulus) % modulus;
    const double angle = -kPi * static_cast<double>(squared) / static_cast<double>(size);
    *re = std::cos(angle);
    *im = std::sin(angle);
}

}  // namespace

RealDft::RealDft(std::size_t size)
    : size_(size < 2 ? 2 : size),
      convolution_(Fft64::nextPowerOfTwo(2 * (size < 2 ? 2 : size) - 1)),
      fft_(convolution_),
      chirp_re_(size_), chirp_im_(size_),
      kernel_re_(convolution_, 0.0), kernel_im_(convolution_, 0.0),
      work_re_(convolution_), work_im_(convolution_) {
    for (std::size_t n = 0; n < size_; ++n) {
        chirpAt(n, size_, &chirp_re_[n], &chirp_im_[n]);
    }

    // The convolution kernel is the conjugate chirp, mirrored into the tail so
    // that the linear convolution the FFT computes is the cyclic one the
    // algorithm wants. Everything between the two copies stays zero, which is
    // what the padding to M >= 2N-1 buys.
    kernel_re_[0] = chirp_re_[0];
    kernel_im_[0] = -chirp_im_[0];
    for (std::size_t n = 1; n < size_; ++n) {
        kernel_re_[n] = chirp_re_[n];
        kernel_im_[n] = -chirp_im_[n];
        kernel_re_[convolution_ - n] = chirp_re_[n];
        kernel_im_[convolution_ - n] = -chirp_im_[n];
    }
    fft_.forward(kernel_re_.data(), kernel_im_.data());
}

void RealDft::magnitude(const float* input, float* out) {
    assert(input != nullptr && out != nullptr);

    for (std::size_t n = 0; n < size_; ++n) {
        const double x = static_cast<double>(input[n]);
        work_re_[n] = x * chirp_re_[n];
        work_im_[n] = x * chirp_im_[n];
    }
    for (std::size_t n = size_; n < convolution_; ++n) {
        work_re_[n] = 0.0;
        work_im_[n] = 0.0;
    }

    fft_.forward(work_re_.data(), work_im_.data());
    for (std::size_t k = 0; k < convolution_; ++k) {
        const double re = work_re_[k] * kernel_re_[k] - work_im_[k] * kernel_im_[k];
        const double im = work_re_[k] * kernel_im_[k] + work_im_[k] * kernel_re_[k];
        work_re_[k] = re;
        work_im_[k] = im;
    }
    // The inverse already carries the 1/M, so the convolution comes out scaled.
    fft_.inverse(work_re_.data(), work_im_.data());

    // Post-multiplying by the chirp again turns the convolution back into the
    // transform. Only the magnitude survives, so the final phase is dropped,
    // but the multiply still has to happen — it is not a phase-only factor once
    // the two complex parts are mixed.
    for (std::size_t k = 0; k < spectrumSize(); ++k) {
        const double re = work_re_[k] * chirp_re_[k] - work_im_[k] * chirp_im_[k];
        const double im = work_re_[k] * chirp_im_[k] + work_im_[k] * chirp_re_[k];
        out[k] = static_cast<float>(std::sqrt(re * re + im * im));
    }
}

}  // namespace tiktak::dsp
