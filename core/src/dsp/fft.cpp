#include "dsp/fft.hpp"

#include <cassert>
#include <cmath>

namespace tiktak::dsp {
namespace {

constexpr double kTwoPi = 6.283185307179586476925286766559;

std::size_t reverseBits(std::size_t value, int bits) {
    std::size_t result = 0;
    for (int i = 0; i < bits; ++i) {
        result = (result << 1) | (value & 1u);
        value >>= 1;
    }
    return result;
}

}  // namespace

Fft::Fft(std::size_t size) : size_(size) {
    assert(isPowerOfTwo(size) && "FFT size must be a power of two >= 2");

    int bits = 0;
    while ((std::size_t{1} << bits) < size_) ++bits;

    reversed_.resize(size_);
    for (std::size_t i = 0; i < size_; ++i) {
        reversed_[i] = reverseBits(i, bits);
    }

    const std::size_t half = size_ / 2;
    cos_.resize(half);
    sin_.resize(half);
    for (std::size_t k = 0; k < half; ++k) {
        const double angle = kTwoPi * static_cast<double>(k) / static_cast<double>(size_);
        cos_[k] = static_cast<float>(std::cos(angle));
        sin_[k] = static_cast<float>(std::sin(angle));
    }

    scratchRe_.resize(size_);
    scratchIm_.resize(size_);
}

void Fft::transform(float* re, float* im, bool inverseTransform) {
    // Reorder into bit-reversed index order so the butterflies below can run in
    // place with unit stride.
    for (std::size_t i = 0; i < size_; ++i) {
        const std::size_t j = reversed_[i];
        if (i < j) {
            std::swap(re[i], re[j]);
            std::swap(im[i], im[j]);
        }
    }

    // The twiddle table holds one full turn at the transform's own resolution,
    // so each stage samples it with stride size_/len.
    const float sign = inverseTransform ? 1.0f : -1.0f;

    for (std::size_t len = 2; len <= size_; len <<= 1) {
        const std::size_t half = len >> 1;
        const std::size_t stride = size_ / len;

        for (std::size_t base = 0; base < size_; base += len) {
            for (std::size_t k = 0; k < half; ++k) {
                const std::size_t twiddle = k * stride;
                const float wr = cos_[twiddle];
                const float wi = sign * sin_[twiddle];

                const std::size_t a = base + k;
                const std::size_t b = a + half;

                const float tr = re[b] * wr - im[b] * wi;
                const float ti = re[b] * wi + im[b] * wr;

                re[b] = re[a] - tr;
                im[b] = im[a] - ti;
                re[a] += tr;
                im[a] += ti;
            }
        }
    }

    if (inverseTransform) {
        const float scale = 1.0f / static_cast<float>(size_);
        for (std::size_t i = 0; i < size_; ++i) {
            re[i] *= scale;
            im[i] *= scale;
        }
    }
}

void Fft::forward(float* re, float* im) { transform(re, im, false); }

void Fft::inverse(float* re, float* im) { transform(re, im, true); }

void Fft::forwardReal(const float* input, float* re, float* im) {
    // A real-input-specific transform would halve this work. Not worth the
    // complexity until profiling says the FFT is hot.
    for (std::size_t i = 0; i < size_; ++i) {
        scratchRe_[i] = input[i];
        scratchIm_[i] = 0.0f;
    }

    transform(scratchRe_.data(), scratchIm_.data(), false);

    const std::size_t bins = spectrumSize();
    for (std::size_t k = 0; k < bins; ++k) {
        re[k] = scratchRe_[k];
        im[k] = scratchIm_[k];
    }
}

void Fft::magnitudeReal(const float* input, float* mag) {
    for (std::size_t i = 0; i < size_; ++i) {
        scratchRe_[i] = input[i];
        scratchIm_[i] = 0.0f;
    }

    transform(scratchRe_.data(), scratchIm_.data(), false);

    const std::size_t bins = spectrumSize();
    for (std::size_t k = 0; k < bins; ++k) {
        const float r = scratchRe_[k];
        const float i = scratchIm_[k];
        mag[k] = std::sqrt(r * r + i * i);
    }
}

}  // namespace tiktak::dsp
