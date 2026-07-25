#include "dsp/fft.hpp"

#include <cassert>
#include <cmath>
#include <utility>

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

template <typename T>
std::size_t FftT<T>::nextPowerOfTwo(std::size_t v) {
    std::size_t size = 2;
    while (size < v) size <<= 1;
    return size;
}

template <typename T>
FftT<T>::FftT(std::size_t size) : size_(size) {
    assert(isPowerOfTwo(size) && "FFT size must be a power of two >= 2");

    int bits = 0;
    while ((std::size_t{1} << bits) < size_) ++bits;

    reversed_.resize(size_);
    for (std::size_t i = 0; i < size_; ++i) {
        reversed_[i] = reverseBits(i, bits);
    }

    // The twiddles are computed in double regardless of T: they are the one
    // part of the transform whose error is systematic rather than random, so
    // rounding them to the working type only at the end keeps the float
    // instantiation as accurate as its storage allows.
    const std::size_t half = size_ / 2;
    cos_.resize(half);
    sin_.resize(half);
    for (std::size_t k = 0; k < half; ++k) {
        const double angle = kTwoPi * static_cast<double>(k) / static_cast<double>(size_);
        cos_[k] = static_cast<T>(std::cos(angle));
        sin_[k] = static_cast<T>(std::sin(angle));
    }

    scratchRe_.resize(size_);
    scratchIm_.resize(size_);
}

template <typename T>
void FftT<T>::transform(T* re, T* im, bool inverseTransform) {
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
    const T sign = inverseTransform ? T{1} : T{-1};

    for (std::size_t len = 2; len <= size_; len <<= 1) {
        const std::size_t half = len >> 1;
        const std::size_t stride = size_ / len;

        for (std::size_t base = 0; base < size_; base += len) {
            for (std::size_t k = 0; k < half; ++k) {
                const std::size_t twiddle = k * stride;
                const T wr = cos_[twiddle];
                const T wi = sign * sin_[twiddle];

                const std::size_t a = base + k;
                const std::size_t b = a + half;

                const T tr = re[b] * wr - im[b] * wi;
                const T ti = re[b] * wi + im[b] * wr;

                re[b] = re[a] - tr;
                im[b] = im[a] - ti;
                re[a] += tr;
                im[a] += ti;
            }
        }
    }

    if (inverseTransform) {
        const T scale = T{1} / static_cast<T>(size_);
        for (std::size_t i = 0; i < size_; ++i) {
            re[i] *= scale;
            im[i] *= scale;
        }
    }
}

template <typename T>
void FftT<T>::forward(T* re, T* im) {
    transform(re, im, false);
}

template <typename T>
void FftT<T>::inverse(T* re, T* im) {
    transform(re, im, true);
}

template <typename T>
void FftT<T>::forwardReal(const T* input, T* re, T* im) {
    // A real-input-specific transform would halve this work. Not worth the
    // complexity until profiling says the FFT is hot.
    for (std::size_t i = 0; i < size_; ++i) {
        scratchRe_[i] = input[i];
        scratchIm_[i] = T{0};
    }

    transform(scratchRe_.data(), scratchIm_.data(), false);

    const std::size_t bins = spectrumSize();
    for (std::size_t k = 0; k < bins; ++k) {
        re[k] = scratchRe_[k];
        im[k] = scratchIm_[k];
    }
}

template <typename T>
void FftT<T>::magnitudeReal(const T* input, T* mag) {
    for (std::size_t i = 0; i < size_; ++i) {
        scratchRe_[i] = input[i];
        scratchIm_[i] = T{0};
    }

    transform(scratchRe_.data(), scratchIm_.data(), false);

    const std::size_t bins = spectrumSize();
    for (std::size_t k = 0; k < bins; ++k) {
        const T r = scratchRe_[k];
        const T i = scratchIm_[k];
        mag[k] = std::sqrt(r * r + i * i);
    }
}

template class FftT<float>;
template class FftT<double>;

}  // namespace tiktak::dsp
