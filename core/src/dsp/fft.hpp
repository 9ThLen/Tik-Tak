#pragma once

#include <cstddef>
#include <vector>

namespace tiktak::dsp {

// Radix-2 Cooley-Tukey FFT.
//
// Deliberately a plain, portable implementation rather than vDSP or PFFFT: the
// interface is what matters at this stage, and keeping it dependency-free lets
// the core cross-compile anywhere from day one. Profiling decides later whether
// a SIMD backend is worth pulling in behind this same interface.
//
// Templated on the scalar type because the two callers need different
// precision. The STFT runs per audio block on values that came from 24-bit
// samples, where float is ample and costs half the memory traffic. Tempo
// estimation autocorrelates thousands of ODF frames and then picks the largest
// of hundreds of nearly tied candidates, and there float32 rounding is enough
// to flip the winner — so it uses double. Instantiated for float and double.
//
// Every buffer is allocated in the constructor; the transform methods allocate
// nothing and are safe to call from an audio callback.
template <typename T>
class FftT {
public:
    // `size` must be a power of two and >= 2.
    explicit FftT(std::size_t size);

    std::size_t size() const { return size_; }

    // Number of non-redundant bins for a real input: size/2 + 1.
    std::size_t spectrumSize() const { return size_ / 2 + 1; }

    // Forward transform of `size` real samples.
    // `re` and `im` each receive spectrumSize() values.
    void forwardReal(const T* input, T* re, T* im);

    // Magnitude spectrum of `size` real samples; `mag` receives spectrumSize().
    void magnitudeReal(const T* input, T* mag);

    // Complex transform in place over `size` values. Used mainly to verify the
    // forward path against a round trip. The inverse is scaled by 1/size, so
    // forward followed by inverse reproduces the input.
    void forward(T* re, T* im);
    void inverse(T* re, T* im);

    static bool isPowerOfTwo(std::size_t v) { return v >= 2 && (v & (v - 1)) == 0; }

    // Smallest power of two >= v, at least 2. Callers sizing a zero-padded
    // transform need this and should not each reinvent it.
    static std::size_t nextPowerOfTwo(std::size_t v);

private:
    void transform(T* re, T* im, bool inverseTransform);

    std::size_t size_;
    std::vector<std::size_t> reversed_;  // bit-reversal permutation
    std::vector<T> cos_;                 // cos(2*pi*k/size), k < size/2
    std::vector<T> sin_;                 // sin(2*pi*k/size), k < size/2
    std::vector<T> scratchRe_;
    std::vector<T> scratchIm_;
};

extern template class FftT<float>;
extern template class FftT<double>;

using Fft = FftT<float>;
using Fft64 = FftT<double>;

}  // namespace tiktak::dsp
