#pragma once

#include <cstddef>
#include <vector>

namespace tiktak::dsp {

// Radix-2 Cooley-Tukey FFT.
//
// Deliberately a plain, portable implementation rather than vDSP or PFFFT: the
// interface is what matters at this stage, and keeping it dependency-free lets
// the core cross-compile anywhere from day one. Profiling in Phase 3 decides
// whether a SIMD backend is worth pulling in behind this same interface.
//
// Every buffer is allocated in the constructor; the transform methods allocate
// nothing and are safe to call from an audio callback.
class Fft {
public:
    // `size` must be a power of two and >= 2.
    explicit Fft(std::size_t size);

    std::size_t size() const { return size_; }

    // Number of non-redundant bins for a real input: size/2 + 1.
    std::size_t spectrumSize() const { return size_ / 2 + 1; }

    // Forward transform of `size` real samples.
    // `re` and `im` each receive spectrumSize() values.
    void forwardReal(const float* input, float* re, float* im);

    // Magnitude spectrum of `size` real samples; `mag` receives spectrumSize().
    void magnitudeReal(const float* input, float* mag);

    // Complex transform in place over `size` values. Used mainly to verify the
    // forward path against a round trip. The inverse is scaled by 1/size, so
    // forward followed by inverse reproduces the input.
    void forward(float* re, float* im);
    void inverse(float* re, float* im);

    static bool isPowerOfTwo(std::size_t v) { return v >= 2 && (v & (v - 1)) == 0; }

private:
    void transform(float* re, float* im, bool inverseTransform);

    std::size_t size_;
    std::vector<std::size_t> reversed_;  // bit-reversal permutation
    std::vector<float> cos_;             // cos(2*pi*k/size), k < size/2
    std::vector<float> sin_;             // sin(2*pi*k/size), k < size/2
    std::vector<float> scratchRe_;
    std::vector<float> scratchIm_;
};

}  // namespace tiktak::dsp
