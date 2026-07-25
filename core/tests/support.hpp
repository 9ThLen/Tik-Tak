#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

namespace tiktak::test {

constexpr double kTwoPi = 6.283185307179586476925286766559;

inline std::vector<float> sine(std::size_t n, double freqHz, double sampleRate,
                               float amplitude = 1.0f, double phase = 0.0) {
    std::vector<float> out(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double t = static_cast<double>(i) / sampleRate;
        out[i] = amplitude * static_cast<float>(std::sin(kTwoPi * freqHz * t + phase));
    }
    return out;
}

inline std::vector<float> silence(std::size_t n) { return std::vector<float>(n, 0.0f); }

// A short burst of a sine, faded in over one sample and out over `n` samples —
// the crude stand-in for a percussive attack used by the onset tests.
inline void addBurst(std::vector<float>& buffer, std::size_t at, std::size_t length,
                     double freqHz, double sampleRate, float amplitude = 1.0f) {
    for (std::size_t i = 0; i < length && at + i < buffer.size(); ++i) {
        const double t = static_cast<double>(i) / sampleRate;
        const double envelope = 1.0 - static_cast<double>(i) / static_cast<double>(length);
        buffer[at + i] += amplitude * static_cast<float>(
                                          envelope * std::sin(kTwoPi * freqHz * t));
    }
}

}  // namespace tiktak::test
