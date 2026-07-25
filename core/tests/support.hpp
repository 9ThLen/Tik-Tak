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

// A metronome-like click track: a percussive burst on every beat, alternating
// between a low "kick" and a higher "snare" so the material has the two-beat
// pattern that makes naive autocorrelation pick the wrong period.
inline std::vector<float> clickTrack(double bpm, double durationSec, double sampleRate,
                                     double leadSec = 0.0) {
    const auto total = static_cast<std::size_t>(durationSec * sampleRate);
    std::vector<float> out(total, 0.0f);

    const double interval = 60.0 / bpm;
    const auto burst = static_cast<std::size_t>(0.03 * sampleRate);

    std::size_t beat = 0;
    for (double t = leadSec; t < durationSec; t += interval, ++beat) {
        const auto at = static_cast<std::size_t>(t * sampleRate);
        const bool strong = beat % 2 == 0;
        addBurst(out, at, burst, strong ? 60.0 : 900.0, sampleRate, strong ? 1.0f : 0.6f);
        // A little broadband energy, so the attack looks like a hit rather than
        // a pure tone switching on.
        addBurst(out, at, burst / 3, strong ? 1800.0 : 5000.0, sampleRate, 0.4f);
    }
    return out;
}

// The onset function such a track produces, built directly: an impulse train
// with `spacing` frames between beats. Lets the tempo and tracker tests work on
// a known-exact input instead of inheriting the ODF's own behaviour.
inline std::vector<double> impulseTrain(std::size_t frames, double spacing, double offset = 0.0,
                                        double strong = 1.0, double weak = 0.6) {
    std::vector<double> out(frames, 0.0);
    std::size_t beat = 0;
    for (double position = offset; position < static_cast<double>(frames);
         position += spacing, ++beat) {
        const auto index = static_cast<std::size_t>(position + 0.5);
        if (index < frames) out[index] = beat % 2 == 0 ? strong : weak;
    }
    return out;
}

}  // namespace tiktak::test
