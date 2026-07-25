#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <vector>

#include "dsp/fft.hpp"

namespace tiktak::dsp {

// Streaming short-time Fourier transform.
//
// Accepts arbitrarily sized blocks of samples and emits one magnitude spectrum
// per hop. Storage is exactly one analysis window regardless of block size, and
// process() allocates nothing, so it is safe to drive from an audio callback.
class Stft {
public:
    // `hopSize` must be in [1, frameSize]; `frameSize` must be a power of two.
    Stft(std::size_t frameSize, std::size_t hopSize);

    std::size_t frameSize() const { return frameSize_; }
    std::size_t hopSize() const { return hopSize_; }
    std::size_t spectrumSize() const { return fft_.spectrumSize(); }

    // Samples already buffered towards the next frame.
    std::size_t pending() const { return fill_; }

    // Frames that feeding `n` more samples would produce right now.
    std::size_t framesAvailable(std::size_t n) const {
        const std::size_t total = fill_ + n;
        if (total < frameSize_) return 0;
        return (total - frameSize_) / hopSize_ + 1;
    }

    // Drops buffered samples and restarts the frame clock at zero.
    void reset();

    // Feeds `n` samples, invoking
    //     onFrame(const float* magnitude, std::size_t bins, std::size_t frameStartSample)
    // once per completed frame. `frameStartSample` is the index of the window's
    // first sample counted from the last reset().
    template <typename Fn>
    void process(const float* samples, std::size_t n, Fn&& onFrame) {
        std::size_t pos = 0;
        while (pos < n) {
            const std::size_t take = std::min(frameSize_ - fill_, n - pos);
            std::memcpy(buffer_.data() + fill_, samples + pos, take * sizeof(float));
            fill_ += take;
            pos += take;

            if (fill_ < frameSize_) break;

            computeSpectrum();
            onFrame(magnitude_.data(), magnitude_.size(), frameStartSample_);

            // Slide the window forward by one hop, keeping the overlap.
            frameStartSample_ += hopSize_;
            const std::size_t keep = frameSize_ - hopSize_;
            if (keep > 0) {
                std::memmove(buffer_.data(), buffer_.data() + hopSize_, keep * sizeof(float));
            }
            fill_ = keep;
        }
    }

private:
    void computeSpectrum();

    std::size_t frameSize_;
    std::size_t hopSize_;
    Fft fft_;
    std::vector<float> window_;
    std::vector<float> buffer_;     // one analysis window of input
    std::vector<float> windowed_;   // buffer_ * window_
    std::vector<float> magnitude_;
    std::size_t fill_ = 0;
    std::size_t frameStartSample_ = 0;
};

}  // namespace tiktak::dsp
