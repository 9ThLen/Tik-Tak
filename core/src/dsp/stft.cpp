#include "dsp/stft.hpp"

#include "dsp/window.hpp"

namespace tiktak::dsp {

Stft::Stft(std::size_t frameSize, std::size_t hopSize)
    : frameSize_(frameSize), hopSize_(hopSize), fft_(frameSize) {
    assert(Fft::isPowerOfTwo(frameSize) && "STFT frame size must be a power of two");
    assert(hopSize >= 1 && hopSize <= frameSize && "hop must be within the window");

    window_ = hannWindow(frameSize_);
    buffer_.assign(frameSize_, 0.0f);
    windowed_.assign(frameSize_, 0.0f);
    magnitude_.assign(fft_.spectrumSize(), 0.0f);
}

void Stft::reset() {
    std::fill(buffer_.begin(), buffer_.end(), 0.0f);
    std::fill(magnitude_.begin(), magnitude_.end(), 0.0f);
    fill_ = 0;
    frameStartSample_ = 0;
}

void Stft::computeSpectrum() {
    for (std::size_t i = 0; i < frameSize_; ++i) {
        windowed_[i] = buffer_[i] * window_[i];
    }
    fft_.magnitudeReal(windowed_.data(), magnitude_.data());
}

}  // namespace tiktak::dsp
