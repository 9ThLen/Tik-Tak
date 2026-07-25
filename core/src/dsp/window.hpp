#pragma once

#include <cstddef>
#include <vector>

namespace tiktak::dsp {

// Periodic Hann window — the DFT-even variant, w[n] = 0.5 * (1 - cos(2*pi*n/N)).
//
// Periodic rather than symmetric because this window is used for STFT analysis,
// where the periodic form is the one that satisfies constant overlap-add at
// hop = N/2 and N/4.
std::vector<float> hannWindow(std::size_t size);

}  // namespace tiktak::dsp
