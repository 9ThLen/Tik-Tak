#include "dsp/window.hpp"

#include <cmath>

namespace tiktak::dsp {
namespace {
constexpr double kTwoPi = 6.283185307179586476925286766559;
}

std::vector<float> hannWindow(std::size_t size) {
    std::vector<float> window(size);
    if (size == 0) return window;
    if (size == 1) {
        window[0] = 1.0f;
        return window;
    }

    for (std::size_t n = 0; n < size; ++n) {
        const double phase = kTwoPi * static_cast<double>(n) / static_cast<double>(size);
        window[n] = static_cast<float>(0.5 * (1.0 - std::cos(phase)));
    }
    return window;
}

}  // namespace tiktak::dsp
