#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace tiktak::desktop {

// Writes 16-bit mono PCM. Deliberately hand-rolled and tiny: the harness only
// ever needs to dump what it rendered or recorded, and pulling in an encoder
// for that would make the harness's dependencies larger than the core's.
//
// Samples outside [-1, 1] are clipped rather than wrapped, because a wrap turns
// an overload into something that looks like a completely different signal.
bool writeWav(const std::string& path, const std::vector<float>& samples,
              double sample_rate);

}  // namespace tiktak::desktop
