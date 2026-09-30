#include "tracking/octave_hold.hpp"

#include <cmath>

namespace tiktak::tracking {

int OctaveHold::octaveIndex(double measured_bpm, double committed_bpm) {
    if (!(measured_bpm > 0.0) || !(committed_bpm > 0.0)) return 0;
    const double exponent = std::log2(measured_bpm / committed_bpm);
    const int k = static_cast<int>(std::lround(exponent));
    if (k == 0) return 0;
    return std::fabs(exponent - k) > std::log2(1.08) ? 0 : k;
}

double OctaveHold::resolve(double time_sec, double measured_bpm) {
    if (!enabled() || !(measured_bpm > 0.0)) return measured_bpm;
    if (!(committed_bpm_ > 0.0)) {
        committed_bpm_ = measured_bpm;
        return measured_bpm;
    }
    const int k = octaveIndex(measured_bpm, committed_bpm_);
    if (k == 0) {
        committed_bpm_ = measured_bpm;
        disagreement_since_sec_ = -1.0;
        return measured_bpm;
    }
    if (disagreement_since_sec_ < 0.0) disagreement_since_sec_ = time_sec;
    if (time_sec - disagreement_since_sec_ < seconds_) {
        // Held: the measured tempo moved back by the whole octaves it jumped,
        // so the tempo inside the octave is still the estimator's.
        return measured_bpm * std::exp2(-static_cast<double>(k));
    }
    committed_bpm_ = measured_bpm;
    disagreement_since_sec_ = -1.0;
    return measured_bpm;
}

void OctaveHold::reset() {
    committed_bpm_ = 0.0;
    disagreement_since_sec_ = -1.0;
}

}  // namespace tiktak::tracking
