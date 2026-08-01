#pragma once

#include <string>
#include <vector>

namespace tiktak::desktop {

// A human tapping along, compared against the grid the core found.
//
// Every number this project has about the offline path comes from an
// annotation someone else made, on recordings chosen by whoever assembled a
// corpus. That is the right way to measure, and it has one blind spot: it
// cannot be pointed at *this* recording, the one that sounded wrong yesterday,
// and it never says what the failure sounded like. A person tapping a key is a
// worse annotator than a corpus and an immeasurably better one than nothing for
// the recording in front of them.
//
// **The click is off unless it is asked for, and that is the whole design.**
// Playing our own click while the listener taps turns the test into a duet:
// they follow the grid, the taps confirm it, and a grid on the off-beat passes
// as easily as a right one. The bench plays the music and nothing else.
//
// What it can and cannot see. It cannot measure accuracy: a person taps early
// by a few tens of milliseconds, unevenly, and drops beats while turning a
// page. It can see the three failures that matter and that a corpus F-measure
// hides — a grid on the off-beat, a grid at half or double the tempo a listener
// would choose, and a grid that drifts away over the length of a song. All
// three survive the noise of human tapping because all three are large.
struct TapComparison {
    std::size_t taps = 0;
    std::size_t matched = 0;         // within the tolerance of a grid beat
    double tap_period_sec = 0.0;     // median interval between taps
    double grid_period_sec = 0.0;
    double octave_ratio = 0.0;       // tap period / grid period
    double median_offset_sec = 0.0;  // taps minus the nearest beat
    double offset_spread_sec = 0.0;  // interquartile spread of that
    // Share of taps landing within the tolerance once each person's own
    // constant anticipation is removed. Reported beside the raw share because
    // the difference between the two is the difference between "the grid is
    // late" and "the grid is elsewhere".
    double matched_after_offset = 0.0;
    std::string verdict;
};

// `taps` and `beats` are seconds on the same clock. Both are sorted here.
TapComparison compareTaps(std::vector<double> taps, std::vector<double> beats,
                          double tolerance_sec);

}  // namespace tiktak::desktop
