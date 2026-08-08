// The bar in the live path: what it decides, what it counts, and what it must
// not touch.
//
// Nothing here is a claim about real music. The material is synthetic and the
// downbeat channel is supplied by the test, so what is under test is the
// plumbing and the counting — that the salience is sampled the way the offline
// backends sample it, that the resolver's answer is carried into the tracker's
// own beat numbering, that a bar line decided from past beats names future ones
// by arithmetic, and that a user's meter outranks all of it.
//
// Whether BeatNet's downbeat head can carry a *causal* metre decision on real
// recordings is unmeasured and is not asserted anywhere in this file. The
// offline audit found the head carries the metre decisively (83% against a 30%
// null) and does not carry the octave; that was whole-recording and does not
// transfer to a trailing window by argument.

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "support.hpp"
#include "tracking/bar.hpp"
#include "tracking/live.hpp"

using tiktak::tracking::BarTracker;
using tiktak::tracking::LiveConfig;
using tiktak::tracking::LiveTracker;

namespace {

constexpr double kFps = 50.0;
constexpr double kRate = 48000.0;

BarTracker::Config barConfig() {
    BarTracker::Config config;
    config.min_beats = 12;
    config.window_beats = 32;
    return config;
}

// Where a drive left off. Carried explicitly because a second segment that
// restarted the clock and the beat numbering would hand the tracker times
// running backwards and indices it had already used — which is not an
// ambiguous passage, it is a corrupt one.
struct Drive {
    double time = 0.0;
    long long index = 0;
};

// Feeds `bars` bars of `m` beats at `bpm`, with the downbeat channel high on
// every bar line and low elsewhere. Returns the beat times handed in.
std::vector<double> driveBarsFrom(BarTracker& bar, Drive& at, int m, int bars,
                                  double bpm, double high = 0.9,
                                  double low = 0.05, int phase = 0) {
    const double period = 60.0 / bpm;
    const int beats = m * bars;
    const double start = at.time;
    const long long base = at.index;
    std::vector<double> times;
    times.reserve(static_cast<std::size_t>(beats));

    int beat = 0;
    // A second of slack past the last beat, so its salience window closes and
    // it is scored before the segment ends.
    const double end = start + period * beats + 1.0;
    for (double t = start; t < end; t += 1.0 / kFps) {
        // A beat is published a little before it falls, as takeBeat does. No
        // upper bound on the comparison: beat zero's publish time is already
        // in the past when the loop starts, and a window would miss it and
        // then stall on it forever.
        if (beat < beats && t >= start + period * beat - 0.05) {
            const double when = start + period * beat;
            bar.addBeat(when, at.index++);
            times.push_back(when);
            ++beat;
        }
        // The activation: a spike on the frame nearest each bar line, with the
        // bar phase continuing across segments.
        const double since = (t - start) / period;
        const double nearest = std::round(since);
        double value = 0.0;
        if (std::fabs(since - nearest) < 0.5 / (kFps * period)) {
            const long long n = base + static_cast<long long>(nearest);
            value = ((n - phase) % m + m) % m == 0 ? high : low;
        }
        bar.observe(t, value);
        bar.update(t);
    }
    at.time = end;
    return times;
}

std::vector<double> driveBars(BarTracker& bar, int m, int bars, double bpm,
                              double high = 0.9, double low = 0.05,
                              int phase = 0) {
    Drive at;
    return driveBarsFrom(bar, at, m, bars, bpm, high, low, phase);
}

}  // namespace

// -------------------------------------------------------- the tracker alone

TEST(BarTracker, SaysNothingBeforeItHasSeenEnough) {
    BarTracker bar(barConfig());
    EXPECT_EQ(bar.beatsPerBar(), 0);
    EXPECT_EQ(bar.positionOf(0), -1);
    EXPECT_FALSE(bar.confident());

    // Two bars of four is eight beats, below min_beats and below the
    // resolver's own min_bars for most meters.
    driveBars(bar, 4, 2, 120.0);
    EXPECT_EQ(bar.beatsPerBar(), 0);
}

TEST(BarTracker, FindsFourFourAndItsBarLine) {
    BarTracker bar(barConfig());
    driveBars(bar, 4, 6, 120.0);

    EXPECT_EQ(bar.beatsPerBar(), 4);
    EXPECT_TRUE(bar.confident());
    // Beat 0 was a bar line, so every fourth beat from it is too.
    EXPECT_EQ(bar.positionOf(0), 0);
    EXPECT_EQ(bar.positionOf(1), 1);
    EXPECT_EQ(bar.positionOf(4), 0);
    EXPECT_EQ(bar.positionOf(7), 3);
}

TEST(BarTracker, FindsAShiftedBarLine) {
    // The failure a listener notices first is not the wrong length, it is the
    // right length starting on the wrong beat.
    BarTracker bar(barConfig());
    driveBars(bar, 4, 6, 120.0, 0.9, 0.05, /*phase=*/2);

    ASSERT_EQ(bar.beatsPerBar(), 4);
    EXPECT_EQ(bar.positionOf(2), 0);
    EXPECT_EQ(bar.positionOf(6), 0);
    EXPECT_EQ(bar.positionOf(0), 2);
}

TEST(BarTracker, FindsThree) {
    BarTracker bar(barConfig());
    driveBars(bar, 3, 8, 120.0);
    EXPECT_EQ(bar.beatsPerBar(), 3);
    EXPECT_EQ(bar.positionOf(3), 0);
    EXPECT_EQ(bar.positionOf(4), 1);
}

TEST(BarTracker, NamesBeatsItHasNotSeenYet) {
    // The whole reason a bar line can be accented at the instant it falls: the
    // decision comes from the past and the counting goes forward.
    BarTracker bar(barConfig());
    driveBars(bar, 4, 6, 120.0);
    ASSERT_EQ(bar.beatsPerBar(), 4);

    EXPECT_EQ(bar.positionOf(1000), 0);
    EXPECT_EQ(bar.positionOf(1001), 1);
    EXPECT_EQ(bar.positionOf(1003), 3);
}

TEST(BarTracker, AFlatChannelDecidesNothing) {
    // No bar-level pattern in the audio means no bar lines, not a bar length
    // invented from rounding noise.
    BarTracker bar(barConfig());
    driveBars(bar, 4, 8, 120.0, /*high=*/0.5, /*low=*/0.5);

    EXPECT_FALSE(bar.confident());
    EXPECT_EQ(bar.result().beats_per_bar, 0);
}

TEST(BarTracker, KeepsTheLastDecisionThroughAmbiguity) {
    // A window that decides nothing does not erase one that did: a bar or two
    // of ambiguity mid-song is not evidence the meter changed, and blanking the
    // display on it would flicker.
    //
    // What is asserted is the invariant and not the held *value*. Between a
    // structured passage and a flat one the window holds both at once, and a
    // resolver deciding something else from that mixture has decided it from
    // real evidence — that is a different question, and pinning a number here
    // would be testing the material rather than the mechanism.
    BarTracker bar(barConfig());
    Drive at;
    driveBarsFrom(bar, at, 4, 8, 120.0);
    ASSERT_EQ(bar.beatsPerBar(), 4);

    BarTracker flat(barConfig());
    driveBars(flat, 4, 8, 120.0, 0.5, 0.5);
    ASSERT_EQ(flat.beatsPerBar(), 0);

    // Long enough that the flat beats have replaced the whole 32-beat window.
    driveBarsFrom(bar, at, 4, 10, 120.0, 0.5, 0.5);

    // The fresh answer says it does not know...
    EXPECT_EQ(bar.result().beats_per_bar, 0);
    EXPECT_FALSE(bar.confident());
    // ...and something is still on screen, flagged as stale by the line above.
    EXPECT_GT(bar.beatsPerBar(), 0);
    EXPECT_GE(bar.positionOf(at.index), 0);
}

TEST(BarTracker, ResetForgetsEverything) {
    BarTracker bar(barConfig());
    driveBars(bar, 4, 6, 120.0);
    ASSERT_EQ(bar.beatsPerBar(), 4);

    bar.reset();
    EXPECT_EQ(bar.beatsPerBar(), 0);
    EXPECT_EQ(bar.positionOf(0), -1);
    EXPECT_EQ(bar.scoredBeats(), 0u);
}

TEST(BarTracker, AnUnscoredBeatStillCounts) {
    // A beat whose window holds no frame gets zero salience rather than being
    // dropped: the resolver refuses a salience vector that does not match its
    // beat list, and a dropped beat would also move every later beat one
    // position round the bar.
    BarTracker bar(barConfig());
    for (long long i = 0; i < 16; ++i) {
        bar.addBeat(0.5 * static_cast<double>(i), i);
    }
    bar.observe(100.0, 0.0);  // far from every beat
    bar.update(100.0);
    EXPECT_EQ(bar.scoredBeats(), 0u);  // window not full enough after the drops

    BarTracker other(barConfig());
    const auto times = driveBars(other, 4, 6, 120.0);
    EXPECT_EQ(other.scoredBeats(), times.size());
}

// ------------------------------------------------------- through LiveTracker

namespace {

LiveConfig liveBarConfig() {
    LiveConfig config = tiktak::tracking::liveConfigFor(kRate);
    config.bar_tracking = true;
    config.bar = barConfig();
    return config;
}

// Alternating strong and weak beats so the tempo is unambiguous, with the
// downbeat channel high every `m` beats.
//
// `from_sec` is not decoration: a second call that started at zero again would
// feed the tracker a clock running backwards, and takeBeat would hand out
// nothing at all for the rest of the test.
std::vector<int> positionsOf(LiveTracker& tracker, double bpm, int m,
                             double seconds, double from_sec = 0.0) {
    const double period = 60.0 / bpm;
    const auto frames = static_cast<std::size_t>(seconds * kFps);
    std::vector<int> out;
    for (std::size_t i = 0; i < frames; ++i) {
        const double time = from_sec + static_cast<double>(i) / kFps;
        const double since = time / period;
        const double nearest = std::round(since);
        double beat = 0.02;
        double down = 0.0;
        if (std::fabs(since - nearest) < 0.5 / (kFps * period)) {
            const int n = static_cast<int>(nearest);
            beat = n % 2 == 0 ? 0.95 : 0.45;
            down = n % m == 0 ? 0.9 : 0.05;
        }
        tracker.observe(time, beat, down);
        double when = 0.0;
        if (tracker.takeBeat(time, 0.05, &when)) out.push_back(tracker.barPosition());
    }
    return out;
}

}  // namespace

TEST(LiveBar, IsOffUnlessAskedFor) {
    // Every published live number was measured without this, and the default
    // has to stay where those numbers were taken.
    LiveConfig config = tiktak::tracking::liveConfigFor(kRate);
    EXPECT_FALSE(config.bar_tracking);

    LiveTracker tracker(config);
    positionsOf(tracker, 120.0, 4, 20.0);
    EXPECT_EQ(tracker.beatsPerBar(), 0);
    EXPECT_EQ(tracker.barPosition(), -1);
}

TEST(LiveBar, DoesNotMoveASingleBeat) {
    // The bar decision reads a channel nothing else reads and writes nothing
    // back. If it ever moves a beat, that is a leak and not a feature.
    LiveTracker with(liveBarConfig());
    LiveTracker without(tiktak::tracking::liveConfigFor(kRate));

    const double period = 60.0 / 120.0;
    const auto frames = static_cast<std::size_t>(30.0 * kFps);
    std::vector<double> a;
    std::vector<double> b;
    for (std::size_t i = 0; i < frames; ++i) {
        const double time = static_cast<double>(i) / kFps;
        const double since = time / period;
        const double nearest = std::round(since);
        double beat = 0.02;
        double down = 0.0;
        if (std::fabs(since - nearest) < 0.5 / (kFps * period)) {
            const int n = static_cast<int>(nearest);
            beat = n % 2 == 0 ? 0.95 : 0.45;
            down = n % 4 == 0 ? 0.9 : 0.05;
        }
        with.observe(time, beat, down);
        without.observe(time, beat);
        double when = 0.0;
        if (with.takeBeat(time, 0.05, &when)) a.push_back(when);
        if (without.takeBeat(time, 0.05, &when)) b.push_back(when);
    }

    ASSERT_EQ(a.size(), b.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        EXPECT_DOUBLE_EQ(a[i], b[i]) << "beat " << i;
    }
}

TEST(LiveBar, CountsFourAcrossHandedOutBeats) {
    LiveTracker tracker(liveBarConfig());
    const auto positions = positionsOf(tracker, 120.0, 4, 30.0);

    ASSERT_GT(positions.size(), 20u);
    ASSERT_EQ(tracker.beatsPerBar(), 4);

    // Once decided, the positions must simply cycle. Which beat the cycle
    // starts on is the resolver's business and is tested above; that it never
    // stutters or repeats is this one's.
    std::size_t start = positions.size() - 12;
    for (std::size_t i = start + 1; i < positions.size(); ++i) {
        ASSERT_GE(positions[i], 0);
        EXPECT_EQ(positions[i], (positions[i - 1] + 1) % 4) << "beat " << i;
    }
}

// --------------------------------------------------------- the user's meter

TEST(LiveBar, AUserMeterOutranksTheTracker) {
    LiveTracker tracker(liveBarConfig());
    positionsOf(tracker, 120.0, 4, 30.0);
    ASSERT_EQ(tracker.beatsPerBar(), 4);

    tracker.setMeter(3, 0);
    EXPECT_EQ(tracker.beatsPerBar(), 3);
    EXPECT_TRUE(tracker.meterIsManual());
    // Never in doubt: the user is the authority on how they are counting.
    EXPECT_TRUE(tracker.meterConfident());
}

TEST(LiveBar, TapOnTheOneNamesTheNextBeat) {
    LiveTracker tracker(liveBarConfig());
    positionsOf(tracker, 120.0, 4, 20.0);

    tracker.setMeter(4, 0);
    const auto after = positionsOf(tracker, 120.0, 4, 6.0, 20.0);
    ASSERT_GE(after.size(), 4u);
    EXPECT_EQ(after[0], 0);
    EXPECT_EQ(after[1], 1);
    EXPECT_EQ(after[2], 2);
    EXPECT_EQ(after[3], 3);
}

TEST(LiveBar, ClearingHandsTheQuestionBack) {
    LiveTracker tracker(liveBarConfig());
    positionsOf(tracker, 120.0, 4, 30.0);
    ASSERT_EQ(tracker.beatsPerBar(), 4);

    tracker.setMeter(3, 0);
    ASSERT_EQ(tracker.beatsPerBar(), 3);

    tracker.clearMeter();
    EXPECT_FALSE(tracker.meterIsManual());
    EXPECT_EQ(tracker.beatsPerBar(), 4);
}

TEST(LiveBar, TheUserMeterNeedsNeitherModelNorFlag) {
    // Counting is arithmetic. A user who has told us the meter has supplied the
    // evidence themselves, so this must work with bar tracking off and with the
    // built-in front end.
    LiveTracker tracker(tiktak::tracking::liveConfigFor(kRate));
    positionsOf(tracker, 120.0, 4, 20.0);

    tracker.setMeter(4, 0);
    const auto after = positionsOf(tracker, 120.0, 4, 6.0, 20.0);
    ASSERT_GE(after.size(), 4u);
    EXPECT_EQ(after[0], 0);
    EXPECT_EQ(after[3], 3);
}

TEST(LiveBar, TheUserMeterSurvivesResetAndKeepsCounting) {
    // A reset forgets audio, not the user -- and it must not restart the bar
    // either, because a dropped capture buffer is not a new bar line.
    LiveTracker tracker(liveBarConfig());
    positionsOf(tracker, 120.0, 4, 20.0);
    tracker.setMeter(4, 0);
    const auto before = positionsOf(tracker, 120.0, 4, 4.0, 20.0);
    ASSERT_GE(before.size(), 2u);
    const int last = before.back();

    tracker.reset();
    EXPECT_TRUE(tracker.meterIsManual());
    EXPECT_EQ(tracker.beatsPerBar(), 4);

    const auto after = positionsOf(tracker, 120.0, 4, 20.0, 24.0);
    ASSERT_GE(after.size(), 1u);
    EXPECT_EQ(after[0], (last + 1) % 4);
}

TEST(LiveBar, ResetForgetsTheTrackersOwnMeter) {
    LiveTracker tracker(liveBarConfig());
    positionsOf(tracker, 120.0, 4, 30.0);
    ASSERT_EQ(tracker.beatsPerBar(), 4);

    tracker.reset();
    EXPECT_EQ(tracker.beatsPerBar(), 0);
    EXPECT_EQ(tracker.barPosition(), -1);
}
