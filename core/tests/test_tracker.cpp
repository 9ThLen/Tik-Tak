#include "analysis/tracker.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "support.hpp"

using tiktak::analysis::BeatResult;
using tiktak::analysis::BeatTracker;
using tiktak::analysis::TempoConfig;
using tiktak::analysis::TrackerConfig;
using tiktak::test::impulseTrain;

namespace {

constexpr double kFps = 48000.0 / 512.0;

double spacingForBpm(double bpm) { return 60.0 * kFps / bpm; }

std::vector<double> frameTimes(std::size_t n) {
    std::vector<double> times(n);
    for (std::size_t i = 0; i < n; ++i) times[i] = static_cast<double>(i) / kFps;
    return times;
}

BeatTracker makeTracker(double tightness = 100.0, bool trim = true) {
    TrackerConfig cfg;
    cfg.tightness = tightness;
    cfg.trim = trim;
    return BeatTracker{cfg, TempoConfig{}};
}

// Fraction of true beats matched by an estimated beat within `tolerance`. This
// is the recall half of the mir_eval F-measure, which the Python harness scores
// properly; here it only has to be good enough to tell "tracked it" from "did
// not".
double recall(const std::vector<double>& truth, const std::vector<double>& found,
              double tolerance = 0.07) {
    if (truth.empty()) return 0.0;
    std::size_t hits = 0;
    for (double expected : truth) {
        const bool matched = std::any_of(found.begin(), found.end(), [&](double beat) {
            return std::abs(beat - expected) <= tolerance;
        });
        if (matched) ++hits;
    }
    return static_cast<double>(hits) / static_cast<double>(truth.size());
}

std::vector<double> expectedBeats(double bpm, std::size_t frames, double offsetFrames = 0.0) {
    std::vector<double> beats;
    const double spacing = spacingForBpm(bpm);
    for (double p = offsetFrames; p < static_cast<double>(frames); p += spacing) {
        beats.push_back(p / kFps);
    }
    return beats;
}

}  // namespace

TEST(TrackerConfigValidity, RejectsANonPositiveTightness) {
    TrackerConfig cfg;
    EXPECT_TRUE(cfg.valid());
    cfg.tightness = 0.0;
    EXPECT_FALSE(cfg.valid());
    cfg.tightness = -1.0;
    EXPECT_FALSE(cfg.valid());
}

TEST(Tracker, PlacesBeatsOnTheOnsetsOfASteadyTrain) {
    constexpr std::size_t kFrames = 2000;
    const std::vector<double> odf = impulseTrain(kFrames, spacingForBpm(120.0));
    const std::vector<double> times = frameTimes(kFrames);

    BeatTracker tracker = makeTracker();
    const BeatResult result = tracker.track(odf.data(), times.data(), kFrames, kFps);

    EXPECT_GT(recall(expectedBeats(120.0, kFrames), result.beats), 0.95);
    EXPECT_EQ(result.beats.size(), result.frames.size());
}

TEST(Tracker, FollowsThePhaseOfTheMusicRatherThanTheStartOfTheFile) {
    // The whole point of the sync feature: the grid must land on the melody's
    // beats, not on frame zero. A third of a beat of offset is enough that a
    // tracker ignoring phase would fail outright.
    constexpr std::size_t kFrames = 2000;
    const double spacing = spacingForBpm(120.0);
    const double offset = spacing / 3.0;

    const std::vector<double> odf = impulseTrain(kFrames, spacing, offset);
    const std::vector<double> times = frameTimes(kFrames);

    BeatTracker tracker = makeTracker();
    const BeatResult result = tracker.track(odf.data(), times.data(), kFrames, kFps);

    EXPECT_GT(recall(expectedBeats(120.0, kFrames, offset), result.beats), 0.95);
}

TEST(Tracker, HonoursAFixedTempoInsteadOfEstimatingOne) {
    constexpr std::size_t kFrames = 2000;
    const std::vector<double> odf = impulseTrain(kFrames, spacingForBpm(120.0));
    const std::vector<double> times = frameTimes(kFrames);

    BeatTracker tracker = makeTracker();
    const BeatResult result = tracker.track(odf.data(), times.data(), kFrames, kFps, 90.0);

    EXPECT_DOUBLE_EQ(result.bpm, 90.0);
    EXPECT_DOUBLE_EQ(result.tempo_confidence, 1.0);   // the caller asserted it
}

// Manual mode has to survive a user who taps in a tempo a few per cent off. The
// transition penalty is soft, so the onset term can still pull the grid onto
// the real beats as long as the true interval is inside the search window of
// half to twice the given period.
TEST(Tracker, RecoversFromASlightlyWrongManualTempo) {
    constexpr std::size_t kFrames = 2000;
    const std::vector<double> odf = impulseTrain(kFrames, spacingForBpm(120.0));
    const std::vector<double> times = frameTimes(kFrames);

    BeatTracker tracker = makeTracker();
    const BeatResult result = tracker.track(odf.data(), times.data(), kFrames, kFps, 113.0);

    EXPECT_GT(recall(expectedBeats(120.0, kFrames), result.beats), 0.9);
}

TEST(Tracker, AHintBeyondTheSearchWindowDoesNotRecover) {
    // At 40 against 120 the true interval is a third of the given period, well
    // outside the [period/2, 2*period] window the dynamic programme searches.
    // Documented rather than papered over: manual mode must sanity-check the
    // tempo the user typed instead of assuming the tracker will cope.
    constexpr std::size_t kFrames = 2000;
    const std::vector<double> odf = impulseTrain(kFrames, spacingForBpm(120.0));
    const std::vector<double> times = frameTimes(kFrames);

    BeatTracker tracker = makeTracker();
    const BeatResult result = tracker.track(odf.data(), times.data(), kFrames, kFps, 40.0);

    EXPECT_LT(recall(expectedBeats(120.0, kFrames), result.beats), 0.6);
}

TEST(Tracker, KeepsTheGridSteadyThroughAGapInTheOnsets) {
    // A held note or a rest leaves the onset function empty for a bar or two.
    // The tempo-consistency term is what carries the grid across it; without it
    // the beats would simply stop.
    constexpr std::size_t kFrames = 2000;
    const double spacing = spacingForBpm(120.0);
    std::vector<double> odf = impulseTrain(kFrames, spacing);

    const auto gap_start = static_cast<std::size_t>(spacing * 8.0);
    const auto gap_end = static_cast<std::size_t>(spacing * 12.0);
    std::fill(odf.begin() + static_cast<std::ptrdiff_t>(gap_start),
              odf.begin() + static_cast<std::ptrdiff_t>(gap_end), 0.0);

    const std::vector<double> times = frameTimes(kFrames);
    BeatTracker tracker = makeTracker();
    const BeatResult result = tracker.track(odf.data(), times.data(), kFrames, kFps);

    // Beats inside the silent stretch are what this is about.
    const double from = static_cast<double>(gap_start) / kFps;
    const double to = static_cast<double>(gap_end) / kFps;
    const auto inside = std::count_if(result.beats.begin(), result.beats.end(),
                                      [&](double t) { return t > from && t < to; });
    EXPECT_GE(inside, 3);
    EXPECT_GT(recall(expectedBeats(120.0, kFrames), result.beats), 0.9);
}

TEST(Tracker, TrimmingRemovesBeatsInventedOverLeadingSilence) {
    // The dynamic programme extends its grid into silence to keep the sequence
    // regular. Those beats are not wrong musically, but clicking through the
    // count-in before the music starts is exactly what the app must not do.
    constexpr std::size_t kFrames = 2000;
    const double spacing = spacingForBpm(120.0);
    const auto lead = static_cast<std::size_t>(spacing * 6.0);

    std::vector<double> odf(kFrames, 0.0);
    const std::vector<double> train = impulseTrain(kFrames - lead, spacing);
    std::copy(train.begin(), train.end(), odf.begin() + static_cast<std::ptrdiff_t>(lead));

    const std::vector<double> times = frameTimes(kFrames);
    const double music_starts = static_cast<double>(lead) / kFps;

    BeatTracker trimming = makeTracker(100.0, true);
    const BeatResult trimmed = trimming.track(odf.data(), times.data(), kFrames, kFps);

    BeatTracker keeping = makeTracker(100.0, false);
    const BeatResult untrimmed = keeping.track(odf.data(), times.data(), kFrames, kFps);

    const auto before = [&](const BeatResult& r) {
        return std::count_if(r.beats.begin(), r.beats.end(),
                             [&](double t) { return t < music_starts - 0.05; });
    };

    EXPECT_GT(before(untrimmed), before(trimmed));
    EXPECT_LE(before(trimmed), 1);
}

TEST(Tracker, ProducesMonotonicNonRepeatingBeats) {
    constexpr std::size_t kFrames = 2000;
    const std::vector<double> odf = impulseTrain(kFrames, spacingForBpm(140.0));
    const std::vector<double> times = frameTimes(kFrames);

    BeatTracker tracker = makeTracker();
    const BeatResult result = tracker.track(odf.data(), times.data(), kFrames, kFps);

    ASSERT_GT(result.beats.size(), 2u);
    for (std::size_t i = 1; i < result.beats.size(); ++i) {
        EXPECT_GT(result.beats[i], result.beats[i - 1]) << "at beat " << i;
    }
    for (std::size_t i = 1; i < result.frames.size(); ++i) {
        EXPECT_GT(result.frames[i], result.frames[i - 1]) << "at beat " << i;
    }
}

TEST(Tracker, ReturnsNothingRatherThanGuessingOnSilence) {
    constexpr std::size_t kFrames = 1000;
    const std::vector<double> odf(kFrames, 0.0);
    const std::vector<double> times = frameTimes(kFrames);

    BeatTracker tracker = makeTracker();
    const BeatResult result = tracker.track(odf.data(), times.data(), kFrames, kFps);

    EXPECT_TRUE(result.beats.empty());
    EXPECT_DOUBLE_EQ(result.bpm, 120.0);          // the prior centre, not a claim
    EXPECT_DOUBLE_EQ(result.tempo_confidence, 0.0);
}

TEST(Tracker, RejectsMalformedInput) {
    const std::vector<double> odf(100, 1.0);
    const std::vector<double> times = frameTimes(100);
    BeatTracker tracker = makeTracker();

    EXPECT_TRUE(tracker.track(nullptr, times.data(), 100, kFps).beats.empty());
    EXPECT_TRUE(tracker.track(odf.data(), nullptr, 100, kFps).beats.empty());
    EXPECT_TRUE(tracker.track(odf.data(), times.data(), 100, 0.0).beats.empty());
    EXPECT_TRUE(tracker.track(odf.data(), times.data(), 2, kFps).beats.empty());
}

TEST(Tracker, LooserTightnessFollowsTempoChangeMoreClosely) {
    // A rubato singer needs the grid to bend; a rock band needs it rigid. The
    // knob has to actually do that, so this pins the direction of its effect.
    constexpr std::size_t kFrames = 3000;
    std::vector<double> odf(kFrames, 0.0);

    // Accelerating: the interval shrinks by 0.02 frames per beat.
    double spacing = spacingForBpm(100.0);
    std::vector<double> truth;
    for (double p = 0.0; p < static_cast<double>(kFrames); p += spacing) {
        odf[static_cast<std::size_t>(p + 0.5)] = 1.0;
        truth.push_back(p / kFps);
        spacing -= 0.02;
    }

    const std::vector<double> times = frameTimes(kFrames);
    BeatTracker rigid = makeTracker(2000.0);
    BeatTracker loose = makeTracker(20.0);

    const double rigid_recall =
        recall(truth, rigid.track(odf.data(), times.data(), kFrames, kFps).beats);
    const double loose_recall =
        recall(truth, loose.track(odf.data(), times.data(), kFrames, kFps).beats);

    EXPECT_GE(loose_recall, rigid_recall);
}
