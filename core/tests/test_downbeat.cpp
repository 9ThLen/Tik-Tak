#include "analysis/downbeat.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <utility>
#include <vector>

namespace {

using tiktak::analysis::BeatFeature;
using tiktak::analysis::BeatFeatureInput;
using tiktak::analysis::beatFeatures;
using tiktak::analysis::DownbeatConfig;
using tiktak::analysis::findDownbeats;

constexpr double kBeat = 0.5;  // 120 BPM

// A run of beats in which every `per_bar`-th one, counted from `phase`, is
// marked by whichever cue the test is about. Everything else is flat, so a
// failure points at the decision and not at the mixture of cues.
enum class Cue { kLow, kAccent, kHarmony };

std::vector<BeatFeature> bars(int per_bar, int phase, int count, Cue cue, double strong = 1.0,
                              double weak = 0.2) {
    std::vector<BeatFeature> out(static_cast<std::size_t>(count));
    for (int i = 0; i < count; ++i) {
        BeatFeature& f = out[static_cast<std::size_t>(i)];
        f.time_sec = i * kBeat;
        const bool down = ((i - phase) % per_bar + per_bar) % per_bar == 0;
        const double v = down ? strong : weak;
        switch (cue) {
            case Cue::kLow: f.low = v; break;
            case Cue::kAccent: f.accent = v; break;
            case Cue::kHarmony: f.harmonic_change = v; break;
        }
    }
    return out;
}

TEST(Downbeat, FindsFourFourAndWhereTheBarStarts) {
    const auto result = findDownbeats(bars(4, 0, 32, Cue::kLow), DownbeatConfig{});

    EXPECT_EQ(result.beats_per_bar, 4);
    EXPECT_EQ(result.phase, 0);
    EXPECT_GT(result.strength, 1.0);
    EXPECT_GT(result.phase_margin, 1.0);

    ASSERT_EQ(result.downbeats.size(), 8u);
    for (std::size_t i = 0; i < result.downbeats.size(); ++i) {
        EXPECT_NEAR(result.downbeats[i], static_cast<double>(i) * 4 * kBeat, 1e-9);
    }
}

TEST(Downbeat, TheBarNeedNotStartWhereTheRecordingDoes) {
    // A track faded in mid-bar. Getting this wrong is the failure a listener
    // hears instantly — an accent on beat three.
    for (int phase = 0; phase < 4; ++phase) {
        const auto result = findDownbeats(bars(4, phase, 32, Cue::kLow), DownbeatConfig{});
        EXPECT_EQ(result.beats_per_bar, 4) << "phase " << phase;
        EXPECT_EQ(result.phase, phase) << "phase " << phase;
        EXPECT_NEAR(result.downbeats.front(), phase * kBeat, 1e-9) << "phase " << phase;
    }
}

TEST(Downbeat, AWaltzIsNotFourFour) {
    const auto result = findDownbeats(bars(3, 0, 30, Cue::kLow), DownbeatConfig{});

    EXPECT_EQ(result.beats_per_bar, 3);
    EXPECT_EQ(result.phase, 0);
    EXPECT_GT(result.phase_margin, 1.0);

    ASSERT_EQ(result.downbeats.size(), 10u);
    EXPECT_NEAR(result.downbeats[1], 3 * kBeat, 1e-9);
}

TEST(Downbeat, HalfABarIsNotABar) {
    // The reason the score is a contrast and not a sum. A four-four pattern
    // also has a strong beat every two beats — every other one of them. A sum
    // over the chosen beats would hand two-four the win for claiming twice as
    // many; the mean difference does not.
    const auto result = findDownbeats(bars(4, 0, 32, Cue::kLow), DownbeatConfig{});
    ASSERT_EQ(result.beats_per_bar, 4);

    ASSERT_GE(result.candidates.size(), 2u);
    EXPECT_EQ(result.candidates.front().beats_per_bar, 4);
    for (const auto& c : result.candidates) {
        if (c.beats_per_bar == 2) EXPECT_LT(c.score, result.candidates.front().score);
    }
}

TEST(Downbeat, AChordChangeIsEnoughOnItsOwn) {
    // Harmony alone, with the drums saying nothing. This is the cue that finds
    // the bar line in music with no kick on the one.
    const auto result = findDownbeats(bars(4, 2, 32, Cue::kHarmony), DownbeatConfig{});

    EXPECT_EQ(result.beats_per_bar, 4);
    EXPECT_EQ(result.phase, 2);
    EXPECT_GT(result.phase_margin, 1.0);
}

TEST(Downbeat, TheCuesAddUpRatherThanFight) {
    // Each cue alone is weak and noisy; together they should still land on the
    // same bar line, which is the only reason to carry three of them.
    auto features = bars(4, 1, 32, Cue::kLow, 0.6, 0.5);
    const auto harmony = bars(4, 1, 32, Cue::kHarmony, 0.6, 0.5);
    for (std::size_t i = 0; i < features.size(); ++i) {
        features[i].harmonic_change = harmony[i].harmonic_change;
    }

    const auto result = findDownbeats(features, DownbeatConfig{});
    EXPECT_EQ(result.beats_per_bar, 4);
    EXPECT_EQ(result.phase, 1);
}

TEST(Downbeat, MusicWithNoBarLineGetsNone) {
    // Every beat identical. There is a meter to name only because something has
    // to be returned; what matters is that the strength says not to believe it.
    std::vector<BeatFeature> flat(32);
    for (std::size_t i = 0; i < flat.size(); ++i) {
        flat[i].time_sec = static_cast<double>(i) * kBeat;
        flat[i].low = 0.4;
        flat[i].accent = 0.4;
    }

    const auto result = findDownbeats(flat, DownbeatConfig{});
    EXPECT_DOUBLE_EQ(result.strength, 0.0);
    EXPECT_DOUBLE_EQ(result.phase_margin, 0.0);
}

TEST(Downbeat, OneBarIsNotAPattern) {
    // Four beats can be divided into one bar of four in exactly one way, and
    // that is not evidence of anything. A meter has to repeat to be seen.
    const auto result = findDownbeats(bars(4, 0, 5, Cue::kLow), DownbeatConfig{});

    EXPECT_EQ(result.beats_per_bar, 0);
    EXPECT_TRUE(result.downbeats.empty());
    EXPECT_TRUE(result.candidates.empty());
}

TEST(Downbeat, NoBeatsAtAllIsNotACrash) {
    const auto result = findDownbeats({}, DownbeatConfig{});
    EXPECT_EQ(result.beats_per_bar, 0);
    EXPECT_TRUE(result.downbeats.empty());
}

TEST(Downbeat, ASixEightBarCanBeAskedForSpecifically) {
    // Six is in the default list but carries the weakest prior, because it is
    // only distinguishable from three by which of its two accents is bigger.
    // A caller who knows the piece can say so, and then it must be found.
    DownbeatConfig config;
    config.meters = {{6, 1.0}, {3, 1.0}};

    const auto result = findDownbeats(bars(6, 0, 36, Cue::kLow), config);
    EXPECT_EQ(result.beats_per_bar, 6);
    EXPECT_EQ(result.phase, 0);
}

TEST(Downbeat, ThePriorOnlyBreaksTies) {
    // A waltz must stay a waltz even though four-four is the likelier meter a
    // priori: the prior is a thumb on the scale, not a thumb on the answer.
    DownbeatConfig config;
    config.meters = {{4, 1.0}, {3, 0.5}};

    const auto result = findDownbeats(bars(3, 0, 30, Cue::kLow), config);
    EXPECT_EQ(result.beats_per_bar, 3);
}

TEST(DownbeatConfigTest, RejectsTheImpossible) {
    EXPECT_TRUE(DownbeatConfig{}.valid());

    auto no_meters = DownbeatConfig{};
    no_meters.meters.clear();
    EXPECT_FALSE(no_meters.valid());

    auto one_beat_bar = DownbeatConfig{};
    one_beat_bar.meters = {{1, 1.0}};
    EXPECT_FALSE(one_beat_bar.valid());

    auto no_weight = DownbeatConfig{};
    no_weight.low_weight = 0.0;
    no_weight.accent_weight = 0.0;
    no_weight.harmony_weight = 0.0;
    EXPECT_FALSE(no_weight.valid());

    auto reaches_next_beat = DownbeatConfig{};
    reaches_next_beat.window_after = 1.5;
    EXPECT_FALSE(reaches_next_beat.valid());

    auto one_bar = DownbeatConfig{};
    one_bar.min_bars = 1;
    EXPECT_FALSE(one_bar.valid());
}

// --------------------------------------------------------------- features --

// ODF frames at a fixed rate, with a bump of `height` on the frame nearest each
// beat listed in `hits`.
struct Frames {
    std::vector<double> times;
    std::vector<double> full;
    std::vector<double> low;
};

Frames framesWith(double fps, double duration, const std::vector<double>& hits, double height) {
    Frames f;
    const auto n = static_cast<std::size_t>(duration * fps);
    for (std::size_t i = 0; i < n; ++i) {
        f.times.push_back(static_cast<double>(i) / fps);
        f.full.push_back(0.01);
        f.low.push_back(0.01);
    }
    for (double t : hits) {
        const auto i = static_cast<std::size_t>(t * fps + 0.5);
        if (i < n) {
            f.full[i] = height;
            f.low[i] = height;
        }
    }
    return f;
}

TEST(BeatFeatures, PicksUpTheOnsetSittingOnTheBeat) {
    const std::vector<double> beats = {1.0, 1.5, 2.0, 2.5};
    const Frames frames = framesWith(100.0, 4.0, {1.0, 2.0}, 1.0);

    BeatFeatureInput input;
    input.frame_times = frames.times.data();
    input.odf_full = frames.full.data();
    input.odf_low = frames.low.data();
    input.frame_count = frames.times.size();
    input.beats = beats.data();
    input.beat_count = beats.size();

    const auto features = beatFeatures(input, DownbeatConfig{});
    ASSERT_EQ(features.size(), 4u);

    EXPECT_NEAR(features[0].low, 1.0, 1e-9);
    EXPECT_NEAR(features[2].low, 1.0, 1e-9);
    // The window must stop well short of the next beat, or every beat would
    // inherit its neighbour's onset and no pattern could survive.
    EXPECT_LT(features[1].low, 0.1);
    EXPECT_LT(features[3].low, 0.1);
}

TEST(BeatFeatures, WithoutChromaTheHarmonyCueIsSilentRatherThanInvented) {
    const std::vector<double> beats = {0.5, 1.0, 1.5};
    const Frames frames = framesWith(100.0, 2.0, {0.5, 1.0, 1.5}, 1.0);

    BeatFeatureInput input;
    input.frame_times = frames.times.data();
    input.odf_full = frames.full.data();
    input.odf_low = frames.low.data();
    input.chroma = nullptr;
    input.frame_count = frames.times.size();
    input.beats = beats.data();
    input.beat_count = beats.size();

    const auto features = beatFeatures(input, DownbeatConfig{});
    ASSERT_EQ(features.size(), 3u);
    for (const auto& f : features) EXPECT_DOUBLE_EQ(f.harmonic_change, 0.0);
}

TEST(BeatFeatures, ReadsTheChordChangeOffTheProfiles) {
    const double fps = 100.0;
    const std::vector<double> beats = {0.0, 0.5, 1.0, 1.5};
    const Frames frames = framesWith(fps, 2.0, {}, 0.0);

    // Two beats of one chord, then two of another.
    const std::size_t n = frames.times.size();
    std::vector<float> chroma(n * 12, 0.0f);
    for (std::size_t i = 0; i < n; ++i) {
        const bool second_half = frames.times[i] >= 1.0;
        chroma[i * 12 + (second_half ? 7 : 0)] = 1.0f;
    }

    BeatFeatureInput input;
    input.frame_times = frames.times.data();
    input.odf_full = frames.full.data();
    input.odf_low = frames.low.data();
    input.chroma = chroma.data();
    input.frame_count = n;
    input.beats = beats.data();
    input.beat_count = beats.size();

    const auto features = beatFeatures(input, DownbeatConfig{});
    ASSERT_EQ(features.size(), 4u);

    EXPECT_DOUBLE_EQ(features[0].harmonic_change, 0.0);  // nothing to compare with
    EXPECT_LT(features[1].harmonic_change, 1e-6);        // same chord
    EXPECT_GT(features[2].harmonic_change, 0.9);         // the change
    EXPECT_LT(features[3].harmonic_change, 1e-6);        // and settled again
}

TEST(BeatFeatures, IgnoresNothing) {
    EXPECT_TRUE(beatFeatures(BeatFeatureInput{}, DownbeatConfig{}).empty());
}

// -------------------------------------------------------------- two doubts --

// The failure the meter margin exists to catch, and the reason it cannot be
// derived from the phase margin: every beat carries the same cue, so all four
// phases of a four-beat bar are equally good and all three of a three-beat bar
// are too. Nothing here says anything about the meter, and the answer — whatever
// it is — must not be presented as settled.
TEST(Downbeat, MetreIsNotSettledWhenEveryMetreFitsEquallyWell) {
    std::vector<BeatFeature> flat(48);
    for (std::size_t i = 0; i < flat.size(); ++i) {
        flat[i].time_sec = static_cast<double>(i) * kBeat;
        flat[i].low = 1.0;
    }
    const auto result = findDownbeats(flat, DownbeatConfig{});
    EXPECT_LT(result.meter_margin, 0.05);
    EXPECT_FALSE(result.confident());
}

TEST(Downbeat, AClearFourFourIsSettledOnBothCounts) {
    const auto result = findDownbeats(bars(4, 0, 32, Cue::kLow), DownbeatConfig{});
    EXPECT_GT(result.phase_margin, 1.0);
    EXPECT_GT(result.meter_margin, 0.4);
    EXPECT_TRUE(result.confident());
}

// A six-beat pattern contains a three-beat one: mark every sixth beat and the
// bar lines of three land on a subset of them, so three scores respectably.
// The phase margin cannot express that doubt, because within three the phase is
// perfectly clear — which is exactly how a confidently wrong meter arises.
TEST(Downbeat, ARelatedMetreShowsUpInTheMetreMarginNotThePhaseMargin) {
    const auto result = findDownbeats(bars(6, 0, 48, Cue::kLow), DownbeatConfig{});

    ASSERT_GE(result.candidates.size(), 2u);
    EXPECT_GT(result.phase_margin, 0.5) << "the phase within the winning metre is clear";
    EXPECT_LT(result.meter_margin, result.phase_margin)
        << "and yet a related metre is close behind, which only this can say";
}

TEST(Downbeat, ConfidenceNeedsBothMarginsAndNotJustOne) {
    tiktak::analysis::DownbeatResult result;
    result.beats_per_bar = 4;
    result.downbeats = {0.0, 2.0};
    result.phase_margin = 2.0;
    result.meter_margin = 0.01;
    EXPECT_FALSE(result.confident()) << "a settled phase inside the wrong metre";

    result.phase_margin = 0.01;
    result.meter_margin = 2.0;
    EXPECT_FALSE(result.confident()) << "the right metre started on a coin toss";

    result.phase_margin = 2.0;
    EXPECT_TRUE(result.confident());

    // Nothing found at all is never confident, whatever the margins say.
    result.beats_per_bar = 0;
    EXPECT_FALSE(result.confident());
}

TEST(Downbeat, WithOneMetreOfferedThereIsNoRivalToLoseTo) {
    DownbeatConfig config;
    config.meters = {{4, 1.0}};
    const auto result = findDownbeats(bars(4, 0, 32, Cue::kLow), config);

    EXPECT_EQ(result.beats_per_bar, 4);
    EXPECT_GT(result.meter_margin, 0.0) << "the winner keeps its whole score";
}

TEST(Downbeat, MusicWithNoBarLineIsConfidentAboutNothing) {
    std::vector<BeatFeature> noise(48);
    for (std::size_t i = 0; i < noise.size(); ++i) {
        noise[i].time_sec = static_cast<double>(i) * kBeat;
        noise[i].low = (i % 7 == 0) ? 0.31 : 0.29;   // a period no meter offers
    }
    EXPECT_FALSE(findDownbeats(noise, DownbeatConfig{}).confident());
}

// ----------------------------------------------------------------- the seam --
//
// The resolver takes a per-beat salience and nothing else, which is what makes
// a learned scorer droppable in later. These tests reach it directly, without
// going through the cues, because that is exactly what a second backend will
// do — and because a seam nothing crosses in a test is a seam that has not been
// shown to hold.

using tiktak::analysis::cueSalience;
using tiktak::analysis::resolveMeter;

// Per-beat salience with every `per_bar`-th value raised, and beat times to go
// with it. No cues, no features — the resolver's whole input.
std::pair<std::vector<double>, std::vector<double>> salienceBars(
        int per_bar, int phase, int count, double strong = 1.0, double weak = 0.0) {
    std::vector<double> salience(static_cast<std::size_t>(count));
    std::vector<double> times(static_cast<std::size_t>(count));
    for (int i = 0; i < count; ++i) {
        const bool down = ((i - phase) % per_bar + per_bar) % per_bar == 0;
        salience[static_cast<std::size_t>(i)] = down ? strong : weak;
        times[static_cast<std::size_t>(i)] = i * kBeat;
    }
    return {salience, times};
}

TEST(Resolver, CountsBarsFromASalienceItDidNotProduce) {
    const auto [salience, times] = salienceBars(3, 1, 30);
    const auto result = resolveMeter(salience, times, DownbeatConfig{});

    EXPECT_EQ(result.beats_per_bar, 3);
    EXPECT_EQ(result.phase, 1);
    EXPECT_TRUE(result.confident());
    ASSERT_FALSE(result.downbeats.empty());
    EXPECT_NEAR(result.downbeats.front(), kBeat, 1e-12);
}

TEST(Resolver, TheMarginsDoNotDependOnTheScaleOfTheSalience) {
    // The point of standardising inside the resolver rather than in each
    // scorer: a model emitting probabilities in [0, 1] and cues emitting
    // arbitrary weighted sums must land on the same margins, or every threshold
    // has to be recalibrated the day the scorer changes.
    const auto [unit, times] = salienceBars(4, 0, 32, 1.0, 0.0);
    std::vector<double> loud(unit.size());
    for (std::size_t i = 0; i < unit.size(); ++i) loud[i] = unit[i] * 250.0 - 7.0;

    const auto a = resolveMeter(unit, times, DownbeatConfig{});
    const auto b = resolveMeter(loud, times, DownbeatConfig{});

    EXPECT_EQ(a.beats_per_bar, b.beats_per_bar);
    EXPECT_EQ(a.phase, b.phase);
    EXPECT_NEAR(a.strength, b.strength, 1e-9);
    EXPECT_NEAR(a.phase_margin, b.phase_margin, 1e-9);
    EXPECT_NEAR(a.meter_margin, b.meter_margin, 1e-9);
}

TEST(Resolver, MismatchedLengthsAreRefusedRatherThanGuessedAt) {
    const auto [salience, times] = salienceBars(4, 0, 32);
    std::vector<double> short_times(times.begin(), times.end() - 1);

    const auto result = resolveMeter(salience, short_times, DownbeatConfig{});
    EXPECT_EQ(result.beats_per_bar, 0);
    EXPECT_TRUE(result.downbeats.empty());
}

TEST(Resolver, TheBarLinesAreTheTimesHandedInAndNotAReconstruction) {
    // A resolver that recomputed times from an assumed tempo would pass every
    // test above and be wrong on any recording that drifts. The times are data.
    auto [salience, times] = salienceBars(4, 0, 32);
    for (std::size_t i = 0; i < times.size(); ++i) times[i] += 0.004 * i * i;

    const auto result = resolveMeter(salience, times, DownbeatConfig{});
    ASSERT_EQ(result.beats_per_bar, 4);
    ASSERT_GE(result.downbeats.size(), 3u);
    EXPECT_NEAR(result.downbeats[1], times[4], 1e-12);
    EXPECT_NEAR(result.downbeats[2], times[8], 1e-12);
}

TEST(Resolver, TheSplitIsBehaviourPreserving) {
    // findDownbeats must remain exactly the two halves composed, or the seam
    // has quietly become a second implementation of the same thing.
    const DownbeatConfig config;
    const auto features = bars(4, 2, 40, Cue::kLow);

    std::vector<double> times(features.size());
    for (std::size_t i = 0; i < features.size(); ++i) times[i] = features[i].time_sec;

    const auto whole = findDownbeats(features, config);
    const auto halves = resolveMeter(cueSalience(features, config), times, config);

    EXPECT_EQ(whole.beats_per_bar, halves.beats_per_bar);
    EXPECT_EQ(whole.phase, halves.phase);
    EXPECT_DOUBLE_EQ(whole.strength, halves.strength);
    EXPECT_DOUBLE_EQ(whole.phase_margin, halves.phase_margin);
    EXPECT_DOUBLE_EQ(whole.meter_margin, halves.meter_margin);
    EXPECT_EQ(whole.downbeats, halves.downbeats);
}

TEST(Resolver, AFlatSalienceDecidesNothingHoweverLargeItIs) {
    std::vector<double> flat(48, 5.0);
    std::vector<double> times(48);
    for (std::size_t i = 0; i < times.size(); ++i) times[i] = static_cast<double>(i) * kBeat;

    EXPECT_FALSE(resolveMeter(flat, times, DownbeatConfig{}).confident());
}

}  // namespace
