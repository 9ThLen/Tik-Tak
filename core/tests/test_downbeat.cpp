#include "analysis/downbeat.hpp"

#include <gtest/gtest.h>

#include <cmath>
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
    EXPECT_GT(result.margin, 1.0);

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
    EXPECT_GT(result.margin, 1.0);

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
    EXPECT_GT(result.margin, 1.0);
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
    EXPECT_DOUBLE_EQ(result.margin, 0.0);
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

}  // namespace
