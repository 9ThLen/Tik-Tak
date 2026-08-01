#include "analysis/downbeat.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
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

TEST(Downbeat, TinyHarmonicLeakageIsNotPromotedIntoAChordChange) {
    // The absolute scale of chroma distance carries information. This is the
    // drums-only failure seen in practice: a periodic numerical ripple around
    // 0.01 is not a quiet but certain chord progression.
    const auto result =
        findDownbeats(bars(4, 0, 32, Cue::kHarmony, 0.011, 0.010),
                      DownbeatConfig{});

    EXPECT_EQ(result.beats_per_bar, 0);
    EXPECT_TRUE(result.downbeats.empty());
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
    // Every beat identical. The scorer's range gate makes this no evidence,
    // rather than returning an arbitrary first candidate with zero strength.
    std::vector<BeatFeature> flat(32);
    for (std::size_t i = 0; i < flat.size(); ++i) {
        flat[i].time_sec = static_cast<double>(i) * kBeat;
        flat[i].low = 0.4;
        flat[i].accent = 0.4;
    }

    const auto result = findDownbeats(flat, DownbeatConfig{});
    EXPECT_EQ(result.beats_per_bar, 0);
    EXPECT_TRUE(result.downbeats.empty());
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

TEST(Downbeat, ASixPulseCycleCanBeAskedForSpecifically) {
    // This says only that six pulses of the supplied grid repeat. It is 6/8 if
    // those pulses are eighth notes; with a tactus-level grid, ordinary 6/8 is
    // usually two pulses and cannot be distinguished from 2/4 here.
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

    auto negative_range = DownbeatConfig{};
    negative_range.min_salience_range = -0.01;
    EXPECT_FALSE(negative_range.valid());

    auto infinite_range = DownbeatConfig{};
    infinite_range.min_salience_range = std::numeric_limits<double>::infinity();
    EXPECT_FALSE(infinite_range.valid());

    auto infinite_prior = DownbeatConfig{};
    infinite_prior.meters.front().prior = std::numeric_limits<double>::infinity();
    EXPECT_FALSE(infinite_prior.valid());

    auto infinite_weight = DownbeatConfig{};
    infinite_weight.low_weight = std::numeric_limits<double>::infinity();
    EXPECT_FALSE(infinite_weight.valid());

    auto infinite_margin = DownbeatConfig{};
    infinite_margin.min_phase_margin = std::numeric_limits<double>::infinity();
    EXPECT_FALSE(infinite_margin.valid());
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

TEST(Resolver, IneligibleMeterPriorDoesNotConstrainNumericalResolution) {
    const auto [salience, times] = salienceBars(4, 0, 32);
    DownbeatConfig config;
    config.meters = {
        {4, 1.0},
        {1000, std::numeric_limits<double>::max()},
    };

    const auto result = resolveMeter(salience, times, config);

    EXPECT_EQ(result.beats_per_bar, 4);
    EXPECT_EQ(result.phase, 0);
}

TEST(Resolver, ZeroPhaseThresholdCannotMakeARoundedTieConfident) {
    std::vector<double> salience(18, 0.0);
    std::vector<double> times(18);
    salience[0] = 1e10;
    salience[1] = 1e10;
    salience[7] = 1e-7;
    salience[13] = 1e-7;
    for (std::size_t i = 0; i < times.size(); ++i) {
        times[i] = static_cast<double>(i) * kBeat;
    }
    DownbeatConfig config;
    config.meters = {{6, 1.0}};
    config.min_phase_margin = 0.0;

    const auto result = resolveMeter(salience, times, config);

    EXPECT_EQ(result.beats_per_bar, 0);
    EXPECT_TRUE(result.downbeats.empty());
}

TEST(Resolver, ZeroMeterThresholdNeedsAtMostOneEligibleMeter) {
    const auto [salience, times] = salienceBars(4, 0, 32);
    DownbeatConfig config;
    config.min_meter_margin = 0.0;

    const auto ambiguous = resolveMeter(salience, times, config);
    EXPECT_EQ(ambiguous.beats_per_bar, 0);

    config.meters = {{4, 1.0}};
    const auto asserted = resolveMeter(salience, times, config);
    EXPECT_EQ(asserted.beats_per_bar, 4);
    EXPECT_EQ(asserted.phase, 0);
}

TEST(Resolver, DuplicateCandidatesAreOneMeterForTheZeroMarginGate) {
    const auto [salience, times] = salienceBars(4, 0, 12);
    DownbeatConfig config;
    config.meters = {{4, 1.0}, {4, 2.0}};
    config.min_meter_margin = 0.0;

    const auto result = resolveMeter(salience, times, config);

    EXPECT_EQ(result.beats_per_bar, 4);
    EXPECT_EQ(result.phase, 0);
}

TEST(Resolver, PreservesBackendScaleButIgnoresAnOffset) {
    // A backend owns its units and calibrates the three evidence thresholds in
    // those units. Adding a DC offset changes nothing; multiplying the evidence
    // must multiply the margins rather than turn weak and strong signals into
    // the same answer.
    const auto [unit, times] = salienceBars(4, 0, 32, 1.0, 0.0);
    std::vector<double> loud(unit.size());
    for (std::size_t i = 0; i < unit.size(); ++i) loud[i] = unit[i] * 250.0 - 7.0;

    const auto a = resolveMeter(unit, times, DownbeatConfig{});
    const auto b = resolveMeter(loud, times, DownbeatConfig{});

    EXPECT_EQ(a.beats_per_bar, b.beats_per_bar);
    EXPECT_EQ(a.phase, b.phase);
    EXPECT_NEAR(a.strength * 250.0, b.strength, 1e-9);
    EXPECT_NEAR(a.phase_margin * 250.0, b.phase_margin, 1e-9);
    EXPECT_NEAR(a.meter_margin * 250.0, b.meter_margin, 1e-9);
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

    const auto result = resolveMeter(flat, times, DownbeatConfig{});
    EXPECT_EQ(result.beats_per_bar, 0);
    EXPECT_TRUE(result.downbeats.empty());
}

TEST(Resolver, ANearlyFlatPeriodicRippleIsNotFullScaleEvidence) {
    const auto [salience, times] =
        salienceBars(4, 0, 32, 0.500003, 0.500001);

    const auto rejected = resolveMeter(salience, times, DownbeatConfig{});
    EXPECT_EQ(rejected.beats_per_bar, 0);
    EXPECT_TRUE(rejected.downbeats.empty());

    // A backend may deliberately lower its own range gate, but the resolver
    // still preserves the tiny scale: the default margin thresholds therefore
    // do not call this confident.
    DownbeatConfig model_config;
    model_config.min_salience_range = 1e-7;
    const auto admitted = resolveMeter(salience, times, model_config);
    EXPECT_EQ(admitted.beats_per_bar, 4);
    EXPECT_EQ(admitted.phase, 0);
    EXPECT_GT(admitted.strength, 0.0);
    EXPECT_LT(admitted.strength, 1e-4);
    EXPECT_FALSE(admitted.confident(model_config.min_phase_margin,
                                    model_config.min_meter_margin));
}

TEST(Resolver, NonFiniteBackendOutputIsNotAnAnswer) {
    const auto [finite_salience, times] = salienceBars(4, 0, 32);
    for (double bad : {std::numeric_limits<double>::quiet_NaN(),
                       std::numeric_limits<double>::infinity(),
                       -std::numeric_limits<double>::infinity()}) {
        auto salience = finite_salience;
        salience[3] = bad;
        const auto result = resolveMeter(salience, times, DownbeatConfig{});
        EXPECT_EQ(result.beats_per_bar, 0);
        EXPECT_TRUE(result.downbeats.empty());
    }
}

TEST(Resolver, ExtremeFiniteBackendOutputKeepsEveryMetricFinite) {
    // A two-level salience can be represented safely even at the edge of the
    // finite range. Repeated DBL_MAX values used to overflow per-phase sums; a
    // positive and negative extreme could also overflow ranges and margins.
    const double limit = std::numeric_limits<double>::max();
    const auto [salience, times] = salienceBars(4, 0, 32, limit, -limit);
    DownbeatConfig config;
    config.meters = {{4, 2.0}, {3, 1.0}, {2, 0.75}, {6, 0.6}};
    config.min_salience_range = limit / 8.0;
    config.min_phase_margin = limit / 8.0;
    config.min_meter_margin = limit / 4.0;

    const auto result = resolveMeter(salience, times, config);

    EXPECT_EQ(result.beats_per_bar, 4);
    EXPECT_EQ(result.phase, 0);
    EXPECT_TRUE(std::isfinite(result.strength));
    EXPECT_TRUE(std::isfinite(result.phase_margin));
    EXPECT_TRUE(std::isfinite(result.meter_margin));
    ASSERT_FALSE(result.candidates.empty());
    for (const auto& candidate : result.candidates) {
        EXPECT_TRUE(std::isfinite(candidate.score))
            << "meter " << candidate.beats_per_bar;
    }
}

TEST(Resolver, ExtremeFiniteOffsetDoesNotCollapseDistinctSalienceLevels) {
    // Subtracting -DBL_MAX in the public double scale would saturate both zero
    // and +DBL_MAX to the same value. The resolver must form its internal
    // affine weights in a shared power-of-two scale so that the mathematical
    // six-pulse pattern and its phase survive before ProductKey ranks meters.
    const double limit = std::numeric_limits<double>::max();
    const std::vector<double> cycle = {
        -limit, 0.0, -limit, limit, -limit, -limit,
    };
    std::vector<double> salience;
    std::vector<double> times;
    for (int repeat = 0; repeat < 4; ++repeat) {
        for (double value : cycle) {
            salience.push_back(value);
            times.push_back(static_cast<double>(times.size()) * kBeat);
        }
    }

    DownbeatConfig config;
    config.min_salience_range = limit / 8.0;
    config.min_phase_margin = limit / 8.0;
    config.min_meter_margin = limit / 8.0;
    const auto result = resolveMeter(salience, times, config);

    EXPECT_EQ(result.beats_per_bar, 6);
    EXPECT_EQ(result.phase, 3);
    EXPECT_TRUE(std::isfinite(result.strength));
    EXPECT_TRUE(std::isfinite(result.phase_margin));
    EXPECT_TRUE(std::isfinite(result.meter_margin));
}

TEST(Resolver, UnrepresentableFiniteDynamicRangeIsNotAnAnswer) {
    // max-min is exactly DBL_MAX here, but after subtracting the minimum both
    // -1 and 0 round to DBL_MAX. Treating those distinct levels as equal creates
    // a large periodic contrast that does not exist in the input. Until an
    // exact/binned accumulator is justified, the safe contract is to withhold.
    const double limit = std::numeric_limits<double>::max();
    std::vector<double> salience(48, -1.0);
    std::vector<double> times(48);
    for (std::size_t i = 0; i < salience.size(); ++i) {
        if (i < 12) {
            salience[i] = -limit;
        } else if (i % 6 == 3) {
            salience[i] = 0.0;
        }
        times[i] = static_cast<double>(i) * kBeat;
    }

    const auto result = resolveMeter(salience, times, DownbeatConfig{});

    EXPECT_EQ(result.beats_per_bar, 0);
    EXPECT_TRUE(result.downbeats.empty());
    EXPECT_TRUE(result.candidates.empty());
}

TEST(Resolver, UnsafeScaleWithLargeCommonTermsIsNotAnAnswer) {
    // Each phase sees the same share of the first twelve DBL_MAX values, so
    // their exact contribution to every contrast is zero. But a backend that
    // calls both DBL_MAX and 1 meaningful under ordinary sub-unit thresholds
    // has not supplied a numerically usable scale, so it must be rejected
    // before arithmetic can manufacture a roughly 5e291 contrast.
    const double limit = std::numeric_limits<double>::max();
    std::vector<double> salience(48, 0.0);
    std::vector<double> times(48);
    for (std::size_t i = 0; i < salience.size(); ++i) {
        if (i < 12) {
            salience[i] = limit;
        } else if (i % 6 == 3) {
            salience[i] = 1.0;
        }
        times[i] = static_cast<double>(i) * kBeat;
    }

    const auto result = resolveMeter(salience, times, DownbeatConfig{});

    EXPECT_EQ(result.beats_per_bar, 0);
    EXPECT_TRUE(result.downbeats.empty());
    EXPECT_TRUE(result.candidates.empty());
}

TEST(Resolver, UnsafeScaleCannotUnderflowBeforeItIsRejected) {
    // After normalising DBL_MAX, 2^-50 becomes DBL_TRUE_MIN. Dividing that
    // numerator by the phase counts in double would round every contrast to
    // zero before the common 2^1024 exponent was restored. The numerical range
    // contract rejects that backend scale instead of choosing an arbitrary
    // phase.
    const double limit = std::numeric_limits<double>::max();
    const double small = 0x1p-50;
    std::vector<double> salience(48, 0.0);
    std::vector<double> times(48);
    for (std::size_t i = 0; i < salience.size(); ++i) {
        if (i < 12) {
            salience[i] = limit;
        } else if (i == 15 || i == 21 || i == 27) {
            salience[i] = small;
        }
        times[i] = static_cast<double>(i) * kBeat;
    }

    const auto result = resolveMeter(salience, times, DownbeatConfig{});

    EXPECT_EQ(result.beats_per_bar, 0);
    EXPECT_TRUE(result.downbeats.empty());
    EXPECT_TRUE(result.candidates.empty());
}

TEST(Resolver, UnsafeScaleCannotHideAPhaseDifferenceInRounding) {
    // The exact phase-one contrast beats phase zero by 0.8, but beside
    // DBL_MAX that difference is below the calibrated numerical resolution.
    // Collapsing the expansion to one double used to turn it into an accidental
    // earliest-phase tie; the range contract must withhold before ranking.
    const double limit = std::numeric_limits<double>::max();
    std::vector<double> salience(18, 1.0);
    std::vector<double> times(18);
    salience[0] = limit;
    salience[1] = limit;
    salience[7] = 2.0;
    salience[13] = 2.0;
    for (std::size_t i = 0; i < times.size(); ++i) {
        times[i] = static_cast<double>(i) * kBeat;
    }
    DownbeatConfig config;
    config.meters = {{6, 1.0}};

    const auto result = resolveMeter(salience, times, config);

    EXPECT_EQ(result.beats_per_bar, 0);
    EXPECT_TRUE(result.downbeats.empty());
    EXPECT_TRUE(result.candidates.empty());
}

TEST(Resolver, ProductAtTheRoundedOverflowBoundaryIsSaturated) {
    // For this exact pair DBL_MAX / prior rounds back to contrast, so checking
    // `contrast > DBL_MAX / prior` misses the overflow even though the product
    // is +Inf. The reported diagnostic must remain valid JSON data.
    constexpr double contrast = 0x1.0776248bc6c27p+753;
    constexpr double prior = 0x1.f17fe8a887318p+270;
    const auto [salience, times] = salienceBars(4, 0, 32, contrast, 0.0);
    DownbeatConfig config;
    config.meters = {{4, prior}};
    config.min_salience_range = contrast / 2.0;
    config.min_phase_margin = contrast / 2.0;
    config.min_meter_margin = std::numeric_limits<double>::max();

    const auto result = resolveMeter(salience, times, config);

    ASSERT_EQ(result.candidates.size(), 1u);
    EXPECT_EQ(result.candidates.front().score,
              std::numeric_limits<double>::max());
    EXPECT_TRUE(std::isfinite(result.meter_margin));
}

TEST(Resolver, ProductAtTheSubnormalHalfwayBoundaryRoundsOnce) {
    // high + low followed by scalbn double-rounds this exact product to zero.
    // Direct rounding in DBL_TRUE_MIN units must see that it lies just above
    // the halfway point and retain the smallest positive double.
    constexpr double contrast = 0x1.0000000000001p-538;
    constexpr double prior = 0x1.fffffffffffffp-538;
    const auto [salience, times] =
        salienceBars(4, 0, 32, contrast, 0.0);
    DownbeatConfig config;
    config.meters = {{4, prior}};
    config.min_salience_range = 0.0;

    const auto result = resolveMeter(salience, times, config);

    ASSERT_EQ(result.candidates.size(), 1u);
    EXPECT_EQ(result.candidates.front().score,
              std::numeric_limits<double>::denorm_min());
    EXPECT_EQ(result.meter_margin,
              std::numeric_limits<double>::denorm_min());
}

TEST(Resolver, OverflowedReportedScoresStillUseTheMathematicalOrder) {
    // On this four-beat pattern both candidate products overflow:
    //   3-beat contrast * 30 < 4-beat contrast * 2.
    // Saturating the public scores is necessary, but using those saturated
    // values as the ordering key would turn the real inequality into a tie and
    // incorrectly preserve the deliberately adverse input order.
    const double limit = std::numeric_limits<double>::max();
    const auto [salience, times] = salienceBars(4, 0, 32, limit, 0.0);
    DownbeatConfig config;
    config.meters = {{3, 30.0}, {4, 2.0}};
    config.min_salience_range = limit / 4.0;
    config.min_phase_margin = limit / 4.0;
    config.min_meter_margin = limit;

    const auto result = resolveMeter(salience, times, config);

    ASSERT_EQ(result.candidates.size(), 2u);
    EXPECT_EQ(result.candidates[0].score, limit);
    EXPECT_EQ(result.candidates[1].score, limit);
    EXPECT_EQ(result.candidates[0].beats_per_bar, 4);
    EXPECT_EQ(result.beats_per_bar, 4);
    EXPECT_EQ(result.phase, 0);
    EXPECT_TRUE(std::isfinite(result.meter_margin));
    EXPECT_GT(result.meter_margin, 0.0);
}

TEST(Resolver, ExactProductTiesKeepTheConfiguredOrder) {
    // On a 4-pulse pattern, meter 2 has half the contrast of meter 4. These
    // priors make the mathematical products exactly equal, so neither the
    // internal key nor sorting may invent a preference behind the caller's
    // configured order.
    const auto [salience, times] = salienceBars(4, 0, 32);
    DownbeatConfig config;
    config.meters = {{2, 2.0}, {4, 1.0}};

    const auto result = resolveMeter(salience, times, config);

    ASSERT_EQ(result.candidates.size(), 2u);
    EXPECT_DOUBLE_EQ(result.candidates[0].score,
                     result.candidates[1].score);
    EXPECT_EQ(result.candidates[0].beats_per_bar, 2);
    EXPECT_EQ(result.beats_per_bar, 2);
}

// ------------------------------------------------------- the movable phase --
//
// A single global phase is right on only one side of an inserted or dropped
// bar, and 14.1% of full-length annotated songs contain one. These pin the
// three things that decide whether letting it move is an improvement or a way
// of chasing noise: that it is off unless asked for, that it follows a real
// slip, and that it does not move when there is nothing to follow.

using tiktak::analysis::barPositions;

// Bars of `per_bar`, with one bar of `inserted` beats spliced in at `at`, so
// everything after it sits on a different phase. No single phase describes the
// whole run — which is the case the decoder exists for and the one a corpus of
// thirty-second excerpts almost never contains.
std::pair<std::vector<double>, std::vector<double>> slippedBars(
        int per_bar, int bars_before, int inserted, int bars_after,
        double strong = 1.0, double weak = 0.0) {
    std::vector<double> salience;
    std::vector<double> times;
    const auto push = [&](int length) {
        for (int i = 0; i < length; ++i) {
            salience.push_back(i == 0 ? strong : weak);
            times.push_back(static_cast<double>(times.size()) * kBeat);
        }
    };
    for (int b = 0; b < bars_before; ++b) push(per_bar);
    push(inserted);
    for (int b = 0; b < bars_after; ++b) push(per_bar);
    return {salience, times};
}

TEST(MovablePhase, ShipsExpensiveEnoughToBeInertOnExcerpts) {
    // 64 is the only cost in the sweep that gains on full-length songs and
    // loses on nothing; at 8 it takes 0.007 of downbeat F off the recordings
    // whose grid is already right. A change that lowers this has to bring the
    // corpus numbers with it — the tables are beside the field.
    const DownbeatConfig config;
    EXPECT_DOUBLE_EQ(config.phase_switch_cost, 64.0);
    EXPECT_TRUE(config.valid());
}

TEST(MovablePhase, PinnedIsStillReachableAndStillTheOldAnswer) {
    // Every offline number measured before the decoder existed was measured
    // with the phase pinned, so the setting that reproduces them has to remain
    // available and has to keep producing the plain arithmetic sequence.
    const auto [salience, times] = slippedBars(4, 6, 3, 6);
    DownbeatConfig config;
    config.phase_switch_cost = std::numeric_limits<double>::infinity();

    const auto result = resolveMeter(salience, times, config);
    ASSERT_EQ(result.beats_per_bar, 4);
    for (std::size_t i = 0; i < result.downbeats.size(); ++i) {
        const auto beat = static_cast<std::size_t>(result.phase) + i * 4u;
        ASSERT_LT(beat, times.size());
        EXPECT_DOUBLE_EQ(result.downbeats[i], times[beat]);
    }
}

TEST(MovablePhase, FollowsASlipTheSingleAnswerCannotDescribe) {
    const auto [salience, times] = slippedBars(4, 6, 3, 6);

    DownbeatConfig pinned;
    const auto before = resolveMeter(salience, times, pinned);

    DownbeatConfig movable;
    movable.phase_switch_cost = 1.0;
    const auto after = resolveMeter(salience, times, movable);

    ASSERT_EQ(after.beats_per_bar, 4) << "the metre is not the decoder's to move";

    // Every beat that actually starts a bar, by construction.
    std::vector<double> truth;
    for (std::size_t i = 0; i < salience.size(); ++i) {
        if (salience[i] > 0.5) truth.push_back(times[i]);
    }
    const auto hits = [&](const std::vector<double>& found) {
        std::size_t n = 0;
        for (double t : truth) {
            for (double f : found) {
                if (std::abs(f - t) < 1e-9) { ++n; break; }
            }
        }
        return n;
    };
    EXPECT_EQ(hits(after.downbeats), truth.size())
        << "a movable phase should place every bar line on this signal";
    EXPECT_LT(hits(before.downbeats), truth.size())
        << "if one phase already described it, the test signal is wrong";
}

TEST(MovablePhase, DoesNotWanderOnMusicThatNeverSlips) {
    // Regular bars with noise on every beat. The decoder is free to switch at
    // a cost that the slip test above shows is cheap enough to follow a real
    // one, so if it moves here it is following the noise, and a bar line that
    // moves for no reason is worse for a player than one that is merely wrong.
    std::vector<double> salience(96);
    std::vector<double> times(96);
    unsigned seed = 12345u;
    for (std::size_t i = 0; i < salience.size(); ++i) {
        seed = seed * 1664525u + 1013904223u;
        const double noise = 0.15 * (static_cast<double>(seed >> 8) / 16777215.0);
        salience[i] = (i % 4 == 0 ? 1.0 : 0.0) + noise;
        times[i] = static_cast<double>(i) * kBeat;
    }

    const auto path = barPositions(salience, 4, 1.0);
    ASSERT_EQ(path.size(), salience.size());
    for (std::size_t i = 1; i < path.size(); ++i) {
        EXPECT_EQ(path[i], path[0]) << "the phase moved at beat " << i;
    }
}

TEST(MovablePhase, AnUnpayableCostIsTheSameAnswerAsNoDecoderAtAll) {
    // The two code paths are separate — infinity skips the decoder entirely —
    // so this is what says they agree rather than merely both existing.
    const auto [salience, times] = salienceBars(4, 2, 40);
    DownbeatConfig pinned;
    DownbeatConfig expensive;
    expensive.phase_switch_cost = 1e6;

    const auto a = resolveMeter(salience, times, pinned);
    const auto b = resolveMeter(salience, times, expensive);

    EXPECT_EQ(a.beats_per_bar, b.beats_per_bar);
    EXPECT_EQ(a.phase, b.phase);
    ASSERT_EQ(a.downbeats.size(), b.downbeats.size());
    for (std::size_t i = 0; i < a.downbeats.size(); ++i) {
        EXPECT_DOUBLE_EQ(a.downbeats[i], b.downbeats[i]);
    }
}

TEST(MovablePhase, ANegativeCostIsRefusedRatherThanObeyed) {
    // A negative cost pays the decoder to switch. It would shatter the phase on
    // any material at all, and the failure would look like a tracker fault
    // rather than a configuration one.
    DownbeatConfig config;
    config.phase_switch_cost = -1.0;
    EXPECT_FALSE(config.valid());
    config.phase_switch_cost = std::numeric_limits<double>::quiet_NaN();
    EXPECT_FALSE(config.valid());
    config.phase_switch_cost = 0.0;
    EXPECT_TRUE(config.valid()) << "free switching is a legitimate experiment";
}

}  // namespace
