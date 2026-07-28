#include "ml/beat_this.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "support.hpp"

using tiktak::ml::BeatThisFeatures;
using tiktak::test::clickTrack;
using tiktak::test::sine;

namespace {

constexpr double kRate = BeatThisFeatures::kModelRate;
constexpr std::size_t kMels = BeatThisFeatures::kMels;

}  // namespace

TEST(BeatThisFeatures, ProducesFiftyFramesASecond) {
    BeatThisFeatures features;
    const auto audio = clickTrack(120.0, 4.0, kRate);
    const auto mel = features.compute(audio.data(), audio.size());

    ASSERT_EQ(mel.size() % kMels, 0u);
    const std::size_t frames = mel.size() / kMels;
    // Centred frames plus reflect padding means four seconds gives 200 frames
    // and one more for the window that starts inside the tail padding.
    EXPECT_NEAR(static_cast<double>(frames), 200.0, 2.0);
    EXPECT_EQ(frames, BeatThisFeatures::frameCount(audio.size()));
}

TEST(BeatThisFeatures, SilenceIsAllZeros) {
    // log1p(1000 * 1e-10) is 1e-7, which rounds to zero in float32 — so
    // digital silence has to come out flat. Anything else here would be the
    // front end reporting its own arithmetic to the network.
    BeatThisFeatures features;
    const std::vector<float> quiet(static_cast<std::size_t>(kRate * 2.0), 0.0f);
    const auto mel = features.compute(quiet.data(), quiet.size());

    ASSERT_FALSE(mel.empty());
    for (float value : mel) EXPECT_NEAR(value, 0.0f, 1e-6f);
}

TEST(BeatThisFeatures, IsNeverNegative) {
    // log1p of a non-negative energy cannot be. A negative value would mean the
    // floor was skipped somewhere.
    BeatThisFeatures features;
    const auto audio = clickTrack(120.0, 3.0, kRate);
    const auto mel = features.compute(audio.data(), audio.size());
    for (float value : mel) ASSERT_GE(value, 0.0f);
}

TEST(BeatThisFeatures, PutsAToneInABandNearItsPitch) {
    BeatThisFeatures features;

    std::size_t previous = 0;
    for (double hz : {100.0, 440.0, 2000.0, 8000.0}) {
        const auto tone = sine(static_cast<std::size_t>(kRate), hz, kRate);
        const auto mel = features.compute(tone.data(), tone.size());
        ASSERT_GE(mel.size(), kMels);

        // A frame from the middle, clear of the padded edges.
        const std::size_t frames = mel.size() / kMels;
        const float* row = mel.data() + (frames / 2) * kMels;
        std::size_t loudest = 0;
        for (std::size_t m = 1; m < kMels; ++m) {
            if (row[m] > row[loudest]) loudest = m;
        }
        EXPECT_GT(loudest, previous) << hz << " Hz landed in band " << loudest;
        previous = loudest;
    }
}

TEST(BeatThisFeatures, ShortAudioIsRefusedRatherThanInvented) {
    BeatThisFeatures features;
    const std::vector<float> tiny(100, 0.5f);
    EXPECT_TRUE(features.compute(tiny.data(), tiny.size()).empty());
    EXPECT_EQ(BeatThisFeatures::frameCount(100), 0u);
    EXPECT_TRUE(features.compute(nullptr, 0).empty());
}

TEST(BeatThisFeatures, ReusingTheObjectGivesTheSameAnswer) {
    // Every buffer is a member and reused between calls; a scratch left dirty
    // would show up as the second answer differing from the first.
    BeatThisFeatures features;
    const auto a = clickTrack(120.0, 2.0, kRate);
    const auto b = clickTrack(90.0, 3.0, kRate);

    const auto first = features.compute(a.data(), a.size());
    features.compute(b.data(), b.size());
    const auto again = features.compute(a.data(), a.size());

    ASSERT_EQ(first.size(), again.size());
    for (std::size_t i = 0; i < first.size(); ++i) EXPECT_FLOAT_EQ(first[i], again[i]);
}

TEST(BeatThisFeatures, TheWindowIsPeriodicNotSymmetric) {
    // One sample in 1024, invisible in any picture of a spectrogram, and a slow
    // leak of accuracy against a model trained with the periodic form. Detected
    // through the thing it actually changes: a tone exactly on a bin centre
    // leaks less through a periodic window than a symmetric one.
    BeatThisFeatures features;
    const double binHz = kRate / static_cast<double>(BeatThisFeatures::kFftSize);
    const auto tone = sine(static_cast<std::size_t>(kRate), 64.0 * binHz, kRate);
    const auto mel = features.compute(tone.data(), tone.size());

    const std::size_t frames = mel.size() / kMels;
    const float* row = mel.data() + (frames / 2) * kMels;

    std::size_t loudest = 0;
    double total = 0.0;
    for (std::size_t m = 0; m < kMels; ++m) {
        if (row[m] > row[loudest]) loudest = m;
        total += row[m];
    }
    // The peak band carries a real share of the frame rather than the energy
    // being smeared across the bank, which is what a windowing mistake does.
    EXPECT_GT(row[loudest] / total, 0.02);
}
