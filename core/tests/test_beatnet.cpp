#include "ml/beatnet.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <utility>
#include <vector>

#include "support.hpp"

using tiktak::ml::BeatNetActivation;
using tiktak::ml::BeatNetFeatures;
using tiktak::ml::BeatNetModel;
using tiktak::ml::BeatNetWeights;
using tiktak::test::clickTrack;
using tiktak::test::sine;

namespace {

// A weight file with the published shapes and made-up numbers.
//
// The real weights are 1.6 MB that git does not carry and CI does not have, so
// everything here that is about *structure* — the loader, the state, and the
// arithmetic's shape — is tested against a blob built on the spot. Reference
// behavior with production weights is verified separately.
// `step` varies the made-up numbers, so two calls can stand in for two of
// BeatNet's published checkpoints where what is under test is that several are
// combined correctly rather than what any one of them says.
std::vector<unsigned char> makeWeightFile(std::uint32_t version = 1,
                                          std::uint32_t features = BeatNetWeights::kFeatures,
                                          double step = 0.7) {
    std::vector<unsigned char> out(BeatNetWeights::kFileBytes);
    std::memcpy(out.data(), "TTBN", 4);

    const std::uint32_t header[7] = {
        version, features, BeatNetWeights::kConvChannels, BeatNetWeights::kKernel,
        BeatNetWeights::kHidden, BeatNetWeights::kLayers, BeatNetWeights::kClasses,
    };
    for (std::size_t i = 0; i < 7; ++i) {
        for (std::size_t b = 0; b < 4; ++b) {
            out[4 + i * 4 + b] = static_cast<unsigned char>((header[i] >> (8 * b)) & 0xFF);
        }
    }

    // Small, varied and deterministic. Large weights would saturate every
    // sigmoid and make the state tests pass for the wrong reason.
    for (std::size_t i = 0; i < BeatNetWeights::kParameters; ++i) {
        const float value = 0.05f * static_cast<float>(std::sin(step * static_cast<double>(i)));
        std::uint32_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        for (std::size_t b = 0; b < 4; ++b) {
            out[BeatNetWeights::kHeaderBytes + i * 4 + b] =
                static_cast<unsigned char>((bits >> (8 * b)) & 0xFF);
        }
    }
    return out;
}

std::vector<float> ramp(std::size_t n) {
    std::vector<float> out(n);
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = static_cast<float>(std::sin(0.3 * static_cast<double>(i)));
    }
    return out;
}

}  // namespace

// ----------------------------------------------------------------- loading --

TEST(BeatNetWeights, LoadsAFileWithTheShapesTheNetworkExpects) {
    const auto blob = makeWeightFile();
    BeatNetWeights weights;
    ASSERT_TRUE(weights.load(blob.data(), blob.size()));
    EXPECT_TRUE(weights.valid());

    // Every tensor points somewhere, and they run in order without overlapping.
    EXPECT_NE(weights.conv_weight, nullptr);
    EXPECT_EQ(weights.conv_bias, weights.conv_weight + BeatNetWeights::kConvChannels *
                                                           BeatNetWeights::kKernel);
    EXPECT_EQ(weights.out_bias,
              weights.out_weight + BeatNetWeights::kClasses * BeatNetWeights::kHidden);
    EXPECT_EQ(weights.out_bias + BeatNetWeights::kClasses,
              weights.conv_weight + BeatNetWeights::kParameters);
}

TEST(BeatNetWeights, RefusesAFileThatIsNotOne) {
    BeatNetWeights weights;

    auto blob = makeWeightFile();
    blob[1] = 'X';
    EXPECT_FALSE(weights.load(blob.data(), blob.size())) << "wrong magic";

    EXPECT_FALSE(weights.load(nullptr, BeatNetWeights::kFileBytes));

    blob = makeWeightFile();
    EXPECT_FALSE(weights.load(blob.data(), blob.size() - 4)) << "truncated";
    EXPECT_FALSE(weights.load(blob.data(), blob.size() + 4)) << "trailing bytes";
}

TEST(BeatNetWeights, RefusesAFileThatDisagreesAboutAShape) {
    // The failure this guards against is not corruption, it is a front end that
    // was changed on one side only: 84 filters offered to a layer expecting
    // 136 is a silent wrong answer unless something refuses.
    BeatNetWeights weights;

    auto blob = makeWeightFile(1, 84);
    EXPECT_FALSE(weights.load(blob.data(), blob.size())) << "wrong feature count";

    blob = makeWeightFile(2);
    EXPECT_FALSE(weights.load(blob.data(), blob.size())) << "future version";
}

TEST(BeatNetWeights, AFailedLoadLeavesNothingBehind) {
    const auto good = makeWeightFile();
    BeatNetWeights weights;
    ASSERT_TRUE(weights.load(good.data(), good.size()));

    auto bad = makeWeightFile();
    bad[0] = 'Z';
    EXPECT_FALSE(weights.load(bad.data(), bad.size()));
    EXPECT_FALSE(weights.valid()) << "a refused file must not leave the old one loaded";
}

// ------------------------------------------------------------------- model --

TEST(BeatNetModel, ProbabilitiesAreProbabilities) {
    const auto blob = makeWeightFile();
    BeatNetWeights weights;
    ASSERT_TRUE(weights.load(blob.data(), blob.size()));
    BeatNetModel model(weights);

    const auto features = ramp(BeatNetWeights::kFeatures);
    float probabilities[BeatNetWeights::kClasses];
    model.forward(features.data(), probabilities);

    float total = 0.0f;
    for (float p : probabilities) {
        EXPECT_GE(p, 0.0f);
        EXPECT_LE(p, 1.0f);
        total += p;
    }
    EXPECT_NEAR(total, 1.0f, 1e-5f);
}

TEST(BeatNetModel, RemembersWhatItHasSeen) {
    const auto blob = makeWeightFile();
    BeatNetWeights weights;
    ASSERT_TRUE(weights.load(blob.data(), blob.size()));
    BeatNetModel model(weights);

    const auto features = ramp(BeatNetWeights::kFeatures);
    float first[BeatNetWeights::kClasses];
    float second[BeatNetWeights::kClasses];
    model.forward(features.data(), first);
    model.forward(features.data(), second);

    // The same input twice must not give the same answer twice: an LSTM whose
    // state is not carried is two linear layers wearing a recurrent name.
    bool differ = false;
    for (std::size_t i = 0; i < BeatNetWeights::kClasses; ++i) {
        differ = differ || std::fabs(first[i] - second[i]) > 1e-7f;
    }
    EXPECT_TRUE(differ);
}

TEST(BeatNetModel, ResetForgetsIt) {
    const auto blob = makeWeightFile();
    BeatNetWeights weights;
    ASSERT_TRUE(weights.load(blob.data(), blob.size()));
    BeatNetModel model(weights);

    const auto features = ramp(BeatNetWeights::kFeatures);
    float first[BeatNetWeights::kClasses];
    float again[BeatNetWeights::kClasses];
    model.forward(features.data(), first);

    for (int i = 0; i < 20; ++i) model.forward(features.data(), again);
    model.reset();
    model.forward(features.data(), again);

    for (std::size_t i = 0; i < BeatNetWeights::kClasses; ++i) {
        EXPECT_FLOAT_EQ(first[i], again[i]);
    }
}

// ---------------------------------------------------------------- features --

TEST(BeatNetFeatures, ProducesFiftyFramesASecond) {
    BeatNetFeatures features(48000.0);
    const auto audio = clickTrack(120.0, 4.0, 48000.0);

    std::size_t frames = 0;
    double last = -1.0;
    features.process(audio.data(), audio.size(),
                     [&](const float*, std::size_t count, double time_sec) {
                         EXPECT_EQ(count, BeatNetFeatures::kFeatures);
                         EXPECT_GT(time_sec, last);
                         last = time_sec;
                         ++frames;
                     });

    // Four seconds is 200 frames; the last one cannot be completed without the
    // 32 ms of audio that follows it, which live there is no way to have.
    EXPECT_GE(frames, 198u);
    EXPECT_LE(frames, 200u);
    EXPECT_NEAR(last, static_cast<double>(frames - 1) * 0.02, 1e-9);
}

TEST(BeatNetFeatures, TheBlockSizeDoesNotChangeTheResult) {
    // The same property the STFT and the ODF are held to: a device hands over
    // whatever block size it likes, and the framing must not care.
    const auto audio = clickTrack(120.0, 3.0, 48000.0);

    auto collect = [&](std::size_t block) {
        BeatNetFeatures features(48000.0);
        std::vector<float> out;
        for (std::size_t pos = 0; pos < audio.size(); pos += block) {
            const std::size_t take = std::min(block, audio.size() - pos);
            features.process(audio.data() + pos, take,
                             [&](const float* row, std::size_t count, double) {
                                 out.insert(out.end(), row, row + count);
                             });
        }
        return out;
    };

    const auto reference = collect(64);
    for (std::size_t block : {1u, 137u, 441u, 1024u, 4096u}) {
        const auto other = collect(block);
        ASSERT_EQ(other.size(), reference.size()) << "block " << block;
        for (std::size_t i = 0; i < reference.size(); ++i) {
            ASSERT_FLOAT_EQ(other[i], reference[i]) << "block " << block << ", value " << i;
        }
    }
}

TEST(BeatNetFeatures, SilenceIsAllZeros) {
    // log10(1 + 0) is 0, and the difference of two zeros is zero. A front end
    // that reports anything here would be reporting its own arithmetic.
    BeatNetFeatures features(48000.0);
    const std::vector<float> quiet(48000, 0.0f);

    std::size_t frames = 0;
    features.process(quiet.data(), quiet.size(),
                     [&](const float* row, std::size_t count, double) {
                         for (std::size_t i = 0; i < count; ++i) {
                             ASSERT_NEAR(row[i], 0.0f, 1e-9f) << "frame " << frames
                                                              << ", value " << i;
                         }
                         ++frames;
                     });
    EXPECT_GT(frames, 0u);
}

TEST(BeatNetFeatures, TheDifferenceHalfIsNeverNegative) {
    BeatNetFeatures features(48000.0);
    const auto audio = clickTrack(120.0, 3.0, 48000.0);

    features.process(audio.data(), audio.size(),
                     [](const float* row, std::size_t count, double) {
                         for (std::size_t i = BeatNetFeatures::kFilters; i < count; ++i) {
                             ASSERT_GE(row[i], 0.0f) << "value " << i;
                         }
                     });
}

TEST(BeatNetFeatures, TheFirstFrameHasNothingToDifferenceAgainst) {
    BeatNetFeatures features(48000.0);
    const auto audio = clickTrack(120.0, 1.0, 48000.0);

    bool checked = false;
    features.process(audio.data(), audio.size(),
                     [&](const float* row, std::size_t count, double time_sec) {
                         if (checked || time_sec != 0.0) return;
                         for (std::size_t i = BeatNetFeatures::kFilters; i < count; ++i) {
                             EXPECT_FLOAT_EQ(row[i], 0.0f) << "value " << i;
                         }
                         checked = true;
                     });
    EXPECT_TRUE(checked);
}

TEST(BeatNetFeatures, ResetStartsTheFrameClockOver) {
    BeatNetFeatures features(48000.0);
    const auto audio = clickTrack(120.0, 2.0, 48000.0);

    auto firstFrame = [&]() {
        std::vector<float> out;
        double when = -1.0;
        features.process(audio.data(), audio.size(),
                         [&](const float* row, std::size_t count, double time_sec) {
                             if (!out.empty()) return;
                             out.assign(row, row + count);
                             when = time_sec;
                         });
        return std::make_pair(out, when);
    };

    const auto before = firstFrame();
    features.reset();
    const auto after = firstFrame();

    EXPECT_EQ(before.second, 0.0);
    EXPECT_EQ(after.second, 0.0);
    ASSERT_EQ(before.first.size(), after.first.size());
    for (std::size_t i = 0; i < before.first.size(); ++i) {
        EXPECT_FLOAT_EQ(before.first[i], after.first[i]) << "value " << i;
    }
}

// -------------------------------------------------------------- activation --

TEST(BeatNetActivation, IsCausal) {
    // The property the whole choice of model rests on: what the network says
    // about second one must not depend on what happens in second three. A
    // non-causal model can score better offline and is useless to a tracker
    // that has to answer now.
    const auto blob = makeWeightFile();
    BeatNetWeights weights;
    ASSERT_TRUE(weights.load(blob.data(), blob.size()));

    const auto audio = clickTrack(120.0, 4.0, 48000.0);
    const std::size_t half = audio.size() / 2;

    auto run = [&](std::size_t length) {
        BeatNetActivation activation(48000.0, weights);
        std::vector<double> out;
        activation.process(audio.data(), length,
                           [&](double, double beat, double) { out.push_back(beat); });
        return out;
    };

    const auto whole = run(audio.size());
    const auto truncated = run(half);
    ASSERT_GT(truncated.size(), 10u);
    ASSERT_LE(truncated.size(), whole.size());
    for (std::size_t i = 0; i < truncated.size(); ++i) {
        EXPECT_DOUBLE_EQ(truncated[i], whole[i]) << "frame " << i;
    }
}

TEST(BeatNetActivation, CountsFramesFromItsOwnZero) {
    // Frames are reported from the last reset(), not on a stream clock, and the
    // caller adds its own origin — the same contract dsp::Odf has. It matters
    // because when a device drops a buffer the audio's timestamps move while
    // its sample count does not, and only the caller can see that happen.
    const auto blob = makeWeightFile();
    BeatNetWeights weights;
    ASSERT_TRUE(weights.load(blob.data(), blob.size()));

    BeatNetActivation activation(48000.0, weights);
    const auto audio = clickTrack(120.0, 2.0, 48000.0);

    std::vector<double> times;
    activation.process(audio.data(), audio.size(),
                       [&](double t, double, double) { times.push_back(t); });

    ASSERT_GT(times.size(), 10u);
    EXPECT_NEAR(times.front(), 0.0, 1e-12);
    for (std::size_t i = 1; i < times.size(); ++i) {
        EXPECT_NEAR(times[i] - times[i - 1], 0.02, 1e-9) << "frame " << i;
    }
}

// ---------------------------------------------------------------- averaging --
//
// The measured gain from averaging BeatNet's three checkpoints is the largest
// available to the live path without training anything, so what this class has
// to guarantee is that it computes the same mean the research seam did — not
// merely something that combines several networks. These tests pin the
// arithmetic, because a bug here would not crash: it would produce a working
// tracker several points below the one that was measured.

namespace {

// Every frame of the activation, so two arrangements can be compared exactly.
std::vector<std::pair<double, double>> runActivation(BeatNetActivation& activation,
                                                     const std::vector<float>& audio) {
    std::vector<std::pair<double, double>> out;
    activation.process(audio.data(), audio.size(),
                       [&](double, double beat, double downbeat) {
                           out.emplace_back(beat, downbeat);
                       });
    return out;
}

}  // namespace

TEST(BeatNetActivation, TheAverageIsTheMeanOfTheNetworksItAverages) {
    // Two different weight sets, run separately and together. Frame by frame,
    // the ensemble must equal the arithmetic mean of the two — in probability
    // space, which is where the seam averaged them and therefore what every
    // measured number describes. Averaging logits instead would also pass a
    // test that only checked the result lay between the two.
    const auto blobA = makeWeightFile(1, BeatNetWeights::kFeatures, 0.7);
    const auto blobB = makeWeightFile(1, BeatNetWeights::kFeatures, 0.31);
    BeatNetWeights a, b;
    ASSERT_TRUE(a.load(blobA.data(), blobA.size()));
    ASSERT_TRUE(b.load(blobB.data(), blobB.size()));

    const auto audio = clickTrack(120.0, 3.0, 48000.0);
    BeatNetActivation alone_a(48000.0, a);
    BeatNetActivation alone_b(48000.0, b);
    const auto only_a = runActivation(alone_a, audio);
    const auto only_b = runActivation(alone_b, audio);

    const BeatNetWeights* both[] = {&a, &b};
    BeatNetActivation ensemble(48000.0, both, 2);
    const auto averaged = runActivation(ensemble, audio);

    ASSERT_EQ(ensemble.networks(), 2u);
    ASSERT_EQ(averaged.size(), only_a.size());
    ASSERT_EQ(averaged.size(), only_b.size());
    ASSERT_GT(averaged.size(), 10u);
    for (std::size_t i = 0; i < averaged.size(); ++i) {
        EXPECT_NEAR(averaged[i].first,
                    0.5 * (only_a[i].first + only_b[i].first), 1e-9) << "frame " << i;
        EXPECT_NEAR(averaged[i].second,
                    0.5 * (only_a[i].second + only_b[i].second), 1e-9) << "frame " << i;
    }

    // And the two networks must actually have disagreed, or the assertion above
    // is satisfied by any combining rule at all.
    double spread = 0.0;
    for (std::size_t i = 0; i < averaged.size(); ++i) {
        spread = std::max(spread, std::abs(only_a[i].first - only_b[i].first));
    }
    EXPECT_GT(spread, 1e-4) << "the two weight sets produced the same activation";
}

TEST(BeatNetActivation, OneNetworkAveragedWithItselfIsThatNetwork) {
    // The degenerate case, which is worth pinning because it is the one an
    // off-by-one in the scale factor would break while leaving the two-network
    // case looking plausible.
    const auto blob = makeWeightFile();
    BeatNetWeights weights;
    ASSERT_TRUE(weights.load(blob.data(), blob.size()));

    const auto audio = clickTrack(120.0, 2.0, 48000.0);
    BeatNetActivation single(48000.0, weights);
    const auto alone = runActivation(single, audio);

    const BeatNetWeights* twice[] = {&weights, &weights};
    BeatNetActivation doubled(48000.0, twice, 2);
    const auto both = runActivation(doubled, audio);

    ASSERT_EQ(both.size(), alone.size());
    for (std::size_t i = 0; i < both.size(); ++i) {
        EXPECT_NEAR(both[i].first, alone[i].first, 1e-12) << "frame " << i;
        EXPECT_NEAR(both[i].second, alone[i].second, 1e-12) << "frame " << i;
    }
}

TEST(BeatNetActivation, ResetForgetsEveryNetworkNotJustTheFirst) {
    // Each checkpoint carries its own LSTM state. Resetting only the first
    // would leave the ensemble remembering music that is no longer playing, on
    // two thirds of its networks, and the symptom would be an activation that
    // is subtly wrong after a stream restart rather than an obvious failure.
    const auto blobA = makeWeightFile(1, BeatNetWeights::kFeatures, 0.7);
    const auto blobB = makeWeightFile(1, BeatNetWeights::kFeatures, 0.31);
    BeatNetWeights a, b;
    ASSERT_TRUE(a.load(blobA.data(), blobA.size()));
    ASSERT_TRUE(b.load(blobB.data(), blobB.size()));

    const auto audio = clickTrack(120.0, 2.0, 48000.0);
    const BeatNetWeights* both[] = {&a, &b};
    BeatNetActivation ensemble(48000.0, both, 2);

    const auto first = runActivation(ensemble, audio);
    ensemble.reset();
    const auto second = runActivation(ensemble, audio);

    ASSERT_EQ(first.size(), second.size());
    ASSERT_GT(first.size(), 10u);
    for (std::size_t i = 0; i < first.size(); ++i) {
        EXPECT_NEAR(first[i].first, second[i].first, 1e-12) << "frame " << i;
        EXPECT_NEAR(first[i].second, second[i].second, 1e-12) << "frame " << i;
    }
}

TEST(BeatNetActivation, ADownbeatIsAlsoABeat) {
    const auto blob = makeWeightFile();
    BeatNetWeights weights;
    ASSERT_TRUE(weights.load(blob.data(), blob.size()));

    BeatNetActivation activation(48000.0, weights);
    const auto audio = clickTrack(120.0, 2.0, 48000.0);

    activation.process(audio.data(), audio.size(),
                       [](double, double beat, double downbeat) {
                           EXPECT_GE(beat, downbeat);
                           EXPECT_GE(beat, 0.0);
                           EXPECT_LE(beat, 1.0);
                       });
}
