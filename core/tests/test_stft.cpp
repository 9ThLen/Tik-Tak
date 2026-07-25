#include "dsp/stft.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

#include "support.hpp"

using tiktak::dsp::Stft;

namespace {

struct Captured {
    std::vector<std::size_t> frameStarts;
    std::vector<std::vector<float>> spectra;

    auto collector() {
        return [this](const float* magnitude, std::size_t bins, std::size_t start) {
            frameStarts.push_back(start);
            spectra.emplace_back(magnitude, magnitude + bins);
        };
    }
};

}  // namespace

TEST(Stft, ReportsItsShape) {
    Stft stft(1024, 256);
    EXPECT_EQ(stft.frameSize(), 1024u);
    EXPECT_EQ(stft.hopSize(), 256u);
    EXPECT_EQ(stft.spectrumSize(), 513u);
    EXPECT_EQ(stft.pending(), 0u);
}

TEST(Stft, EmitsNothingBeforeTheFirstFullWindow) {
    Stft stft(64, 16);
    Captured captured;

    const auto input = tiktak::test::silence(63);
    stft.process(input.data(), input.size(), captured.collector());

    EXPECT_TRUE(captured.frameStarts.empty());
    EXPECT_EQ(stft.pending(), 63u);
}

TEST(Stft, EmitsOneFramePerHopAfterTheFirstWindow) {
    constexpr std::size_t frame = 64;
    constexpr std::size_t hop = 16;
    constexpr std::size_t n = 64 + 5 * 16;  // first window, then five hops

    Stft stft(frame, hop);
    Captured captured;

    const auto input = tiktak::test::silence(n);
    stft.process(input.data(), input.size(), captured.collector());

    ASSERT_EQ(captured.frameStarts.size(), 6u);
    for (std::size_t i = 0; i < captured.frameStarts.size(); ++i) {
        EXPECT_EQ(captured.frameStarts[i], i * hop) << "frame " << i;
    }
}

TEST(Stft, FramesAvailableMatchesWhatProcessEmits) {
    constexpr std::size_t frame = 64;
    constexpr std::size_t hop = 16;

    for (std::size_t n : {0u, 1u, 63u, 64u, 65u, 80u, 200u, 1000u}) {
        Stft stft(frame, hop);
        Captured captured;

        const std::size_t predicted = stft.framesAvailable(n);
        const auto input = tiktak::test::silence(n);
        stft.process(input.data(), n, captured.collector());

        EXPECT_EQ(predicted, captured.frameStarts.size()) << "for n = " << n;
    }
}

TEST(Stft, BlockSizeDoesNotChangeTheResult) {
    // The analyser must be indifferent to how the host chops up the stream —
    // an audio callback delivers whatever block size the device feels like.
    constexpr std::size_t frame = 64;
    constexpr std::size_t hop = 16;
    const auto input = tiktak::test::sine(512, 1000.0, 48000.0);

    Captured wholeBuffer;
    Stft a(frame, hop);
    a.process(input.data(), input.size(), wholeBuffer.collector());

    for (std::size_t block : {1u, 3u, 7u, 16u, 100u}) {
        Captured chunked;
        Stft b(frame, hop);
        for (std::size_t pos = 0; pos < input.size(); pos += block) {
            const std::size_t take = std::min(block, input.size() - pos);
            b.process(input.data() + pos, take, chunked.collector());
        }

        ASSERT_EQ(chunked.spectra.size(), wholeBuffer.spectra.size())
            << "block size " << block;
        EXPECT_EQ(chunked.frameStarts, wholeBuffer.frameStarts) << "block size " << block;

        for (std::size_t f = 0; f < chunked.spectra.size(); ++f) {
            for (std::size_t k = 0; k < chunked.spectra[f].size(); ++k) {
                EXPECT_NEAR(chunked.spectra[f][k], wholeBuffer.spectra[f][k], 1e-4f)
                    << "block " << block << ", frame " << f << ", bin " << k;
            }
        }
    }
}

TEST(Stft, ToneAppearsAtTheExpectedBin) {
    constexpr std::size_t frame = 1024;
    constexpr double sampleRate = 48000.0;
    constexpr double freq = 1000.0;

    Stft stft(frame, frame / 4);
    Captured captured;

    const auto input = tiktak::test::sine(frame * 2, freq, sampleRate);
    stft.process(input.data(), input.size(), captured.collector());

    ASSERT_FALSE(captured.spectra.empty());
    const auto& spectrum = captured.spectra.front();
    const auto peak = std::max_element(spectrum.begin(), spectrum.end());
    const std::size_t peakBin = static_cast<std::size_t>(peak - spectrum.begin());

    const double expectedBin = freq * static_cast<double>(frame) / sampleRate;
    EXPECT_NEAR(static_cast<double>(peakBin), expectedBin, 1.0);
}

TEST(Stft, ResetRestartsTheFrameClock) {
    Stft stft(64, 16);
    Captured first;

    const auto input = tiktak::test::silence(128);
    stft.process(input.data(), input.size(), first.collector());
    ASSERT_FALSE(first.frameStarts.empty());
    EXPECT_NE(first.frameStarts.back(), 0u);

    stft.reset();
    EXPECT_EQ(stft.pending(), 0u);

    Captured second;
    stft.process(input.data(), input.size(), second.collector());
    ASSERT_FALSE(second.frameStarts.empty());
    EXPECT_EQ(second.frameStarts.front(), 0u);
    EXPECT_EQ(second.frameStarts, first.frameStarts);
}

TEST(Stft, SupportsHopEqualToFrameSize) {
    // Degenerate but legal: no overlap at all.
    Stft stft(64, 64);
    Captured captured;

    const auto input = tiktak::test::silence(256);
    stft.process(input.data(), input.size(), captured.collector());

    ASSERT_EQ(captured.frameStarts.size(), 4u);
    EXPECT_EQ(captured.frameStarts[1], 64u);
}
