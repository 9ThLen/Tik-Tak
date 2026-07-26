#include "dsp/chroma.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include "dsp/stft.hpp"
#include "support.hpp"

namespace {

using tiktak::dsp::ChromaFilterbank;
using tiktak::dsp::chromaDistance;

constexpr double kRate = 48000.0;
constexpr std::size_t kFrame = 4096;

// Equal temperament. Counted in semitones from A4 = 440, which is the only way
// to name a pitch here that cannot quietly be off by an octave.
double pitch(int semitones_from_a4) {
    return 440.0 * std::pow(2.0, semitones_from_a4 / 12.0);
}

// Pitch classes as offsets from A4, so a test reads as notes and not numbers.
// The octave is chosen to sit above what a 4096-sample window at 48 kHz can
// resolve (about 405 Hz) — below that the filterbank deliberately hears
// nothing, and a test written on middle C would be testing the clamp.
constexpr int kC5 = 3;
constexpr int kE5 = 7;
constexpr int kG5 = 10;
constexpr int kA5 = 12;

// Runs one analysis window of a signal through the filterbank.
std::array<float, ChromaFilterbank::kBins> profile(const std::vector<float>& signal,
                                                   const ChromaFilterbank& bank) {
    tiktak::dsp::Stft stft(kFrame, kFrame);
    std::array<float, ChromaFilterbank::kBins> out{};
    bool got = false;
    stft.process(signal.data(), signal.size(),
                 [&](const float* magnitude, std::size_t, std::size_t) {
                     if (got) return;
                     bank.apply(magnitude, out.data());
                     got = true;
                 });
    EXPECT_TRUE(got);
    return out;
}

std::vector<float> chord(const std::vector<int>& notes, float amplitude = 1.0f) {
    std::vector<float> out(kFrame, 0.0f);
    for (int n : notes) {
        const auto tone = tiktak::test::sine(kFrame, pitch(n), kRate, amplitude);
        for (std::size_t i = 0; i < kFrame; ++i) out[i] += tone[i];
    }
    return out;
}

std::size_t loudest(const std::array<float, ChromaFilterbank::kBins>& p) {
    return static_cast<std::size_t>(std::max_element(p.begin(), p.end()) - p.begin());
}

// Bin 0 is C, so a note's bin is its distance above C in semitones.
std::size_t binOf(int semitones_from_a4) {
    return static_cast<std::size_t>(((semitones_from_a4 - kC5) % 12 + 12) % 12);
}

class Chroma : public ::testing::Test {
protected:
    ChromaFilterbank bank_{kFrame, kRate};
};

TEST_F(Chroma, PutsANoteOnItsOwnPitchClass) {
    EXPECT_EQ(loudest(profile(chord({kC5}), bank_)), binOf(kC5));
    EXPECT_EQ(loudest(profile(chord({kA5}), bank_)), binOf(kA5));
    EXPECT_EQ(loudest(profile(chord({kE5}), bank_)), binOf(kE5));
}

TEST_F(Chroma, TheSameNoteAnOctaveUpIsTheSameNote) {
    // The whole point of folding octaves together: a chord voiced high and the
    // same chord voiced low have to compare as identical, or every change of
    // register would read as a change of harmony.
    const auto low = profile(chord({kC5}), bank_);
    const auto high = profile(chord({kC5 + 12}), bank_);

    EXPECT_EQ(loudest(low), loudest(high));
    EXPECT_LT(chromaDistance(low.data(), high.data()), 0.05);
}

TEST_F(Chroma, ADifferentChordIsADifferentDirection) {
    const auto c_major = profile(chord({kC5, kE5, kG5}), bank_);
    const auto a_minor = profile(chord({kA5, kC5 + 12, kE5 + 12}), bank_);

    // They share two notes out of three, so they are not opposites — but they
    // must be far enough apart that a bar line built on this can see them.
    const double d = chromaDistance(c_major.data(), a_minor.data());
    EXPECT_GT(d, 0.15);
    EXPECT_LT(d, 1.0);
}

TEST_F(Chroma, TheSameChordTwiceHasNotChanged) {
    const auto a = profile(chord({kC5, kE5, kG5}), bank_);
    const auto b = profile(chord({kC5, kE5, kG5}), bank_);
    EXPECT_NEAR(chromaDistance(a.data(), b.data()), 0.0, 1e-6);
}

TEST_F(Chroma, GettingLouderIsNotAChordChange) {
    // A crescendo must not read as a modulation, which is exactly what an
    // un-normalised profile would report.
    const auto quiet = profile(chord({kC5, kE5, kG5}, 0.05f), bank_);
    const auto loud = profile(chord({kC5, kE5, kG5}, 1.0f), bank_);
    EXPECT_LT(chromaDistance(quiet.data(), loud.data()), 0.02);
}

TEST_F(Chroma, SilenceIsNotAModulation) {
    const auto nothing = profile(tiktak::test::silence(kFrame), bank_);
    for (float v : nothing) EXPECT_FLOAT_EQ(v, 0.0f);

    const auto something = profile(chord({kC5, kE5, kG5}), bank_);
    EXPECT_DOUBLE_EQ(chromaDistance(nothing.data(), something.data()), 0.0);
    EXPECT_DOUBLE_EQ(chromaDistance(nothing.data(), nothing.data()), 0.0);
}

TEST_F(Chroma, EnergyBetweenSemitonesIsHeldBack) {
    // A tone a quarter-tone off is not evidence for either neighbour, and the
    // weighting is what keeps drums and noise from reading as chords. It cannot
    // be suppressed entirely — the analysis window smears every tone across
    // several bins — but it must land well below a tuned note of the same size.
    const auto tuned = profile(chord({kC5}), bank_);
    std::vector<float> detuned(kFrame, 0.0f);
    const auto tone = tiktak::test::sine(kFrame, pitch(kC5) * std::pow(2.0, 0.5 / 12.0), kRate);
    for (std::size_t i = 0; i < kFrame; ++i) detuned[i] = tone[i];
    const auto between = profile(detuned, bank_);

    // Tuned energy concentrates on one class; detuned energy is split between
    // the two it sits between, so the peak is markedly lower after normalising.
    EXPECT_GT(tuned[loudest(tuned)], between[loudest(between)] * 1.3f);
}

TEST_F(Chroma, WillNotAnswerBelowWhatItCanResolve) {
    // Asking for A1 does not get A1. A semitone down there is a couple of hertz
    // wide and an FFT bin is a couple of dozen, so any answer would be built
    // from neighbouring semitones sharing bins. The range is raised instead,
    // and says so, rather than reporting smear as harmony.
    EXPECT_GT(bank_.minHz(), 55.0);
    EXPECT_NEAR(bank_.minHz(), ChromaFilterbank::resolvedMinHz(kFrame, kRate), 1e-9);

    // Doubling the window halves the bin width and halves the floor with it —
    // the limit is the transform's, not a constant someone picked.
    EXPECT_NEAR(ChromaFilterbank::resolvedMinHz(2 * kFrame, kRate),
                ChromaFilterbank::resolvedMinHz(kFrame, kRate) * 0.5, 1e-9);

    // A note below the floor contributes nothing at all.
    const auto below = profile(chord({kC5 - 24}), bank_);  // C3, a pure tone
    for (float v : below) EXPECT_FLOAT_EQ(v, 0.0f);
}

TEST_F(Chroma, ARangeOutsideTheSpectrumIsSilentRatherThanWrong) {
    ChromaFilterbank empty(kFrame, kRate, 30000.0, 40000.0);
    EXPECT_EQ(empty.contributingBins(), 0u);

    const auto p = profile(chord({kC5, kE5, kG5}), empty);
    for (float v : p) EXPECT_FLOAT_EQ(v, 0.0f);
}

TEST_F(Chroma, IgnoresNothing) {
    std::array<float, ChromaFilterbank::kBins> out{};
    bank_.apply(nullptr, out.data());
    for (float v : out) EXPECT_FLOAT_EQ(v, 0.0f);

    EXPECT_DOUBLE_EQ(chromaDistance(nullptr, out.data()), 0.0);
    EXPECT_DOUBLE_EQ(chromaDistance(out.data(), nullptr), 0.0);
}

}  // namespace
