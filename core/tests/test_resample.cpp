#include "dsp/resample.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <numeric>
#include <vector>

#include "dsp/fft.hpp"
#include "support.hpp"

using tiktak::dsp::Fft;
using tiktak::dsp::Resampler;
using tiktak::test::sine;

namespace {

// Energy at a frequency, measured by projecting onto it. Cheaper to read than
// a spectrum and immune to which bin a tone happens to land in.
double energyAt(const std::vector<float>& x, double hz, double rate) {
    double re = 0.0;
    double im = 0.0;
    for (std::size_t n = 0; n < x.size(); ++n) {
        const double t = 2.0 * 3.14159265358979323846 * hz * static_cast<double>(n) / rate;
        re += x[n] * std::cos(t);
        im -= x[n] * std::sin(t);
    }
    return 2.0 * std::sqrt(re * re + im * im) / static_cast<double>(x.size());
}

}  // namespace

TEST(Resampler, ReducesTheRatio) {
    const Resampler r(48000.0, 22050.0);
    EXPECT_EQ(r.up(), 147u);
    EXPECT_EQ(r.down(), 320u);

    const Resampler half(44100.0, 22050.0);
    EXPECT_EQ(half.up(), 1u);
    EXPECT_EQ(half.down(), 2u);
}

TEST(Resampler, TheSameRateIsACopy) {
    const Resampler r(22050.0, 22050.0);
    const auto in = sine(1000, 440.0, 22050.0);
    const auto out = r.apply(in.data(), in.size());
    ASSERT_EQ(out.size(), in.size());
    for (std::size_t i = 0; i < in.size(); ++i) EXPECT_FLOAT_EQ(out[i], in[i]);
}

TEST(Resampler, LengthFollowsTheRatio) {
    const Resampler r(48000.0, 22050.0);
    EXPECT_EQ(r.outputLength(48000), 22050u);
    EXPECT_EQ(r.outputLength(0), 0u);
    const Resampler half(44100.0, 22050.0);
    EXPECT_EQ(half.outputLength(1000), 500u);
}

TEST(Resampler, KeepsATonePassingThrough) {
    // Amplitude and frequency both survive. A resampler that got its gain wrong
    // would still look fine on a spectrogram and would quietly rescale every
    // feature a model reads.
    const Resampler r(48000.0, 22050.0);
    for (double hz : {110.0, 440.0, 1000.0, 4000.0}) {
        const auto in = sine(48000, hz, 48000.0);
        const auto out = r.apply(in.data(), in.size());
        ASSERT_EQ(out.size(), 22050u);

        // Skip the filter's settling at each edge.
        const std::vector<float> middle(out.begin() + 2000, out.end() - 2000);
        EXPECT_NEAR(energyAt(middle, hz, 22050.0), 1.0, 0.03) << hz << " Hz";
    }
}

TEST(Resampler, StopsWhatWouldOtherwiseFoldBack) {
    // The whole reason this class exists. A 16 kHz tone cannot exist below
    // 11025 Hz, so decimating without a filter puts it at 22050 - 16000 =
    // 6050 Hz, right in the middle of the music.
    const Resampler r(48000.0, 22050.0);
    const auto in = sine(48000, 16000.0, 48000.0);
    const auto out = r.apply(in.data(), in.size());
    const std::vector<float> middle(out.begin() + 2000, out.end() - 2000);

    const double alias = energyAt(middle, 22050.0 - 16000.0, 22050.0);
    EXPECT_LT(alias, 0.01) << "a 16 kHz tone folded back at " << alias;
}

TEST(Resampler, LinearInterpolationWouldHaveFailedThatTest) {
    // Stated as a test rather than a comment, so the claim in the header is
    // checked: this is what the shortcut actually does with the same input.
    const auto in = sine(48000, 16000.0, 48000.0);
    std::vector<float> naive(22050);
    const double ratio = 48000.0 / 22050.0;
    for (std::size_t n = 0; n < naive.size(); ++n) {
        const double position = static_cast<double>(n) * ratio;
        const auto i = static_cast<std::size_t>(position);
        const double fraction = position - static_cast<double>(i);
        naive[n] = i + 1 < in.size()
                       ? static_cast<float>(in[i] + fraction * (in[i + 1] - in[i]))
                       : in[i];
    }
    const std::vector<float> middle(naive.begin() + 2000, naive.end() - 2000);
    EXPECT_GT(energyAt(middle, 22050.0 - 16000.0, 22050.0), 0.1);
}

TEST(Resampler, SilenceStaysSilent) {
    const Resampler r(44100.0, 22050.0);
    const std::vector<float> quiet(4410, 0.0f);
    const auto out = r.apply(quiet.data(), quiet.size());
    for (float v : out) EXPECT_FLOAT_EQ(v, 0.0f);
}

TEST(Resampler, DoesNotShiftTheSignalInTime) {
    // A resampler that forgets its filter's group delay puts every onset half a
    // filter late — 65 ms at this ratio, which is a third of a beat at 120 BPM
    // and would look like a tracker that drags.
    const Resampler r(44100.0, 22050.0);
    std::vector<float> impulse(4410, 0.0f);
    impulse[2000] = 1.0f;

    const auto out = r.apply(impulse.data(), impulse.size());
    std::size_t peak = 0;
    for (std::size_t n = 1; n < out.size(); ++n) {
        if (std::fabs(out[n]) > std::fabs(out[peak])) peak = n;
    }
    EXPECT_NEAR(static_cast<double>(peak), 1000.0, 1.0);
}

TEST(Resampler, HandlesNothing) {
    const Resampler r(48000.0, 22050.0);
    EXPECT_TRUE(r.apply(nullptr, 0).empty());
    const std::vector<float> one(1, 1.0f);
    EXPECT_EQ(r.apply(one.data(), one.size()).size(), r.outputLength(1));
}
