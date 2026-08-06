// What the octave freeze must do before any corpus number about it means
// anything. Specified in eval/PREREGISTERED_octave_freeze.md and written
// before the arm was measured on anything.
//
// The arm changes which octave the filter is anchored to when the activation
// tempo estimator stops being sure. Everything below is about the state
// machine and about what it must *not* touch — the tempo inside the octave,
// and the phase.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "support.hpp"
#include "tracking/live.hpp"

using tiktak::tracking::LiveConfig;
using tiktak::tracking::LiveTracker;
using tiktak::tracking::octaveNearest;

namespace {

constexpr double kRate = 48000.0;
constexpr double kFps = 50.0;

struct ResolverProbe {
    std::size_t calls = 0;
    double factor = 1.0;
    bool invalid = false;
};

double resolveAnchor(void* context, double, double measured_bpm) {
    auto& probe = *static_cast<ResolverProbe*>(context);
    ++probe.calls;
    if (probe.invalid) return std::numeric_limits<double>::quiet_NaN();
    return measured_bpm * probe.factor;
}

LiveConfig freezeConfig(double margin, bool freeze = true) {
    LiveConfig config = tiktak::tracking::liveConfigFor(kRate);
    config.anchor_octave_margin = margin;
    config.anchor_octave_freeze = freeze;
    return config;
}

// Drives the tracker with an activation that is confident about its metrical
// level: alternating strong and weak hits, so the beat period is preferred
// over its subdivision rather than merely tied with it.
double driveAlternating(LiveTracker& tracker, double bpm, double seconds,
                        double from_sec = 0.0) {
    const double period = 60.0 / bpm;
    const auto frames = static_cast<std::size_t>(seconds * kFps);
    std::size_t beat = 0;
    double last = from_sec;
    for (std::size_t i = 0; i < frames; ++i) {
        const double time = from_sec + static_cast<double>(i) / kFps;
        const double since = time / period;
        const double nearest = std::round(since);
        double value = 0.02;
        if (std::fabs(since - nearest) < 0.5 / (kFps * period)) {
            beat = static_cast<std::size_t>(nearest);
            value = beat % 2 == 0 ? 0.95 : 0.45;
        }
        tracker.observe(time, value);
        last = time;
    }
    return last;
}

// Nothing at all, for the timeout: an absence of evidence rather than weak
// evidence, which is the case the expiry has to cover and the one it missed.
double driveSilence(LiveTracker& tracker, double seconds, double from_sec) {
    const auto frames = static_cast<std::size_t>(seconds * kFps);
    double last = from_sec;
    for (std::size_t i = 0; i < frames; ++i) {
        const double time = from_sec + static_cast<double>(i) / kFps;
        tracker.observe(time, 0.0);
        last = time;
    }
    return last;
}

std::vector<double> beatsOf(LiveTracker& tracker, double bpm, double seconds) {
    std::vector<double> out;
    const double period = 60.0 / bpm;
    const auto frames = static_cast<std::size_t>(seconds * kFps);
    for (std::size_t i = 0; i < frames; ++i) {
        const double time = static_cast<double>(i) / kFps;
        const double since = time / period;
        const double nearest = std::round(since);
        double value = 0.02;
        if (std::fabs(since - nearest) < 0.5 / (kFps * period)) {
            value = static_cast<std::size_t>(nearest) % 2 == 0 ? 0.95 : 0.45;
        }
        tracker.observe(time, value);
        double beat = 0.0;
        if (tracker.takeBeat(time, 0.05, &beat)) out.push_back(beat);
    }
    return out;
}

}  // namespace

// ------------------------------------------------- the mapping, on its own

TEST(OctaveFreeze, MapsToTheNearestOctaveOfTheHold) {
    // Pre-registration test 2. A hold of 120 with the estimator reporting 60,
    // 240 and 61 must anchor at 120, 120 and 122 — the nearest octave
    // equivalent of what the estimator said, never the held value itself.
    EXPECT_DOUBLE_EQ(octaveNearest(60.0, 120.0), 120.0);
    EXPECT_DOUBLE_EQ(octaveNearest(240.0, 120.0), 120.0);
    EXPECT_DOUBLE_EQ(octaveNearest(61.0, 120.0), 122.0);
    EXPECT_DOUBLE_EQ(octaveNearest(30.0, 120.0), 120.0);
}

TEST(OctaveFreeze, DoesNotFreezeTheTempoInsideTheOctave) {
    // Pre-registration test 3, and the distinction the whole arm rests on:
    // the doubt is over which multiple of the pulse is the beat, never over
    // the pulse. A band drifting 128 -> 132 under a hold of 120 anchors at
    // 132, not at 120 and not at 128.
    EXPECT_DOUBLE_EQ(octaveNearest(132.0, 120.0), 132.0);
    EXPECT_DOUBLE_EQ(octaveNearest(128.0, 120.0), 128.0);
    // And the boundary is in log tempo: 170 is nearer 120 doubled-down than
    // left alone only past sqrt(2) * 120 = 169.7.
    EXPECT_DOUBLE_EQ(octaveNearest(168.0, 120.0), 168.0);
    EXPECT_DOUBLE_EQ(octaveNearest(172.0, 120.0), 86.0);
}

TEST(OctaveFreeze, LeavesAnEmptyHoldAlone) {
    // Nothing held is not the same as holding zero, and a caller in that state
    // must be handed back exactly what it passed in.
    EXPECT_DOUBLE_EQ(octaveNearest(120.0, 0.0), 120.0);
    EXPECT_DOUBLE_EQ(octaveNearest(0.0, 120.0), 0.0);
}

TEST(AnchorResolver, ChangesOnlyTheAnchorBpmAtTheExistingSeam) {
    ResolverProbe probe{0, 0.5, false};
    LiveConfig config = freezeConfig(0.0, false);
    config.anchor_bpm_resolver = &resolveAnchor;
    config.anchor_bpm_resolver_context = &probe;
    LiveTracker tracker(config);

    driveAlternating(tracker, 120.0, 12.0);

    ASSERT_GT(probe.calls, 0u);
    EXPECT_NEAR(tracker.heldOctaveBpm(), 60.0, 3.0);
}

TEST(AnchorResolver, InvalidAnswerFallsBackToTheMeasuredBpm) {
    ResolverProbe probe{0, 1.0, true};
    LiveConfig baseline_config = freezeConfig(0.0, false);
    LiveConfig resolver_config = baseline_config;
    resolver_config.anchor_bpm_resolver = &resolveAnchor;
    resolver_config.anchor_bpm_resolver_context = &probe;
    LiveTracker baseline(baseline_config);
    LiveTracker resolver(resolver_config);

    driveAlternating(baseline, 120.0, 12.0);
    driveAlternating(resolver, 120.0, 12.0);

    ASSERT_GT(probe.calls, 0u);
    EXPECT_DOUBLE_EQ(resolver.heldOctaveBpm(), baseline.heldOctaveBpm());
}

TEST(AnchorResolver, IsInertWhenTheAnchorIsOff) {
    ResolverProbe probe{0, 0.5, false};
    LiveConfig config = freezeConfig(0.0, false);
    config.anchor_tempo = false;
    config.anchor_bpm_resolver = &resolveAnchor;
    config.anchor_bpm_resolver_context = &probe;
    LiveTracker tracker(config);

    driveAlternating(tracker, 120.0, 12.0);

    EXPECT_EQ(probe.calls, 0u);
    EXPECT_DOUBLE_EQ(tracker.heldOctaveBpm(), 0.0);
}

// --------------------------------------------- the arm against the baseline

TEST(OctaveFreeze, IsTheBaselineWhileNothingHasBeenHeld) {
    // Pre-registration test 1. With a margin no estimate can clear, no hold is
    // ever taken, and every anchor decision must be the baseline's — refusing
    // to anchor at the start of a recording is the arm that already lost.
    LiveConfig frozen = freezeConfig(1.1);
    LiveConfig plain = tiktak::tracking::liveConfigFor(kRate);

    LiveTracker a(frozen);
    LiveTracker b(plain);
    const auto with = beatsOf(a, 120.0, 20.0);
    const auto without = beatsOf(b, 120.0, 20.0);

    EXPECT_DOUBLE_EQ(a.heldOctaveBpm(), 0.0);
    ASSERT_EQ(with.size(), without.size());
    for (std::size_t i = 0; i < with.size(); ++i) {
        EXPECT_DOUBLE_EQ(with[i], without[i]) << "beat " << i;
    }
}

TEST(OctaveFreeze, DoesNotTouchThePhaseWhileTheMarginHolds) {
    // Pre-registration test 6, and the one worth writing first: it fails
    // loudly if the octave mapping is implemented anywhere near the phase.
    // With a margin every estimate clears, the freeze branch is unreachable
    // and the beats must be identical to the baseline's.
    LiveConfig frozen = freezeConfig(0.0);
    LiveConfig plain = tiktak::tracking::liveConfigFor(kRate);

    LiveTracker a(frozen);
    LiveTracker b(plain);
    const auto with = beatsOf(a, 120.0, 20.0);
    const auto without = beatsOf(b, 120.0, 20.0);

    ASSERT_EQ(with.size(), without.size());
    for (std::size_t i = 0; i < with.size(); ++i) {
        EXPECT_DOUBLE_EQ(with[i], without[i]) << "beat " << i;
    }
}

// ------------------------------------------------------- the state machine

TEST(OctaveFreeze, TakesAHoldOnlyFromAConfidentAnchor) {
    LiveTracker tracker(freezeConfig(0.5));
    EXPECT_DOUBLE_EQ(tracker.heldOctaveBpm(), 0.0);

    driveAlternating(tracker, 120.0, 20.0);
    ASSERT_GE(tracker.tempoFromActivation().octave_margin, 0.5)
        << "the alternating drive is meant to be metrically unambiguous; "
           "if this fails the test's material is wrong, not the tracker";
    EXPECT_GT(tracker.heldOctaveBpm(), 0.0);
}

TEST(OctaveFreeze, AConfidentAnchorReplacesTheHoldImmediately) {
    // Pre-registration test 4. The freeze survives an absence of evidence and
    // never outvotes evidence, so a confident anchor elsewhere wins outright.
    //
    // 120 then 60, an octave apart, because both are material this estimator
    // is decided about — it clears 0.5 comfortably at 60, 90, 100 and 120 and
    // does not at 140 or above, so a test that used a fast second tempo would
    // be measuring the estimator's range and not the freeze.
    LiveTracker tracker(freezeConfig(0.5));
    const double after = driveAlternating(tracker, 120.0, 20.0);
    const double first = tracker.heldOctaveBpm();
    ASSERT_NEAR(first, 120.0, 6.0);

    driveAlternating(tracker, 60.0, 30.0, after + 0.02);
    ASSERT_GE(tracker.tempoFromActivation().octave_margin, 0.5)
        << "the second drive is meant to be metrically unambiguous; if this "
           "fails the test's material is wrong, not the tracker";
    EXPECT_NEAR(tracker.heldOctaveBpm(), 60.0, 3.0)
        << "a confident anchor an octave away must replace the hold, not be "
           "mapped back onto it";
}

TEST(OctaveFreeze, TheHoldExpiresAfterTheTimeout) {
    // Pre-registration test 5, and the concession to live.cpp's objection: a
    // tempo measured in the first chorus must not outlive the evidence for it.
    //
    // Silence rather than ambiguous material, because silence is the case that
    // exposed the hole this test was first written around: with no estimate at
    // all there is no margin to be weak, so an expiry checked only on the
    // weak-margin branch never fired and the hold was immortal.
    LiveConfig config = freezeConfig(0.5);
    config.anchor_freeze_timeout_sec = 4.0;
    LiveTracker tracker(config);

    const double after = driveAlternating(tracker, 120.0, 20.0);
    ASSERT_GT(tracker.heldOctaveBpm(), 0.0);

    // Generously past the timeout: the estimator stays confident for a few
    // seconds into silence on the history it already has, and the clock starts
    // from the last confident anchor rather than from the last sound.
    driveSilence(tracker, 20.0, after + 0.02);
    ASSERT_FALSE(tracker.tempoFromActivation().answered());
    EXPECT_DOUBLE_EQ(tracker.heldOctaveBpm(), 0.0);
}

TEST(OctaveFreeze, TheHoldSurvivesASilenceShorterThanTheTimeout) {
    // The other side of the same rule, so that "expires" cannot be satisfied by
    // an implementation that simply drops the hold whenever the estimator goes
    // quiet.
    LiveConfig config = freezeConfig(0.5);
    config.anchor_freeze_timeout_sec = 30.0;
    LiveTracker tracker(config);

    const double after = driveAlternating(tracker, 120.0, 20.0);
    const double held = tracker.heldOctaveBpm();
    ASSERT_GT(held, 0.0);

    driveSilence(tracker, 20.0, after + 0.02);
    ASSERT_FALSE(tracker.tempoFromActivation().answered());
    EXPECT_DOUBLE_EQ(tracker.heldOctaveBpm(), held);
}

// ------------------------------------------------------------- the abstain arm

TEST(OctaveFreeze, AbstainReportsNoConfidenceWhileTheMarginIsWeak) {
    LiveConfig config = tiktak::tracking::liveConfigFor(kRate);
    config.anchor_octave_margin = 1.1;  // nothing can clear it
    config.anchor_margin_abstain = true;
    LiveTracker tracker(config);

    const double after = driveAlternating(tracker, 120.0, 20.0);
    EXPECT_DOUBLE_EQ(tracker.estimate(after).confidence, 0.0);

    // The tempo is deliberately left in place: a caller showing the last known
    // BPM greyed out is being honest, and zeroing it would claim the tracker
    // had forgotten.
    EXPECT_GT(tracker.estimate(after).bpm, 0.0);
}

TEST(OctaveFreeze, AbstainWithholdsTheBeatsAsWellAsTheMeter) {
    // "Publish nothing" has to reach the clicks. A tracker reporting no
    // confidence while still handing out beats would be counted as silent by
    // the lock and as speaking by the beat measure, and the arm would be a
    // bound on neither.
    // Not "no beats at all": before the estimator has answered there is no
    // margin to be weak, and the arm behaves as the baseline for the same
    // reason the freeze does when nothing is held yet. What must be true is
    // that the beats stop once it starts answering.
    LiveConfig config = tiktak::tracking::liveConfigFor(kRate);
    config.anchor_octave_margin = 1.1;  // nothing can clear it
    config.anchor_margin_abstain = true;
    LiveTracker tracker(config);

    const auto beats = beatsOf(tracker, 120.0, 25.0);
    const auto late = std::count_if(beats.begin(), beats.end(),
                                    [](double t) { return t > 12.0; });
    EXPECT_EQ(late, 0) << "the arm went on publishing after the estimator had "
                          "an opinion to withhold";
}

TEST(OctaveFreeze, AbstainDoesNotAlsoDropTheAnchor) {
    // The arm is a bound on what silence alone can buy, so it must not carry a
    // second change. Its threshold is a publishing threshold: the anchor keeps
    // the baseline's behaviour, and the held octave — which only the freeze
    // touches — stays out of it entirely.
    LiveConfig abstaining = tiktak::tracking::liveConfigFor(kRate);
    abstaining.anchor_octave_margin = 1.1;
    abstaining.anchor_margin_abstain = true;
    LiveTracker muted(abstaining);
    LiveTracker plain(tiktak::tracking::liveConfigFor(kRate));

    driveAlternating(muted, 120.0, 20.0);
    driveAlternating(plain, 120.0, 20.0);

    // Same tempo underneath, because the same anchors were applied.
    EXPECT_NEAR(muted.estimate(20.0).bpm, plain.estimate(20.0).bpm, 1e-9);
    EXPECT_GT(plain.estimate(20.0).confidence, 0.0);
    EXPECT_DOUBLE_EQ(muted.estimate(20.0).confidence, 0.0);
}

TEST(OctaveFreeze, AbstainIsInertWhenTheMarginIsAlwaysMet) {
    LiveConfig config = tiktak::tracking::liveConfigFor(kRate);
    config.anchor_margin_abstain = true;  // margin defaults to 0.0
    LiveTracker abstaining(config);
    LiveTracker plain(tiktak::tracking::liveConfigFor(kRate));

    const auto with = beatsOf(abstaining, 120.0, 20.0);
    const auto without = beatsOf(plain, 120.0, 20.0);

    ASSERT_EQ(with.size(), without.size());
    for (std::size_t i = 0; i < with.size(); ++i) {
        EXPECT_DOUBLE_EQ(with[i], without[i]) << "beat " << i;
    }
}
