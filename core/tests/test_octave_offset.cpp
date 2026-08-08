// What the user's octave control must do, and what it must not touch.
//
// The control exists because the metrical level is the largest measured loss on
// full-length material and is not recoverable from the model's own outputs —
// see tracking/live.hpp for both halves of that. A person deciding it is the
// remaining route, and the ceiling on it is 39.0% -> 60.0% usable on RWC-Pop.
//
// The claim under test is narrow and is the whole point: a press outranks the
// tracker's own octave *and keeps doing so*. LiveTracker::submit writes an
// anchor from the activation-tempo estimator on every submitted frame, so a
// control that only moved the cloud would be overruled within about a second
// and would read as a button that does not work.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "support.hpp"
#include "tracking/live.hpp"
#include "tracking/particle.hpp"

using tiktak::tracking::BeatEstimate;
using tiktak::tracking::BeatParticleFilter;
using tiktak::tracking::LiveConfig;
using tiktak::tracking::LiveTracker;
using tiktak::tracking::octaveNearest;
using tiktak::tracking::ParticleFilterConfig;

namespace {

constexpr double kRate = 48000.0;
constexpr double kFps = 50.0;

// 100 BPM throughout: ×2 is 200 and ÷2 is 50, both inside the default 40..220,
// so a refusal in these tests is a real refusal and never the range guard. The
// range case has its own test below.
constexpr double kBpm = 100.0;

LiveConfig offsetConfig(bool freeze = true, double margin = 0.0) {
    LiveConfig config = tiktak::tracking::liveConfigFor(kRate);
    config.anchor_octave_freeze = freeze;
    config.anchor_octave_margin = margin;
    return config;
}

// Strong and weak hits alternating, so the beat period is preferred over its
// subdivision rather than merely tied with it — the estimator needs a margin
// before the anchor is written at all.
double driveAlternating(LiveTracker& tracker, double bpm, double seconds,
                        double from_sec = 0.0) {
    const double period = 60.0 / bpm;
    const auto frames = static_cast<std::size_t>(seconds * kFps);
    double last = from_sec;
    for (std::size_t i = 0; i < frames; ++i) {
        const double time = from_sec + static_cast<double>(i) / kFps;
        const double since = time / period;
        const double nearest = std::round(since);
        double value = 0.02;
        if (std::fabs(since - nearest) < 0.5 / (kFps * period)) {
            value = static_cast<std::size_t>(nearest) % 2 == 0 ? 0.95 : 0.45;
        }
        tracker.observe(time, value);
        last = time;
    }
    return last;
}

// Hits at `bpm`, every one the same strength. Driven at twice the beat rate
// this is the ambiguous case — the subdivision is indistinguishable from the
// beat, and the estimator's octave margin collapses — which is how the
// weak-margin branch is reached without changing configuration mid-recording.
// Note that uniform hits at the beat rate are *not* ambiguous: nothing falls
// between them, so the doubled period has nothing to correlate with.
double driveUniform(LiveTracker& tracker, double bpm, double seconds,
                    double from_sec = 0.0) {
    const double period = 60.0 / bpm;
    const auto frames = static_cast<std::size_t>(seconds * kFps);
    double last = from_sec;
    for (std::size_t i = 0; i < frames; ++i) {
        const double time = from_sec + static_cast<double>(i) / kFps;
        const double since = time / period;
        const double nearest = std::round(since);
        double value = 0.02;
        if (std::fabs(since - nearest) < 0.5 / (kFps * period)) value = 0.95;
        tracker.observe(time, value);
        last = time;
    }
    return last;
}

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

// Drives one filter directly, for the primitive's own tests.
double driveFilter(BeatParticleFilter& filter, double bpm, double seconds) {
    const double period = 60.0 / bpm;
    const auto frames = static_cast<std::size_t>(seconds * kFps);
    double last = 0.0;
    for (std::size_t i = 0; i < frames; ++i) {
        const double time = static_cast<double>(i) / kFps;
        const double since = time / period;
        const double nearest = std::round(since);
        double value = 0.0;
        if (std::fabs(since - nearest) < 0.5 / (kFps * period)) value = 1.0;
        filter.observe(time, value);
        last = time;
    }
    return last;
}

}  // namespace

// ------------------------------------------------------- the primitive alone

TEST(ScalePeriod, MovesTheTempoByTheFactorAndKeepsTheGrid) {
    ParticleFilterConfig config;
    BeatParticleFilter filter(config);
    const double now = driveFilter(filter, kBpm, 20.0);

    const BeatEstimate before = filter.estimate(now);
    ASSERT_NEAR(before.bpm, kBpm, 6.0);

    filter.scalePeriod(0.5);
    const BeatEstimate after = filter.estimate(now);

    // Twice the tempo, to within what the cloud's own spread allows.
    EXPECT_NEAR(after.bpm, 2.0 * before.bpm, 0.1 * before.bpm);

    // And the beat it now names is on the grid the old one belonged to: either
    // the same beat, or the one halfway to it.
    const double old_period = 60.0 / before.bpm;
    const double delta = before.next_beat_sec - after.next_beat_sec;
    const double halves = delta / (0.5 * old_period);
    EXPECT_LT(std::fabs(halves - std::round(halves)), 0.15);
}

TEST(ScalePeriod, DoesNotSilenceTheClick) {
    // The reason this is not seedTempo(): re-drawing the cloud flattens the
    // weights, confidence collapses through the release threshold, and the user
    // hears the click stop because they asked it to count differently.
    ParticleFilterConfig config;
    BeatParticleFilter filter(config);
    const double now = driveFilter(filter, kBpm, 20.0);

    const double before = filter.estimate(now).confidence;
    ASSERT_GT(before, 0.25);

    filter.scalePeriod(0.5);
    const double after = filter.estimate(now).confidence;

    // Not asserted equal: phase agreement is a spread measured as a fraction of
    // a beat, so the same absolute uncertainty reads larger against a shorter
    // one. What must survive is the lock.
    EXPECT_GT(after, LiveConfig{}.release_confidence);

    // Against the alternative, on the same cloud driven the same way.
    BeatParticleFilter reseeded(config);
    driveFilter(reseeded, kBpm, 20.0);
    reseeded.seedTempo(2.0 * kBpm, 0.05);
    EXPECT_LT(reseeded.estimate(now).confidence, after);
}

TEST(ScalePeriod, NamesTheNearestBeatAndNotOneItSkipped) {
    // Halving the period without walking the grid back leaves "next" up to a
    // full new period late — one missed click, every press.
    ParticleFilterConfig config;
    BeatParticleFilter filter(config);
    const double now = driveFilter(filter, kBpm, 20.0);

    filter.scalePeriod(0.5);
    const BeatEstimate after = filter.estimate(now);
    const double period = 60.0 / after.bpm;

    EXPECT_GT(after.next_beat_sec, now - 1e-9);
    EXPECT_LT(after.next_beat_sec, now + period + 1e-9);
}

TEST(OctaveShift, ZeroIsExactlyTheConfiguredWorld) {
    // The guarantee the whole design rests on: with no press, nothing about the
    // filter differs from a filter that has never heard of this feature. Shifted
    // and not widened is what buys it -- a wider range would coarsen estimate()'s
    // 48 log-period bins and raise confidence for a cloud that had not moved.
    ParticleFilterConfig config;
    BeatParticleFilter untouched(config);
    BeatParticleFilter shifted(config);
    shifted.setOctaveShift(2);
    shifted.setOctaveShift(0);

    const double now = driveFilter(untouched, kBpm, 20.0);
    driveFilter(shifted, kBpm, 20.0);

    const BeatEstimate a = untouched.estimate(now);
    const BeatEstimate b = shifted.estimate(now);
    EXPECT_DOUBLE_EQ(a.bpm, b.bpm);
    EXPECT_DOUBLE_EQ(a.next_beat_sec, b.next_beat_sec);
    EXPECT_DOUBLE_EQ(a.confidence, b.confidence);
    EXPECT_EQ(shifted.octaveShift(), 0);
}

TEST(OctaveShift, DoesNotCompound) {
    // Always computed from the configured world. Two up then one down has to
    // land exactly where one up did, not three octaves away.
    ParticleFilterConfig config;
    BeatParticleFilter a(config);
    BeatParticleFilter b(config);
    a.setOctaveShift(1);
    b.setOctaveShift(2);
    b.setOctaveShift(-1);
    b.setOctaveShift(1);

    const double now = driveFilter(a, 2.0 * kBpm, 20.0);
    driveFilter(b, 2.0 * kBpm, 20.0);
    EXPECT_DOUBLE_EQ(a.estimate(now).bpm, b.estimate(now).bpm);
    EXPECT_EQ(a.octaveShift(), b.octaveShift());
}

TEST(OctaveShift, LetsTheCloudLiveOutsideTheConfiguredRange) {
    // 400 BPM against a configured maximum of 220. Without the shift the clamp
    // would hold every particle at 220 and the tracker would look merely slow.
    ParticleFilterConfig config;
    BeatParticleFilter filter(config);
    filter.setOctaveShift(1);
    filter.seedTempo(400.0, 0.02);
    EXPECT_NEAR(filter.estimate(0.0).bpm, 400.0, 20.0);

    BeatParticleFilter clamped(config);
    clamped.seedTempo(400.0, 0.02);
    EXPECT_NEAR(clamped.estimate(0.0).bpm, config.max_bpm, 1.0);
}

TEST(OctaveShift, DoesNothingWhilePinned) {
    // The pin already fixes the period; moving a range around it could only
    // drift off a number somebody typed.
    ParticleFilterConfig config;
    BeatParticleFilter filter(config);
    filter.pinPeriod(60.0 / kBpm);
    filter.setOctaveShift(1);
    EXPECT_NEAR(filter.estimate(1.0).bpm, kBpm, 1e-9);
}

TEST(ScalePeriod, DoesNothingWhilePinned) {
    // The pinned period is a number somebody typed, and scaling it would
    // quietly mean a tempo they did not type.
    ParticleFilterConfig config;
    BeatParticleFilter filter(config);
    driveFilter(filter, kBpm, 5.0);
    filter.pinPeriod(60.0 / kBpm);

    const double before = filter.estimate(5.0).bpm;
    filter.scalePeriod(0.5);
    EXPECT_DOUBLE_EQ(filter.estimate(5.0).bpm, before);
}

// ------------------------------------------------------------ what it refuses

TEST(OctaveOffset, RefusesBeforeThereIsAnEstimateToMove) {
    LiveTracker tracker(offsetConfig());
    EXPECT_FALSE(tracker.setOctaveOffset(1));
    EXPECT_EQ(tracker.octaveOffset(), 0);
}

TEST(OctaveOffset, RefusesInManualMode) {
    // The tempo there is already the user's, and setManualTempo is how it
    // changes. Answering here would leave two places holding one period.
    LiveTracker tracker(offsetConfig());
    driveAlternating(tracker, kBpm, 20.0);
    ASSERT_TRUE(tracker.tempoFromActivation().answered());

    tracker.setManualTempo(kBpm);
    EXPECT_FALSE(tracker.setOctaveOffset(1));
    EXPECT_EQ(tracker.octaveOffset(), 0);
}

TEST(OctaveOffset, TheConfiguredRangeDoesNotOverruleThePress) {
    // The behaviour this test used to assert -- refuse anything outside
    // min_bpm..max_bpm -- was measured on RWC and refused 57.8% of a simulated
    // listener's presses, 342 of them x2, because x2 is unavailable above 110.
    // The range is a belief about what tempo music is likely to be, and
    // pinPeriod already records that such a belief does not overrule a person.
    // So it moves with the press.
    //
    // 100 doubled twice is 400, which is well outside the configured 40..220
    // and must now be taken.
    LiveTracker tracker(offsetConfig());
    double now = driveAlternating(tracker, kBpm, 20.0);
    ASSERT_TRUE(tracker.tempoFromActivation().answered());

    ASSERT_TRUE(tracker.setOctaveOffset(1));
    EXPECT_TRUE(tracker.setOctaveOffset(2));
    EXPECT_EQ(tracker.octaveOffset(), 2);

    now = driveAlternating(tracker, kBpm, 8.0, now + 1.0 / kFps);
    const double measured = tracker.tempoFromActivation().bpm;
    // And it reaches the anchor rather than piling against a boundary.
    EXPECT_NEAR(tracker.anchoredTempo(), 4.0 * measured, 1e-9);
    EXPECT_GT(tracker.estimate(now).bpm, 2.0 * kBpm);
}

TEST(OctaveOffset, StillRefusesWhatNoFilterCouldTrack) {
    // The one refusal left, and it is physical: two beats inside a single
    // evidence window cannot be separated by any filter, so there is nothing
    // there to track. At 48 kHz the window is 2048 samples, so the floor is
    // about 703 BPM and three octaves above 100 is 800.
    LiveTracker tracker(offsetConfig());
    driveAlternating(tracker, kBpm, 20.0);
    ASSERT_TRUE(tracker.tempoFromActivation().answered());

    EXPECT_FALSE(tracker.setOctaveOffset(3));
    EXPECT_EQ(tracker.octaveOffset(), 0);
}

TEST(OctaveOffset, ADownwardPressIsNotBoundedAtAll) {
    // There is no physical floor on how slow a pulse can be -- only a belief,
    // and beliefs do not overrule a person. 100 halved twice is 25, below the
    // configured 40.
    LiveTracker tracker(offsetConfig());
    double now = driveAlternating(tracker, kBpm, 20.0);
    ASSERT_TRUE(tracker.setOctaveOffset(-2));

    now = driveAlternating(tracker, kBpm, 8.0, now + 1.0 / kFps);
    const double measured = tracker.tempoFromActivation().bpm;
    EXPECT_NEAR(tracker.anchoredTempo(), 0.25 * measured, 1e-9);
}

TEST(OctaveOffset, SettingWhatIsAlreadySetIsAcceptedAndChangesNothing) {
    LiveTracker tracker(offsetConfig());
    EXPECT_TRUE(tracker.setOctaveOffset(0));
    EXPECT_EQ(tracker.octaveOffset(), 0);
}

// ---------------------------------------------------- what a press actually does

TEST(OctaveOffset, OutranksTheEstimatorAndKeepsDoingSo) {
    // The claim the control rests on. The estimator goes on saying 100 for as
    // long as the music says 100; the anchor is written from it fifty times a
    // second; and the press has to survive all of that.
    LiveTracker tracker(offsetConfig());
    double now = driveAlternating(tracker, kBpm, 20.0);
    ASSERT_TRUE(tracker.tempoFromActivation().answered());
    ASSERT_NEAR(tracker.heldOctaveBpm(), kBpm, 8.0);

    ASSERT_TRUE(tracker.setOctaveOffset(1));

    // Ten more seconds of the same music, which is about ten refreshes of the
    // estimate and five hundred anchor writes.
    now = driveAlternating(tracker, kBpm, 10.0, now + 1.0 / kFps);

    // The estimator is untouched: the press was never a claim about the pulse.
    const double measured = tracker.tempoFromActivation().bpm;
    EXPECT_NEAR(measured, kBpm, 8.0);
    // The anchor is exactly the user's octave of it, still, after all of that.
    EXPECT_NEAR(tracker.anchoredTempo(), 2.0 * measured, 1e-9);
    EXPECT_NEAR(tracker.heldOctaveBpm(), 2.0 * measured, 1e-9);
    // And the filter went with it.
    EXPECT_NEAR(tracker.estimate(now).bpm, 2.0 * kBpm, 20.0);
}

TEST(OctaveOffset, HalvingWorksTheSameWayRound) {
    LiveTracker tracker(offsetConfig());
    double now = driveAlternating(tracker, kBpm, 20.0);
    ASSERT_TRUE(tracker.setOctaveOffset(-1));
    now = driveAlternating(tracker, kBpm, 10.0, now + 1.0 / kFps);

    EXPECT_NEAR(tracker.tempoFromActivation().bpm, kBpm, 8.0);
    EXPECT_NEAR(tracker.heldOctaveBpm(), 0.5 * kBpm, 8.0);
    EXPECT_NEAR(tracker.estimate(now).bpm, 0.5 * kBpm, 10.0);
}

TEST(OctaveOffset, RidesTheEstimatorRatherThanRememberingATempo) {
    // A press is a claim about the multiple, not about the tempo. A band
    // drifting 100 -> 108 under a ÷2 must be followed to 54, not frozen at 50.
    LiveTracker tracker(offsetConfig());
    double now = driveAlternating(tracker, kBpm, 20.0);
    ASSERT_TRUE(tracker.setOctaveOffset(-1));

    now = driveAlternating(tracker, 108.0, 20.0, now + 1.0 / kFps);
    ASSERT_NEAR(tracker.tempoFromActivation().bpm, 108.0, 8.0);
    EXPECT_NEAR(tracker.heldOctaveBpm(), 54.0, 5.0);
}

TEST(OctaveOffset, TheWeakMarginFreezeDoesNotApplyItTwice) {
    // The branch that must *not* call withUserOctave, and the one a later
    // reader is most likely to "fix". It anchors toward the hold, which already
    // carries the offset, so applying it again would put the anchor an octave
    // past what was asked for.
    //
    // Asserted on the anchor and not on the beats, deliberately. A doubled
    // offset here lands at a quarter of the estimator's tempo, which for most
    // material is below the filter's range: the cloud would pile against the
    // clamp and merely look slow, and a tolerance loose enough to describe a
    // drifting cloud is loose enough to accept the bug.
    LiveConfig config = offsetConfig(/*freeze=*/true, /*margin=*/0.4);
    // The hold expires four seconds after the last confident anchor, and the
    // estimator's window is longer than that — so with the shipped timeout the
    // margin cannot collapse before the hold is already gone, and every frame
    // lands in the *next* branch down instead. That branch applies the offset
    // and produces the same anchor as a correct freeze, which is exactly why an
    // earlier version of this test passed with the bug deliberately inserted.
    config.anchor_freeze_timeout_sec = 60.0;
    LiveTracker tracker(config);

    // Metrically unambiguous first, so a hold is taken at all.
    double now = driveAlternating(tracker, kBpm, 20.0);
    ASSERT_GE(tracker.tempoFromActivation().octave_margin, 0.4)
        << "the alternating drive is meant to clear the gate; if this fails "
           "the test's material is wrong, not the tracker";
    ASSERT_TRUE(tracker.setOctaveOffset(-1));
    const double held = tracker.heldOctaveBpm();
    ASSERT_GT(held, 0.0);

    // Then ambiguous, which is what puts every subsequent frame in the freeze
    // branch: the estimator still answers, and its octave is a coin toss.
    now = driveUniform(tracker, 2.0 * kBpm, 12.0, now + 1.0 / kFps);
    const auto measured = tracker.tempoFromActivation();
    ASSERT_TRUE(measured.answered());
    ASSERT_LT(measured.octave_margin, 0.4)
        << "the uniform drive is meant to collapse the margin";

    // The witness that this is the freeze branch and not the one below it: that
    // one zeroes the hold before anchoring, this one leaves it alone.
    ASSERT_DOUBLE_EQ(tracker.heldOctaveBpm(), held);

    // Exactly what the branch is specified to write: the estimator's own tempo
    // moved to the octave nearest the hold. Not that halved again.
    const double expected = octaveNearest(measured.bpm, held);
    EXPECT_NEAR(tracker.anchoredTempo(), expected, 1e-9);
}

TEST(OctaveOffset, MovesTheHoldWithThePress) {
    // Without this the hold outvotes the press on every weak-margin frame until
    // the next confident one — and on material where the octave is genuinely in
    // doubt, that is most of them.
    LiveTracker tracker(offsetConfig(/*freeze=*/true));
    driveAlternating(tracker, kBpm, 20.0);
    const double before = tracker.heldOctaveBpm();
    ASSERT_GT(before, 0.0);

    ASSERT_TRUE(tracker.setOctaveOffset(-1));
    EXPECT_NEAR(tracker.heldOctaveBpm(), 0.5 * before, 1e-9);
}

TEST(OctaveOffset, AppliesWithTheFreezeOff) {
    // The other branch that writes an anchor from the estimator directly. Same
    // press, different path through submit.
    LiveTracker tracker(offsetConfig(/*freeze=*/false));
    double now = driveAlternating(tracker, kBpm, 20.0);
    ASSERT_TRUE(tracker.setOctaveOffset(-1));
    now = driveAlternating(tracker, kBpm, 10.0, now + 1.0 / kFps);

    EXPECT_NEAR(tracker.estimate(now).bpm, 0.5 * kBpm, 10.0);
}

TEST(OctaveOffset, SurvivesReset) {
    // A reset forgets audio, not the user — the rule reset() already applies to
    // a pinned manual tempo. It is also the safer way round: a shell resets
    // after a capture discontinuity, and a dropout that undid somebody's press
    // would look like the tracker changing its mind on its own.
    LiveTracker tracker(offsetConfig());
    driveAlternating(tracker, kBpm, 20.0);
    ASSERT_TRUE(tracker.setOctaveOffset(-1));

    tracker.reset();
    EXPECT_EQ(tracker.octaveOffset(), -1);
    // The hold is a conclusion about audio and goes with it.
    EXPECT_DOUBLE_EQ(tracker.heldOctaveBpm(), 0.0);

    // And it re-applies to whatever the next confident anchor turns out to be,
    // which is the point of storing an octave rather than a tempo.
    const double now = driveAlternating(tracker, kBpm, 20.0);
    EXPECT_NEAR(tracker.heldOctaveBpm(), 0.5 * kBpm, 8.0);
    EXPECT_NEAR(tracker.estimate(now).bpm, 0.5 * kBpm, 10.0);
}

TEST(OctaveOffset, DoesNotMoveThePhase) {
    // The press changes how often the click comes, never where the beats are.
    // A beat the tracker was about to play must still be a beat afterwards.
    LiveTracker tracker(offsetConfig());
    const double now = driveAlternating(tracker, kBpm, 20.0);

    const double before = tracker.estimate(now).next_beat_sec;
    ASSERT_TRUE(tracker.setOctaveOffset(1));
    const double after = tracker.estimate(now).next_beat_sec;

    // Same grid at twice the rate: either the same beat, or the new one that
    // falls between now and it.
    const double half = 0.5 * (60.0 / kBpm);
    const double steps = (before - after) / half;
    EXPECT_LT(std::fabs(steps - std::round(steps)), 0.2);
    EXPECT_GE(after, now - 1e-9);
}
