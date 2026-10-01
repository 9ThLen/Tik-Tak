#include "tracking/octave_hold.hpp"

#include <gtest/gtest.h>

using tiktak::tracking::OctaveHold;

TEST(OctaveHold, OffPassesEverythingThrough) {
    OctaveHold hold;
    EXPECT_FALSE(hold.enabled());
    EXPECT_DOUBLE_EQ(hold.resolve(0.0, 120.0), 120.0);
    EXPECT_DOUBLE_EQ(hold.resolve(1.0, 240.0), 240.0);
}

TEST(OctaveHold, FollowsFreelyInsideTheOctave) {
    // A band drifting 128 to 132 is never a proposal.
    OctaveHold hold(20.0);
    EXPECT_DOUBLE_EQ(hold.resolve(0.0, 128.0), 128.0);
    EXPECT_DOUBLE_EQ(hold.resolve(1.0, 130.0), 130.0);
    EXPECT_DOUBLE_EQ(hold.resolve(2.0, 132.0), 132.0);
    EXPECT_DOUBLE_EQ(hold.heldBpm(), 132.0);
}

TEST(OctaveHold, ABriefFlipIsHeldAndTheTempoInsideItKept) {
    OctaveHold hold(20.0);
    hold.resolve(0.0, 120.0);
    // Eight seconds at double: held at the old level, but the tempo within the
    // octave is still the estimator's (242 -> 121, not 120).
    for (int t = 1; t <= 8; ++t) {
        EXPECT_DOUBLE_EQ(hold.resolve(static_cast<double>(t), 242.0), 121.0) << t;
    }
    EXPECT_DOUBLE_EQ(hold.resolve(9.0, 121.0), 121.0);
    EXPECT_DOUBLE_EQ(hold.heldBpm(), 121.0);
}

TEST(OctaveHold, APersistentProposalIsAcceptedAfterTheHold) {
    OctaveHold hold(20.0);
    hold.resolve(0.0, 120.0);
    EXPECT_DOUBLE_EQ(hold.resolve(1.0, 60.0), 120.0);
    EXPECT_DOUBLE_EQ(hold.resolve(20.9, 60.0), 120.0);
    EXPECT_DOUBLE_EQ(hold.resolve(21.0, 60.0), 60.0) << "twenty seconds of it is a change";
    EXPECT_DOUBLE_EQ(hold.heldBpm(), 60.0);
    EXPECT_DOUBLE_EQ(hold.resolve(22.0, 61.0), 61.0);
}

TEST(OctaveHold, ReturningToTheOctaveRestartsTheClock) {
    // The measured research seam's behaviour, kept on purpose: a flickering
    // proposal never accumulates its seconds.
    OctaveHold hold(20.0);
    hold.resolve(0.0, 120.0);
    hold.resolve(1.0, 240.0);
    hold.resolve(15.0, 240.0);
    hold.resolve(16.0, 120.0);                      // back inside
    EXPECT_DOUBLE_EQ(hold.resolve(30.0, 240.0), 120.0) << "the clock restarted at 30 s";
    EXPECT_DOUBLE_EQ(hold.resolve(50.0, 240.0), 240.0);
}

TEST(OctaveHold, AThreeTwoRelationIsNotAnOctave) {
    // 120 -> 180 is not near a power of two, so it is followed, not held: the
    // hold is about metrical level, and 3:2 is a different tempo, not a level.
    OctaveHold hold(20.0);
    hold.resolve(0.0, 120.0);
    EXPECT_DOUBLE_EQ(hold.resolve(1.0, 180.0), 180.0);
}

TEST(OctaveHold, ResetForgetsTheLevel) {
    OctaveHold hold(20.0);
    hold.resolve(0.0, 120.0);
    hold.reset();
    EXPECT_DOUBLE_EQ(hold.heldBpm(), 0.0);
    EXPECT_DOUBLE_EQ(hold.resolve(1.0, 240.0), 240.0);
}
