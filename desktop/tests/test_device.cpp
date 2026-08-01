// The device layer used to be described here as "thin, and verified by running
// it". That was wrong, and these tests exist because of how it was wrong: three
// defects sat in it at once, and every one of them was invisible to running it
// on this machine.
//
//   * `--device` in capture mode looked the name up among the *playback*
//     devices. The two lists share no names, so naming a microphone always
//     missed — and the message it produced was "no playback device named
//     <your microphone>", which reads as a typo rather than a bug.
//   * the rate was decided before the device was opened, so a tracker built for
//     48 kHz could be handed a 44.1 kHz stream. Every machine here runs at
//     48 kHz, so nothing showed; phones commonly do not.
//   * beats were pushed into a `std::vector` from the audio callback while the
//     main thread read its size.
//
// None of that is caught by a passing run, which is the argument for the file.
// What can be tested without a sound card is tested unconditionally; what needs
// a device skips rather than fails, because CI has none.

#include "device.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <string>

namespace {

using tiktak::desktop::Device;
using tiktak::desktop::DeviceList;

bool listed(const std::vector<tiktak::desktop::DeviceInfo>& devices,
            const std::string& name) {
    return std::any_of(devices.begin(), devices.end(),
                       [&](const tiktak::desktop::DeviceInfo& d) { return d.name == name; });
}

void nothing(void*, double, const float*, float*, std::size_t) {}

// A name nothing will ever be called, so a match means the lookup matched
// something it should not have.
constexpr const char* kAbsent = "tiktak::no such device exists";

TEST(DeviceNames, AnUnknownCaptureNameIsRefusedAsACaptureName) {
    Device device;
    if (device.open(nothing, nullptr, 0.0, true, kAbsent)) {
        FAIL() << "opened a device by a name nothing has";
    }
    // The wording is the test. "No playback device named <microphone>" sent the
    // reader to look for a typo in a list the name was never going to be in.
    EXPECT_NE(device.error().find("capture"), std::string::npos) << device.error();
}

TEST(DeviceNames, AnUnknownPlaybackNameIsStillRefusedAsAPlaybackName) {
    Device device;
    if (device.open(nothing, nullptr, 0.0, false, kAbsent)) {
        FAIL() << "opened a device by a name nothing has";
    }
    EXPECT_NE(device.error().find("playback"), std::string::npos) << device.error();
}

// The defect itself, rather than its symptom: ask for capture by the name of a
// speaker. This used to succeed — the name was found in the playback list, the
// id was handed to the playback side, and the *default* microphone opened. The
// bench then reported the device the user had not chosen, under the name they
// had.
TEST(DeviceNames, ASpeakerIsNotAValidMicrophone) {
    const DeviceList devices = tiktak::desktop::listDevices();
    if (!devices.ok || devices.playback.empty()) GTEST_SKIP() << "no audio devices here";

    const auto speaker = std::find_if(
        devices.playback.begin(), devices.playback.end(),
        [&](const tiktak::desktop::DeviceInfo& d) { return !listed(devices.capture, d.name); });
    if (speaker == devices.playback.end()) {
        GTEST_SKIP() << "every playback device here is also a capture device";
    }

    Device device;
    EXPECT_FALSE(device.open(nothing, nullptr, 0.0, true, speaker->name))
        << "opened '" << speaker->name << "' as a microphone";
}

// Why `open` and `begin` are separate at all. Anything built to match the
// stream — a tracker's window, a click — has to be built after the rate is
// known, and the rate is only known once the device has settled on a format.
TEST(DeviceRate, TheRealRateIsKnownBeforeTheStreamRuns) {
    Device device;
    if (!device.open(nothing, nullptr, 0.0, false)) {
        GTEST_SKIP() << "no playable device here: " << device.error();
    }
    EXPECT_GT(device.sample_rate(), 0.0);
    EXPECT_FALSE(device.name().empty());
    device.stop();
}

TEST(DeviceRate, BeginningWithoutOpeningIsRefusedRatherThanCrashing) {
    Device device;
    EXPECT_FALSE(device.begin());
    EXPECT_FALSE(device.error().empty());
}

}  // namespace
