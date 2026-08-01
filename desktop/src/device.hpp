#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace tiktak::desktop {

struct DeviceInfo {
    std::string name;
    bool is_default = false;
};

struct DeviceList {
    std::vector<DeviceInfo> playback;
    std::vector<DeviceInfo> capture;
    bool ok = false;
    std::string error;
    std::string backend;
};

DeviceList listDevices();

// What the harness hands the audio thread. `frames` mono samples are due at
// `stream_time_sec` — the moment they are submitted to the device, not the
// moment they are heard, which is the clock the core's latency compensation is
// expressed in.
//
// `input` is the captured signal for the same block, or nullptr when the device
// is playback-only. Called on the audio thread: nothing in here may allocate.
using AudioCallback = void (*)(void* user, double stream_time_sec,
                               const float* input, float* output, std::size_t frames);

// A running mono float32 device. Playback, or duplex when `capture` is asked
// for, in which case the same callback receives both.
//
// Kept this thin on purpose. Everything the harness exists to measure lives in
// the core; this file is the part that cannot be portable, and the less of it
// there is the less there is to get wrong differently from the phone.
class Device {
public:
    Device() = default;
    ~Device();

    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

    // `sample_rate` of 0 takes whatever the device prefers, which is the right
    // default: forcing a rate makes the driver resample behind our back, and a
    // resampler between the click and the speaker is exactly what a timing
    // harness must not have.
    bool start(AudioCallback callback, void* user, double sample_rate, bool capture,
               const std::string& preferred_name = {});
    void stop();

    double sample_rate() const { return sample_rate_; }
    std::size_t period_frames() const { return period_frames_; }
    const std::string& name() const { return name_; }
    const std::string& backend() const { return backend_; }
    const std::string& error() const { return error_; }

    // Latency the device reports, in seconds. It is the driver's own figure and
    // not to be trusted as the truth — `tiktak measure` exists because the
    // truth has to be measured — but it is the right starting point.
    double reported_output_latency() const { return output_latency_; }
    double reported_input_latency() const { return input_latency_; }

private:
    struct Impl;
    Impl* impl_ = nullptr;

    double sample_rate_ = 0.0;
    std::size_t period_frames_ = 0;
    double output_latency_ = 0.0;
    double input_latency_ = 0.0;
    std::string name_;
    std::string backend_;
    std::string error_;
};

}  // namespace tiktak::desktop
