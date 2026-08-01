#include "device.hpp"

#include <atomic>
#include <cstring>
#include <new>

#include "miniaudio.h"

namespace tiktak::desktop {
namespace {

// The clock the whole design hangs on: stream time is counted in samples
// submitted, not read from a wall clock. A wall clock read in the callback
// carries the scheduling delay of the callback itself — tens of microseconds,
// different every time — and that jitter would land straight on the beat.
struct Shared {
    AudioCallback callback = nullptr;
    void* user = nullptr;
    double sample_rate = 0.0;
    std::atomic<unsigned long long> frames_played{0};
};

void dataCallback(ma_device* device, void* output, const void* input, ma_uint32 frames) {
    auto* shared = static_cast<Shared*>(device->pUserData);
    if (!shared || !shared->callback) return;

    auto* out = static_cast<float*>(output);
    const auto submitted = shared->frames_played.load(std::memory_order_relaxed);
    const double stream_time = static_cast<double>(submitted) / shared->sample_rate;

    // The device hands over whatever was in the buffer. The core mixes rather
    // than fills — it has to, so a click can play over a backing track — so
    // clearing is this side's job.
    if (out) std::memset(out, 0, static_cast<std::size_t>(frames) * sizeof(float));

    shared->callback(shared->user, stream_time, static_cast<const float*>(input), out,
                     static_cast<std::size_t>(frames));

    shared->frames_played.store(submitted + frames, std::memory_order_relaxed);
}

std::string deviceName(const ma_device_info& info) { return std::string(info.name); }

}  // namespace

struct Device::Impl {
    ma_context context{};
    ma_device device{};
    Shared shared;
    bool context_ready = false;
    bool device_ready = false;
};

Device::~Device() {
    stop();
    delete impl_;
}

DeviceList listDevices() {
    DeviceList list;

    ma_context context;
    if (ma_context_init(nullptr, 0, nullptr, &context) != MA_SUCCESS) {
        list.error = "no audio backend available";
        return list;
    }
    list.backend = ma_get_backend_name(context.backend);

    ma_device_info* playback = nullptr;
    ma_uint32 playback_count = 0;
    ma_device_info* capture = nullptr;
    ma_uint32 capture_count = 0;

    if (ma_context_get_devices(&context, &playback, &playback_count, &capture,
                               &capture_count) != MA_SUCCESS) {
        ma_context_uninit(&context);
        list.error = "could not enumerate devices";
        return list;
    }

    for (ma_uint32 i = 0; i < playback_count; ++i) {
        list.playback.push_back({deviceName(playback[i]), playback[i].isDefault != 0});
    }
    for (ma_uint32 i = 0; i < capture_count; ++i) {
        list.capture.push_back({deviceName(capture[i]), capture[i].isDefault != 0});
    }

    ma_context_uninit(&context);
    list.ok = true;
    return list;
}

bool Device::start(AudioCallback callback, void* user, double sample_rate, bool capture,
                   const std::string& preferred_name) {
    stop();

    impl_ = new (std::nothrow) Impl();
    if (!impl_) {
        error_ = "out of memory";
        return false;
    }

    if (ma_context_init(nullptr, 0, nullptr, &impl_->context) != MA_SUCCESS) {
        error_ = "no audio backend available";
        return false;
    }
    impl_->context_ready = true;
    backend_ = ma_get_backend_name(impl_->context.backend);

    // Resolving a name to an id has to happen against the enumerated list;
    // miniaudio has no lookup by name.
    ma_device_id chosen_id{};
    bool have_id = false;
    if (!preferred_name.empty()) {
        ma_device_info* infos = nullptr;
        ma_uint32 count = 0;
        ma_device_info* capture_infos = nullptr;
        ma_uint32 capture_count = 0;
        if (ma_context_get_devices(&impl_->context, &infos, &count, &capture_infos,
                                   &capture_count) == MA_SUCCESS) {
            for (ma_uint32 i = 0; i < count; ++i) {
                if (preferred_name == infos[i].name) {
                    chosen_id = infos[i].id;
                    have_id = true;
                    break;
                }
            }
        }
        if (!have_id) {
            error_ = "no playback device named '" + preferred_name + "'";
            return false;
        }
    }

    ma_device_config config =
        ma_device_config_init(capture ? ma_device_type_duplex : ma_device_type_playback);
    config.playback.format = ma_format_f32;
    config.playback.channels = 1;
    if (have_id) config.playback.pDeviceID = &chosen_id;
    if (capture) {
        config.capture.format = ma_format_f32;
        config.capture.channels = 1;
    }
    config.sampleRate = static_cast<ma_uint32>(sample_rate > 0.0 ? sample_rate : 0.0);
    config.dataCallback = dataCallback;
    config.pUserData = &impl_->shared;

    if (ma_device_init(&impl_->context, &config, &impl_->device) != MA_SUCCESS) {
        error_ = "could not open the audio device";
        return false;
    }
    impl_->device_ready = true;

    sample_rate_ = impl_->device.sampleRate;
    name_ = impl_->device.playback.name;
    period_frames_ = impl_->device.playback.internalPeriodSizeInFrames;

    // What the driver claims. Reported, never assumed — see the header.
    const double periods = impl_->device.playback.internalPeriods > 0
                               ? impl_->device.playback.internalPeriods
                               : 1.0;
    output_latency_ = sample_rate_ > 0.0
                          ? static_cast<double>(period_frames_) * periods / sample_rate_
                          : 0.0;
    if (capture) {
        input_latency_ =
            sample_rate_ > 0.0
                ? static_cast<double>(impl_->device.capture.internalPeriodSizeInFrames) *
                      (impl_->device.capture.internalPeriods > 0
                           ? impl_->device.capture.internalPeriods
                           : 1) /
                      sample_rate_
                : 0.0;
    }

    impl_->shared.callback = callback;
    impl_->shared.user = user;
    impl_->shared.sample_rate = sample_rate_;
    impl_->shared.frames_played.store(0, std::memory_order_relaxed);

    if (ma_device_start(&impl_->device) != MA_SUCCESS) {
        error_ = "could not start the audio device";
        return false;
    }
    return true;
}

void Device::stop() {
    if (!impl_) return;
    if (impl_->device_ready) {
        ma_device_uninit(&impl_->device);
        impl_->device_ready = false;
    }
    if (impl_->context_ready) {
        ma_context_uninit(&impl_->context);
        impl_->context_ready = false;
    }
    delete impl_;
    impl_ = nullptr;
}

}  // namespace tiktak::desktop
