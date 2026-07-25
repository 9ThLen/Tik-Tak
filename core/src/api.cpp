#include "tiktak/tiktak.h"

#include <algorithm>
#include <new>

#include "dsp/odf.hpp"
#include "schedule/scheduler.hpp"

namespace {

constexpr const char* kVersion = "0.1.0";

// Zero means "use the default" for every numeric field, so a caller can memset
// the config, set sample_rate, and get a sane analyser.
tiktak::dsp::OdfConfig resolve(const tt_odf_config& in) {
    tiktak::dsp::OdfConfig out;
    const double nyquist = in.sample_rate * 0.5;

    out.sampleRate = in.sample_rate;
    out.frameSize = in.frame_size ? in.frame_size : 2048;
    out.hopSize = in.hop_size ? in.hop_size : 512;
    out.melBands = in.mel_bands ? in.mel_bands : 81;
    out.melMinHz = in.mel_min_hz > 0.0 ? in.mel_min_hz : 27.5;
    out.melMaxHz = in.mel_max_hz > 0.0 ? in.mel_max_hz : std::min(16000.0, nyquist);
    out.lowBandHz = in.low_band_hz > 0.0 ? in.low_band_hz : 200.0;
    out.highBandHz = in.high_band_hz > 0.0 ? in.high_band_hz : 4000.0;
    out.whitening = in.whitening != 0;
    out.whiteningTau = in.whitening_tau > 0.0 ? in.whitening_tau : 1.0;
    out.whiteningFloorRel =
        in.whitening_floor_rel > 0.0 ? in.whitening_floor_rel : 1e-3;
    // Negative means "explicitly zero", since zero itself means "default".
    out.whiteningStrength = in.whitening_strength < 0.0    ? 0.0
                            : in.whitening_strength > 0.0  ? in.whitening_strength
                                                           : 0.5;

    if (out.melMaxHz > nyquist) out.melMaxHz = nyquist;
    return out;
}

}  // namespace

struct tt_odf {
    explicit tt_odf(const tiktak::dsp::OdfConfig& cfg) : impl(cfg) {}
    tiktak::dsp::Odf impl;
};

const char* tt_version(void) { return kVersion; }

const char* tt_status_string(tt_status status) {
    switch (status) {
        case TT_OK: return "ok";
        case TT_ERR_INVALID_ARG: return "invalid argument";
        case TT_ERR_OUT_OF_MEMORY: return "out of memory";
        case TT_ERR_UNSUPPORTED: return "unsupported";
    }
    return "unknown status";
}

void tt_odf_config_defaults(tt_odf_config* cfg, double sample_rate) {
    if (!cfg) return;
    cfg->sample_rate = sample_rate;
    cfg->frame_size = 2048;
    cfg->hop_size = 512;
    cfg->mel_bands = 81;
    cfg->mel_min_hz = 27.5;
    cfg->mel_max_hz = std::min(16000.0, sample_rate * 0.5);
    cfg->low_band_hz = 200.0;
    cfg->high_band_hz = 4000.0;
    cfg->whitening = 1;
    cfg->whitening_tau = 1.0;
    cfg->whitening_floor_rel = 1e-3;
    cfg->whitening_strength = 0.5;
}

tt_odf* tt_odf_create(const tt_odf_config* cfg, tt_status* status) {
    const auto fail = [status](tt_status code) -> tt_odf* {
        if (status) *status = code;
        return nullptr;
    };

    if (!cfg) return fail(TT_ERR_INVALID_ARG);

    const tiktak::dsp::OdfConfig resolved = resolve(*cfg);
    if (!resolved.valid()) return fail(TT_ERR_INVALID_ARG);

    tt_odf* handle = new (std::nothrow) tt_odf(resolved);
    if (!handle) return fail(TT_ERR_OUT_OF_MEMORY);

    if (status) *status = TT_OK;
    return handle;
}

void tt_odf_destroy(tt_odf* odf) { delete odf; }

size_t tt_odf_process(tt_odf* odf,
                      const float* samples, size_t n,
                      tt_odf_frame* out, size_t cap,
                      size_t* dropped) {
    if (dropped) *dropped = 0;
    if (!odf || (n > 0 && !samples)) return 0;

    size_t written = 0;
    size_t lost = 0;

    odf->impl.process(samples, n, [&](const tiktak::dsp::OdfFrame& frame) {
        if (!out || written >= cap) {
            ++lost;
            return;
        }
        out[written].time_sec = frame.timeSec;
        out[written].full = frame.full;
        out[written].low = frame.low;
        out[written].high = frame.high;
        ++written;
    });

    if (dropped) *dropped = lost;
    return written;
}

size_t tt_odf_frames_available(const tt_odf* odf, size_t n) {
    if (!odf) return 0;
    return odf->impl.framesAvailable(n);
}

void tt_odf_reset(tt_odf* odf) {
    if (!odf) return;
    odf->impl.reset();
}

double tt_odf_latency_sec(const tt_odf* odf) {
    if (!odf) return 0.0;
    return odf->impl.latencySec();
}

/* -------------------------------------------------------------- scheduler -- */

namespace {

tiktak::schedule::SchedulerConfig resolve(const tt_scheduler_config& in) {
    tiktak::schedule::SchedulerConfig out;
    out.bpm = in.bpm > 0.0 ? in.bpm : 120.0;
    out.beats_per_bar = in.beats_per_bar > 0 ? in.beats_per_bar : 4;
    out.subdivisions = in.subdivisions > 0 ? in.subdivisions : 1;
    out.lookahead_sec = in.lookahead_sec > 0.0 ? in.lookahead_sec : 0.25;

    for (int i = 0; i < TT_CHANNEL_COUNT; ++i) {
        out.latency_sec[static_cast<std::size_t>(i)] =
            in.latency_sec[i] > 0.0 ? in.latency_sec[i] : 0.0;
        out.channel_enabled[static_cast<std::size_t>(i)] = in.channel_enabled[i] != 0;
    }
    return out;
}

tt_event to_c(const tiktak::schedule::Event& event) {
    tt_event out;
    out.time_sec = event.time_sec;
    out.beat_time_sec = event.beat_time_sec;
    out.step = static_cast<long long>(event.step);
    out.bar = static_cast<long long>(event.bar);
    out.channel = static_cast<int>(event.channel);
    out.kind = static_cast<int>(event.kind);
    out.beat_in_bar = event.beat_in_bar;
    out.subdivision = event.subdivision;
    return out;
}

}  // namespace

struct tt_scheduler {
    explicit tt_scheduler(const tiktak::schedule::SchedulerConfig& cfg) : impl(cfg) {}
    tiktak::schedule::Scheduler impl;
};

void tt_scheduler_config_defaults(tt_scheduler_config* cfg) {
    if (!cfg) return;
    cfg->bpm = 120.0;
    cfg->beats_per_bar = 4;
    cfg->subdivisions = 1;
    cfg->lookahead_sec = 0.25;
    for (int i = 0; i < TT_CHANNEL_COUNT; ++i) {
        cfg->latency_sec[i] = 0.0;
        cfg->channel_enabled[i] = 1;
    }
}

tt_scheduler* tt_scheduler_create(const tt_scheduler_config* cfg, tt_status* status) {
    const auto fail = [status](tt_status code) -> tt_scheduler* {
        if (status) *status = code;
        return nullptr;
    };

    if (!cfg) return fail(TT_ERR_INVALID_ARG);

    const tiktak::schedule::SchedulerConfig resolved = resolve(*cfg);
    if (!resolved.valid()) return fail(TT_ERR_INVALID_ARG);

    tt_scheduler* handle = new (std::nothrow) tt_scheduler(resolved);
    if (!handle) return fail(TT_ERR_OUT_OF_MEMORY);

    if (status) *status = TT_OK;
    return handle;
}

void tt_scheduler_destroy(tt_scheduler* scheduler) { delete scheduler; }

void tt_scheduler_start(tt_scheduler* scheduler, double now_sec) {
    if (scheduler) scheduler->impl.start(now_sec);
}

void tt_scheduler_stop(tt_scheduler* scheduler) {
    if (scheduler) scheduler->impl.stop();
}

int tt_scheduler_running(const tt_scheduler* scheduler) {
    return scheduler && scheduler->impl.running() ? 1 : 0;
}

void tt_scheduler_set_tempo(tt_scheduler* scheduler, double bpm) {
    if (scheduler) scheduler->impl.set_tempo(bpm);
}

void tt_scheduler_align_to(tt_scheduler* scheduler, double beat_time_sec, double now_sec) {
    if (scheduler) scheduler->impl.align_to(beat_time_sec, now_sec);
}

size_t tt_scheduler_pull(tt_scheduler* scheduler, double now_sec,
                         tt_event* out, size_t capacity, size_t* dropped_late) {
    if (dropped_late) *dropped_late = 0;
    if (!scheduler || !out || capacity == 0) return 0;

    // Bounded stack staging: the C++ side writes its own Event type, and this
    // converts in place without allocating, so the call stays real-time safe.
    constexpr std::size_t kBatch = 32;
    tiktak::schedule::Event staging[kBatch];

    std::size_t written = 0;
    std::size_t late_total = 0;

    while (written < capacity) {
        const std::size_t want = std::min(kBatch, capacity - written);
        std::size_t late = 0;
        const std::size_t got = scheduler->impl.pull(now_sec, staging, want, &late);

        late_total += late;
        for (std::size_t i = 0; i < got; ++i) out[written + i] = to_c(staging[i]);
        written += got;

        if (got < want) break;
    }

    if (dropped_late) *dropped_late = late_total;
    return written;
}

double tt_scheduler_step_time(const tt_scheduler* scheduler, long long step) {
    return scheduler ? scheduler->impl.step_time(step) : 0.0;
}

size_t tt_scheduler_late_count(const tt_scheduler* scheduler) {
    return scheduler ? scheduler->impl.late_count() : 0;
}
