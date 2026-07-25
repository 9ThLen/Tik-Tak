#include "tiktak/tiktak.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <new>
#include <utility>
#include <vector>

#include "analysis/grid_cache.hpp"
#include "analysis/offline.hpp"
#include "render/click.hpp"
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

tiktak::render::ClickTone resolve(const tt_click_tone& in, const tiktak::render::ClickTone& fallback) {
    tiktak::render::ClickTone out = fallback;
    if (in.frequency_hz > 0.0) out.frequency_hz = in.frequency_hz;
    if (in.length_sec > 0.0) out.length_sec = in.length_sec;
    // Negative means "explicitly silent", since zero itself means "default" —
    // the same convention the ODF config uses for whitening_strength.
    if (in.gain > 0.0) out.gain = in.gain;
    else if (in.gain < 0.0) out.gain = 0.0;
    return out;
}

tiktak::render::ClickConfig resolve(const tt_click_config& in) {
    const tiktak::render::ClickConfig defaults;
    tiktak::render::ClickConfig out;

    out.sample_rate = in.sample_rate;
    out.downbeat = resolve(in.downbeat, defaults.downbeat);
    out.beat = resolve(in.beat, defaults.beat);
    out.subdivision = resolve(in.subdivision, defaults.subdivision);
    out.max_voices = in.max_voices > 0 ? in.max_voices : defaults.max_voices;
    out.max_pending = in.max_pending > 0 ? in.max_pending : defaults.max_pending;
    out.late_tolerance_sec = in.late_tolerance_sec > 0.0 ? in.late_tolerance_sec
                                                         : defaults.late_tolerance_sec;
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

/* --------------------------------------------------------------- offline -- */

namespace {

tiktak::analysis::OfflineConfig resolve(const tt_offline_config& in) {
    tiktak::analysis::OfflineConfig out;
    out.odf = resolve(in.odf);

    out.tempo.min_bpm = in.min_bpm > 0.0 ? in.min_bpm : 40.0;
    out.tempo.max_bpm = in.max_bpm > 0.0 ? in.max_bpm : 220.0;
    out.tempo.prior_centre_bpm = in.prior_centre_bpm > 0.0 ? in.prior_centre_bpm : 120.0;
    out.tempo.prior_width_octaves =
        in.prior_width_octaves > 0.0 ? in.prior_width_octaves : 0.7;
    out.tempo.grid_size = in.tempo_grid_size > 0 ? in.tempo_grid_size : 512;
    out.tempo.comb_harmonics = in.comb_harmonics > 0 ? in.comb_harmonics : 1;
    out.tempo.comb_weight_decay = in.comb_weight_decay > 0.0 ? in.comb_weight_decay : 1.0;

    out.tracker.tightness = in.tightness > 0.0 ? in.tightness : 100.0;
    // Negative means "explicitly off", since zero means "default".
    out.tracker.trim = in.trim >= 0;
    out.bpm_hint = in.bpm_hint;
    return out;
}

}  // namespace

struct tt_offline {
    explicit tt_offline(const tiktak::analysis::OfflineConfig& cfg) : impl(cfg) {}
    tiktak::analysis::OfflineAnalyzer impl;
    tiktak::analysis::OfflineResult result;
    bool finished = false;
};

void tt_offline_config_defaults(tt_offline_config* cfg, double sample_rate) {
    if (!cfg) return;
    tt_odf_config_defaults(&cfg->odf, sample_rate);
    cfg->min_bpm = 40.0;
    cfg->max_bpm = 220.0;
    cfg->prior_centre_bpm = 120.0;
    cfg->prior_width_octaves = 0.7;
    cfg->tempo_grid_size = 512;
    cfg->comb_harmonics = 1;
    cfg->comb_weight_decay = 1.0;
    cfg->tightness = 100.0;
    cfg->trim = 1;
    cfg->bpm_hint = 0.0;
}

tt_offline* tt_offline_create(const tt_offline_config* cfg, tt_status* status) {
    const auto fail = [status](tt_status code) -> tt_offline* {
        if (status) *status = code;
        return nullptr;
    };

    if (!cfg) return fail(TT_ERR_INVALID_ARG);

    const tiktak::analysis::OfflineConfig resolved = resolve(*cfg);
    if (!resolved.odf.valid() || !resolved.tempo.valid() || !resolved.tracker.valid()) {
        return fail(TT_ERR_INVALID_ARG);
    }

    tt_offline* handle = new (std::nothrow) tt_offline(resolved);
    if (!handle) return fail(TT_ERR_OUT_OF_MEMORY);

    if (status) *status = TT_OK;
    return handle;
}

void tt_offline_destroy(tt_offline* offline) { delete offline; }

tt_status tt_offline_feed(tt_offline* offline, const float* samples, size_t n) {
    if (!offline || (n > 0 && !samples)) return TT_ERR_INVALID_ARG;
    offline->impl.feed(samples, n);
    // More audio invalidates the previous answer rather than extending it, so
    // a caller that forgets to finish again reads nothing instead of stale
    // beats.
    offline->finished = false;
    return TT_OK;
}

tt_status tt_offline_finish(tt_offline* offline) {
    if (!offline) return TT_ERR_INVALID_ARG;
    offline->result = offline->impl.finish();
    offline->finished = true;
    return TT_OK;
}

void tt_offline_reset(tt_offline* offline) {
    if (!offline) return;
    offline->impl.reset();
    offline->result = tiktak::analysis::OfflineResult{};
    offline->finished = false;
}

double tt_offline_bpm(const tt_offline* offline) {
    return offline && offline->finished ? offline->result.bpm : 0.0;
}

double tt_offline_estimated_bpm(const tt_offline* offline) {
    return offline && offline->finished ? offline->result.estimated_bpm : 0.0;
}

double tt_offline_confidence(const tt_offline* offline) {
    return offline && offline->finished ? offline->result.tempo_confidence : 0.0;
}

size_t tt_offline_beat_count(const tt_offline* offline) {
    return offline && offline->finished ? offline->result.beats.size() : 0;
}

size_t tt_offline_beats(const tt_offline* offline, double* out, size_t capacity) {
    if (!offline || !offline->finished || !out) return 0;
    const std::size_t count = std::min(capacity, offline->result.beats.size());
    std::copy(offline->result.beats.begin(),
              offline->result.beats.begin() + static_cast<std::ptrdiff_t>(count), out);
    return count;
}

size_t tt_offline_tempo_candidates(const tt_offline* offline, tt_tempo_candidate* out,
                                   size_t capacity) {
    if (!offline || !offline->finished || !out || capacity == 0) return 0;

    // Staged through the C++ type rather than reinterpreted: the two structs
    // happen to have the same layout today, and relying on that would be a
    // silent trap the first time either gains a field.
    std::vector<tiktak::analysis::TempoCandidate> staging(capacity);
    const std::size_t count = offline->impl.tempoCandidates(staging.data(), capacity);
    for (std::size_t i = 0; i < count; ++i) {
        out[i].bpm = staging[i].bpm;
        out[i].strength = staging[i].strength;
    }
    return count;
}

size_t tt_offline_frame_count(const tt_offline* offline) {
    return offline ? offline->impl.odfValues().size() : 0;
}

/* ------------------------------------------------------------- grid cache -- */

tt_status tt_grid_key(const void* bytes, size_t n, char* out, size_t cap) {
    if (!out || (n > 0 && !bytes)) return TT_ERR_INVALID_ARG;
    if (cap < TT_GRID_KEY_HEX + 1) return TT_ERR_INVALID_ARG;
    const std::string key = tiktak::analysis::gridCacheKey(bytes, n);
    std::memcpy(out, key.c_str(), key.size() + 1);
    return TT_OK;
}

size_t tt_offline_grid_size(const tt_offline* offline) {
    if (!offline || !offline->finished) return 0;
    return tiktak::analysis::serializeGrid(offline->result, offline->impl.config()).size();
}

size_t tt_offline_grid_serialize(const tt_offline* offline, void* out, size_t cap) {
    if (!offline || !offline->finished || !out) return 0;
    const std::vector<std::uint8_t> blob =
        tiktak::analysis::serializeGrid(offline->result, offline->impl.config());
    if (cap < blob.size()) return 0;
    std::memcpy(out, blob.data(), blob.size());
    return blob.size();
}

tt_status tt_offline_grid_restore(tt_offline* offline, const void* bytes, size_t n) {
    if (!offline || !bytes) return TT_ERR_INVALID_ARG;
    tiktak::analysis::OfflineResult restored;
    if (!tiktak::analysis::deserializeGrid(static_cast<const std::uint8_t*>(bytes), n,
                                           offline->impl.config(), &restored)) {
        return TT_ERR_UNSUPPORTED;
    }
    offline->result = std::move(restored);
    offline->finished = true;
    return TT_OK;
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

/* ------------------------------------------------------------------ click -- */

struct tt_click {
    explicit tt_click(const tiktak::render::ClickConfig& cfg) : impl(cfg) {}
    tiktak::render::ClickRenderer impl;
};

void tt_click_config_defaults(tt_click_config* cfg, double sample_rate) {
    if (!cfg) return;
    const tiktak::render::ClickConfig defaults;

    const auto copy = [](tt_click_tone& out, const tiktak::render::ClickTone& in) {
        out.frequency_hz = in.frequency_hz;
        out.length_sec = in.length_sec;
        out.gain = in.gain;
    };

    cfg->sample_rate = sample_rate;
    copy(cfg->downbeat, defaults.downbeat);
    copy(cfg->beat, defaults.beat);
    copy(cfg->subdivision, defaults.subdivision);
    cfg->max_voices = defaults.max_voices;
    cfg->max_pending = defaults.max_pending;
    cfg->late_tolerance_sec = defaults.late_tolerance_sec;
}

tt_click* tt_click_create(const tt_click_config* cfg, tt_status* status) {
    const auto fail = [status](tt_status code) -> tt_click* {
        if (status) *status = code;
        return nullptr;
    };

    if (!cfg) return fail(TT_ERR_INVALID_ARG);

    const tiktak::render::ClickConfig resolved = resolve(*cfg);
    if (!resolved.valid()) return fail(TT_ERR_INVALID_ARG);

    tt_click* handle = new (std::nothrow) tt_click(resolved);
    if (!handle) return fail(TT_ERR_OUT_OF_MEMORY);

    if (status) *status = TT_OK;
    return handle;
}

void tt_click_destroy(tt_click* click) { delete click; }

int tt_click_schedule(tt_click* click, double time_sec, int kind) {
    if (!click) return 0;
    if (kind < 0 || kind > static_cast<int>(tiktak::schedule::BeatKind::Subdivision)) return 0;
    return click->impl.schedule(time_sec, static_cast<tiktak::schedule::BeatKind>(kind)) ? 1 : 0;
}

void tt_click_mix(tt_click* click, double start_time_sec, float* out, size_t frames) {
    if (!click) return;
    click->impl.mix(start_time_sec, out, frames);
}

void tt_click_reset(tt_click* click) {
    if (click) click->impl.reset();
}

size_t tt_click_pending(const tt_click* click) {
    return click ? click->impl.pending_count() : 0;
}

size_t tt_click_active_voices(const tt_click* click) {
    return click ? click->impl.active_voice_count() : 0;
}

size_t tt_click_dropped_late(const tt_click* click) {
    return click ? click->impl.dropped_late() : 0;
}

size_t tt_click_dropped_overflow(const tt_click* click) {
    return click ? click->impl.dropped_overflow() : 0;
}

size_t tt_click_stolen(const tt_click* click) {
    return click ? click->impl.stolen() : 0;
}

size_t tt_click_discontinuities(const tt_click* click) {
    return click ? click->impl.discontinuities() : 0;
}
