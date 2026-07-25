/*
 * tik-tak core — public C API.
 *
 * The boundary between the portable analysis core and the platform shells is a
 * flat C API on purpose: Swift calls it directly through a module map (no
 * Objective-C++ shim needed) and JNI binds to C signatures naturally.
 * See docs/adr/0001-portable-cpp-core.md.
 *
 * Threading: every object is single-threaded. Create it on whichever thread will
 * use it and do not share it. tt_odf_process is real-time safe — it allocates
 * nothing and takes no locks, so it is safe to call from an audio callback.
 */
#ifndef TIKTAK_H
#define TIKTAK_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#  define TT_API __declspec(dllexport)
#else
#  define TT_API __attribute__((visibility("default")))
#endif

/* ---------------------------------------------------------------- version -- */

#define TT_VERSION_MAJOR 0
#define TT_VERSION_MINOR 1
#define TT_VERSION_PATCH 0

/* Returns a static string such as "0.1.0". Never NULL. */
TT_API const char* tt_version(void);

/* ----------------------------------------------------------------- status -- */

typedef enum tt_status {
    TT_OK = 0,
    TT_ERR_INVALID_ARG = 1,
    TT_ERR_OUT_OF_MEMORY = 2,
    TT_ERR_UNSUPPORTED = 3
} tt_status;

/* Human-readable description of a status code. Never NULL. */
TT_API const char* tt_status_string(tt_status status);

/* -------------------------------------------------------------------- ODF -- */

/*
 * Onset detection function: the shared front-end of every analysis mode.
 *
 * Audio in, a stream of onset-strength frames out. Three bands are emitted in
 * parallel because they carry different rhythmic information:
 *
 *   full  — everything, the general-purpose onset signal
 *   low   — below `low_band_hz`; kick and bass, the best downbeat cue
 *   high  — above `high_band_hz`; hi-hat, the best subdivision cue
 */

typedef struct tt_odf tt_odf;

typedef struct tt_odf_config {
    double sample_rate;    /* Hz. Must be > 0.                                  */
    size_t frame_size;     /* STFT window, samples. Power of two. 0 -> 2048.    */
    size_t hop_size;       /* Hop, samples. Must divide into frame_size. 0 -> 512. */
    size_t mel_bands;      /* Mel filters. 0 -> 81.                             */
    double mel_min_hz;     /* Low edge of the filterbank. 0 -> 27.5 (A0).       */
    double mel_max_hz;     /* High edge. 0 -> min(16000, nyquist).              */
    double low_band_hz;    /* Upper edge of the low band. 0 -> 200.             */
    double high_band_hz;   /* Lower edge of the high band. 0 -> 4000.           */
    int    whitening;      /* Adaptive whitening on (1) or off (0).             */
    double whitening_tau;  /* Whitening time constant, seconds. 0 -> 1.0.       */
    /* Whitening floor as a fraction of the loudest recent band, in (0, 1).
       Bands below it count as noise instead of normalising up to full scale.
       0 -> 1e-3 (-60 dB). */
    double whitening_floor_rel;
    /* Whitening exponent in [0, 1]: bands are divided by peak^strength.
       1.0 fully equalises loudness but erases the balance between bands, so
       low/high stop being comparable. 0.0 disables normalisation.
       0 -> 0.5. Pass a negative value to mean exactly 0.0. */
    double whitening_strength;
} tt_odf_config;

/* Fills `cfg` with the defaults documented above. */
TT_API void tt_odf_config_defaults(tt_odf_config* cfg, double sample_rate);

/* Onset strengths are mean rise per mel band, so the three are on a common
   scale and can be compared with one another directly. */
typedef struct tt_odf_frame {
    double time_sec;  /* Centre of the analysis window, seconds from stream start. */
    float  full;      /* Onset strength, full band. >= 0.                          */
    float  low;       /* Onset strength below low_band_hz. >= 0.                   */
    float  high;      /* Onset strength above high_band_hz. >= 0.                  */
} tt_odf_frame;

/*
 * Creates an ODF analyser. Returns NULL on invalid config or allocation failure;
 * pass `status` to tell the two apart, or NULL if you do not care.
 * Every buffer is allocated here so that tt_odf_process never allocates.
 */
TT_API tt_odf* tt_odf_create(const tt_odf_config* cfg, tt_status* status);

TT_API void tt_odf_destroy(tt_odf* odf);

/*
 * Feeds `n` mono samples and writes completed frames to `out`.
 *
 * Real-time safe. Returns the number of frames written, or 0 if more input is
 * needed. Frames are produced once every hop_size samples, after an initial
 * frame_size samples of fill.
 *
 * If `cap` is too small for the frames this input produced, the surplus is
 * dropped and `dropped` (when non-NULL) is set to how many. Sizing `cap` at
 * n / hop_size + 2 always suffices.
 */
TT_API size_t tt_odf_process(tt_odf* odf,
                             const float* samples, size_t n,
                             tt_odf_frame* out, size_t cap,
                             size_t* dropped);

/* Number of frames a call with `n` samples can produce right now. */
TT_API size_t tt_odf_frames_available(const tt_odf* odf, size_t n);

/*
 * Clears all history: buffered samples, the previous spectrum and the whitening
 * state. The stream clock restarts at zero. Use when the input jumps — a new
 * file, a seek, or the microphone restarting.
 */
TT_API void tt_odf_reset(tt_odf* odf);

/* Latency between a sample arriving and the frame covering it being emitted. */
TT_API double tt_odf_latency_sec(const tt_odf* odf);

/* --------------------------------------------------------------- offline -- */

/*
 * Whole-file beat analysis: audio in, a beat grid out.
 *
 * This is the accurate path. It sees the entire piece before deciding anything,
 * so it picks the globally best beat sequence rather than committing frame by
 * frame the way the microphone path must. Use it for imported tracks.
 *
 * Audio is fed in blocks and reduced to onset frames as it arrives, so a long
 * file never has to be held in memory at once. Nothing here is real-time safe —
 * it allocates. Drive it from a file-reading thread, never an audio callback.
 *
 * Usage: create, feed repeatedly, finish, then read the results.
 */

typedef struct tt_offline tt_offline;

typedef struct tt_offline_config {
    tt_odf_config odf;

    /* Tempo search. */
    double min_bpm;              /* 0 -> 40                                    */
    double max_bpm;              /* 0 -> 220                                   */
    double prior_centre_bpm;     /* 0 -> 120                                   */
    double prior_width_octaves;  /* Prior width in octaves. 0 -> 0.7           */
    int    tempo_grid_size;      /* Candidate tempi. 0 -> 512                  */
    /* Comb scoring over metrical multiples. 0 -> 1, which disables it: it
       measured worse than plain autocorrelation on every metric available.
       Kept configurable because that evidence is synthetic. */
    int    comb_harmonics;
    double comb_weight_decay;    /* 0 -> 1.0                                   */

    /* Beat tracking. */
    /* Weight of the tempo-consistency penalty. Higher keeps the grid rigid
       through weak passages; lower follows a rubato performer. 0 -> 100. */
    double tightness;
    /* Drop beats at the start and end that sit on no real onset. The tracker
       extends its grid into silence to stay regular, and clicking through the
       silence before the music starts is exactly what the app must not do.
       Non-zero enables; pass a negative value to disable. 0 -> enabled. */
    int    trim;
    /* Fix the tempo instead of estimating it, for manual mode. <= 0 estimates.
       The tempo is measured either way — see tt_offline_estimated_bpm. */
    double bpm_hint;
} tt_offline_config;

TT_API void tt_offline_config_defaults(tt_offline_config* cfg, double sample_rate);

TT_API tt_offline* tt_offline_create(const tt_offline_config* cfg, tt_status* status);
TT_API void tt_offline_destroy(tt_offline* offline);

/* Appends `n` mono samples. Any block size. */
TT_API tt_status tt_offline_feed(tt_offline* offline, const float* samples, size_t n);

/*
 * Runs tempo estimation and beat tracking over everything fed so far. Call
 * before reading any result. Repeatable: feeding more and calling again extends
 * the analysis rather than restarting it.
 */
TT_API tt_status tt_offline_finish(tt_offline* offline);

/* Clears the collected audio and results, ready for another file. */
TT_API void tt_offline_reset(tt_offline* offline);

/* The tempo the beats were tracked at: the hint if one was given, else the
   estimate. 0 before the first tt_offline_finish. */
TT_API double tt_offline_bpm(const tt_offline* offline);

/* What the audio itself says, measured even when a hint overrode it. Lets
   manual mode warn that the user's 120 sounds like 90. */
TT_API double tt_offline_estimated_bpm(const tt_offline* offline);

/* 0..1: how strongly the onset function repeats at that tempo. 0 means no
   periodicity was found, which is not the same as a slow tempo. */
TT_API double tt_offline_confidence(const tt_offline* offline);

TT_API size_t tt_offline_beat_count(const tt_offline* offline);

/*
 * Copies beat times, in seconds from the start of the audio, into `out`.
 * Returns how many were written: min(capacity, beat count). The core keeps
 * ownership of its own storage, so callers across the FFI boundary never have
 * to free anything.
 */
TT_API size_t tt_offline_beats(const tt_offline* offline, double* out, size_t capacity);

typedef struct tt_tempo_candidate {
    double bpm;
    double strength;   /* 0..1, relative to the winner */
} tt_tempo_candidate;

/*
 * Alternative readings of the tempo, strongest first. When the runner-up sits
 * an octave from the winner with a similar strength, the estimate is a coin
 * toss — show the ambiguity and let the user resolve it rather than committing
 * silently. Returns how many were written.
 */
TT_API size_t tt_offline_tempo_candidates(const tt_offline* offline,
                                          tt_tempo_candidate* out, size_t capacity);

/* Onset frames collected so far — diagnostics and the parity harness. */
TT_API size_t tt_offline_frame_count(const tt_offline* offline);

/* -------------------------------------------------------------- scheduler -- */

/*
 * Turns a tempo into precisely timed events, ahead of time.
 *
 * Nothing is ever played on demand. Calling the audio device at the moment of
 * the beat inherits every scheduling delay between the decision and the
 * speaker — tens of milliseconds, different every time. Instead the grid is
 * computed in advance and each event carries the timestamp the device should
 * place it at.
 *
 * One clock domain throughout: the host's monotonic clock, in seconds. The core
 * never reads a clock itself; `now_sec` is always supplied by the caller.
 */

typedef struct tt_scheduler tt_scheduler;

typedef enum tt_channel {
    TT_CHANNEL_AUDIO = 0,
    TT_CHANNEL_HAPTIC = 1,
    TT_CHANNEL_VISUAL = 2,
    TT_CHANNEL_COUNT = 3
} tt_channel;

typedef enum tt_beat_kind {
    TT_BEAT_DOWNBEAT = 0,     /* first beat of the bar */
    TT_BEAT_BEAT = 1,         /* any other beat        */
    TT_BEAT_SUBDIVISION = 2   /* between beats         */
} tt_beat_kind;

typedef struct tt_scheduler_config {
    double bpm;               /* Must be > 0. 0 -> 120.                        */
    int    beats_per_bar;     /* 0 -> 4.                                       */
    int    subdivisions;      /* 1 = beats only, 2 = eighths. 0 -> 1.          */
    /* How far ahead events are handed out. Must comfortably exceed the host's
       polling interval, or beats fall through the gap between polls.
       0 -> 0.25. */
    double lookahead_sec;
    /* Per-channel output latency. They differ — the audio device buffer, the
       taptic engine and the next display frame are not the same delay — and
       compensating them with one number makes the vibration drift audibly
       against the click. Indexed by tt_channel. */
    double latency_sec[TT_CHANNEL_COUNT];
    int    channel_enabled[TT_CHANNEL_COUNT];
} tt_scheduler_config;

typedef struct tt_event {
    /* When to hand this to its device: the musical instant minus that
       channel's latency. Schedule against this, never against "now". */
    double time_sec;
    /* The musical instant itself, shared by every channel of the same step. */
    double beat_time_sec;
    long long step;           /* grid position, counting subdivisions */
    long long bar;
    int channel;              /* tt_channel   */
    int kind;                 /* tt_beat_kind */
    int beat_in_bar;          /* 0-based */
    int subdivision;          /* 0 on the beat itself */
} tt_event;

TT_API void tt_scheduler_config_defaults(tt_scheduler_config* cfg);

TT_API tt_scheduler* tt_scheduler_create(const tt_scheduler_config* cfg, tt_status* status);
TT_API void tt_scheduler_destroy(tt_scheduler* scheduler);

/* Starts the grid with step 0 at `now_sec`. Resets the late counter. */
TT_API void tt_scheduler_start(tt_scheduler* scheduler, double now_sec);
TT_API void tt_scheduler_stop(tt_scheduler* scheduler);
TT_API int  tt_scheduler_running(const tt_scheduler* scheduler);

/*
 * Changes tempo without moving anything already handed out. The grid is
 * re-anchored on the last emitted event, so a beat the device is already
 * holding keeps its time — re-anchoring on "now" is heard as a stumble.
 */
TT_API void tt_scheduler_set_tempo(tt_scheduler* scheduler, double bpm);

/*
 * Shifts the grid's phase so a beat lands on `beat_time_sec`, for the manual
 * mode where the tempo is fixed and only the offset must be found. Events
 * already handed out are untouched; a shift that would put the next event in
 * the past makes the grid skip forward instead.
 */
TT_API void tt_scheduler_align_to(tt_scheduler* scheduler, double beat_time_sec,
                                  double now_sec);

/*
 * Writes every event due between `now_sec` and the lookahead horizon, and
 * advances past them. Returns how many were written.
 *
 * Real-time safe. An event whose compensated time has already passed is not
 * emitted — a late click actively misleads the player, so a gap is the lesser
 * evil — and `dropped_late` (when non-NULL) counts those, so the host can widen
 * its lookahead or report the overload rather than silently limping.
 *
 * A step's enabled channels are emitted together or not at all, so a caller
 * with a small buffer never receives half a beat.
 */
TT_API size_t tt_scheduler_pull(tt_scheduler* scheduler, double now_sec,
                                tt_event* out, size_t capacity, size_t* dropped_late);

/* Host time of a grid step — for drawing the grid ahead of the events. */
TT_API double tt_scheduler_step_time(const tt_scheduler* scheduler, long long step);

/* Events dropped for lateness since the last start(). */
TT_API size_t tt_scheduler_late_count(const tt_scheduler* scheduler);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* TIKTAK_H */
