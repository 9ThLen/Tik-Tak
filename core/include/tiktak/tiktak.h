/*
 * tik-tak core — public C API.
 *
 * The boundary between the portable analysis core and the platform shells is a
 * flat C API on purpose: Swift calls it directly through a module map (no
 * Objective-C++ shim needed) and JNI binds to C signatures naturally.
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
    double prior_centre_bpm;     /* 0 -> 140                                   */
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

    /* Bar lines. Also find where each bar starts and how many beats it holds,
       so the click can accent the downbeat instead of every beat equally.
       Non-zero enables; pass a negative value to disable. 0 -> enabled.

       Turning it off is a real option and not just a debugging one: it makes
       the analyser skip the harmony front end, which is the expensive half.
       A caller that only wants a click on every beat should not pay for it. */
    int    find_downbeats;
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

/*
 * Bar lines.
 *
 * Beats per bar, or 0 when no meter was decided — the track was too short for
 * any meter to repeat, or bar lines were not asked for. Bar lines themselves
 * are a subset of the beats, copied out the same way.
 */
TT_API int tt_offline_beats_per_bar(const tt_offline* offline);
TT_API size_t tt_offline_downbeat_count(const tt_offline* offline);
TT_API size_t tt_offline_downbeats(const tt_offline* offline, double* out, size_t capacity);

/*
 * How far to trust those bar lines. All three are in the active salience
 * backend's units and answer different questions. They may be compared with
 * thresholds calibrated for that backend, but not with raw values from a
 * different scorer.
 *
 * `strength` is how much louder the chosen bar lines are than the beats around
 * them. Near zero means the audio has no bar-level pattern at all, and the
 * honest thing to show is no accent.
 *
 * `phase_margin` is how far ahead the winning bar line is of the next best
 * place to put it *within the same meter*. A strong pattern with a small phase
 * margin means the bars are clear but which beat starts them is a coin toss.
 *
 * `meter_margin` is how far ahead the winning meter is of the next best meter.
 * It has to be asked separately: every rival the phase margin considers has
 * already accepted the bar length, so a piece read in three can look completely
 * settled on that scale while four fits it nearly as well.
 *
 * Gate a UI on `tt_offline_downbeat_confident`, which requires both, rather
 * than on either margin alone — that mistake is what this API separated.
 */
TT_API double tt_offline_downbeat_strength(const tt_offline* offline);
TT_API double tt_offline_downbeat_phase_margin(const tt_offline* offline);
TT_API double tt_offline_downbeat_meter_margin(const tt_offline* offline);

/*
 * Whether the bar lines are worth accenting at all: non-zero when a pattern
 * exists and both margins clear their thresholds.
 *
 * When this is zero a player should count from the first beat and accent
 * nothing. Accenting "every fourth beat starting from the first" as a fallback
 * is not a neutral default — it is an arbitrary accent presented with the same
 * confidence as a real one, and a player following it is worse off than with a
 * plain click.
 */
TT_API int tt_offline_downbeat_confident(const tt_offline* offline);

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

/* ------------------------------------------------------------- grid cache -- */

/*
 * The beat grid cache: an analysed track serialised for reuse, so importing the
 * same file twice costs seconds once and nothing after.
 *
 * The core serialises to bytes and leaves storage to the shell, for the same
 * reason decoding accepts bytes rather than a path: every shell can persist
 * bytes, while a portable "cache directory" does not exist. The intended flow:
 *
 *   key <- tt_grid_key(file bytes)              -- before decoding anything
 *   have cached bytes under `key`?  tt_offline_grid_restore and skip analysis
 *   otherwise: decode, feed, finish, tt_offline_grid_serialize, store as `key`
 *
 * The key is the SHA-256 of the *encoded file bytes*, not the file's name or
 * its decoded samples — a renamed file still hits, a re-encoded one correctly
 * misses, and nothing needs decoding just to ask.
 */

/* Hex characters in a grid key, excluding the terminator. */
#define TT_GRID_KEY_HEX 64

/*
 * Writes the cache key for `bytes` into `out` as lowercase hex plus a
 * terminating NUL. `cap` must be at least TT_GRID_KEY_HEX + 1.
 */
TT_API tt_status tt_grid_key(const void* bytes, size_t n, char* out, size_t cap);

/* Size in bytes of the serialized grid. 0 before tt_offline_finish. */
TT_API size_t tt_offline_grid_size(const tt_offline* offline);

/*
 * Serialises the finished analysis into `out`. Returns the bytes written, or 0
 * when there is no finished result or `cap` is smaller than
 * tt_offline_grid_size — a partial blob is useless, so none is written.
 */
TT_API size_t tt_offline_grid_serialize(const tt_offline* offline, void* out, size_t cap);

/*
 * Restores a previously serialized grid into the handle, after which the
 * tt_offline_* accessors read it exactly as if the analysis had just run.
 *
 * The handle's configuration must match the one the grid was analysed under —
 * the same audio under a different config (a manual-mode bpm_hint, say) is a
 * different grid, and serving one for the other would put beats in the wrong
 * place. A mismatch, a truncated blob, or a foreign format all return
 * TT_ERR_UNSUPPORTED, and they all mean the same thing: re-analyse.
 */
TT_API tt_status tt_offline_grid_restore(tt_offline* offline, const void* bytes, size_t n);

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

/* ------------------------------------------------------------------ click -- */

/*
 * Turns scheduled beats into audio.
 *
 * The scheduler decides when, this decides what it sounds like, and they are
 * separate because the shells differ in the first and must not differ in the
 * second: a click that sounded different on iOS and on the desktop harness
 * would make every timing measurement taken on the harness unusable.
 *
 * Placement is sample-accurate — the click is written at the sample nearest its
 * time, not at the start of the buffer that contains it. The difference is up
 * to a whole buffer, grossly audible at 10 ms; the residual rounding is half a
 * sample, which is not.
 *
 * Real-time safe: everything is sized at creation and nothing allocates.
 */

typedef struct tt_click tt_click;

typedef struct tt_click_tone {
    double frequency_hz;
    /* How long the click lasts, measured as the time to fall 60 dB — the
       number you would see on a stopwatch, not a time constant. It has to stay
       well under a beat or fast tempos turn into a drone. 0 -> default. */
    double length_sec;
    double gain;              /* 0 -> default. Negative is rejected. */
} tt_click_tone;

typedef struct tt_click_config {
    double sample_rate;       /* Must be > 0. */

    /* Roughly G6 / C6 / G5 by default: a fifth apart so the downbeat reads as
       "the one" rather than as a different instrument, and the subdivision an
       octave under the beat so it stays beneath it. */
    tt_click_tone downbeat;
    tt_click_tone beat;
    tt_click_tone subdivision;

    int max_voices;           /* Overlapping clicks. 0 -> 8.  */
    int max_pending;          /* Queued clicks.      0 -> 64. */

    /* A click landing before the buffer it is handed to is nudged to the start
       of that buffer if it is late by less than this, and dropped if it is
       later. Both halves matter: a host that polls a hair late produces
       sub-millisecond lateness that is inaudible when nudged and audible as a
       hole when dropped, while a truly late click misleads a player and is
       better missing. 0 -> 0.002. */
    double late_tolerance_sec;
} tt_click_config;

TT_API void tt_click_config_defaults(tt_click_config* cfg, double sample_rate);

TT_API tt_click* tt_click_create(const tt_click_config* cfg, tt_status* status);
TT_API void tt_click_destroy(tt_click* click);

/*
 * Queues a click at `time_sec`, in the same clock domain the mix times use —
 * hand it tt_event.time_sec straight from the scheduler. `kind` is a
 * tt_beat_kind. Returns 0 if the queue is full.
 */
TT_API int tt_click_schedule(tt_click* click, double time_sec, int kind);

/*
 * Adds into `out`; it does not clear it. Mixing rather than filling is the
 * contract because the click plays over a backing track, and a fill would
 * silently erase it. `start_time_sec` is the time of the first sample of `out`.
 */
TT_API void tt_click_mix(tt_click* click, double start_time_sec, float* out, size_t frames);

/* Silences everything sounding and queued, without tearing the renderer down.
   The counters below survive, since they describe the run, not the take. */
TT_API void tt_click_reset(tt_click* click);

TT_API size_t tt_click_pending(const tt_click* click);
TT_API size_t tt_click_active_voices(const tt_click* click);
/* Clicks that arrived too late to place — see late_tolerance_sec. */
TT_API size_t tt_click_dropped_late(const tt_click* click);
/* Clicks refused because the queue was full. */
TT_API size_t tt_click_dropped_overflow(const tt_click* click);
/* Clicks cut short because every voice was busy. */
TT_API size_t tt_click_stolen(const tt_click* click);
/* Mixes whose start time did not follow on from the previous buffer: the
   device dropped or repeated one. A sounding click assumes time is continuous,
   so this is counted rather than hidden. */
TT_API size_t tt_click_discontinuities(const tt_click* click);

/* ----------------------------------------------------------------- player -- */

/*
 * Playback of an analysed track with the metronome riding its beat grid — the
 * app's main scenario: the backing track with the click exactly on the beats
 * the offline analysis found.
 *
 * A separate object rather than a mode of the metronome because the time source
 * is inverted: a metronome generates its grid from a tempo, the player follows
 * a grid the analysis already fixed. The track, not the clock, says where the
 * beats are.
 *
 * The click needs no latency compensation against the track: both leave
 * through the same device buffer, so a click written on the beat's sample
 * arrives with it whatever the output latency is. Only haptic and visual cues
 * carry latency arithmetic — see tt_player_config.latency_sec.
 *
 * tt_player_process is real-time safe. Everything else allocates or takes
 * setup decisions; call it before starting, from a normal thread.
 */

typedef struct tt_player tt_player;

typedef struct tt_player_config {
    double sample_rate;         /* Must be > 0, and the track's rate.          */
    tt_click_config click;      /* click.sample_rate 0 -> sample_rate.         */

    /* Grid beat `downbeat_offset` is a bar's first beat, and every
       beats_per_bar-th after it. Both come from the offline analysis —
       tt_offline_beats_per_bar and the first of tt_offline_downbeats — with
       the offset left settable so the user can shift which beat is "the one"
       when the analysis is unsure or simply wrong. */
    int beats_per_bar;          /* 0 -> 4                                      */
    int downbeat_offset;        /* 0-based grid index; negative rejected       */
    /* Whether bar starts are distinguished at all. Read literally:
       tt_player_config_defaults enables it; set to 0 when
       tt_offline_downbeat_confident returns 0 so every beat sounds and is
       reported alike. Bars remain available for looping and positioning. */
    int accent_downbeats;

    /* Count-in clicks before the music, at the local beat interval read off
       the grid at the entry point, with the track silent underneath. */
    int count_in_beats;         /* 0 -> none                                   */

    /* How far ahead haptic/visual cues are handed out. The click needs none —
       it renders in the same callback. 0 -> 0.25.                             */
    double cue_lookahead_sec;

    /* latency_sec[TT_CHANNEL_AUDIO] is the device's output latency, used only
       to know when a beat is *heard*; haptic and visual entries are those
       channels' own delays, compensated against that moment.                  */
    double latency_sec[TT_CHANNEL_COUNT];
    /* Read literally (no zero-means-default): audio on and cues off is what
       tt_player_config_defaults fills in.                                     */
    int    channel_enabled[TT_CHANNEL_COUNT];
} tt_player_config;

TT_API void tt_player_config_defaults(tt_player_config* cfg, double sample_rate);

TT_API tt_player* tt_player_create(const tt_player_config* cfg, tt_status* status);
TT_API void tt_player_destroy(tt_player* player);

/*
 * The decoded track, mono at the configured rate. NOT copied — the caller
 * already holds it for the analysis, and five minutes of audio is tens of
 * megabytes. The buffer must stay alive and unmoved until playback is done.
 */
TT_API tt_status tt_player_set_track(tt_player* player, const float* samples,
                                     size_t frames);

/* The analysed beat grid, seconds from the track's start, ascending. Copied —
   the caller may free its analysis once the player is loaded. */
TT_API tt_status tt_player_set_grid(tt_player* player, const double* beat_times,
                                    size_t count);

/*
 * Loops bars [start_bar, end_bar): the track jumps from the end bar's first
 * beat back to the start bar's, sample-exactly. Set before starting.
 * TT_ERR_INVALID_ARG when the bars are not in the grid or the range is empty.
 */
TT_API tt_status tt_player_set_loop(tt_player* player, long long start_bar,
                                    long long end_bar);
TT_API void tt_player_clear_loop(tt_player* player);

/*
 * Starts at `from_bar`'s first beat, count-in first if configured;
 * `stream_time_sec` is when the first sample leaves. TT_ERR_INVALID_ARG when
 * there is no track, the bar is not in the grid, or a count-in is asked of a
 * grid too short to define a beat interval.
 */
TT_API tt_status tt_player_start(tt_player* player, double stream_time_sec,
                                 long long from_bar);

/* Stops advancing; sounding clicks ring out. Silence also cuts them. */
TT_API void tt_player_stop(tt_player* player);
TT_API void tt_player_silence(tt_player* player);

TT_API int tt_player_running(const tt_player* player);

/* Current position, seconds into the track. */
TT_API double tt_player_position_sec(const tt_player* player);

/*
 * The audio callback. Mixes into `out` — does not clear it. `stream_time_sec`
 * is the time of out[0], in the same clock tt_player_start was given. Haptic
 * and visual cues are written to `cues` (pass NULL to discard); events beyond
 * `cue_capacity` are counted in stats rather than silently lost.
 */
TT_API void tt_player_process(tt_player* player, double stream_time_sec,
                              float* out, size_t frames,
                              tt_event* cues, size_t cue_capacity,
                              size_t* cue_count);

/* Counters that should stay zero on a healthy run (beats and loops count the
   run itself). Mirrors the metronome's stats discipline. */
typedef struct tt_player_stats {
    size_t beats;              /* clicks scheduled since start                 */
    size_t loops;              /* times the loop wrapped                       */
    size_t clicks_late;        /* clicks that arrived past their buffer        */
    size_t clicks_overflowed;  /* clicks refused, queue full                   */
    size_t voices_stolen;      /* clicks cut short, all voices busy            */
    size_t discontinuities;    /* buffers that did not follow the previous one */
    size_t cues_dropped;       /* cue events the caller had no room for        */
    int    clean;              /* 1 when nothing above went wrong              */
} tt_player_stats;

TT_API void tt_player_stats_get(const tt_player* player, tt_player_stats* out);

/* ------------------------------------------------------------ live input -- */

/*
 * The microphone path: captured audio in, beat predictions out.
 *
 * Predictions, not detections. A click has to be written into a buffer before
 * its beat is heard, so the question this answers is "where is the next beat",
 * which a tracker announcing beats it has already seen could never answer
 * however accurate it was.
 *
 * Two things the shell must do for this to work at all:
 *
 * - Feed capture and ask for beats in the *same* clock. Whatever a platform
 *   calls it, the number passed to tt_live_process and tt_live_take_beat has to
 *   come from one timeline, or the beats come out shifted by the difference.
 *
 * - Declare its own click through tt_live_gate_click. A metronome listening
 *   through a microphone hears itself, and a click is the most onset-like
 *   sound there is; ungated, the tracker locks onto its own output, reports
 *   full confidence and stops following the room.
 *
 * tt_live_process, tt_live_take_beat, tt_live_estimate and tt_live_gate_click
 * are real-time safe. Create, seed and reset are not.
 */

typedef struct tt_live tt_live;

typedef struct tt_live_config {
    double sample_rate;        /* Must be > 0, the capture rate.               */

    /* Tempo range and prior, the same belief the offline estimator applies. */
    double min_bpm;            /* 0 -> 40                                      */
    double max_bpm;            /* 0 -> 220                                     */
    double prior_centre_bpm;   /* 0 -> 150                                     */

    /* Particles carried. More is steadier and costs linearly. 0 -> 512. */
    int particles;

    /* How long our own click blinds the microphone, around the moment it is
       heard. 0 -> 5 ms before, 50 ms after.                                   */
    double gate_before_sec;
    double gate_after_sec;

    /* Confidence to start handing out beats at, and to stop at. Between them
       the tracker coasts at the last tempo it was sure of, which is what a
       musician does when the band drops out for a bar. 0 -> 0.35 / 0.15.      */
    double lock_confidence;
    double release_confidence;
} tt_live_config;

TT_API void tt_live_config_defaults(tt_live_config* cfg, double sample_rate);

TT_API tt_live* tt_live_create(const tt_live_config* cfg, tt_status* status);
TT_API void tt_live_destroy(tt_live* live);

/*
 * Captured mono audio. `stream_time_sec` is the time of samples[0]; a value
 * that does not follow on from the last call is treated as the device having
 * dropped or repeated a buffer, and counted.
 */
TT_API void tt_live_process(tt_live* live, double stream_time_sec,
                            const float* samples, size_t frames);

/*
 * When our own click will reach the microphone: the moment it is *heard*,
 * output latency and room delay already added by the caller. The core cannot
 * work it out — only the shell knows what the round trip measured.
 */
TT_API void tt_live_gate_click(tt_live* live, double heard_time_sec);

typedef struct tt_live_estimate {
    double bpm;
    double next_beat_sec;         /* prediction, in the caller's clock         */
    double confidence;            /* 0..1: the cloud agrees AND onsets land    */
    double tempo_spread_octaves;  /* how undecided the period itself is        */
} tt_live_estimate;

TT_API void tt_live_estimate_get(const tt_live* live, double now_sec,
                                 tt_live_estimate* out);

/*
 * The next beat to play, handed out once, when it comes within
 * `lookahead_sec` of `now_sec`. Returns 1 and writes `beat_sec` when there is
 * one, 0 otherwise — including whenever confidence is too low to claim a beat
 * at all, which is the tracker's way of saying it cannot hear the music.
 *
 * A beat, once handed out, is never revised: by then the click is in a buffer
 * on its way to the device, and moving it would be a stutter rather than a
 * correction. Refinements land on the beat after it.
 */
TT_API int tt_live_take_beat(tt_live* live, double now_sec, double lookahead_sec,
                             double* beat_sec);

/*
 * Concentrates the cloud on a known tempo: an offline analysis of the same
 * song, or a tempo the user typed. `spread_octaves` 0 -> 0.05.
 */
TT_API void tt_live_seed_tempo(tt_live* live, double bpm, double spread_octaves);

/*
 * The ×2 / ÷2 control: which multiple of the pulse the user says is the beat,
 * in whole octaves. +1 doubles, -1 halves, 0 leaves it to the tracker.
 * Returns 1 when the press was taken, 0 when it was refused.
 *
 * A press is a claim about the *multiple*, not about the tempo, and is stored
 * as one — so a band drifting from 128 to 132 is still followed, at 66. It
 * outranks the tracker's own octave for as long as it is set: the tracker
 * re-decides the level continuously, and a control that only nudged the
 * estimate would be overruled within about a second.
 *
 * Refused, changing nothing, in three cases: in manual mode, where the tempo is
 * already the user's and tt_live_set_manual_tempo is the way to change it;
 * before the tracker has an estimate to move; and when doubling or halving
 * would leave the configured BPM range. The last is refused rather than clamped
 * so that a press either means what it says or visibly does nothing.
 *
 * Survives tt_live_reset, which forgets audio and not the user.
 */
TT_API int tt_live_set_octave_offset(tt_live* live, int octaves);
TT_API int tt_live_octave_offset(const tt_live* live);

/*
 * Manual mode: the tempo is the user's and the room is asked only where the
 * beat falls. 0 goes back to tracking the tempo too.
 *
 * A different promise from auto mode, in two ways that a shell has to show:
 *
 * - Nothing comes out until the room has been heard. The user sets a tempo and
 *   starts; the click waits, catches the first phrase and falls in on it, which
 *   is what makes a count-in of its own unnecessary. tt_live_waiting is that
 *   state, and what a UI shows as "listening...".
 *
 * - Once it has fallen in, it does not stop. In auto mode a room that goes
 *   quiet has taken the tempo with it; here the tempo was never the room's, so
 *   the click holds it through a silent bar, a solo or a cough, indefinitely.
 *
 * The room may nudge the click by up to 2% of a beat at a time, so it follows a
 * player drifting within about 2% of the tempo set and free-runs against
 * anything further off. Finding the phase is a far smaller problem than finding
 * a tempo, which is why this mode works on material the auto tracker cannot
 * follow at all.
 *
 * The tempo is taken as given, including outside min_bpm..max_bpm: that range
 * is a belief about what music is likely to be, and it does not overrule a
 * number somebody typed.
 */
TT_API void tt_live_set_manual_tempo(tt_live* live, double bpm);
TT_API double tt_live_manual_tempo(const tt_live* live);

/* Manual mode, still listening for something to fall in with. 0 otherwise. */
TT_API int tt_live_waiting(const tt_live* live);

/*
 * How concentrated the room's onsets are at one phase, 0..1 — the meter behind
 * that "listening...". Manual mode only; 0 in auto mode.
 */
TT_API double tt_live_sync_strength(const tt_live* live);

/* Forgets the audio and the clock — a new session. The manual tempo survives:
   it was typed, not heard. */
TT_API void tt_live_reset(tt_live* live);

typedef struct tt_live_stats {
    size_t frames;           /* onset frames produced                          */
    size_t gated;            /* frames withheld because our own click was in   */
    size_t beats;            /* beats handed out                               */
    size_t beats_late;       /* beats predicted into the past, skipped         */
    size_t discontinuities;  /* capture buffers that did not follow the last   */
    size_t resamples;        /* filter resampling steps, diagnostics           */
    size_t reanchors;        /* gaps in the stream the cloud was re-anchored on*/
} tt_live_stats;

TT_API void tt_live_stats_get(const tt_live* live, tt_live_stats* out);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* TIKTAK_H */
