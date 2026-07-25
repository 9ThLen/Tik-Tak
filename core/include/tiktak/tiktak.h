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

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* TIKTAK_H */
