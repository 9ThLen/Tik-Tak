/*
 * tik-tak decoding — public C API.
 *
 * Deliberately a separate header and a separate library from tiktak.h. The
 * analysis core has no third-party dependencies and cross-compiles anywhere
 * unchanged; decoding cannot, because it needs codec implementations. A
 * platform that would rather decode with AVAssetReader or MediaCodec links
 * tiktak_core alone and never sees this file.
 *
 * Threading: a decoder is single-threaded, like everything else in the core.
 * Nothing here is real-time safe — it reads files and allocates. Drive it from
 * a file-reading thread.
 */
#ifndef TIKTAK_DECODE_H
#define TIKTAK_DECODE_H

#include <stddef.h>

#include "tiktak/tiktak.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum tt_format {
    TT_FORMAT_UNKNOWN = 0,
    TT_FORMAT_WAV = 1,
    TT_FORMAT_FLAC = 2,
    TT_FORMAT_MP3 = 3
} tt_format;

/* Human-readable format name, e.g. "mp3". Never NULL. */
TT_API const char* tt_format_name(tt_format format);

/*
 * Identifies a format from the first bytes of a file without decoding it.
 * 64 bytes is plenty. Returns TT_FORMAT_UNKNOWN for anything unsupported —
 * which is the answer, not a failure to try harder: an MP3 decoder handed an
 * arbitrary byte stream produces noise rather than an error, and silently
 * analysing noise is worse than refusing the file.
 */
TT_API tt_format tt_sniff_format(const void* data, size_t bytes);

typedef struct tt_decoder tt_decoder;

typedef struct tt_audio_info {
    double sample_rate;
    unsigned channels;      /* of the source; decoding always yields mono */
    unsigned long long frames;   /* per channel */
    double duration_sec;
    tt_format format;
} tt_audio_info;

/*
 * Opens a decoder over encoded bytes the caller owns and keeps alive until
 * tt_decoder_close.
 *
 * This is the primary entry point rather than a convenience. A picked file
 * arrives as a content:// descriptor on Android and a security-scoped URL on
 * iOS, and on Windows a path containing non-ASCII characters does not survive
 * the narrow-char fopen a path-based API implies. Every shell can produce
 * bytes; not every shell can produce a path that works.
 *
 * Returns NULL if the data is not a supported format.
 */
TT_API tt_decoder* tt_decoder_open_memory(const void* data, size_t bytes, tt_status* status);

/*
 * Convenience for hosts where a plain path is meaningful — the desktop harness
 * and the tests. Reads the file into memory. Not what the mobile shells should
 * use; see above.
 */
TT_API tt_decoder* tt_decoder_open_file(const char* path, tt_status* status);

TT_API void tt_decoder_close(tt_decoder* decoder);

TT_API tt_audio_info tt_decoder_info(const tt_decoder* decoder);

/*
 * Reads up to `frames` mono frames into `out`. Returns how many were read; a
 * short read means the end of the stream.
 *
 * Always mono: everything downstream is. The onset function has no use for a
 * stereo image, and downmixing at the source halves the work instead of
 * carrying two channels through the STFT to discard one. Channels are averaged,
 * not summed, so material already near full scale does not clip.
 *
 * No resampling either. Configure the analyser with tt_audio_info.sample_rate
 * and beat times still come out in seconds.
 */
TT_API size_t tt_decoder_read(tt_decoder* decoder, float* out, size_t frames);

/* Seeks to a frame index. Non-zero on success. */
TT_API int tt_decoder_seek(tt_decoder* decoder, unsigned long long frame);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* TIKTAK_DECODE_H */
