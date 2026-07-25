#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace tiktak::decode {

enum class Format {
    Unknown,
    Wav,
    Flac,
    Mp3,
};

const char* formatName(Format format);

struct AudioInfo {
    double sample_rate = 0.0;
    unsigned channels = 0;
    // Frames per channel. Zero means the container did not say — some MP3
    // streams only reveal their length by being decoded to the end.
    std::uint64_t frames = 0;
    Format format = Format::Unknown;

    double durationSec() const {
        return sample_rate > 0.0 ? static_cast<double>(frames) / sample_rate : 0.0;
    }
};

// Decodes WAV, FLAC and MP3 to mono float, for the offline analysis path.
//
// Mono because everything downstream is mono: the onset function has no use for
// a stereo image, and downmixing at the source halves the work rather than
// carrying two channels through the STFT to throw one away.
//
// No resampling. The analyser is configured with whatever rate the file is at,
// which avoids both the cost and the phase distortion of a resampler — and beat
// times come out in seconds, so nothing downstream cares about the rate.
//
// The format is identified by content, not by file extension, and a file that
// matches nothing is refused rather than guessed at. MP3 in particular will
// decode almost any byte stream into noise if simply handed to the decoder, and
// silently analysing noise is worse than reporting an unsupported file.
//
// Offline component: opens files and allocates. Never call it from an audio
// callback.
class Decoder {
public:
    ~Decoder();

    Decoder(const Decoder&) = delete;
    Decoder& operator=(const Decoder&) = delete;

    // Decodes from a buffer the caller owns and keeps alive. This is the
    // primary entry point, not a convenience: on Android a picked file arrives
    // as a content:// descriptor rather than a path, on iOS as a
    // security-scoped URL, and on Windows a path with Cyrillic in it does not
    // survive the narrow-char fopen that a path-based API implies. Every shell
    // can produce bytes; not every shell can produce a path that works.
    //
    // A five-minute MP3 is a few megabytes, so holding the encoded file is
    // cheap. If that ever stops being true, dr_libs takes read/seek callbacks
    // and this grows a streaming overload without changing anything else.
    static std::unique_ptr<Decoder> openMemory(const void* data, std::size_t bytes);

    // Convenience for the desktop harness and the tests, where a plain path is
    // meaningful. Not the API the mobile shells should use — see above.
    static std::unique_ptr<Decoder> open(const char* path);

    // Both return nullptr if the input cannot be read or is not a supported
    // format.

    // Identifies the format of a buffer without decoding it. Exposed because a
    // caller often wants to reject a file before committing to reading it.
    static Format sniff(const void* data, std::size_t bytes);

    const AudioInfo& info() const { return info_; }

    // Reads up to `frames` mono frames. Returns how many were read; a short
    // read means the end of the stream.
    std::size_t readMono(float* out, std::size_t frames);

    // Seeks to a frame index. False if the format or stream does not support
    // it. The analysis path reads straight through, so this exists for playback
    // and for re-analysing a section.
    bool seek(std::uint64_t frame);

private:
    struct Impl;

    Decoder();

    std::unique_ptr<Impl> impl_;
    AudioInfo info_;
    std::vector<float> interleaved_;         // scratch for the downmix
    std::vector<unsigned char> owned_;       // file bytes, when we read them ourselves
};

}  // namespace tiktak::decode
