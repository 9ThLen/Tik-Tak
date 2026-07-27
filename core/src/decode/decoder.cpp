#include "decode/decoder.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <new>

#include "dr_flac.h"
#include "dr_mp3.h"
#include "dr_wav.h"

namespace tiktak::decode {
namespace {

// Read in chunks rather than one frame at a time: dr_libs has real per-call
// overhead, and the analysis reads the whole file straight through.
constexpr std::size_t kChunkFrames = 4096;

// The largest file this decoder will load. It exists because the class holds
// the whole encoded file in memory by design, so any size it accepts is a size
// it must be able to hold: an hour of CD-quality WAV is about six hundred
// megabytes, and past a gigabyte the whole-file approach is the wrong one
// rather than a tight fit.
//
// It also catches sizes that are not really sizes. ftell on a directory
// returns LONG_MAX on glibc rather than failing, and turning that into an
// allocation is what made open() abort instead of returning null.
constexpr long kMaxFileBytes = 1L << 30;

// Real recordings are mono through 7.1, and ambisonic material reaches a few
// dozen. Past that the count came from a corrupt header, not a microphone, and
// it would otherwise size the interleaved buffer: 4096 frames times the 65535
// a WAV header can claim is a gigabyte of scratch space asked for on the
// strength of two bad bytes. Everything is downmixed to mono anyway.
constexpr unsigned kMaxChannels = 64;

bool startsWith(const unsigned char* data, std::size_t bytes, const char* magic,
                std::size_t at = 0) {
    const std::size_t length = std::strlen(magic);
    if (bytes < at + length) return false;
    return std::memcmp(data + at, magic, length) == 0;
}

// Total length of a leading ID3v2 tag, or 0 if there is not a valid one.
//
// The header is ten bytes: "ID3", two version bytes, flags, and a four-byte
// size that is *syncsafe* — seven bits per byte, so the high bit is always
// clear and the pattern can never be mistaken for an MP3 sync word. Checking
// those four high bits is what separates a real tag from three bytes that
// happen to spell ID3. Version 2.4 may repeat the header as a footer, which
// bit 4 of the flags announces.
std::size_t id3TagLength(const unsigned char* data, std::size_t bytes) {
    if (!startsWith(data, bytes, "ID3") || bytes < 10) return 0;
    if (data[3] == 0xFF || data[4] == 0xFF) return 0;   // reserved versions

    std::size_t size = 0;
    for (std::size_t i = 6; i < 10; ++i) {
        if ((data[i] & 0x80) != 0) return 0;            // not syncsafe
        size = (size << 7) | data[i];
    }
    const std::size_t footer = (data[5] & 0x10) != 0 ? 10u : 0u;
    return 10 + size + footer;
}

}  // namespace

const char* formatName(Format format) {
    switch (format) {
        case Format::Wav: return "wav";
        case Format::Flac: return "flac";
        case Format::Mp3: return "mp3";
        case Format::Unknown: break;
    }
    return "unknown";
}

struct Decoder::Impl {
    Format format = Format::Unknown;
    bool initialised = false;

    drwav wav{};
    drmp3 mp3{};
    drflac* flac = nullptr;

    ~Impl() {
        if (!initialised) return;
        switch (format) {
            case Format::Wav: drwav_uninit(&wav); break;
            case Format::Mp3: drmp3_uninit(&mp3); break;
            case Format::Flac: drflac_close(flac); break;
            case Format::Unknown: break;
        }
    }
};

Decoder::Decoder() : impl_(std::make_unique<Impl>()) {}

Decoder::~Decoder() = default;

Format Decoder::sniff(const void* data, std::size_t bytes) {
    if (data == nullptr) return Format::Unknown;
    const auto* bytes_in = static_cast<const unsigned char*>(data);

    // RIFF....WAVE
    if (startsWith(bytes_in, bytes, "RIFF") && startsWith(bytes_in, bytes, "WAVE", 8)) {
        return Format::Wav;
    }
    if (startsWith(bytes_in, bytes, "fLaC")) return Format::Flac;

    // MP3 has no header of its own: either an ID3 tag, or a raw frame whose
    // sync word is eleven set bits. Checking the layer and bitrate nibbles too,
    // because eleven set bits alone appear in plenty of binary files and the
    // MP3 decoder will gladly turn any of them into noise.
    // A tagger will put an ID3 tag on anything it can write metadata to, FLAC
    // included, so an ID3 tag on its own does not mean MP3. Look underneath it
    // for a real magic number first. Skipping this is precisely the failure the
    // paragraph above warns about, reached from the other side: the MP3 decoder
    // gets handed FLAC data, and either refuses a perfectly good file or turns
    // it into noise.
    if (startsWith(bytes_in, bytes, "ID3")) {
        if (const std::size_t skip = id3TagLength(bytes_in, bytes)) {
            if (startsWith(bytes_in, bytes, "fLaC", skip)) return Format::Flac;
            if (startsWith(bytes_in, bytes, "RIFF", skip) &&
                startsWith(bytes_in, bytes, "WAVE", skip + 8)) {
                return Format::Wav;
            }
        }
        return Format::Mp3;
    }
    if (bytes >= 4 && bytes_in[0] == 0xFF && (bytes_in[1] & 0xE0) == 0xE0) {
        const unsigned layer = (bytes_in[1] >> 1) & 0x03;
        const unsigned bitrate = (bytes_in[2] >> 4) & 0x0F;
        const unsigned rate = (bytes_in[2] >> 2) & 0x03;
        if (layer != 0 && bitrate != 0x0F && rate != 0x03) return Format::Mp3;
    }

    return Format::Unknown;
}

std::unique_ptr<Decoder> Decoder::openMemory(const void* data, std::size_t bytes) {
    const Format format = sniff(data, bytes);
    if (format == Format::Unknown) return nullptr;

    std::unique_ptr<Decoder> decoder{new Decoder()};
    Impl& impl = *decoder->impl_;
    impl.format = format;

    // The WAV and FLAC decoders expect their own magic at byte zero and will
    // refuse a buffer that opens with an ID3 tag, so the tag is stepped over
    // here. The MP3 decoder handles tags itself and is left alone. The pointer
    // aims into the caller's buffer, which the path overload keeps alive by
    // moving it into the decoder afterwards — moving a vector does not move
    // its heap storage, so the offset stays valid.
    std::size_t offset = 0;
    if (format != Format::Mp3) {
        const std::size_t skip =
            id3TagLength(static_cast<const unsigned char*>(data), bytes);
        if (skip > 0 && skip < bytes) offset = skip;
    }
    const void* payload = static_cast<const unsigned char*>(data) + offset;
    const std::size_t payload_bytes = bytes - offset;

    switch (format) {
        case Format::Wav:
            if (!drwav_init_memory(&impl.wav, payload, payload_bytes, nullptr)) return nullptr;
            impl.initialised = true;
            decoder->info_.sample_rate = impl.wav.sampleRate;
            decoder->info_.channels = impl.wav.channels;
            decoder->info_.frames = impl.wav.totalPCMFrameCount;
            break;

        case Format::Flac:
            impl.flac = drflac_open_memory(payload, payload_bytes, nullptr);
            if (impl.flac == nullptr) return nullptr;
            impl.initialised = true;
            decoder->info_.sample_rate = impl.flac->sampleRate;
            decoder->info_.channels = impl.flac->channels;
            decoder->info_.frames = impl.flac->totalPCMFrameCount;
            break;

        case Format::Mp3:
            if (!drmp3_init_memory(&impl.mp3, data, bytes, nullptr)) return nullptr;
            impl.initialised = true;
            decoder->info_.sample_rate = impl.mp3.sampleRate;
            decoder->info_.channels = impl.mp3.channels;
            // Counting MP3 frames means decoding the whole stream. Worth it:
            // the analyser wants to size its buffers up front, and a progress
            // bar with no total is worse than the one-off cost.
            decoder->info_.frames = drmp3_get_pcm_frame_count(&impl.mp3);
            drmp3_seek_to_pcm_frame(&impl.mp3, 0);
            break;

        case Format::Unknown:
            return nullptr;
    }

    decoder->info_.format = format;
    if (decoder->info_.channels == 0 || decoder->info_.sample_rate <= 0.0) return nullptr;
    if (static_cast<unsigned>(decoder->info_.channels) > kMaxChannels) return nullptr;

    try {
        decoder->interleaved_.resize(kChunkFrames * decoder->info_.channels);
    } catch (const std::bad_alloc&) {
        // Bounded above, so reaching here means the machine is out of memory
        // rather than the header being absurd. Either way the contract is the
        // same: a decoder that cannot be built comes back as null.
        return nullptr;
    }
    return decoder;
}

std::unique_ptr<Decoder> Decoder::open(const char* path) {
    if (path == nullptr) return nullptr;

    std::FILE* file = std::fopen(path, "rb");
    if (file == nullptr) return nullptr;

    std::fseek(file, 0, SEEK_END);
    const long size = std::ftell(file);
    std::fseek(file, 0, SEEK_SET);
    // A directory opens successfully on glibc and reports LONG_MAX bytes, so
    // the upper bound is doing real work here and not only guarding against
    // enormous files. Refusing is the whole contract of this function: a path
    // it cannot turn into audio comes back as null.
    if (size <= 0 || size > kMaxFileBytes) {
        std::fclose(file);
        return nullptr;
    }

    std::vector<unsigned char> bytes;
    try {
        bytes.resize(static_cast<std::size_t>(size));
    } catch (const std::bad_alloc&) {
        std::fclose(file);
        return nullptr;
    }
    const std::size_t read = std::fread(bytes.data(), 1, bytes.size(), file);
    std::fclose(file);
    if (read != bytes.size()) return nullptr;

    std::unique_ptr<Decoder> decoder = openMemory(bytes.data(), bytes.size());
    if (decoder == nullptr) return nullptr;

    // The decoders read from the buffer lazily, so it has to outlive them.
    // Moving it into the decoder is what makes the path overload safe to use
    // the same way as the memory one.
    decoder->owned_ = std::move(bytes);
    return decoder;
}

std::size_t Decoder::readMono(float* out, std::size_t frames) {
    if (out == nullptr || frames == 0 || !impl_->initialised) return 0;

    const std::size_t channels = info_.channels;
    std::size_t done = 0;

    while (done < frames) {
        const std::size_t want = std::min(kChunkFrames, frames - done);
        std::size_t got = 0;

        switch (impl_->format) {
            case Format::Wav:
                got = drwav_read_pcm_frames_f32(&impl_->wav, want, interleaved_.data());
                break;
            case Format::Flac:
                got = drflac_read_pcm_frames_f32(impl_->flac, want, interleaved_.data());
                break;
            case Format::Mp3:
                got = drmp3_read_pcm_frames_f32(&impl_->mp3, want, interleaved_.data());
                break;
            case Format::Unknown:
                return done;
        }

        if (got == 0) break;

        if (channels == 1) {
            std::copy(interleaved_.begin(),
                      interleaved_.begin() + static_cast<std::ptrdiff_t>(got), out + done);
        } else {
            // Mean, not sum: a sum would clip on anything already near full
            // scale, and the onset function cares about the shape of the
            // attack, which clipping flattens.
            const float scale = 1.0f / static_cast<float>(channels);
            for (std::size_t frame = 0; frame < got; ++frame) {
                float sum = 0.0f;
                for (std::size_t c = 0; c < channels; ++c) {
                    sum += interleaved_[frame * channels + c];
                }
                out[done + frame] = sum * scale;
            }
        }

        done += got;
        if (got < want) break;   // end of stream
    }

    return done;
}

bool Decoder::seek(std::uint64_t frame) {
    if (!impl_->initialised) return false;

    switch (impl_->format) {
        case Format::Wav:
            return drwav_seek_to_pcm_frame(&impl_->wav, frame) != 0;
        case Format::Flac:
            return drflac_seek_to_pcm_frame(impl_->flac, frame) != 0;
        case Format::Mp3:
            return drmp3_seek_to_pcm_frame(&impl_->mp3, frame) != 0;
        case Format::Unknown:
            break;
    }
    return false;
}

}  // namespace tiktak::decode
