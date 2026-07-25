#include "decode/decoder.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>

#include "dr_flac.h"
#include "dr_mp3.h"
#include "dr_wav.h"

namespace tiktak::decode {
namespace {

// Read in chunks rather than one frame at a time: dr_libs has real per-call
// overhead, and the analysis reads the whole file straight through.
constexpr std::size_t kChunkFrames = 4096;

bool startsWith(const unsigned char* data, std::size_t bytes, const char* magic,
                std::size_t at = 0) {
    const std::size_t length = std::strlen(magic);
    if (bytes < at + length) return false;
    return std::memcmp(data + at, magic, length) == 0;
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
    if (startsWith(bytes_in, bytes, "ID3")) return Format::Mp3;
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

    switch (format) {
        case Format::Wav:
            if (!drwav_init_memory(&impl.wav, data, bytes, nullptr)) return nullptr;
            impl.initialised = true;
            decoder->info_.sample_rate = impl.wav.sampleRate;
            decoder->info_.channels = impl.wav.channels;
            decoder->info_.frames = impl.wav.totalPCMFrameCount;
            break;

        case Format::Flac:
            impl.flac = drflac_open_memory(data, bytes, nullptr);
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

    decoder->interleaved_.resize(kChunkFrames * decoder->info_.channels);
    return decoder;
}

std::unique_ptr<Decoder> Decoder::open(const char* path) {
    if (path == nullptr) return nullptr;

    std::FILE* file = std::fopen(path, "rb");
    if (file == nullptr) return nullptr;

    std::fseek(file, 0, SEEK_END);
    const long size = std::ftell(file);
    std::fseek(file, 0, SEEK_SET);
    if (size <= 0) {
        std::fclose(file);
        return nullptr;
    }

    std::vector<unsigned char> bytes(static_cast<std::size_t>(size));
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
