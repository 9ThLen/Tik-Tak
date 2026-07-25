#include "tiktak/tiktak_decode.h"

#include <memory>
#include <new>

#include "decode/decoder.hpp"

namespace {

tt_format to_c(tiktak::decode::Format format) {
    switch (format) {
        case tiktak::decode::Format::Wav: return TT_FORMAT_WAV;
        case tiktak::decode::Format::Flac: return TT_FORMAT_FLAC;
        case tiktak::decode::Format::Mp3: return TT_FORMAT_MP3;
        case tiktak::decode::Format::Unknown: break;
    }
    return TT_FORMAT_UNKNOWN;
}

}  // namespace

struct tt_decoder {
    explicit tt_decoder(std::unique_ptr<tiktak::decode::Decoder> decoder)
        : impl(std::move(decoder)) {}
    std::unique_ptr<tiktak::decode::Decoder> impl;
};

const char* tt_format_name(tt_format format) {
    switch (format) {
        case TT_FORMAT_WAV: return tiktak::decode::formatName(tiktak::decode::Format::Wav);
        case TT_FORMAT_FLAC: return tiktak::decode::formatName(tiktak::decode::Format::Flac);
        case TT_FORMAT_MP3: return tiktak::decode::formatName(tiktak::decode::Format::Mp3);
        case TT_FORMAT_UNKNOWN: break;
    }
    return tiktak::decode::formatName(tiktak::decode::Format::Unknown);
}

tt_format tt_sniff_format(const void* data, size_t bytes) {
    return to_c(tiktak::decode::Decoder::sniff(data, bytes));
}

namespace {

tt_decoder* wrap(std::unique_ptr<tiktak::decode::Decoder> decoder, tt_status* status) {
    if (decoder == nullptr) {
        if (status) *status = TT_ERR_UNSUPPORTED;
        return nullptr;
    }

    auto* handle = new (std::nothrow) tt_decoder(std::move(decoder));
    if (handle == nullptr) {
        if (status) *status = TT_ERR_OUT_OF_MEMORY;
        return nullptr;
    }

    if (status) *status = TT_OK;
    return handle;
}

}  // namespace

tt_decoder* tt_decoder_open_memory(const void* data, size_t bytes, tt_status* status) {
    if (data == nullptr || bytes == 0) {
        if (status) *status = TT_ERR_INVALID_ARG;
        return nullptr;
    }
    return wrap(tiktak::decode::Decoder::openMemory(data, bytes), status);
}

tt_decoder* tt_decoder_open_file(const char* path, tt_status* status) {
    if (path == nullptr) {
        if (status) *status = TT_ERR_INVALID_ARG;
        return nullptr;
    }
    return wrap(tiktak::decode::Decoder::open(path), status);
}

void tt_decoder_close(tt_decoder* decoder) { delete decoder; }

tt_audio_info tt_decoder_info(const tt_decoder* decoder) {
    tt_audio_info out{};
    out.format = TT_FORMAT_UNKNOWN;
    if (decoder == nullptr) return out;

    const tiktak::decode::AudioInfo& info = decoder->impl->info();
    out.sample_rate = info.sample_rate;
    out.channels = info.channels;
    out.frames = info.frames;
    out.duration_sec = info.durationSec();
    out.format = to_c(info.format);
    return out;
}

size_t tt_decoder_read(tt_decoder* decoder, float* out, size_t frames) {
    if (decoder == nullptr || out == nullptr) return 0;
    return decoder->impl->readMono(out, frames);
}

int tt_decoder_seek(tt_decoder* decoder, unsigned long long frame) {
    if (decoder == nullptr) return 0;
    return decoder->impl->seek(frame) ? 1 : 0;
}
