#include "wav.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace tiktak::desktop {
namespace {

void put32(std::vector<unsigned char>& out, std::uint32_t v) {
    out.push_back(static_cast<unsigned char>(v & 0xFF));
    out.push_back(static_cast<unsigned char>((v >> 8) & 0xFF));
    out.push_back(static_cast<unsigned char>((v >> 16) & 0xFF));
    out.push_back(static_cast<unsigned char>((v >> 24) & 0xFF));
}

void put16(std::vector<unsigned char>& out, std::uint16_t v) {
    out.push_back(static_cast<unsigned char>(v & 0xFF));
    out.push_back(static_cast<unsigned char>((v >> 8) & 0xFF));
}

void putTag(std::vector<unsigned char>& out, const char* tag) {
    for (int i = 0; i < 4; ++i) out.push_back(static_cast<unsigned char>(tag[i]));
}

}  // namespace

bool writeWav(const std::string& path, const std::vector<float>& samples,
              double sample_rate) {
    const auto rate = static_cast<std::uint32_t>(sample_rate + 0.5);
    const auto data_bytes = static_cast<std::uint32_t>(samples.size() * 2);

    std::vector<unsigned char> bytes;
    bytes.reserve(44 + samples.size() * 2);

    putTag(bytes, "RIFF");
    put32(bytes, 36 + data_bytes);
    putTag(bytes, "WAVE");
    putTag(bytes, "fmt ");
    put32(bytes, 16);           // PCM header size
    put16(bytes, 1);            // PCM
    put16(bytes, 1);            // mono
    put32(bytes, rate);
    put32(bytes, rate * 2);     // byte rate
    put16(bytes, 2);            // block align
    put16(bytes, 16);           // bits
    putTag(bytes, "data");
    put32(bytes, data_bytes);

    for (float sample : samples) {
        double v = static_cast<double>(sample);
        if (v > 1.0) v = 1.0;
        if (v < -1.0) v = -1.0;
        const auto q = static_cast<std::int16_t>(std::lround(v * 32767.0));
        put16(bytes, static_cast<std::uint16_t>(q));
    }

    std::FILE* file = std::fopen(path.c_str(), "wb");
    if (!file) return false;
    const std::size_t written = std::fwrite(bytes.data(), 1, bytes.size(), file);
    std::fclose(file);
    return written == bytes.size();
}

bool writeWavFloat(const std::string& path, const std::vector<float>& samples,
                   double sample_rate) {
    const auto rate = static_cast<std::uint32_t>(sample_rate + 0.5);
    const auto data_bytes = static_cast<std::uint32_t>(samples.size() * 4);

    std::vector<unsigned char> bytes;
    bytes.reserve(58 + samples.size() * 4);

    putTag(bytes, "RIFF");
    put32(bytes, 50 + data_bytes);
    putTag(bytes, "WAVE");
    putTag(bytes, "fmt ");
    put32(bytes, 18);           // non-PCM header size, with cbSize
    put16(bytes, 3);            // IEEE float
    put16(bytes, 1);            // mono
    put32(bytes, rate);
    put32(bytes, rate * 4);     // byte rate
    put16(bytes, 4);            // block align
    put16(bytes, 32);           // bits
    put16(bytes, 0);            // cbSize
    putTag(bytes, "fact");      // required for any format but PCM
    put32(bytes, 4);
    put32(bytes, static_cast<std::uint32_t>(samples.size()));
    putTag(bytes, "data");
    put32(bytes, data_bytes);

    for (float sample : samples) {
        std::uint32_t word = 0;
        std::memcpy(&word, &sample, sizeof word);
        put32(bytes, word);
    }

    std::FILE* file = std::fopen(path.c_str(), "wb");
    if (!file) return false;
    const std::size_t written = std::fwrite(bytes.data(), 1, bytes.size(), file);
    std::fclose(file);
    return written == bytes.size();
}

}  // namespace tiktak::desktop
