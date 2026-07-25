#include "render/player.hpp"

#include <cmath>

namespace tiktak::render {

bool PlayerConfig::valid() const {
    if (sample_rate <= 0.0 || !click.valid()) return false;
    // One rate throughout: the track, the grid times and the click all meet in
    // the same buffer, and a mismatch would be a resampling bug wearing a
    // configuration's clothes.
    if (click.sample_rate != sample_rate) return false;
    if (beats_per_bar < 1 || downbeat_offset < 0) return false;
    if (count_in_beats < 0 || cue_lookahead_sec < 0.0) return false;
    return true;
}

bool TrackPlayer::Stats::clean() const {
    return clicks_late == 0 && clicks_overflowed == 0 && voices_stolen == 0 &&
           discontinuities == 0 && cues_dropped == 0;
}

TrackPlayer::TrackPlayer(const PlayerConfig& config)
    : config_(config), click_(config.click) {}

void TrackPlayer::setTrack(const float* samples, std::size_t frames) {
    track_ = samples;
    track_frames_ = frames;
}

void TrackPlayer::setGrid(const double* beat_times, std::size_t count) {
    grid_.assign(beat_times, beat_times + count);
}

bool TrackPlayer::setLoop(std::int64_t start_bar, std::int64_t end_bar) {
    if (started_ || grid_.empty() || end_bar <= start_bar) return false;

    const std::int64_t bpb = config_.beats_per_bar;
    const std::int64_t first = config_.downbeat_offset + start_bar * bpb;
    const std::int64_t last = config_.downbeat_offset + end_bar * bpb;
    if (first < 0 || first >= static_cast<std::int64_t>(grid_.size())) return false;
    if (last > static_cast<std::int64_t>(grid_.size())) return false;

    double end_sec;
    if (last < static_cast<std::int64_t>(grid_.size())) {
        end_sec = grid_[static_cast<std::size_t>(last)];
    } else if (grid_.size() >= 2) {
        // Looping through the final bar: its end is one beat past the last
        // beat, extrapolated at the closing tempo.
        end_sec = grid_.back() + (grid_.back() - grid_[grid_.size() - 2]);
    } else {
        return false;
    }

    loop_set_ = true;
    loop_start_beat_ = static_cast<std::size_t>(first);
    loop_end_beat_ = static_cast<std::size_t>(last);
    loop_start_sec_ = grid_[loop_start_beat_];
    loop_end_sec_ = end_sec;
    return true;
}

void TrackPlayer::clearLoop() {
    if (!started_) loop_set_ = false;
}

bool TrackPlayer::start(double stream_time_sec, std::int64_t from_bar) {
    if (!track_ || track_frames_ == 0) return false;

    if (grid_.empty()) {
        // No grid: a plain player, from the top, no clicks and no count-in.
        if (config_.count_in_beats > 0 || from_bar != 0) return false;
        start_beat_ = 0;
        start_beat_sec_ = 0.0;
        count_in_interval_sec_ = 0.0;
    } else {
        const std::int64_t s =
            config_.downbeat_offset + from_bar * config_.beats_per_bar;
        if (s < 0 || s >= static_cast<std::int64_t>(grid_.size())) return false;
        if (loop_set_ && static_cast<std::size_t>(s) >= loop_end_beat_) return false;
        start_beat_ = static_cast<std::size_t>(s);
        start_beat_sec_ = grid_[start_beat_];

        // The count-in clicks at the tempo the music is about to have, so the
        // interval is read off the grid at the entry point, not from a global
        // bpm — on a rubato track those differ.
        if (start_beat_ + 1 < grid_.size()) {
            count_in_interval_sec_ = grid_[start_beat_ + 1] - grid_[start_beat_];
        } else if (start_beat_ >= 1) {
            count_in_interval_sec_ = grid_[start_beat_] - grid_[start_beat_ - 1];
        } else if (config_.count_in_beats > 0) {
            return false;  // one lone beat gives no interval to count at
        } else {
            count_in_interval_sec_ = 0.0;
        }
    }

    count_in_sec_ = config_.count_in_beats * count_in_interval_sec_;
    track_entry_frame_ = std::llround(start_beat_sec_ * config_.sample_rate);
    if (track_entry_frame_ >= static_cast<std::int64_t>(track_frames_)) return false;

    start_stream_sec_ = stream_time_sec;
    click_cursor_ = BeatCursor{start_beat_, 0, 0.0};
    cue_cursor_ = BeatCursor{start_beat_, 0, 0.0};
    beats_ = 0;
    loops_ = 0;
    cues_dropped_ = 0;
    timeline_end_frame_ = 0;
    click_.reset();
    started_ = true;
    ended_ = false;
    return true;
}

void TrackPlayer::stop() { started_ = false; }

void TrackPlayer::silence() {
    started_ = false;
    click_.reset();
}

double TrackPlayer::positionSec() const {
    if (!started_ && !ended_) return 0.0;
    const std::int64_t count_in_frames =
        std::llround(count_in_sec_ * config_.sample_rate);
    if (timeline_end_frame_ <= count_in_frames) return start_beat_sec_;

    std::int64_t frame = track_entry_frame_ + (timeline_end_frame_ - count_in_frames);
    if (loop_set_) {
        const std::int64_t ls = std::llround(loop_start_sec_ * config_.sample_rate);
        const std::int64_t le = std::llround(loop_end_sec_ * config_.sample_rate);
        while (frame >= le && le > ls) frame -= le - ls;
    }
    if (frame > static_cast<std::int64_t>(track_frames_)) {
        frame = static_cast<std::int64_t>(track_frames_);
    }
    return static_cast<double>(frame) / config_.sample_rate;
}

schedule::BeatKind TrackPlayer::kindOf(std::size_t beat_index) const {
    const std::int64_t rel =
        static_cast<std::int64_t>(beat_index) - config_.downbeat_offset;
    if (rel >= 0 && rel % config_.beats_per_bar == 0) {
        return schedule::BeatKind::Downbeat;
    }
    return schedule::BeatKind::Beat;
}

std::int64_t TrackPlayer::barOf(std::size_t beat_index) const {
    const std::int64_t rel =
        static_cast<std::int64_t>(beat_index) - config_.downbeat_offset;
    const std::int64_t bpb = config_.beats_per_bar;
    return rel >= 0 ? rel / bpb : -((-rel + bpb - 1) / bpb);
}

int TrackPlayer::beatInBar(std::size_t beat_index) const {
    const std::int64_t rel =
        static_cast<std::int64_t>(beat_index) - config_.downbeat_offset;
    const std::int64_t bpb = config_.beats_per_bar;
    return static_cast<int>(((rel % bpb) + bpb) % bpb);
}

bool TrackPlayer::nextEvent(BeatCursor& cursor, double* when_sec,
                            std::size_t* beat_index, bool* is_count_in) {
    if (cursor.count_in < static_cast<std::size_t>(config_.count_in_beats)) {
        *when_sec = static_cast<double>(cursor.count_in) * count_in_interval_sec_;
        *beat_index = cursor.count_in;
        *is_count_in = true;
        ++cursor.count_in;
        return true;
    }

    if (grid_.empty()) return false;

    if (loop_set_ && cursor.beat >= loop_end_beat_) {
        cursor.beat = loop_start_beat_;
        cursor.loop_shift_sec += loop_end_sec_ - loop_start_sec_;
    }
    if (cursor.beat >= grid_.size()) return false;

    *when_sec =
        count_in_sec_ + (grid_[cursor.beat] - start_beat_sec_) + cursor.loop_shift_sec;
    *beat_index = cursor.beat;
    *is_count_in = false;
    ++cursor.beat;
    return true;
}

void TrackPlayer::mixTrack(double stream_time_sec, float* out, std::size_t frames) {
    const double sr = config_.sample_rate;
    const std::int64_t p0 = std::llround((stream_time_sec - start_stream_sec_) * sr);
    const std::int64_t count_in_frames = std::llround(count_in_sec_ * sr);
    const std::int64_t ls =
        loop_set_ ? std::llround(loop_start_sec_ * sr) : 0;
    const std::int64_t le = loop_set_ ? std::llround(loop_end_sec_ * sr) : 0;

    for (std::size_t k = 0; k < frames; ++k) {
        const std::int64_t p = p0 + static_cast<std::int64_t>(k);
        if (p < count_in_frames) continue;  // count-in: clicks only

        std::int64_t f = track_entry_frame_ + (p - count_in_frames);
        if (loop_set_ && le > ls) {
            std::size_t folds = 0;
            while (f >= le) {
                f -= le - ls;
                ++folds;
            }
            // The loop count is read off the track itself rather than the
            // click cursor, so it stays truthful with the click muted.
            if (folds > loops_) loops_ = folds;
        }
        if (f >= static_cast<std::int64_t>(track_frames_)) {
            ended_ = true;
            break;
        }
        out[k] += track_[f];
    }

    timeline_end_frame_ = p0 + static_cast<std::int64_t>(frames);
}

void TrackPlayer::process(double stream_time_sec, float* out, std::size_t frames,
                          schedule::Event* cues, std::size_t cue_capacity,
                          std::size_t* cue_count) {
    if (cue_count) *cue_count = 0;

    if (started_ && !ended_) {
        const double sr = config_.sample_rate;
        const double buffer_end = stream_time_sec + static_cast<double>(frames) / sr;

        // Clicks due in this buffer, scheduled and rendered in the same call.
        // A beat rounding to the first sample of the *next* buffer stays in
        // the renderer's pending queue — that case is its regression test.
        if (config_.channel_enabled[static_cast<int>(schedule::Channel::Audio)]) {
            const double edge = buffer_end + 0.5 / sr;
            for (;;) {
                BeatCursor peek = click_cursor_;
                double when;
                std::size_t index;
                bool is_count_in;
                if (!nextEvent(peek, &when, &index, &is_count_in)) break;
                const double at = start_stream_sec_ + when;
                if (at >= edge) break;
                click_cursor_ = peek;
                click_.schedule(at,
                                is_count_in ? schedule::BeatKind::Beat : kindOf(index));
                ++beats_;
            }
        }

        // Haptic and visual cues, handed out ahead: the shell has to schedule
        // the taptic engine and the next frame in advance, which the click
        // does not need — it leaves through this very buffer.
        const bool haptic =
            config_.channel_enabled[static_cast<int>(schedule::Channel::Haptic)];
        const bool visual =
            config_.channel_enabled[static_cast<int>(schedule::Channel::Visual)];
        if (haptic || visual) {
            const double horizon = buffer_end + config_.cue_lookahead_sec;
            const double heard_delay =
                config_.latency_sec[static_cast<int>(schedule::Channel::Audio)];
            std::size_t written = cue_count ? *cue_count : 0;
            for (;;) {
                BeatCursor peek = cue_cursor_;
                double when;
                std::size_t index;
                bool is_count_in;
                if (!nextEvent(peek, &when, &index, &is_count_in)) break;
                const double at = start_stream_sec_ + when;
                if (at >= horizon) break;
                cue_cursor_ = peek;

                // The beat is *heard* one output latency after its sample is
                // handed over; each cue channel is compensated against that
                // moment with its own delay.
                const double heard = at + heard_delay;
                for (const schedule::Channel channel :
                     {schedule::Channel::Haptic, schedule::Channel::Visual}) {
                    if (channel == schedule::Channel::Haptic && !haptic) continue;
                    if (channel == schedule::Channel::Visual && !visual) continue;

                    schedule::Event event;
                    event.time_sec =
                        heard - config_.latency_sec[static_cast<int>(channel)];
                    event.beat_time_sec = heard;
                    event.channel = channel;
                    event.kind =
                        is_count_in ? schedule::BeatKind::Beat : kindOf(index);
                    event.step = is_count_in
                                     ? static_cast<std::int64_t>(index) -
                                           config_.count_in_beats
                                     : static_cast<std::int64_t>(index);
                    event.bar = is_count_in ? -1 : barOf(index);
                    event.beat_in_bar =
                        is_count_in ? static_cast<int>(index) : beatInBar(index);
                    event.subdivision = 0;

                    if (cues && written < cue_capacity) {
                        cues[written++] = event;
                    } else {
                        ++cues_dropped_;
                    }
                }
            }
            if (cue_count) *cue_count = written;
        }

        mixTrack(stream_time_sec, out, frames);
    }

    // Always mixed, even stopped or ended: a click that started before the
    // stop rings out instead of being cut, exactly as Metronome does.
    click_.mix(stream_time_sec, out, frames);
}

TrackPlayer::Stats TrackPlayer::stats() const {
    Stats s;
    s.beats = beats_;
    s.loops = loops_;
    s.clicks_late = click_.dropped_late();
    s.clicks_overflowed = click_.dropped_overflow();
    s.voices_stolen = click_.stolen();
    s.discontinuities = click_.discontinuities();
    s.cues_dropped = cues_dropped_;
    return s;
}

}  // namespace tiktak::render
