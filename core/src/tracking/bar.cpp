#include "tracking/bar.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>

namespace tiktak::tracking {

bool BarTracker::Config::valid() const {
    return salience_window_sec > 0.0 && window_beats >= 2 && min_beats >= 2 &&
           min_beats <= window_beats && resolver.valid();
}

BarTracker::BarTracker(const Config& config) : config_(config) {
    frame_time_.assign(kFrames, 0.0);
    frame_value_.assign(kFrames, 0.0);
    pending_time_.assign(kPending, 0.0);
    pending_index_.assign(kPending, 0);

    // Sized once so that the scored window never reallocates on the path a
    // beat arrives on.
    beat_time_.reserve(config_.window_beats);
    salience_.reserve(config_.window_beats);
    beat_index_.reserve(config_.window_beats);
}

void BarTracker::reset() {
    frame_next_ = 0;
    frames_ = 0;
    pending_ = 0;
    beat_time_.clear();
    salience_.clear();
    beat_index_.clear();
    result_ = analysis::DownbeatResult{};
    held_beats_per_bar_ = 0;
    held_downbeat_index_ = 0;
    decided_ = false;
    path_beats_per_bar_ = 0;
    path_downbeat_index_ = 0;
    path_decided_ = false;
    scored_ = 0;
}

void BarTracker::observe(double time_sec, double downbeat) {
    if (!std::isfinite(time_sec) || !std::isfinite(downbeat)) return;
    frame_time_[frame_next_] = time_sec;
    frame_value_[frame_next_] = downbeat;
    frame_next_ = (frame_next_ + 1) % kFrames;
    if (frames_ < kFrames) ++frames_;
}

void BarTracker::addBeat(double beat_sec, long long index) {
    if (!std::isfinite(beat_sec)) return;
    if (pending_ == kPending) {
        // Beats are consumed within a beat of being added, so a full queue
        // means update() is not being called. Drop the oldest rather than the
        // newest: the newest is the one still scoreable.
        for (std::size_t i = 1; i < kPending; ++i) {
            pending_time_[i - 1] = pending_time_[i];
            pending_index_[i - 1] = pending_index_[i];
        }
        --pending_;
    }
    pending_time_[pending_] = beat_sec;
    pending_index_[pending_] = index;
    ++pending_;
}

double BarTracker::peakAround(double centre_sec) const {
    const double low = centre_sec - config_.salience_window_sec;
    const double high = centre_sec + config_.salience_window_sec;
    double peak = 0.0;
    bool seen = false;
    for (std::size_t i = 0; i < frames_; ++i) {
        const double t = frame_time_[i];
        if (t < low || t > high) continue;
        if (!seen || frame_value_[i] > peak) {
            peak = frame_value_[i];
            seen = true;
        }
    }
    return seen ? peak : 0.0;
}

bool BarTracker::update(double now_sec) {
    std::size_t taken = 0;
    while (taken < pending_ &&
           pending_time_[taken] + config_.salience_window_sec <= now_sec) {
        if (beat_time_.size() == config_.window_beats) {
            beat_time_.erase(beat_time_.begin());
            salience_.erase(salience_.begin());
            beat_index_.erase(beat_index_.begin());
        }
        beat_time_.push_back(pending_time_[taken]);
        salience_.push_back(peakAround(pending_time_[taken]));
        beat_index_.push_back(pending_index_[taken]);
        ++taken;
    }
    if (taken == 0) return false;

    for (std::size_t i = taken; i < pending_; ++i) {
        pending_time_[i - taken] = pending_time_[i];
        pending_index_[i - taken] = pending_index_[i];
    }
    pending_ -= taken;

    if (beat_time_.size() < config_.min_beats) return false;

    // `result_` is always the latest window's answer, including when that
    // answer is "nothing". The held pair below is what is displayed. Keeping
    // them apart is the difference between reporting a stale margin as though
    // it were fresh and reporting an old decision as what is still on screen.
    result_ = analysis::resolveMeter(salience_, beat_time_, config_.resolver);
    scored_ = beat_time_.size();

    if (result_.beats_per_bar > 0) {
        held_beats_per_bar_ = result_.beats_per_bar;
        // `phase` indexes the beat list handed in, so the bar line it names has
        // to be carried back out into the tracker's own numbering before the
        // window slides and that index means a different beat.
        std::size_t anchor = static_cast<std::size_t>(result_.phase);
        if (!result_.downbeats.empty()) {
            const auto found = std::find(beat_time_.rbegin(), beat_time_.rend(),
                                         result_.downbeats.back());
            if (found != beat_time_.rend()) {
                const std::size_t path_anchor = beat_time_.size() - 1 -
                    static_cast<std::size_t>(
                        std::distance(beat_time_.rbegin(), found));
                path_beats_per_bar_ = result_.beats_per_bar;
                path_downbeat_index_ = beat_index_[path_anchor];
                path_decided_ = true;
                if (config_.use_latest_path_downbeat) anchor = path_anchor;
            }
        }
        held_downbeat_index_ = beat_index_[anchor];
        decided_ = true;
    }
    // A window that decided nothing does not erase one that did. A bar or two
    // of ambiguity mid-song is not evidence that the meter changed, and
    // blanking the display on it would flicker on exactly the material this is
    // for. What it does do is show through in result(), so a caller that wants
    // to grey out the display on stale evidence can.
    return true;
}

int BarTracker::positionOf(long long index) const {
    if (!decided_ || held_beats_per_bar_ <= 0) return -1;
    const long long m = held_beats_per_bar_;
    long long delta = (index - held_downbeat_index_) % m;
    if (delta < 0) delta += m;
    return static_cast<int>(delta);
}

int BarTracker::pathPositionOf(long long index) const {
    if (!path_decided_ || path_beats_per_bar_ <= 0) return -1;
    const long long m = path_beats_per_bar_;
    long long delta = (index - path_downbeat_index_) % m;
    if (delta < 0) delta += m;
    return static_cast<int>(delta);
}

}  // namespace tiktak::tracking
