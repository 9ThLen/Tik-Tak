#include "tracking/live.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace tiktak::tracking {

bool LiveConfig::valid() const {
    return odf.valid() && filter.valid() && sync.valid() && onset_peak_tau_sec > 0.0 &&
           gate_before_sec >= 0.0 && gate_after_sec >= 0.0 && lock_confidence > 0.0 &&
           lock_confidence <= 1.0 && release_confidence >= 0.0 &&
           release_confidence < lock_confidence && discontinuity_tolerance_sec > 0.0 &&
           activation_tempo.valid() && anchor_width_octaves > 0.0 &&
           anchor_octave_margin >= 0.0 && anchor_octave_margin <= 1.0 &&
           anchor_freeze_timeout_sec > 0.0 && bar_tempo.valid() &&
           bar_ratio_margin > 0.0;
}

double octaveNearest(double bpm, double held_bpm) {
    if (!(bpm > 0.0) || !(held_bpm > 0.0)) return bpm;
    return bpm * std::exp2(std::round(std::log2(held_bpm / bpm)));
}

bool barEndorsedOctave(double beat_period_sec, double bar_period_sec,
                       double margin, int* octave) {
    if (octave == nullptr) return false;
    if (!(beat_period_sec > 0.0) || !(bar_period_sec > 0.0)) return false;

    // 2, 3, 4 and 6. Not 5 or 7, which are not bars this product is for, and
    // not 8 or 12 on purpose: those are exactly what a doubled tempo implies,
    // so admitting them would make the wrong octave look like a long bar.
    static constexpr double kBars[] = {2.0, 3.0, 4.0, 6.0};

    double score[3];
    for (int k = -1; k <= 1; ++k) {
        const double candidate = beat_period_sec * std::exp2(k);
        const double ratio = bar_period_sec / candidate;
        double closest = std::numeric_limits<double>::infinity();
        for (const double bars : kBars) {
            closest = std::min(closest, std::fabs(std::log2(ratio / bars)));
        }
        score[k + 1] = closest;
    }

    int best_octave = -1;
    for (int k = 0; k <= 2; ++k) {
        if (score[k] < score[best_octave + 1]) best_octave = k - 1;
    }
    const double best = score[best_octave + 1];
    double runner_up = std::numeric_limits<double>::infinity();
    for (int k = -1; k <= 1; ++k) {
        if (k != best_octave) runner_up = std::min(runner_up, score[k + 1]);
    }

    // The winner has to be close to a plausible bar in the first place. Without
    // this, three candidates that are all equally wrong would still elect one
    // of themselves as soon as they were unequally wrong.
    if (best > margin * 2.0) return false;
    if (runner_up - best < margin) return false;

    *octave = best_octave;
    return true;
}

LiveConfig liveConfigFor(double sample_rate) {
    LiveConfig out;
    out.odf.sampleRate = sample_rate;
    if (!(sample_rate > 0.0)) return out;

    const double scale = sample_rate / 48000.0;

    // The hop is any integer; the window has to be a power of two for the
    // radix-2 transform, so it is the nearest one to the scaled size.
    out.odf.hopSize = std::max<std::size_t>(
        64, static_cast<std::size_t>(std::lround(512.0 * scale)));

    const double target = 2048.0 * scale;
    std::size_t frame = 256;
    while (frame < 8192 && static_cast<double>(frame) * 1.5 < target) frame *= 2;
    out.odf.frameSize = std::max(frame, out.odf.hopSize * 2);
    return out;
}

ParticleFilterConfig LiveTracker::resolveFilter(const LiveConfig& config) {
    ParticleFilterConfig out = config.filter;

    // The window says how late an onset may be and still count as on the beat,
    // and it cannot usefully be tighter than the front-end's own resolution: at
    // 22 kHz a hop is 23 ms, so a 25 ms window is one frame wide and most of a
    // real onset's energy falls outside it however well the tempo is tracked.
    // Confidence then reads as doubt about the audio when it is really doubt
    // about the ODF's time resolution. A frame and a half is the floor.
    const double hop_sec = static_cast<double>(config.odf.hopSize) / config.odf.sampleRate;
    const double period = 60.0 / out.prior_centre_bpm;
    out.beat_window = std::max(out.beat_window, 1.5 * hop_sec / period);
    return out;
}

LiveTracker::LiveTracker(const LiveConfig& config)
    : config_(config), odf_(config.odf), filter_(resolveFilter(config)), sync_(config.sync),
      activation_tempo_(config.activation_tempo),
      bar_tempo_(config.bar_tempo),
      evidence_half_sec_(0.5 * static_cast<double>(config.odf.frameSize) /
                         config.odf.sampleRate) {
    gate_start_.fill(std::numeric_limits<double>::infinity());
    gate_end_.fill(-std::numeric_limits<double>::infinity());
}

LiveTracker::LiveTracker(const LiveConfig& config, const ml::BeatNetWeights& weights)
    : LiveTracker(config) {
    if (!weights.valid()) return;
    model_.emplace(config.odf.sampleRate, weights);
    evidence_half_sec_ = 0.5 * static_cast<double>(ml::BeatNetFeatures::kFrameSize) /
                         ml::BeatNetFeatures::kModelRate;
}

LiveTracker::LiveTracker(const LiveConfig& config,
                         const ml::BeatNetWeights* const* weights, std::size_t count)
    : LiveTracker(config) {
    // Same contract as the single-weight form: an invalid set leaves the
    // tracker on spectral flux rather than half-built. Checked before anything
    // is constructed, so an ensemble is all of its checkpoints or none of them
    // -- averaging two where three were asked for is a different estimator with
    // different numbers, and would be indistinguishable from the right one at
    // the call site.
    if (weights == nullptr || count == 0) return;
    for (std::size_t i = 0; i < count; ++i) {
        if (weights[i] == nullptr || !weights[i]->valid()) return;
    }
    model_.emplace(config.odf.sampleRate, weights, count);
    evidence_half_sec_ = 0.5 * static_cast<double>(ml::BeatNetFeatures::kFrameSize) /
                         ml::BeatNetFeatures::kModelRate;
}

void LiveTracker::gateClick(double heard_time_sec) {
    gate_start_[gate_next_] = heard_time_sec - config_.gate_before_sec;
    gate_end_[gate_next_] = heard_time_sec + config_.gate_after_sec;
    gate_next_ = (gate_next_ + 1) % kGates;
}

bool LiveTracker::gatedAt(double frame_time_sec) const {
    // A frame is not a moment: it is a window, and a click anywhere inside it
    // colours the whole thing. The comparison is therefore window against
    // window, not centre against window.
    const double start = frame_time_sec - evidence_half_sec_;
    const double end = frame_time_sec + evidence_half_sec_;
    for (std::size_t i = 0; i < kGates; ++i) {
        if (end > gate_start_[i] && start < gate_end_[i]) return true;
    }
    return false;
}

void LiveTracker::process(double stream_time_sec, const float* samples, std::size_t n) {
    if (samples == nullptr || n == 0) return;

    const double sample_rate = config_.odf.sampleRate;

    if (!started_) {
        started_ = true;
        origin_sec_ = stream_time_sec;
        consumed_ = 0;
    } else {
        const double expected = origin_sec_ + static_cast<double>(consumed_) / sample_rate;
        if (std::fabs(stream_time_sec - expected) > config_.discontinuity_tolerance_sec) {
            // The device dropped or repeated a buffer. Follow the clock rather
            // than the sample count — the samples in flight are real, their
            // timestamps are what moved — by shifting the origin. Resetting the
            // ODF instead would throw away a window of audio and a beat with
            // it, for what is usually a one-buffer hiccup.
            ++stats_.discontinuities;
            origin_sec_ = stream_time_sec - static_cast<double>(consumed_) / sample_rate;
        }
    }

    if (model_) {
        // The learned front end replaces spectral flux and nothing else. Note
        // what does *not* happen here: no peak follower, because an activation
        // already answers the question on its own scale, and no separate
        // clock, because the model reports frames from its own zero exactly as
        // the ODF does and the origin correction above applies to both.
        //
        // One honest asymmetry. Gating drops the click's frames before the
        // filter sees them, as it does for the ODF — but spectral flux is
        // computed frame by frame with no memory, while an LSTM has already
        // taken the click into its state by the time the frame is dropped.
        // What the gate can still prevent is the click moving the filter,
        // which is the failure that matters: a tracker that locks onto itself.
        // What it cannot prevent is the model having heard it.
        model_->process(samples, n, [&](double frame_sec, double beat,
                                        double downbeat) {
            ++stats_.frames;
            const double time_sec = origin_sec_ + frame_sec;
            if (gatedAt(time_sec)) {
                ++stats_.gated;
                return;
            }
            // The downbeat goes in before the beat does, so that the anchor
            // decision inside submit() sees a bar rate that includes this
            // frame rather than one frame behind it. Gated exactly as the beat
            // is: a frame the click contaminated is contaminated in all three
            // channels.
            if (config_.bar_channel) {
                bar_tempo_.observe(time_sec,
                                   std::min(1.0, std::max(0.0, downbeat)));
            }
            submit(time_sec, std::min(1.0, std::max(0.0, beat)));
        });
        consumed_ += n;
        return;
    }

    const double hop_sec = static_cast<double>(config_.odf.hopSize) / sample_rate;
    const double decay = std::exp(-hop_sec / config_.onset_peak_tau_sec);

    odf_.process(samples, n, [&](const dsp::OdfFrame& frame) {
        ++stats_.frames;
        const double time_sec = origin_sec_ + frame.timeSec;
        const double value = static_cast<double>(frame.full);

        // Gated frames are dropped before the level tracker sees them, not
        // just before the filter does. Our own click is the loudest onset in
        // the room, and letting it into the running level would raise the bar
        // the actual music has to clear.
        if (gatedAt(time_sec)) {
            ++stats_.gated;
            return;
        }

        // A decaying peak follower: it rises instantly to a new loudest onset
        // and forgets one over a few seconds, so the scale tracks the room
        // without a loud hit permanently deafening the tracker.
        onset_peak_ = std::max(value, onset_peak_ * decay);

        // The epsilon is what keeps digital silence and room hiss from being
        // divided up into full-scale onsets.
        const double normalised = std::min(1.0, value / (onset_peak_ + 1e-6));
        submit(time_sec, normalised);
    });

    consumed_ += n;
}

void LiveTracker::submit(double time_sec, double normalised) {
    if (manual_bpm_ > 0.0) {
        sync_.observe(time_sec, normalised);
        // Acquisition happens once. Handing the filter a fresh phase every
        // frame would keep flattening the cloud it has been building, and the
        // correlation is the coarser of the two answers — it is a mean over the
        // last few seconds, so a syncopated bar drags it, whereas the filter's
        // window is local and merely lowers the particles the stray hit missed.
        if (!acquired_ && sync_.ready()) {
            filter_.seedPhase(sync_.nextBeat(time_sec));
            acquired_ = true;
        }
    }

    // Fed whatever the front end produced, before the filter sees it and
    // whether or not the anchor is on: the estimate is worth reporting even
    // when nothing acts on it, and a history that only exists while a feature
    // is enabled cannot be compared against one where it is not.
    activation_tempo_.observe(time_sec, normalised);

    // Manual mode already has a period and did not get it from the room, so
    // the anchor has nothing to add and no business overruling it.
    if (config_.anchor_tempo && manual_bpm_ <= 0.0) {
        const auto measured = activation_tempo_.estimate();
        last_octave_margin_ = measured.octave_margin;
        has_margin_ = measured.answered();

        // The hold expires on its own clock, before anything is decided with
        // it, and whatever the estimator is doing.
        //
        // Checking this only on the weak-margin branch left it immortal in the
        // one case that matters most: silence produces no estimate at all, so
        // no branch that could expire it was ever reached, and a hold taken in
        // the first chorus survived an arbitrarily long quiet passage. Not
        // answering is not evidence for the hold.
        if (config_.anchor_octave_freeze && held_octave_bpm_ > 0.0 &&
            time_sec - held_since_sec_ >= config_.anchor_freeze_timeout_sec) {
            held_octave_bpm_ = 0.0;
        }

        // The abstain arm changes what is published, never what is tracked, so
        // its threshold is a publishing threshold and the anchor keeps the
        // baseline's behaviour. Without this the arm would be silence *and* a
        // dropped anchor at once, which is the `clear` arm with a mute on it
        // and measures neither of the two things separately.
        const double anchor_gate =
            config_.anchor_margin_abstain ? 0.0 : config_.anchor_octave_margin;

        if (!measured.answered()) {
            // No estimate at all, which is a different state from an estimate
            // that is not trusted: there is nothing to hold either.
            filter_.clearAnchor();
        } else if (measured.octave_margin >= anchor_gate) {
            // The bar rate gets to move the octave, and only the octave. It is
            // the one piece of evidence here that did not come from re-reading
            // the beat channel, which is the entire reason this arm exists —
            // see eval/PREREGISTERED_downbeat_channel.md, and the octave freeze
            // it replaces, which acted correctly on no new evidence and moved
            // nothing.
            double bpm = measured.bpm;
            int octave = 0;
            if (config_.bar_channel) {
                const auto bar = bar_tempo_.estimate();
                if (bar.answered() &&
                    barEndorsedOctave(60.0 / measured.bpm, 60.0 / bar.bpm,
                                      config_.bar_ratio_margin, &octave)) {
                    // A candidate period of P * 2^k is a tempo of bpm / 2^k.
                    bpm = measured.bpm * std::exp2(-octave);
                }
            }

            // One width, unconditionally. Making it depend on whether the
            // filter and the anchor sit at the same metrical level was built,
            // measured on both corpus families, and removed: see
            // LiveConfig::anchor_width_octaves for the numbers and for why the
            // obvious version of that rule points the wrong way.
            filter_.anchorTempo(bpm, config_.anchor_width_octaves);
            // A confident anchor always refreshes the hold, including when it
            // lands on a different octave from the one being held. The freeze
            // exists to survive an absence of evidence, never to outvote it.
            // What was actually anchored, not what the estimator said. With the
            // bar channel on, those differ exactly when it moved the octave,
            // and holding the estimator's answer would let the freeze undo the
            // bar rate the moment the margin weakened.
            held_octave_bpm_ = bpm;
            held_since_sec_ = time_sec;
        } else if (config_.anchor_octave_freeze && held_octave_bpm_ > 0.0) {
            // Weak margin, and an octave recently worth keeping. Move the
            // estimator's own tempo by whole octaves to the equivalent nearest
            // the hold.
            //
            // The tempo *inside* the octave is not frozen: a band drifting from
            // 128 to 132 anchors at 132, because the doubt is over which
            // multiple of the pulse is the beat and never over the pulse. And
            // the phase is not touched — this writes an anchor, which is a
            // prior over period, and nothing here reaches where a beat falls.
            filter_.anchorTempo(octaveNearest(measured.bpm, held_octave_bpm_),
                                config_.anchor_width_octaves);
        } else if (config_.anchor_octave_freeze) {
            // Either nothing has been held yet, or the hold has outlived its
            // evidence. Both fall back to the baseline exactly: accepting a
            // weak anchor is what ships, and refusing to anchor at the start of
            // a recording is the arm that already lost.
            held_octave_bpm_ = 0.0;
            filter_.anchorTempo(measured.bpm, config_.anchor_width_octaves);
        } else {
            // Dropped rather than held. An anchor is a claim that the metrical
            // level is known, and when the estimator stops saying so the claim
            // has to go with it — otherwise a tempo measured in the first
            // chorus outlives the evidence for it.
            //
            // Measured and rejected as a policy of its own; see
            // LiveConfig::anchor_octave_margin. It stays reachable because it
            // is the control the freeze arm has to beat.
            filter_.clearAnchor();
        }
    }

    filter_.observe(time_sec, normalised);
}

void LiveTracker::observe(double time_sec, double activation) {
    ++stats_.frames;

    // Gated exactly as an ODF frame is: a learned front end hears our own
    // click too, and rather better than spectral flux does.
    if (gatedAt(time_sec)) {
        ++stats_.gated;
        return;
    }

    // No peak follower. An activation already answers "how much does this look
    // like a beat" on its own scale, and dividing it by a running maximum
    // would undo that — a quiet passage the model correctly reports as
    // uncertain would be renormalised back up into confidence.
    submit(time_sec, std::min(1.0, std::max(0.0, activation)));
}

bool LiveTracker::takeBeat(double now_sec, double lookahead_sec, double* beat_sec) {
    // The abstain arm, and the only other place it acts. "Publish nothing"
    // has to mean the clicks too — a tracker that reported no confidence and
    // went on handing out beats would be scored as silent by anything reading
    // the meter and as speaking by anything counting beats.
    if (abstaining()) return false;

    const BeatEstimate current = filter_.estimate(now_sec);

    double candidate = 0.0;
    if (manual_bpm_ > 0.0) {
        // Manual mode. There is no confidence gate past acquisition, and that
        // is the whole difference between the two modes: the tempo was never
        // the room's, so a room that falls silent takes nothing with it. The
        // grid continues because the pinned cloud continues — silence moves no
        // weights, so no special case is needed to keep it going.
        if (!acquired_) return false;
        held_period_sec_ = 60.0 / manual_bpm_;

        const double free_run = last_beat_sec_ + held_period_sec_;
        if (!published_ || free_run < now_sec - 2.0 * held_period_sec_) {
            // Nothing to continue from — the first beat after acquisition, or a
            // grid left so far behind that walking it back into the present one
            // period at a time would be a loop of unknown length in an audio
            // callback. Take the filter's answer whole.
            candidate = current.next_beat_sec;
        } else {
            // Afterwards the grid runs at the user's tempo and the room is only
            // allowed to nudge it. Folding the correction into half a period
            // either way is what makes "nudge" meaningful: the filter names a
            // beat, not a grid, and the beat it names may be the next one along
            // from the one being corrected.
            double correction = current.next_beat_sec - free_run;
            correction -= std::round(correction / held_period_sec_) * held_period_sec_;
            const double limit = config_.sync.max_drift * held_period_sec_;
            candidate = free_run + std::max(-limit, std::min(limit, correction));
        }
    } else {
        if (current.confidence >= config_.lock_confidence) {
            locked_ = true;
        } else if (current.confidence < config_.release_confidence) {
            locked_ = false;
            published_ = false;
        }
        if (!locked_) return false;

        if (current.confidence >= config_.lock_confidence) {
            candidate = current.next_beat_sec;
            held_period_sec_ = 60.0 / current.bpm;
        } else if (published_) {
            // Coasting: the cloud has lost the phase but the music has not
            // necessarily stopped. Carry on from the last beat at the last
            // tempo we were sure of, so a quiet bar sounds like a metronome
            // rather than like a fault.
            candidate = last_beat_sec_ + held_period_sec_;
        } else {
            return false;
        }
    }

    // One beat is handed out once. Between publishing a beat and that beat
    // arriving, the estimate keeps naming it; without this guard every callback
    // in that window would schedule another click on top of the same beat.
    //
    // Deliberately not conditioned on `published_`, which is where this used to
    // go wrong. A confidence dip below the release threshold clears that flag,
    // so a tracker flickering across the threshold — which is exactly what a
    // hard passage looks like — skipped the guard every time it came back and
    // handed out a beat however close it fell to the last one. On a piece with
    // a 0.9 s beat that produced clicks 30 ms apart: not a fast tempo, a
    // stutter. Nothing here needs the flag, because a genuine restart after a
    // long silence leaves last_beat_sec_ far enough back that the comparison
    // passes on its own.
    if (candidate < last_beat_sec_ + 0.5 * held_period_sec_) return false;

    if (candidate > now_sec + lookahead_sec) return false;
    if (candidate < now_sec) {
        // Predicted into the past: the buffer it belonged in has already gone
        // to the device. Skip it rather than play it late, and count it.
        ++stats_.beats_late;
        last_beat_sec_ = candidate;
        published_ = true;
        return false;
    }

    last_beat_sec_ = candidate;
    published_ = true;
    ++stats_.beats;
    if (beat_sec != nullptr) *beat_sec = candidate;
    return true;
}

void LiveTracker::seedTempo(double bpm, double spread_octaves) {
    filter_.seedTempo(bpm, spread_octaves);
}

void LiveTracker::setManualTempo(double bpm) {
    if (!(bpm > 0.0)) {
        if (!(manual_bpm_ > 0.0)) return;
        manual_bpm_ = 0.0;
        sync_.setPeriod(0.0);
        filter_.unpinPeriod();
        acquired_ = false;
        locked_ = false;
        published_ = false;
        return;
    }
    if (bpm == manual_bpm_) return;

    // Whether there is a phase worth keeping. `published_` is precisely "a
    // click is currently coming out", in either mode.
    const bool carry = published_;

    manual_bpm_ = bpm;
    // A typed tempo outranks a measured one. Leaving the anchor set would do
    // nothing while pinned — the prior is switched off there — but it would
    // come back the moment manual mode was left, carrying a tempo from before
    // the user overruled it.
    filter_.clearAnchor();
    const double period = 60.0 / bpm;
    held_period_sec_ = period;
    locked_ = false;

    // The correlation is measured in angles relative to a period, so a new
    // period discards it. The phase is not so fragile, and a user nudging the
    // BPM slider is asking for a different spacing between clicks, not for the
    // click to fall silent and resynchronise — so if one was already playing,
    // the new grid is anchored on the last beat actually played and carries
    // straight on from it.
    sync_.setPeriod(period);
    filter_.pinPeriod(period);
    if (carry) {
        filter_.seedPhase(last_beat_sec_ + period);
    } else {
        acquired_ = false;
    }
}

void LiveTracker::reset() {
    odf_.reset();
    // Carrying an LSTM across a reset is carrying a memory of music that is no
    // longer playing, and it would take several seconds to fade on its own.
    if (model_) model_->reset();
    filter_.reset();  // keeps the pin: a reset forgets audio, not the user
    // The anchor is dropped with it, by the same rule read the other way: the
    // activation history is audio, and this is what was concluded from it.
    activation_tempo_.reset();
    bar_tempo_.reset();
    sync_.reset();
    acquired_ = false;
    // The hold is a conclusion about audio, and goes with the audio.
    held_octave_bpm_ = 0.0;
    held_since_sec_ = 0.0;
    last_octave_margin_ = 0.0;
    has_margin_ = false;
    origin_sec_ = 0.0;
    consumed_ = 0;
    started_ = false;
    onset_peak_ = 0.0;
    gate_start_.fill(std::numeric_limits<double>::infinity());
    gate_end_.fill(-std::numeric_limits<double>::infinity());
    gate_next_ = 0;
    locked_ = false;
    published_ = false;
    last_beat_sec_ = -std::numeric_limits<double>::infinity();
    held_period_sec_ = 0.5;
    stats_ = Stats{};
}

LiveTracker::Stats LiveTracker::stats() const {
    Stats out = stats_;
    out.filter = filter_.stats();
    return out;
}

}  // namespace tiktak::tracking
