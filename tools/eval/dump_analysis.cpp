// Dumps one file's offline analysis as JSON, for research/eval to score.
//
// Why a tool and not a Python port: the thing being measured has to be the
// thing that ships. A reimplementation of the downbeat scorer in Python would
// produce numbers about the reimplementation, and the first time the two drifted
// the evaluation would be measuring the wrong program without saying so. This
// runs the same core the app runs and prints what it concluded.
//
//   dump_analysis <song.mp3>                 decode WAV, FLAC or MP3
//   dump_analysis <clip.f32> <sample_rate>   raw 32-bit float mono, native order
//   dump_analysis <audio> [rate] --salience <file> [calibration]
//                                            replace the built-in cues with a
//                                            per-beat salience read from a file
//
// The second form exists so the synthetic clips in research/tiktak/synth.py can
// be scored without inventing a file format between here and there — they are
// already float arrays in memory.
//
// --salience is the seam in analysis/downbeat.hpp made reachable from outside:
// one finite number per beat, whitespace-separated, `#` starts a comment. The
// values keep their original scale: the resolver will not turn an almost-flat
// model output into unit-variance evidence. The beat grid still comes from the
// core's own tracker, and the bar length and phase still come from the core's
// own resolver — only the per-beat scorer is swapped. That is exactly the
// substitution an ONNX model will make, which is what lets a model be *scored*
// through the shipping resolver before a line of it is ported: run once to get
// the beats, sample the model's activation at those beat times, run again with
// the file. The count must match the beat count exactly; a mismatch is an
// error, not an alignment guess.
//
// A backend's calibration is three numbers — --salience-min-range,
// --salience-min-phase-margin, --salience-min-meter-margin — and they are
// passed together or not at all. Since the resolver no longer rescales
// arbitrary backend output, all three live in that backend's own units and
// none of them has a universal model-independent value. Supplying a range gate
// while inheriting the cue backend's margins would report `downbeat_confident`
// judged by numbers belonging to a different scorer, so it is refused.
//
// Output is one JSON object on stdout. Times are printed at full double
// precision because this is a machine format: a diff between two runs should
// show a change in behaviour, never a change in rounding.
//
// The grid cache is deliberately not consulted or written. An evaluation that
// silently scored a blob analysed under an older configuration would be the
// worst possible failure here, and the run is cheap enough not to need it.
#include "tiktak/tiktak.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <string>
#include <deque>
#include <vector>

// Internal, on purpose: resolveMeter is the seam itself, and going through the
// public C API instead would mean growing that API for a research need. The
// parity tools set the precedent.
#include "analysis/downbeat.hpp"
#include "dsp/resample.hpp"
#include "ml/beatnet.hpp"
#if TIKTAK_HAVE_ML
#include "ml/beat_this.hpp"
#include "ml/beat_this_session.hpp"
#endif
#include "analysis/offline.hpp"
#include "tracking/live.hpp"
#include "tracking/particle.hpp"

#if defined(TIKTAK_HAVE_DECODE)
#include "decode/decoder.hpp"
#endif

namespace {

std::vector<float> readRaw(const char* path) {
    std::FILE* file = std::fopen(path, "rb");
    if (!file) return {};
    std::fseek(file, 0, SEEK_END);
    const long bytes = std::ftell(file);
    std::fseek(file, 0, SEEK_SET);
    if (bytes <= 0) {
        std::fclose(file);
        return {};
    }
    std::vector<float> samples(static_cast<std::size_t>(bytes) / sizeof(float));
    if (std::fread(samples.data(), sizeof(float), samples.size(), file) != samples.size()) {
        std::fclose(file);
        return {};
    }
    std::fclose(file);
    return samples;
}

// One value per beat, whitespace-separated, `#` to end of line. Text rather
// than binary because the writer is a numpy one-liner and the file is worth
// being able to look at when a result surprises.
// The model's weights, as bytes. The core does no I/O and takes the blob.
bool readBytes(const char* path, std::vector<unsigned char>& out) {
    std::FILE* file = std::fopen(path, "rb");
    if (!file) return false;
    std::fseek(file, 0, SEEK_END);
    const long bytes = std::ftell(file);
    std::fseek(file, 0, SEEK_SET);
    out.resize(static_cast<std::size_t>(bytes));
    const bool ok = std::fread(out.data(), 1, out.size(), file) == out.size();
    std::fclose(file);
    return ok;
}

bool readSalience(const char* path, std::vector<double>& out,
                  std::string& error) {
    std::FILE* file = std::fopen(path, "rb");
    if (!file) {
        error = "cannot open salience file";
        return false;
    }

    std::string text;
    char block[4096];
    std::size_t got;
    while ((got = std::fread(block, 1, sizeof(block), file)) > 0) {
        text.append(block, got);
    }
    std::fclose(file);

    std::size_t i = 0;
    while (i < text.size()) {
        const char c = text[i];
        if (c == '#') {
            while (i < text.size() && text[i] != '\n') ++i;
        } else if (std::isspace(static_cast<unsigned char>(c))) {
            ++i;
        } else {
            char* end = nullptr;
            const double value = std::strtod(text.c_str() + i, &end);
            const std::size_t consumed = static_cast<std::size_t>(end - (text.c_str() + i));
            if (consumed == 0) {
                error = "invalid salience token at byte " + std::to_string(i);
                return false;
            }
            if (!std::isfinite(value)) {
                error = "non-finite salience value " +
                        std::to_string(out.size() + 1) + " at byte " +
                        std::to_string(i);
                return false;
            }
            out.push_back(value);
            i += consumed;
        }
    }
    return true;
}

struct VetoInterval {
    double onset_sec = 0.0;
    double close_sec = 0.0;
    double committed_bpm = 0.0;
};

// A fixed schedule is the smallest seam that lets the Python experiment replay
// a stateful policy through the real particle filter. The schedule is rebuilt
// from each replay until it reaches a fixed point; this class only applies it.
// It deliberately knows nothing about the decoder or annotations.
struct AnchorVetoSchedule {
    std::vector<VetoInterval> intervals;
    std::size_t next = 0;
    std::size_t applied_frames = 0;
    // Proposal onsets are observed on the audio-callback clock. Model frames
    // carry the start of their 64 ms feature window, so comparing a schedule
    // with that timestamp would apply every decision roughly one window late.
    // dump_analysis sets this to the current callback time before observations;
    // the fallback keeps the seam usable outside cached-model replay.
    double decision_time_sec = -1.0;

    double resolve(double time_sec, double measured_bpm) {
        if (decision_time_sec >= 0.0) time_sec = decision_time_sec;
        while (next < intervals.size() &&
               time_sec >= intervals[next].close_sec) {
            ++next;
        }
        if (next >= intervals.size()) return measured_bpm;
        const VetoInterval& interval = intervals[next];
        if (time_sec < interval.onset_sec) return measured_bpm;
        ++applied_frames;
        return tiktak::tracking::octaveNearest(measured_bpm,
                                               interval.committed_bpm);
    }

    static double callback(void* context, double time_sec, double measured_bpm) {
        return static_cast<AnchorVetoSchedule*>(context)->resolve(time_sec,
                                                                  measured_bpm);
    }
};

// The registered matched-cost comparison policies, decided **online**.
//
// They were first built as fixed schedules, like the decoder, and that does not
// work: a schedule has to name every proposal in advance, but delaying an
// octave change shifts the whole trajectory and creates proposals the baseline
// never had, so the schedule must grow and each growth creates more. Measured
// on RWC_C003, `debounce_1.5` never reached a fixed point in forty passes, and
// not for want of precision — its *decisions* kept changing, while every one of
// the 21 decoder arms settled. The decoder is threshold-gated and contracts;
// debounce is not and need not.
//
// Here they see their own consequences, which is what they would do in the
// product, and no iteration is involved at all.
struct OnlineOctavePolicy {
    enum class Kind { None, Debounce, RateLimit, TotalBan };

    Kind kind = Kind::None;
    double seconds = 0.0;  // D for debounce, N for the rate limit

    // The held level, and the same rule the decoder's event extraction uses:
    // it follows the estimator while that stays inside the octave and freezes
    // the moment an octave away is proposed. Following `measured` rather than
    // the published BPM matches what `held_octave_bpm_` already does in the
    // core's freeze arm; the tracker's own published tempo is not reachable
    // from here without re-entering `estimate()` on the audio thread.
    double committed_bpm = 0.0;
    double disagreement_since_sec = -1.0;
    double last_change_sec = -1.0;

    // Set from the poll loop, like `decision_time_sec`. Only the total ban
    // reads it: §7 words that policy as "no octave change after first lock",
    // and without this it would start banning during acquisition, which is a
    // different and already-measured problem.
    bool locked = false;
    double decision_time_sec = -1.0;
    std::size_t applied_frames = 0;

    // Near a power of two, at the 8% the labels and the live benchmark use.
    // Plain rounding in log space would call every ratio in (1.41, 2.83) a
    // doubling, and a 3:2 tempo relation sits inside it.
    static int octaveIndex(double measured_bpm, double committed_bpm) {
        if (!(measured_bpm > 0.0) || !(committed_bpm > 0.0)) return 0;
        const double exponent = std::log2(measured_bpm / committed_bpm);
        const int k = static_cast<int>(std::lround(exponent));
        if (k == 0) return 0;
        return std::fabs(exponent - k) > std::log2(1.08) ? 0 : k;
    }

    double resolve(double time_sec, double measured_bpm) {
        const double now = decision_time_sec >= 0.0 ? decision_time_sec : time_sec;
        if (kind == Kind::None || !(measured_bpm > 0.0)) return measured_bpm;
        if (!(committed_bpm > 0.0)) {
            committed_bpm = measured_bpm;
            return measured_bpm;
        }
        if (octaveIndex(measured_bpm, committed_bpm) == 0) {
            // Inside the octave the level simply follows, so a band drifting
            // 128 to 132 is never a proposal and never debounced.
            committed_bpm = measured_bpm;
            disagreement_since_sec = -1.0;
            return measured_bpm;
        }
        if (disagreement_since_sec < 0.0) disagreement_since_sec = now;

        bool block = false;
        switch (kind) {
            case Kind::Debounce:
                block = (now - disagreement_since_sec) < seconds;
                break;
            case Kind::RateLimit:
                block = last_change_sec >= 0.0 &&
                        (now - last_change_sec) < seconds;
                break;
            case Kind::TotalBan:
                block = locked;
                break;
            case Kind::None:
                break;
        }
        if (block) {
            ++applied_frames;
            return tiktak::tracking::octaveNearest(measured_bpm, committed_bpm);
        }
        committed_bpm = measured_bpm;
        last_change_sec = now;
        disagreement_since_sec = -1.0;
        return measured_bpm;
    }

    static double callback(void* context, double time_sec, double measured_bpm) {
        return static_cast<OnlineOctavePolicy*>(context)->resolve(time_sec,
                                                                  measured_bpm);
    }
};

bool readVetoSchedule(const char* path, AnchorVetoSchedule& schedule,
                      std::string& error) {
    std::vector<double> values;
    if (!readSalience(path, values, error)) return false;
    if (values.size() % 3 != 0) {
        error = "anchor veto schedule needs onset, close and committed BPM per row";
        return false;
    }
    schedule.intervals.reserve(values.size() / 3);
    for (std::size_t i = 0; i < values.size(); i += 3) {
        const VetoInterval interval{values[i], values[i + 1], values[i + 2]};
        if (!(interval.onset_sec >= 0.0) ||
            !(interval.close_sec > interval.onset_sec) ||
            !(interval.committed_bpm > 0.0)) {
            error = "invalid anchor veto interval " + std::to_string(i / 3 + 1);
            return false;
        }
        if (!schedule.intervals.empty() &&
            interval.onset_sec < schedule.intervals.back().close_sec) {
            error = "anchor veto intervals must be sorted and non-overlapping";
            return false;
        }
        schedule.intervals.push_back(interval);
    }
    return true;
}

// JSON has no way to say "not a number", and a reader that meets NaN either
// throws or silently invents null. Analysis of silence legitimately produces
// none of these values, so they are reported as 0 with the empty beat list
// alongside saying why.
double finiteOrZero(double value) {
    return std::isfinite(value) ? value : 0.0;
}

void printTimes(const char* name, const std::vector<double>& times, bool last) {
    std::printf("  \"%s\": [", name);
    for (std::size_t i = 0; i < times.size(); ++i) {
        std::printf("%s%.17g", i ? ", " : "", times[i]);
    }
    std::printf("]%s\n", last ? "" : ",");
}

// The bare minimum of JSON string escaping — enough for a file path, which is
// the only string this prints. A path with a quote or a backslash in it is
// ordinary on Windows and must not produce a broken document.
std::string escape(const std::string& text) {
    std::string out;
    out.reserve(text.size() + 8);
    for (char c : text) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

using tiktak::analysis::BeatFeature;

std::vector<double> cueColumn(const std::vector<BeatFeature>& features,
                              double BeatFeature::*member) {
    std::vector<double> out;
    out.reserve(features.size());
    for (const BeatFeature& f : features) out.push_back(f.*member);
    return out;
}

bool nonnegativeFinite(const char* text, double& value) {
    char* end = nullptr;
    value = std::strtod(text, &end);
    return end != text && *end == '\0' && value >= 0.0 && std::isfinite(value);
}

// The device-sized buffer the live path is driven in. At namespace scope
// because the activation dump has to feed the model in exactly these blocks:
// its emission schedule is what the replay reproduces, and a schedule recorded
// under a different block size is a schedule for a different run.
constexpr std::size_t kLiveBlock = 512;

}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string> positional;
    std::string salience_path;
    // A beat grid supplied from outside, so that a bad grid and a bad scorer
    // can be told apart. The bar-line stage takes the beats as given, and when
    // the tracker is an octave out every bar line is wrong for a reason that
    // has nothing to do with the downbeat cues. Replacing the grid isolates
    // the resolver; it is a research seam and never a product path.
    std::string beats_path;
    // The three numbers a backend calibrates as one set. Defaulted from the
    // built-in cue backend and overridden together — see the loop below for
    // why passing only some of them is refused.
    const tiktak::analysis::DownbeatConfig cue_defaults;
    double salience_min_range = cue_defaults.min_salience_range;
    double salience_min_phase = cue_defaults.min_phase_margin;
    double salience_min_meter = cue_defaults.min_meter_margin;
    int calibration_given = 0;
    // Fixes the tempo instead of estimating it, so an evaluation can separate
    // "the tempo hypothesis was wrong" from "the phase drifted at the right
    // tempo" — two failures that look identical in a beat grid and need
    // opposite fixes.
    double bpm_hint = 0.0;
    // Runs the causal microphone tracker over the file instead of the offline
    // analyser. Everything measured so far has been the offline path; the live
    // one ships in the microphone mode and had never been scored on real music
    // at all, which this exists to fix.
    bool live = false;
    // Seeds the causal tracker with the tempo the offline analyser found, which
    // is what the backing-track case can actually do: the app is playing the
    // file, so it has already analysed it and the microphone does not have to
    // rediscover the tempo from a dense mix.
    bool live_seed = false;
    // Prints the onset function itself, for experiments on the statistics the
    // live tracker derives from it — measuring a proposed confidence change in
    // Python first is a rebuild per idea cheaper.
    bool dump_odf = false;
    // Prints the learned activation the causal path runs on, before any decoder
    // has touched it.
    //
    // This is the measurement the live work has been missing. The benchmark's
    // largest single failure is recall — beats that were there and were not
    // found — and every experiment so far has asked which *decoder* loses them,
    // which cannot be answered while nobody has checked whether the activation
    // has a peak at those beats at all. With this dumped the question becomes
    // arithmetic: the maximum activation inside the same 70 ms window the score
    // uses, around each annotated beat, with no decoder involved in the answer.
    //
    // Computed by a second pass over the same samples through
    // ml::BeatNetActivation — the class LiveTracker itself holds — rather than
    // by a hook inside the tracker. A hook would put a research need inside a
    // real-time class that ships; a second pass through the same class cannot
    // disagree with the tracker about what the activation is.
    bool dump_activation = false;
    // Overrides for the live tracker's lock/release hysteresis, so a threshold
    // can be chosen on one batch and validated on another without a rebuild.
    double live_lock = 0.0;
    double live_release = 0.0;
    // Drives the particle filter from an activation computed elsewhere instead
    // of from the built-in onset function. The same seam as --salience on the
    // offline resolver, and for the same reason: the observation model is what
    // the live path's confidence was measured down to, and swapping it is the
    // only way to find out whether that diagnosis was right.
    std::string activation_path;
    // The model's second channel, cached beside the first. Supplying it as a
    // separate file is what makes the controls possible: a shuffled or
    // substituted downbeat channel is a different file through the same
    // binary, so the arms differ in the evidence and in nothing else.
    std::string downbeat_path;
    bool live_bars = false;
    double activation_fps = 50.0;
    // Reproduce when a BeatNet frame becomes available, not only the timestamp
    // written on it. Off by default so earlier external-activation experiments
    // retain their callback cadence; octave-veto cached replay turns it on.
    bool activation_model_timing = false;

    // Recorded frame-release times, one per activation frame, from a
    // --dump-activation run on the same audio. Supersedes the analytic delay
    // in `activation_model_timing`: that modelled only the feature window and
    // ignored the resampler's filter delay and the stream's own buffering, and
    // twenty of twenty RWC recordings failed parity under it.
    std::string activation_emit_path;

    // The model's own frame timestamps, one per activation frame. Supplied
    // rather than reconstructed because the reconstruction is a different
    // double: see the note on `activation_times` where it is printed.
    std::string activation_time_path;

    // The matched-cost comparison policies, one at a time.
    OnlineOctavePolicy online_policy;
    // The same swap, but made by the core itself rather than handed to it: the
    // tracker computes the activation from the audio through its own front end.
    // --live-activation answers "would a better observation help"; this answers
    // "does the thing we would ship actually do it", which is a different
    // question and the one that matters now.
    //
    // Repeatable. Given more than once the core averages the checkpoints over a
    // single front end, which is the arm eval/PREREGISTERED_ensemble_in_core.md
    // is about; the order does not matter to a mean, and it is not recorded.
    std::vector<std::string> model_paths;
    // The offline path's replacement for a beat tracker from 2007. Decodes,
    // resamples to the model's rate, runs the network and picks the peaks —
    // the same code an app would run, not a research approximation of it.
    std::string beat_this_path;
    // The tempo posterior's shape, so the octave choice can be swept over a real
    // annotated corpus without a rebuild per point. Zero means "leave the
    // shipped default alone", the same convention --live-lock already uses.
    // Deliberately not part of the calibration table below: those three are one
    // indivisible backend calibration, and these are independent knobs.
    // What it costs the bar line to move. NaN means "leave the shipped default
    // alone", which is infinity; anything finite turns the decoder on, so the
    // sweep that chose the costs can be repeated against the real resolver
    // rather than against the Python prototype.
    double phase_switch_cost = std::numeric_limits<double>::quiet_NaN();
    double tempo_prior_centre = 0.0;
    double tempo_prior_width = 0.0;
    double tempo_comb_harmonics = 0.0;
    double tempo_comb_decay = 0.0;
    // The same two numbers for the causal path, which carries its own copy of
    // the belief in ParticleFilterConfig. They are separate flags rather than
    // one, because the two paths were calibrated apart and a sweep that moved
    // both at once could not say which one paid.
    double live_prior_centre = 0.0;
    double live_prior_width = 0.0;
    // The filter's observation constants. They were calibrated against spectral
    // flux, in units where "a beat is worth about one"; BeatNet hands it a
    // probability instead, which is a different distribution with the same
    // nominal scale, and nothing re-derived these when the model arrived.
    // Manual mode: the period is pinned and the room is asked only where the
    // beat falls. Not a shippable path on its own — the tempo has to come from
    // somewhere — but it is the only way to measure the ceiling of "the period
    // is supplied by something else", which is a different question from
    // --live-seeded's "the period starts out right and may drift".
    double live_manual_bpm = 0.0;
    double live_prior_rate = 0.0;
    double live_observation_gain = 0.0;
    double live_onset_exponent = 0.0;
    double live_beat_gain = 0.0;
    // How far the cloud is allowed to move and to jump between resamples. These
    // are the filter's tempo agility, and they are exposed because the oracle
    // experiment made agility the question: fed a pulse at every annotated beat,
    // the filter still recalls 92% of GTZAN and 52% of SMC, and the shortfall
    // correlates with how much the recording's tempo moves. Before reaching for
    // somebody else's decoder it is worth knowing whether ours has a setting.
    double live_roughening = 0.0;
    double live_regeneration = -1.0;  // 0 is a meaningful value here

    // Soft octave holding: the filter's tempo prior is re-centred on what an
    // autocorrelation over the activation history makes of the tempo, instead
    // of on a fixed belief about musical tempo. Off in the core by default,
    // so these are how the corpus gets to answer whether it should be.
    // Follows the core rather than overriding it. Initialising this to false
    // silently forced the anchor off on every run that did not pass
    // --live-anchor, so a sweep of "the shipped defaults" reproduced free
    // running exactly and looked like the feature doing nothing.
    bool live_anchor = tiktak::tracking::LiveConfig{}.anchor_tempo;
    double live_anchor_width = 0.0;
    double live_anchor_margin = -1.0;
    double live_anchor_window = 0.0;
    double live_anchor_min_window = 0.0;

    // How often the per-second series below are sampled. One a second is what
    // every experiment before this one was measured at and stays the default,
    // so nothing already published moves.
    //
    // The octave-veto pre-registration needs much finer: it defines a proposal
    // as a run of frames of one sign of `k`, closing after a second at `k = 0`,
    // and a series sampled at exactly that period cannot resolve the thing it
    // is defined by. Fifty is the activation frame rate, which is as fine as
    // this can usefully go — `estimate()` and `tempoFromActivation()` are both
    // const and both read state that only moves once a frame, so asking more
    // often returns the same answer and, crucially, cannot change the run.
    double live_sample_hz = 1.0;
    // A research replay schedule: onset, close, committed BPM. The decoder
    // remains in Python as the single implementation of the registered formula;
    // this file only gives its decisions the real live-core consequences.
    std::string live_anchor_veto_path;
    // The arms of eval/PREREGISTERED_octave_freeze.md that need code. The
    // third, clearing the anchor at a raised threshold, is already reachable
    // through --live-anchor-margin on its own.
    bool live_octave_freeze = false;
    bool live_margin_abstain = false;
    double live_freeze_timeout = 0.0;

    struct Threshold {
        const char* flag;
        double* target;
    };
    const Threshold thresholds[] = {
        {"--salience-min-range", &salience_min_range},
        {"--salience-min-phase-margin", &salience_min_phase},
        {"--salience-min-meter-margin", &salience_min_meter},
    };
    const Threshold tempo_knobs[] = {
        {"--tempo-prior-centre", &tempo_prior_centre},
        {"--tempo-prior-width", &tempo_prior_width},
        {"--tempo-comb-harmonics", &tempo_comb_harmonics},
        {"--tempo-comb-decay", &tempo_comb_decay},
        {"--live-prior-centre", &live_prior_centre},
        {"--live-prior-width", &live_prior_width},
        {"--live-manual-bpm", &live_manual_bpm},
        {"--live-prior-rate", &live_prior_rate},
        {"--live-observation-gain", &live_observation_gain},
        {"--live-onset-exponent", &live_onset_exponent},
        {"--live-beat-gain", &live_beat_gain},
        {"--live-roughening", &live_roughening},
        {"--live-regeneration", &live_regeneration},
        {"--live-anchor-width", &live_anchor_width},
        {"--live-anchor-margin", &live_anchor_margin},
        {"--live-anchor-window", &live_anchor_window},
        {"--live-anchor-min-window", &live_anchor_min_window},
        {"--live-freeze-timeout", &live_freeze_timeout},
        {"--live-sample-hz", &live_sample_hz},
    };

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--salience") == 0) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "--salience needs a file\n");
                return 2;
            }
            salience_path = argv[++i];
            continue;
        }
        if (std::strcmp(argv[i], "--dump-odf") == 0) {
            dump_odf = true;
            continue;
        }
        if (std::strcmp(argv[i], "--dump-activation") == 0) {
            dump_activation = true;
            continue;
        }
        if (std::strcmp(argv[i], "--live-anchor-veto") == 0) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "--live-anchor-veto needs a file\n");
                return 2;
            }
            live = true;
            live_anchor_veto_path = argv[++i];
            continue;
        }
        if (std::strcmp(argv[i], "--live") == 0) {
            live = true;
            continue;
        }
        if (std::strcmp(argv[i], "--live-downbeat") == 0) {
            if (i + 1 >= argc) { std::fprintf(stderr, "--live-downbeat needs a file\n"); return 2; }
            downbeat_path = argv[++i];
            continue;
        }
        if (std::strcmp(argv[i], "--live-bars") == 0) {
            live_bars = true;
            continue;
        }
        if (std::strcmp(argv[i], "--live-activation") == 0) {
            if (i + 1 >= argc) { std::fprintf(stderr, "--live-activation needs a file\n"); return 2; }
            activation_path = argv[++i];
            continue;
        }
        if (std::strcmp(argv[i], "--live-octave-debounce") == 0) {
            if (i + 1 >= argc) { std::fprintf(stderr, "--live-octave-debounce needs seconds\n"); return 2; }
            online_policy.kind = OnlineOctavePolicy::Kind::Debounce;
            online_policy.seconds = std::atof(argv[++i]);
            continue;
        }
        if (std::strcmp(argv[i], "--live-octave-rate-limit") == 0) {
            if (i + 1 >= argc) { std::fprintf(stderr, "--live-octave-rate-limit needs seconds\n"); return 2; }
            online_policy.kind = OnlineOctavePolicy::Kind::RateLimit;
            online_policy.seconds = std::atof(argv[++i]);
            continue;
        }
        if (std::strcmp(argv[i], "--live-octave-ban") == 0) {
            online_policy.kind = OnlineOctavePolicy::Kind::TotalBan;
            continue;
        }
        if (std::strcmp(argv[i], "--activation-times") == 0) {
            if (i + 1 >= argc) { std::fprintf(stderr, "--activation-times needs a file\n"); return 2; }
            activation_time_path = argv[++i];
            continue;
        }
        if (std::strcmp(argv[i], "--activation-emit") == 0) {
            if (i + 1 >= argc) { std::fprintf(stderr, "--activation-emit needs a file\n"); return 2; }
            activation_emit_path = argv[++i];
            continue;
        }
        if (std::strcmp(argv[i], "--activation-model-timing") == 0) {
            activation_model_timing = true;
            continue;
        }
        if (std::strcmp(argv[i], "--beat-this") == 0) {
            if (i + 1 >= argc) { std::fprintf(stderr, "--beat-this needs a model\n"); return 2; }
            beat_this_path = argv[++i];
            continue;
        }
        if (std::strcmp(argv[i], "--live-model") == 0) {
            if (i + 1 >= argc) { std::fprintf(stderr, "--live-model needs a file\n"); return 2; }
            model_paths.emplace_back(argv[++i]);
            continue;
        }
        if (std::strcmp(argv[i], "--activation-fps") == 0) {
            if (i + 1 >= argc) { std::fprintf(stderr, "--activation-fps needs a value\n"); return 2; }
            activation_fps = std::atof(argv[++i]);
            continue;
        }
        if (std::strcmp(argv[i], "--live-lock") == 0) {
            if (i + 1 >= argc) { std::fprintf(stderr, "--live-lock needs a value\n"); return 2; }
            live_lock = std::atof(argv[++i]);
            continue;
        }
        if (std::strcmp(argv[i], "--live-release") == 0) {
            if (i + 1 >= argc) { std::fprintf(stderr, "--live-release needs a value\n"); return 2; }
            live_release = std::atof(argv[++i]);
            continue;
        }
        if (std::strcmp(argv[i], "--live-seeded") == 0) {
            live = true;
            live_seed = true;
            continue;
        }
        if (std::strcmp(argv[i], "--live-anchor") == 0) {
            live = true;
            live_anchor = true;
            continue;
        }
        if (std::strcmp(argv[i], "--live-no-anchor") == 0) {
            live = true;
            live_anchor = false;
            continue;
        }
        if (std::strcmp(argv[i], "--live-octave-freeze") == 0) {
            live = true;
            live_octave_freeze = true;
            continue;
        }
        if (std::strcmp(argv[i], "--live-margin-abstain") == 0) {
            live = true;
            live_margin_abstain = true;
            continue;
        }
        // Parsed here rather than in the table above because that table's
        // convention is "> 0 means set", and zero is a legitimate switch cost:
        // free switching is the bottom of the sweep and has to be reachable.
        if (std::strcmp(argv[i], "--phase-switch-cost") == 0) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "--phase-switch-cost needs a value\n");
                return 2;
            }
            phase_switch_cost = std::atof(argv[++i]);
            if (!(phase_switch_cost >= 0.0)) {
                std::fprintf(stderr,
                             "--phase-switch-cost must be >= 0 (inf pins the "
                             "phase, which is the default)\n");
                return 2;
            }
            continue;
        }
        if (std::strcmp(argv[i], "--bpm") == 0) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "--bpm needs a value\n");
                return 2;
            }
            bpm_hint = std::atof(argv[++i]);
            if (!(bpm_hint > 0.0)) {
                std::fprintf(stderr, "--bpm needs a positive value\n");
                return 2;
            }
            continue;
        }
        if (std::strcmp(argv[i], "--beats") == 0) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "--beats needs a file\n");
                return 2;
            }
            beats_path = argv[++i];
            continue;
        }

        bool tempo_matched = false;
        for (const Threshold& knob : tempo_knobs) {
            if (std::strcmp(argv[i], knob.flag) != 0) continue;
            tempo_matched = true;
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s needs a value\n", knob.flag);
                return 2;
            }
            if (!nonnegativeFinite(argv[++i], *knob.target)) {
                std::fprintf(stderr, "%s must be a finite, non-negative number\n",
                             knob.flag);
                return 2;
            }
            break;
        }
        if (tempo_matched) continue;

        bool matched = false;
        for (const Threshold& threshold : thresholds) {
            if (std::strcmp(argv[i], threshold.flag) != 0) continue;
            matched = true;
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s needs a value\n", threshold.flag);
                return 2;
            }
            if (!nonnegativeFinite(argv[++i], *threshold.target)) {
                std::fprintf(stderr,
                             "%s must be a finite, non-negative number\n",
                             threshold.flag);
                return 2;
            }
            ++calibration_given;
            break;
        }
        if (!matched) positional.push_back(argv[i]);
    }

    const int kCalibrationSize = static_cast<int>(std::size(thresholds));

    // A foreign grid without a foreign scorer would be a lie: the built-in cues
    // are gathered from ODF frames against the tracker's own beats and cannot
    // be recomputed here for someone else's grid, so the metre and phase would
    // still be the original analysis's while the beats printed beside them came
    // from the file.
    if (!beats_path.empty() && salience_path.empty()) {
        std::fprintf(stderr, "--beats replaces the grid the scorer runs on, so it "
                             "needs --salience for that grid too\n");
        return 2;
    }

    if (calibration_given > 0 && salience_path.empty()) {
        std::fprintf(stderr,
                     "the calibration flags only apply with --salience\n");
        return 2;
    }
    // All three or none. They are one calibration: a backend that supplied a
    // range gate in its own units while silently inheriting the cue backend's
    // margins would be judged confident by numbers belonging to a different
    // scorer, and `downbeat_confident` would be a claim about nothing. Half a
    // calibration is the failure this refuses to let anyone make quietly.
    if (calibration_given > 0 && calibration_given < kCalibrationSize) {
        std::fprintf(stderr,
                     "a backend calibration is all %d of --salience-min-range, "
                     "--salience-min-phase-margin and --salience-min-meter-margin, "
                     "or none of them — %d given\n",
                     kCalibrationSize, calibration_given);
        return 2;
    }

    if (positional.empty() || positional.size() > 2) {
        std::fprintf(stderr,
                     "usage: %s <song.mp3|song.wav|song.flac> [--salience <file>"
                     " [calibration]]\n"
                     "       %s <clip.f32> <sample_rate> [--salience <file>"
                     " [calibration]]\n"
                     "  --bpm <value>      track at this tempo instead of estimating it\n"
                     "  --live             use the causal microphone tracker, not the offline one\n"
                     "  --live-seeded      the same, seeded with the offline tempo\n"
                     "  --live-anchor      the same, holding the octave the activation says\n"
                     "  --live-no-anchor   the same, with that holding turned off\n"
                     "  --live-octave-freeze     hold the last confidently chosen octave\n"
                     "                           while --live-anchor-margin is not met\n"
                     "  --live-freeze-timeout <s>  how long that hold may outlive its anchor\n"
                     "  --live-sample-hz <N>  how often to sample the live series;\n"
                     "                     1 is the default and what everything before\n"
                     "                     the octave-veto work was measured at\n"
                     "  --live-anchor-veto <file> onset close committed_bpm rows\n"
                     "  --live-margin-abstain    publish nothing while the margin is weak\n"
                     "                           (a diagnostic bound, not a shippable mode)\n"
                     "  --live-activation <file> [--activation-fps N]\n"
                     "                     drive the particle filter from this activation\n"
                     "  --activation-model-timing  replay BeatNet's frame availability\n"
                     "  --live-model <file>\n"
                     "                     the core computes the activation itself; repeat\n"
                     "                     the flag to average several checkpoints\n"
                     "  calibration: --salience-min-range <v> "
                     "--salience-min-phase-margin <v> "
                     "--salience-min-meter-margin <v>\n"
                     "               (all three together, in the backend's own units)\n",
                     argv[0], argv[0]);
        return 2;
    }

    const std::string path = positional[0];
    const bool raw = positional.size() == 2;

    std::vector<float> samples;
    double rate = 0.0;

    if (raw) {
        rate = std::atof(positional[1].c_str());
        if (rate <= 0.0) {
            std::fprintf(stderr, "bad sample rate\n");
            return 2;
        }
        samples = readRaw(path.c_str());
        if (samples.empty()) {
            std::fprintf(stderr, "cannot read %s as raw float32\n", path.c_str());
            return 1;
        }
    } else {
#if defined(TIKTAK_HAVE_DECODE)
        auto decoder = tiktak::decode::Decoder::open(path.c_str());
        if (!decoder) {
            std::fprintf(stderr, "%s is not a WAV, FLAC or MP3 file\n", path.c_str());
            return 1;
        }
        rate = decoder->info().sample_rate;
        samples.reserve(static_cast<std::size_t>(decoder->info().frames));
        std::vector<float> block(65536);
        for (;;) {
            const std::size_t got = decoder->readMono(block.data(), block.size());
            if (got == 0) break;
            samples.insert(samples.end(), block.begin(), block.begin() + got);
        }
        if (samples.empty()) {
            std::fprintf(stderr, "%s decoded to nothing\n", path.c_str());
            return 1;
        }
#else
        std::fprintf(stderr,
                     "this build has no decoder — rebuild with -DTIKTAK_BUILD_DECODE=ON, "
                     "or pass a raw .f32 file and its sample rate\n");
        return 1;
#endif
    }

    // The C++ analyser rather than the C API, so the per-beat cues and the
    // runner-up tempos are reachable. Growing the product API to expose either
    // would be paying for a research need in a header that ships; the parity
    // tools set this precedent already.
    tiktak::analysis::OfflineConfig config;
    config.odf.sampleRate = rate;
    // The same clamp tt_odf_config_defaults applies. Without it a file below
    // 32 kHz gets mel bands above its own Nyquist and a different onset
    // function from the one the app would compute — the tool would be
    // measuring a configuration that never ships.
    config.odf.melMaxHz = std::min(16000.0, rate * 0.5);
    config.find_downbeats = true;
    config.bpm_hint = bpm_hint;
    if (tempo_prior_centre > 0.0) config.tempo.prior_centre_bpm = tempo_prior_centre;
    if (tempo_prior_width > 0.0) config.tempo.prior_width_octaves = tempo_prior_width;
    if (tempo_comb_harmonics > 0.0) {
        config.tempo.comb_harmonics = static_cast<int>(tempo_comb_harmonics);
    }
    if (tempo_comb_decay > 0.0) config.tempo.comb_weight_decay = tempo_comb_decay;
    if (!std::isnan(phase_switch_cost)) {
        config.downbeat.phase_switch_cost = phase_switch_cost;
    }
    tiktak::analysis::OfflineAnalyzer analyzer(config);

    // Fed in blocks that are not a multiple of the hop, for the same reason
    // dump_beats does it: a decoder hands over whatever size it likes and the
    // framing must not change the answer.
    constexpr std::size_t kBlock = 4099;
    for (std::size_t pos = 0; pos < samples.size(); pos += kBlock) {
        const std::size_t take = std::min(kBlock, samples.size() - pos);
        analyzer.feed(samples.data() + pos, take);
    }
    const tiktak::analysis::OfflineResult analysis = analyzer.finish();
    std::vector<double> beats = analysis.beats;

    // The causal path, driven over the same file. Fed in the same odd-sized
    // blocks, and read the way the shell reads it: ask for the next beat as
    // soon as it comes within the lookahead, and never revise one already
    // handed out. Scoring anything else would be scoring a tracker that does
    // not exist.
    std::vector<double> live_beats;
    std::vector<double> live_bar_positions;
    std::vector<double> live_bar_meters;
    std::vector<double> live_bar_confident;
    int live_beats_per_bar = 0;
    double live_bpm = 0.0;
    double live_confidence = 0.0;
    double live_tempo_spread_octaves = 0.0;
    tiktak::tracking::LiveTracker::Stats live_stats;
    std::vector<double> live_times, live_bpms, live_confidences;
    std::vector<double> live_tempo_spreads_octaves;
    std::vector<double> live_share, live_agreement, live_coincidence;
    // `anchor_*` without the `live_` prefix that the *flags* above use, so the
    // series and the knob that shapes it cannot be confused for each other.
    std::vector<double> anchor_bpm, anchor_confidence, anchor_margin;
    AnchorVetoSchedule anchor_veto_schedule;
    if (!live_anchor_veto_path.empty()) {
        std::string complaint;
        if (!readVetoSchedule(live_anchor_veto_path.c_str(), anchor_veto_schedule,
                              complaint)) {
            std::fprintf(stderr, "%s\n", complaint.c_str());
            return 1;
        }
    }

    // The whole live path, driven by an activation from outside instead of by
    // the built-in onset function. The same tracker, the same hysteresis and
    // the same publishing rules — only the evidence differs, so the grids that
    // come out are comparable with --live's.
    std::vector<double> activation;
    if (!activation_path.empty()) {
        std::string complaint;
        if (!readSalience(activation_path.c_str(), activation, complaint)) {
            std::fprintf(stderr, "%s\n", complaint.c_str());
            return 1;
        }
        live = true;
    }
    std::vector<double> downbeat_activation;
    if (!downbeat_path.empty()) {
        std::string complaint;
        if (!readSalience(downbeat_path.c_str(), downbeat_activation, complaint)) {
            std::fprintf(stderr, "%s\n", complaint.c_str());
            return 1;
        }
        if (activation_path.empty()) {
            std::fprintf(stderr,
                         "--live-downbeat is the second channel of a cached "
                         "activation and needs --live-activation for the first\n");
            return 1;
        }
        if (downbeat_activation.size() != activation.size()) {
            std::fprintf(stderr,
                         "--live-downbeat has %zu frame(s) against %zu in "
                         "--live-activation; they are two channels of one "
                         "stream and a mismatch is a different recording\n",
                         downbeat_activation.size(), activation.size());
            return 1;
        }
    }
    std::vector<double> activation_emit;
    if (!activation_emit_path.empty()) {
        std::string complaint;
        if (!readSalience(activation_emit_path.c_str(), activation_emit, complaint)) {
            std::fprintf(stderr, "%s\n", complaint.c_str());
            return 1;
        }
        if (activation_emit.size() != activation.size()) {
            std::fprintf(stderr,
                         "--activation-emit has %zu times for %zu frames\n",
                         activation_emit.size(), activation.size());
            return 1;
        }
    }
    std::vector<double> activation_frame_times;
    if (!activation_time_path.empty()) {
        std::string complaint;
        if (!readSalience(activation_time_path.c_str(), activation_frame_times,
                          complaint)) {
            std::fprintf(stderr, "%s\n", complaint.c_str());
            return 1;
        }
        if (activation_frame_times.size() != activation.size()) {
            std::fprintf(stderr,
                         "--activation-times has %zu times for %zu frames\n",
                         activation_frame_times.size(), activation.size());
            return 1;
        }
    }
    std::vector<double> model_downbeats;
    // Held by value in a deque rather than a vector: BeatNetWeights owns the
    // storage its member pointers point into, so a vector reallocating on push
    // would leave every earlier set pointing at freed bytes. A deque never
    // moves what it already holds.
    std::deque<tiktak::ml::BeatNetWeights> model_weights;
    std::vector<const tiktak::ml::BeatNetWeights*> model_refs;
    for (const std::string& path : model_paths) {
        std::vector<unsigned char> blob;
        if (!readBytes(path.c_str(), blob)) {
            std::fprintf(stderr, "cannot read %s\n", path.c_str());
            return 1;
        }
        tiktak::ml::BeatNetWeights& weights = model_weights.emplace_back();
        if (!weights.load(blob.data(), blob.size())) {
            std::fprintf(stderr, "%s is not a weight file this build can run\n",
                         path.c_str());
            return 1;
        }
        model_refs.push_back(&weights);
        live = true;
    }
    if (live) {
        tiktak::tracking::LiveConfig live_config;
        live_config.odf = config.odf;
        if (live_lock > 0.0) live_config.lock_confidence = live_lock;
        if (live_release > 0.0) live_config.release_confidence = live_release;
        if (live_prior_centre > 0.0) {
            live_config.filter.prior_centre_bpm = live_prior_centre;
        }
        if (live_prior_width > 0.0) {
            live_config.filter.prior_width_octaves = live_prior_width;
        }
        if (live_prior_rate > 0.0) live_config.filter.prior_rate = live_prior_rate;
        if (live_observation_gain > 0.0) {
            live_config.filter.observation_gain = live_observation_gain;
        }
        if (live_onset_exponent > 0.0) {
            live_config.filter.onset_exponent = live_onset_exponent;
        }
        if (live_beat_gain > 0.0) live_config.filter.beat_gain = live_beat_gain;
        if (live_roughening > 0.0) {
            live_config.filter.roughening_octaves = live_roughening;
        }
        if (live_regeneration >= 0.0) {
            live_config.filter.regeneration = live_regeneration;
        }
        live_config.anchor_tempo = live_anchor;
        if (!live_anchor_veto_path.empty()) {
            live_config.anchor_bpm_resolver = &AnchorVetoSchedule::callback;
            live_config.anchor_bpm_resolver_context = &anchor_veto_schedule;
        } else if (online_policy.kind != OnlineOctavePolicy::Kind::None) {
            // Mutually exclusive: one resolver, and a schedule combined with a
            // live policy would be two policies reported as one.
            live_config.anchor_bpm_resolver = &OnlineOctavePolicy::callback;
            live_config.anchor_bpm_resolver_context = &online_policy;
        }
        if (live_anchor_width > 0.0) {
            live_config.anchor_width_octaves = live_anchor_width;
        }
        if (live_anchor_margin >= 0.0) {
            live_config.anchor_octave_margin = live_anchor_margin;
        }
        live_config.anchor_octave_freeze = live_octave_freeze;
        live_config.bar_tracking = live_bars;
        live_config.anchor_margin_abstain = live_margin_abstain;
        if (live_freeze_timeout > 0.0) {
            live_config.anchor_freeze_timeout_sec = live_freeze_timeout;
        }
        if (live_anchor_window > 0.0) {
            live_config.activation_tempo.window_sec = live_anchor_window;
        }
        if (live_anchor_min_window > 0.0) {
            live_config.activation_tempo.min_window_sec = live_anchor_min_window;
        }
        tiktak::tracking::LiveTracker tracker =
            model_refs.empty()
                ? tiktak::tracking::LiveTracker(live_config)
                : tiktak::tracking::LiveTracker(live_config, model_refs.data(),
                                                model_refs.size());
        if (live_seed && analysis.bpm > 0.0) {
            tracker.seedTempo(analysis.bpm);
        }
        // After the seed, deliberately: pinning the period supersedes
        // concentrating the cloud on one, and asking for both should behave as
        // the stronger of the two rather than as an ordering accident.
        if (live_manual_bpm > 0.0) tracker.setManualTempo(live_manual_bpm);

        // A device-sized buffer, not the odd block above. takeBeat only hands
        // over a beat once it is within the lookahead of now, so the polling
        // rate is part of the algorithm: poll every 4099 samples and most
        // beats fall between checks and are simply never played. A real shell
        // polls once per audio callback, and scoring anything slower would be
        // measuring the harness rather than the tracker.
        constexpr double kLookahead = 0.05;
        double now = 0.0;
        const double sample_period =
            live_sample_hz > 0.0 ? 1.0 / live_sample_hz : 1.0;
        double next_sample = sample_period;
        const auto poll = [&]() {
            double beat = 0.0;
            // Outside the sampling guard, deliberately. Beats are handed out on
            // the block clock and must not depend on how often the series are
            // read, which is what makes --live-sample-hz observational: the
            // beat list at 50 Hz has to equal the beat list at 1 Hz.
            while (tracker.takeBeat(now, kLookahead, &beat)) {
                live_beats.push_back(beat);
                // One per beat, in the same order, so the two columns can be
                // read together. -1 is "nothing decided yet", which is a state
                // and not a position.
                live_bar_positions.push_back(
                    static_cast<double>(tracker.barPosition()));
                // The held metre as it stood when this beat was handed out. The
                // final one alone cannot say whether the answer was reached
                // early and kept or arrived on the last bar, and it cannot see
                // a decoder that flickers through every candidate on its way.
                live_bar_meters.push_back(
                    static_cast<double>(tracker.beatsPerBar()));
                live_bar_confident.push_back(tracker.meterConfident() ? 1.0 : 0.0);
            }
            // The total ban is worded "no octave change after first lock", so
            // the policy needs the publishing state. Latched with the shipped
            // hysteresis and never released: what §7 asks about is whether the
            // tracker has ever settled, not whether it is settled now.
            if (online_policy.kind == OnlineOctavePolicy::Kind::TotalBan &&
                !online_policy.locked &&
                tracker.estimate(now).confidence >= live_config.lock_confidence) {
                online_policy.locked = true;
            }
            if (now >= next_sample) {
                // Which of confidence's three factors is low is the diagnosis,
                // and the product alone cannot say.
                const tiktak::tracking::BeatEstimate e = tracker.estimate(now);
                live_times.push_back(now);
                live_bpms.push_back(e.bpm);
                live_confidences.push_back(e.confidence);
                live_tempo_spreads_octaves.push_back(e.tempo_spread_octaves);
                live_share.push_back(e.cluster_share);
                live_agreement.push_back(e.phase_agreement);
                live_coincidence.push_back(e.onset_coincidence);
                // The anchor's own answer, beside the filter's. Without both,
                // a run at the wrong metrical level cannot say which half of
                // the path chose it — the estimator that re-aims the prior
                // every second, or the filter that was aimed and went
                // elsewhere. Those need opposite work, and the product of the
                // two is all the beat list shows.
                const tiktak::tracking::ActivationTempoEstimate a =
                    tracker.tempoFromActivation();
                anchor_bpm.push_back(a.bpm);
                anchor_confidence.push_back(a.confidence);
                anchor_margin.push_back(a.octave_margin);
                next_sample += sample_period;
            }
        };

        if (!activation.empty()) {
            // Driven by a block clock, not by the frame index, so that this
            // path and the audio path below ask takeBeat the same question at
            // the same moments and differ only in where the evidence came from.
            //
            // The first version of this loop polled once per activation frame
            // and let `now` jump a frame at a time. That is a different
            // cadence from a device's, and takeBeat is sensitive to it: while
            // coasting it hands out one beat per call until it catches up with
            // now + lookahead, so a coarser clock flushes out several beats
            // where a device would have taken one. It inflated the beat count
            // roughly threefold — visible afterwards as beats 50 ms apart,
            // which is the lookahead, not a tempo — without much changing
            // which beats were right. The accuracy measured through it stands;
            // the count of beats emitted through it did not.
            const double frame_sec = 1.0 / activation_fps;
            std::size_t next_frame = 0;
            if (activation_model_timing) {
                // A frame is timestamped at its start but is not available
                // until the model has released it, and the two differ by more
                // than the feature window: the resampler carries a filter delay
                // and the stream buffers whatever does not fill a hop. The
                // analytic form below modelled only the window, released frames
                // early, and failed parity on twenty RWC recordings out of
                // twenty — identical observations at identical timestamps, but
                // a different amount of them accumulated whenever the block
                // clock reached `estimate()` and `takeBeat()`.
                //
                // With --activation-emit the schedule is not modelled at all.
                // It is replayed from what a model fed the same 512-sample
                // blocks actually did. The observation timestamp is still the
                // registered frame time; only availability moves.
                constexpr double kAvailabilityDelay =
                    static_cast<double>(tiktak::ml::BeatNetFeatures::kFrameSize) /
                    tiktak::ml::BeatNetFeatures::kModelRate;
                const bool recorded = !activation_emit.empty();
                const bool recorded_times = !activation_frame_times.empty();
                const auto observation_time = [&](std::size_t frame) {
                    return recorded_times
                               ? activation_frame_times[frame]
                               : static_cast<double>(frame) * frame_sec;
                };
                double block_index = 0.0;
                for (std::size_t pos = 0; pos < samples.size(); pos += kLiveBlock) {
                    const std::size_t take =
                        std::min(kLiveBlock, samples.size() - pos);
                    now += static_cast<double>(take) / rate;
                    block_index += 1.0;
                    anchor_veto_schedule.decision_time_sec = now;
                    online_policy.decision_time_sec = now;
                    while (next_frame < activation.size() &&
                           (recorded
                                ? activation_emit[next_frame] <= block_index
                                : static_cast<double>(next_frame) * frame_sec +
                                      kAvailabilityDelay <= now)) {
                        if (downbeat_activation.empty()) {
                            tracker.observe(observation_time(next_frame),
                                            activation[next_frame]);
                        } else {
                            tracker.observe(observation_time(next_frame),
                                            activation[next_frame],
                                            downbeat_activation[next_frame]);
                        }
                        ++next_frame;
                    }
                    poll();
                }
            } else {
                const double block_sec = static_cast<double>(kLiveBlock) / rate;
                const double last_sec =
                    static_cast<double>(activation.size()) * frame_sec;
                for (double clock = 0.0; clock < last_sec; clock += block_sec) {
                    anchor_veto_schedule.decision_time_sec = clock;
                    online_policy.decision_time_sec = clock;
                    while (next_frame < activation.size() &&
                           static_cast<double>(next_frame) * frame_sec <= clock) {
                        if (downbeat_activation.empty()) {
                            tracker.observe(
                                static_cast<double>(next_frame) * frame_sec,
                                activation[next_frame]);
                        } else {
                            tracker.observe(
                                static_cast<double>(next_frame) * frame_sec,
                                activation[next_frame],
                                downbeat_activation[next_frame]);
                        }
                        ++next_frame;
                    }
                    now = clock;
                    poll();
                }
            }
        } else {
            for (std::size_t pos = 0; pos < samples.size(); pos += kLiveBlock) {
                const std::size_t take = std::min(kLiveBlock, samples.size() - pos);
                anchor_veto_schedule.decision_time_sec =
                    now + static_cast<double>(take) / rate;
                online_policy.decision_time_sec =
                    now + static_cast<double>(take) / rate;
                tracker.process(now, samples.data() + pos, take);
                now += static_cast<double>(take) / rate;
                poll();
            }
        }
        const tiktak::tracking::BeatEstimate final = tracker.estimate(now);
        live_bpm = final.bpm;
        live_confidence = final.confidence;
        live_tempo_spread_octaves = final.tempo_spread_octaves;
        live_stats = tracker.stats();
        live_beats_per_bar = tracker.beatsPerBar();
        beats = live_beats;
    }

    bool beats_replaced = false;
    const char* beats_source = "tracker";

#if TIKTAK_HAVE_ML
    if (!beat_this_path.empty()) {
        tiktak::ml::BeatThisSession session;
        if (!session.open(beat_this_path)) {
            std::fprintf(stderr, "%s\n", session.reason().c_str());
            return 1;
        }

        const tiktak::dsp::Resampler resampler(rate, tiktak::ml::BeatThisFeatures::kModelRate);
        const std::vector<float> model_audio = resampler.apply(samples.data(), samples.size());

        tiktak::ml::BeatThisFeatures features;
        const std::vector<float> mel =
            features.compute(model_audio.data(), model_audio.size());
        const std::size_t mels = tiktak::ml::BeatThisFeatures::kMels;
        const auto activations = session.run(mel.data(), mel.size() / mels, mels);
        const auto grid = tiktak::ml::pickBeats(activations.beat.data(),
                                                activations.downbeat.data(),
                                                activations.beat.size());
        if (grid.beats.size() < 2) {
            std::fprintf(stderr, "the model found %zu beat(s); a grid needs at least 2\n",
                         grid.beats.size());
            return 1;
        }
        beats = grid.beats;
        model_downbeats = grid.downbeats;
        beats_replaced = true;
        beats_source = "beat_this";
    }
#endif

    if (!beats_path.empty()) {
        std::vector<double> supplied;
        std::string complaint;
        if (!readSalience(beats_path.c_str(), supplied, complaint)) {
            std::fprintf(stderr, "%s: %s\n", beats_path.c_str(), complaint.c_str());
            return 1;
        }
        if (supplied.size() < 2) {
            std::fprintf(stderr, "%s holds %zu beat time(s); a grid needs at least 2\n",
                         beats_path.c_str(), supplied.size());
            return 1;
        }
        // Sorted and strictly increasing, because everything downstream indexes
        // bars by position in this list. An unsorted grid would not fail, it
        // would silently answer about a different piece of music.
        for (std::size_t i = 1; i < supplied.size(); ++i) {
            if (!(supplied[i] > supplied[i - 1])) {
                std::fprintf(stderr,
                             "%s: beat times must strictly increase (%zu: %.17g <= %.17g)\n",
                             beats_path.c_str(), i, supplied[i], supplied[i - 1]);
                return 1;
            }
        }
        beats = supplied;
        beats_replaced = true;
        beats_source = "file";
    }

    std::vector<double> downbeats = analysis.downbeats;
    // The model has its own opinion about bar lines, and when it was the one
    // that found the beats it is the one to ask. The resolver still runs — its
    // metre and margins are what the rest of this tool reports — but the bar
    // lines themselves come from the head that was trained to find them.
    if (!model_downbeats.empty()) downbeats = model_downbeats;
    int beats_per_bar = analysis.beats_per_bar;
    double strength = analysis.downbeat_strength;
    double phase_margin = analysis.downbeat_phase_margin;
    double meter_margin = analysis.downbeat_meter_margin;
    bool confident = analysis.downbeat_confident;

    if (!salience_path.empty()) {
        std::vector<double> salience;
        std::string salience_error;
        if (!readSalience(salience_path.c_str(), salience, salience_error)) {
            std::fprintf(stderr, "%s: %s\n", salience_path.c_str(),
                         salience_error.c_str());
            return 1;
        }
        if (salience.size() != beats.size()) {
            std::fprintf(stderr,
                         "%s holds %zu value(s) but the analysis found %zu beat(s) — "
                         "one number per beat, in beat order\n",
                         salience_path.c_str(), salience.size(), beats.size());
            return 1;
        }
        // The C API offers no way to override the downbeat configuration, so
        // this research seam reaches the resolver directly. All three
        // calibration numbers travel together: the resolver no longer rescales
        // arbitrary backend output, so margins are in the backend's own units
        // and `downbeat_confident` means nothing unless judged by that
        // backend's own thresholds.
        tiktak::analysis::DownbeatConfig db_config;
        db_config.min_salience_range = salience_min_range;
        db_config.min_phase_margin = salience_min_phase;
        db_config.min_meter_margin = salience_min_meter;
        if (!std::isnan(phase_switch_cost)) {
            db_config.phase_switch_cost = phase_switch_cost;
        }
        const tiktak::analysis::DownbeatResult resolved =
            tiktak::analysis::resolveMeter(salience, beats, db_config);
        downbeats = resolved.downbeats;
        beats_per_bar = resolved.beats_per_bar;
        strength = resolved.strength;
        phase_margin = resolved.phase_margin;
        meter_margin = resolved.meter_margin;
        confident = resolved.confident(db_config.min_phase_margin,
                                       db_config.min_meter_margin);
    }

    std::printf("{\n");
    std::printf("  \"path\": \"%s\",\n", escape(path).c_str());
    std::printf("  \"salience_source\": \"%s\",\n",
                salience_path.empty() ? "cues" : "file");
    std::printf("  \"beats_source\": \"%s\",\n",
                beats_source);
    std::printf("  \"sample_rate\": %.17g,\n", rate);
    std::printf("  \"duration_sec\": %.17g,\n", static_cast<double>(samples.size()) / rate);
    std::printf("  \"bpm\": %.17g,\n", finiteOrZero(analysis.bpm));
    std::printf("  \"confidence\": %.17g,\n",
                finiteOrZero(analysis.tempo_confidence));
    std::printf("  \"beat_objective_per_beat\": %.9g,\n",
                finiteOrZero(analysis.beat_objective_per_beat));
    std::printf("  \"beats_causal\": %s,\n", live ? "true" : "false");
    if (dump_odf) {
        const std::vector<double>& odf_values = analyzer.odfValues();
        const std::vector<double>& odf_times = analyzer.frameTimes();
        std::printf("  \"odf\": [");
        for (std::size_t i = 0; i < odf_values.size(); ++i) {
            std::printf("%s%.6g", i == 0 ? "" : ",", odf_values[i]);
        }
        std::printf("],\n  \"odf_times\": [");
        for (std::size_t i = 0; i < odf_times.size(); ++i) {
            std::printf("%s%.6f", i == 0 ? "" : ",", odf_times[i]);
        }
        std::printf("],\n");
    }
    if (dump_activation && !model_refs.empty()) {
        // A fresh pass, and fresh state: the tracker's own instance has been
        // run to the end of the file and an LSTM that has already seen it would
        // not answer the same way twice. Whatever the tracker was given, the
        // dump is of the same thing — one checkpoint or the mean of several —
        // so an activation dumped here can be handed back through
        // --live-activation and reproduce the run it came from.
        tiktak::ml::BeatNetActivation activation_pass(rate, model_refs.data(),
                                                      model_refs.size());
        std::vector<double> at, beat_p, downbeat_p, emit;
        at.reserve(samples.size() / 441);
        beat_p.reserve(at.capacity());
        downbeat_p.reserve(at.capacity());
        emit.reserve(at.capacity());
        // Fed in the tracker's own 512-sample blocks, not the whole file at
        // once, and each frame is stamped with the block clock it was *released*
        // on rather than the time it describes.
        //
        // Those are not the same instant and the gap is not analytic. A frame
        // cannot exist before its 64 ms window does, but on top of that the
        // resampler carries its own filter delay and the feature stream buffers
        // whatever does not fill a hop. Modelling only the window left the
        // replay releasing frames early, which changed nothing the filter
        // observed and everything about what `estimate()` and `takeBeat()` had
        // accumulated when the block clock reached them: measured on twenty RWC
        // recordings, **none** reproduced the live core, with beat counts off by
        // as much as 116 against 80. Recording the schedule removes the model
        // and the residual with it.
        {
            // A block *index*, not a block time. The first version recorded the
            // clock and printed it at nine significant digits, which is enough
            // to round trip a float and not a double: the seventh boundary is
            // 0.081269841269..., prints as 0.0812698413, and comes back
            // fractionally *larger* than the clock it has to be compared with.
            // One frame then arrives a block late, and on a filter that is a
            // different run — visible as an activation replay whose BPM already
            // disagreed at 0.08 s. An integer has no such edge.
            std::size_t block_index = 0;
            for (std::size_t pos = 0; pos < samples.size(); pos += kLiveBlock) {
                const std::size_t take = std::min(kLiveBlock, samples.size() - pos);
                ++block_index;
                activation_pass.process(samples.data() + pos, take,
                                        [&](double t, double beat, double downbeat) {
                                            at.push_back(t);
                                            beat_p.push_back(beat);
                                            downbeat_p.push_back(downbeat);
                                            emit.push_back(
                                                static_cast<double>(block_index));
                                        });
            }
        }
        const auto series = [](const char* name, const std::vector<double>& values,
                               const char* format) {
            std::printf("  \"%s\": [", name);
            for (std::size_t i = 0; i < values.size(); ++i) {
                std::printf(i == 0 ? "" : ",");
                std::printf(format, values[i]);
            }
            std::printf("],\n");
        };
        // Seventeen digits, because the replay observes at exactly these
        // instants. `BeatNetFeatures::frameTimeSec()` is
        // `(n * 441.0) / 22050.0` — an exact product and one division — while a
        // replay reconstructing `n * (1.0 / 50.0)` multiplies an already
        // rounded 0.02 by n and drifts from it in the last bits. The filter
        // integrates over inter-observation gaps, so that is enough to be a
        // different run, and it was the last of three reasons activation
        // replay could not reproduce the live core.
        series("activation_times", at, "%.17g");
        // Which 512-sample block released each frame, counted from one. Handed
        // back through --activation-emit so the replay releases frames when the
        // model did rather than when their timestamps say they begin.
        series("activation_emit", emit, "%.0f");
        // Seventeen, not nine. The note this replaces said BeatNet returns
        // floats and that nine digits round trip a float exactly. Both halves
        // are true and the conclusion is not: `beatnet.hpp` computes
        // `(double)p[0] + (double)p[1]`, scaled by `1 / models`, so what leaves
        // the callback is a genuine double whose value is generally not
        // representable as a float. Nine digits therefore truncated it, the
        // truncation moved the particle weights, and the byte-parity gate on
        // activation replay could not pass — 0 of 20 RWC recordings, with beat
        // counts as far apart as 116 against 74. Seventeen round trips a double.
        series("activation_beat", beat_p, "%.17g");
        series("activation_downbeat", downbeat_p, "%.17g");
    }
    if (live) {
        std::printf("  \"live_bpm\": %.17g,\n", finiteOrZero(live_bpm));
        std::printf("  \"live_confidence\": %.9g,\n", finiteOrZero(live_confidence));
        std::printf("  \"live_tempo_spread_octaves\": %.9g,\n",
                    finiteOrZero(live_tempo_spread_octaves));
        // Beats the tracker decided on but could not play, because by the time
        // it was asked the beat was already behind the clock. Free to report
        // and the only way to tell a tracker that lost the grid from one that
        // found it too late — which look identical in the beat list.
        std::printf("  \"live_frames\": %zu,\n", live_stats.frames);
        std::printf("  \"live_gated\": %zu,\n", live_stats.gated);
        std::printf("  \"live_beats\": %zu,\n", live_stats.beats);
        std::printf("  \"live_beats_late\": %zu,\n", live_stats.beats_late);
        std::printf("  \"live_anchor_veto_intervals\": %zu,\n",
                    anchor_veto_schedule.intervals.size());
        std::printf("  \"live_anchor_veto_frames\": %zu,\n",
                    anchor_veto_schedule.applied_frames);
        std::printf("  \"live_seeded\": %s,\n", live_seed ? "true" : "false");
        const auto column = [](const char* name, const std::vector<double>& values) {
            std::printf("  \"%s\": [", name);
            for (std::size_t i = 0; i < values.size(); ++i) {
                std::printf("%s%.4g", i == 0 ? "" : ",", values[i]);
            }
            std::printf("],\n");
        };
        column("live_times", live_times);
        column("live_bpms", live_bpms);
        column("live_confidences", live_confidences);
        column("live_tempo_spreads_octaves", live_tempo_spreads_octaves);
        column("live_share", live_share);
        column("live_agreement", live_agreement);
        column("live_coincidence", live_coincidence);
        column("live_anchor_bpm", anchor_bpm);
        column("live_anchor_confidence", anchor_confidence);
        column("live_anchor_margin", anchor_margin);
        // Per beat rather than per sample, and therefore not affected by
        // --live-sample-hz: a bar position belongs to a beat, not to a clock.
        std::printf("  \"live_beats_per_bar\": %d,\n", live_beats_per_bar);
        column("live_bar_positions", live_bar_positions);
        column("live_bar_meters", live_bar_meters);
        column("live_bar_confident", live_bar_confident);
    }
    std::printf("  \"beats_per_bar\": %d,\n", beats_per_bar);
    std::printf("  \"downbeat_strength\": %.17g,\n", finiteOrZero(strength));
    std::printf("  \"downbeat_phase_margin\": %.17g,\n",
                finiteOrZero(phase_margin));
    std::printf("  \"downbeat_meter_margin\": %.17g,\n",
                finiteOrZero(meter_margin));
    std::printf("  \"downbeat_confident\": %s,\n", confident ? "true" : "false");
    // The three cues, one value per beat, in the order the beats are printed.
    // Measured on real music the metre is usually right and the phase usually
    // wrong, which is a claim about these numbers; printing them is what lets
    // that be investigated, and what lets cue weights be swept outside the core
    // by recombining them and feeding the result back through --salience.
    printTimes("cue_low", cueColumn(analysis.beat_features, &BeatFeature::low), false);
    printTimes("cue_accent", cueColumn(analysis.beat_features, &BeatFeature::accent), false);
    printTimes("cue_harmony",
               cueColumn(analysis.beat_features, &BeatFeature::harmonic_change), false);

    // Runner-up tempos, strongest first. An octave-away runner-up with a
    // similar strength is the classic ambiguity, and it is the difference
    // between a tracker that is wrong and one that was handed a coin toss.
    std::printf("  \"tempo_candidates\": [");
    tiktak::analysis::TempoCandidate candidates[8];
    const std::size_t found = analyzer.tempoCandidates(candidates, 8);
    for (std::size_t i = 0; i < found; ++i) {
        std::printf("%s{\"bpm\": %.17g, \"strength\": %.17g}", i ? ", " : "",
                    finiteOrZero(candidates[i].bpm),
                    finiteOrZero(candidates[i].strength));
    }
    std::printf("],\n");

    printTimes("beats", beats, false);
    printTimes("downbeats", downbeats, true);
    std::printf("}\n");

    return 0;
}
