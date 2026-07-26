#pragma once

#include <cstddef>
#include <vector>

namespace tiktak::analysis {

// What one beat looks like, reduced to the three things that distinguish a bar
// line from the beats around it.
//
// Beat-synchronous by construction: the beat grid is already known when this is
// built, so a bar is a pattern over *beats* rather than over time. That is the
// whole reason downbeat detection is a separate stage from beat tracking and
// not a harder version of it — the hard part, finding the beats, is done.
struct BeatFeature {
    double time_sec = 0.0;
    // Onset energy below OdfConfig::lowBandHz at this beat. The kick drum is
    // the plainest downbeat marker there is in most popular music.
    double low = 0.0;
    // Full-band onset energy: how hard the beat is struck at all.
    double accent = 0.0;
    // Cosine distance between this beat's pitch class profile and the previous
    // beat's, in [0, 1]. Chords change on bar lines.
    double harmonic_change = 0.0;
};

struct MeterCandidate {
    int beats_per_bar = 0;
    // Multiplies the meter's score. Not a probability — a thumb on the scale
    // for breaking ties, in the same spirit as the tempo estimator's log-normal
    // prior over BPM, and for the same reason: the evidence is genuinely
    // ambiguous between a meter and its divisors, so something has to break it
    // and a stated prior is better than an accident of arithmetic.
    double prior = 1.0;
};

struct DownbeatConfig {
    // Bar lengths considered, strongest prior first.
    //
    // 4, 3 and 2 cover very nearly all of the repertoire this app is for. 6 is
    // here for 6/8 counted in eighths, and carries the weakest prior because
    // it is only ever distinguishable from 3 by which of the two accents in the
    // bar is bigger — a genuinely subtle question that this stage should be
    // allowed to decline rather than answer badly.
    //
    // The priors are ratios against 4/4 and encode how common each meter is,
    // nothing more. They matter only when the audio is close to a tie.
    std::vector<MeterCandidate> meters = {{4, 1.0}, {3, 0.9}, {2, 0.75}, {6, 0.6}};

    // Relative weight of each cue in the per-beat score.
    double low_weight = 1.0;

    // Off by default, and this is the one number here that was decided by
    // measurement rather than by argument.
    //
    // How hard a beat is struck sounds like a downbeat cue and is very nearly
    // its opposite. In the ordinary rock and pop pattern — kick on one, snare
    // on the backbeat — the snare is broadband and the kick is not, so the
    // full-band onset function peaks on beats two and four while the low band
    // peaks on one. Measured on the reference clip: accent averages 1.2 on the
    // backbeats against 0.68 on the downbeat, while the low band averages 1.13
    // on the downbeat against 0.73. The two cues point at different beats, and
    // the accent points at the wrong one. At a weight of 0.5 it won, and the
    // analysis confidently accented beat four.
    //
    // It is not wrong everywhere — in orchestral or choral music the downbeat
    // really is the loudest event — so it stays as a parameter for callers who
    // know their material. It is simply not a safe default.
    double accent_weight = 0.0;

    // Harmony is *not* standardised before being weighted, unlike the onset
    // cues, because it does not need to be: a cosine distance between two pitch
    // class profiles already means something absolute, where an onset value
    // means nothing until compared with the rest of the piece. Standardising it
    // would be actively wrong — on a drums-only track the chord "changes" by
    // 0.01 from beat to beat, pure noise, and forcing that to unit variance
    // would promote it to an equal vote with the kick drum.
    double harmony_weight = 1.0;

    // Where a beat's onset energy is collected from, as a fraction of the gap
    // to the next beat. Slightly before, because the tracker's beat time can
    // land just after an attack; well short of the next beat, because energy
    // from the following beat must not count towards this one.
    double window_before = 0.15;
    double window_after = 0.45;

    // A meter is only offered if the piece holds at least this many bars of it.
    // Below that there is no repetition to see and any answer is the first
    // accident in the audio.
    int min_bars = 3;

    // How convincing the answer has to be before a caller should accent
    // anything. They live in the config rather than at the call site so that
    // research/eval sweeps the same numbers the app uses, not a copy of them.
    //
    // **Provisional, on synthetic material only.** A metre threshold above the
    // two observed wrong answers took the wrong rate from 14% to zero on seven
    // held-out clips; 0.40 is a deliberately conservative round value inside
    // that separation, not an optimum inferred from six validation clips.
    // Six clips is not a calibration and the material has no chord changes in
    // it at all, so the harmony cue had no say. Real recordings will move both
    // numbers. The phase threshold has had no such check and remains a plain
    // guess.
    //
    // A consequence worth knowing rather than discovering: a metre that divides
    // another is inherently less separable, because the longer bar fits the
    // shorter pattern exactly. Two-beat bars score a metre margin near zero
    // against four and are usually withheld. That is the measure being honest —
    // 4/4 really does fit a 2/4 accent pattern — but it means 2/4 mostly does
    // not get an automatic accent. `--beats` can assert the bar length, but it
    // still needs a phase supported by the audio rather than inventing one.
    double min_phase_margin = 0.25;
    double min_meter_margin = 0.40;

    bool valid() const;
};

struct MeterScore {
    int beats_per_bar = 0;
    int phase = 0;       // index of the first downbeat within the beat list
    double score = 0.0;  // best contrast for this meter, prior applied
};

struct DownbeatResult {
    // Times of the bar lines, a subset of the beats handed in. Empty when no
    // meter could be decided.
    std::vector<double> downbeats;
    // Beats per bar, or 0 when undecided. `phase` is the index into the beat
    // list of the first downbeat.
    int beats_per_bar = 0;
    int phase = 0;

    // How far the winning answer stands above the alternatives, in standard
    // deviations of the per-beat score. Three separate doubts, kept separate
    // because they fail differently and a caller should be able to tell which
    // one it has:
    //
    //   `strength` is the winner's own contrast — how much louder, in the
    //   combined cue, the chosen bar lines are than everything else. Near zero
    //   means the audio simply has no bar-level pattern, and the honest display
    //   is no bar lines at all.
    //
    //   `phase_margin` is how far ahead of the next best *phase of the same
    //   meter* the answer is. A large strength with a small phase margin means
    //   the bars are clear but which beat starts them is a coin toss — the
    //   failure a listener notices immediately, since a metronome accenting
    //   beat 3 is worse than one accenting nothing.
    //
    //   `meter_margin` is how far ahead of the best *other meter* it is. This
    //   is a genuinely different question, and conflating the two was a real
    //   bug: a piece scored in three can be perfectly unambiguous about which
    //   beat starts its bars — a large phase margin — while four fits it very
    //   nearly as well. The phase margin cannot see that, because every rival
    //   it considers has already accepted the wrong meter.
    //
    // A caller that accents downbeats needs *both* to be convincing. See
    // DownbeatResult::confident().
    double strength = 0.0;
    double phase_margin = 0.0;
    double meter_margin = 0.0;

    // Every meter considered, best first, for diagnostics and for a UI that
    // wants to offer "no, it is a waltz".
    std::vector<MeterScore> candidates;

    // Whether the answer is worth acting on: a bar-level pattern exists, the
    // beat it starts on is settled, and no other meter is nearly as good.
    //
    // The thresholds are the caller's, because the cost of being wrong is the
    // caller's. Both default to placeholders rather than calibrated numbers —
    // see research/eval/README.md, which is what will replace them.
    bool confident(double min_phase_margin = 0.25,
                   double min_meter_margin = 0.40) const {
        return beats_per_bar > 0 && !downbeats.empty() &&
               phase_margin >= min_phase_margin && meter_margin >= min_meter_margin;
    }
};

// The per-beat cues, gathered from ODF frames and their pitch class profiles.
//
// `chroma` is ChromaFilterbank::kBins values per frame, laid out frame-major,
// or nullptr — in which case harmonic_change comes back zero throughout and the
// decision rests on rhythm alone.
struct BeatFeatureInput {
    const double* frame_times = nullptr;
    const double* odf_full = nullptr;
    const double* odf_low = nullptr;
    const float* chroma = nullptr;
    std::size_t frame_count = 0;

    const double* beats = nullptr;
    std::size_t beat_count = 0;
};

std::vector<BeatFeature> beatFeatures(const BeatFeatureInput& input,
                                      const DownbeatConfig& config);

// Picks the bar length and where the bar lines fall.
//
// There is no dynamic programming here, and that is a finding rather than a
// shortcut. The plan called for Markov smoothing over a per-beat downbeat
// probability, so that one loud beat in the wrong place could not move the bar
// line. But smoothing is only needed because the underlying model is allowed to
// answer per beat — and a bar length that does not change has exactly one
// degree of freedom beyond itself, the phase. Fix the meter M and the phase p
// and every bar line in the piece is determined. So the entire hypothesis space
// is the handful of (M, p) pairs, each of which can be scored exactly, and the
// smoothing it would take a Viterbi pass to approximate is already total.
//
// The cost of that collapse is stated plainly: a piece that *changes* meter
// partway cannot be represented, and will come back as whichever meter holds
// for longer with a poor margin. Bringing back the DP is what that would need,
// and it should be brought back for that reason and not for smoothing.
//
// Scoring is a contrast, not a sum: how far the chosen beats stand above the
// ones they were chosen out of. A sum would make short bars win automatically
// by containing more beats.
//
// Offline component: allocates, and reads the whole beat list at once.
DownbeatResult findDownbeats(const std::vector<BeatFeature>& features,
                             const DownbeatConfig& config);

// ------------------------------------------------------------------ the seam
//
// findDownbeats is these two steps, and they are separable because they answer
// unrelated questions. The first asks how much each beat *looks* like a bar
// line; the second asks which bar length and phase that pattern implies. Only
// the first is a perception problem, and only the first is what a learned model
// would replace — Beat This! and BeatNet both emit a per-beat (or per-frame)
// downbeat activation and leave the counting to something else.
//
// The seam is therefore a plain `std::vector<double>`, one value per beat, and
// not an abstract class. A backend has to produce that vector from whatever it
// likes — these cues, an ONNX session, a file of activations dumped by a Python
// experiment — and nothing in the resolver knows or asks which. A virtual
// interface would add a vtable and a factory in exchange for nothing that a
// free function taking a vector does not already give, and would have to be
// designed now against a model that cannot even be downloaded in this
// environment. When a second backend exists and the shape of its needs is
// known, an interface can be introduced over two working implementations
// instead of one imagined one.

// Per-beat downbeat salience from the built-in cues.
//
// The onset cues are standardised here because their units are arbitrary and
// cue-specific; harmony is not, because a chroma distance already means
// something absolute. Both are decisions about *these* cues and belong on this
// side of the seam. What comes out is not normalised and need not be — see
// resolveMeter.
std::vector<double> cueSalience(const std::vector<BeatFeature>& features,
                                const DownbeatConfig& config);

// Bar length and phase from any per-beat salience, whatever produced it.
//
// `salience` and `beat_times` must be the same length; a mismatch returns an
// empty result rather than guessing which is right.
//
// Standardising the salience is done *here*, deliberately, and it is the reason
// a threshold calibrated on one backend means anything on another. The margins
// this returns are quoted in standard deviations of the incoming salience, so a
// model emitting probabilities in [0, 1] and these cues emitting arbitrary
// weighted sums land on the same scale, and `min_meter_margin` does not have to
// be recalibrated from scratch the day the scorer changes. It would still be
// worth re-checking — the *shape* of a distribution is not fixed by its mean and
// spread — but it starts from a comparable number rather than an unrelated one.
DownbeatResult resolveMeter(const std::vector<double>& salience,
                            const std::vector<double>& beat_times,
                            const DownbeatConfig& config);

}  // namespace tiktak::analysis
