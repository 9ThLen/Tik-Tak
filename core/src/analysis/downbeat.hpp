#pragma once

#include <cstddef>
#include <limits>
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
    // 4, 3 and 2 cover very nearly all of the repertoire this app is for. 6
    // means six pulses of the *incoming beat grid* per cycle and carries the
    // weakest prior because it is only ever distinguishable from 3 by which of
    // the two accents in the cycle is bigger. It describes 6/8 only when the
    // upstream tracker counted eighth-note pulses. A tracker working at the
    // usual tactus often represents 6/8 as two dotted-quarter beats; this
    // resolver sees neither a denominator nor subdivisions and cannot tell
    // that apart from 2/4.
    //
    // The priors are ratios against 4/4 and encode how common each meter is,
    // nothing more. They matter only when the audio is close to a tie.
    std::vector<MeterCandidate> meters = {{4, 1.0}, {3, 0.9}, {2, 0.75}, {6, 0.6}};

    // Relative weight of each cue in the per-beat score.
    //
    // **The two onset cues and harmony answer different questions**, which is
    // the thing measurement showed and argument had not. On real recordings the
    // low band gets the bar *length* right and the bar *line* wrong: a kick
    // pattern repeats every N beats whether or not the kick is on the one, so
    // it establishes N and says little about the phase. A chord change is the
    // opposite — it is weak evidence about the length and strong evidence about
    // where the bar starts.
    //
    // Measured over five recordings whose reference bar grid is regular enough
    // to trust: with the low band dominant the metre came back right four times
    // out of five and the phase zero times out of five. Shifting the balance
    // towards harmony kept the metre at four and took the phase to two — and
    // broke the percussion-only reference clip completely, where harmony is
    // nothing but noise and the low band is the only cue there is.
    //
    // So the weight stays where it was, and that is a report of a dead end
    // rather than a result. **No single fixed mixture serves both**: the useful
    // conclusion is that these are not interchangeable evidence for one
    // question and should not be blended into one number at all. Scoring the
    // metre from the onset cues and the phase from harmony, with a fallback to
    // the onsets when harmony has nothing to say, is the change this points to.
    // It is a change to the resolver rather than to a constant, five recordings
    // do not justify making it, and pretending a reweighting solved it would
    // have buried the finding under a number that did nothing.
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
    //
    // That reasoning is right and it hid a plain bug. A standardised onset cue
    // has a spread of exactly 1 by construction; a chroma distance, measured
    // across real recordings, has a spread of about 0.086. So the two weights
    // were multiplying quantities an order of magnitude apart, and
    // `harmony_weight = 1.0` was really 0.086 — the harmony cue could be given
    // a nominal weight of sixteen and still lose to the low band. The weights
    // were not describing the mixture they produced.
    //
    // kHarmonyScale fixes that without standardising: a **fixed** factor, not
    // one derived from the piece, so a drums-only track's chroma noise is
    // magnified by the same constant as a real chord change instead of being
    // stretched to fill the range. What it buys is that the weights above now
    // mean what they say.
    static constexpr double kHarmonyScale = 12.0;

    // Below this much movement, a chroma distance is measurement noise and is
    // discarded rather than scaled up with everything else. An absolute floor,
    // which is only defensible because the quantity is absolute — the whole
    // reason harmony is not standardised in the first place.
    //
    // Measured: the percussion-only reference clip, which has no chord changes
    // at all, never exceeds 0.043 and averages 0.010. The weakest real
    // recording to hand peaks at 0.165 and the rest reach 0.35 to 0.64. The
    // floor sits in that gap, so silence from the harmony cue on drum-only
    // material is a statement rather than an accident.
    static constexpr double kHarmonyFloor = 0.05;
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

    // The smallest peak-to-peak range that this scorer considers evidence.
    // A nearly constant output is ignorance, even if its tiny numerical ripple
    // happens to repeat every four beats; normalising that ripple would turn
    // floating-point leakage into a full-scale accent.
    //
    // This number is in the scorer's own units. The default belongs to the
    // built-in cue backend and rejects the roughly 0.01 chroma variation seen
    // on drums-only material.
    // A learned backend must calibrate and pass its own value together with its
    // margin thresholds; the resolver deliberately does not rescale arbitrary
    // backend output.
    double min_salience_range = 0.05;

    // How convincing the answer has to be before a caller should accent
    // anything. These values and min_salience_range are one backend-specific
    // calibration. They live in the config rather than at the call site so
    // every caller uses the same values.
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

    // What it costs the bar line to move, in this backend's salience units.
    //
    // Infinity — the default — means it may not, and the resolver behaves
    // exactly as it did before this existed: one metre and one phase for the
    // whole recording. Any finite value turns on a Viterbi over bar position,
    // where holding a phase is free and changing costs this much, so the phase
    // moves only where the evidence pays for it.
    //
    // Songs have intros, breaks and half-bar turns, and a single global phase
    // is right on only one side of an inserted or dropped bar. On 554
    // full-length annotated songs, 14.1% cannot be described by one phase at
    // all; the ceiling by corpus is 0.999 for ballroom, 0.995 for GTZAN, 0.957
    // for Harmonix and 0.893 for Beatles. Thirty-second excerpts rarely contain
    // a section boundary, which is why this looked worthless for so long.
    //
    // **What it is worth, and on what.** Measured with the annotated grid and
    // metre so only the phase rule varies, out of fold, both settings on the
    // same recording, interval bootstrapped over paired differences:
    //
    //     cost    db F    difference   95% interval        better/worse
    //      inf   0.7620        —             —                  —
    //       20   0.7686     +0.0066   [+0.0022, +0.0116]      18 / 4
    //        8   0.7684     +0.0064   [-0.0014, +0.0144]      35 / 21
    //        4   0.7682     +0.0062   [-0.0026, +0.0150]      45 / 27
    //
    // 20 is the only cost whose interval clears zero, and the split says why:
    // on the recordings one phase cannot describe it is worth +0.049 [+0.023,
    // +0.080], and on the rest -0.0003 [-0.0027, +0.0022] — not a small price,
    // no price. Every cheaper switch buys more where it is needed and spends
    // more than that everywhere else, which is the degeneracy the cost prices.
    //
    // **On the built-in cues it is worth nothing**, and that is measured too,
    // on the core's own grid and metre: at 64 it moves 13 recordings of 453 for
    // +0.0055, at 8 it is +0.0078 with the interval straddling zero, and at 4
    // it is -0.0135. Restricted to recordings whose grid is already right the
    // same shape holds. So the value here is conditional on the salience, and
    // the ordering the rest of this file argues for is not negotiable: a
    // learned activation first, this second. The default is infinity because
    // today the resolver is fed the cues.
    //
    // The cost is in salience units and therefore backend-specific, exactly
    // like min_salience_range and the two margins. It is not part of their
    // calibration triple because those three decide whether to answer and this
    // decides the shape of the answer.
    //
    // **64, from a sweep through this resolver rather than through a
    // prototype.** The tables above were decoded in Python over dumped cue
    // columns; these are `dump_analysis --phase-switch-cost` on 2244
    // recordings, paired, with the metre search and the gates all the real
    // ones:
    //
    //     cost   harmonix (554)        ballroom (698)      GTZAN (992)
    //      64   +0.0045 [+.0017,+.0080]  +0.0003          +0.0007
    //      20   +0.0050 [-.0015,+.0114]  +0.0010          +0.0010
    //       8   +0.0067 [-.0033,+.0169]  -0.0041          -0.0001
    //
    // Only one interval in the table clears zero and it is Harmonix at 64:
    // 13 recordings better, 1 worse. Harmonix is the full-length corpus, which
    // is the material a movable phase exists for; ballroom and GTZAN are
    // thirty-second excerpts where four recordings in total move at all.
    // Cheaper switching buys more on Harmonix and starts losing on ballroom,
    // and on the recordings whose beat grid is already right it loses there
    // too: pooled -0.0072 [-0.0144, -0.0000] at cost 8.
    //
    // So this is on, at the one cost that gains where the idea applies and is
    // inert everywhere else. It is a small change and it is honest about being
    // one: seventeen recordings in 2244 move, seventeen better against five.
    // The larger number this file argues for is still upstream of here.
    //
    // The metre is untouched at every cost — 89.4%, 73.9% and 76.6% before and
    // after, on all three corpora — which is the decoder being confined to the
    // question it was given rather than a happy accident.
    //
    // **This number is already a minimum dwell, which is why there is not a
    // second knob for one.** The obvious next idea is to require a phase to
    // hold for several bars before it may move again, separating "how much
    // evidence a change needs" from "how often one may happen". Measured on
    // 600 recordings, how far apart the changes this decoder actually makes
    // are:
    //
    //     cost   recordings that switch   switches each   median gap   under 2
    //      64            0.5%                  0.01        17.7 bars     0.0%
    //      20            9.7%                  0.12        19.0          0.0%
    //       8           36.3%                  0.74        15.2          3.2%
    //       4           69.7%                  3.68         5.2         28.0%
    //       2           94.3%                 16.89         1.2         65.2%
    //       1           98.7%                 46.19         0.5         86.3%
    //
    // At the shipped cost, and at 20, not one change in 600 recordings lands
    // within two bars of another. Paying a cost of λ requires accumulating λ
    // of advantage, which takes λ divided by the per-beat emission — so the
    // cost buys a dwell whether or not one is asked for, and here it buys
    // fifteen to nineteen bars of it.
    //
    // Built anyway and swept, because "it should be inert" is a prediction.
    // A floor of 1, 2, 4 and 8 bars, on 600 recordings, pooled downbeat F
    // against 0.4047 for the pinned phase:
    //
    //     cost      1 bar    2 bars    4 bars    8 bars
    //       64     0.4052    0.4052    0.4052    0.4052
    //       20     0.4059    0.4059    0.4059    0.4059
    //        8     0.4066    0.4064    0.4062    0.4069
    //        4     0.3801    0.3817    0.3850    0.3863
    //        2     0.3575    0.3619    0.3670    0.3718
    //        1     0.3388    0.3453    0.3537    0.3608
    //
    // Two readings, and both matter. The floor is **exactly** inert where the
    // decoder is any good — identical to four decimal places at 64 and 20, and
    // within noise at 8. And it does precisely what its mechanism promises
    // where the decoder is broken: at cost 1 it recovers 0.022 of the 0.066
    // that cheap switching threw away, monotonically in the length of the
    // floor. It is a real repair of a real failure.
    //
    // It is still not worth having, because the repair never climbs back to
    // where the cost alone already sits: the best cheap combination in the
    // table, 4 with an eight-bar floor, is 0.0184 below the pinned phase and
    // 0.0189 below what ships. Separating "how much evidence" from "how often"
    // works, and then finds that the setting worth using was never in the
    // region where the separation helps.
    //
    // The aggregating version of the same idea — score a four-bar pattern
    // rather than a per-beat contrast — was tried on the activation and moved
    // 85.0% to 85.3%, which is noise. See the phrase-term note further down.
    double phase_switch_cost = 64.0;

    bool valid() const;
};

struct MeterScore {
    int beats_per_bar = 0;
    int phase = 0;       // index of the first downbeat within the beat list
    // Best contrast for this meter, prior applied. Saturated at the largest
    // finite double for diagnostics; the resolver retains an overflow-safe
    // internal key so saturation never changes which meter wins.
    double score = 0.0;
};

struct DownbeatResult {
    // Times of the bar lines, a subset of the beats handed in. Empty when no
    // meter could be decided.
    std::vector<double> downbeats;
    // Beats per bar, or 0 when undecided. `phase` is the index into the beat
    // list of the first downbeat.
    int beats_per_bar = 0;
    int phase = 0;

    // How far the winning answer stands above the alternatives, in the units
    // supplied by the salience backend. The built-in backend standardises its
    // onset components but keeps harmony in absolute cosine-distance units;
    // resolveMeter preserves that mixture or a learned backend's calibrated
    // scale. Three separate doubts, kept separate because they fail differently
    // and a caller should be able to tell which one it has:
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
    //   It does carry information about correctness. Measured against a Beat
    //   This! reference over eighty-one recordings from fifty-five releases,
    //   its AUC for predicting agreement is 0.713 — and 0.792 if the one batch
    //   that came from only eight releases is set aside, which the three
    //   batches taken in order argue for: 0.562 on eight groups, then 0.777 on
    //   thirty-five and 0.821 on twelve. The low reading is the outlier, and it
    //   is the underpowered one.
    //
    //   That still does not make a large margin on any individual track mean
    //   much. On the eight-group batch the two largest margins in the set,
    //   1.782 and 1.768, were an agreement and a disagreement. What the
    //   quantity measures is how far the *mixture* won by, which is weak
    //   evidence when one cue dominates the mixture and is wrong.
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
    // caller's. Both default to conservative placeholders rather than
    // backend-specific calibrated numbers.
    //
    // Measured against a Beat This! reference on eighty-one recordings from
    // fifty-five releases, this does separate: where it returns true the
    // reference agreed 29 times in 36, and where it returns false, 30 in 45 —
    // about fourteen points of lift, and twelve on the two better-grouped
    // batches alone. Useful, and a long way from good enough to accent on
    // silently.
    //
    // An earlier version of this comment said the opposite, that the gate
    // "partitions the material into two halves of equal accuracy". That was
    // measured on twenty-six recordings from only eight releases, leaning
    // heavily on one drone-heavy session, and it did not survive the next
    // forty-seven releases. The lesson is recorded because it will otherwise
    // be repeated: the sample size that matters is the number of independent
    // releases, not the number of tracks.
    //
    // The same comment also proposed agreement between the cues as a better
    // signal than the margin of their sum, on a measured 55-point gap. That
    // is now withdrawn. Tested on two batches gathered after it was proposed,
    // the gap fell to 19 points and then to *minus* 20 — on the third batch
    // the tracks whose cues disagreed were the ones the reference agreed
    // with, every one of them. Pooling all three still shows 29 points, but
    // that number is made entirely by the eight-release batch and drops to 7
    // without it, with confidence intervals that then almost entirely
    // overlap. An effect whose sign depends on the batch is not an effect
    // yet. beat_features on OfflineResult is still what makes trying such
    // things possible from outside the core.
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
// **Measured on full-length songs, that reasoning is wrong, and wrong in a way
// the paragraph above did not anticipate.** It holds for a clip. It fails for a
// track, and the failure is not the metre changing — it is the *phase*
// changing while the metre stays put. An inserted or dropped bar at a section
// boundary shifts every bar line after it, and a single global p can be right
// on only one side of that shift.
//
// The numbers, taken from a reference downbeat grid on five recordings whose
// bar structure is regular enough to trust:
//
//   LUNCH        one phase throughout        a global phase reaches 1.00
//   makko        74% of bars on one phase                       0.74
//   東方          45% / 40% split across two                      0.45
//   Загадай      43% / 39% split                                 0.43
//   Dopamine     42% / 38% split                                 0.42
//
// So on three of the five, **no** method that commits to one phase for the
// whole song can place even half the bar lines, whatever its cues are. That
// ceiling is 2 of 5, and a state-of-the-art model's activations run through
// this resolver score exactly 2 of 5 — it is already at the limit, and the
// limit is here, not in the scorer.
//
// This is the reason to bring the DP back, and it is a much more ordinary
// reason than a metre change: songs have intros, breaks and half-bar turns,
// and a Viterbi pass over (M, p) with a small cost for changing p represents
// them where a single pair cannot. The rest of the paragraph above still holds
// — smoothing was never the motivation, and is still not.
//
// That pass has since been prototyped outside the core and measured on
// twenty-six recordings across eight releases, and the result carries one
// condition that has to travel with the recommendation:
//
//   salience fed in            single global p      Viterbi over (M, p)
//   the built-in cues              F 0.415              F 0.423
//   a learned activation           F 0.772              F 0.933
//
// The decoder is worth roughly nothing on the cues this file computes, and a
// great deal on a good activation. Both numbers come from the same grid and
// the same reference, so the difference is the salience, not the decoding.
// Ordering therefore matters: build the DP against the current cues and it
// will look like the idea failed, when what failed was the evidence handed to
// it. Activations first, decoder second, and the two are not separable pieces
// of work.
//
// The obvious objection — that a decoder free to move the phase simply drifts
// until it matches anything — was tested and does not hold. On the eleven
// tracks whose reference bar spacing is regular it changes phase 0.5 to 1.4
// times per track across the whole range of switch costs, against 42 to 63 on
// the irregular ones. It switches where there is something to switch on.
//
// The absolute F values are inflated: the reference is the same model's own
// peak picking, so anything built on that activation is being marked by a
// relative. What survives is the comparison down each column, and the
// discrimination test above, neither of which depends on the reference being
// right in absolute terms. Ranking these cues against a learned activation
// fairly still needs human annotation.
//
// **That human annotation has now been run, and it does not reproduce.** The
// same decoder — `research/eval/moving_phase.py`, unmodified, the code that
// produced the table above — scored against annotated bar lines on 1677
// recordings, each held out from the checkpoint that scored it, with the
// annotated grid and the annotated metre handed over so only the phase rule
// varies:
//
//     decoder                      GTZAN db F     Ballroom db F
//     one global phase (ships)        0.838           0.982
//     movable phase, switch cost 20   0.838           0.982
//                             ... 4   0.839           0.982
//                             ... 1   0.828           0.980
//                          ... 0.01   0.774           0.963
//
// Nothing wins. The reason is not the decoder, and it is worth stating before
// anyone tunes the switch cost again: **there is nothing on these corpora to
// win.** How well one global phase can describe the annotated bar lines at
// all, computed from the annotations alone — no audio, no model:
//
//     corpus       beats (median)   ceiling F   recordings it cannot describe
//     ballroom          61           0.9994               0.3%
//     gtzan             57           0.9947               1.6%
//     hainsworth       102           0.9892               3.2%
//     rwc              393           0.9815              14.2%
//     harmonix         395           0.9571              13.9%
//     beatles          288           0.8925              34.1%
//
// GTZAN and Ballroom are thirty-second excerpts. A phase slip needs a section
// boundary and fourteen bars rarely contain one, so a movable phase has 0.006
// and 0.0006 F available to it there. The decoder does work where it is
// needed — on the 2.6% of GTZAN that one phase cannot describe it gains 0.06
// to 0.09 F — and loses more than that on the 97.4% where it is free to slip
// and has no reason to.
//
// Two corrections follow, and both were errors of sample size rather than of
// reasoning:
//
// *Five recordings overstated the problem by an order of magnitude.* The table
// above says three of five songs cannot place half their bar lines from one
// phase. On 911 annotated full-length songs the figure below 0.50 is 2%, and
// the corpus ceiling is 0.957. Full-length songs do need a movable phase more
// than excerpts do; they do not need it anything like that often.
//
// *The circularity is worse than the paragraph above admits.* It grants that
// comparison down a column is unsafe and claims comparison along a row
// survives, because the salience and the reference are held fixed and only the
// phase constraint moves. That does not hold here: the reference is Beat
// This!'s own peak picking, and its postprocessor is itself a decoder that
// moves the phase freely. A movable-phase decoder reproduces a movable-phase
// reference more closely whether or not either is right, so the 0.772 -> 0.933
// belongs to the reference's freedom as much as to the music's.
//
// **On full-length songs it does win, and only when barely allowed to move.**
// Harmonix is full-length and human-annotated but ships no audio; 554
// recordings of it have since been aligned against the Set's own official
// mel-spectrograms — a reference that owes nothing to any beat or downbeat
// model, which is what stops the selection from answering its own question —
// and rewritten onto the annotation's timeline. Scored out of fold, annotated
// grid and annotated metre, both settings on the same recording, interval
// bootstrapped over the paired differences:
//
//     switch cost    db F    difference   95% interval        better/worse
//        inf        0.7620        —             —                  —
//         20        0.7686     +0.0066   [+0.0022, +0.0116]      18 / 4
//          8        0.7684     +0.0064   [-0.0014, +0.0144]      35 / 21
//          4        0.7682     +0.0062   [-0.0026, +0.0150]      45 / 27
//
// Only a switch cost of 20 clears zero, and the split says why it is the right
// one rather than merely the luckiest. On the 14.1% of recordings a single
// phase cannot describe it is worth +0.049 [+0.023, +0.080]; on the other
// 85.9% it is worth -0.0003 [-0.0027, +0.0022] — not a small cost, no cost.
// Every smaller cost buys more where it is needed (+0.083 at 4) and spends
// more than that everywhere else, which is exactly the degeneracy the switch
// cost exists to price.
//
// So the decoder is real, it is small, and it is the *second* half of a job.
// The first half is that resolveMeter is not fed this activation at all —
// OfflineAnalyser calls findDownbeats, which is the built-in cues, and on
// those cues this same decoder measured 0.415 -> 0.415, 0.658 -> 0.658 and
// 0.665 -> 0.474 across three batches. Building it before the salience is
// wired through would move the shipping number down by a tenth while the
// measurement above says up by seven thousandths. Activations first, decoder
// second: the ordering the earlier paragraphs assert on a model's own peak
// picking now holds on human annotation, out of fold, with an interval.
//
// **On the path that actually ships, it is worth almost nothing, and the three
// small batches above were not enough to say which way.** Same decoder, same
// 554 songs, but now the grid is the core's own, the metre is the core's own,
// the salience is these cues, and the score is the time-based downbeat F the
// product is judged by — so a switch cost of infinity has to reproduce the
// resolver's own bar lines, and on all 453 scorable recordings it does:
//
//     switch cost    db F    difference   95% interval        better/worse
//        inf        0.3518        —             —                  —
//         64        0.3572     +0.0055   [+0.0021, +0.0098]      13 / 1
//         20        0.3579     +0.0061   [-0.0015, +0.0140]      55 / 40
//          8        0.3596     +0.0078   [-0.0048, +0.0204]     128 / 125
//          4        0.3383     -0.0135   [-0.0288, +0.0018]     164 / 200
//
// So "measured harmful on the cues" was a reading of sixteen recordings. On
// 453 it is not harmful at high switch cost and not helpful either: the only
// row whose interval clears zero moves 14 recordings out of 453, and the row
// that moves everything loses. Both statements — the earlier harm and this
// gain — are smaller than the 0.41 of downbeat F that is simply missing, and
// neither is a reason to build anything.
//
// The obvious objection is that most of those 453 have a broken beat grid, and
// a bar-phase decoder cannot be blamed for beats that are in the wrong place.
// Restricting to the 153 whose grid is already right (beat F >= 0.8) does not
// rescue it — it sharpens the same answer:
//
//     switch cost    db F    difference   95% interval        better/worse
//        inf        0.6264        —             —                  —
//         64        0.6327     +0.0063   [+0.0000, +0.0163]       3 / 0
//         20        0.6204     -0.0060   [-0.0224, +0.0101]       9 / 11
//          8        0.6217     -0.0048   [-0.0345, +0.0254]      33 / 44
//          4        0.5743     -0.0522   [-0.0887, -0.0158]      42 / 82
//
// On a good grid and these cues the decoder either does nothing (three
// recordings move at cost 64) or does harm. The same decoder on the same
// corpus with a learned activation and the annotated grid gained +0.0066 with
// an interval clear of zero. Same code, same songs, same standard: the
// difference is the salience. That is the third independent way this file has
// arrived at the same ordering, and the first on the path that ships.
//
// One caveat on the absolute values in the last two tables, not on the
// differences: about 40% of that corpus is still misaligned, so 0.3518 and
// 0.6264 are lower than this resolver deserves. See analysis/offline.hpp,
// which splits the corpus with an out-of-fold tracker. Both settings run on
// the same recording with the same grid on both sides, so the paired
// differences and the intervals are unaffected — a displaced recording
// displaces the two arms together.
//
// **Where the loss actually is, since it is not the decoding.** On GTZAN, with
// the annotated grid and metre and a learned activation, one global phase
// reaches 0.838 against a ceiling of 0.994. That 0.15 is the phase being
// chosen wrong, and it is not a near miss: measured as a fraction of the whole
// spread of the phase scores, the wrong phase led the right one by a median of
// 0.70, and by 0.99 at the third quartile — a quarter of the failures put the
// right phase at or near the *minimum*. Only 6.9% were within 0.05 of a tie.
// Sixty per cent of them are exactly half a bar out.
//
// Both halves of that matter. A confident wrong answer is not recoverable by
// any rule reading the same numbers, which is why swapping the mean for a
// median or adding a four-bar phrase term moved 85.0% to 85.3% and 85.1% —
// noise. And accumulating evidence across bars, whatever the mechanism, cannot
// break a half-bar symmetry: the wrong phase repeats on exactly the period
// being accumulated over, so more bars make it more confident, not less. The
// cue that distinguishes beat 1 from beat 3 has to be one that does not repeat
// at the half bar — which is the harmony argument made from five recordings at
// the top of this file, now with a population attached to it.
//
// **The harmony cue is that cue, and reweighting it still does not ship.**
// This file has argued twice that the way out is evidence which does not
// repeat at the half bar, and named harmony as the candidate. It was swept:
// only the ratio between the harmony and low weights matters to the phase, the
// decision being an argmax over a scaled salience, and the Python single-phase
// decoder reproduces this resolver's bar lines exactly (checked, 453 of 453).
// On the 1121 recordings across all three corpora whose beat grid is already
// right, metre held at what the core chose so only the phase moves:
//
//     harmony/low   harmonix   ballroom   gtzan   pooled   95% interval
//         0.50       0.6175     0.6725   0.5213   0.5854  [-0.0301, -0.0109]
//         1.00       0.6264     0.6906   0.5458   0.6056        — (ships)
//         1.75       0.6472     0.7089   0.5597   0.6219  [+0.0064, +0.0267]
//         3.00       0.6622     0.6916   0.5688   0.6230  [+0.0017, +0.0330]
//         8.00       0.6593     0.6479   0.5630   0.6048  [-0.0239, +0.0218]
//
// A broad optimum, a clear direction, two of the three held-out folds choosing
// 1.75, and an interval clear of zero. Everything this project asks of a
// fitted constant.
//
// It does not survive the metre. Recombined and fed back through the real
// resolver, where the metre search reads the same salience and is free to move
// again:
//
//     harmony/low   pooled db F      difference   metre: hx / ballroom / gtzan
//        0.00          0.5146   -0.0919 [-.111,-.073]   90.2 / 87.0 / 81.2
//        0.50          0.5721   -0.0345 [-.046,-.023]   88.9 / 90.7 / 79.8
//        1.00          0.6066            — (ships)      90.2 / 88.9 / 81.5
//        1.25          0.6115   +0.0049 [-.003,+.013]   90.2 / 87.8 / 80.2
//        1.75          0.6164   +0.0098 [-.002,+.022]   90.2 / 83.1 / 81.0
//
// Nothing clears zero upward, ballroom's metre falls six points by 1.75, and
// every step away from 1.0 in the other direction is worse on both counts. The
// shipped weighting is a genuine optimum — for the joint decision.
//
// The finding is not that harmony is useless — the phase table above is
// unambiguous that it carries the half-bar-breaking evidence. It is that one
// scalar cannot spend it, because the same mixture answers two questions and
// the answers want different weights.
//
// **Which is itself a measurement, and it says what to build.** The table above
// *is* the split: the metre came from the shipped mixture and only the phase
// from the reweighted one. Repeated over every recording rather than only those
// with a good grid, so the result does not depend on that filter:
//
//     harmony/low for the phase only, metre unchanged, n = 2244
//                   harmonix   ballroom    gtzan   pooled   95% interval
//         1.00       0.2879     0.4873    0.4167   0.4069        — (ships)
//         1.75       0.2958     0.4977    0.4277   0.4169  [+0.0046, +0.0157]
//
// Pooled +0.0100 clear of zero, and clear of zero on ballroom [+0.0008,
// +0.0208] and GTZAN [+0.0030, +0.0195] on their own. Same optimum, same shape
// of curve, whichever subset it is asked on. So resolveMeter wants two inputs,
// not one: the evidence that says how long the bar is and the evidence that
// says where it starts are not the same evidence, and the seam that made this
// measurable — one vector in, one answer out — is also what makes the gain
// unreachable today. That is the change worth designing, and it is worth about
// a hundredth of downbeat F, which is small but is more than anything else on
// this side of the salience has been worth.
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
// The seam is therefore a plain `std::vector<double>`, one finite value per
// beat, where a larger value means stronger downbeat evidence. A backend has to
// produce that vector from whatever it likes — these cues, an ONNX session, a
// file of activations dumped by a Python experiment — and declare its scale
// through DownbeatConfig: min_salience_range, min_phase_margin and
// min_meter_margin are calibrated as a set. Nothing in the resolver knows or
// asks which backend produced them. A virtual interface would add a vtable and
// a factory in exchange for nothing that a free function taking a vector does
// not already give, and would have to be designed now against a model that
// cannot even be downloaded in this environment. When a second backend exists
// and the shape of its needs is known, an interface can be introduced over two
// working implementations instead of one imagined one.

// Per-beat downbeat salience from the built-in cues.
//
// The onset cues are standardised here because their units are arbitrary and
// cue-specific; harmony is not, because a chroma distance already means
// something absolute. The mixture keeps those units; the range gate is applied
// once by resolveMeter to the actual vector crossing the seam. Those are
// decisions about *this* backend and belong on this side of the seam.
std::vector<double> cueSalience(const std::vector<BeatFeature>& features,
                                const DownbeatConfig& config);

// Bar length and phase from any per-beat salience, whatever produced it.
//
// `salience` and `beat_times` must be the same length and every salience value
// must be finite; invalid input returns an empty result rather than guessing.
// The same applies when finite values span so many orders of magnitude that
// their distinct levels cannot all survive the resolver's affine double
// representation: merging evidence levels can create a false periodic
// contrast, so no answer is safer than a numerically invented meter. The
// calibrated range and margin thresholds also define the numerical resolution:
// a dynamic range that leaves fewer than eight guard bits after accounting for
// the number of beats is rejected. A zero phase margin cannot provide that
// contract and is therefore no answer; a zero meter margin is allowed only
// when at most one meter is eligible. A range below config.min_salience_range
// is likewise no answer.
//
// The resolver removes an irrelevant constant offset but does not divide by
// the range or standard deviation. Scale is evidence: turning probabilities
// 0.500001 and 0.500003 into unit variance would make an ignorant model look
// certain. Consequently strength and margins remain in this backend's units,
// and replacing a scorer requires calibrating the range and margin thresholds
// on held-out material rather than inheriting another backend's numbers.
DownbeatResult resolveMeter(const std::vector<double>& salience,
                            const std::vector<double>& beat_times,
                            const DownbeatConfig& config);

// Which position in the bar each beat holds, allowing that to change.
//
// Exposed rather than hidden inside resolveMeter because it is the piece worth
// testing on its own: whether it follows a real slip, whether it stays put when
// there is nothing to follow, and whether it agrees with the research
// prototype the switch costs were measured with. A decoder that is only
// reachable through the resolver can only be tested through the resolver's
// metre search as well, which would mix two failures into every red test.
//
// `switch_cost` is in salience units; infinity pins the position and returns a
// constant path. Returns one position per beat, so beat i begins a bar exactly
// when `i % m == path[i]`.
std::vector<std::size_t> barPositions(const std::vector<double>& salience,
                                      std::size_t m, double switch_cost);

}  // namespace tiktak::analysis
