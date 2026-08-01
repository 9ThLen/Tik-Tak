#pragma once

#include <array>
#include <cstddef>
#include <limits>
#include <optional>

#include "dsp/odf.hpp"
#include "ml/beatnet.hpp"
#include "tracking/activation_tempo.hpp"
#include "tracking/particle.hpp"
#include "tracking/sync.hpp"

namespace tiktak::tracking {

struct LiveConfig {
    dsp::OdfConfig odf;
    ParticleFilterConfig filter;
    SyncConfig sync;  // manual mode only: finding the phase at a known tempo

    // Half-life of the peak the onset function is normalised against. The
    // filter's constants are calibrated in units where a beat is worth about
    // one, so the front-end has to deliver that whatever the room's level —
    // hence dividing by the loudest onset of the last few seconds rather than
    // by a standard deviation. A z-score would also be level-invariant, but its
    // scale moves with the *material*: dense music raises the running mean and
    // shrinks every peak towards it, so the same beat arrives as weaker
    // evidence purely because there is more going on around it.
    double onset_peak_tau_sec = 3.0;

    // How long our own click is treated as blinding the microphone, relative to
    // the moment it is heard.
    double gate_before_sec = 0.005;
    double gate_after_sec = 0.050;

    // Hysteresis on publishing beats. Above `lock_confidence` the tracker
    // follows the cloud; below `release_confidence` it stops handing out beats
    // altogether; in between it coasts at the last tempo it was sure of, which
    // is what a musician does when the band drops out for a bar.
    //
    // 0.25 / 0.02, not the 0.35 / 0.15 this carried before. The old band was
    // chosen without a public corpus behind it, and measured against one it was
    // withholding most of the track: driven by BeatNet it emitted 0.48 beats
    // for every annotated one on GTZAN and said nothing at all on 2.1% of
    // recordings. Both corpora, prior held at what it shipped:
    //
    //     lock/release    GTZAN F   CMLt   beats/ref   silent
    //     0.45 / 0.15       0.408  0.322      0.42       8.0%
    //     0.35 / 0.15       0.486  0.362      0.48       2.1%
    //     0.25 / 0.02       0.513  0.376      0.60       0.9%
    //     0.15 / 0.02       0.522  0.375      0.65       0.2%
    //
    // The sweep wanted to keep going, and it is stopped at 0.25 by a test
    // rather than by a preference. Below it, ADipInConfidenceDoesNotLetTwo-
    // BeatsOutTogether fails: at 0.20 the tracker publishes two beats 0.163 s
    // apart on a half-second pulse, which is the stutter that test was written
    // for after it had already shipped once. Bisected, and the boundary is the
    // lock alone: 0.25 passes and 0.20 does not, at every release from 0.10
    // down to 0.01. An F-measure has no column for "clicked twice", so the
    // invariant decides here and the corpus does not get a vote.
    //
    // Giving up that last 0.01 F costs nothing anyway. With the prior re-aimed
    // as well, 0.25 beats 0.15 on CMLt on both corpora, 0.428 against 0.413 and
    // 0.495 against 0.484. The safe point is also the better one.
    //
    // What the band cannot do is improve the beats it lets through: CMLt moves
    // 0.014 across the whole sweep at a fixed prior. The withheld beats were not
    // being kept from a tracker that knew where they were, so opening the gate
    // lets out more of the same. Where they land is the prior's job, and see
    // ParticleFilterConfig::prior_centre_bpm for it: that is the larger of the
    // two changes by some way.
    double lock_confidence = 0.25;
    double release_confidence = 0.02;

    // Tempo measured from the activation history, used to aim the filter's
    // prior at the metrical level the recording is actually in.
    ActivationTempoConfig activation_tempo;

    // On. Over 698 ballroom and 999 GTZAN recordings, against free running:
    //
    //                   ballroom            GTZAN
    //     free       0.700  0.584        0.632  0.508
    //     anchored   0.794  0.705        0.666  0.565
    //     hard pin   0.778  0.782        0.697  0.637
    //     the answer 0.881  0.873        0.737  0.790
    //
    // On ballroom that is better than holding the period outright, which is
    // the thing the corpora were expected to prefer and the thing that cannot
    // ship. It is not a compromise reached against them: at a six-second
    // window the anchor is both better on the corpus and quicker to follow a
    // tempo change than any longer-window setting, so nothing was traded away
    // to get it. See ActivationTempoConfig::window_sec.
    bool anchor_tempo = true;

    // How much room the anchored tempo is left, in octaves. A fifth of an
    // octave is about 15% either way, so half and double sit four widths out
    // and are outvoted, while a singer drifting a few percent is followed
    // rather than fought.
    //
    // 0.10 against 0.20, at the six-second window both were measured on:
    //
    //     width   ballroom F   GTZAN F   worst lag over six tempo changes
    //     0.10      0.794       0.666           6.8 s
    //     0.20      0.760       0.659           5.6 s
    //
    // A point and a bit of latency for three and a half points of F. Note how
    // much smaller the choice is than it was at a thirty-second window, where
    // the same widths spanned 23.1 seconds of worst-case lag: once the window
    // is short the width stops being the thing that matters, which is the
    // right way round, because the width is a belief and the window is
    // evidence.
    double anchor_width_octaves = 0.1;

    // How decided the estimator has to be before its answer is used, as the
    // gap to the best rival at another metrical level.
    //
    // Zero, meaning "use it whenever there is one", and that is a measured
    // result rather than an omission. Gating looked obviously right — a
    // half-certain octave is exactly what one would not want to hold — and on
    // 120 ballroom recordings it is worse at every setting tried: F 0.752 with
    // no gate, 0.738 at 0.15, 0.714 at 0.30. The reason it does not help is
    // that a tie in the estimator is not a coin toss downstream. Both rivals
    // are metrical relatives of each other, so anchoring the wrong one still
    // puts the filter on a grid the right beats fall on, whereas refusing to
    // anchor leaves it with the fixed prior, which is worse than either.
    //
    // Kept as a parameter rather than deleted because the question is a real
    // one and the answer may not survive a corpus that is not ballroom.
    double anchor_octave_margin = 0.0;

    // A stream time this far from where the sample count says it should be
    // means the device dropped or repeated a buffer.
    double discontinuity_tolerance_sec = 0.002;

    bool valid() const;
};

// ------------------------------- what this path is worth to a person using it
//
// An average F-measure cannot answer the question the product asks. Two
// recordings with the same CMLt are not the same experience: one takes four
// seconds to start clicking and then holds, the other starts at once and jumps
// an octave twice in the middle. So the live benchmark also scores each
// recording pass or fail — starts within 8 s, at least 80% of the beats it
// emits land within 70 ms, *finds* at least 80% of the beats there were, and
// never spends more than 4 s at the wrong metrical level — and reports the
// share that passes. 1914 recordings, shipped thresholds:
//
//     front end          ballroom*   GTZAN    SMC   median acquire
//     spectral flux         5.7%     13.4%   1.4%       6 s
//     BeatNet activation   57.2%     41.1%   3.2%       5 s
//
//     * in this model's training set — see below, and do not quote it
//
// The third criterion is matched recall and not the ratio of the two beat
// counts, which is what it was first written as. A tracker can emit exactly as
// many beats as there were and put all of them somewhere else: precision 0.80
// with a count ratio of 0.80 bounds the beats actually found at 64%. The
// figures above are after that correction and are about a point lower for it.
//
// **Quote GTZAN.** The weights are `beatnet_model_1`. BeatNet trains on five
// corpora — Ballroom, Beatles, Carnatic, GTZAN and Rock Corpus — and ships
// three models, holding out GTZAN, Ballroom and Rock Corpus respectively;
// docs/ml-models.md records both. Model 1 holds out GTZAN, so ballroom is in
// its training set and its 57.2% is *not an independent estimate*. That is
// weaker than saying it memorised those recordings, which nothing here shows,
// and it is enough: a number measured on training material cannot be quoted as
// performance whatever produced it. SMC is in none of the five, so its 3.2% is
// out of training — it is a domain shift rather than a held-out twin of GTZAN,
// and that is why it is reported apart rather than averaged in.
//
// An earlier revision averaged all three into "34.4% usable" and a later one
// wrote the honest headline as 42.6%, which is the *pooled* rate over all
// 1914 recordings and therefore has ballroom in it too. Both are wrong for the
// same reason. The headline is **41.1%, GTZAN, per corpus**.
//
// **Acquisition is not the problem, with one caveat.** It was, and the
// diagnosis chain in research/eval/README.md is about it; it is fixed. The
// median is five seconds, and slow acquisition is a listed failure on 38.7% of
// SMC and 9.2% of everything else. The caveat is that "acquired" means the
// first time confidence crossed the lock threshold, whether or not the level it
// locked to was right, and that is not merely a labelling problem: lock at 2 s
// on half tempo, release, re-lock correctly at 10 s, and if the wrong stretch
// stayed under four seconds the recording passes the acquisition criterion on
// the strength of a lock that was wrong. So the rates above can contain false
// passes. The benchmark now also reports `settled_at` — the first locked
// stretch at the annotated level that lasts four seconds — beside the old
// figure rather than in place of it, so the two columns can be compared instead
// of one silently replacing the other.
//
// **Where the metrical level comes from — withdrawn pending a re-run.** This
// paragraph used to give two conditional rates, 85% and 92%, for how often the
// anchor and the filter are at the wrong level together. They were computed by
// keeping every observation above the lock threshold, which is not the
// tracker's notion of tracking: it samples the tracker's most *confident*
// seconds, and confidence is not independent of what is being measured —
// throwing away the shaky seconds throws away the seconds where the two
// disagree. `octave_blame.py` now follows the lock/release hysteresis instead,
// and no number replaces those two until it has been run again. What survives
// is only the direction, and only weakly: the anchor and the filter are at the
// wrong level together far more often than apart. Three ways of improving the
// anchor from what is already computed were tried and all lost (below).
//
// That does not establish that the filter is blameless, and an earlier
// revision of this comment said so on the strength of the filter agreeing with
// a correct anchor 94% of the time. That figure cannot carry the claim.
// `anchorTempo` is applied on every submitted frame — fifty a second with
// BeatNet, see LiveTracker::submit — with a prior a tenth of an octave wide,
// so the agreement is largely enforced rather than observed; only the estimate
// behind it is refreshed once a second. The 15.4% residual quoted here is
// withdrawn with the rest of that measurement. Separating the two needs the
// filter run *without* the anchor and against an oracle level — which is now a
// flag, `--live-no-anchor`, and an arm of the benchmark, `--no-anchor`. Until
// it has been run, no statement here apportions blame between the estimator
// and the filter.
//
// **The level is not where the recordings are lost.** Scored per *recording*
// rather than per second, with the grid read at half (both phases) or twice
// its rate and judged at whichever agrees best — an oracle correction applied
// to the whole recording, and therefore an upper bound on any control the
// player could be given, while also removing the wrong-level criterion
// outright rather than modelling one press:
//
//     usable        ballroom*   GTZAN    SMC
//     as it stands     57.2%    41.1%   3.2%
//     any level        59.0%    46.0%   4.6%
//
// Five points on GTZAN, and of the GTZAN recordings that fail today only 8.3%
// become usable at another level.
//
// **On whole songs it is the other way round, and that reverses the reading.**
// RWC 2.0 arrived after the numbers above: 328 full-length recordings, 23.4
// hours, beats for all five collections, and out of `beatnet_model_1`'s five
// training corpora — so it is held out, correctly aligned (Beat This! scores
// 0.993 on RWC-Pop, which is train-on-test for *it* and therefore an alignment
// check rather than a score; Harmonix scored 0.490 on the same check and turned
// out to be displaced), and not made of thirty-second excerpts:
//
//     usable            n    as it stands   any level    F    CMLt
//     RWC-Pop         100        33.0%        56.0%    0.786  0.703
//     RWC royalty-free 15        33.3%        40.0%    0.618  0.562
//     RWC-Genre       102        10.8%        21.6%    0.574  0.432
//     RWC-Jazz         50         6.0%        12.0%    0.523  0.346
//     RWC-Classical    61         0.0%         1.6%    0.374  0.155
//
// Beat placement on full-length pop is *better* than on GTZAN — F 0.786 against
// 0.665 — and the recordings are lost to the metrical level instead. The
// failure list inverts with it: on RWC-Pop the wrong level is a listed failure
// on 62% of recordings against too-few-beats on 38%, where GTZAN reads 37.7%
// against 54.3%.
//
// So "a ×2 control in the product would recover little" was true of excerpts and
// is false of songs: the oracle level is worth 23 points here against 4.9 on
// GTZAN. It remains an oracle over the whole recording and therefore an upper
// bound on any control a player could be given, not a forecast of one. But the
// direction of the work follows the corpus that looks like the product, and
// this one does: a thirty-second excerpt simply does not last long enough to
// spend four seconds at the wrong level and then be judged on it.
//
// Classical at 0.0% and jazz at 6.0% are not a surprise and not the same
// problem; see the note on sparse material below.
//
// Why GTZAN's recordings fail, as a share of all 999 of them — a recording can
// fail several ways at once, so these overlap and do not sum to the 58.9% that
// fail:
//
//     too few beats found     54.3%
//     wrong beats             51.4%
//     wrong level over 4 s    37.7%
//     slow to acquire          9.2%
//     never acquired           0.5%
//
// The largest is recall: the tracker is not putting beats where the beats are.
// That is what the count-ratio version of the criterion was hiding, and an
// earlier revision of this comment misreported these as 52.6/43.6 against an
// unstated denominator. They are shares of the corpus, not of the failures.
//
// Note what this does *not* say. A grid at the right level but the wrong local
// tempo, or one that drifts, arrives as both of those failures too, so nothing
// here separates placement from tempo. It says the level is not the way out,
// and no more than that.
//
// **What the published comparison is, and is not.** BeatNet's paper reports
// 0.754 beat F on GTZAN and BeatNet+ 0.806, both full systems — activation
// plus their two-level cascade particle filter, not activations alone. The
// same BeatNet activation through this tracker gives 0.666. That is a
// published score against a local one, measured by different code on
// different framing, and it is a *lead*, not a decoder gap: nothing here has
// yet run their filter on our activations. The A/B that would make it a
// measurement is the next piece of work.
//
// Every figure above was measured twice, by
//
//     python -m eval.live_corpus_benchmark --model models/beatnet_model_1.ttw \
//         --include-root-audio --mode model --output results/live_usable.json
//
// which writes the commit, the weight file's SHA-256, the per-corpus file
// counts and whether the tree was clean beside the numbers. Quoting a rate
// without those is neither reproducible nor falsifiable: a later disagreement
// cannot be settled, because nobody can tell whether the code moved or the
// corpus did. Note that a run from a dirty tree records a commit that does not
// identify the binary that produced it, so the artifact only becomes provenance
// once the tree is clean. The audio is not in the repository (see .gitignore),
// which is why the numbers are written out here as well.
//
// **Where the missing beats go, measured without a decoder and then without a
// front end.** Two experiments, `research/eval/activation_recall.py` and
// `oracle_activation.py`. The first takes as many of the strongest peaks of the
// BeatNet activation as there are annotated beats — no rhythm, no filter, just
// height — and the second feeds the filter a pulse at every annotated beat and
// nothing else. Recall at 70 ms, after the same five-second warm-up:
//
//                                     GTZAN     SMC
//     strongest N peaks               64.7%    35.8%
//     the shipping path               66.7%    26.1%
//     told the beats, as shipped      92.7%    54.6%
//     told the beats, anchor off      95.3%    59.8%
//
// Whole corpora: 998 GTZAN and 217 SMC, every genre. An earlier revision quoted
// 92.2% from a subset that took every sixth recording and then the first 150 of
// those, which stops at reggae and omits the rock genre entirely on a corpus
// filed by genre. The bias turned out to be worth half a point here, but it was
// not knowable in advance and the sampling is fixed (`sample()` rounds the
// stride up so the stride itself does the limiting).
//
// Read it as a budget. On GTZAN the front end costs about 26 points and
// everything after it about 7 — so most of the recall is lost before the
// decoder, and a better observation is worth more than a better decoder. On SMC
// the same arithmetic gives 29 points to the front end and **45 to what follows
// it**: told exactly where every beat is, this tracker still finds barely half.
// That is the opposite of what this comment used to say about SMC, which was
// that the symptom was consistent with a front-end limit.
//
// The last row is why the row above it is not "the filter". `anchor_tempo`
// ships true and `submit` applies it on every frame, so an oracle run measures
// the six-second autocorrelation and the filter together. Separated, the anchor
// costs **2.6 points on GTZAN and 5.2 on SMC** when the observation is perfect,
// and on GTZAN it costs 6.1 points of the recordings that clear the 80% bar.
// That is what an insurance premium looks like: with a clean observation the
// anchor has nothing to protect against and its six-second lag is pure cost.
//
// **And here is the payout.** The same switch on the *real* activation, scored
// by the product's criterion over all 1914 recordings:
//
//     usable          ballroom   GTZAN    SMC   mean F   tracks that switch level
//     anchor on         57.2%   41.1%   3.2%    0.665           27.2%
//     anchor off        38.3%   34.3%   2.8%    0.608           46.8%
//
// **Read the GTZAN column and only that one.** It costs 2.6 points of GTZAN
// recall to buy 6.8 points of usable recordings — a payout between two and three
// times the premium, on the corpus this model has not been trained on. Ballroom
// shows nineteen points and SMC shows almost nothing, and neither is admissible
// as the size of the effect: ballroom is in `beatnet_model_1`'s training set,
// and SMC is so far from usable at either setting that a difference there is
// measured between two failures. The mechanism is visible in the last column
// the tracker ends on, which falls from 67.2% correct to 58.7% on GTZAN when
// the anchor goes: what the anchor sells is octave stability, which is exactly
// what it was added for and what `ActivationTempo`'s own comment claims for it.
//
// So the anchor stays, and the conclusion is narrower and more useful than
// either "keep" or "remove": its strength should depend on how much the
// observation is worth at that moment. It is paying a fixed premium against a
// risk that varies, and the two numbers above are the first measurement of both
// sides of that trade rather than of one.
//
// And the decoder's loss has a name — but not the one this comment gave it
// first. The rank correlation between a recording's annotated tempo spread and
// its oracle-fed recall is -0.51 on GTZAN, which was read here as "the filter
// cannot follow a tempo that changes". That reading was wrong, because the
// spread it correlates with is two things at once. Split the interval series
// into a slow trend (a nine-beat running median) and the residual around it:
//
//                  median drift   median jitter   rho(drift)   rho(jitter)
//     GTZAN           0.0084          0.0122        -0.35        -0.39
//     SMC             0.0423          0.0706        -0.20        -0.38
//
// Whole corpora, from research/eval/timing_irregularity.py. Jitter is the
// stronger predictor on both, and on SMC it is nearly twice as strong — but the
// separation is modest and **drift is not zero anywhere**. An earlier revision
// read a subset figure of -0.02 as "drift has no relationship with recall on
// SMC" and built a conclusion on it; at full size that number is -0.20. The
// supportable claim is only that irregularity predicts failure better than
// drift does, not that drift is innocent.
//
// A synthetic bench agrees on the direction from the other side, passing every
// ramp of ±2/5/10% over 15 or 45 seconds at F70 ≈ 0.995 while failing four of
// five clips at 40 ms of jitter. Note that at 70 ms of jitter its failure is
// close to tautological, the deviation being the width of the scoring window.
//
// Nor is the filter over-smoothing, which is the next thing one would assume.
// Score a locally steady pulse — the annotated beats with every interval
// replaced by its local median, so it follows drift and never follows jitter:
//
//     recall at 70 ms          GTZAN    SMC
//     a locally steady pulse   83.5%   33.2%
//     our filter, oracle-fed   92.7%   54.6%
//
// Nine points above that bound on GTZAN and twenty-one on SMC. The filter
// already follows a great deal of expressive timing. And 33.2% says something
// about SMC rather than about us: two in three of its annotated beats sit more
// than 70 ms from any locally regular pulse, so a large part of the remaining
// 45 points is deviation a causal system cannot predict in principle. How much
// is irreducible is not known; that it is a substantial share is.
//
// **Where the anchor helps and where it hurts, per recording.** With the same
// oracle observation, the anchor changes some recordings by more than five
// points and most by less. Grouping them and looking at what the groups have in
// common is how one would find a run-time signal to make its strength
// conditional — and the honest result is that these features do not provide one:
//
//     GTZAN            n    drift    jitter   recall on the real activation
//     anchor saves    56   0.0149    0.0400              44.2%
//     no effect      808   0.0074    0.0100              83.5%
//     anchor costs   134   0.0139    0.0212              35.0%
//
//     SMC              n    drift    jitter
//     anchor saves    60   0.0436    0.0713
//     no effect       61   0.0406    0.0627
//     anchor costs    96   0.0425    0.0742
//
// The recordings the anchor touches at all are the irregular ones the front end
// is also failing on; the 808 it leaves alone are the easy ones. But saved and
// cost are not separated by drift on either corpus, and on SMC they are not
// separated by anything here. Only GTZAN's jitter column shows a gap worth
// noticing, 0.040 against 0.021.
//
// So "widen the prior when the tempo is changing" is not supported yet: on the
// evidence here it would fire on the saved and the cost recordings alike. A
// conditional anchor is still the right shape — the on/off measurement supports
// that much — but the condition has to come from quantities the tracker has at
// run time, the octave margin and the size and persistence of the disagreement
// between the anchor and the filter, and those have not been joined to this
// outcome yet.
//
// **The filter cannot be made agile enough, and this is why.** If the decoder's
// loss is tempo agility, the filter has a knob for exactly that:
// `roughening_octaves`, the spread added to every resampled particle's period,
// which ships at 0.01. Swept against the oracle activation it looks like a free
// win — GTZAN flat within noise while the share of recordings clearing 80% rises
// 87.4% -> 91.4%, and SMC recall 54.6% -> 66.8% at 0.08. Run on the *real*
// activation and scored by the product's own criterion, every one of those
// numbers reverses:
//
//     usable            ballroom   GTZAN    SMC   mean F
//     0.01, as shipped     57.2%   41.1%   3.2%    0.665
//     0.02                 56.3%   40.2%   3.2%    0.659
//     0.08                 40.4%   30.0%   2.3%    0.615
//
// Eleven points of GTZAN gone, and SMC — the corpus the widening was *for* —
// worse too, despite gaining thirteen points of oracle recall. The reason is in
// the paragraph above: 8.7 noise peaks a second at a median height of 0.0018. A
// cloud wide enough to follow a tempo that moves is wide enough to chase those,
// and agility and noise-immunity are not two knobs but one.
//
// So 0.01 stays. It is worth being explicit that an earlier version of this
// paragraph used the same evidence to argue for a predominant-local-pulse
// decoder, on the grounds that a random-walk tempo cannot follow tempo change.
// The drift/jitter split above withdraws that: tempo change is not what fails,
// and a decoder that produces a *smooth* local pulse would sit nearer the
// 84.2%/30.7% steady-pulse bound than to what this filter already does. If one
// is tried, it should be tried for the octave, where the same evidence still
// favours a window over a recursion — `ActivationTempo` beats the filter 81.7%
// to 66.6% on ballroom at choosing the level — and not for recall.
//
// A caution for anyone sweeping anything else here: an oracle observation
// removes precisely the pressure most of these settings exist to resist, so a
// parameter tuned under one is tuned against a threat model that does not
// exist. Attribute loss with the oracle; never choose a value with it.
//
// A caution about the first row. It is a *lower* bound on what is extractable,
// not an upper one: it ignores periodicity entirely, which is the whole
// advantage a tracker has, and on GTZAN the shipping path duly beats it. On SMC
// the shipping path is ten points *below* peak height alone — periodicity
// machinery actively destroying information that was present.
//
// Do not quote "a peak exists within 70 ms of the beat", which is 98.7%. Random
// times score 78.7% on the same test: the activation carries 8.7 local maxima a
// second against 1.8 beats, at a median height of 0.0018, so that measure reads
// the density of the noise floor. Every number above is reported against its own
// null for this reason.
//
// Material without percussion remains a separate case: on SMC this path is
// usable on 3.2% of recordings, with 94.9% of *all* its recordings failing on
// wrong beats and 94.5% on too few. The same symptom follows from a decoder
// that
// cannot hold a sparse pulse, and only an A/B on identical activations tells
// them apart. It is the material BeatNet+ claims to address, which makes it
// worth the A/B rather than worth assuming.
//
// **Listening before answering has no headroom, and the ceiling says so.**
// The proposal is reasonable and recurs: not every song has an audible beat in
// its first bar, so buffer some seconds of microphone and orient on them
// before committing. Its ceiling can be measured without building any of it,
// by handing the tracker the tempo an offline analysis of the *whole* file
// found — strictly more than any buffer could recover:
//
//     seeded with the whole file's tempo   ballroom*   GTZAN   SMC*
//     no                                     57.4%     42.6%   3.2%
//     yes                                    56.6%     44.0%   2.8%
//
// A point and a half on GTZAN, and it moves in different directions on
// different corpora. The same experiment on spectral flux was already
// negative; this replicates it with the front end that works, which is what
// makes it worth believing rather than a property of the old evidence.
//
// The reason is `anchor_tempo`, and it is structural rather than a shortfall.
// The activation-tempo estimator re-aims the prior from a six-second window,
// and applies it every frame, so any tempo put into the cloud at the start is
// gone within six seconds of audio whether it was right or not. Seeding a
// tracker that continuously re-anchors cannot do anything by construction —
// which is worth knowing before building a buffer to do it more slowly.
//
// What the ceiling does not test is the *phase*: seedTempo concentrates the
// cloud on a tempo and says nothing about where the beat falls. That half is
// still open. But note which failures are actually left — wrong octave on 27%
// of ballroom and 38% of GTZAN, *after* being told the tempo, and slightly
// worse than without. The octave is not failing for want of knowing the tempo.
// It is failing because the tracker leaves the tempo it was given, which puts
// this back with everything else in this comment: the decoder.

// A live configuration for a capture rate, with the front-end sized in
// milliseconds rather than samples.
//
// The ODF's defaults are 2048/512 *samples*, which at 48 kHz is a 43 ms window
// every 11 ms and at 22 kHz is 93 ms every 23 ms — the same numbers describing
// a front-end twice as coarse. The filter is tuned against how much onset
// energy a beat is worth per frame, so the coarser front-end quietly halves the
// evidence per beat while the charge per predicted beat stays put, and the
// tracker that was steady at 48 kHz wanders at 22. Scaling both with the rate
// keeps the tracker's world the same whatever the device hands it.
LiveConfig liveConfigFor(double sample_rate);

// The microphone path, whole: audio in, beat predictions out.
//
// This is the online counterpart of analysis::OfflineAnalyzer and is composed
// here rather than in each shell for the same reason render::Metronome is —
// the shells differ in how they obtain a buffer and a clock, and must not
// differ in what happens between them.
//
// Two things it owns that a shell would otherwise have to reinvent:
//
// *Level normalisation.* The particle filter's observation gain is a constant,
// so what it multiplies must not depend on how loud the room is.
//
// *Own-click gating.* A metronome listening through a microphone hears its own
// click, and a click is the most onset-like sound there is. Left alone the
// tracker locks onto itself: confidence goes to one, the tempo stops responding
// to the music, and nothing about the output looks wrong. The click cannot be
// subtracted — the room's response to it is unknown — so instead the tracker
// declines to look during the window it occupies. That costs information but
// does not bias the filter, because the observation is zero-mean: a dropped
// frame changes no weights at all, while a frame merely ignored by a
// single-hypothesis tracker would still shift its estimate.
//
// Real-time safe: process() allocates nothing and reads no clock.
class LiveTracker {
public:
    explicit LiveTracker(const LiveConfig& config);

    // With a learned front end instead of spectral flux.
    //
    // `weights` must be valid() and must outlive the tracker; the shell owns
    // the bytes, because the core does no I/O. Everything else about the
    // tracker is the same object it was — the same filter, the same gating, the
    // same publishing thresholds — and only the evidence differs. That is not
    // a coincidence of the implementation, it is the point: it is what makes
    // the measured before and after comparable.
    //
    // Measured on 107 produced recordings against reference beats, the causal
    // tracker's accuracy and coverage improve materially without changing its
    // thresholds. See ml/beatnet.hpp.
    //
    // Not the default. Spectral flux costs a few hundred kFLOP a second and
    // this costs tens of MFLOP plus 1.6 MB of weights, and which of those a
    // given device should spend is a decision that needs measurements from that
    // device, not from a workstation.
    LiveTracker(const LiveConfig& config, const ml::BeatNetWeights& weights);

    const LiveConfig& config() const { return config_; }

    // True when the learned front end is the one feeding the filter.
    bool usingModel() const { return model_.has_value(); }

    // Feeds captured audio. `stream_time_sec` is the time of samples[0], in the
    // same clock the shell schedules output in.
    void process(double stream_time_sec, const float* samples, std::size_t n);

    // Feeds one already-computed observation instead of audio, at a time in
    // the same clock: how much this instant looks like a beat, 0 to 1.
    //
    // This is the seam a learned front end arrives through, and it is here
    // rather than in the research harness because that is where the front end
    // is going. The built-in onset function is spectral flux, and measured
    // against a reference on 106 produced recordings it does not concentrate
    // on the beat: the filter's own coincidence term sat at 0.226 where
    // perfect tracking of that same evidence could only have reached 0.39, so
    // the gate stayed shut on the material the product exists for. Fed a
    // causal model's activation instead — same filter, same recordings — that
    // term reaches 0.535 and the lock rate goes from 1% of tracks to 45%.
    //
    // Everything downstream is unchanged and deliberately so: gating, level
    // normalisation and the publishing hysteresis are the tracker's, and only
    // the evidence is swapped. Callers use one of process() or observe(), not
    // both — mixing them feeds the filter two clocks and two scales.
    void observe(double time_sec, double activation);

    // Tells the tracker when its own click will reach the microphone — that is
    // the moment the click is *heard*, output latency and room delay already
    // added by the caller. The core cannot compute it: only the shell knows
    // what the round trip measured.
    void gateClick(double heard_time_sec);

    BeatEstimate estimate(double now_sec) const { return filter_.estimate(now_sec); }

    // What the autocorrelation over the activation history currently makes of
    // the tempo, whether or not it is being used. Reported separately from
    // estimate() because it answers a different question — which metrical
    // level the recording is in, rather than where the next beat falls — and
    // because a bench that cannot see both cannot tell which of the two is
    // wrong when the beats are.
    ActivationTempoEstimate tempoFromActivation() const {
        return activation_tempo_.estimate();
    }

    // Hands out the next beat to play, once, when it comes within
    // `lookahead_sec` of now. True when `beat_sec` was written.
    //
    // A beat, once handed out, is never revised: by then the click is in a
    // buffer on its way to the device, and moving it would be a click that
    // stutters rather than a click that corrects. Refinements land on the beat
    // after it.
    bool takeBeat(double now_sec, double lookahead_sec, double* beat_sec);

    // Concentrates the cloud on a known tempo — an offline analysis of the same
    // song, or a tempo the user typed.
    void seedTempo(double bpm, double spread_octaves = 0.05);

    // Manual mode: the tempo is the user's and the room is asked only where the
    // beat falls. Zero goes back to tracking the tempo too.
    //
    // This is a different promise from auto mode, and the difference is worth
    // being explicit about, because it is what the mode is for:
    //
    // - Nothing is played until the room has been heard. The user sets a tempo
    //   and starts; the click waits, catches the first phrase, and falls in on
    //   it. That waiting is the feature — a metronome that starts on the beat
    //   the user's own count-in landed on needs no count-in of its own.
    //
    // - Once it has fallen in, it does not stop. In auto mode a room that goes
    //   quiet has taken the tempo with it, so the tracker coasts and eventually
    //   gives up; here the tempo was never the room's to take. The click keeps
    //   the user's BPM through a silent bar, a solo, a cough, indefinitely.
    //   That falls out of the filter rather than being special-cased: with the
    //   period pinned and the observation zero-mean, silence moves no weights
    //   and the grid simply continues.
    //
    // The tempo is taken as given even outside the configured BPM range — see
    // BeatParticleFilter::pinPeriod.
    void setManualTempo(double bpm);
    double manualTempo() const { return manual_bpm_; }

    // Manual mode, still waiting for something to synchronise to. What a shell
    // shows as "listening…", and the reason no beats are coming out.
    bool waiting() const { return manual_bpm_ > 0.0 && !acquired_; }

    // How concentrated the room's onsets are at one phase, 0..1. Manual mode
    // only, and useful mainly as the meter behind that "listening…".
    double syncStrength() const { return sync_.strength(); }

    void reset();

    struct Stats {
        std::size_t frames = 0;          // ODF frames produced
        std::size_t gated = 0;           // frames withheld, our own click
        std::size_t beats = 0;           // beats handed out
        std::size_t beats_late = 0;      // predicted beats already in the past
        std::size_t discontinuities = 0; // capture buffers that did not follow
        BeatParticleFilter::Stats filter;
    };

    Stats stats() const;

private:
    // The filter's beat window, widened if the ODF is too coarse to support the
    // configured one. See the definition.
    static ParticleFilterConfig resolveFilter(const LiveConfig& config);

    bool gatedAt(double frame_time_sec) const;

    // Feeds one already-normalised observation to the filter and, in manual
    // mode, to the phase correlator. What process() and observe() share once
    // each has produced a number the filter can use.
    void submit(double time_sec, double normalised);

    LiveConfig config_;
    dsp::Odf odf_;
    BeatParticleFilter filter_;
    PhaseSync sync_;
    ActivationTempo activation_tempo_;

    // Engaged only by the constructor that was handed weights. Held by value
    // rather than behind a pointer so that the audio path has no indirection
    // and no chance of a null to check.
    std::optional<ml::BeatNetActivation> model_;

    // Half the width of one evidence window, for gating. The two front ends
    // disagree about it — the ODF's frame is the configured one, the model's is
    // a fixed 64 ms — and a gate measured against the wrong one either lets the
    // click through or blinds the tracker either side of it.
    double evidence_half_sec_ = 0.0;

    double manual_bpm_ = 0.0;
    bool acquired_ = false;

    double origin_sec_ = 0.0;  // stream time of the ODF's sample zero
    std::size_t consumed_ = 0;
    bool started_ = false;

    double onset_peak_ = 0.0;

    // A handful of pending gates is plenty: they are consumed within a beat of
    // being added, and a shell that has queued eight of them is not running.
    static constexpr std::size_t kGates = 8;
    std::array<double, kGates> gate_start_{};
    std::array<double, kGates> gate_end_{};
    std::size_t gate_next_ = 0;

    bool locked_ = false;
    bool published_ = false;
    // Far enough back that the first beat of a stream is never mistaken for a
    // repeat of one that was never handed out.
    double last_beat_sec_ = -std::numeric_limits<double>::infinity();
    double held_period_sec_ = 0.5;

    Stats stats_;
};

}  // namespace tiktak::tracking
