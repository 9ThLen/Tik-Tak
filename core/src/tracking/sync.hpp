#pragma once

namespace tiktak::tracking {

struct SyncConfig {
    // How long the correlation remembers, as an exponential half-life. Long
    // enough to hold several beats at any tempo we support, short enough that
    // an answer arrives while the singer is still on the first phrase.
    double tau_sec = 2.5;

    // Exponent on the normalised onset. Sharper than the particle filter's
    // square, and deliberately so: there, an off-beat hit merely lowers the
    // particles that missed it, while here a hit exactly between the beats sits
    // at the opposite angle and *cancels* beat energy outright. A hi-hat at 0.7
    // of the beat's level leaves a resultant of 0.18 weighed by amplitude and
    // 0.34 by energy — in both cases the answer is still correct and looks too
    // weak to act on. Cubed it is 0.49, which is convincing, and measurably so:
    // on a set of twelve scenarios spanning clean beats, subdivisions of every
    // strength, noise, silence and rooms at the wrong tempo, the square accepts
    // six of the seven it should and the cube accepts all seven, neither being
    // fooled by any of the five it should refuse.
    double onset_exponent = 3.0;

    // How concentrated the onset energy has to be at one phase before the
    // answer is worth acting on, and how far it may fall before the answer is
    // withdrawn. A resultant length, so 0 is "onsets everywhere" and 1 is "all
    // of them on one phase".
    double acquire_strength = 0.15;
    double release_strength = 0.08;

    // The concentration alone cannot say whether it means anything, and this is
    // the term that can. Onsets at random phases still add up to a resultant of
    // roughly one over the square root of how many there were, so what counts
    // as convincing depends on how many there have been — six onsets landing
    // 0.4 concentrated is exactly what chance produces, while three hundred
    // doing it would be remarkable.
    //
    // That is the Rayleigh statistic, resultant squared times the effective
    // number of onsets, and it is what stops the mode falling in with a room
    // whose beat is simply not the one the user asked for. A band at 150 while
    // the dial says 120 has no 120 phase to find: its onsets walk steadily
    // round the circle, and with only a handful of them in the window at a time
    // the walk keeps producing a resultant above any fixed threshold. Weighed
    // against how little evidence it rests on, it stops being convincing.
    //
    // It has to be above one whatever else is chosen, because one is exactly
    // what a single isolated hit scores — a resultant of one on one onset — and
    // that score does not decay, so at one a lone door slam would acquire a
    // beat and keep it.
    double min_rayleigh = 1.5;

    // How long both conditions have to keep holding before the answer is
    // handed over. A moment of coincidence is not a beat: the statistic above
    // is a random quantity, and over a few seconds of noise it crosses any
    // fixed threshold now and then simply because it wanders. What noise cannot
    // do is keep being convincing, and this is the difference — real music
    // holds the same phase indefinitely, so waiting costs it a second and costs
    // an empty room the acquisition altogether.
    double hold_sec = 1.0;

    // How far the click may be nudged from where the user's tempo puts it, per
    // beat, as a fraction of the period. Applied by tracking::LiveTracker once
    // manual mode is playing.
    //
    // This is the entire difference between "the room decides the phase" and
    // "the room decides the tempo", and without it manual mode does not keep
    // its promise. A cloud pinned to 120 listening to a room at 150 has no
    // stable phase to find — the two grids slide past each other twice a second
    // — and a tracker that published the filter's answer directly would put its
    // clicks anywhere from 0.7 to 0.9 seconds apart while insisting it was at
    // 120 BPM. Correcting by a fiftieth of a beat at a time is fast enough to
    // follow a singer drifting and far too slow to chase a tempo that is simply
    // not the one that was asked for.
    double max_drift = 0.02;

    // A gap longer than this is the stream stopping, not the music pausing.
    double max_gap_sec = 1.0;

    bool valid() const;
};

// Where the beat sits, when the tempo is already known.
//
// This is the acquisition half of manual mode: the user has given a BPM, so the
// period is not in question and the only unknown is the offset. That is a much
// smaller problem than tracking a tempo, which is why manual mode works on
// material the auto tracker cannot follow at all.
//
// The plan called for cross-correlating the last few seconds of the onset
// function against a comb of impulses at the given period and taking the
// maximum. That is the right question, and the answer to it is one complex
// number: correlating against a comb is asking how the onset energy is
// distributed *modulo the period*, and the first Fourier coefficient at that
// period is exactly that distribution's centre of mass on the circle. So there
// is no buffer to keep and no grid to search — two accumulators, decayed, and
// the phase falls out of an atan2 with no quantisation to the frame rate.
//
// Using only the fundamental makes the comb a raised cosine rather than a train
// of impulses, which is a deliberate difference and not an approximation of
// one: a sharp comb has to be sampled somewhere, and at the ODF's frame rate
// the samples land differently depending on the phase being tested. The smooth
// comb has no such preference.
//
// Its second job is deciding whether there is any music yet. That is not a
// separate question with a separate threshold — room noise has onsets but no
// phase, so the concentration measure that answers "where is the beat" already
// answers "is there one". Asking twice would only add a level threshold to get
// wrong.
//
// Real-time safe: no allocation, no clock, constant work per frame.
class PhaseSync {
public:
    PhaseSync() = default;
    explicit PhaseSync(const SyncConfig& config) : config_(config) {}

    const SyncConfig& config() const { return config_; }

    // The period to look for. Changing it discards what has been accumulated:
    // the angles were measured against the old period and mean nothing at a new
    // one. Zero stops the sync entirely.
    void setPeriod(double period_sec);
    double period() const { return period_sec_; }

    // One onset frame, normalised for level the same way the particle filter's
    // input is.
    void observe(double time_sec, double onset);

    // 0..1: the share of onset energy that sits at one phase rather than spread
    // over the period. Doubles as the confidence in the answer below.
    double strength() const;

    // How many onsets the answer effectively rests on. Not a count of frames:
    // the accumulator is a weighted sum, and one loud hit among quiet ones
    // carries the weight of several, so what matters is the weights' own
    // concentration. This is what the strength above has to be judged against.
    double evidence() const;

    // True once the phase is worth acting on, false again only when the room
    // has fallen well below that — hysteresis, so a strength hovering at the
    // threshold does not switch the metronome on and off.
    bool ready() const { return ready_; }

    // The first beat strictly after `now_sec`, on the grid found so far. Only
    // meaningful while ready().
    double nextBeat(double now_sec) const;

    void reset();

private:
    void forget();

    SyncConfig config_;

    double period_sec_ = 0.0;

    // The correlation, as a vector on the unit circle, and the onset energy it
    // was built from. Their ratio is the resultant length.
    double x_ = 0.0;
    double y_ = 0.0;
    double mass_ = 0.0;
    double mass_sq_ = 0.0;  // the same energy squared, for evidence() above

    double held_sec_ = 0.0;  // how long the acquire conditions have kept holding
    double last_time_sec_ = 0.0;
    bool started_ = false;
    bool ready_ = false;
};

}  // namespace tiktak::tracking
