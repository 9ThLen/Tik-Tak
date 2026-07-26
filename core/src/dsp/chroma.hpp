#pragma once

#include <cstddef>
#include <vector>

namespace tiktak::dsp {

// Twelve-bin pitch class profile over a real FFT magnitude spectrum.
//
// Chroma is here for one reason: harmony changes on the bar line far more often
// than anywhere else inside a bar, and no amount of onset energy can see that.
// The low band hears the kick, the ODF hears the accent — both are rhythm. A
// chord change is the one downbeat cue that is not.
//
// Folding every octave onto the same twelve bins is what makes it a harmony
// measure rather than a spectrum: a chord voiced high and the same chord voiced
// low are the same chord, and should compare as identical.
//
// Two decisions worth stating, because both discard information deliberately:
//
//   Bins that fall *between* semitones are attenuated, not assigned. A bin
//   halfway between C and C# is not evidence for either — it is noise, a
//   transient, or a drum. Weighting by squared cosine of the deviation lets
//   tuned material through and quietly drops the rest, which is what keeps a
//   drum fill from reading as a chord change.
//
//   The range stops well below Nyquist. High partials of one note land on the
//   pitch classes of other notes, so a wide band does not measure more harmony,
//   it measures more smear. Everything above the top of the treble staff is
//   almost entirely overtones of something already counted lower down.
//
// And one limit that is imposed rather than chosen — see resolvedMinHz(). A
// linear-frequency transform has the same absolute resolution everywhere, while
// a semitone gets narrower the lower it goes, so below some frequency
// neighbouring semitones share bins and no weighting can separate them. The
// bottom of the requested range is raised to wherever that is, because the
// alternative is to return a confident answer computed from smear.
//
// Filters are precomputed once; apply() allocates nothing.
class ChromaFilterbank {
public:
    // `fftSize` is the transform length; the spectrum passed to apply() has
    // fftSize/2 + 1 bins. Bins outside [minHz, maxHz] are ignored; maxHz is
    // clamped to Nyquist.
    // `floorRel` is how much of the spectrum's energy has to land on tuned
    // pitches inside the range before the frame is called harmony at all.
    // Below it the profile comes back as zeros rather than being normalised —
    // see apply().
    ChromaFilterbank(std::size_t fftSize, double sampleRate, double minHz = 55.0,
                     double maxHz = 2093.0, double floorRel = 1e-4);

    static constexpr std::size_t kBins = 12;

    // The lowest frequency at which this transform can still tell one semitone
    // from the next: where a semitone is at least `binsPerSemitone` FFT bins
    // wide. A semitone around f spans about 0.058·f Hz and a bin is
    // sampleRate/fftSize wide, so the two cross at a frequency proportional to
    // the bin width — halve the hop and this halves with it.
    //
    // Worth knowing before configuring the front end: at 48 kHz a 2048-sample
    // window resolves nothing below roughly 800 Hz, so chroma from it is built
    // from the upper partials of the harmony rather than its roots. That is
    // usable — the partials move when the chord does — but a dedicated longer
    // window, or a constant-Q transform, is the real fix and is not here.
    static double resolvedMinHz(std::size_t fftSize, double sampleRate,
                                double binsPerSemitone = 2.0);

    // The range actually in use, after the low end was raised to what the
    // transform can resolve.
    double minHz() const { return minHz_; }
    double maxHz() const { return maxHz_; }

    std::size_t spectrumSize() const { return fftSize_ / 2 + 1; }
    // How many FFT bins actually contribute. Zero means the range fell outside
    // the spectrum entirely and apply() will only ever produce silence.
    std::size_t contributingBins() const { return bin_.size(); }

    // Projects a magnitude spectrum onto twelve pitch classes in `out`, which
    // must have room for kBins values. Bin 0 is C.
    //
    // The result is L2-normalised, so it says which pitches are present and not
    // how loud they are — the caller compares consecutive frames, and a
    // crescendo is not a chord change. A frame whose tuned energy is a
    // negligible fraction of everything in the spectrum comes back as twelve
    // zeros instead: normalising there would scale spectral leakage up to full
    // scale and report a confident chord made of nothing.
    void apply(const float* magnitude, float* out) const;

private:
    std::size_t fftSize_;
    double minHz_ = 0.0;
    double floorRel_ = 0.0;
    double maxHz_ = 0.0;
    std::vector<std::size_t> bin_;     // contributing FFT bin indices
    std::vector<std::size_t> pitch_;   // pitch class each one lands on, 0 = C
    std::vector<float> weight_;        // how squarely it sits on that semitone
};

// How much two pitch class profiles differ, in [0, 1].
//
// Cosine distance rather than a Euclidean one: after L2 normalisation the
// profiles are directions, and the angle between them is exactly "how different
// are these two chords" with loudness already divided out. Zero when either
// side carries no tuned energy — silence is not a modulation.
double chromaDistance(const float* a, const float* b);

}  // namespace tiktak::dsp
