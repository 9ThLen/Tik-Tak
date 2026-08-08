#!/usr/bin/env python3
"""Score a recording made in an actual room against the file it was played from.

`room_degradation.py` sweeps reverberation and noise in simulation. This takes
the real thing: someone played a corpus recording through a speaker, captured it
with a microphone, and the capture is scored against the same annotations the
clean baseline used. One room is one sample, so the number that matters is the
**gap** against the clean run on the same recording — and, once there is a gap,
which simulated cell it corresponds to, which is what turns the rest of the
sweep from a guess into an extrapolation from a measured point.

## Alignment is the whole risk

The annotations belong to the original file and were not moved, so a capture
that is offset or stretched against it scores a time shift rather than a room.
Two things are therefore checked before any beat is scored, and either one
failing voids the recording:

**Offset.** Found by cross-correlating onset envelopes rather than waveforms.
A room inverts phase, colours the spectrum and adds a tail; the waveform
correlation of a reverberant capture against its source is weak and can peak in
the wrong place, while the envelope survives all three.

**Drift.** The offset is fitted twice — over the first third and over the last
third — and the two must agree. Playback and capture run off different clocks,
and a part-per-thousand difference is 0.2 s over a three-minute song, which is
three beats. A constant offset is fine and is corrected; a drifting one is not
correctable by a constant and the run is void.

## What it does not do

It does not resample the capture to remove drift. That would be repairing the
measurement rather than reporting it, and a setup whose clocks disagree is a
fact about the setup that the log should carry.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from eval.analysis import Estimate  # noqa: E402
from eval.live_corpus_benchmark import _score_one, load_corpus  # noqa: E402

SAMPLE_HZ = 50.0
# The envelope the alignment runs on. 100 Hz is ten times finer than a beat at
# 200 BPM and coarse enough that a three-minute correlation is instant.
ENVELOPE_HZ = 100.0
# Above this, the two ends of the recording disagree about where it starts by
# more than a beat at any sane tempo, and no constant offset describes it.
MAX_DRIFT_SEC = 0.030
# How late playback may plausibly have started after the recorder did. Past
# this a correlation peak is a repeat of the music, not a late start.
MAX_OFFSET_SEC = 30.0


def read_audio(path: pathlib.Path) -> tuple[np.ndarray, float]:
    """Mono float samples and the rate, from anything ffmpeg can open.

    Phones record AAC in an .m4a, which libsndfile does not decode and which
    dr_libs does not either — `docs/PLAN.md` puts m4a on the platform side
    deliberately, behind AVAssetReader and MediaCodec. Nothing in the product
    depends on decoding it here; this is a research path reading a file a phone
    produced, so ffmpeg is the right tool and the fallback is explicit rather
    than a mysterious failure.
    """
    import soundfile

    try:
        audio, rate = soundfile.read(str(path), dtype="float32", always_2d=True)
        return audio.mean(axis=1), float(rate)
    except Exception:  # noqa: BLE001 -- format, not logic
        pass

    with tempfile.TemporaryDirectory() as directory:
        decoded = pathlib.Path(directory) / "decoded.wav"
        done = subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(path),
             "-ac", "1", "-c:a", "pcm_s16le", str(decoded)],
            capture_output=True, text=True, check=False)
        if done.returncode != 0 or not decoded.exists():
            raise RuntimeError(
                f"cannot decode {path.name}: {done.stderr.strip()[:200]}")
        audio, rate = soundfile.read(str(decoded), dtype="float32",
                                     always_2d=True)
        return audio.mean(axis=1), float(rate)


def match_name(stem: str, names: list[str]) -> str | None:
    """Which corpus recording a capture file is of.

    Capture filenames are typed by a person on a phone, so they carry dropped
    digits, doubled letters and spaces for underscores. Matching is therefore
    fuzzy — and the caller prints what matched what, because a fuzzy match that
    nobody reads is a silent way to score the wrong recording.
    """
    import difflib

    def normalise(text: str) -> str:
        return "".join(c for c in text.lower() if c.isalnum())

    table = {normalise(name): name for name in names}
    key = normalise(stem)
    if key in table:
        return table[key]
    close = difflib.get_close_matches(key, list(table), n=2, cutoff=0.75)
    if not close:
        return None
    if len(close) > 1:
        best = difflib.SequenceMatcher(None, key, close[0]).ratio()
        second = difflib.SequenceMatcher(None, key, close[1]).ratio()
        # Two plausible corpus entries for one file is not a match, it is a
        # coin toss, and a coin toss here scores against the wrong annotations.
        if best - second < 0.05:
            return None
    return table[close[0]]


def envelope(mono: np.ndarray, rate: float) -> np.ndarray:
    """A crude onset envelope: rectified first difference of a smoothed level.

    Resampled onto an **exact** ENVELOPE_HZ grid rather than left on the hop
    grid, and both of those words are load-bearing. `rate / step` is only
    approximately ENVELOPE_HZ — at 22.05 kHz a 220-sample hop is 100.23 Hz, and
    reading the correlation index as if it were 100 Hz puts a 4.7 s offset out
    by 10 ms. Worse, the corpus is 22.05 kHz and a phone records at 44.1 or 48,
    so the two signals would land on grids of *different* pitch and the
    correlation would measure the mismatch. A common exact grid removes both.
    """
    step = max(1, int(rate / ENVELOPE_HZ))
    trimmed = mono[: len(mono) // step * step]
    if len(trimmed) == 0:
        return np.zeros(0, dtype=np.float64)
    level = np.sqrt(np.mean(trimmed.reshape(-1, step) ** 2, axis=1))
    hop_hz = rate / step
    out = np.diff(level, prepend=level[:1])
    np.maximum(out, 0.0, out=out)

    duration = len(out) / hop_hz
    grid = np.arange(0.0, duration, 1.0 / ENVELOPE_HZ)
    out = np.interp(grid, np.arange(len(out)) / hop_hz, out)

    peak = float(np.max(out)) if len(out) else 0.0
    return out / peak if peak > 0 else out


def correlation_curve(reference: np.ndarray, capture: np.ndarray,
                      max_lag_sec: float = MAX_OFFSET_SEC) -> np.ndarray:
    """Normalised cross-correlation against lag, bounded and unit-scaled.

    Bounded because a short reference window correlated against a whole capture
    will happily match a later chorus: music repeats, and unbounded, windows
    from these recordings returned 120, 99 and 79 seconds where the answer was
    0.76. Unit-scaled so that windows of different loudness contribute equally
    when several are summed.
    """
    if len(reference) == 0 or len(capture) == 0:
        return np.zeros(0, dtype=np.float64)
    size = 1
    while size < len(reference) + len(capture):
        size *= 2
    spectrum = (np.fft.rfft(capture, size)
                * np.conj(np.fft.rfft(reference, size)))
    correlation = np.fft.irfft(spectrum, size)
    limit = min(len(capture), max(2, int(max_lag_sec * ENVELOPE_HZ)))
    curve = correlation[:limit]
    norm = float(np.linalg.norm(reference) * np.linalg.norm(capture))
    return curve / norm if norm > 0 else curve


def refine(curve: np.ndarray) -> tuple[float, float]:
    """Peak lag in seconds and its height, with sub-grid parabolic refinement.

    Without the refinement the answer is quantised to the envelope's 10 ms,
    which is a seventh of the 70 ms beat tolerance and lands on the same side
    every time, so it would be a bias rather than noise.
    """
    if len(curve) == 0:
        return (0.0, 0.0)
    index = int(np.argmax(curve))
    peak = float(curve[index])
    shift = 0.0
    if 0 < index < len(curve) - 1:
        left, right = curve[index - 1], curve[index + 1]
        denominator = left - 2.0 * peak + right
        if denominator != 0.0:
            shift = float(np.clip(0.5 * (left - right) / denominator, -0.5, 0.5))
    return ((index + shift) / ENVELOPE_HZ, peak)


def align(original: np.ndarray, capture: np.ndarray, rate_a: float,
          rate_b: float, beat_sec: float = 0.0) -> dict:
    """One constant offset, found by summing the evidence from every window.

    The offset is the same everywhere by construction — the recorder started
    once — so the windows are not independent estimates to be averaged, they
    are evidence about one number. Summing their correlation curves and taking
    one peak lets a consistent lag reinforce across all of them while a
    spurious match reinforces nothing, because different windows land on
    different repeats.

    Taking a per-window argmax and then a median does not work, and was tried:
    on music with a verse structure most windows can each pick a *different*
    wrong repeat, and a median of wrong answers is a wrong answer with a
    majority behind it.

    The per-window peaks are still computed, but only as diagnosis:

    * residuals near a whole number of beats are *slips*, the artifact of
      correlating periodic material;
    * a residual growing monotonically with position is *drift*, two clocks
      disagreeing, which no constant offset can describe.
    """
    a = envelope(original, rate_a)
    b = envelope(capture, rate_b)

    windows = 7
    span = len(a) // (windows + 1)
    curves, fits, centres = [], [], []
    for index in range(windows):
        start = index * span
        stop = start + 2 * span
        if span <= 0 or stop > len(a) or start >= len(b):
            continue
        curve = correlation_curve(a[start:stop], b[start:])
        if len(curve) == 0:
            continue
        curves.append(curve)
        fits.append(refine(curve)[0])
        centres.append((start + span) / ENVELOPE_HZ)

    if not curves:
        return {"offset_sec": 0.0, "quality": 0.0, "fits": [], "windows": 0,
                "agreeing_windows": 0, "slipped_windows": 0,
                "drift_sec": float("inf"), "drift_ok": False,
                "note": "too short to fit"}

    width = min(len(c) for c in curves)
    total = np.sum([c[:width] for c in curves], axis=0)
    offset, quality = refine(total)

    fits = np.asarray(fits)
    centres = np.asarray(centres)
    residual = fits - offset
    agreeing = np.abs(residual) <= 0.030

    slipped = np.zeros(len(fits), dtype=bool)
    if beat_sec > 0.0:
        beats_off = residual / beat_sec
        slipped = (np.abs(beats_off - np.round(beats_off)) < 0.25) & ~agreeing

    drift_total = 0.0
    if agreeing.sum() >= 3:
        slope = float(np.polyfit(centres[agreeing], fits[agreeing], 1)[0])
        drift_total = abs(slope) * float(centres[-1] - centres[0])

    return {
        "offset_sec": offset,
        "quality": quality / len(curves),
        "fits": [float(f) for f in fits],
        "windows": int(len(fits)),
        "agreeing_windows": int(agreeing.sum()),
        "slipped_windows": int(slipped.sum()),
        "drift_sec": drift_total,
        # A third of the windows agreeing with the summed answer is enough: the
        # answer came from all of them coherently, and the count is a check on
        # it rather than the estimator.
        "drift_ok": bool(agreeing.sum() * 3 >= len(fits)
                         and drift_total <= MAX_DRIFT_SEC),
    }


def measure_one(item: dict, capture_path: pathlib.Path, binary: pathlib.Path,
                model: pathlib.Path) -> dict:
    import soundfile

    original, rate_a = read_audio(pathlib.Path(item["audio"]))
    capture, rate_b = read_audio(capture_path)

    from eval.live_corpus_benchmark import load_reference_beats
    reference = load_reference_beats(item["annotation"])
    beat_sec = (float(np.median(np.diff(reference)))
                if len(reference) > 4 else 0.0)
    alignment = align(original, capture, float(rate_a), float(rate_b), beat_sec)
    offset = alignment["offset_sec"]

    # Trimmed so sample zero of the written file is sample zero of the original,
    # which is what makes the untouched annotations apply.
    start = int(round(offset * rate_b))
    trimmed = capture[max(0, start):]
    alignment["captured_sec"] = len(trimmed) / float(rate_b)

    out: dict = {"name": item["name"], "corpus": item["corpus"],
                 "capture": str(capture_path), "alignment": alignment}
    if not alignment["drift_ok"]:
        out["void"] = ("no constant offset fits: see agreeing_windows against "
                       "windows, and slipped_windows for whether the dissent is "
                       "a periodic-correlation slip or a clock disagreement")
        return out

    with tempfile.TemporaryDirectory() as directory:
        path = pathlib.Path(directory) / "aligned.wav"
        soundfile.write(str(path), trimmed, int(rate_b))
        arms = {"room": str(path), "clean": str(item["audio"])}
        for arm, audio in arms.items():
            done = subprocess.run(
                [str(binary), audio, "--live", "--live-model", str(model),
                 "--live-sample-hz", repr(SAMPLE_HZ)],
                capture_output=True, text=True, encoding="utf-8",
                errors="replace", check=False)
            if done.returncode != 0:
                raise RuntimeError(done.stderr.strip()[:200])
            scored = _score_one(item, "model", binary, model,
                                estimate=Estimate.from_json(json.loads(done.stdout)))
            out[arm] = {
                "usable": bool(scored.get("usable", False)),
                "reasons": list(scored.get("reasons", [])),
                "f_measure": scored.get("f_measure"),
                "p70": scored.get("p70"),
                "r70": scored.get("r70"),
                "acquired_at": scored.get("acquired_at"),
                "switches": scored.get("switches"),
                "correct_share_of_eligible": scored.get("correct_share_of_eligible"),
            }
    return out


def main(argv: list[str] | None = None) -> int:
    repository = pathlib.Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--music", type=pathlib.Path, required=True)
    parser.add_argument("--corpora", nargs="+", default=["harmonix"])
    parser.add_argument("--binary", type=pathlib.Path, required=True)
    parser.add_argument("--model", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument(
        "--captures", type=pathlib.Path, required=True,
        help="directory of room recordings, each named <track>.wav")
    parser.add_argument("--notes", type=str, default="",
                        help="speaker, microphone, room, distance, levels")
    args = parser.parse_args(argv)

    items = {i["name"]: i for i in
             load_corpus(args.manifest, args.music, False, frozenset(args.corpora))}
    captures = sorted(p for p in args.captures.iterdir()
                      if p.suffix.lower() in {".wav", ".m4a", ".mp3", ".flac",
                                              ".aac", ".ogg", ".opus"})
    if not captures:
        print(f"no audio files in {args.captures}", file=sys.stderr)
        return 1

    records, failures = [], []
    matched: dict[str, str] = {}
    for capture in captures:
        name = match_name(capture.stem, list(items))
        item = items.get(name) if name else None
        if item is None:
            failures.append({"capture": capture.name,
                             "error": f"{capture.stem} matches no corpus entry"})
            continue
        matched[capture.name] = name
        # Printed, always: a fuzzy match nobody reads is a silent way to score
        # against the wrong annotations.
        print(f"  {capture.name}  ->  {name}", file=sys.stderr)
        try:
            records.append(measure_one(item, capture, args.binary, args.model))
        except Exception as error:  # noqa: BLE001
            failures.append({"capture": capture.name, "error": str(error)[:300]})
        print(f"{len(records) + len(failures)}/{len(captures)}",
              file=sys.stderr, flush=True)

    scored = [r for r in records if "room" in r]
    summary = {}
    if scored:
        for arm in ("clean", "room"):
            summary[arm] = {
                "n": len(scored),
                "usable_rate": float(np.mean([r[arm]["usable"] for r in scored])),
                "mean_f": float(np.mean([r[arm]["f_measure"] for r in scored])),
                "mean_r70": float(np.mean([r[arm]["r70"] for r in scored])),
                "mean_p70": float(np.mean([r[arm]["p70"] for r in scored])),
            }
        summary["voided"] = len([r for r in records if "void" in r])

    payload = {
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                 text=True, cwd=repository).stdout.strip(),
        "notes": args.notes, "captures": str(args.captures),
        "matched": matched,
        "failures": failures, "summary": summary, "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
