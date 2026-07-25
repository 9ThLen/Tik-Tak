#!/usr/bin/env python3
"""Regenerates the decoder test fixtures.

    research/.venv/bin/python core/tests/data/generate.py

The files are committed rather than generated at build time on purpose. They
have to be produced by a real encoder — the delay MP3 adds, the block structure
FLAC uses — and requiring libsndfile to build the C++ tests would drag a
research dependency into the core's build for no gain. Regenerate only when the
fixtures need to change; the checked-in files are the contract the tests assert
against.

Kept deliberately small: the tone fixtures are half a second each, the click
track ten seconds, about 125 KB in total.
"""

import pathlib
import sys

import numpy as np
import soundfile as sf

# The synthesiser lives in the research package, which is why this script is not
# part of the C++ build.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "research"))

from tiktak.synth import make_clip  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent

SAMPLE_RATE = 22050
DURATION_SEC = 0.5
TONE_HZ = 440.0
AMPLITUDE = 0.5

CLICK_BPM = 120.0
CLICK_SECONDS = 10.0


def main() -> None:
    frames = int(DURATION_SEC * SAMPLE_RATE)
    t = np.arange(frames) / SAMPLE_RATE

    # Quantised here, not by libsndfile. Handing it float and asking for PCM_16
    # produced files that differed by one LSB between WAV and FLAC, because its
    # two writers round differently — which would have made the "lossless
    # formats agree exactly" test assert something about libsndfile instead of
    # about our decoders.
    mono = np.round(AMPLITUDE * 32767 * np.sin(2 * np.pi * TONE_HZ * t)).astype(np.int16)

    sf.write(HERE / "tone_mono.wav", mono, SAMPLE_RATE, subtype="PCM_16")
    sf.write(HERE / "tone_mono.flac", mono, SAMPLE_RATE, subtype="PCM_16")
    sf.write(HERE / "tone_mono.mp3", mono, SAMPLE_RATE)

    # Signal in the left channel only: a correct mono downmix averages and so
    # returns exactly half, while a summing one would return the whole thing.
    stereo = np.stack([mono, np.zeros_like(mono)], axis=1)
    sf.write(HERE / "tone_stereo.wav", stereo, SAMPLE_RATE, subtype="PCM_16")

    # A real encoded track for the end-to-end test: decode and analyse, the
    # actual job the file path exists to do. MP3 because it is both the format
    # users will import and the one that could plausibly break the analysis —
    # its pre-echo lands exactly on the transients the onset function looks for.
    clip = make_clip(bpm=CLICK_BPM, duration_sec=CLICK_SECONDS,
                     sample_rate=SAMPLE_RATE, seed=42)
    audio = np.round(clip.audio * 32767 * 0.8).astype(np.int16)
    sf.write(HERE / "click_120.mp3", audio, clip.sample_rate)
    print(f"click_120.mp3 truth: {clip.bpm:g} BPM, {len(clip.beats)} beats, "
          f"first at {clip.beats[0]:.3f} s")

    for path in sorted(p for p in HERE.iterdir() if p.suffix != ".py"):
        info = sf.info(path)
        print(f"{path.name:18} {path.stat().st_size:7d} B  {info.samplerate} Hz  "
              f"{info.channels} ch  {info.frames} frames  {info.subtype}")


if __name__ == "__main__":
    main()
