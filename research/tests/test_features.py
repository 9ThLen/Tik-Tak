"""The feature dump round-trips, and says so when it does not.

A dump that is silently wrong — halves swapped, a frame short, the wrong band
count — would not fail anywhere downstream. It would produce a peak map, and a
number, and the number would be about nothing. So the format is tested before
anything reads it for a result.
"""
import pathlib
import struct
import subprocess

import numpy as np
import pytest

from eval.features import BYTE_ORDER, MAGIC, VERSION, read

ROOT = pathlib.Path(__file__).resolve().parents[2]
BINARY = ROOT / "tools/eval/build/RelWithDebInfo/dump_analysis.exe"
if not BINARY.exists():
    BINARY = ROOT / "tools/eval/build/dump_analysis"

HEADER = struct.Struct("<7I2d")


def write_header(**over) -> bytes:
    fields = dict(magic=MAGIC, version=VERSION, order=BYTE_ORDER, value_bytes=4,
                  frames=2, filters=3, deltas=3, frame_rate=50.0,
                  sample_rate=48000.0)
    fields.update(over)
    head = HEADER.pack(fields["magic"], fields["version"], fields["order"],
                       fields["value_bytes"], fields["frames"],
                       fields["filters"], fields["deltas"],
                       fields["frame_rate"], fields["sample_rate"])
    body = np.arange(fields["frames"], dtype="<f8").tobytes()
    width = fields["filters"] + fields["deltas"]
    body += np.arange(fields["frames"] * width, dtype="<f4").tobytes()
    return head + body


def test_round_trip(tmp_path):
    path = tmp_path / "ok.ttfd"
    path.write_bytes(write_header())
    got = read(path)
    assert got.frames == 2
    assert got.filterbank.shape == (2, 3)
    assert got.difference.shape == (2, 3)
    assert got.frame_rate == 50.0
    # The halves are split at the seam and not interleaved: row 0 is
    # 0,1,2 | 3,4,5, so the first difference value is 3 and not 1.
    assert got.filterbank[0].tolist() == [0.0, 1.0, 2.0]
    assert got.difference[0].tolist() == [3.0, 4.0, 5.0]


@pytest.mark.parametrize("field,value,message", [
    ("magic", 0xDEADBEEF, "not a feature dump"),
    ("order", 0x04030201, "byte order"),
    ("version", VERSION + 1, "version"),
    ("value_bytes", 8, "float32"),
])
def test_rejects_a_header_it_cannot_trust(tmp_path, field, value, message):
    path = tmp_path / "bad.ttfd"
    path.write_bytes(write_header(**{field: value}))
    with pytest.raises(ValueError, match=message):
        read(path)


def test_rejects_a_truncated_file(tmp_path):
    path = tmp_path / "short.ttfd"
    path.write_bytes(write_header()[:-4])
    with pytest.raises(ValueError, match="header describes"):
        read(path)


@pytest.mark.skipif(not BINARY.exists(), reason="dump_analysis is not built")
def test_the_tool_writes_what_the_reader_expects(tmp_path):
    # Two seconds of a click every half second, as raw f32 at 48 kHz — the
    # tool's second input form, so the test needs no audio file.
    rate = 48000
    audio = np.zeros(rate * 2, dtype="<f4")
    audio[::rate // 2] = 1.0
    clip = tmp_path / "clicks.f32"
    clip.write_bytes(audio.tobytes())

    out = tmp_path / "features.ttfd"
    done = subprocess.run([str(BINARY), str(clip), str(rate),
                           "--dump-features", str(out)],
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr

    got = read(out)
    assert got.sample_rate == rate
    assert got.frame_rate == pytest.approx(50.0)
    assert got.filterbank.shape[1] == 136
    assert got.difference.shape[1] == 136
    # 50 frames a second over two seconds, give or take the framing at the end.
    assert 90 <= got.frames <= 101
    assert got.times[0] == pytest.approx(0.0, abs=0.03)
    assert np.all(np.diff(got.times) > 0)
    # The difference half is a positive rise, by construction.
    assert np.all(got.difference >= 0.0)
    assert np.isfinite(got.filterbank).all()
    # Clicks are transients: the rise has to actually rise somewhere, or the
    # halves are swapped and the test above would not have caught it.
    assert got.difference.sum() > 0.0


@pytest.mark.skipif(not BINARY.exists(), reason="dump_analysis is not built")
def test_the_dump_is_deterministic(tmp_path):
    rate = 22050
    rng = np.random.default_rng(20260809)
    audio = rng.standard_normal(rate).astype("<f4") * 0.1
    clip = tmp_path / "noise.f32"
    clip.write_bytes(audio.tobytes())

    digests = []
    for run in range(2):
        out = tmp_path / f"run{run}.ttfd"
        done = subprocess.run([str(BINARY), str(clip), str(rate),
                               "--dump-features", str(out)],
                              capture_output=True, text=True)
        assert done.returncode == 0, done.stderr
        digests.append(out.read_bytes())
    assert digests[0] == digests[1]
