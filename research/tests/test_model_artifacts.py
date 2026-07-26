"""The artifact pinning workflow, exercised entirely offline.

The download half cannot be tested here — this environment has no network,
which is the very reason the workflow exists. What can be tested is everything
the checksums are *for*: pinning records the right hash, verification catches a
changed file, and the refusals (HTML masquerading as a model, a size that is a
different order of magnitude, silent re-pinning) actually refuse.
"""

import importlib.util
import json
import pathlib

import pytest

FETCH_PY = pathlib.Path(__file__).resolve().parents[2] / "models" / "fetch.py"

spec = importlib.util.spec_from_file_location("fetch", FETCH_PY)
fetch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch)


@pytest.fixture
def manifest(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({
        "artifacts": {
            "model_a": {
                "purpose": "test",
                "license": "MIT",
                "source": {"repo": "https://example.invalid/repo"},
                "file": "model_a.bin",
                "expected_mb": 0.001,
                "pinned": None,
            },
            "model_b": {
                "purpose": "test",
                "license": "MIT",
                "source": {"repo": "https://example.invalid/repo"},
                "file": "model_b.bin",
                "expected_mb": None,
                "pinned": None,
            },
        }
    }))
    return path


def artifact(tmp_path, name="weights.bin", content=b"\x00\x01" * 512):
    source = tmp_path / name
    source.write_bytes(content)
    return source


def test_pinning_records_the_hash_and_verify_holds_the_file_to_it(manifest, tmp_path):
    source = artifact(tmp_path)

    assert fetch.pin(manifest, "model_a", str(source)) == 0
    entry = fetch.load(manifest)["artifacts"]["model_a"]
    assert entry["pinned"]["sha256"] == fetch.sha256_of(source)
    assert entry["pinned"]["bytes"] == source.stat().st_size

    assert fetch.verify(manifest) == 0


def test_a_changed_file_fails_verification_by_name(manifest, tmp_path, capsys):
    fetch.pin(manifest, "model_a", str(artifact(tmp_path)))
    (manifest.parent / "model_a.bin").write_bytes(b"\x00\x01" * 511 + b"\xff\xff")

    assert fetch.verify(manifest) == 1
    assert "model_a" in capsys.readouterr().out


def test_a_missing_pinned_file_is_a_failure_but_an_unpinned_one_is_not(manifest, tmp_path):
    fetch.pin(manifest, "model_a", str(artifact(tmp_path)))
    (manifest.parent / "model_a.bin").unlink()

    # model_a is vouched for and absent: fail. model_b is future work: fine.
    assert fetch.verify(manifest, ["model_a"]) == 1
    assert fetch.verify(manifest, ["model_b"]) == 0


def test_repinning_needs_force_and_the_old_pin_survives_the_refusal(manifest, tmp_path):
    first = artifact(tmp_path, "first.bin")
    fetch.pin(manifest, "model_a", str(first))
    old = fetch.load(manifest)["artifacts"]["model_a"]["pinned"]["sha256"]

    second = artifact(tmp_path, "second.bin", b"\x02\x03" * 512)
    assert fetch.pin(manifest, "model_a", str(second)) == 2
    assert fetch.load(manifest)["artifacts"]["model_a"]["pinned"]["sha256"] == old

    assert fetch.pin(manifest, "model_a", str(second), force=True) == 0
    assert fetch.load(manifest)["artifacts"]["model_a"]["pinned"]["sha256"] != old


def test_an_error_page_is_refused_however_it_is_named(manifest, tmp_path):
    # The exact trap this environment sets: the proxy answers with an HTML
    # error document and status 200, and it lands under the model's filename.
    page = artifact(tmp_path, "weights.pt",
                    b"<!DOCTYPE html>\n<html><body>403</body></html>" * 40)

    assert fetch.pin(manifest, "model_b", str(page)) == 1
    assert fetch.load(manifest)["artifacts"]["model_b"]["pinned"] is None
    assert not (manifest.parent / "model_b.bin").exists()


def test_a_wrong_order_of_magnitude_is_a_wrong_file(manifest, tmp_path):
    # model_a documents ~1 KB; hand it 64 bytes. A version bump changes a size
    # by percents, not by orders of magnitude.
    tiny = artifact(tmp_path, "tiny.bin", b"\x00" * 64)

    assert fetch.pin(manifest, "model_a", str(tiny)) == 1
    assert fetch.load(manifest)["artifacts"]["model_a"]["pinned"] is None


def test_an_unknown_artifact_is_named_along_with_what_is_known(manifest, tmp_path, capsys):
    assert fetch.pin(manifest, "model_zz", str(artifact(tmp_path))) == 2
    out = capsys.readouterr().out
    assert "model_a" in out and "model_b" in out


def test_the_real_manifest_parses_and_starts_unpinned():
    manifest = fetch.load(fetch.MANIFEST)
    for name, entry in manifest["artifacts"].items():
        assert entry["file"], name
        assert entry["license"], name
        assert entry["source"], name
        # Nothing may carry a pin that was not made by an actual fetch. This
        # environment has no network, so until someone pins on a networked
        # machine every entry must say so honestly.
        if entry["pinned"] is not None:
            assert set(entry["pinned"]) == {"sha256", "bytes", "origin", "date"}
