"""The artifact pinning workflow, exercised entirely offline.

The download half cannot be tested here — this environment has no network,
which is the very reason the workflow exists. What can be tested is everything
the checksums are *for*: pinning records the right hash, verification catches a
changed file, and the refusals (HTML masquerading as a model, a size that is a
different order of magnitude, silent trust-root replacement) actually refuse.
"""

import importlib.util
import json
import pathlib
import subprocess
import sys
import time

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
                "source": {
                    "repo": "https://example.invalid/repo",
                    "revision": "0123456789abcdef",
                    "path": "weights/model_a.bin",
                },
                "file": "model_a.bin",
                "expected_mb": 0.001,
                "pinned": None,
            },
            "model_b": {
                "purpose": "test",
                "license": "MIT",
                "source": {
                    "repo": "https://example.invalid/repo",
                    "revision": "fedcba9876543210",
                    "path": "exports/model_b.onnx",
                },
                "conversion": {
                    "model_variant": "test",
                    "source_weights_sha256": "a" * 64,
                    "exporter": {
                        "repo": "https://example.invalid/exporter",
                        "revision": "1111222233334444",
                        "version": "1.2.3",
                    },
                    "onnx_opset": 17,
                    "preprocessing": {"sample_rate": 22050},
                    "io_contract": {
                        "inputs": ["audio: float32[batch,samples]"],
                        "outputs": ["salience: float32[batch,frames]"],
                    },
                },
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
    assert entry["pinned"]["provenance"] == {"source": entry["source"]}
    assert "origin" not in entry["pinned"]
    assert str(source) not in manifest.read_text(encoding="utf-8")

    assert fetch.verify(manifest) == 0


def test_derived_artifact_provenance_carries_the_conversion_contract(
        manifest, tmp_path):
    source = artifact(tmp_path, "export.onnx")

    assert fetch.pin(manifest, "model_b", str(source)) == 0
    entry = fetch.load(manifest)["artifacts"]["model_b"]
    assert entry["pinned"]["provenance"] == {
        "source": entry["source"],
        "conversion": entry["conversion"],
    }
    assert str(source) not in manifest.read_text(encoding="utf-8")


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
    target = manifest.parent / "model_a.bin"
    old_bytes = target.read_bytes()

    second = artifact(tmp_path, "second.bin", b"\x02\x03" * 512)
    assert fetch.pin(manifest, "model_a", str(second)) == 2
    assert fetch.load(manifest)["artifacts"]["model_a"]["pinned"]["sha256"] == old
    assert target.read_bytes() == old_bytes

    assert fetch.pin(manifest, "model_a", str(second), force=True) == 0
    assert fetch.load(manifest)["artifacts"]["model_a"]["pinned"]["sha256"] != old


def test_matching_bytes_restore_a_missing_pinned_artifact_without_repinning(
        manifest, tmp_path):
    source = artifact(tmp_path, "trusted.bin")
    assert fetch.pin(manifest, "model_a", str(source)) == 0
    manifest_before = manifest.read_bytes()
    target = manifest.parent / "model_a.bin"
    target.unlink()

    assert fetch.pin(manifest, "model_a", str(source)) == 0

    assert target.read_bytes() == source.read_bytes()
    assert manifest.read_bytes() == manifest_before


def test_a_failed_forced_repin_preserves_the_old_file_and_pin(manifest, tmp_path):
    first = artifact(tmp_path, "first.bin")
    assert fetch.pin(manifest, "model_a", str(first)) == 0
    target = manifest.parent / "model_a.bin"
    old_bytes = target.read_bytes()
    old_pin = fetch.load(manifest)["artifacts"]["model_a"]["pinned"]

    error_page = artifact(
        tmp_path, "replacement.pt",
        b"<!DOCTYPE html>\n<html><body>403</body></html>" * 40)
    assert fetch.pin(manifest, "model_a", str(error_page), force=True) == 1

    assert target.read_bytes() == old_bytes
    assert fetch.load(manifest)["artifacts"]["model_a"]["pinned"] == old_pin


def test_a_manifest_save_failure_rolls_back_a_valid_repin(
        manifest, tmp_path, monkeypatch):
    first = artifact(tmp_path, "first.bin")
    assert fetch.pin(manifest, "model_a", str(first)) == 0
    target = manifest.parent / "model_a.bin"
    old_bytes = target.read_bytes()
    old_manifest = manifest.read_bytes()

    replacement = artifact(tmp_path, "replacement.bin", b"\x02\x03" * 512)
    real_replace = pathlib.Path.replace

    def fail_staged_manifest_replace(path, target):
        target = pathlib.Path(target)
        if (target == manifest
                and path.name.startswith(f".{manifest.name}.")
                and path.suffix == ".tmp"):
            raise OSError("simulated manifest replace failure")
        return real_replace(path, target)

    monkeypatch.setattr(pathlib.Path, "replace", fail_staged_manifest_replace)
    with pytest.raises(OSError, match="simulated manifest replace failure"):
        fetch.pin(manifest, "model_a", str(replacement), force=True)

    assert target.read_bytes() == old_bytes
    assert manifest.read_bytes() == old_manifest
    assert not list(manifest.parent.glob(f".{manifest.name}.*.tmp"))
    assert not list(manifest.parent.glob(f".{target.name}.*.tmp"))
    assert not list(manifest.parent.glob(f".{target.name}.*.backup"))


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


def test_an_unpinned_file_is_rejected_by_verify(manifest, tmp_path, capsys):
    target = manifest.parent / "model_b.bin"
    target.write_bytes(b"untrusted")

    assert fetch.verify(manifest, ["model_b"]) == 1
    assert "exists but is not pinned" in capsys.readouterr().out


def test_an_unknown_artifact_is_named_along_with_what_is_known(manifest, tmp_path, capsys):
    assert fetch.pin(manifest, "model_zz", str(artifact(tmp_path))) == 2
    out = capsys.readouterr().out
    assert "model_a" in out and "model_b" in out


def test_verify_remains_compatible_with_a_legacy_origin_pin(manifest):
    target = manifest.parent / "model_a.bin"
    target.write_bytes(b"\x00\x01" * 512)
    contents = fetch.load(manifest)
    contents["artifacts"]["model_a"]["pinned"] = {
        "sha256": fetch.sha256_of(target),
        "bytes": target.stat().st_size,
        "origin": r"C:\legacy\download\weights.bin",
        "date": "2026-01-01",
    }
    fetch.save(manifest, contents)

    assert fetch.verify(manifest, ["model_a"]) == 0


def test_concurrent_pins_do_not_lose_the_first_manifest_update(
        manifest, tmp_path):
    """A waiting process must load only after the current transaction saves."""
    child_script = tmp_path / "pin_child.py"
    child_script.write_text(
        """
import importlib.util
import pathlib
import sys

spec = importlib.util.spec_from_file_location("fetch_child", sys.argv[1])
fetch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch)
pathlib.Path(sys.argv[2]).write_text("ready", encoding="utf-8")
raise SystemExit(fetch.pin(
    pathlib.Path(sys.argv[3]), sys.argv[4], sys.argv[5]))
""".lstrip(),
        encoding="utf-8",
    )
    ready = tmp_path / "child.ready"
    source_a = artifact(tmp_path, "concurrent-a.bin")
    source_b = artifact(
        tmp_path, "concurrent-b.bin", b"\x02\x03" * (4 * 1024 * 1024))

    with fetch.artifact_lock(manifest):
        child = subprocess.Popen(
            [
                sys.executable,
                str(child_script),
                str(FETCH_PY),
                str(ready),
                str(manifest),
                "model_b",
                str(source_b),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        deadline = time.monotonic() + 10
        while not ready.exists() and child.poll() is None:
            if time.monotonic() >= deadline:
                child.kill()
                pytest.fail("concurrent pin child did not start")
            time.sleep(0.01)
        assert ready.exists()
        assert child.poll() is None, "child bypassed the held artifact lock"
        assert fetch._pin_locked(
            manifest, "model_a", str(source_a)) == 0

    out, err = child.communicate(timeout=20)
    assert child.returncode == 0, out + err
    contents = fetch.load(manifest)["artifacts"]
    assert contents["model_a"]["pinned"] is not None
    assert contents["model_b"]["pinned"] is not None
    assert fetch.verify(manifest) == 0


def test_the_real_manifest_records_reproducible_provenance():
    manifest = fetch.load(fetch.MANIFEST)
    for name, entry in manifest["artifacts"].items():
        assert entry["file"], name
        assert entry["license"], name
        assert entry["source"], name
        if "conversion" in entry:
            assert {
                "model_variant",
                "source_weights_sha256",
                "exporter",
                "onnx_opset",
                "preprocessing",
                "io_contract",
            } <= set(entry["conversion"]), name
        # Nothing may carry a pin that was not made by an actual fetch. Entries
        # not obtained yet remain null; obtained entries carry bytes, hash and
        # canonical provenance rather than a local delivery path.
        if entry["pinned"] is not None:
            assert {"sha256", "bytes", "date"} <= set(entry["pinned"])
            assert "origin" not in entry["pinned"]
            assert entry["pinned"]["provenance"]["source"]


# ----------------------------------------------------------- attribution ----
#
# Using these models is conditional on crediting their authors, and the way
# that condition gets broken is never a decision — it is a model added later
# whose entry nobody copied into a hand-written notice. So the notice is
# generated, and these hold it to the manifest.

def _notice():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "tiktak_notice", fetch.MANIFEST.parent / "notice.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_artifact_credits_its_authors():
    manifest = fetch.load(fetch.MANIFEST)
    for name, entry in manifest["artifacts"].items():
        attribution = entry.get("attribution")
        assert attribution, name
        for field in ("work", "creators", "license_id", "license_uri",
                      "source_uri", "citation"):
            assert attribution.get(field), f"{name} is missing {field}"
        assert isinstance(attribution["creators"], list)
        assert all(attribution["creators"]), name
        # A licence has to be named as an identifier, not described. "MIT-ish"
        # or "open source" is what a hurried entry looks like.
        assert attribution["license_id"] in {"MIT", "CC-BY-4.0"}, name
        assert attribution["license_uri"].startswith("https://"), name
        # Present but null means "unmodified"; absent means nobody considered
        # the question, and CC BY 4.0 requires that modifications be indicated.
        assert "modifications" in attribution, name


def test_the_licence_texts_are_in_the_repository():
    # MIT requires its text to travel with the work. A link is not a copy, and
    # the link is to someone else's branch, which can move.
    manifest = fetch.load(fetch.MANIFEST)
    for name, entry in manifest["artifacts"].items():
        relative = entry["attribution"].get("license_text")
        assert relative, name
        path = fetch.MANIFEST.parent / relative
        assert path.is_file(), f"{name}: {path} is missing"
        assert len(path.read_text(encoding="utf-8").strip()) > 200, name


def test_the_notice_is_regenerated_from_the_manifest():
    # The check that makes the rest of this enforceable rather than advisory.
    notice = _notice()
    manifest = fetch.load(fetch.MANIFEST)
    rendered = notice.render(manifest)
    assert notice.NOTICE.is_file(), "NOTICE.md is missing"
    assert notice.NOTICE.read_text(encoding="utf-8") == rendered, (
        "NOTICE.md is out of date — run: python models/notice.py")


def test_the_notice_names_every_creator_and_licence():
    notice = _notice()
    manifest = fetch.load(fetch.MANIFEST)
    rendered = notice.render(manifest)
    for entry in manifest["artifacts"].values():
        attribution = entry["attribution"]
        for creator in attribution["creators"]:
            assert creator in rendered, creator
        assert attribution["license_uri"] in rendered
        assert attribution["source_uri"] in rendered
        # And what we did to it, which is the CC BY 4.0 obligation that a
        # generic notice would silently drop.
        if attribution["modifications"]:
            assert attribution["modifications"] in rendered, entry["file"]


def test_an_artifact_without_attribution_is_refused():
    # The failure mode this exists for, exercised rather than assumed: the
    # generator must stop, not skip the entry and produce a notice that looks
    # complete.
    notice = _notice()
    manifest = fetch.load(fetch.MANIFEST)
    manifest["artifacts"]["nameless_model"] = {"file": "x.onnx", "pinned": None}
    with pytest.raises(SystemExit):
        notice.render(manifest)
