"""The launcher's two failure modes, which are the only reason it exists."""
import json
import pathlib

import pytest

from training.beatnet import c1_launch as launch


def _args(tmp_path, **overrides):
    import argparse
    base = dict(subset=tmp_path / "subset.json", config=tmp_path / "config.json",
                source=tmp_path / "source.pt", cache=tmp_path / "cache.json",
                baseline=tmp_path / "a0.json", binary=tmp_path / "bin.exe",
                manifest=tmp_path / "manifest.json", m0e=tmp_path / "m0e.json",
                music_root=tmp_path / "music", output_root=tmp_path / "out",
                pause_file=None, device="cuda", eval_workers=4, dry_run=False)
    base.update(overrides)
    return argparse.Namespace(**base)


def test_the_plan_is_six_jobs_and_never_the_anchor():
    jobs = launch.plan(pathlib.Path("/out"))
    assert len(jobs) == 6
    assert {job["fraction"] for job in jobs} == {0.25, 0.50}
    assert {job["seed"] for job in jobs} == set(launch.SEEDS)
    assert 1.00 not in {job["fraction"] for job in jobs}
    assert len({job["root"] for job in jobs}) == 6


def test_resume_is_inferred_per_job_not_applied_to_all(tmp_path):
    """After a pause one run has a checkpoint and the rest have not started."""
    complete = tmp_path / "complete"
    complete.mkdir()
    (complete / "result.json").write_text("{}", encoding="utf-8")
    interrupted = tmp_path / "interrupted"
    interrupted.mkdir()
    (interrupted / "checkpoint.pt").write_bytes(b"x")
    pending = tmp_path / "pending"

    (complete / "result.json").write_text(json.dumps({
        "arm": launch.ARM, "seed": 17,
        "identity": {"c1": {"fraction": 0.25}}}), encoding="utf-8")
    assert launch.job_state(complete, 0.25, 17) == "complete"
    assert launch.job_state(interrupted, 0.25, 29) == "interrupted"
    assert launch.job_state(pending, 0.25, 43) == "pending"

    args = _args(tmp_path)
    job = {"fraction": 0.25, "seed": 17, "root": interrupted}
    assert "--resume" in launch.command(job, args, "interrupted")
    assert "--resume" not in launch.command(job, args, "pending")


def test_the_command_carries_the_fraction_and_the_stateful_arm(tmp_path):
    args = _args(tmp_path)
    got = launch.command({"fraction": 0.50, "seed": 29,
                          "root": tmp_path / "r"}, args, "pending")
    assert got[got.index("--fraction") + 1] == "0.50"
    assert got[got.index("--arm") + 1] == "A3_stateful"
    assert got[got.index("--subset") + 1] == str(args.subset)


def _inputs(tmp_path):
    subset = {"total_frames": 10566912, "registered_corpus": True,
              "frame_fraction_deviations": {}}
    for name in ("subset.json", "config.json", "cache.json", "a0.json",
                 "manifest.json", "m0e.json"):
        (tmp_path / name).write_text(json.dumps(subset), encoding="utf-8")
    (tmp_path / "source.pt").write_bytes(b"x")
    (tmp_path / "bin.exe").write_bytes(b"x")
    (tmp_path / "music").mkdir(exist_ok=True)
    return ["--subset", str(tmp_path / "subset.json"),
            "--config", str(tmp_path / "config.json"),
            "--source", str(tmp_path / "source.pt"),
            "--cache", str(tmp_path / "cache.json"),
            "--baseline", str(tmp_path / "a0.json"),
            "--binary", str(tmp_path / "bin.exe"),
            "--manifest", str(tmp_path / "manifest.json"),
            "--m0e", str(tmp_path / "m0e.json"),
            "--music-root", str(tmp_path / "music"),
            "--output-root", str(tmp_path / "out")]


class _Result:
    def __init__(self, code):
        self.returncode = code


def test_a_pause_stops_the_sweep_instead_of_starting_the_next_job(
        tmp_path, monkeypatch):
    """The failure the M0b operational revision warns about.

    A loop that reads 75 as "this one finished" would ignore the pause and keep
    the machine busy for another four runs. `main` has to stop, report where,
    and return 75 itself so whatever launched it can stop too.
    """
    started = []

    def fake_run(argv, **kwargs):
        started.append(argv[argv.index("--seed") + 1])
        return _Result(launch.PAUSED_EXIT_CODE if len(started) == 2 else 0)

    monkeypatch.setattr(launch.subprocess, "run", fake_run)
    code = launch.main(_inputs(tmp_path))

    assert code == launch.PAUSED_EXIT_CODE
    assert len(started) == 2, "the sweep continued past the pause"
    state = json.loads((tmp_path / "out" / "launch.json").read_text())
    assert state["status"] == "paused"
    assert state["paused"]["seed"] == 29
    assert len(state["completed"]) == 1


def test_a_failed_job_stops_the_sweep_and_returns_non_zero(
        tmp_path, monkeypatch):
    started = []

    def fake_run(argv, **kwargs):
        started.append(argv)
        return _Result(0 if len(started) == 1 else 2)

    monkeypatch.setattr(launch.subprocess, "run", fake_run)
    assert launch.main(_inputs(tmp_path)) == 1
    assert len(started) == 2
    state = json.loads((tmp_path / "out" / "launch.json").read_text())
    assert state["status"] == "failed"
    assert state["failed"]["returncode"] == 2


def test_relaunching_after_a_pause_skips_what_finished_and_resumes_one(
        tmp_path, monkeypatch):
    """The whole point of inferring resume per job rather than per sweep."""
    out = tmp_path / "out"
    (out / "f0.25-seed-17").mkdir(parents=True)
    (out / "f0.25-seed-17" / "result.json").write_text(json.dumps({
        "arm": launch.ARM, "seed": 17,
        "identity": {"c1": {"fraction": 0.25}}}), encoding="utf-8")
    (out / "f0.25-seed-29").mkdir(parents=True)
    (out / "f0.25-seed-29" / "checkpoint.pt").write_bytes(b"x")

    seen = []

    def fake_run(argv, **kwargs):
        seen.append((argv[argv.index("--seed") + 1], "--resume" in argv))
        return _Result(0)

    monkeypatch.setattr(launch.subprocess, "run", fake_run)
    assert launch.main(_inputs(tmp_path)) == 0
    assert seen[0] == ("29", True), "the interrupted job must resume"
    assert all(not resumed for _, resumed in seen[1:]), "later jobs are fresh"
    assert len(seen) == 5, "the finished job must not run again"


def test_a_subset_that_is_not_the_registered_corpus_is_refused(tmp_path):
    from training.beatnet import c1_subsets

    with pytest.raises(ValueError, match="is not the registered"):
        c1_subsets.require_registered_corpus(
            {"total_frames": 1, "registered_corpus": False})


def test_end_to_end_dry_run_plans_six_commands(tmp_path, capsys):
    code = launch.main(_inputs(tmp_path) + ["--dry-run"])
    assert code == 0
    events = [json.loads(line) for line in
              capsys.readouterr().out.strip().splitlines()]
    assert sum(1 for e in events if e["event"] == "start") == 6
    assert events[-1] == {"event": "done", "status": "complete",
                          "completed": 6, "of": 6}


def test_a_directory_holding_another_jobs_result_is_not_called_complete(tmp_path):
    """Otherwise the sweep skips it and trains five runs while reporting six."""
    root = tmp_path / "f0.25-seed-17"
    root.mkdir()
    (root / "result.json").write_text(json.dumps({
        "arm": launch.ARM, "seed": 29,
        "identity": {"c1": {"fraction": 0.50}}}), encoding="utf-8")
    with pytest.raises(ValueError, match="another job's result"):
        launch.job_state(root, 0.25, 17)
