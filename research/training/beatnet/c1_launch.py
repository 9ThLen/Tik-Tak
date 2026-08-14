#!/usr/bin/env python3
"""Run the six new C1 training jobs in sequence, and stop when told to.

Sequential because they share one GPU. The only thing this adds over six
invocations by hand is that it gets the two failure modes right.

**Exit code 75 stops the sweep.** `run.py` returns 75 when the pause file
appears, and a loop that treats that as "this one finished, start the next"
would ignore the pause and keep the machine busy -- the M0b operational
revision warns about exactly this.

**Resume is per run, and inferred rather than asked for.** After a pause one run
has a checkpoint and no `result.json` while the rest have not started, so a
finished run is skipped, an interrupted one gets `--resume`, and an untouched one
starts fresh. Passing `--resume` to a run that never began fails closed inside
`run.py`, which is why this decides per run instead of applying one flag to all.

The 100% arm is not here. It is the S1 runs reused as an anchor.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

from .cache import _atomic_json, _outside_repository

SCHEMA = "tiktak.c1_launch/v1"
FRACTIONS = (0.25, 0.50)
SEEDS = (17, 29, 43)
ARM = "A3_stateful"
PAUSED_EXIT_CODE = 75


def plan(output_root: pathlib.Path) -> list[dict]:
    """Every job, in the order they run, with what each one needs."""
    jobs = []
    for fraction in FRACTIONS:
        for seed in SEEDS:
            root = output_root / f"f{fraction:.2f}-seed-{seed}"
            jobs.append({"fraction": fraction, "seed": seed, "root": root})
    return jobs


def job_state(root: pathlib.Path) -> str:
    if (root / "result.json").is_file():
        return "complete"
    if (root / "checkpoint.pt").is_file():
        return "interrupted"
    return "pending"


def command(job: dict, args: argparse.Namespace, state: str) -> list[str]:
    argv = [sys.executable, "-m", "training.beatnet.run",
            "--arm", ARM, "--seed", str(job["seed"]),
            "--config", str(args.config), "--source", str(args.source),
            "--cache", str(args.cache), "--output-root", str(job["root"]),
            "--baseline", str(args.baseline), "--binary", str(args.binary),
            "--manifest", str(args.manifest), "--m0e", str(args.m0e),
            "--music-root", str(args.music_root),
            "--subset", str(args.subset),
            "--fraction", f"{job['fraction']:.2f}",
            "--device", args.device,
            "--eval-workers", str(args.eval_workers)]
    if args.pause_file is not None:
        argv += ["--pause-file", str(args.pause_file)]
    if state == "interrupted":
        argv.append("--resume")
    return argv


def main(argv: list[str] | None = None) -> int:
    repository = pathlib.Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", type=pathlib.Path, required=True)
    parser.add_argument("--config", type=pathlib.Path, required=True)
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--cache", type=pathlib.Path, required=True)
    parser.add_argument("--baseline", type=pathlib.Path, required=True)
    parser.add_argument("--binary", type=pathlib.Path, required=True)
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--m0e", type=pathlib.Path, required=True)
    parser.add_argument("--music-root", type=pathlib.Path, required=True)
    parser.add_argument("--output-root", type=pathlib.Path, required=True)
    parser.add_argument("--pause-file", type=pathlib.Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    try:
        _outside_repository(args.output_root, repository)
        if args.pause_file is not None:
            _outside_repository(args.pause_file, repository)
            if args.pause_file.exists():
                raise ValueError(
                    f"remove the pause file before starting: {args.pause_file}")
        subset = json.loads(args.subset.read_text(encoding="utf-8"))
        # Refuse before any GPU time rather than on the first job.
        from . import c1_subsets
        c1_subsets.require_registered_corpus(subset)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    jobs = plan(args.output_root)
    done, paused, failed = [], None, None
    for job in jobs:
        state = job_state(job["root"])
        argv_job = command(job, args, state)
        label = {"fraction": job["fraction"], "seed": job["seed"],
                 "state": state}
        if state == "complete":
            print(json.dumps({"event": "skip", **label}), flush=True)
            done.append(label)
            continue
        print(json.dumps({"event": "start", **label}), flush=True)
        if args.dry_run:
            done.append({**label, "command": argv_job})
            continue
        code = subprocess.run(argv_job, cwd=str(repository / "research")).returncode
        if code == PAUSED_EXIT_CODE:
            paused = label
            print(json.dumps({"event": "paused", **label}), flush=True)
            break
        if code != 0:
            failed = {**label, "returncode": code}
            print(json.dumps({"event": "failed", **failed}), flush=True)
            break
        done.append(label)

    status = ("paused" if paused else "failed" if failed else
              "complete" if len(done) == len(jobs) else "partial")
    state_path = args.output_root / "launch.json"
    if not args.dry_run:
        _atomic_json(state_path, {
            "schema": SCHEMA, "status": status, "jobs": len(jobs),
            "completed": done, "paused": paused, "failed": failed})
    print(json.dumps({"event": "done", "status": status,
                      "completed": len(done), "of": len(jobs)}))
    if failed:
        return 1
    return PAUSED_EXIT_CODE if paused else 0


if __name__ == "__main__":
    raise SystemExit(main())
