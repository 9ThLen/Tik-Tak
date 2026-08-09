"""Regression tests for the evaluation artifact provenance contract."""

from __future__ import annotations

import importlib
import pathlib
import subprocess

import pytest

from eval.provenance import (PROVENANCE_SCHEMA, experiment_provenance,
                             provenance)


def _git(repository: pathlib.Path, *arguments: str) -> None:
    subprocess.run(
        ("git", "-C", str(repository), *arguments),
        check=True,
        capture_output=True,
        text=True,
    )


def _repository(tmp_path: pathlib.Path) -> pathlib.Path:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "--quiet")
    _git(repository, "config", "user.name", "TikTak test")
    _git(repository, "config", "user.email", "test@example.invalid")
    (repository / "tracked.txt").write_text("tracked\n", encoding="utf-8")
    _git(repository, "add", "tracked.txt")
    _git(repository, "commit", "--quiet", "-m", "fixture")
    return repository


def test_clean_repository_uses_the_versioned_schema(tmp_path):
    record = provenance(_repository(tmp_path))

    assert record["schema"] == PROVENANCE_SCHEMA
    assert record["commit"] and len(record["commit"]) == 40
    assert record["tree_clean"] is True
    assert "provenance_error" not in record


def test_dirty_repository_is_not_clean(tmp_path):
    repository = _repository(tmp_path)
    (repository / "untracked.txt").write_text("dirty\n", encoding="utf-8")

    record = provenance(repository)

    assert record["tree_clean"] is False
    assert "provenance_error" not in record


def test_experimental_artifact_refuses_a_dirty_repository(tmp_path):
    repository = _repository(tmp_path)
    (repository / "untracked.txt").write_text("dirty\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="tree_clean is True"):
        experiment_provenance(repository)


def test_experimental_artifact_refuses_an_unreadable_input(tmp_path):
    repository = _repository(tmp_path)

    with pytest.raises(RuntimeError, match="unreadable provenance inputs: model"):
        experiment_provenance(
            repository, {"optional": None, "model": tmp_path / "missing.ttw"})


def test_not_a_repository_is_unknown_not_dirty(tmp_path):
    # The test basetemp may live below the repository under review. An empty
    # `.git` boundary prevents Git from walking up and discovering that parent.
    (tmp_path / ".git").write_text("gitdir: missing\n", encoding="utf-8")
    record = provenance(tmp_path)

    assert record["commit"] is None
    assert record["tree_clean"] is None
    assert record["provenance_error"]

    with pytest.raises(RuntimeError, match="got None"):
        experiment_provenance(tmp_path)


def test_git_spawn_failure_is_fail_closed(monkeypatch, tmp_path):
    provenance_module = importlib.import_module("eval.provenance")

    def fail(*_args, **_kwargs):
        raise OSError("git unavailable")

    monkeypatch.setattr(provenance_module.subprocess, "run", fail)

    record = provenance(tmp_path)

    assert record["commit"] is None
    assert record["tree_clean"] is None
    assert len(record["provenance_error"]) == 2


def _mock_git(monkeypatch, status_stderr: str) -> None:
    provenance_module = importlib.import_module("eval.provenance")

    def run(command, **_kwargs):
        arguments = tuple(command[-2:])
        if arguments == ("rev-parse", "HEAD"):
            return subprocess.CompletedProcess(command, 0, "a" * 40 + "\n", "")
        if arguments == ("status", "--porcelain"):
            return subprocess.CompletedProcess(command, 0, "", status_stderr)
        raise AssertionError(f"unexpected git command: {command}")

    monkeypatch.setattr(provenance_module.subprocess, "run", run)


def test_captured_global_ignore_warning_is_benign(monkeypatch, tmp_path):
    _mock_git(
        monkeypatch,
        "warning: unable to access "
        "'C:\\Users\\fixture/.config/git/ignore': Permission denied\n",
    )

    record = provenance(tmp_path)

    assert record["tree_clean"] is True
    assert "provenance_error" not in record


@pytest.mark.parametrize(
    "stderr",
    [
        "warning: could not open directory 'private/': Permission denied\n",
        "warning: unable to read tree object deadbeef\n",
        "warning: a warning not present in the captured allowlist\n",
    ],
)
def test_incomplete_or_unknown_status_warning_is_fail_closed(
        monkeypatch, tmp_path, stderr):
    _mock_git(monkeypatch, stderr)

    record = provenance(tmp_path)

    assert record["commit"] == "a" * 40
    assert record["tree_clean"] is None
    assert "unexpected stderr" in " ".join(record["provenance_error"])


def test_eval_harnesses_do_not_bypass_the_shared_status_check():
    eval_directory = pathlib.Path(__file__).resolve().parents[1] / "eval"
    bypasses = []
    for path in eval_directory.glob("*.py"):
        if path.name == "provenance.py":
            continue
        text = path.read_text(encoding="utf-8")
        if ('"status", "--porcelain"' in text
                or "'status', '--porcelain'" in text):
            bypasses.append(path.name)

    assert bypasses == []


def test_eval_harnesses_use_the_fail_closed_experiment_wrapper():
    eval_directory = pathlib.Path(__file__).resolve().parents[1] / "eval"
    unsafe_imports = []
    for path in eval_directory.glob("*.py"):
        if path.name == "provenance.py":
            continue
        text = path.read_text(encoding="utf-8")
        if "from eval.provenance import provenance" in text:
            unsafe_imports.append(path.name)

    assert unsafe_imports == []
