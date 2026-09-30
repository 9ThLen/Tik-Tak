import json

from eval.compare_live_runs import load, macro, paired, corpora_of


def _write(tmp_path, name, records):
    path = tmp_path / name
    path.write_text(json.dumps(records), encoding="utf-8")
    return str(path)


def _record(corpus, name, usable, worst=0.0, f=0.5):
    return {"ok": True, "annotated": True, "corpus": corpus, "name": name,
            "usable": usable, "usable_strict": usable, "worst_wrong_octave_sec": worst,
            "f_measure": f, "correct_share_of_eligible": f}


def test_macro_ignores_corpora_too_small_to_speak(tmp_path):
    records = [_record("big", f"b{i}", i % 2 == 0) for i in range(40)]
    records += [_record("small", f"s{i}", True) for i in range(5)]
    run = load(_write(tmp_path, "a.json", records))
    by = corpora_of(run.keys())
    assert set(by) == {"big"}
    assert macro(run, by, "usable") == 0.5


def test_episode_free_is_the_four_second_rule(tmp_path):
    records = [_record("big", f"b{i}", True, worst=5.0 if i < 10 else 2.0) for i in range(40)]
    run = load(_write(tmp_path, "a.json", records))
    assert macro(run, corpora_of(run.keys()), "episode_free") == 0.75


def test_paired_counts_what_moved(tmp_path):
    base = [_record("big", f"b{i}", False) for i in range(40)]
    arm = [_record("big", f"b{i}", i < 4) for i in range(40)]
    delta, _, better, worse, before, after = paired(
        load(_write(tmp_path, "base.json", base)), load(_write(tmp_path, "arm.json", arm)),
        "usable")
    assert (better, worse) == (4, 0)
    assert abs(delta - 0.1) < 1e-12 and before == 0.0 and abs(after - 0.1) < 1e-12
