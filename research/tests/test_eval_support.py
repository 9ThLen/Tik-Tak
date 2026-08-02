"""Regression tests for shared evaluation statistics and provenance."""

import hashlib

import numpy as np
import pytest

from eval.provenance import provenance
from eval.statistics import spearman


def test_spearman_assigns_average_ranks_to_tied_values():
    x = np.arange(20, dtype=np.float64)
    y = np.repeat((0.0, 1.0), 10)
    expected_y_ranks = np.repeat((4.5, 14.5), 10)
    expected = float(np.corrcoef(x, expected_y_ranks)[0, 1])

    assert spearman(x, y, min_samples=20) == pytest.approx(expected)


def test_spearman_filters_non_finite_pairs_and_rejects_no_variation():
    x = np.arange(21, dtype=np.float64)
    y = x.copy()
    x[-1] = np.nan
    y[-2] = np.inf

    assert spearman(x, y, min_samples=19) == pytest.approx(1.0)
    assert spearman(np.ones(20), np.arange(20)) is None
    assert spearman(np.arange(10), np.arange(10), min_samples=20) is None


def test_provenance_content_addresses_upstream_artifacts(tmp_path):
    source = tmp_path / "oracle.json"
    payload = b'{"result": 0.927}'
    source.write_bytes(payload)

    report = provenance(tmp_path, {"oracle_0": source})

    assert report["oracle_0"] == {
        "name": "oracle.json",
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
