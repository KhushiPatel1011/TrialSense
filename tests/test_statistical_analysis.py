
import csv

import pytest

from src.evaluation.statistical_analysis import (
    load_paired_results,
    paired_bootstrap,
    percentile,
    METRICS,
)


def test_percentile():
    assert percentile([0, 10, 20], 0.5) == 10
    assert percentile([0, 10, 20], 0.25) == 5


def test_positive_constant_difference():
    result = paired_bootstrap(
        [0.1, 0.2, 0.3],
        [0.2, 0.3, 0.4],
        iterations=100,
    )

    assert result["observed_difference"] == pytest.approx(
        0.1
    )

    assert result["ci_lower"] == pytest.approx(0.1)
    assert result["ci_upper"] == pytest.approx(0.1)


def test_identical_methods():
    result = paired_bootstrap(
        [0.1, 0.2, 0.3],
        [0.1, 0.2, 0.3],
        iterations=100,
    )

    assert result["observed_difference"] == 0
    assert result["ci_lower"] == 0
    assert result["ci_upper"] == 0


def test_bootstrap_reproducible():
    baseline = [0.0, 0.1, 0.2, 0.4]
    proposed = [0.1, 0.2, 0.2, 0.5]

    first = paired_bootstrap(
        baseline, proposed, 200, seed=42
    )

    second = paired_bootstrap(
        baseline, proposed, 200, seed=42
    )

    assert first == second


def test_mismatched_pairs():
    with pytest.raises(ValueError):
        paired_bootstrap([0.1], [0.2, 0.3])


def test_empty_pairs():
    with pytest.raises(ValueError):
        paired_bootstrap([], [])


def test_invalid_iterations():
    with pytest.raises(ValueError):
        paired_bootstrap([0.1], [0.2], iterations=0)


def test_load_paired_results(tmp_path):
    path = tmp_path / "results.csv"

    fields = ["topic_id"]

    for suffix in METRICS.values():
        fields.extend([
            "v1_" + suffix,
            "v4_" + suffix,
        ])

    row = {"topic_id": "1"}

    for suffix in METRICS.values():
        row["v1_" + suffix] = 0.2
        row["v4_" + suffix] = 0.3

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerow(row)

    result = load_paired_results(path)

    assert len(result) == 1
    assert result[0]["v1_precision_at_10"] == 0.2
    assert result[0]["v4_precision_at_10"] == 0.3


def test_duplicate_topic_rejected(tmp_path):
    path = tmp_path / "results.csv"

    fields = ["topic_id"]

    for suffix in METRICS.values():
        fields.extend([
            "v1_" + suffix,
            "v4_" + suffix,
        ])

    row = {"topic_id": "1"}

    for suffix in METRICS.values():
        row["v1_" + suffix] = 0.2
        row["v4_" + suffix] = 0.3

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerow(row)
        writer.writerow(row)

    with pytest.raises(ValueError):
        load_paired_results(path)
