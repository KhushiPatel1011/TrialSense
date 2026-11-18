
import pytest

from src.evaluation.ablation_study import (
    screening_status,
    rank_for_method,
)


def make_screen(demographic, numerical):
    conflict = (
        demographic == "CONFLICT"
        or numerical == "CONFLICT"
    )

    return {
        "status": (
            "CONFLICT"
            if conflict
            else "INSUFFICIENT_EVIDENCE"
        ),
        "demographics": {
            "status": demographic
        },
        "numerical": {
            "status": numerical
        },
    }


def test_demographic_conflict_only():
    screen = make_screen(
        "CONFLICT", "SATISFIED"
    )

    assert screening_status(
        screen, "demographic"
    ) == "CONFLICT"

    assert screening_status(
        screen, "numerical"
    ) != "CONFLICT"


def test_numerical_conflict_only():
    screen = make_screen(
        "PASS", "CONFLICT"
    )

    assert screening_status(
        screen, "numerical"
    ) == "CONFLICT"

    assert screening_status(
        screen, "demographic"
    ) != "CONFLICT"


def test_combined_detects_either_conflict():
    screen = make_screen(
        "CONFLICT", "UNKNOWN"
    )

    assert screening_status(
        screen, "combined"
    ) == "CONFLICT"


def test_unknown_is_not_a_conflict():
    screen = make_screen(
        "PASS", "UNKNOWN"
    )

    for method in (
        "demographic",
        "numerical",
        "combined",
    ):
        assert screening_status(
            screen, method
        ) != "CONFLICT"


def test_ablation_ranking():
    candidates = [
        {"nct_id": "A"},
        {"nct_id": "B"},
        {"nct_id": "C"},
    ]

    screens = {
        "A": make_screen(
            "CONFLICT", "SATISFIED"
        ),
        "B": make_screen(
            "PASS", "CONFLICT"
        ),
        "C": make_screen(
            "PASS", "UNKNOWN"
        ),
    }

    demographic = rank_for_method(
        candidates, screens, "demographic"
    )

    numerical = rank_for_method(
        candidates, screens, "numerical"
    )

    combined = rank_for_method(
        candidates, screens, "combined"
    )

    assert [
        item["nct_id"] for item in demographic
    ] == ["B", "C", "A"]

    assert [
        item["nct_id"] for item in numerical
    ] == ["A", "C", "B"]

    assert [
        item["nct_id"] for item in combined
    ] == ["C", "A", "B"]


def test_baseline_preserves_original_order():
    candidates = [
        {"nct_id": "A"},
        {"nct_id": "B"},
    ]

    result = rank_for_method(
        candidates, {}, "baseline"
    )

    assert result == candidates


def test_unsupported_method():
    screen = make_screen(
        "PASS", "UNKNOWN"
    )

    with pytest.raises(ValueError):
        screening_status(
            screen, "unsupported"
        )
