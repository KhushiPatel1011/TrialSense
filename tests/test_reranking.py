
from src.evaluation.evaluate_reranking import (
    rerank_candidates,
)


def test_conflicts_are_demoted():
    candidates = [
        {"nct_id": "A"},
        {"nct_id": "B"},
        {"nct_id": "C"},
    ]

    screenings = {
        "A": {"status": "CONFLICT"},
        "B": {"status": "INSUFFICIENT_EVIDENCE"},
        "C": {"status": "NO_DETECTED_CONFLICT"},
    }

    result = rerank_candidates(
        candidates, screenings
    )

    assert [r["nct_id"] for r in result] == [
        "B", "C", "A"
    ]


def test_original_order_preserved_within_groups():
    candidates = [
        {"nct_id": "A"},
        {"nct_id": "B"},
        {"nct_id": "C"},
        {"nct_id": "D"},
    ]

    screenings = {
        "A": {"status": "CONFLICT"},
        "B": {"status": "INSUFFICIENT_EVIDENCE"},
        "C": {"status": "CONFLICT"},
        "D": {"status": "NO_DETECTED_CONFLICT"},
    }

    result = rerank_candidates(
        candidates, screenings
    )

    assert [r["nct_id"] for r in result] == [
        "B", "D", "A", "C"
    ]


def test_no_conflicts_preserves_bm25_order():
    candidates = [
        {"nct_id": "A"},
        {"nct_id": "B"},
    ]

    screenings = {
        "A": {"status": "INSUFFICIENT_EVIDENCE"},
        "B": {"status": "NO_DETECTED_CONFLICT"},
    }

    result = rerank_candidates(
        candidates, screenings
    )

    assert [r["nct_id"] for r in result] == [
        "A", "B"
    ]
