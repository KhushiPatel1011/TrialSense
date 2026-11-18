
from src.reasoning.numerical_eligibility import (
    extract_testosterone_thresholds,
    evaluate_numerical_criteria,
)


def test_compound_range_extraction():
    criteria = (
        "Total serum testosterone level (TT) "
        "< 350 ng/dL and > 100 ng/dL."
    )

    thresholds = extract_testosterone_thresholds(criteria)

    assert len(thresholds) == 2
    assert thresholds[0]["operator"] == "<"
    assert thresholds[0]["value"] == 350.0
    assert thresholds[1]["operator"] == ">"
    assert thresholds[1]["value"] == 100.0


def test_compound_range_conflict():
    patient = "Serum testosterone level of 65 ng/dL."
    criteria = (
        "Total serum testosterone level (TT) "
        "< 350 ng/dL and > 100 ng/dL."
    )

    result = evaluate_numerical_criteria(
        patient, criteria
    )

    assert result["status"] == "CONFLICT"
    assert len(result["checks"]) == 2
    assert result["checks"][0]["satisfied"] is True
    assert result["checks"][1]["satisfied"] is False


def test_compound_range_satisfied():
    patient = "Testosterone 150 ng/dL."
    criteria = (
        "Total serum testosterone level (TT) "
        "< 350 ng/dL and > 100 ng/dL."
    )

    result = evaluate_numerical_criteria(
        patient, criteria
    )

    assert result["status"] == "SATISFIED"
    assert len(result["checks"]) == 2


def test_exclusion_only_threshold_not_treated_as_inclusion():
    patient = "Testosterone 65 ng/dL."
    criteria = (
        "Exclusion Criteria: "
        "Testosterone < 100 ng/dL."
    )

    result = evaluate_numerical_criteria(
        patient, criteria
    )

    assert result["status"] == "UNKNOWN"


def test_labeled_inclusion_and_exclusion():
    patient = "Testosterone 65 ng/dL."
    criteria = (
        "Inclusion Criteria: Testosterone < 100 ng/dL. "
        "Exclusion Criteria: Testosterone < 50 ng/dL."
    )

    result = evaluate_numerical_criteria(
        patient, criteria
    )

    assert result["status"] == "SATISFIED"
    assert len(result["checks"]) == 1
