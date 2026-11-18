from src.reasoning.numerical_eligibility import (
    extract_testosterone_thresholds,
    compare_values,
    evaluate_numerical_criteria,
)


def test_extract_threshold():
    criteria = "Serum testosterone < 100 ng/dL"

    result = extract_testosterone_thresholds(criteria)

    assert len(result) == 1
    assert result[0]["operator"] == "<"
    assert result[0]["value"] == 100.0


def test_less_than_satisfied():
    assert compare_values(65, "<", 100) is True


def test_less_than_conflict():
    assert compare_values(125, "<", 100) is False


def test_boundary_strict():
    assert compare_values(100, "<", 100) is False


def test_boundary_inclusive():
    assert compare_values(100, "<=", 100) is True


def test_numerical_criterion_satisfied():
    patient = "Serum testosterone level of 65 ng/dL."
    criteria = "Serum testosterone < 100 ng/dL."

    result = evaluate_numerical_criteria(
        patient, criteria
    )

    assert result["status"] == "SATISFIED"


def test_numerical_conflict():
    patient = "Testosterone 125 ng/dL."
    criteria = "Testosterone < 100 ng/dL."

    result = evaluate_numerical_criteria(
        patient, criteria
    )

    assert result["status"] == "CONFLICT"


def test_missing_patient_measurement():
    patient = "Patient has low testosterone."
    criteria = "Testosterone < 100 ng/dL."

    result = evaluate_numerical_criteria(
        patient, criteria
    )

    assert result["status"] == "UNKNOWN"


def test_missing_supported_threshold():
    patient = "Testosterone 65 ng/dL."
    criteria = "Patients must have low testosterone."

    result = evaluate_numerical_criteria(
        patient, criteria
    )

    assert result["status"] == "UNKNOWN"


def test_multiple_patient_measurements():
    patient = (
        "Testosterone 65 ng/dL initially, "
        "testosterone 125 ng/dL after treatment."
    )
    criteria = "Testosterone < 100 ng/dL."

    result = evaluate_numerical_criteria(
        patient, criteria
    )

    assert result["status"] == "UNKNOWN"


def test_multiple_thresholds():
    patient = "Testosterone 125 ng/dL."
    criteria = (
        "Testosterone < 350 ng/dL "
        "and testosterone > 100 ng/dL."
    )

    result = evaluate_numerical_criteria(
        patient, criteria
    )

    assert result["status"] == "SATISFIED"


def test_multiple_thresholds_conflict():
    patient = "Testosterone 65 ng/dL."
    criteria = (
        "Testosterone < 350 ng/dL "
        "and testosterone > 100 ng/dL."
    )

    result = evaluate_numerical_criteria(
        patient, criteria
    )

    assert result["status"] == "CONFLICT"
