from src.reasoning.clinical_evidence import (
    extract_testosterone_measurements,
    extract_clinical_evidence,
)


def test_testosterone_level_of():
    text = "Serum testosterone level of 65 ng/dL."

    result = extract_testosterone_measurements(text)

    assert len(result) == 1
    assert result[0]["value"] == 65.0
    assert result[0]["unit"] == "ng/dL"


def test_testosterone_direct_value():
    text = "Testosterone 125 ng/dL."

    result = extract_testosterone_measurements(text)

    assert len(result) == 1
    assert result[0]["value"] == 125.0


def test_testosterone_decimal():
    text = "Total testosterone was 65.5 ng/dL."

    result = extract_testosterone_measurements(text)

    assert len(result) == 1
    assert result[0]["value"] == 65.5


def test_missing_testosterone():
    text = "A patient has elevated serum glucose."

    result = extract_testosterone_measurements(text)

    assert result == []


def test_unsupported_unit():
    text = "Testosterone 2.5 nmol/L."

    result = extract_testosterone_measurements(text)

    assert result == []


def test_no_inferred_measurement():
    text = "Patient has low testosterone."

    result = extract_testosterone_measurements(text)

    assert result == []


def test_multiple_measurements():
    text = (
        "Testosterone 65 ng/dL initially. "
        "Testosterone 80 ng/dL after treatment."
    )

    result = extract_testosterone_measurements(text)

    assert len(result) == 2
    assert result[0]["value"] == 65.0
    assert result[1]["value"] == 80.0


def test_clinical_evidence_structure():
    text = "Serum testosterone level of 65 ng/dL."

    result = extract_clinical_evidence(text)

    assert "testosterone" in result
    assert len(result["testosterone"]) == 1
    assert result["testosterone"][0]["biomarker"] == "testosterone"
