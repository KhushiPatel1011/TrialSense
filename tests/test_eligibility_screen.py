import pytest

from src.reasoning.eligibility_screen import (
    extract_patient_age,
    extract_patient_sex,
    parse_age_limit,
    evaluate_demographics,
)


def make_trial(
    minimum_age="18 Years",
    maximum_age="65 Years",
    gender="All",
):
    return {
        "minimum_age": minimum_age,
        "maximum_age": maximum_age,
        "gender": gender,
    }


def test_extract_patient_age():
    text = "A 19-year-old male presents with low testosterone."
    assert extract_patient_age(text) == 19


def test_extract_age_alternative_format():
    text = "A patient aged 45 has hypertension."
    assert extract_patient_age(text) == 45


def test_extract_patient_sex():
    assert extract_patient_sex(
        "A 19-year-old male presents with fatigue."
    ) == "Male"

    assert extract_patient_sex(
        "A 32-year-old woman presents with pain."
    ) == "Female"


def test_patient_sex_with_other_people():
    text = (
        "A 32-year-old woman has multiple male partners."
    )
    assert extract_patient_sex(text) == "Female"


def test_explicit_female_patient():
    text = (
        "A female patient reports multiple male partners."
    )
    assert extract_patient_sex(text) == "Female"


def test_explicit_male_patient():
    text = (
        "A male patient reports a female partner."
    )
    assert extract_patient_sex(text) == "Male"


def test_unrelated_sex_mention():
    text = (
        "The patient's male partner attended the visit."
    )
    assert extract_patient_sex(text) is None


def test_missing_sex_descriptor():
    text = (
        "A 45-year-old patient presents with headache."
    )
    assert extract_patient_sex(text) is None


def test_age_unit_conversion():
    assert parse_age_limit("18 Years") == 18
    assert parse_age_limit("24 Months") == 2
    assert parse_age_limit("N/A") is None


def test_eligible_demographics():
    trial = make_trial(
        minimum_age="18 Years",
        maximum_age="65 Years",
        gender="Male",
    )

    result = evaluate_demographics(
        patient_age=19,
        patient_sex="Male",
        trial=trial,
    )

    assert result["status"] == "PASS"
    assert len(result["conflicts"]) == 0


def test_age_conflict():
    trial = make_trial(
        minimum_age="65 Years",
        maximum_age="N/A",
        gender="Male",
    )

    result = evaluate_demographics(
        patient_age=19,
        patient_sex="Male",
        trial=trial,
    )

    assert result["status"] == "CONFLICT"
    assert any(
        "below minimum" in conflict
        for conflict in result["conflicts"]
    )


def test_sex_conflict():
    trial = make_trial(
        minimum_age="18 Years",
        maximum_age="65 Years",
        gender="Female",
    )

    result = evaluate_demographics(
        patient_age=19,
        patient_sex="Male",
        trial=trial,
    )

    assert result["status"] == "CONFLICT"


def test_missing_patient_information():
    trial = make_trial()

    result = evaluate_demographics(
        patient_age=None,
        patient_sex=None,
        trial=trial,
    )

    assert result["status"] == "UNKNOWN"
    assert len(result["unknowns"]) == 2


def test_upper_age_limit_conflict():
    trial = make_trial(
        minimum_age="18 Years",
        maximum_age="40 Years",
        gender="All",
    )

    result = evaluate_demographics(
        patient_age=70,
        patient_sex="Female",
        trial=trial,
    )

    assert result["status"] == "CONFLICT"
    assert any(
        "exceeds maximum" in conflict
        for conflict in result["conflicts"]
    )


def test_missing_trial_sex_requirement():
    trial = make_trial(gender="Unknown")

    result = evaluate_demographics(
        patient_age=30,
        patient_sex="Male",
        trial=trial,
    )

    assert result["status"] == "UNKNOWN"


def test_unparseable_age_limit():
    trial = make_trial(
        minimum_age="Adults",
        maximum_age="N/A",
        gender="All",
    )

    result = evaluate_demographics(
        patient_age=30,
        patient_sex="Female",
        trial=trial,
    )

    assert result["status"] == "UNKNOWN"
