
from src.reasoning.combined_screen import (
    combined_eligibility_screen,
)


PATIENT = (
    "A 19-year-old male has serum testosterone "
    "level of 65 ng/dL."
)


def make_trial(
    minimum_age="18 Years",
    maximum_age="65 Years",
    gender="Male",
    criteria="Serum testosterone < 100 ng/dL.",
):
    return {
        "minimum_age": minimum_age,
        "maximum_age": maximum_age,
        "gender": gender,
        "eligibility_criteria": criteria,
    }


def test_all_supported_checks_satisfied():
    result = combined_eligibility_screen(
        PATIENT, make_trial()
    )

    assert result["status"] == "NO_DETECTED_CONFLICT"
    assert result["demographics"]["status"] == "PASS"
    assert result["numerical"]["status"] == "SATISFIED"
    assert result["clinical_eligibility"] == "NOT DETERMINED"


def test_demographic_conflict():
    trial = make_trial(minimum_age="65 Years")

    result = combined_eligibility_screen(PATIENT, trial)

    assert result["status"] == "CONFLICT"
    assert result["demographics"]["status"] == "CONFLICT"


def test_numerical_conflict():
    trial = make_trial(
        criteria=(
            "Total serum testosterone level (TT) "
            "< 350 ng/dL and > 100 ng/dL."
        )
    )

    result = combined_eligibility_screen(PATIENT, trial)

    assert result["status"] == "CONFLICT"
    assert result["numerical"]["status"] == "CONFLICT"


def test_two_independent_conflicts():
    trial = make_trial(
        minimum_age="65 Years",
        criteria=(
            "Total serum testosterone level (TT) "
            "< 350 ng/dL and > 100 ng/dL."
        ),
    )

    result = combined_eligibility_screen(PATIENT, trial)

    assert result["status"] == "CONFLICT"
    assert result["demographics"]["status"] == "CONFLICT"
    assert result["numerical"]["status"] == "CONFLICT"


def test_unknown_numerical_information():
    trial = make_trial(
        criteria="Patients must have normal hormone levels."
    )

    result = combined_eligibility_screen(PATIENT, trial)

    assert result["status"] == "INSUFFICIENT_EVIDENCE"
    assert result["demographics"]["status"] == "PASS"
    assert result["numerical"]["status"] == "UNKNOWN"


def test_missing_patient_demographics():
    patient = "Serum testosterone level of 65 ng/dL."

    result = combined_eligibility_screen(
        patient, make_trial()
    )

    assert result["status"] == "INSUFFICIENT_EVIDENCE"
    assert result["demographics"]["status"] == "UNKNOWN"


def test_known_conflict_overrides_unknown():
    patient = "A 19-year-old patient has no lab measurements."
    trial = make_trial(minimum_age="65 Years")

    result = combined_eligibility_screen(patient, trial)

    assert result["status"] == "CONFLICT"
    assert result["numerical"]["status"] == "UNKNOWN"
