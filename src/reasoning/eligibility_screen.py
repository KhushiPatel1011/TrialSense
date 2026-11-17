import argparse
import re

from src.reasoning.show_trial_evidence import (
    load_selected_trials,
    LABELS,
)
from src.retrieval.search_bm25 import (
    load_patient_topic,
    load_relevance_judgments,
    search_bm25,
)


def extract_patient_age(text):
    """Extract an explicitly stated patient age."""
    patterns = [
        r"\b(\d{1,3})\s*[- ]year[- ]old\b",
        r"\b(\d{1,3})\s*years?\s+of\s+age\b",
        r"\baged?\s+(\d{1,3})\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            age = int(match.group(1))
            if 0 <= age <= 120:
                return age

    return None


def extract_patient_sex(text):
    """Extract explicitly stated patient sex descriptors."""
    patterns = [
        (r"\b(?:male|man|boy)\b", "Male"),
        (r"\b(?:female|woman|girl)\b", "Female"),
    ]

    matches = []

    for pattern, label in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            matches.append(label)

    # Avoid guessing when descriptors conflict.
    return matches[0] if len(matches) == 1 else None


def parse_age_limit(value):
    """Convert trial age values to years."""
    if not value:
        return None

    value = value.strip().lower()

    if value in ("n/a", "not applicable", "unknown"):
        return None

    match = re.fullmatch(
        r"(\d+(?:\.\d+)?)\s*"
        r"(years?|months?|weeks?|days?)",
        value,
    )

    if not match:
        return None

    number = float(match.group(1))
    unit = match.group(2)

    if unit.startswith("year"):
        return number

    if unit.startswith("month"):
        return number / 12

    if unit.startswith("week"):
        return number / 52.1775

    if unit.startswith("day"):
        return number / 365.25

    return None


def evaluate_demographics(patient_age, patient_sex, trial):
    """Check age and sex without inferring clinical eligibility."""

    checks = []
    conflicts = []
    unknowns = []

    min_age = parse_age_limit(trial.get("minimum_age"))
    max_age = parse_age_limit(trial.get("maximum_age"))

    if patient_age is None:
        unknowns.append("Patient age not established")

    elif min_age is not None and patient_age < min_age:
        conflicts.append(
            f"Patient age {patient_age} is below "
            f"minimum {min_age:g} years"
        )

    elif max_age is not None and patient_age > max_age:
        conflicts.append(
            f"Patient age {patient_age} exceeds "
            f"maximum {max_age:g} years"
        )

    elif min_age is None and max_age is None:
        if (
            trial.get("minimum_age", "").strip().lower()
            in ("", "n/a")
            and trial.get("maximum_age", "").strip().lower()
            in ("", "n/a")
        ):
            checks.append("No explicit numeric age restriction")
        else:
            unknowns.append("Trial age limits could not be parsed")

    else:
        checks.append("Age requirement satisfied")

    trial_sex = (trial.get("gender") or "").strip().lower()

    if patient_sex is None:
        unknowns.append("Patient sex not established")

    elif trial_sex in ("all", "both"):
        checks.append("Trial accepts all sexes")

    elif trial_sex in ("male", "female"):
        if patient_sex.lower() == trial_sex:
            checks.append("Sex requirement satisfied")
        else:
            conflicts.append(
                f"Patient sex {patient_sex} conflicts "
                f"with trial requirement {trial_sex.title()}"
            )

    else:
        unknowns.append(
            "Trial sex requirement not established"
        )

    if conflicts:
        status = "CONFLICT"
    elif unknowns:
        status = "UNKNOWN"
    else:
        status = "PASS"

    return {
        "status": status,
        "checks": checks,
        "conflicts": conflicts,
        "unknowns": unknowns,
    }


def main():
    parser = argparse.ArgumentParser(
        description="TrialSense demographic eligibility screening"
    )
    parser.add_argument("--topic", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=3)

    args = parser.parse_args()

    patient = load_patient_topic(args.topic)
    judgments = load_relevance_judgments(args.topic)

    patient_age = extract_patient_age(patient)
    patient_sex = extract_patient_sex(patient)

    retrieved, latency = search_bm25(
        patient, top_k=args.top_k
    )

    trial_ids = {
        result["nct_id"] for result in retrieved
    }

    trials = load_selected_trials(trial_ids)

    print("=" * 70)
    print("TRIALSENSE - STRUCTURED ELIGIBILITY SCREENING")
    print("=" * 70)

    print(f"\nPatient topic: {args.topic}")
    print(f"Extracted age: {patient_age}")
    print(f"Extracted sex: {patient_sex}")
    print(f"Retrieval latency: {latency * 1000:.2f} ms")

    for rank, result in enumerate(retrieved, start=1):
        nct_id = result["nct_id"]
        trial = trials.get(nct_id)

        print("\n" + "-" * 70)
        print(f"RANK {rank}: {nct_id}")
        print("-" * 70)

        if trial is None:
            print("Trial record missing")
            continue

        evidence = evaluate_demographics(
            patient_age, patient_sex, trial
        )

        grade = judgments.get(nct_id)

        print(f"Title: {trial['title']}")
        print(f"TREC reference label: {LABELS[grade]}")
        print(f"Trial age range: "
              f"{trial['minimum_age']} to {trial['maximum_age']}")
        print(f"Trial sex requirement: {trial['gender']}")
        print(f"Demographic screen: {evidence['status']}")

        for check in evidence["checks"]:
            print(f"  [PASS] {check}")

        for conflict in evidence["conflicts"]:
            print(f"  [CONFLICT] {conflict}")

        for unknown in evidence["unknowns"]:
            print(f"  [UNKNOWN] {unknown}")

        print(
            "Clinical eligibility: NOT DETERMINED"
        )
        print(
            "Note: Other inclusion/exclusion criteria "
            "have not been automatically assessed."
        )

    print("\n" + "=" * 70)
    print("DEMOGRAPHIC SCREENING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
