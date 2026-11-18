
import argparse

from src.reasoning.eligibility_screen import (
    extract_patient_age,
    extract_patient_sex,
    evaluate_demographics,
)
from src.reasoning.numerical_eligibility import (
    evaluate_numerical_criteria,
)
from src.reasoning.show_trial_evidence import (
    load_selected_trials,
    LABELS,
)
from src.retrieval.search_bm25 import (
    load_patient_topic,
    load_relevance_judgments,
    search_bm25,
)


def combined_eligibility_screen(patient_text, trial):
    """
    Combine supported demographic and numerical checks.

    This is a limited evidence-based screening system.
    It does not determine full clinical eligibility.
    """

    patient_age = extract_patient_age(patient_text)
    patient_sex = extract_patient_sex(patient_text)

    demographics = evaluate_demographics(
        patient_age,
        patient_sex,
        trial,
    )

    numerical = evaluate_numerical_criteria(
        patient_text,
        trial.get("eligibility_criteria", ""),
    )

    demographic_conflict = (
        demographics["status"] == "CONFLICT"
    )

    numerical_conflict = (
        numerical["status"] == "CONFLICT"
    )

    if demographic_conflict or numerical_conflict:
        status = "CONFLICT"

    elif (
        demographics["status"] == "UNKNOWN"
        or numerical["status"] == "UNKNOWN"
    ):
        status = "INSUFFICIENT_EVIDENCE"

    else:
        status = "NO_DETECTED_CONFLICT"

    return {
        "status": status,
        "patient_age": patient_age,
        "patient_sex": patient_sex,
        "demographics": demographics,
        "numerical": numerical,
        "clinical_eligibility": "NOT DETERMINED",
    }


def display_screening(result):
    print(f"Combined screening: {result['status']}")

    print(
        "Demographic screening: "
        f"{result['demographics']['status']}"
    )

    for item in result["demographics"]["checks"]:
        print(f"  [PASS] {item}")

    for item in result["demographics"]["conflicts"]:
        print(f"  [CONFLICT] {item}")

    for item in result["demographics"]["unknowns"]:
        print(f"  [UNKNOWN] {item}")

    numerical = result["numerical"]

    print(f"Numerical screening: {numerical['status']}")
    print(f"  Reason: {numerical['reason']}")

    for check in numerical["checks"]:
        label = (
            "SATISFIED"
            if check["satisfied"]
            else "CONFLICT"
        )

        print(
            f"  [{label}] "
            f"{check['patient_value']} {check['unit']} "
            f"{check['operator']} "
            f"{check['threshold']} {check['unit']}"
        )

        print(
            f"    Patient evidence: "
            f"{check['patient_evidence']}"
        )

        print(
            f"    Trial evidence: "
            f"{check['trial_evidence']}"
        )

    print(
        "Overall clinical eligibility: "
        f"{result['clinical_eligibility']}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="TrialSense combined eligibility screening"
    )

    parser.add_argument(
        "--topic", type=int, default=1
    )

    parser.add_argument(
        "--top-k", type=int, default=3
    )

    args = parser.parse_args()

    patient = load_patient_topic(args.topic)
    judgments = load_relevance_judgments(args.topic)

    retrieved, latency = search_bm25(
        patient, top_k=args.top_k
    )

    selected_ids = {
        trial["nct_id"] for trial in retrieved
    }

    trial_records = load_selected_trials(selected_ids)

    print("=" * 70)
    print("TRIALSENSE - COMBINED ELIGIBILITY SCREENING")
    print("=" * 70)

    print(f"\nPatient topic: {args.topic}")
    print(f"Patient age: {extract_patient_age(patient)}")
    print(f"Patient sex: {extract_patient_sex(patient)}")
    print(f"Retrieval latency: {latency * 1000:.2f} ms")

    for rank, item in enumerate(retrieved, start=1):
        nct_id = item["nct_id"]
        trial = trial_records.get(nct_id)

        print("\n" + "-" * 70)
        print(f"RANK {rank} | {nct_id}")
        print("-" * 70)

        if trial is None:
            print("Trial record unavailable")
            continue

        print(f"Title: {trial['title']}")
        print(
            "TREC reference judgment: "
            f"{LABELS[judgments.get(nct_id)]}"
        )

        result = combined_eligibility_screen(
            patient, trial
        )

        display_screening(result)

    print("\n" + "=" * 70)
    print("COMBINED SCREENING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
