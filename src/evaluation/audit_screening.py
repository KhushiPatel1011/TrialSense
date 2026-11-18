
import csv
import json
from pathlib import Path

from src.reasoning.combined_screen import (
    combined_eligibility_screen,
)
from src.retrieval.search_bm25 import (
    load_patient_topic,
)


ROOT = Path(__file__).resolve().parents[2]

RESULTS_PATH = (
    ROOT / "results" / "screening_evaluation.csv"
)

TRIALS_PATH = (
    ROOT / "data" / "processed" / "trials_part1.jsonl"
)


def find_disagreements():
    """
    Find pairs labeled eligible by TREC but
    flagged as conflicting by our screening.
    """
    disagreements = []

    with RESULTS_PATH.open(
        "r", encoding="utf-8", newline=""
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            if (
                row["trec_label"] == "eligible"
                and row["screen_status"] == "CONFLICT"
            ):
                disagreements.append(row)

    return disagreements


def load_trials(required_ids):
    """Load only the trials needed for error analysis."""
    selected = {}

    with TRIALS_PATH.open(
        "r", encoding="utf-8"
    ) as file:
        for line in file:
            trial = json.loads(line)
            nct_id = trial["nct_id"]

            if nct_id in required_ids:
                selected[nct_id] = trial

                if len(selected) == len(required_ids):
                    break

    missing = required_ids - set(selected)

    if missing:
        raise RuntimeError(
            f"Missing trial records: {sorted(missing)}"
        )

    return selected


def show_evidence_context(criteria, evidence, window=180):
    """Show surrounding text for a matched criterion."""
    if not evidence:
        return

    position = criteria.lower().find(
        evidence.strip().lower()
    )

    if position < 0:
        print("  Evidence not located in original criteria")
        return

    start = max(0, position - window)
    end = min(
        len(criteria),
        position + len(evidence) + window,
    )

    excerpt = criteria[start:end]

    print("  Source context:")
    print(f"  ...{excerpt}...")


def audit_case(row, trial):
    topic_id = row["topic_id"]
    nct_id = row["nct_id"]

    patient = load_patient_topic(topic_id)

    result = combined_eligibility_screen(
        patient, trial
    )

    print("\n" + "=" * 75)
    print(f"TOPIC {topic_id} | {nct_id}")
    print("=" * 75)

    print(f"\nRetrieval rank: {row['rank']}")
    print(f"Trial title: {trial['title']}")
    print(f"TREC judgment: Eligible")

    print("\nPATIENT DESCRIPTION")
    print(patient)

    print("\nEXTRACTED PATIENT INFORMATION")
    print(f"Age: {result['patient_age']}")
    print(f"Sex: {result['patient_sex']}")

    print("\nTRIAL REQUIREMENTS")
    print(f"Minimum age: {trial['minimum_age']}")
    print(f"Maximum age: {trial['maximum_age']}")
    print(f"Sex requirement: {trial['gender']}")

    print("\nSCREENING RESULTS")
    print(f"Combined: {result['status']}")
    print(
        "Demographic: "
        f"{result['demographics']['status']}"
    )
    print(
        "Numerical: "
        f"{result['numerical']['status']}"
    )

    for conflict in result["demographics"]["conflicts"]:
        print(f"\n[DEMOGRAPHIC CONFLICT] {conflict}")

    for unknown in result["demographics"]["unknowns"]:
        print(f"\n[DEMOGRAPHIC UNKNOWN] {unknown}")

    for check in result["numerical"]["checks"]:
        status = (
            "SATISFIED"
            if check["satisfied"]
            else "CONFLICT"
        )

        print(f"\n[NUMERICAL {status}]")
        print(
            f"Patient value: {check['patient_value']} "
            f"{check['unit']}"
        )
        print(
            f"Requirement: {check['operator']} "
            f"{check['threshold']} {check['unit']}"
        )
        print(
            f"Patient evidence: {check['patient_evidence']}"
        )
        print(
            f"Trial evidence: {check['trial_evidence']}"
        )

        if not check["satisfied"]:
            show_evidence_context(
                trial["eligibility_criteria"],
                check["trial_evidence"],
            )

    print("\nORIGINAL ELIGIBILITY CRITERIA")
    print(trial["eligibility_criteria"][:3000])

    if len(trial["eligibility_criteria"]) > 3000:
        print("\n[Criteria truncated for display]")

    print("\nClinical eligibility: NOT DETERMINED")


def main():
    print("=" * 75)
    print("TRIALSENSE - SCREENING ERROR ANALYSIS")
    print("=" * 75)

    disagreements = find_disagreements()

    print(
        "\nEligible-labeled pairs flagged as conflicts: "
        f"{len(disagreements)}"
    )

    if not disagreements:
        print("No disagreements found.")
        return

    required_ids = {
        row["nct_id"]
        for row in disagreements
    }

    trials = load_trials(required_ids)

    for row in disagreements:
        audit_case(
            row,
            trials[row["nct_id"]],
        )

    print("\n" + "=" * 75)
    print("AUDIT COMPLETE")
    print("=" * 75)

    print(
        "\nNote: A disagreement does not automatically "
        "establish an incorrect TREC label or parser."
    )


if __name__ == "__main__":
    main()
