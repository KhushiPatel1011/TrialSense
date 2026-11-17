
import argparse
import json
from pathlib import Path

from src.retrieval.search_bm25 import (
    load_patient_topic,
    load_relevance_judgments,
    search_bm25,
)

ROOT = Path(__file__).resolve().parents[2]

TRIALS_PATH = (
    ROOT / "data" / "processed" / "trials_part1.jsonl"
)

LABELS = {
    0: "Non-relevant",
    1: "Excluded",
    2: "Eligible",
    None: "Unjudged",
}


def load_selected_trials(nct_ids):
    """Read only required records from the JSONL corpus."""

    selected = {}

    with TRIALS_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            trial = json.loads(line)
            nct_id = trial["nct_id"]

            if nct_id in nct_ids:
                selected[nct_id] = trial

                if len(selected) == len(nct_ids):
                    break

    return selected


def shorten(text, max_chars=1200):
    """Limit long fields for readable terminal output."""
    text = " ".join((text or "").split())

    if len(text) <= max_chars:
        return text or "Not specified"

    return text[:max_chars] + " ... [truncated]"


def main():
    parser = argparse.ArgumentParser(
        description="TrialSense evidence viewer"
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

    results, latency = search_bm25(
        patient, top_k=args.top_k
    )

    selected_ids = {
        result["nct_id"] for result in results
    }

    trials = load_selected_trials(selected_ids)

    print("=" * 70)
    print("TRIALSENSE - CLINICAL TRIAL EVIDENCE VIEWER")
    print("=" * 70)

    print(f"\nPatient topic: {args.topic}")
    print(f"Patient description:\n{patient}")

    print(f"\nSearch latency: {latency * 1000:.2f} ms")

    for rank, result in enumerate(results, start=1):
        nct_id = result["nct_id"]
        trial = trials.get(nct_id)

        print("\n" + "=" * 70)
        print(f"RANK {rank} | {nct_id}")
        print("=" * 70)

        if trial is None:
            print("Trial record not found")
            continue

        grade = judgments.get(nct_id)

        print(f"\nTitle: {trial['title']}")
        print(f"Conditions: {trial['conditions']}")
        print(f"TREC judgment: {LABELS[grade]}")

        print("\nStructured eligibility information:")
        print(f"Sex: {trial['gender'] or 'Not specified'}")
        print(
            f"Minimum age: "
            f"{trial['minimum_age'] or 'Not specified'}"
        )
        print(
            f"Maximum age: "
            f"{trial['maximum_age'] or 'Not specified'}"
        )
        print(f"Study status: {trial['status']}")

        print("\nOfficial eligibility criteria:")
        print(
            shorten(trial["eligibility_criteria"])
        )

        print("\nInterpretation:")
        print(
            "The TREC label is an external relevance "
            "judgment, not a prediction made by TrialSense."
        )
        print(
            "Eligibility criteria are displayed as "
            "source evidence; automatic eligibility "
            "reasoning has not yet been implemented."
        )

    print("\n" + "=" * 70)
    print("EVIDENCE VIEWER COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
