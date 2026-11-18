
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from src.reasoning.combined_screen import (
    combined_eligibility_screen,
)
from src.retrieval.search_bm25 import (
    load_patient_topic,
    search_bm25,
)
from src.evaluation.evaluate_bm25 import (
    load_all_qrels,
)


ROOT = Path(__file__).resolve().parents[2]

TRIALS_PATH = (
    ROOT / "data" / "processed" / "trials_part1.jsonl"
)

OUTPUT_PATH = (
    ROOT / "results" / "screening_evaluation.csv"
)


def load_required_trials(required_ids):
    """
    Read the JSONL corpus once and retain only
    documents needed for the evaluation.
    """
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
            f"Missing {len(missing)} trial documents"
        )

    return selected


def collect_candidates(topic_ids, top_k):
    """Retrieve BM25 candidates for every topic."""

    all_candidates = {}
    required_ids = set()

    for topic_id in topic_ids:
        patient = load_patient_topic(topic_id)

        results, latency = search_bm25(
            patient,
            top_k=top_k,
        )

        if len(results) != top_k:
            raise RuntimeError(
                f"Topic {topic_id}: expected {top_k} "
                f"results, received {len(results)}"
            )

        all_candidates[topic_id] = {
            "patient": patient,
            "results": results,
            "latency_ms": latency * 1000,
        }

        required_ids.update(
            result["nct_id"] for result in results
        )

    return all_candidates, required_ids


def safe_rate(numerator, denominator):
    if denominator == 0:
        return None

    return numerator / denominator


def format_rate(value):
    if value is None:
        return "N/A"

    return f"{value:.2%}"


def evaluate(top_k):
    print("=" * 68)
    print("TRIALSENSE - ELIGIBILITY SCREENING EVALUATION")
    print("=" * 68)

    qrels = load_all_qrels()
    topic_ids = sorted(qrels, key=int)

    print(f"\nPatient topics: {len(topic_ids)}")
    print(f"Retrieval depth: {top_k}")

    candidates, required_ids = collect_candidates(
        topic_ids, top_k
    )

    print(
        f"Unique retrieved trial documents: "
        f"{len(required_ids)}"
    )

    print("Loading required trial documents...")
    trials = load_required_trials(required_ids)

    rows = []
    status_counts = Counter()
    label_counts = Counter()

    for topic_id in topic_ids:
        patient = candidates[topic_id]["patient"]
        results = candidates[topic_id]["results"]
        judgments = qrels[topic_id]

        for rank, item in enumerate(results, start=1):
            nct_id = item["nct_id"]
            trial = trials[nct_id]

            screen = combined_eligibility_screen(
                patient,
                trial,
            )

            grade = judgments.get(nct_id)

            label = {
                0: "non_relevant",
                1: "excluded",
                2: "eligible",
                None: "unjudged",
            }[grade]

            status = screen["status"]

            status_counts[status] += 1
            label_counts[label] += 1

            demographic_conflicts = (
                screen["demographics"]["conflicts"]
            )

            numerical_conflicts = [
                check
                for check in screen["numerical"]["checks"]
                if not check["satisfied"]
            ]

            rows.append({
                "topic_id": topic_id,
                "rank": rank,
                "nct_id": nct_id,
                "trec_grade": (
                    grade if grade is not None else ""
                ),
                "trec_label": label,
                "screen_status": status,
                "demographic_status":
                    screen["demographics"]["status"],
                "numerical_status":
                    screen["numerical"]["status"],
                "demographic_conflicts":
                    len(demographic_conflicts),
                "numerical_conflicts":
                    len(numerical_conflicts),
                "demographic_evidence":
                    " | ".join(demographic_conflicts),
                "numerical_evidence":
                    " | ".join(
                        check["trial_evidence"]
                        for check in numerical_conflicts
                    ),
            })

        print(
            f"Topic {int(topic_id):02d}: "
            f"{len(results)} pairs screened"
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    with OUTPUT_PATH.open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)

    excluded = [
        row for row in rows
        if row["trec_label"] == "excluded"
    ]

    eligible = [
        row for row in rows
        if row["trec_label"] == "eligible"
    ]

    excluded_conflicts = [
        row for row in excluded
        if row["screen_status"] == "CONFLICT"
    ]

    eligible_conflicts = [
        row for row in eligible
        if row["screen_status"] == "CONFLICT"
    ]

    print("\n" + "=" * 68)
    print("AGGREGATE SCREENING RESULTS")
    print("=" * 68)

    print(f"Total patient-trial pairs: {total}")

    print("\nScreening status distribution:")
    for status in (
        "CONFLICT",
        "NO_DETECTED_CONFLICT",
        "INSUFFICIENT_EVIDENCE",
    ):
        count = status_counts[status]
        print(
            f"  {status}: "
            f"{count} ({count / total:.2%})"
        )

    print("\nTREC judgment distribution:")
    for label in (
        "eligible",
        "excluded",
        "non_relevant",
        "unjudged",
    ):
        print(f"  {label}: {label_counts[label]}")

    judged_count = total - label_counts["unjudged"]

    print(
        "\nJudgment coverage: "
        f"{format_rate(safe_rate(judged_count, total))}"
    )

    print(
        "Overall conflict detection rate: "
        f"{format_rate(safe_rate(status_counts['CONFLICT'], total))}"
    )

    print(
        "\nExcluded-label conflict coverage: "
        f"{len(excluded_conflicts)}/{len(excluded)} "
        f"({format_rate(safe_rate(len(excluded_conflicts), len(excluded)))})"
    )

    print(
        "Eligible-label conflict flags: "
        f"{len(eligible_conflicts)}/{len(eligible)} "
        f"({format_rate(safe_rate(len(eligible_conflicts), len(eligible)))})"
    )

    print("\nEligible-labeled pairs flagged as conflicting:")
    if not eligible_conflicts:
        print("  None in this evaluation")
    else:
        for row in eligible_conflicts[:10]:
            print(
                f"  Topic {row['topic_id']} | "
                f"{row['nct_id']} | "
                f"Rank {row['rank']}"
            )

    print(f"\nSaved detailed results: {OUTPUT_PATH}")

    print("\nInterpretation:")
    print("- Conflict means a supported rule detected a mismatch.")
    print("- A non-conflict is NOT an eligibility prediction.")
    print("- TREC judgments are external reference labels.")
    print("- These results use retrieved Part 1 pairs only.")
    print("- Manual review is needed for conflicting")
    print("  results with eligible TREC judgments.")

    print("\nSCREENING EVALUATION COMPLETE")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k must be positive")

    evaluate(args.top_k)


if __name__ == "__main__":
    main()
