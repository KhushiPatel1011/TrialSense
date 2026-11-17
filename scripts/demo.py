
import argparse
import csv
import json
import sqlite3
from pathlib import Path

from src.reasoning.eligibility_screen import (
    evaluate_demographics,
    extract_patient_age,
    extract_patient_sex,
)
from src.reasoning.show_trial_evidence import (
    LABELS,
    load_selected_trials,
    shorten,
)
from src.retrieval.search_bm25 import (
    DB_PATH,
    load_patient_topic,
    load_relevance_judgments,
    search_bm25,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

MODEL_FILES = {
    "BM25 V1": RESULTS_DIR / "bm25_baseline_v1.csv",
    "Clinical V2": RESULTS_DIR / "bm25_v2_evaluation.csv",
    "Fusion V3": RESULTS_DIR / "bm25_rrf_v3.csv",
}


def read_metrics(path):
    """Read previously measured retrieval metrics."""
    if not path.exists():
        return None

    with path.open(
        "r", encoding="utf-8", newline=""
    ) as file:
        return list(csv.DictReader(file))


def mean_metric(rows, key):
    values = [
        float(row[key])
        for row in rows
        if row.get(key) not in (None, "")
    ]

    return sum(values) / len(values) if values else None


def show_project_summary():
    """Display corpus size and measured model comparisons."""
    print("\n" + "=" * 74)
    print("TRIALSENSE - PROJECT AND EXPERIMENT SUMMARY")
    print("=" * 74)

    with sqlite3.connect(DB_PATH) as connection:
        corpus_size = connection.execute(
            "SELECT COUNT(*) FROM trials"
        ).fetchone()[0]

    print(f"\nIndexed clinical trials: {corpus_size:,}")
    print("Patient topics: 50")
    print("Corpus: TREC Clinical Trials 2022, Part 1")

    print("\nRetrieval evaluation:")
    print(
        f"{'Model':<17}"
        f"{'P@10':>10}"
        f"{'Recall@10':>12}"
        f"{'nDCG@10':>12}"
        f"{'Latency(ms)':>14}"
    )

    for model_name, path in MODEL_FILES.items():
        rows = read_metrics(path)

        if not rows:
            print(f"{model_name:<17} Evaluation unavailable")
            continue

        precision = mean_metric(rows, "precision_at_10")
        recall = mean_metric(rows, "recall_at_10")
        ndcg = mean_metric(rows, "ndcg_at_10")
        latency = mean_metric(rows, "latency_ms")

        print(
            f"{model_name:<17}"
            f"{precision:>10.4f}"
            f"{recall:>12.4f}"
            f"{ndcg:>12.4f}"
            f"{latency:>14.2f}"
        )

    print("\nEvaluation limitations:")
    print("- This demonstration uses Part 1 of the corpus.")
    print("- Only grade 2 counts as confirmed eligible.")
    print("- Unjudged results are not proof of ineligibility.")
    print("- Scores are exploratory, not official TREC results.")


def show_patient_results(topic_id, top_k):
    """Run retrieval and show demographic reasoning evidence."""

    patient = load_patient_topic(topic_id)
    judgments = load_relevance_judgments(topic_id)

    patient_age = extract_patient_age(patient)
    patient_sex = extract_patient_sex(patient)

    results, latency = search_bm25(
        patient, top_k=top_k
    )

    selected_ids = {
        trial["nct_id"] for trial in results
    }

    trial_records = load_selected_trials(selected_ids)

    print("\n" + "=" * 74)
    print(f"PATIENT TOPIC {topic_id}")
    print("=" * 74)

    print(f"\nPatient description:\n{patient}")
    print("\nExtracted patient attributes:")
    print(f"  Age: {patient_age}")
    print(f"  Sex: {patient_sex}")

    print(f"\nBM25 retrieval latency: {latency * 1000:.2f} ms")

    for rank, item in enumerate(results, 1):
        nct_id = item["nct_id"]
        trial = trial_records.get(nct_id)

        print("\n" + "-" * 74)
        print(f"RANK {rank} | {nct_id}")
        print("-" * 74)

        if trial is None:
            print("Trial document unavailable")
            continue

        grade = judgments.get(nct_id)

        evidence = evaluate_demographics(
            patient_age,
            patient_sex,
            trial,
        )

        print(f"Title: {trial['title']}")
        print(f"Conditions: {', '.join(trial['conditions'])}")
        print(f"BM25 score: {item['display_score']:.4f}")
        print(f"TREC reference judgment: {LABELS[grade]}")
        print(f"Demographic screening: {evidence['status']}")

        for check in evidence["checks"]:
            print(f"  [PASS] {check}")

        for conflict in evidence["conflicts"]:
            print(f"  [CONFLICT] {conflict}")

        for unknown in evidence["unknowns"]:
            print(f"  [UNKNOWN] {unknown}")

        print("\nTrial eligibility requirements:")
        print(
            f"  Age: {trial['minimum_age']} "
            f"to {trial['maximum_age']}"
        )
        print(f"  Sex: {trial['gender']}")

        print("\nEligibility criteria excerpt:")
        print(shorten(trial["eligibility_criteria"], 550))

        print("\nFull clinical eligibility: NOT DETERMINED")

    print("\n" + "=" * 74)
    print("LIVE DEMONSTRATION COMPLETE")
    print("=" * 74)


def main():
    parser = argparse.ArgumentParser(
        description="TrialSense professor demonstration"
    )

    parser.add_argument(
        "--topic", type=int, default=1
    )
    parser.add_argument(
        "--top-k", type=int, default=3
    )
    parser.add_argument(
        "--summary-only", action="store_true"
    )

    args = parser.parse_args()

    if not DB_PATH.exists():
        raise FileNotFoundError(
            "TrialSense BM25 database is missing."
        )

    show_project_summary()

    if not args.summary_only:
        show_patient_results(
            topic_id=args.topic,
            top_k=args.top_k,
        )


if __name__ == "__main__":
    main()
