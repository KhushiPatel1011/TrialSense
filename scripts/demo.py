import argparse
import csv
import sqlite3
from pathlib import Path

from src.reasoning.combined_screen import (
    combined_eligibility_screen,
    display_screening,
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
    """Read saved evaluation results."""
    if not path.exists():
        return None

    with path.open(
        "r", encoding="utf-8", newline=""
    ) as file:
        return list(csv.DictReader(file))


def mean_metric(rows, key):
    """Calculate the mean of available metric values."""
    values = [
        float(row[key])
        for row in rows
        if row.get(key) not in (None, "")
    ]

    return sum(values) / len(values) if values else None


def show_project_summary():
    """Display corpus statistics and retrieval experiments."""

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

        values = [
            mean_metric(rows, key)
            for key in (
                "precision_at_10",
                "recall_at_10",
                "ndcg_at_10",
                "latency_ms",
            )
        ]

        if any(value is None for value in values):
            print(f"{model_name:<17} Incomplete metrics")
            continue

        precision, recall, ndcg, latency = values

        print(
            f"{model_name:<17}"
            f"{precision:>10.4f}"
            f"{recall:>12.4f}"
            f"{ndcg:>12.4f}"
            f"{latency:>14.2f}"
        )

    print("\nEvaluation limitations:")
    print("- Part 1 corpus only.")
    print("- Grade 2 is the positive eligibility label.")
    print("- Unjudged results receive no positive credit.")
    print("- Results are exploratory subset measurements.")
    print("- Clinical screening accuracy is not yet evaluated.")


def show_patient_results(topic_id, top_k):
    """Run BM25 retrieval and combined evidence screening."""

    patient = load_patient_topic(topic_id)
    judgments = load_relevance_judgments(topic_id)

    results, latency = search_bm25(
        patient,
        top_k=top_k,
    )

    trial_records = load_selected_trials(
        {item["nct_id"] for item in results}
    )

    print("\n" + "=" * 74)
    print(f"PATIENT TOPIC {topic_id}")
    print("=" * 74)

    print(f"\nPatient description:\n{patient}")

    print(
        f"\nBM25 retrieval latency: "
        f"{latency * 1000:.2f} ms"
    )

    counts = {
        "CONFLICT": 0,
        "NO_DETECTED_CONFLICT": 0,
        "INSUFFICIENT_EVIDENCE": 0,
    }

    for rank, item in enumerate(results, start=1):
        nct_id = item["nct_id"]
        trial = trial_records.get(nct_id)

        print("\n" + "-" * 74)
        print(f"RANK {rank} | {nct_id}")
        print("-" * 74)

        if trial is None:
            print("Trial record unavailable")
            continue

        grade = judgments.get(nct_id)

        print(f"Title: {trial['title']}")
        print(
            f"Conditions: "
            f"{', '.join(trial['conditions'])}"
        )

        print(
            f"BM25 score: {item['display_score']:.4f}"
        )

        print(
            f"TREC reference judgment: "
            f"{LABELS[grade]}"
        )

        screening = combined_eligibility_screen(
            patient,
            trial,
        )

        counts[screening["status"]] += 1

        print("\nPatient information:")
        print(f"  Age: {screening['patient_age']}")
        print(f"  Sex: {screening['patient_sex']}")

        print("\nTrial requirements:")
        print(
            f"  Age: {trial['minimum_age']} "
            f"to {trial['maximum_age']}"
        )
        print(f"  Sex: {trial['gender']}")

        print("\nEvidence-based screening:")
        display_screening(screening)

        print("\nOriginal eligibility criteria excerpt:")
        print(
            shorten(
                trial["eligibility_criteria"],
                450,
            )
        )

    print("\n" + "=" * 74)
    print("SCREENING SUMMARY")
    print("=" * 74)

    print(
        f"Explicit conflicts detected: "
        f"{counts['CONFLICT']}"
    )

    print(
        f"No conflicts detected in supported checks: "
        f"{counts['NO_DETECTED_CONFLICT']}"
    )

    print(
        f"Insufficient evidence: "
        f"{counts['INSUFFICIENT_EVIDENCE']}"
    )

    print("\nIMPORTANT:")
    print(
        "Screening checks are not predictions of "
        "overall clinical eligibility."
    )
    print(
        "Official TREC labels are displayed for "
        "reference only."
    )

    print("\nLIVE DEMONSTRATION COMPLETE")


def main():
    parser = argparse.ArgumentParser(
        description="TrialSense unified research demonstration"
    )

    parser.add_argument(
        "--topic",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--summary-only",
        action="store_true",
    )

    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k must be positive")

    if not DB_PATH.exists():
        raise FileNotFoundError(
            "BM25 database missing. Build the index first."
        )

    show_project_summary()

    if not args.summary_only:
        show_patient_results(
            topic_id=args.topic,
            top_k=args.top_k,
        )


if __name__ == "__main__":
    main()
