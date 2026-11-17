
import csv
import statistics
from pathlib import Path

from src.evaluation.evaluate_bm25 import (
    TOP_K,
    dcg,
    load_all_qrels,
    load_indexed_ids,
)
from src.retrieval.search_bm25 import load_patient_topic
from src.retrieval.search_bm25_v2 import search_bm25_v2


ROOT = Path(__file__).resolve().parents[2]

BASELINE_PATH = ROOT / "results" / "bm25_baseline_v1.csv"
V2_PATH = ROOT / "results" / "bm25_v2_evaluation.csv"
COMPARISON_PATH = ROOT / "results" / "bm25_comparison.csv"

METRICS = [
    "precision_at_10",
    "recall_at_10",
    "ndcg_at_10",
    "rr_at_10",
    "latency_ms",
]


def evaluate_v2_topic(topic_id, qrels, indexed_ids):
    """Evaluate clinically focused BM25 on one TREC topic."""

    description = load_patient_topic(topic_id)

    retrieved, latency, terms = search_bm25_v2(
        description,
        top_k=TOP_K,
    )

    judgments = qrels.get(str(topic_id), {})

    eligible_in_subset = {
        nct_id
        for nct_id, grade in judgments.items()
        if grade == 2 and nct_id in indexed_ids
    }

    grades = [
        judgments.get(item["nct_id"])
        for item in retrieved
    ]

    relevance = [
        1 if grade == 2 else 0
        for grade in grades
    ]

    eligible_count = sum(relevance)
    judged_count = sum(
        grade is not None for grade in grades
    )

    precision = eligible_count / TOP_K

    recall = (
        eligible_count / len(eligible_in_subset)
        if eligible_in_subset
        else None
    )

    ideal = [1] * min(
        len(eligible_in_subset), TOP_K
    )
    ideal_dcg = dcg(ideal)

    ndcg = (
        dcg(relevance) / ideal_dcg
        if ideal_dcg > 0
        else None
    )

    reciprocal_rank = next(
        (
            1 / rank
            for rank, value in enumerate(relevance, 1)
            if value == 1
        ),
        0.0,
    )

    return {
        "topic_id": str(topic_id),
        "retrieved": len(retrieved),
        "eligible_in_subset": len(eligible_in_subset),
        "eligible_at_10": eligible_count,
        "judged_at_10": judged_count,
        "precision_at_10": precision,
        "recall_at_10": recall,
        "ndcg_at_10": ndcg,
        "rr_at_10": reciprocal_rank,
        "latency_ms": latency * 1000,
    }


def save_csv(path, rows):
    with path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)


def load_baseline():
    if not BASELINE_PATH.exists():
        raise FileNotFoundError(BASELINE_PATH)

    with BASELINE_PATH.open(
        "r", newline="", encoding="utf-8"
    ) as file:
        rows = list(csv.DictReader(file))

    result = {}

    for row in rows:
        topic_id = str(int(row["topic_id"]))

        if topic_id in result:
            raise ValueError(
                f"Duplicate baseline topic {topic_id}"
            )

        result[topic_id] = {
            key: (
                float(row[key])
                if row[key] not in ("", None)
                else None
            )
            for key in METRICS
        }

    return result


def mean_metric(rows, metric):
    values = [
        row[metric]
        for row in rows
        if row[metric] is not None
    ]
    return statistics.mean(values) if values else None


def main():
    print("=" * 65)
    print("TRIALSENSE - BM25 V1 VS V2 EVALUATION")
    print("=" * 65)

    baseline = load_baseline()
    qrels = load_all_qrels()
    indexed_ids = load_indexed_ids()

    topic_ids = sorted(qrels, key=int)

    if set(baseline) != set(topic_ids):
        raise ValueError(
            "Baseline topics and evaluation topics differ."
        )

    v2_rows = []
    comparison_rows = []

    for topic_id in topic_ids:
        v2 = evaluate_v2_topic(
            topic_id, qrels, indexed_ids
        )

        v1 = baseline[topic_id]

        comparison_rows.append({
            "topic_id": topic_id,
            "v1_eligible_at_10": round(
                v1["precision_at_10"] * TOP_K
            ),
            "v2_eligible_at_10": v2["eligible_at_10"],
            "v1_ndcg_at_10": v1["ndcg_at_10"],
            "v2_ndcg_at_10": v2["ndcg_at_10"],
            "v1_latency_ms": v1["latency_ms"],
            "v2_latency_ms": v2["latency_ms"],
        })

        v2_rows.append(v2)

        print(
            f"Topic {int(topic_id):02d} | "
            f"V1 eligible: "
            f"{round(v1['precision_at_10'] * TOP_K)} | "
            f"V2 eligible: {v2['eligible_at_10']}"
        )

    save_csv(V2_PATH, v2_rows)
    save_csv(COMPARISON_PATH, comparison_rows)

    baseline_rows = list(baseline.values())

    print("\n" + "=" * 65)
    print("AGGREGATE COMPARISON - 50 TOPICS")
    print("=" * 65)

    print(
        f"{'Metric':<20} {'V1':>12} "
        f"{'V2':>12} {'Difference':>12}"
    )

    for metric in METRICS:
        v1_mean = mean_metric(
            baseline_rows, metric
        )
        v2_mean = mean_metric(
            v2_rows, metric
        )

        if v1_mean is None or v2_mean is None:
            print(f"{metric:<20} {'N/A':>12}")
            continue

        difference = v2_mean - v1_mean

        print(
            f"{metric:<20} "
            f"{v1_mean:>12.4f} "
            f"{v2_mean:>12.4f} "
            f"{difference:>+12.4f}"
        )

    improved = sum(
        r["v2_eligible_at_10"] >
        r["v1_eligible_at_10"]
        for r in comparison_rows
    )

    worsened = sum(
        r["v2_eligible_at_10"] <
        r["v1_eligible_at_10"]
        for r in comparison_rows
    )

    unchanged = len(comparison_rows) - improved - worsened

    print("\nPer-topic eligible@10 comparison:")
    print(f"V2 improved: {improved}")
    print(f"V2 worsened: {worsened}")
    print(f"Unchanged: {unchanged}")

    coverage = (
        sum(r["judged_at_10"] for r in v2_rows)
        / sum(r["retrieved"] for r in v2_rows)
    )

    print(f"\nV2 judged coverage: {coverage:.4f}")

    print("\nSaved results:")
    print(V2_PATH)
    print(COMPARISON_PATH)

    print("\nImportant:")
    print("Evaluation uses the Part 1 trial corpus.")
    print("Only grade 2 counts as eligible.")
    print("Unjudged trials receive no positive credit.")
    print("These are preliminary, not official TREC scores.")

    print("\nCOMPARISON COMPLETE")


if __name__ == "__main__":
    main()
