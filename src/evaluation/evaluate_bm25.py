import csv
import math
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path

from src.retrieval.search_bm25 import (
    DB_PATH,
    QRELS_PATH,
    TOPICS_PATH,
    load_patient_topic,
    search_bm25,
)

ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = ROOT / "results" / "bm25_evaluation.csv"
TOP_K = 10


def load_all_qrels():
    """Read official topic-trial relevance judgments."""
    qrels = defaultdict(dict)

    with QRELS_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            topic_id, _, nct_id, grade = line.split()
            qrels[topic_id][nct_id] = int(grade)

    return qrels


def load_indexed_ids():
    """Load only trial IDs, not the full docs"""
    with sqlite3.connect(DB_PATH) as connection:
        rows = connection.execute(
            "SELECT nct_id FROM trials"
        ).fetchall()

    return {row[0] for row in rows}


def dcg(binary_relevance):
    """Compute binary discounted cumulative gain."""
    return sum(
        relevance / math.log2(rank + 1)
        for rank, relevance in enumerate(
            binary_relevance, start=1
        )
    )


def evaluate_topic(topic_id, qrels, indexed_ids):
    """Evaluate 1 patient query on this Part 1 corpus."""
    description = load_patient_topic(topic_id)

    results, latency = search_bm25(
        description, top_k=TOP_K
    )

    judgments = qrels.get(str(topic_id), {})

    # Only officially eligible trials (grade 2) count as positive for this eligibility-focused evaluation.
    eligible_in_subset = {
        nct_id
        for nct_id, grade in judgments.items()
        if grade == 2 and nct_id in indexed_ids
    }

    grades = [
        judgments.get(trial["nct_id"])
        for trial in results
    ]

    binary_relevance = [
        1 if grade == 2 else 0
        for grade in grades
    ]

    eligible_retrieved = sum(binary_relevance)
    judged_retrieved = sum(
        grade is not None for grade in grades
    )

    precision = eligible_retrieved / TOP_K

    recall = (
        eligible_retrieved / len(eligible_in_subset)
        if eligible_in_subset
        else None
    )

    ideal = [1] * min(
        len(eligible_in_subset), TOP_K
    )

    ideal_dcg = dcg(ideal)

    ndcg = (
        dcg(binary_relevance) / ideal_dcg
        if ideal_dcg > 0
        else None
    )

    reciprocal_rank = next(
        (
            1 / rank
            for rank, relevance in enumerate(
                binary_relevance, start=1
            )
            if relevance == 1
        ),
        0.0,
    )

    return {
        "topic_id": topic_id,
        "retrieved": len(results),
        "eligible_in_subset": len(eligible_in_subset),
        "eligible_at_10": eligible_retrieved,
        "judged_at_10": judged_retrieved,
        "precision_at_10": precision,
        "recall_at_10": recall,
        "ndcg_at_10": ndcg,
        "rr_at_10": reciprocal_rank,
        "latency_ms": latency * 1000,
    }


def mean_defined(rows, key):
    """Average only metrics with defined denominators."""
    values = [
        row[key]
        for row in rows
        if row[key] is not None
    ]
    return statistics.mean(values) if values else None


def main():
    print("=" * 65)
    print("TRIALSENSE - BM25 RETRIEVAL EVALUATION")
    print("=" * 65)

    if not DB_PATH.exists():
        raise FileNotFoundError(DB_PATH)

    if not TOPICS_PATH.exists():
        raise FileNotFoundError(TOPICS_PATH)

    qrels = load_all_qrels()
    indexed_ids = load_indexed_ids()

    print(f"\nIndexed trial documents: {len(indexed_ids)}")
    print(f"Topics with judgments: {len(qrels)}")

    results = []

    for topic_id in sorted(qrels, key=int):
        result = evaluate_topic(
            topic_id, qrels, indexed_ids
        )
        results.append(result)

        print(
            f"Topic {int(topic_id):02d} | "
            f"Eligible@10: {result['eligible_at_10']} | "
            f"Judged@10: {result['judged_at_10']} | "
            f"Latency: {result['latency_ms']:.1f} ms"
        )

    RESULTS_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    with RESULTS_PATH.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=list(results[0].keys())
        )
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 65)
    print("AGGREGATE BM25 RESULTS - PART 1 CORPUS")
    print("=" * 65)

    print(f"Evaluated topics: {len(results)}")

    topics_with_eligible = sum(
        row["eligible_in_subset"] > 0
        for row in results
    )

    print(
        "Topics with eligible trials in subset: "
        f"{topics_with_eligible}"
    )

    for key, label in [
        ("precision_at_10", "Precision@10"),
        ("recall_at_10", "Recall@10"),
        ("ndcg_at_10", "nDCG@10"),
        ("rr_at_10", "MRR@10"),
    ]:
        value = mean_defined(results, key)
        print(
            f"{label}: "
            f"{value:.4f}" if value is not None
            else f"{label}: N/A"
        )

    judged_coverage = (
        sum(row["judged_at_10"] for row in results)
        / sum(row["retrieved"] for row in results)
    )

    print(f"Judged@10 coverage: {judged_coverage:.4f}")

    print(
        "Average query latency: "
        f"{mean_defined(results, 'latency_ms'):.2f} ms"
    )

    print(f"\nResults saved to: {RESULTS_PATH}")

    print("\nNotes:")
    print("- Eligible = official grade 2.")
    print("- Excluded = grade 1, not eligible.")
    print("- Unjudged results receive no positive credit.")
    print("- Recall uses known eligible trials in Part 1.")
    print("- These are NOT official full-corpus scores.")

    print("\nEVALUATION COMPLETE")


if __name__ == "__main__":
    main()
