import csv
import math
import statistics
import time
from pathlib import Path

from src.evaluation.evaluate_bm25 import (
    load_all_qrels,
    load_indexed_ids,
    dcg,
)
from src.retrieval.search_bm25 import (
    load_patient_topic,
    search_bm25,
)
from src.retrieval.search_bm25_v2 import search_bm25_v2


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"

BASELINE_FILE = RESULTS_DIR / "bm25_baseline_v1.csv"
V2_FILE = RESULTS_DIR / "bm25_v2_evaluation.csv"
OUTPUT_FILE = RESULTS_DIR / "bm25_rrf_v3.csv"

TOP_K = 10
CANDIDATE_K = 100
RRF_K = 60


def reciprocal_rank_fusion(rankings, top_k=TOP_K):
    """Combine ranked lists using reciprocal rank fusion."""

    scores = {}
    documents = {}

    for ranking in rankings:
        for rank, trial in enumerate(ranking, start=1):
            nct_id = trial["nct_id"]

            scores[nct_id] = (
                scores.get(nct_id, 0.0)
                + 1.0 / (RRF_K + rank)
            )

            documents[nct_id] = trial

    ranked_ids = sorted(
        scores,
        key=lambda nct_id: (
            -scores[nct_id],
            nct_id,
        ),
    )

    results = []

    for nct_id in ranked_ids[:top_k]:
        trial = documents[nct_id].copy()
        trial["rrf_score"] = scores[nct_id]
        results.append(trial)

    return results


def retrieve_fusion(description):
    """Retrieve candidates from both methods and fuse."""

    start = time.perf_counter()

    v1_results, _ = search_bm25(
        description,
        top_k=CANDIDATE_K,
    )

    v2_results, _, _ = search_bm25_v2(
        description,
        top_k=CANDIDATE_K,
    )

    fused = reciprocal_rank_fusion(
        [v1_results, v2_results]
    )

    latency = (time.perf_counter() - start) * 1000

    return fused, latency


def evaluate_topic(topic_id, qrels, indexed_ids):
    """Evaluate V3 against known eligible judgments."""

    description = load_patient_topic(topic_id)

    results, latency = retrieve_fusion(description)

    judgments = qrels.get(str(topic_id), {})

    eligible_ids = {
        nct_id
        for nct_id, grade in judgments.items()
        if grade == 2 and nct_id in indexed_ids
    }

    grades = [
        judgments.get(item["nct_id"])
        for item in results
    ]

    relevance = [
        1 if grade == 2 else 0
        for grade in grades
    ]

    eligible_count = sum(relevance)

    precision = eligible_count / TOP_K

    recall = (
        eligible_count / len(eligible_ids)
        if eligible_ids
        else None
    )

    ideal_relevance = (
        [1] * min(len(eligible_ids), TOP_K)
    )

    ideal_dcg = dcg(ideal_relevance)

    ndcg = (
        dcg(relevance) / ideal_dcg
        if ideal_dcg > 0
        else None
    )

    reciprocal_rank = next(
        (
            1.0 / rank
            for rank, grade in enumerate(
                relevance, start=1
            )
            if grade == 1
        ),
        0.0,
    )

    return {
        "topic_id": str(topic_id),
        "eligible_at_10": eligible_count,
        "judged_at_10": sum(
            grade is not None for grade in grades
        ),
        "precision_at_10": precision,
        "recall_at_10": recall,
        "ndcg_at_10": ndcg,
        "rr_at_10": reciprocal_rank,
        "latency_ms": latency,
    }


def read_results(path):
    """Load evaluation CSV without changing original files."""

    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(
        "r", encoding="utf-8", newline=""
    ) as file:
        return list(csv.DictReader(file))


def mean_metric(rows, metric):
    values = []

    for row in rows:
        value = row.get(metric)

        if value is not None and value != "":
            values.append(float(value))

    return statistics.mean(values) if values else None


def main():
    print("=" * 65)
    print("TRIALSENSE - V3 RECIPROCAL RANK FUSION")
    print("=" * 65)

    qrels = load_all_qrels()
    indexed_ids = load_indexed_ids()

    baseline = read_results(BASELINE_FILE)
    v2 = read_results(V2_FILE)

    topic_ids = sorted(qrels, key=int)

    if (
        len(baseline) != len(topic_ids)
        or len(v2) != len(topic_ids)
        or {str(int(r["topic_id"])) for r in baseline}
            != set(topic_ids)
        or {str(int(r["topic_id"])) for r in v2}
            != set(topic_ids)
    ):
        raise ValueError(
            "Evaluation topics do not match."
        )

    v3_rows = []

    for topic_id in topic_ids:
        row = evaluate_topic(
            topic_id, qrels, indexed_ids
        )

        v3_rows.append(row)

        print(
            f"Topic {int(topic_id):02d} | "
            f"Eligible@10: {row['eligible_at_10']} | "
            f"Latency: {row['latency_ms']:.1f} ms"
        )

    with OUTPUT_FILE.open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=list(v3_rows[0])
        )
        writer.writeheader()
        writer.writerows(v3_rows)

    print("\n" + "=" * 65)
    print("AGGREGATE RESULTS - 50 PATIENT TOPICS")
    print("=" * 65)

    metrics = [
        ("precision_at_10", "Precision@10"),
        ("recall_at_10", "Recall@10"),
        ("ndcg_at_10", "nDCG@10"),
        ("rr_at_10", "MRR@10"),
        ("latency_ms", "Latency (ms)"),
    ]

    print(
        f"{'Metric':<18}"
        f"{'V1':>12}"
        f"{'V2':>12}"
        f"{'V3 RRF':>12}"
    )

    for key, label in metrics:
        values = [
            mean_metric(rows, key)
            for rows in (baseline, v2, v3_rows)
        ]

        formatted = [
            f"{value:.4f}" if value is not None
            else "N/A"
            for value in values
        ]

        print(
            f"{label:<18}"
            + "".join(f"{v:>12}" for v in formatted)
        )

    baseline_by_topic = {
        str(int(row["topic_id"])): row
        for row in baseline
    }

    improved = 0
    worsened = 0
    unchanged = 0

    for row in v3_rows:
        v1_precision = float(
            baseline_by_topic[row["topic_id"]][
                "precision_at_10"
            ]
        )

        v3_precision = row["precision_at_10"]

        if v3_precision > v1_precision + 1e-9:
            improved += 1
        elif v3_precision < v1_precision - 1e-9:
            worsened += 1
        else:
            unchanged += 1

    print("\nV3 compared with V1, by eligible@10:")
    print(f"Improved topics: {improved}")
    print(f"Worsened topics: {worsened}")
    print(f"Unchanged topics: {unchanged}")

    coverage = (
        sum(r["judged_at_10"] for r in v3_rows)
        / (len(v3_rows) * TOP_K)
    )

    print(f"\nV3 judged coverage: {coverage:.4f}")
    print(f"V3 results saved: {OUTPUT_FILE}")

    print("\nEvaluation limitations:")
    print("- Part 1 corpus only.")
    print("- Binary eligibility: grade 2 is positive.")
    print("- Unjudged results receive no positive credit.")
    print("- V3 latency includes both retrieval calls.")
    print("- No official full-corpus TREC claim.")

    print("\nRRF EVALUATION COMPLETE")


if __name__ == "__main__":
    main()
