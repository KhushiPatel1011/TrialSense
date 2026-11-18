
import argparse
import csv
import json
import math
import time
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
    load_indexed_ids,
    dcg,
)

ROOT = Path(__file__).resolve().parents[2]

TRIALS_FILE = ROOT / "data/processed/trials_part1.jsonl"
OUTPUT_FILE = ROOT / "results/bm25_v4_reranking.csv"

TOP_K = 10
CANDIDATE_K = 30


def load_documents(nct_ids):
    """Load only required trial documents in one pass."""
    documents = {}

    with TRIALS_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            trial = json.loads(line)
            nct_id = trial["nct_id"]

            if nct_id in nct_ids:
                documents[nct_id] = trial

                if len(documents) == len(nct_ids):
                    break

    missing = nct_ids - set(documents)

    if missing:
        raise RuntimeError(
            f"Missing {len(missing)} trial documents"
        )

    return documents


def rerank_candidates(candidates, screenings):
    """
    Move explicit conflicts below other candidates.

    Preserve original BM25 order within each group.
    Never use TREC reference labels for reranking.
    """
    return sorted(
        candidates,
        key=lambda item: (
            screenings[item["nct_id"]]["status"]
            == "CONFLICT"
        ),
    )


def calculate_metrics(results, judgments, eligible_ids):
    """Evaluate top-10 eligibility-based retrieval."""
    grades = [
        judgments.get(item["nct_id"])
        for item in results[:TOP_K]
    ]

    relevance = [
        int(grade == 2)
        for grade in grades
    ]

    eligible_count = sum(relevance)

    recall = (
        eligible_count / len(eligible_ids)
        if eligible_ids
        else None
    )

    ideal = [1] * min(len(eligible_ids), TOP_K)
    ideal_dcg = dcg(ideal)

    ndcg = (
        dcg(relevance) / ideal_dcg
        if ideal_dcg
        else None
    )

    reciprocal_rank = next(
        (
            1 / rank
            for rank, relevant in enumerate(
                relevance, start=1
            )
            if relevant
        ),
        0.0,
    )

    return {
        "eligible_at_10": eligible_count,
        "precision_at_10": eligible_count / TOP_K,
        "recall_at_10": recall,
        "ndcg_at_10": ndcg,
        "rr_at_10": reciprocal_rank,
        "judged_at_10": sum(
            grade is not None for grade in grades
        ),
    }


def mean_metric(rows, key):
    values = [
        row[key] for row in rows
        if row[key] is not None
    ]
    return sum(values) / len(values) if values else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=CANDIDATE_K,
    )
    args = parser.parse_args()

    if args.candidate_k < TOP_K:
        parser.error("--candidate-k must be at least 10")

    qrels = load_all_qrels()
    indexed_ids = load_indexed_ids()
    topic_ids = sorted(qrels, key=int)

    print("=" * 68)
    print("TRIALSENSE - V4 ELIGIBILITY-AWARE RERANKING")
    print("=" * 68)

    print(f"Patient topics: {len(topic_ids)}")
    print(f"Candidate depth: {args.candidate_k}")
    print(f"Evaluation depth: {TOP_K}")

    # Retrieve the same BM25 candidates for both
    # experimental conditions.
    queries = {}
    needed_ids = set()

    for topic_id in topic_ids:
        patient = load_patient_topic(topic_id)

        candidates, retrieval_time = search_bm25(
            patient,
            top_k=args.candidate_k,
        )

        if len(candidates) < TOP_K:
            raise RuntimeError(
                f"Insufficient candidates for topic {topic_id}"
            )

        queries[topic_id] = {
            "patient": patient,
            "candidates": candidates,
            "retrieval_ms": retrieval_time * 1000,
        }

        needed_ids.update(
            item["nct_id"] for item in candidates
        )

    print(f"Unique candidate documents: {len(needed_ids)}")
    print("Loading candidate evidence...")

    documents = load_documents(needed_ids)

    rows = []
    label_disagreements = []

    for topic_id in topic_ids:
        data = queries[topic_id]
        patient = data["patient"]
        candidates = data["candidates"]
        judgments = qrels[topic_id]

        eligible_ids = {
            nct_id
            for nct_id, grade in judgments.items()
            if grade == 2 and nct_id in indexed_ids
        }

        baseline_top10 = candidates[:TOP_K]

        start = time.perf_counter()

        screenings = {
            item["nct_id"]: combined_eligibility_screen(
                patient,
                documents[item["nct_id"]],
            )
            for item in candidates
        }

        reranked = rerank_candidates(
            candidates, screenings
        )

        screening_ms = (
            time.perf_counter() - start
        ) * 1000

        v1 = calculate_metrics(
            baseline_top10, judgments, eligible_ids
        )

        v4 = calculate_metrics(
            reranked[:TOP_K], judgments, eligible_ids
        )

        for item in candidates:
            nct_id = item["nct_id"]

            if (
                judgments.get(nct_id) == 2
                and screenings[nct_id]["status"]
                == "CONFLICT"
            ):
                label_disagreements.append(
                    (topic_id, nct_id)
                )

        rows.append({
            "topic_id": topic_id,
            "v1_eligible_at_10": v1["eligible_at_10"],
            "v4_eligible_at_10": v4["eligible_at_10"],
            "v1_precision_at_10": v1["precision_at_10"],
            "v4_precision_at_10": v4["precision_at_10"],
            "v1_recall_at_10": v1["recall_at_10"],
            "v4_recall_at_10": v4["recall_at_10"],
            "v1_ndcg_at_10": v1["ndcg_at_10"],
            "v4_ndcg_at_10": v4["ndcg_at_10"],
            "v1_rr_at_10": v1["rr_at_10"],
            "v4_rr_at_10": v4["rr_at_10"],
            "v1_judged_at_10": v1["judged_at_10"],
            "v4_judged_at_10": v4["judged_at_10"],
            "retrieval_ms": data["retrieval_ms"],
            "screening_reranking_ms": screening_ms,
        })

        print(
            f"Topic {int(topic_id):02d} | "
            f"V1 eligible: {v1['eligible_at_10']} | "
            f"V4 eligible: {v4['eligible_at_10']}"
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True, exist_ok=True
    )

    with OUTPUT_FILE.open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=list(rows[0])
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 68)
    print("AGGREGATE V1 VS V4 RESULTS")
    print("=" * 68)

    for name, suffix in [
        ("Precision@10", "precision_at_10"),
        ("Recall@10", "recall_at_10"),
        ("nDCG@10", "ndcg_at_10"),
        ("MRR@10", "rr_at_10"),
    ]:
        v1 = mean_metric(rows, "v1_" + suffix)
        v4 = mean_metric(rows, "v4_" + suffix)

        print(
            f"{name:<17} "
            f"V1={v1:.4f} "
            f"V4={v4:.4f} "
            f"Delta={v4-v1:+.4f}"
        )

    improved = sum(
        row["v4_eligible_at_10"]
        > row["v1_eligible_at_10"]
        for row in rows
    )

    worsened = sum(
        row["v4_eligible_at_10"]
        < row["v1_eligible_at_10"]
        for row in rows
    )

    print("\nEligible@10 changes:")
    print(f"  Improved topics: {improved}")
    print(f"  Worsened topics: {worsened}")
    print(f"  Unchanged topics: {len(rows)-improved-worsened}")

    print(
        "\nV4 judgment coverage: "
        f"{sum(r['v4_judged_at_10'] for r in rows) / (len(rows)*TOP_K):.2%}"
    )

    print(
        "Average BM25 retrieval time: "
        f"{mean_metric(rows, 'retrieval_ms'):.2f} ms"
    )

    print(
        "Average screening + reranking time: "
        f"{mean_metric(rows, 'screening_reranking_ms'):.2f} ms"
    )

    print(
        "Eligible-labeled candidate conflicts: "
        f"{len(label_disagreements)}"
    )

    print(f"\nResults saved to: {OUTPUT_FILE}")
    print("Corpus: Part 1 only")
    print("Evaluation: exploratory, same 50 development topics")
    print("TREC labels were not used for reranking")
    print("V4 is an experiment, not a clinical eligibility predictor")
    print("\nV4 EVALUATION COMPLETE")


if __name__ == "__main__":
    main()
