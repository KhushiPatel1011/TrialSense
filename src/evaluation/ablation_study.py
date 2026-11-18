
import argparse
import csv
from collections import Counter
from pathlib import Path

from src.evaluation.evaluate_bm25 import (
    load_all_qrels,
    load_indexed_ids,
)
from src.evaluation.evaluate_reranking import (
    load_documents,
    calculate_metrics,
    mean_metric,
    rerank_candidates,
)
from src.reasoning.combined_screen import (
    combined_eligibility_screen,
)
from src.retrieval.search_bm25 import (
    load_patient_topic,
    search_bm25,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results" / "ablation_study.csv"

METHODS = (
    "baseline",
    "demographic",
    "numerical",
    "combined",
)

METRICS = (
    ("Precision@10", "precision_at_10"),
    ("Recall@10", "recall_at_10"),
    ("nDCG@10", "ndcg_at_10"),
    ("MRR@10", "rr_at_10"),
)


def screening_status(screen, method):
    """
    Determine whether a candidate has an explicit
    conflict under the selected ablation method.

    No TREC labels are used here.
    """
    if method == "demographic":
        conflict = (
            screen["demographics"]["status"]
            == "CONFLICT"
        )
    elif method == "numerical":
        conflict = (
            screen["numerical"]["status"]
            == "CONFLICT"
        )
    elif method == "combined":
        conflict = (
            screen["status"] == "CONFLICT"
        )
    else:
        raise ValueError(
            f"Unsupported method: {method}"
        )

    return (
        "CONFLICT"
        if conflict
        else "INSUFFICIENT_EVIDENCE"
    )


def rank_for_method(candidates, screens, method):
    """Preserve BM25 order unless conflicts are demoted."""
    if method == "baseline":
        return list(candidates)

    statuses = {
        item["nct_id"]: {
            "status": screening_status(
                screens[item["nct_id"]],
                method,
            )
        }
        for item in candidates
    }

    return rerank_candidates(
        candidates, statuses
    )


def run_study(candidate_k):
    qrels = load_all_qrels()
    indexed_ids = load_indexed_ids()
    topic_ids = sorted(qrels, key=int)

    print("=" * 70)
    print("TRIALSENSE - ELIGIBILITY RERANKING ABLATION STUDY")
    print("=" * 70)

    print(f"\nPatient topics: {len(topic_ids)}")
    print(f"Candidate depth: {candidate_k}")
    print("Evaluation depth: 10")

    queries = {}
    required_ids = set()

    for topic_id in topic_ids:
        patient = load_patient_topic(topic_id)
        candidates, _ = search_bm25(
            patient,
            top_k=candidate_k,
        )

        queries[topic_id] = {
            "patient": patient,
            "candidates": candidates,
        }

        required_ids.update(
            item["nct_id"]
            for item in candidates
        )

    print(
        f"Unique candidate documents: "
        f"{len(required_ids)}"
    )
    print("Loading trial documents once...")

    documents = load_documents(required_ids)

    rows = []
    conflict_counts = Counter()

    for topic_id in topic_ids:
        patient = queries[topic_id]["patient"]
        candidates = queries[topic_id]["candidates"]
        judgments = qrels[topic_id]

        eligible_ids = {
            nct_id
            for nct_id, grade in judgments.items()
            if grade == 2 and nct_id in indexed_ids
        }

        screens = {
            item["nct_id"]: combined_eligibility_screen(
                patient,
                documents[item["nct_id"]],
            )
            for item in candidates
        }

        row = {"topic_id": topic_id}
        topic_counts = {}

        for method in METHODS:
            ranked = rank_for_method(
                candidates,
                screens,
                method,
            )

            metrics = calculate_metrics(
                ranked[:10],
                judgments,
                eligible_ids,
            )

            for _, metric in METRICS:
                row[f"{method}_{metric}"] = (
                    metrics[metric]
                )

            row[f"{method}_eligible_at_10"] = (
                metrics["eligible_at_10"]
            )

            if method != "baseline":
                count = sum(
                    screening_status(
                        screens[item["nct_id"]],
                        method,
                    ) == "CONFLICT"
                    for item in candidates
                )

                conflict_counts[method] += count
                row[f"{method}_candidate_conflicts"] = (
                    count
                )

            topic_counts[method] = (
                metrics["eligible_at_10"]
            )

        rows.append(row)

        print(
            f"Topic {int(topic_id):02d} | "
            f"V1={topic_counts['baseline']} "
            f"Demog={topic_counts['demographic']} "
            f"Numeric={topic_counts['numerical']} "
            f"Combined={topic_counts['combined']}"
        )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0]),
        )

        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 70)
    print("AGGREGATE ABLATION RESULTS")
    print("=" * 70)

    print(
        f"{'Method':<17}"
        f"{'P@10':>10}"
        f"{'Recall@10':>12}"
        f"{'nDCG@10':>12}"
        f"{'MRR@10':>12}"
    )

    for method in METHODS:
        values = [
            mean_metric(
                rows,
                f"{method}_{metric}",
            )
            for _, metric in METRICS
        ]

        print(
            f"{method:<17}"
            + "".join(
                f"{value:>12.4f}"
                if index > 0
                else f"{value:>10.4f}"
                for index, value in enumerate(values)
            )
        )

    print("\nChanges in eligible trials at rank 10:")

    baseline_key = "baseline_eligible_at_10"

    for method in METHODS[1:]:
        method_key = f"{method}_eligible_at_10"

        improved = sum(
            row[method_key] > row[baseline_key]
            for row in rows
        )

        worsened = sum(
            row[method_key] < row[baseline_key]
            for row in rows
        )

        unchanged = (
            len(rows) - improved - worsened
        )

        print(
            f"  {method}: "
            f"improved={improved}, "
            f"worsened={worsened}, "
            f"unchanged={unchanged}"
        )

    print(
        "\nDetected candidate conflicts "
        "(all 50 topics):"
    )

    for method in METHODS[1:]:
        print(
            f"  {method}: "
            f"{conflict_counts[method]}"
        )

    print(f"\nResults saved to: {OUTPUT}")

    print("\nLimitations:")
    print("- Part 1 corpus only.")
    print("- Same 50 development topics.")
    print("- TREC labels used for evaluation only.")
    print("- Unjudged results receive no positive credit.")
    print("- These are retrieval, not clinical accuracy, metrics.")

    print("\nABLATION STUDY COMPLETE")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--candidate-k",
        type=int,
        default=30,
    )

    args = parser.parse_args()

    if args.candidate_k < 10:
        parser.error(
            "--candidate-k must be at least 10"
        )

    run_study(args.candidate_k)


if __name__ == "__main__":
    main()
