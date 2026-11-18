
import argparse

from src.evaluation.audit_screening import (
    audit_case,
    load_trials,
)
from src.evaluation.evaluate_bm25 import (
    load_all_qrels,
)
from src.reasoning.combined_screen import (
    combined_eligibility_screen,
)
from src.retrieval.search_bm25 import (
    load_patient_topic,
    search_bm25,
)


def main():
    parser = argparse.ArgumentParser(
        description="Audit V4 disagreements with TREC labels"
    )
    parser.add_argument(
        "--candidate-k", type=int, default=30
    )
    args = parser.parse_args()

    if args.candidate_k < 1:
        parser.error("--candidate-k must be positive")

    qrels = load_all_qrels()
    topic_ids = sorted(qrels, key=int)

    print("=" * 72)
    print("TRIALSENSE - V4 ELIGIBLE-LABEL CONFLICT AUDIT")
    print("=" * 72)

    candidates_by_topic = {}
    required_ids = set()

    # Retrieve the same candidate depth used by V4.
    for topic_id in topic_ids:
        patient = load_patient_topic(topic_id)
        results, _ = search_bm25(
            patient, top_k=args.candidate_k
        )

        candidates_by_topic[topic_id] = (
            patient, results
        )

        required_ids.update(
            item["nct_id"] for item in results
        )

    trials = load_trials(required_ids)

    disagreements = []

    for topic_id in topic_ids:
        patient, candidates = candidates_by_topic[topic_id]

        for rank, item in enumerate(candidates, start=1):
            nct_id = item["nct_id"]

            # Reference labels are used only for audit.
            if qrels[topic_id].get(nct_id) != 2:
                continue

            screening = combined_eligibility_screen(
                patient, trials[nct_id]
            )

            if screening["status"] == "CONFLICT":
                disagreements.append({
                    "topic_id": topic_id,
                    "rank": rank,
                    "nct_id": nct_id,
                    "trec_label": "eligible",
                    "screen_status": "CONFLICT",
                })

    print(f"\nCandidate depth: {args.candidate_k}")
    print(f"Patient topics examined: {len(topic_ids)}")
    print(
        "Eligible-labeled candidate conflicts: "
        f"{len(disagreements)}"
    )

    for row in disagreements:
        audit_case(
            row,
            trials[row["nct_id"]],
        )

    print("\n" + "=" * 72)
    print("V4 AUDIT COMPLETE")
    print("=" * 72)

    print(
        "Disagreements require interpretation; "
        "they are not automatically annotation errors."
    )


if __name__ == "__main__":
    main()
