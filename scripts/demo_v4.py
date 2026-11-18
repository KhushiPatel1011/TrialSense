
import argparse
import time

from src.data.trial_store import load_trials_by_id

from src.reasoning.combined_screen import (
    combined_eligibility_screen,
)
from src.reasoning.show_trial_evidence import (
    load_selected_trials,
    LABELS,
)
from src.retrieval.search_bm25 import (
    load_patient_topic,
    load_relevance_judgments,
    search_bm25,
)
from src.evaluation.evaluate_reranking import (
    rerank_candidates,
)


def count_eligible(results, judgments):
    """Count TREC grade-2 trials in a ranked list."""
    return sum(
        judgments.get(item["nct_id"]) == 2
        for item in results
    )


def load_trial_documents(trial_ids, compare_loaders=False):
    """
    Load documents using the SQLite offset index.

    Optionally compare the indexed loader against
    the previous full JSONL scan using identical IDs.
    """
    start = time.perf_counter()

    trials = load_trials_by_id(trial_ids)

    indexed_ms = (
        time.perf_counter() - start
    ) * 1000

    scan_ms = None

    if compare_loaders:
        start = time.perf_counter()

        scanned_trials = load_selected_trials(trial_ids)

        scan_ms = (
            time.perf_counter() - start
        ) * 1000

        if trials != scanned_trials:
            raise RuntimeError(
                "Indexed and scanned trial records differ."
            )

    return trials, indexed_ms, scan_ms


def display_rankings(
    heading,
    candidates,
    trials,
    screenings,
    judgments,
    top_k,
):
    print("\n" + "=" * 78)
    print(heading)
    print("=" * 78)

    print(
        f"{'Rank':<6}"
        f"{'Trial ID':<15}"
        f"{'Screening':<27}"
        f"{'TREC judgment':<17}"
    )
    print("-" * 78)

    for rank, item in enumerate(
        candidates[:top_k],
        start=1,
    ):
        nct_id = item["nct_id"]

        status = screenings[nct_id]["status"]
        reference = LABELS[judgments.get(nct_id)]

        print(
            f"{rank:<6}"
            f"{nct_id:<15}"
            f"{status:<27}"
            f"{reference:<17}"
        )

        print(
            f"      {trials[nct_id]['title'][:70]}"
        )

    eligible = count_eligible(
        candidates[:top_k],
        judgments,
    )

    print(
        f"\nEligible-labeled trials in top {top_k}: "
        f"{eligible}/{top_k}"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "TrialSense V1 versus V4 comparison "
            "with indexed trial loading"
        )
    )

    parser.add_argument(
        "--topic",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--candidate-k",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--compare-loaders",
        action="store_true",
        help=(
            "Benchmark the indexed loader against "
            "the original JSONL scan."
        ),
    )

    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k must be positive")

    if args.candidate_k < args.top_k:
        parser.error(
            "--candidate-k must be >= --top-k"
        )

    patient = load_patient_topic(args.topic)

    judgments = load_relevance_judgments(
        args.topic
    )

    print("=" * 78)
    print("TRIALSENSE - BM25 V1 VS ELIGIBILITY-AWARE V4")
    print("=" * 78)

    print(f"\nPatient topic: {args.topic}")
    print(f"Candidate depth: {args.candidate_k}")
    print(f"Display depth: {args.top_k}")

    print("\nPatient description:")
    print(patient)

    candidates, retrieval_latency = search_bm25(
        patient,
        top_k=args.candidate_k,
    )

    trial_ids = {
        item["nct_id"]
        for item in candidates
    }

    trials, indexed_ms, scan_ms = load_trial_documents(
        trial_ids,
        compare_loaders=args.compare_loaders,
    )

    screening_start = time.perf_counter()

    screenings = {
        item["nct_id"]: combined_eligibility_screen(
            patient,
            trials[item["nct_id"]],
        )
        for item in candidates
    }

    reranked = rerank_candidates(
        candidates,
        screenings,
    )

    screening_ms = (
        time.perf_counter() - screening_start
    ) * 1000

    display_rankings(
        "BASELINE V1 - ORIGINAL BM25 ORDER",
        candidates,
        trials,
        screenings,
        judgments,
        args.top_k,
    )

    display_rankings(
        "V4 - ELIGIBILITY-AWARE RERANKED ORDER",
        reranked,
        trials,
        screenings,
        judgments,
        args.top_k,
    )

    original_ranks = {
        item["nct_id"]: rank
        for rank, item in enumerate(
            candidates,
            start=1,
        )
    }

    print("\n" + "=" * 78)
    print("RANKING CHANGES")
    print("=" * 78)

    changed = 0

    for new_rank, item in enumerate(
        reranked[:args.top_k],
        start=1,
    ):
        nct_id = item["nct_id"]
        old_rank = original_ranks[nct_id]

        if old_rank != new_rank:
            changed += 1

            print(
                f"{nct_id}: "
                f"V1 rank {old_rank} -> "
                f"V4 rank {new_rank}"
            )

    if changed == 0:
        print("No changes among displayed positions.")

    print("\n" + "=" * 78)
    print("COMPARISON SUMMARY")
    print("=" * 78)

    v1_eligible = count_eligible(
        candidates[:args.top_k],
        judgments,
    )

    v4_eligible = count_eligible(
        reranked[:args.top_k],
        judgments,
    )

    conflicts = sum(
        screen["status"] == "CONFLICT"
        for screen in screenings.values()
    )

    print(
        f"V1 eligible-labeled @ {args.top_k}: "
        f"{v1_eligible}"
    )

    print(
        f"V4 eligible-labeled @ {args.top_k}: "
        f"{v4_eligible}"
    )

    print(
        f"Change: {v4_eligible - v1_eligible:+d}"
    )

    print(
        f"Detected conflicts among "
        f"{len(candidates)} candidates: {conflicts}"
    )

    retrieval_ms = retrieval_latency * 1000

    total_pipeline_ms = (
        retrieval_ms + indexed_ms + screening_ms
    )

    print("\nPIPELINE PERFORMANCE")
    print(f"BM25 retrieval: {retrieval_ms:.2f} ms")
    print(f"Indexed document loading: {indexed_ms:.2f} ms")
    print(
        f"Screening + reranking: "
        f"{screening_ms:.2f} ms"
    )
    print(
        f"Combined measured pipeline stages: "
        f"{total_pipeline_ms:.2f} ms"
    )

    if scan_ms is not None:
        print("\nDOCUMENT LOADER BENCHMARK")
        print(f"Documents compared: {len(trial_ids)}")
        print(f"Original JSONL scan: {scan_ms:.2f} ms")
        print(f"SQLite indexed loading: {indexed_ms:.2f} ms")

        if indexed_ms > 0:
            print(
                f"Measured loader speedup: "
                f"{scan_ms / indexed_ms:.2f}x"
            )

        print("Record comparison: IDENTICAL")

    print("\nImportant:")
    print(
        "- TREC labels are displayed only for evaluation."
    )
    print(
        "- Screening does not determine full "
        "clinical eligibility."
    )
    print(
        "- Timing is from one execution, "
        "not a repeated benchmark."
    )
    print(
        "- Combined pipeline timing excludes "
        "console formatting and output."
    )

    print("\nV1 VS V4 COMPARISON COMPLETE")


if __name__ == "__main__":
    main()
