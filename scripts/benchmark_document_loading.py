
import argparse
import statistics
import time

from src.data.trial_store import load_trials_by_id
from src.reasoning.show_trial_evidence import (
    load_selected_trials,
)
from src.retrieval.search_bm25 import (
    load_patient_topic,
    search_bm25,
)


def measure(function, trial_ids):
    start = time.perf_counter()
    records = function(trial_ids)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return records, elapsed_ms


def main():
    parser = argparse.ArgumentParser(
        description="TrialSense repeated document-loading benchmark"
    )
    parser.add_argument("--topic", type=int, default=1)
    parser.add_argument("--candidate-k", type=int, default=30)
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()

    if args.candidate_k < 1 or args.runs < 1:
        parser.error(
            "--candidate-k and --runs must be positive"
        )

    patient = load_patient_topic(args.topic)
    candidates, _ = search_bm25(
        patient,
        top_k=args.candidate_k,
    )

    trial_ids = {
        item["nct_id"] for item in candidates
    }

    # Verify both methods return identical records.
    indexed_records = load_trials_by_id(trial_ids)
    scanned_records = load_selected_trials(trial_ids)

    if indexed_records != scanned_records:
        raise RuntimeError(
            "The loaders returned different trial records."
        )

    indexed_times = []
    scanned_times = []

    print("=" * 68)
    print("TRIALSENSE - REPEATED DOCUMENT LOADING BENCHMARK")
    print("=" * 68)
    print(f"\nPatient topic: {args.topic}")
    print(f"Documents: {len(trial_ids)}")
    print(f"Runs: {args.runs}")
    print("Record equality: VERIFIED")

    for run in range(args.runs):
        # Alternate execution order to reduce
        # consistent first-run ordering effects.
        if run % 2 == 0:
            methods = [
                ("indexed", load_trials_by_id),
                ("scan", load_selected_trials),
            ]
        else:
            methods = [
                ("scan", load_selected_trials),
                ("indexed", load_trials_by_id),
            ]

        measurements = {}

        for name, function in methods:
            records, elapsed = measure(
                function,
                trial_ids,
            )

            if records != indexed_records:
                raise RuntimeError(
                    f"Record mismatch in run {run + 1}"
                )

            measurements[name] = elapsed

        indexed_times.append(measurements["indexed"])
        scanned_times.append(measurements["scan"])

        print(
            f"Run {run + 1:02d} | "
            f"JSONL scan: {measurements['scan']:.2f} ms | "
            f"SQLite index: {measurements['indexed']:.2f} ms"
        )

    scan_mean = statistics.mean(scanned_times)
    index_mean = statistics.mean(indexed_times)

    scan_median = statistics.median(scanned_times)
    index_median = statistics.median(indexed_times)

    print("\n" + "=" * 68)
    print("AGGREGATE BENCHMARK RESULTS")
    print("=" * 68)

    print(f"Original JSONL mean: {scan_mean:.2f} ms")
    print(f"Original JSONL median: {scan_median:.2f} ms")

    print(f"SQLite indexed mean: {index_mean:.2f} ms")
    print(f"SQLite indexed median: {index_median:.2f} ms")

    print(
        f"\nSpeedup from means: "
        f"{scan_mean / index_mean:.2f}x"
    )
    print(
        f"Speedup from medians: "
        f"{scan_median / index_median:.2f}x"
    )

    print("\nBenchmark notes:")
    print("- Both loaders access identical trial IDs.")
    print("- All returned records are checked for equality.")
    print("- Results are local warm-cache measurements.")
    print("- This benchmark excludes BM25 and screening.")
    print("- Timings may vary with hardware and caching.")

    print("\nBENCHMARK COMPLETE")


if __name__ == "__main__":
    main()
