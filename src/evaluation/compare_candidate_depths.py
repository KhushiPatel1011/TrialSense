
import argparse
import math
import statistics
from pathlib import Path

from src.evaluation.statistical_analysis import (
    METRICS,
    load_paired_results,
    paired_bootstrap,
)


ROOT = Path(__file__).resolve().parents[2]

DEFAULT_K30 = (
    ROOT / "results" / "bm25_v4_reranking_k30.csv"
)

DEFAULT_K50 = (
    ROOT / "results" / "bm25_v4_reranking_k50.csv"
)


def compare_depth_results(
    path_k30,
    path_k50,
    iterations=5000,
    seed=42,
):
    """
    Compare V4-K30 and V4-K50 using paired topics.

    The comparison is K50 minus K30.
    """

    rows_k30 = load_paired_results(path_k30)
    rows_k50 = load_paired_results(path_k50)

    by_topic_k30 = {
        row["topic_id"]: row
        for row in rows_k30
    }

    by_topic_k50 = {
        row["topic_id"]: row
        for row in rows_k50
    }

    if set(by_topic_k30) != set(by_topic_k50):
        raise ValueError(
            "K30 and K50 must contain "
            "the same patient topics"
        )

    topic_ids = sorted(
        by_topic_k30,
        key=lambda value: int(value),
    )

    results = {}

    for label, suffix in METRICS.items():

        # Confirm that both experiments use
        # the same original BM25 baseline.
        for topic_id in topic_ids:

            baseline_30 = (
                by_topic_k30[topic_id]["v1_" + suffix]
            )

            baseline_50 = (
                by_topic_k50[topic_id]["v1_" + suffix]
            )

            if not math.isclose(
                baseline_30,
                baseline_50,
                rel_tol=0,
                abs_tol=1e-10,
            ):
                raise ValueError(
                    "Baseline mismatch for "
                    f"topic {topic_id}, metric {suffix}"
                )

        k30_values = [
            by_topic_k30[topic_id]["v4_" + suffix]
            for topic_id in topic_ids
        ]

        k50_values = [
            by_topic_k50[topic_id]["v4_" + suffix]
            for topic_id in topic_ids
        ]

        bootstrap = paired_bootstrap(
            k30_values,
            k50_values,
            iterations=iterations,
            seed=seed,
        )

        differences = [
            v50 - v30
            for v30, v50 in zip(
                k30_values,
                k50_values,
            )
        ]

        results[label] = {
            "k30_mean": statistics.mean(k30_values),
            "k50_mean": statistics.mean(k50_values),
            "delta": bootstrap["observed_difference"],
            "ci_lower": bootstrap["ci_lower"],
            "ci_upper": bootstrap["ci_upper"],
            "improved": sum(
                difference > 1e-10
                for difference in differences
            ),
            "worsened": sum(
                difference < -1e-10
                for difference in differences
            ),
            "unchanged": sum(
                abs(difference) <= 1e-10
                for difference in differences
            ),
        }

    return results


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare TrialSense V4-K30 "
            "against V4-K50"
        )
    )

    parser.add_argument(
        "--k30",
        type=Path,
        default=DEFAULT_K30,
    )

    parser.add_argument(
        "--k50",
        type=Path,
        default=DEFAULT_K50,
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=5000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    results = compare_depth_results(
        args.k30,
        args.k50,
        iterations=args.iterations,
        seed=args.seed,
    )

    print("=" * 78)
    print("TRIALSENSE - K30 VS K50 PAIRED COMPARISON")
    print("=" * 78)

    print("\nComparison: K50 minus K30")
    print(f"Bootstrap iterations: {args.iterations}")
    print(f"Random seed: {args.seed}")

    print(
        f"\n{'Metric':<17}"
        f"{'K30':>9}"
        f"{'K50':>9}"
        f"{'Delta':>10}"
        f"{'95% bootstrap CI':>27}"
    )

    print("-" * 78)

    for label, data in results.items():

        interval = (
            f"[{data['ci_lower']:+.4f}, "
            f"{data['ci_upper']:+.4f}]"
        )

        print(
            f"{label:<17}"
            f"{data['k30_mean']:>9.4f}"
            f"{data['k50_mean']:>9.4f}"
            f"{data['delta']:>+10.4f}"
            f"{interval:>27}"
        )

    print("\nTopic-level changes:")

    for label, data in results.items():

        print(
            f"  {label}: "
            f"improved={data['improved']}, "
            f"worsened={data['worsened']}, "
            f"unchanged={data['unchanged']}"
        )

    print("\nInterpretation:")

    print(
        "- Positive differences favor K50."
    )

    print(
        "- Intervals crossing zero do not "
        "establish a reliable direction of gain."
    )

    print(
        "- Intervals are exploratory because "
        "these topics were reused in development."
    )

    print(
        "- The comparison does not account "
        "for the increased retrieval depth or "
        "resource usage."
    )

    print(
        "- These are retrieval metrics, "
        "not clinical eligibility accuracy."
    )

    print("\nCANDIDATE-DEPTH COMPARISON COMPLETE")


if __name__ == "__main__":
    main()
