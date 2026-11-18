
import argparse
import csv
import math
import random
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

DEFAULT_RESULTS = (
    ROOT / "results" / "bm25_v4_reranking.csv"
)

METRICS = {
    "Precision@10": "precision_at_10",
    "Recall@10": "recall_at_10",
    "nDCG@10": "ndcg_at_10",
    "MRR@10": "rr_at_10",
}


def load_paired_results(path):
    """
    Load paired V1 and V4 topic-level scores.

    Each row must represent one unique patient topic.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        required = {"topic_id"}

        for suffix in METRICS.values():
            required.add("v1_" + suffix)
            required.add("v4_" + suffix)

        missing = required - set(
            reader.fieldnames or []
        )

        if missing:
            raise ValueError(
                f"Missing CSV columns: {sorted(missing)}"
            )

        rows = list(reader)

    if not rows:
        raise ValueError("Evaluation CSV is empty")

    topic_ids = [row["topic_id"] for row in rows]

    if (
        any(not topic_id for topic_id in topic_ids)
        or len(topic_ids) != len(set(topic_ids))
    ):
        raise ValueError(
            "Topic IDs must be present and unique"
        )

    for row in rows:
        for suffix in METRICS.values():
            for prefix in ("v1_", "v4_"):
                key = prefix + suffix

                value = float(row[key])

                if not math.isfinite(value):
                    raise ValueError(
                        f"Non-finite value in {key}"
                    )

                row[key] = value

    return rows


def percentile(sorted_values, probability):
    """Calculate a linearly interpolated percentile."""
    if not sorted_values:
        raise ValueError("No values provided")

    position = (
        (len(sorted_values) - 1) * probability
    )

    lower = math.floor(position)
    upper = math.ceil(position)

    weight = position - lower

    return (
        sorted_values[lower] * (1 - weight)
        + sorted_values[upper] * weight
    )


def paired_bootstrap(
    baseline,
    proposed,
    iterations=5000,
    seed=42,
):
    """
    Bootstrap the paired mean difference V4 - V1.

    Resample matched topic-level differences.
    """
    if len(baseline) != len(proposed):
        raise ValueError(
            "Paired score arrays must have equal length"
        )

    if not baseline:
        raise ValueError("No paired scores provided")

    if iterations < 1:
        raise ValueError(
            "Iterations must be positive"
        )

    differences = [
        v4 - v1
        for v1, v4 in zip(baseline, proposed)
    ]

    observed = statistics.mean(differences)

    rng = random.Random(seed)
    sample_count = len(differences)

    bootstrapped = []

    for _ in range(iterations):
        sample_mean = sum(
            differences[rng.randrange(sample_count)]
            for _ in range(sample_count)
        ) / sample_count

        bootstrapped.append(sample_mean)

    bootstrapped.sort()

    return {
        "observed_difference": observed,
        "ci_lower": percentile(
            bootstrapped, 0.025
        ),
        "ci_upper": percentile(
            bootstrapped, 0.975
        ),
        "iterations": iterations,
        "topics": sample_count,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Paired bootstrap analysis of "
            "TrialSense V1 versus V4"
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_RESULTS,
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

    rows = load_paired_results(args.input)

    print("=" * 72)
    print("TRIALSENSE - PAIRED RETRIEVAL STATISTICAL ANALYSIS")
    print("=" * 72)

    print(f"\nPatient topics: {len(rows)}")
    print(f"Bootstrap iterations: {args.iterations}")
    print(f"Random seed: {args.seed}")

    print("\nMean V4 - V1 differences:")
    print(
        f"{'Metric':<17}"
        f"{'V1':>9}"
        f"{'V4':>9}"
        f"{'Delta':>10}"
        f"{'95% bootstrap CI':>25}"
    )

    print("-" * 72)

    for label, suffix in METRICS.items():
        baseline = [
            row["v1_" + suffix]
            for row in rows
        ]

        proposed = [
            row["v4_" + suffix]
            for row in rows
        ]

        stats = paired_bootstrap(
            baseline,
            proposed,
            iterations=args.iterations,
            seed=args.seed,
        )

        baseline_mean = statistics.mean(baseline)
        proposed_mean = statistics.mean(proposed)

        interval = (
            f"[{stats['ci_lower']:+.4f}, "
            f"{stats['ci_upper']:+.4f}]"
        )

        print(
            f"{label:<17}"
            f"{baseline_mean:>9.4f}"
            f"{proposed_mean:>9.4f}"
            f"{stats['observed_difference']:>+10.4f}"
            f"{interval:>25}"
        )

    print("\nInterpretation:")
    print(
        "- Intervals describe uncertainty under "
        "topic-level resampling."
    )
    print(
        "- These are exploratory bootstrap "
        "intervals, not held-out results."
    )
    print(
        "- A positive observed difference favors V4."
    )
    print(
        "- Intervals containing zero indicate that "
        "the sign of the gain is uncertain."
    )
    print(
        "- Topic resampling does not correct "
        "development-set reuse or label limitations."
    )
    print(
        "- Results apply only to this partial "
        "corpus and evaluation setup."
    )

    print("\nSTATISTICAL ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()
