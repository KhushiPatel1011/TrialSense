
import argparse
import re

from src.reasoning.clinical_evidence import (
    extract_testosterone_measurements,
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


OPERATOR = r"(?P<operator><=|>=|<|>|≤|≥)"
NUMBER = r"(?P<value>\d+(?:\.\d+)?)"
UNIT = r"(?P<unit>ng\s*/\s*d[lL])"

PRIMARY_PATTERN = re.compile(
    r"\b(?:total\s+)?(?:serum\s+)?testosterone"
    r"(?:\s+level)?"
    r"(?:\s*\(TT\))?"
    r"\s*" + OPERATOR +
    r"\s*" + NUMBER +
    r"\s*" + UNIT + r"\b",
    flags=re.IGNORECASE,
)

# Recognize a second comparison that inherits
# the biomarker from the preceding requirement.
CONTINUATION_PATTERN = re.compile(
    r"\s*(?:and|,)\s*"
    r"(?:testosterone\s*)?"
    + OPERATOR +
    r"\s*" + NUMBER +
    r"\s*" + UNIT + r"\b",
    flags=re.IGNORECASE,
)


def normalize_operator(operator):
    return {
        "≤": "<=",
        "≥": ">=",
    }.get(operator, operator)


def get_inclusion_text(criteria):
    """
    Restrict matching to the inclusion section when
    explicitly labeled sections are present.

    This does not implement exclusion-clause reasoning.
    """
    inclusion = re.search(
        r"\bINCLUSION\s+CRITERIA\b\s*:?",
        criteria,
        flags=re.IGNORECASE,
    )

    exclusion = re.search(
        r"\bEXCLUSION\s+CRITERIA\b\s*:?",
        criteria,
        flags=re.IGNORECASE,
    )

    if inclusion:
        start = inclusion.end()
        end = (
            exclusion.start()
            if exclusion and exclusion.start() > start
            else len(criteria)
        )
        return criteria[start:end]

    if exclusion:
        # Without a labeled inclusion section, do not
        # interpret an exclusion-only threshold as
        # an inclusion requirement.
        return criteria[:exclusion.start()]

    return criteria


def make_threshold(match):
    return {
        "biomarker": "testosterone",
        "operator": normalize_operator(
            match.group("operator")
        ),
        "value": float(match.group("value")),
        "unit": "ng/dL",
        "evidence": match.group(0).strip(),
    }


def extract_testosterone_thresholds(criteria):
    """
    Extract supported inclusion thresholds.

    Supports standalone comparisons and connected
    AND comparisons with repeated or inherited
    biomarker names.

    OR alternatives and complex logical structures
    are not considered supported by this extractor.
    """
    inclusion_text = get_inclusion_text(criteria)
    thresholds = []
    consumed_until = -1

    for match in PRIMARY_PATTERN.finditer(inclusion_text):
        if match.start() < consumed_until:
            continue

        thresholds.append(make_threshold(match))
        cursor = match.end()

        while True:
            continuation = CONTINUATION_PATTERN.match(
                inclusion_text, cursor
            )

            if continuation is None:
                break

            thresholds.append(
                make_threshold(continuation)
            )

            cursor = continuation.end()

        consumed_until = cursor

    return thresholds


def compare_values(patient_value, operator, threshold):
    comparisons = {
        "<": patient_value < threshold,
        "<=": patient_value <= threshold,
        ">": patient_value > threshold,
        ">=": patient_value >= threshold,
    }

    if operator not in comparisons:
        raise ValueError(
            f"Unsupported operator: {operator}"
        )

    return comparisons[operator]


def evaluate_numerical_criteria(patient_text, trial_criteria):
    """
    Compare a single explicit testosterone measurement
    against supported inclusion thresholds.

    A SATISFIED result applies only to extracted
    comparisons, never to overall trial eligibility.
    """
    measurements = extract_testosterone_measurements(
        patient_text
    )

    inclusion_text = get_inclusion_text(
        trial_criteria
    )

    thresholds = extract_testosterone_thresholds(
        trial_criteria
    )

    # Fail conservatively on OR-style alternatives
    # involving supported testosterone comparisons.
    if thresholds and re.search(
        r"\b(?:or|and/or)\s*"
        r"(?:testosterone\s*)?"
        r"(?:<=|>=|<|>|≤|≥)\s*\d",
        inclusion_text,
        flags=re.IGNORECASE,
    ):
        return {
            "status": "UNKNOWN",
            "reason": "Unsupported alternative numerical logic",
            "checks": [],
        }

    if not thresholds:
        return {
            "status": "UNKNOWN",
            "reason": "No supported inclusion threshold found",
            "checks": [],
        }

    if not measurements:
        return {
            "status": "UNKNOWN",
            "reason": "Patient measurement unavailable",
            "checks": [],
        }

    if len(measurements) != 1:
        return {
            "status": "UNKNOWN",
            "reason": (
                "Multiple measurements require "
                "temporal context"
            ),
            "checks": [],
        }

    measurement = measurements[0]
    checks = []

    for threshold in thresholds:
        if measurement["unit"] != threshold["unit"]:
            return {
                "status": "UNKNOWN",
                "reason": "Measurement units do not match",
                "checks": [],
            }

        satisfied = compare_values(
            measurement["value"],
            threshold["operator"],
            threshold["value"],
        )

        checks.append({
            "patient_value": measurement["value"],
            "operator": threshold["operator"],
            "threshold": threshold["value"],
            "unit": threshold["unit"],
            "satisfied": satisfied,
            "patient_evidence": measurement["evidence"],
            "trial_evidence": threshold["evidence"],
        })

    status = (
        "SATISFIED"
        if all(item["satisfied"] for item in checks)
        else "CONFLICT"
    )

    return {
        "status": status,
        "reason": (
            "All extracted numerical comparisons satisfied"
            if status == "SATISFIED"
            else "At least one numerical comparison failed"
        ),
        "checks": checks,
    }


def main():
    parser = argparse.ArgumentParser(
        description="TrialSense numerical eligibility screening"
    )
    parser.add_argument("--topic", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    patient = load_patient_topic(args.topic)
    judgments = load_relevance_judgments(args.topic)

    retrieved, latency = search_bm25(
        patient, top_k=args.top_k
    )

    trials = load_selected_trials(
        {item["nct_id"] for item in retrieved}
    )

    print("=" * 70)
    print("TRIALSENSE - NUMERICAL ELIGIBILITY REASONING V2")
    print("=" * 70)

    print(f"\nPatient topic: {args.topic}")
    print(f"Retrieval latency: {latency * 1000:.2f} ms")

    for rank, item in enumerate(retrieved, 1):
        nct_id = item["nct_id"]
        trial = trials.get(nct_id)

        print("\n" + "-" * 70)
        print(f"RANK {rank} | {nct_id}")
        print("-" * 70)

        if trial is None:
            print("Trial record unavailable")
            continue

        evidence = evaluate_numerical_criteria(
            patient,
            trial["eligibility_criteria"],
        )

        print(f"Title: {trial['title']}")
        print(
            f"TREC label: "
            f"{LABELS[judgments.get(nct_id)]}"
        )
        print(
            f"Numerical screening: {evidence['status']}"
        )
        print(f"Reason: {evidence['reason']}")

        for check in evidence["checks"]:
            label = (
                "SATISFIED"
                if check["satisfied"]
                else "CONFLICT"
            )

            print(f"\n  [{label}]")
            print(
                f"  Patient: {check['patient_value']} "
                f"{check['unit']}"
            )
            print(
                f"  Requirement: {check['operator']} "
                f"{check['threshold']} {check['unit']}"
            )
            print(
                f"  Patient evidence: "
                f"{check['patient_evidence']}"
            )
            print(
                f"  Trial evidence: "
                f"{check['trial_evidence']}"
            )

        print(
            "\nOverall clinical eligibility: NOT DETERMINED"
        )

    print("\n" + "=" * 70)
    print("NUMERICAL SCREENING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
