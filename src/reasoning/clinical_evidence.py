
import argparse
import json
import re

from src.retrieval.search_bm25 import load_patient_topic


TESTOSTERONE_PATTERN = re.compile(
    r"\b(?:serum\s+|total\s+)?testosterone\b"
    r"(?:\s+(?:level|concentration))?"
    r"(?:\s+(?:of|is|was|at))?"
    r"\s*(?:=|:)?\s*"
    r"(?P<value>\d+(?:\.\d+)?)"
    r"\s*(?P<unit>ng\s*/\s*d[lL])\b",
    flags=re.IGNORECASE,
)


def extract_testosterone_measurements(text):
    """
    Extract explicit testosterone measurements in ng/dL.

    Limitations:
    - Does not infer missing values.
    - Does not convert different measurement units.
    - Does not determine clinical eligibility.
    - Does not distinguish total from free testosterone.
    - Only matches supported explicit text patterns.
    """

    measurements = []

    for match in TESTOSTERONE_PATTERN.finditer(text):
        value = float(match.group("value"))

        measurements.append({
            "biomarker": "testosterone",
            "value": value,
            "unit": "ng/dL",
            "evidence": match.group(0),
            "start": match.start(),
            "end": match.end(),
        })

    return measurements


def extract_clinical_evidence(text):
    """Return structured clinical measurements."""

    return {
        "testosterone": extract_testosterone_measurements(text)
    }


def main():
    parser = argparse.ArgumentParser(
        description="TrialSense clinical evidence extraction"
    )

    parser.add_argument(
        "--topic",
        type=int,
        default=1,
    )

    args = parser.parse_args()

    patient = load_patient_topic(args.topic)

    evidence = extract_clinical_evidence(patient)

    print("=" * 65)
    print("TRIALSENSE - CLINICAL EVIDENCE EXTRACTION")
    print("=" * 65)

    print(f"\nPatient topic: {args.topic}")

    print("\nExtracted clinical evidence:")
    print(json.dumps(evidence, indent=2))

    print("\nImportant limitations:")
    print("- Only supported explicit measurements are extracted.")
    print("- Unsupported units are not converted.")
    print("- No diagnosis or eligibility is inferred.")
    print("- Extraction is not a clinical interpretation.")

    print("\nCLINICAL EVIDENCE EXTRACTION COMPLETE")


if __name__ == "__main__":
    main()
