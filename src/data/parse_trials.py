import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile

from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]

ZIP_PATH = (
    ROOT
    / "data"
    / "raw"
    / "ClinicalTrials.2021-04-27.part1.zip"
)

OUTPUT_PATH = (
    ROOT
    / "data"
    / "processed"
    / "trials_part1.jsonl"
)


def clean_text(text):
    """Normalize whitespace without destroying medical terminology."""
    if not text:
        return ""

    return re.sub(r"\s+", " ", text).strip()


def extract_text(element, path):
    """Extract complete text from a nested XML field."""
    node = element.find(path)

    if node is None:
        return ""

    return clean_text(" ".join(node.itertext()))


def extract_multiple(element, path):
    """Extract text from all matching XML elements."""
    values = []

    for node in element.findall(path):
        value = clean_text(" ".join(node.itertext()))

        if value:
            values.append(value)

    return values


def parse_trial(xml_file):
    """Convert one ClinicalTrials.gov XML record into structured data."""
    root = ET.parse(xml_file).getroot()

    nct_id = extract_text(root, "id_info/nct_id")

    if not nct_id:
        return None

    title = extract_text(root, "brief_title")
    summary = extract_text(root, "brief_summary")
    description = extract_text(root, "detailed_description")

    conditions = extract_multiple(root, "condition")

    eligibility = extract_text(root, "eligibility/criteria/textblock")

    gender = extract_text(root, "eligibility/gender")
    minimum_age = extract_text(root, "eligibility/minimum_age")
    maximum_age = extract_text(root, "eligibility/maximum_age")
    healthy_volunteers = extract_text(
        root, "eligibility/healthy_volunteers"
    )

    status = extract_text(root, "overall_status")
    study_type = extract_text(root, "study_type")

    # Searchable document text used by retrieval models.
    retrieval_text = clean_text(
        " ".join(
            [
                title,
                " ".join(conditions),
                summary,
                description,
                eligibility,
            ]
        )
    )

    return {
        "nct_id": nct_id,
        "title": title,
        "conditions": conditions,
        "brief_summary": summary,
        "detailed_description": description,
        "eligibility_criteria": eligibility,
        "gender": gender,
        "minimum_age": minimum_age,
        "maximum_age": maximum_age,
        "healthy_volunteers": healthy_volunteers,
        "status": status,
        "study_type": study_type,
        "retrieval_text": retrieval_text,
    }


def main():
    print("=" * 60)
    print("TRIALSENSE - CLINICAL TRIAL DATA PARSER")
    print("=" * 60)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    total_xml = 0
    parsed = 0
    skipped = 0

    seen_ids = set()

    with ZipFile(ZIP_PATH, "r") as archive:
        xml_files = [
            entry
            for entry in archive.infolist()
            if not entry.is_dir()
            and entry.filename.lower().endswith(".xml")
        ]

        total_xml = len(xml_files)

        with OUTPUT_PATH.open("w", encoding="utf-8") as output:

            for entry in tqdm(xml_files, desc="Parsing trials"):

                try:
                    with archive.open(entry) as xml_file:
                        trial = parse_trial(xml_file)

                    if trial is None:
                        skipped += 1
                        continue

                    if trial["nct_id"] in seen_ids:
                        skipped += 1
                        continue

                    seen_ids.add(trial["nct_id"])

                    output.write(
                        json.dumps(trial, ensure_ascii=False) + "\n"
                    )

                    parsed += 1

                except ET.ParseError as error:
                    print(f"\nXML parsing error: {entry.filename}")
                    print(error)
                    skipped += 1

    print("\n" + "=" * 60)
    print("PARSING SUMMARY")
    print("=" * 60)

    print(f"Total XML files: {total_xml}")
    print(f"Successfully parsed trials: {parsed}")
    print(f"Skipped trials: {skipped}")
    print(f"Output file: {OUTPUT_PATH}")

    print("\nDATA PARSING COMPLETE")


if __name__ == "__main__":
    main()
