from pathlib import Path
from collections import Counter
from zipfile import ZipFile, BadZipFile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]

ZIP_PATH = (
    ROOT
    / "data"
    / "raw"
    / "ClinicalTrials.2021-04-27.part1.zip"
)


def main():
    print("=" * 60)
    print("TRIALSENSE - CLINICAL TRIAL ARCHIVE INSPECTION")
    print("=" * 60)

    if not ZIP_PATH.exists():
        raise FileNotFoundError(f"Archive not found: {ZIP_PATH}")

    with ZipFile(ZIP_PATH, "r") as archive:

        entries = archive.infolist()

        xml_files = [
            entry for entry in entries
            if not entry.is_dir()
            and entry.filename.lower().endswith(".xml")
        ]

        extensions = Counter(
            Path(entry.filename).suffix.lower()
            for entry in entries
            if not entry.is_dir()
        )

        print(f"\nTotal archive entries: {len(entries)}")
        print(f"Total XML files: {len(xml_files)}")
        print(f"File extensions: {dict(extensions)}")

        print("\nFirst five XML files:")
        for entry in xml_files[:5]:
            print(f"  {entry.filename}")

        if not xml_files:
            raise ValueError("No XML files found in archive")

        # InspectIing the 1st actual clinical trial document
        sample = xml_files[0]

        with archive.open(sample) as file:
            tree = ET.parse(file)
            root = tree.getroot()

        print("\nSample trial file:")
        print(sample.filename)

        print("\nRoot XML tag:")
        print(root.tag)

        print("\nTop-level XML fields:")
        for child in list(root)[:25]:
            value = " ".join("".join(child.itertext()).split())
            print(f"  {child.tag}: {value[:120]}")

        print("\n" + "=" * 60)
        print("ARCHIVE INSPECTION COMPLETE")
        print("=" * 60)


if __name__ == "__main__":
    main()
