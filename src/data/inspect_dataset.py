from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
TOPICS_FILE = ROOT / "data" / "raw" / "topics2022.xml"
QRELS_FILE = ROOT / "data" / "raw" / "qrels2022.txt"


def main():
    print("=" * 55)
    print("TRIALSENSE - TREC 2022 DATASET INSPECTION")
    print("=" * 55)

    # Load and validate patient topics
    tree = ET.parse(TOPICS_FILE)
    root = tree.getroot()

    topics = {}
    for topic in root.findall("topic"):
        topic_id = topic.get("number")
        description = " ".join(
            "".join(topic.itertext()).split()
        )

        if not topic_id or not description:
            raise ValueError("Invalid or empty topic detected")

        if topic_id in topics:
            raise ValueError(f"Duplicate topic: {topic_id}")

        topics[topic_id] = description

    print(f"\nTotal patient topics: {len(topics)}")

    # Load and validate relevance judgments
    judgments = []
    labels = Counter()
    judged_topics = set()
    unique_trials = set()

    with QRELS_FILE.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            parts = line.split()

            if len(parts) != 4:
                raise ValueError(
                    f"Invalid qrels row at line {line_number}"
                )

            topic_id, iteration, trial_id, grade = parts
            grade = int(grade)

            if grade not in (0, 1, 2):
                raise ValueError(
                    f"Unexpected relevance grade: {grade}"
                )

            judgments.append(
                (topic_id, trial_id, grade)
            )

            labels[grade] += 1
            judged_topics.add(topic_id)
            unique_trials.add(trial_id)

    missing_topics = judged_topics - set(topics)

    print(f"Total relevance judgments: {len(judgments)}")
    print(f"Unique judged trials: {len(unique_trials)}")
    print(f"Topics with judgments: {len(judged_topics)}")
    print(f"Judged topics missing from XML: {len(missing_topics)}")

    print("\nRelevance label distribution:")
    print(f"  Non-relevant (0): {labels[0]}")
    print(f"  Excluded     (1): {labels[1]}")
    print(f"  Eligible     (2): {labels[2]}")

    print("\nExample patient description (Topic 1):")
    print("-" * 55)
    print(topics.get("1", "Topic 1 not found"))

    print("\n" + "=" * 55)
    print("DATASET INSPECTION COMPLETE")
    print("=" * 55)


if __name__ == "__main__":
    main()
