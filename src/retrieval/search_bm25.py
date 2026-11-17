import argparse
import re
import sqlite3
import time
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

DB_PATH = (
    ROOT / "data" / "processed" / "trials_bm25_part1.db"
)
TOPICS_PATH = ROOT / "data" / "raw" / "topics2022.xml"
QRELS_PATH = ROOT / "data" / "raw" / "qrels2022.txt"

# Small list of common words to avoid filling the search query with uninformative terms
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be",
    "been", "but", "by", "came", "can", "for",
    "from", "had", "has", "have", "he", "her",
    "his", "in", "into", "is", "it", "its",
    "of", "on", "or", "patient", "she", "that",
    "the", "their", "there", "these", "they",
    "this", "to", "was", "were", "with",
    "woman", "man", "who", "which", "than",
    "then", "after", "before", "during",
    "also", "some", "other", "not", "no",
    "about", "according", "showed",
}

MAX_QUERY_TERMS = 60


def load_patient_topic(topic_id):
    """Load an official TREC patient description."""
    root = ET.parse(TOPICS_PATH).getroot()

    for topic in root.findall("topic"):
        if topic.get("number") == str(topic_id):
            return " ".join("".join(topic.itertext()).split())

    raise ValueError(f"Patient topic {topic_id} not found")


def load_relevance_judgments(topic_id):
    """Load judged trial relevance grades for one topic."""
    judgments = {}

    with QRELS_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            query_id, iteration, nct_id, grade = line.split()

            if query_id == str(topic_id):
                judgments[nct_id] = int(grade)

    return judgments


def prepare_fts_query(description):
    """
    Create a safe OR query for SQLite FTS5.

    OR retrieval is used for candidate generation so that
    documents do not need to contain every patient term.
    """
    tokens = re.findall(
        r"[^\W_]+", description.lower(), flags=re.UNICODE
    )

    selected = []
    seen = set()

    for token in tokens:
        if token in STOPWORDS:
            continue

        if len(token) < 2 and not token.isdigit():
            continue

        if token in seen:
            continue

        seen.add(token)
        selected.append(token)

        if len(selected) >= MAX_QUERY_TERMS:
            break

    if not selected:
        raise ValueError("No searchable terms in description")

    # Quoted terms prevent special characters from being
    # interpreted as FTS5 query syntax.
    return " OR ".join(f'"{token}"' for token in selected)


def search_bm25(description, top_k=10):
    """Retrieve the top clinical trials using SQLite BM25."""
    if top_k < 1:
        raise ValueError("top_k must be positive")

    query = prepare_fts_query(description)

    sql = """
        SELECT
            trials.nct_id,
            trials.title,
            trials.conditions,
            bm25(trials_fts) AS bm25_score
        FROM trials_fts
        JOIN trials ON trials.id = trials_fts.rowid
        WHERE trials_fts MATCH ?
        ORDER BY bm25_score ASC
        LIMIT ?
    """

    start = time.perf_counter()

    with sqlite3.connect(DB_PATH) as connection:
        rows = connection.execute(
            sql, (query, top_k)
        ).fetchall()

    elapsed = time.perf_counter() - start

    results = []

    for nct_id, title, conditions, score in rows:
        results.append({
            "nct_id": nct_id,
            "title": title,
            "conditions": conditions,
            "raw_bm25_score": score,
            # SQLite FTS5 returns lower-is-better scores,
            # commonly negative. Negate for display.
            "display_score": -score,
        })

    return results, elapsed


def main():
    parser = argparse.ArgumentParser(
        description="TrialSense BM25 clinical trial retrieval"
    )
    parser.add_argument(
        "--topic", type=int, default=1
    )
    parser.add_argument(
        "--top-k", type=int, default=10
    )
    parser.add_argument(
        "--query", type=str, default=None
    )

    args = parser.parse_args()

    if not DB_PATH.exists():
        raise FileNotFoundError(
            "BM25 database not found. Build the index first."
        )

    description = (
        args.query
        if args.query is not None
        else load_patient_topic(args.topic)
    )

    judgments = (
        load_relevance_judgments(args.topic)
        if args.query is None
        else {}
    )

    print("=" * 65)
    print("TRIALSENSE - BM25 CLINICAL TRIAL RETRIEVAL")
    print("=" * 65)

    print(f"\nPatient description:\n{description}")

    results, elapsed = search_bm25(
        description, top_k=args.top_k
    )

    print(f"\nRetrieved trials: {len(results)}")
    print(f"Search latency: {elapsed * 1000:.2f} ms")

    label_names = {
        0: "Non-relevant",
        1: "Excluded",
        2: "Eligible",
    }

    for rank, trial in enumerate(results, start=1):
        print("\n" + "-" * 65)
        print(f"RANK {rank}")
        print("-" * 65)
        print(f"NCT ID: {trial['nct_id']}")
        print(f"Title: {trial['title']}")
        print(f"Conditions: {trial['conditions']}")
        print(f"BM25 ranking score: {trial['display_score']:.4f}")

        if args.query is None:
            grade = judgments.get(trial["nct_id"])

            if grade is None:
                print("TREC judgment: Unjudged")
            else:
                print(
                    "TREC judgment: "
                    f"{grade} ({label_names[grade]})"
                )

    print("\n" + "=" * 65)
    print("RETRIEVAL COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    main()
