import argparse
import json
import re
import sqlite3
import time
from collections import Counter

from src.retrieval.search_bm25 import (
    DB_PATH,
    STOPWORDS,
    load_patient_topic,
    load_relevance_judgments,
    search_bm25,
)


# General clinical vocab for heuristic query selection.
# This list is not a medical diagnosis system.
CLINICAL_TERMS = {
    "syndrome", "disease", "disorder", "deficiency",
    "infection", "cancer", "tumor", "carcinoma",
    "diabetes", "hypertension", "hypotension",
    "pregnancy", "pregnant", "gestational",
    "ectopic", "uterus", "uterine", "vaginal",
    "ovarian", "testosterone", "estrogen",
    "hormone", "hormonal", "gnrh", "hcg",
    "fsh", "lh", "hypogonadism",
    "hypogonadotropic", "kallmann",
    "testes", "testicular", "pubic",
    "endometrium", "ultrasound",
    "menstrual", "bleeding", "spotting",
    "glucose", "insulin", "hemoglobin",
    "creatinine", "serum", "biopsy",
    "diagnosis", "diagnosed", "treatment",
    "therapy", "symptoms", "chronic",
    "acute", "severe", "mild",
    "abnormal", "elevated", "decreased",
    "low", "high", "positive", "negative",
    "pain", "fever", "fatigue", "weight",
    "seizure", "stroke", "cardiac",
    "renal", "hepatic", "pulmonary",
    "neurological", "inflammatory",
    "autoimmune", "metastatic",
}

# Morphological cues frequently found in medical terminology.
CLINICAL_SUFFIXES = (
    "itis", "osis", "emia", "pathy",
    "oma", "ectomy", "scopy",
    "genic", "tropic", "plasia",
)

# Additional narrative terms to remove.
NARRATIVE_STOPWORDS = {
    "recently", "reveals", "revealed", "according",
    "statement", "girlfriend", "boyfriend",
    "classmates", "relationship", "satisfaction",
    "hospital", "clinic", "comes", "came",
    "medical", "history", "physical",
    "examination", "normal", "regular",
    "days", "ago", "years", "old",
    "repeat", "repeating", "significant",
    "multiple", "inconsistent", "unable",
    "shows", "shown", "levels", "level",
    "evaluation", "findings",
}

MAX_QUERY_TERMS = 25


def tokenize(description):
    """Tokenize Unicode clinical text."""
    return re.findall(
        r"[^\W_]+",
        description.lower(),
        flags=re.UNICODE,
    )


def score_term(token, position, total_tokens):
    """
    Assign a heuristic importance score.

    Later clinical findings receive a small priority.
    Recognized medical vocabulary receives a larger one.
    """
    score = 1.0

    if token in CLINICAL_TERMS:
        score += 4.0

    if token.endswith(CLINICAL_SUFFIXES):
        score += 2.0

    if len(token) >= 9:
        score += 0.7

    # Many case descriptions place investigation results
    # toward the end. This is only a heuristic.
    if total_tokens and position / total_tokens >= 0.65:
        score += 0.5

    return score


def prepare_clinical_query(description):
    """Build an FTS5 OR query from prioritized terms."""
    tokens = tokenize(description)

    best_scores = {}
    first_positions = {}

    for position, token in enumerate(tokens):
        if token in STOPWORDS:
            continue

        if token in NARRATIVE_STOPWORDS:
            continue

        if len(token) < 3:
            continue

        # Numeric measurements can introduce unrelated matches.
        if token.isdigit():
            continue

        score = score_term(
            token, position, len(tokens)
        )

        best_scores[token] = max(
            best_scores.get(token, 0),
            score,
        )

        if token not in first_positions:
            first_positions[token] = position

    selected = sorted(
        best_scores,
        key=lambda token: (
            -best_scores[token],
            first_positions[token],
        ),
    )[:MAX_QUERY_TERMS]

    if not selected:
        raise ValueError("No usable clinical query terms")

    # FTS5 supports quoted literal terms joined with OR.
    query = " OR ".join(
        f'"{token}"' for token in selected
    )

    return query, selected


def search_bm25_v2(description, top_k=10):
    """Search clinical trials using focused BM25 queries."""
    if top_k < 1:
        raise ValueError("top_k must be positive")

    query, selected_terms = prepare_clinical_query(
        description
    )

    sql = """
        SELECT
            t.nct_id,
            t.title,
            t.conditions,
            bm25(f) AS score
        FROM trials_fts AS f
        JOIN trials AS t
            ON t.id = f.rowid
        WHERE trials_fts MATCH ?
        ORDER BY score ASC
        LIMIT ?
    """

    # FTS5's bm25() auxiliary function requires the
    # registered FTS table name, not an alias.
    sql = """
        SELECT
            t.nct_id,
            t.title,
            t.conditions,
            bm25(trials_fts) AS score
        FROM trials_fts
        JOIN trials AS t
            ON t.id = trials_fts.rowid
        WHERE trials_fts MATCH ?
        ORDER BY score ASC
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
            "conditions": json.loads(conditions),
            "display_score": -score,
        })

    return results, elapsed, selected_terms


def print_results(label, results, judgments, latency):
    print(f"\n{label}")
    print("-" * 65)
    print(f"Latency: {latency * 1000:.2f} ms")

    eligible = 0

    for rank, trial in enumerate(results, start=1):
        grade = judgments.get(trial["nct_id"])

        if grade == 2:
            eligible += 1

        judgment = {
            0: "Non-relevant",
            1: "Excluded",
            2: "Eligible",
            None: "Unjudged",
        }[grade]

        print(
            f"{rank:2d}. {trial['nct_id']} | "
            f"{judgment:12s} | "
            f"{trial['title'][:65]}"
        )

    print(f"Confirmed eligible in top 10: {eligible}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", type=int, default=1)
    args = parser.parse_args()

    description = load_patient_topic(args.topic)
    judgments = load_relevance_judgments(args.topic)

    print("=" * 65)
    print("TRIALSENSE - BM25 V1 VS CLINICAL QUERY V2")
    print("=" * 65)

    print(f"\nPatient topic: {args.topic}")

    baseline_results, baseline_time = search_bm25(
        description, top_k=10
    )

    improved_results, improved_time, terms = (
        search_bm25_v2(description, top_k=10)
    )

    print("\nV2 selected query terms:")
    print(", ".join(terms))

    print_results(
        "BASELINE BM25 V1",
        baseline_results,
        judgments,
        baseline_time,
    )

    print_results(
        "CLINICALLY FOCUSED BM25 V2",
        improved_results,
        judgments,
        improved_time,
    )

    print("\nCOMPARISON COMPLETE")


if __name__ == "__main__":
    main()
