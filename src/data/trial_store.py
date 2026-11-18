
import json
import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

TRIALS_PATH = (
    ROOT / "data" / "processed" / "trials_part1.jsonl"
)

INDEX_PATH = (
    ROOT / "data" / "processed" / "trials_offset_index.db"
)


def load_trials_by_id(
    required_ids,
    trials_path=TRIALS_PATH,
    index_path=INDEX_PATH,
):
    """
    Retrieve trial records using indexed byte offsets.

    Does not scan or load the entire JSONL corpus.

    Raises an error if the index is stale or a
    requested record is missing.
    """
    required_ids = set(required_ids)

    if not required_ids:
        return {}

    trials_path = Path(trials_path)
    index_path = Path(index_path)

    if not trials_path.exists():
        raise FileNotFoundError(trials_path)

    if not index_path.exists():
        raise FileNotFoundError(
            f"Trial offset index missing: {index_path}. "
            "Build it before using the fast loader."
        )

    source_stat = trials_path.stat()
    records = {}

    with sqlite3.connect(index_path) as connection:
        metadata = dict(
            connection.execute(
                "SELECT key, value FROM metadata"
            ).fetchall()
        )

        if (
            metadata.get("source_size")
            != str(source_stat.st_size)
            or metadata.get("source_mtime_ns")
            != str(source_stat.st_mtime_ns)
        ):
            raise RuntimeError(
                "Trial offset index is stale. "
                "Rebuild it before retrieving records."
            )

        identifiers = sorted(required_ids)

        with trials_path.open("rb") as source:
            # Stay below SQLite's common parameter limit.
            for start in range(0, len(identifiers), 800):
                batch = identifiers[start:start + 800]

                placeholders = ",".join(
                    "?" for _ in batch
                )

                query = (
                    "SELECT nct_id, byte_offset, byte_length "
                    "FROM trial_offsets WHERE nct_id IN ("
                    + placeholders + ")"
                )

                for nct_id, offset, length in (
                    connection.execute(query, batch)
                ):
                    source.seek(offset)
                    raw = source.read(length)
                    trial = json.loads(raw)

                    if trial.get("nct_id") != nct_id:
                        raise RuntimeError(
                            f"Offset index mismatch for {nct_id}"
                        )

                    records[nct_id] = trial

    missing = required_ids - set(records)

    if missing:
        raise KeyError(
            f"Trial IDs missing from index: "
            f"{sorted(missing)[:10]}"
        )

    return records
