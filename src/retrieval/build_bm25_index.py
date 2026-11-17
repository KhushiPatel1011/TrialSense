import json
import sqlite3
from pathlib import Path

from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    ROOT / "data" / "processed" / "trials_part1.jsonl"
)

DB_PATH = (
    ROOT / "data" / "processed" / "trials_bm25_part1.db"
)

BATCH_SIZE = 500


def create_tables(connection):
    """Create metadata and disk-backed full-text search tables."""

    connection.execute("""
        CREATE TABLE trials (
            id INTEGER PRIMARY KEY,
            nct_id TEXT NOT NULL UNIQUE,
            title TEXT,
            conditions TEXT
        )
    """)

    # Contentless FTS5 avoids storing a second copy of
    # the entire retrieval text inside the database.
    connection.execute("""
        CREATE VIRTUAL TABLE trials_fts USING fts5(
            retrieval_text,
            content='',
            tokenize='unicode61'
        )
    """)


def insert_batch(connection, records):
    """Insert a batch of records in one transaction."""

    with connection:
        connection.executemany(
            """
            INSERT INTO trials (
                id, nct_id, title, conditions
            )
            VALUES (?, ?, ?, ?)
            """,
            [
                (r[0], r[1], r[2], r[3])
                for r in records
            ],
        )

        connection.executemany(
            """
            INSERT INTO trials_fts (
                rowid, retrieval_text
            )
            VALUES (?, ?)
            """,
            [
                (r[0], r[4])
                for r in records
            ],
        )


def main():
    print("=" * 60)
    print("TRIALSENSE - BM25 INDEX BUILDER")
    print("=" * 60)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(INPUT_PATH)

    if DB_PATH.exists():
        raise FileExistsError(
            f"Database already exists: {DB_PATH}\n"
            "Keep the existing index unless you intend "
            "to rebuild it."
        )

    # Check FTS5 support before creating the database.
    if not sqlite3.sqlite_version_info >= (3, 9, 0):
        raise RuntimeError("SQLite version is too old")

    with sqlite3.connect(":memory:") as test_db:
        try:
            test_db.execute(
                "CREATE VIRTUAL TABLE fts_test USING fts5(text)"
            )
        except sqlite3.OperationalError as error:
            raise RuntimeError(
                "SQLite FTS5 is unavailable"
            ) from error

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(DB_PATH)

    try:
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=NORMAL")

        create_tables(connection)
        connection.commit()

        batch = []
        indexed = 0

        with INPUT_PATH.open(
            "r", encoding="utf-8"
        ) as file:

            for line_number, line in enumerate(
                tqdm(file, desc="Indexing clinical trials"),
                start=1,
            ):
                if not line.strip():
                    continue

                trial = json.loads(line)

                record = (
                    line_number,
                    trial["nct_id"],
                    trial.get("title", ""),
                    json.dumps(
                        trial.get("conditions", []),
                        ensure_ascii=False,
                    ),
                    trial.get("retrieval_text", ""),
                )

                batch.append(record)

                if len(batch) >= BATCH_SIZE:
                    insert_batch(connection, batch)
                    indexed += len(batch)
                    batch.clear()

            if batch:
                insert_batch(connection, batch)
                indexed += len(batch)

        metadata_count = connection.execute(
            "SELECT COUNT(*) FROM trials"
        ).fetchone()[0]

        fts_count = connection.execute(
            "SELECT COUNT(*) FROM trials_fts"
        ).fetchone()[0]

        integrity = connection.execute(
            "PRAGMA integrity_check"
        ).fetchone()[0]

        connection.execute(
            "INSERT INTO trials_fts(trials_fts) "
            "VALUES ('integrity-check')"
        )

        if (
            metadata_count != indexed
            or fts_count != indexed
            or integrity != "ok"
        ):
            raise RuntimeError("Index verification failed")

        print("\n" + "=" * 60)
        print("BM25 INDEX SUMMARY")
        print("=" * 60)
        print(f"Indexed trials: {indexed}")
        print(f"Metadata records: {metadata_count}")
        print(f"FTS5 records: {fts_count}")
        print(f"Database integrity: {integrity}")
        print(f"SQLite version: {sqlite3.sqlite_version}")
        print(f"Database location: {DB_PATH}")
        print(f"Database size: {DB_PATH.stat().st_size / (1024**2):.2f} MB")
        print("\nBM25 INDEX BUILD COMPLETE")

    finally:
        connection.close()


if __name__ == "__main__":
    main()
