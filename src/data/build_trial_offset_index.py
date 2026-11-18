
import argparse
import json
import sqlite3
from contextlib import closing
from pathlib import Path

from src.data.trial_store import (
    TRIALS_PATH,
    INDEX_PATH,
)


def build_index(trials_path=TRIALS_PATH, index_path=INDEX_PATH):
    """
    Build a SQLite index of JSONL byte offsets.

    Windows-safe implementation:
    - Explicitly closes SQLite before renaming.
    - Builds into a temporary database.
    - Replaces the final index only after success.
    - Does not modify the source JSONL corpus.
    """

    trials_path = Path(trials_path)
    index_path = Path(index_path)

    if not trials_path.exists():
        raise FileNotFoundError(
            f"Trial corpus not found: {trials_path}"
        )

    index_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = index_path.with_name(
        index_path.name + ".building"
    )

    # The previous failed attempt may have left
    # a temporary file on disk.
    if temporary_path.exists():
        temporary_path.unlink()

    source_stat = trials_path.stat()
    count = 0

    try:
        # closing() explicitly closes the database.
        # This is essential for Windows file renaming.
        with closing(
            sqlite3.connect(str(temporary_path))
        ) as connection:

            # Use the normal rollback-journal mode.
            # No persistent WAL sidecar files are needed.
            connection.execute(
                "PRAGMA journal_mode=DELETE"
            )

            connection.execute(
                """
                CREATE TABLE metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )

            connection.execute(
                """
                CREATE TABLE trial_offsets (
                    nct_id TEXT PRIMARY KEY,
                    byte_offset INTEGER NOT NULL,
                    byte_length INTEGER NOT NULL
                )
                """
            )

            batch = []

            with trials_path.open("rb") as source:

                while True:
                    offset = source.tell()
                    raw_line = source.readline()

                    if not raw_line:
                        break

                    if not raw_line.strip():
                        continue

                    trial = json.loads(raw_line)
                    nct_id = trial["nct_id"]

                    batch.append(
                        (
                            nct_id,
                            offset,
                            len(raw_line),
                        )
                    )

                    count += 1

                    if len(batch) >= 1000:

                        connection.executemany(
                            """
                            INSERT INTO trial_offsets (
                                nct_id,
                                byte_offset,
                                byte_length
                            )
                            VALUES (?, ?, ?)
                            """,
                            batch,
                        )

                        batch.clear()

            if batch:

                connection.executemany(
                    """
                    INSERT INTO trial_offsets (
                        nct_id,
                        byte_offset,
                        byte_length
                    )
                    VALUES (?, ?, ?)
                    """,
                    batch,
                )

            end_stat = trials_path.stat()

            if (
                end_stat.st_size != source_stat.st_size
                or end_stat.st_mtime_ns
                != source_stat.st_mtime_ns
            ):
                raise RuntimeError(
                    "The trial corpus changed "
                    "during index construction."
                )

            connection.executemany(
                """
                INSERT INTO metadata (key, value)
                VALUES (?, ?)
                """,
                [
                    (
                        "source_size",
                        str(source_stat.st_size),
                    ),
                    (
                        "source_mtime_ns",
                        str(source_stat.st_mtime_ns),
                    ),
                    (
                        "record_count",
                        str(count),
                    ),
                ],
            )

            # Finish the transaction before closing.
            connection.commit()

            # Confirm that the number of stored
            # offsets matches the source records.
            indexed_count = connection.execute(
                "SELECT COUNT(*) FROM trial_offsets"
            ).fetchone()[0]

            if indexed_count != count:
                raise RuntimeError(
                    "Index count mismatch: "
                    f"{indexed_count} stored, "
                    f"{count} source records"
                )

        # At this point, closing() has released
        # the SQLite connection and Windows handle.

        temporary_path.replace(index_path)

    finally:
        # Clean up incomplete temporary databases.
        # Do not mask the original exception if
        # Windows is temporarily holding the file.
        if temporary_path.exists():
            try:
                temporary_path.unlink()
            except PermissionError:
                pass

    return count


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build the TrialSense SQLite "
            "document offset index"
        )
    )

    parser.add_argument(
        "--trials",
        type=Path,
        default=TRIALS_PATH,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=INDEX_PATH,
    )

    args = parser.parse_args()

    print("=" * 65)
    print("TRIALSENSE - TRIAL DOCUMENT OFFSET INDEX")
    print("=" * 65)

    print("\nBuilding disk-based offset index...")

    count = build_index(
        trials_path=args.trials,
        index_path=args.output,
    )

    size_mb = (
        args.output.stat().st_size / (1024 ** 2)
    )

    print(f"\nIndexed trial records: {count:,}")
    print(f"Index location: {args.output}")
    print(f"Index size: {size_mb:.2f} MB")

    print("\nTRIAL OFFSET INDEX COMPLETE")


if __name__ == "__main__":
    main()
