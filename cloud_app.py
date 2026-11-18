
"""
TrialSense cloud launcher.

Run locally:
    python -m streamlit run cloud_app.py

Streamlit Community Cloud entrypoint:
    cloud_app.py

Existing local trial files are reused.
Missing files are downloaded from Hugging Face.
"""

import os
import runpy
import sqlite3
from pathlib import Path

import streamlit as st
from huggingface_hub import hf_hub_download

from src.data.build_trial_offset_index import build_index
from src.data.trial_store import INDEX_PATH, TRIALS_PATH
from src.retrieval.search_bm25 import DB_PATH

ROOT = Path(__file__).resolve().parent
REPO_ID = "Kits2410/trialsense-data"

TOPICS_PATH = ROOT / "data" / "raw" / "topics2022.xml"
QRELS_PATH = ROOT / "data" / "raw" / "qrels2022.txt"

REQUIRED_FILES = {
    "topics2022.xml": TOPICS_PATH,
    "qrels2022.txt": QRELS_PATH,
    "trials_bm25_part1.db": DB_PATH,
    "trials_part1.jsonl": TRIALS_PATH,
}


def download_missing_files():
    token = os.environ.get("HF_TOKEN")

    for filename, target in REQUIRED_FILES.items():
        if target.is_file() and target.stat().st_size > 0:
            print(f"[TrialSense] Ready: {filename}")
            continue

        target.parent.mkdir(parents=True, exist_ok=True)

        print(f"[TrialSense] Downloading: {filename}")

        downloaded = hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            repo_type="dataset",
            local_dir=str(target.parent),
            token=token,
        )

        if not target.is_file() or target.stat().st_size == 0:
            raise RuntimeError(
                f"Download did not produce a valid file: "
                f"{filename} ({downloaded})"
            )

        print(f"[TrialSense] Downloaded: {filename}")


def offset_index_is_current():
    if not INDEX_PATH.is_file():
        return False

    if not TRIALS_PATH.is_file():
        return False

    try:
        source = TRIALS_PATH.stat()

        with sqlite3.connect(str(INDEX_PATH)) as connection:
            metadata = dict(
                connection.execute(
                    "SELECT key, value FROM metadata"
                ).fetchall()
            )

        return (
            metadata.get("source_size") == str(source.st_size)
            and metadata.get("source_mtime_ns")
            == str(source.st_mtime_ns)
        )

    except (sqlite3.DatabaseError, OSError, ValueError):
        return False


def prepare_data():
    download_missing_files()

    if offset_index_is_current():
        print("[TrialSense] Offset index is ready.")
        return

    print("[TrialSense] Building trial offset index...")

    count = build_index(
        trials_path=TRIALS_PATH,
        index_path=INDEX_PATH,
    )

    print(
        f"[TrialSense] Indexed {count:,} trial records."
    )


def main():
    try:
        prepare_data()

    except Exception as error:
        st.error("TrialSense data initialization failed.")
        st.exception(error)
        st.info(
            "Check dataset accessibility and the "
            "HF_TOKEN secret if the dataset is private."
        )
        st.stop()

    runpy.run_path(
        str(ROOT / "app.py"),
        run_name="__main__",
    )


if __name__ == "__main__":
    main()
