
import json

import pytest

from src.data.build_trial_offset_index import (
    build_index,
)
from src.data.trial_store import (
    load_trials_by_id,
)


def create_sample_corpus(tmp_path):
    trials_path = tmp_path / "trials.jsonl"
    index_path = tmp_path / "offsets.db"

    records = [
        {
            "nct_id": "NCT00000001",
            "title": "Test trial A",
        },
        {
            "nct_id": "NCT00000002",
            "title": "Unicode β-hCG study",
        },
        {
            "nct_id": "NCT00000003",
            "title": "Test trial C",
        },
    ]

    with trials_path.open(
        "w", encoding="utf-8"
    ) as file:
        for record in records:
            file.write(
                json.dumps(record, ensure_ascii=False)
                + "\n"
            )

    return trials_path, index_path


def test_build_index_and_retrieve(tmp_path):
    trials_path, index_path = create_sample_corpus(
        tmp_path
    )

    assert build_index(trials_path, index_path) == 3

    result = load_trials_by_id(
        {"NCT00000002", "NCT00000003"},
        trials_path,
        index_path,
    )

    assert len(result) == 2
    assert result["NCT00000002"]["title"] == (
        "Unicode β-hCG study"
    )


def test_empty_request(tmp_path):
    result = load_trials_by_id(
        set(),
        tmp_path / "missing.jsonl",
        tmp_path / "missing.db",
    )

    assert result == {}


def test_missing_trial_id(tmp_path):
    trials_path, index_path = create_sample_corpus(
        tmp_path
    )
    build_index(trials_path, index_path)

    with pytest.raises(KeyError):
        load_trials_by_id(
            {"NCT99999999"},
            trials_path,
            index_path,
        )


def test_stale_index_detected(tmp_path):
    trials_path, index_path = create_sample_corpus(
        tmp_path
    )
    build_index(trials_path, index_path)

    with trials_path.open(
        "a", encoding="utf-8"
    ) as file:
        file.write(
            '{"nct_id": "NCT00000004"}\n'
        )

    with pytest.raises(RuntimeError):
        load_trials_by_id(
            {"NCT00000001"},
            trials_path,
            index_path,
        )


def test_rebuild_index(tmp_path):
    trials_path, index_path = create_sample_corpus(
        tmp_path
    )

    build_index(trials_path, index_path)
    assert build_index(trials_path, index_path) == 3

    result = load_trials_by_id(
        {"NCT00000001"},
        trials_path,
        index_path,
    )

    assert result["NCT00000001"]["title"] == (
        "Test trial A"
    )
