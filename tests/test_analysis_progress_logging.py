from __future__ import annotations

import json

from autodj.songformer_analyzer import PROGRESS_PREFIX, parse_worker_progress


def test_parse_songformer_worker_progress() -> None:
    payload = {
        "current": 2,
        "total": 5,
        "message": "SongFormer 完成：demo.wav",
        "stage": "track_done",
    }
    line = PROGRESS_PREFIX + json.dumps(payload, ensure_ascii=False)
    assert parse_worker_progress(line) == payload


def test_ignore_regular_worker_log() -> None:
    assert parse_worker_progress("Downloading model files...") == {}
    assert parse_worker_progress(PROGRESS_PREFIX + "not-json") == {}
