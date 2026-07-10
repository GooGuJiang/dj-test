from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np

from autodj.audio_engine import AutoDJEngine
from autodj.models import TrackAnalysis


def analysis(path: str, bpm: float = 120.0) -> TrackAnalysis:
    return TrackAnalysis(
        path=path,
        title=path,
        duration=10.0,
        bpm=bpm,
        beats_per_bar=4,
        beat_times=(0.0, 0.5, 1.0, 1.5),
        beat_numbers=(1, 2, 3, 4),
        downbeat_times=(0.0,),
    )


def test_seek_snaps_to_nearest_beat_and_cancels_transition() -> None:
    engine = AutoDJEngine()
    engine._current = SimpleNamespace(
        total_samples=441000,
        beat_samples=np.asarray([0, 44100, 88200, 132300]),
        analysis=SimpleNamespace(path="a.wav"),
    )
    engine._playing = True
    engine._transitioning = True
    engine._transition_pos = 123
    engine._next_pos = 456

    actual = engine.seek(1.12, snap_to_beat=True)

    assert actual == 1.0
    assert engine._current_pos == 44100
    assert not engine._transitioning
    assert engine._transition_pos == 0
    assert engine._next_pos == 0
    assert engine._seek_fade_remaining > 0


def test_update_playlist_invalidates_changed_next() -> None:
    engine = AutoDJEngine()
    engine._playing = True
    engine._current = SimpleNamespace(analysis=SimpleNamespace(path="a.wav"))
    engine._next = SimpleNamespace(analysis=SimpleNamespace(path="b.wav"))
    engine._next_synced_base = object()
    engine._plan = object()
    engine._playlist = [analysis("a.wav"), analysis("b.wav")]
    engine._index = 0
    engine.service = lambda: None  # type: ignore[method-assign]

    engine.update_playlist_order(
        [analysis("a.wav"), analysis("c.wav"), analysis("b.wav")]
    )

    assert engine._index == 0
    assert engine._playlist[1].path == "c.wav"
    assert engine._next is None
    assert engine._next_synced_base is None
    assert engine._plan is None


def test_preload_pair_prepares_current_and_next(monkeypatch) -> None:
    engine = AutoDJEngine()
    first = analysis("a.wav", 120.0)
    second = analysis("b.wav", 124.0)
    current = SimpleNamespace(original_bpm=120.0)
    incoming = SimpleNamespace(original_bpm=124.0)
    plan = SimpleNamespace(next_end=100)

    def fake_prepare(item, target_bpm=None):
        return current if item.path == "a.wav" else incoming

    monkeypatch.setattr(engine, "prepare_track", fake_prepare)
    monkeypatch.setattr(engine, "_calculate_plan", lambda *args, **kwargs: plan)
    monkeypatch.setattr(engine, "_apply_tempo_restore", lambda track, end: track)
    monkeypatch.setattr(engine, "_render_advanced_transition", lambda p, a, b: p)

    assert engine.preload_pair([first, second], 0)
    deadline = time.time() + 2.0
    while engine._prime_loading and time.time() < deadline:
        time.sleep(0.01)

    assert engine._prime_current is current
    assert engine._prime_next_synced_base is incoming
    assert engine._prime_next is incoming
    assert engine._prime_plan is plan
