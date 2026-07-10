from __future__ import annotations

import threading
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
        beat_times=(0.0, 1.0, 2.0, 3.0, 4.0),
        beat_numbers=(1, 2, 3, 4, 1),
        downbeat_times=(0.0, 4.0),
    )


def test_prepare_track_shares_one_inflight_job(monkeypatch) -> None:
    engine = AutoDJEngine()
    item = analysis("shared.wav")
    started = threading.Event()
    release = threading.Event()
    calls = 0
    result = SimpleNamespace(name="prepared")

    def fake_uncached(track, target_bpm=None):
        nonlocal calls
        calls += 1
        started.set()
        assert release.wait(2.0)
        return result

    monkeypatch.setattr(engine, "_prepare_track_uncached", fake_uncached)
    outputs: list[object] = []

    first = threading.Thread(target=lambda: outputs.append(engine.prepare_track(item, 124.0)))
    second = threading.Thread(target=lambda: outputs.append(engine.prepare_track(item, 124.0)))
    first.start()
    assert started.wait(1.0)
    second.start()
    time.sleep(0.05)
    release.set()
    first.join(2.0)
    second.join(2.0)

    assert calls == 1
    assert outputs == [result, result]
    assert not engine._prepare_jobs


def test_duplicate_pair_preload_does_not_start_second_worker(monkeypatch) -> None:
    engine = AutoDJEngine()
    first = analysis("a.wav")
    entered = threading.Event()
    release = threading.Event()

    def fake_prepare(track, target_bpm=None):
        entered.set()
        assert release.wait(2.0)
        return SimpleNamespace(original_bpm=120.0)

    monkeypatch.setattr(engine, "prepare_track", fake_prepare)
    assert engine.preload_pair([first], 0) is True
    assert entered.wait(1.0)
    assert engine.preload_pair([first], 0) is False
    release.set()
    deadline = time.time() + 2.0
    while engine._prime_loading and time.time() < deadline:
        time.sleep(0.01)
    assert not engine._prime_loading


def test_cancelled_prime_stops_before_preparing_next(monkeypatch) -> None:
    engine = AutoDJEngine()
    tracks = [analysis("a.wav"), analysis("b.wav", 124.0)]
    entered = threading.Event()
    release = threading.Event()
    calls: list[str] = []

    def fake_prepare(track, target_bpm=None):
        calls.append(track.path)
        entered.set()
        assert release.wait(2.0)
        return SimpleNamespace(original_bpm=track.bpm)

    monkeypatch.setattr(engine, "prepare_track", fake_prepare)
    assert engine.preload_pair(tracks, 0)
    assert entered.wait(1.0)
    with engine._lock:
        engine._prime_generation += 1
        engine._playing = True
    release.set()
    deadline = time.time() + 2.0
    while engine._prime_loading and time.time() < deadline:
        time.sleep(0.01)

    assert calls == ["a.wav"]


def _current_track() -> SimpleNamespace:
    return SimpleNamespace(
        total_samples=441000,
        beat_samples=np.asarray([0, 44100, 88200, 132300, 176400, 220500]),
        analysis=SimpleNamespace(path="a.wav"),
    )


def test_seek_keeps_prepared_transition_without_replan(monkeypatch) -> None:
    engine = AutoDJEngine()
    engine._current = _current_track()
    engine._next = SimpleNamespace(analysis=SimpleNamespace(path="b.wav"))
    engine._next_synced_base = object()
    plan = SimpleNamespace(
        current_start=176400,
        current_end=264600,
        length=88200,
        next_start=44100,
    )
    engine._plan = plan
    engine._playing = True
    monkeypatch.setattr(engine, "_replan", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("replan")))

    actual = engine.seek(2.1, snap_to_beat=True)

    assert actual == 2.0
    assert engine._plan is plan
    assert engine._next is not None
    assert not engine._next_loading
    assert not engine._transitioning
    assert not engine._skip_plan_after_seek


def test_seek_inside_existing_transition_uses_rendered_offset() -> None:
    engine = AutoDJEngine()
    engine._current = _current_track()
    engine._next = SimpleNamespace(analysis=SimpleNamespace(path="b.wav"))
    engine._plan = SimpleNamespace(
        current_start=88200,
        current_end=176400,
        length=88200,
        next_start=22050,
    )
    engine._playing = True

    actual = engine.seek(3.1, snap_to_beat=True)

    assert actual == 3.0
    assert engine._transitioning
    assert engine._transition_pos == 44100
    assert engine._next_pos == 66150


def test_seek_after_existing_transition_bypasses_plan_without_replan() -> None:
    engine = AutoDJEngine()
    engine._current = _current_track()
    engine._next = SimpleNamespace(analysis=SimpleNamespace(path="b.wav"))
    plan = SimpleNamespace(
        current_start=44100,
        current_end=88200,
        length=44100,
        next_start=0,
    )
    engine._plan = plan
    engine._playing = True

    actual = engine.seek(4.1, snap_to_beat=True)

    assert actual == 4.0
    assert engine._plan is plan
    assert engine._skip_plan_after_seek
    assert not engine._transitioning
