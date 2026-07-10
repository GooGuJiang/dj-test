from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from autodj.audio_engine import AutoDJEngine, EngineConfig
from autodj.models import TrackAnalysis


def _track(name: str, bpm: float = 120.0) -> TrackAnalysis:
    return TrackAnalysis(
        path=f"{name}.wav",
        title=name,
        duration=180.0,
        bpm=bpm,
        beats_per_bar=4,
        beat_times=tuple(i * 0.5 for i in range(360)),
        beat_numbers=tuple(i % 4 + 1 for i in range(360)),
        downbeat_times=tuple(i * 2.0 for i in range(90)),
    )


def test_hot_pair_ready_does_not_wait_for_future_window(monkeypatch) -> None:
    engine = AutoDJEngine(EngineConfig(preload_window_tracks=4, sample_rate=1000))
    tracks = [_track("A"), _track("B"), _track("C")]
    future_calls: list[str] = []

    def fake_prepare(item, target_bpm=None):
        return SimpleNamespace(
            original_bpm=float(item.bpm),
            audio=np.zeros((128, 2), dtype=np.float32),
            analysis=item,
        )

    plan = SimpleNamespace(
        next_end=64,
        current_start=32,
        next_start=0,
        current_end=96,
        bars=4,
        dj_intent="Outro to Intro",
        transition_mode="test",
        score=0.8,
        metrics={},
    )

    monkeypatch.setattr(engine, "prepare_track", fake_prepare)
    monkeypatch.setattr(engine, "_calculate_plan", lambda *args, **kwargs: plan)
    monkeypatch.setattr(engine, "_apply_tempo_restore", lambda track, end: track)
    monkeypatch.setattr(engine, "_render_advanced_transition", lambda p, a, b: p)
    monkeypatch.setattr(
        engine,
        "_warm_following_tracks",
        lambda *args, **kwargs: future_calls.append("future"),
    )

    assert engine.preload_pair(tracks, 0)
    assert engine._prime_ready_event.wait(timeout=1.0)
    assert engine._prime_next is not None
    assert engine._prime_plan is plan
    # Future B->C planning starts only after playback begins, so it cannot delay
    # the hot A->B pair or compete with the Play thread.
    assert future_calls == []


def test_preload_is_rejected_while_playback_start_is_in_progress() -> None:
    engine = AutoDJEngine()
    engine._start_in_progress = True
    assert not engine.preload_pair([_track("A"), _track("B")], 0)


def test_natural_transitions_establish_incoming_groove_before_bass_swap() -> None:
    from autodj.human_transition import _render_archetype

    sr = 8000
    length = sr * 8
    t = np.arange(length, dtype=np.float32) / sr
    zeros = np.zeros((length, 2), dtype=np.float32)
    perc = np.column_stack([
        0.25 * np.sin(2 * np.pi * 8.0 * t),
        0.25 * np.sin(2 * np.pi * 8.0 * t),
    ]).astype(np.float32)
    harm = np.column_stack([
        0.12 * np.sin(2 * np.pi * 220.0 * t),
        0.12 * np.sin(2 * np.pi * 220.0 * t),
    ]).astype(np.float32)

    for archetype in ("Short Blend", "Bass Swap"):
        _, _, incoming, controls = _render_archetype(
            archetype,
            a_low=zeros,
            b_low=zeros,
            a_harm=zeros,
            b_harm=harm,
            a_perc=perc,
            b_perc=perc,
            sample_rate=sr,
            bpm=120.0,
            beats_per_bar=4,
            bars=16,
            local_gain=1.0,
            effect_strength=0.7,
            vocal_risk=0.1,
        )
        early = incoming[int(0.08 * length) : int(0.22 * length)]
        assert float(np.sqrt(np.mean(np.square(early)))) > 0.006
        assert 0.10 <= float(np.max(controls["drum_b"][: int(0.22 * length)])) <= 0.25
        assert float(np.min(controls["drum_a"][: int(0.22 * length)])) >= 0.95
        cue = int(0.50 * length)
        assert controls["drum_b"][cue] > controls["drum_a"][cue]
        assert controls["drum_a"][int(0.65 * length)] < 0.02


def test_play_pressed_before_prime_current_is_ready_waits_for_same_job(monkeypatch) -> None:
    import threading
    import time
    import autodj.audio_engine as audio_engine_module

    engine = AutoDJEngine(EngineConfig(preload_window_tracks=2, sample_rate=1000))
    tracks = [_track("A"), _track("B")]
    key = engine._pair_key(tracks, 0)
    prepare_calls: list[str] = []

    class FakeStream:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def start(self):
            return None

        def abort(self):
            return None

        def close(self):
            return None

    monkeypatch.setattr(audio_engine_module, "sd", SimpleNamespace(OutputStream=FakeStream))
    monkeypatch.setattr(engine, "stop", lambda clear_preload=False: None)
    monkeypatch.setattr(
        engine,
        "prepare_track",
        lambda item, target_bpm=None: prepare_calls.append(item.title),
    )
    monkeypatch.setattr(engine, "service", lambda: None)

    current = SimpleNamespace(title="A", original_bpm=120.0, analysis=tracks[0])
    incoming = SimpleNamespace(title="B", original_bpm=120.0, analysis=tracks[1])
    plan = SimpleNamespace(current_start=120000, next_start=0)

    with engine._lock:
        engine._prime_key = key
        engine._prime_loading = True
        engine._prime_current = None
        engine._prime_next_synced_base = None
        engine._prime_next = None
        engine._prime_plan = None
        engine._prime_ready_event.clear()

    error: list[BaseException] = []

    def start() -> None:
        try:
            engine.start_playlist(tracks, 0)
        except BaseException as exc:  # pragma: no cover - diagnostic capture
            error.append(exc)

    thread = threading.Thread(target=start)
    thread.start()
    time.sleep(0.05)
    assert thread.is_alive()
    assert prepare_calls == []

    with engine._lock:
        engine._prime_current = current
        engine._prime_next_synced_base = incoming
        engine._prime_next = incoming
        engine._prime_plan = plan
        engine._prime_loading = False
    engine._prime_ready_event.set()

    thread.join(timeout=1.0)
    assert not thread.is_alive()
    assert error == []
    assert prepare_calls == []
    assert engine._current is current
    assert engine._next is incoming
