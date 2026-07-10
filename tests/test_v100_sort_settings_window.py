from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np

from autodj.audio_engine import AutoDJEngine, EngineConfig
from autodj.models import SongFormerProfile, FunctionalSegment, MuQProfile, TrackAnalysis
from autodj.playlist_ranker import rank_playlist, transition_compatibility
from autodj.settings_store import SettingsStore


def _track(name: str, bpm: float) -> TrackAnalysis:
    return TrackAnalysis(
        path=f"{name}.wav",
        title=name,
        duration=120.0,
        bpm=bpm,
        beats_per_bar=4,
        beat_times=(0.0, 0.5, 1.0, 1.5),
        beat_numbers=(1, 2, 3, 4),
        downbeat_times=(0.0,),
    )


def _profile(vector: list[float], energy: float = -1.0) -> MuQProfile:
    value = np.asarray(vector, dtype=np.float32)
    value /= max(float(np.linalg.norm(value)), 1e-8)
    acoustic = np.asarray([energy, 0.20, 0.25, 0.08, 0.30, 0.10], dtype=np.float32)
    return MuQProfile(
        global_embedding=value,
        intro_embedding=value,
        outro_embedding=value,
        timeline_embeddings=np.stack([value, value, value]),
        timeline_positions=np.asarray([0.1, 0.5, 0.9], dtype=np.float32),
        acoustic_features=acoustic,
        backend="test",
    )


def _structure(first: str, last: str) -> SongFormerProfile:
    return SongFormerProfile(
        segments=(
            FunctionalSegment(0.0, 30.0, first),
            FunctionalSegment(30.0, 60.0, last),
        ),
        backend="test",
    )


def test_ranker_penalizes_large_style_tempo_energy_jump() -> None:
    a = _track("A", 120.0)
    b = _track("B", 122.0)
    far = _track("Far", 174.0)
    pa = _profile([1.0, 0.0, 0.0], -1.0)
    pb = _profile([0.98, 0.1, 0.0], -0.95)
    pf = _profile([-1.0, 0.0, 0.0], -2.2)

    good = transition_compatibility(
        a, b, pa, pb, _structure("intro", "outro"), _structure("intro", "verse")
    )
    bad = transition_compatibility(
        a, far, pa, pf, _structure("intro", "chorus"), _structure("chorus", "solo")
    )
    assert good.total > bad.total
    assert bad.jump_penalty > good.jump_penalty

    order, _ = rank_playlist(
        [a, far, b],
        [pa, pf, pb],
        start_index=0,
        structures=[
            _structure("intro", "outro"),
            _structure("chorus", "solo"),
            _structure("intro", "verse"),
        ],
    )
    assert order[:2] == [0, 2]


def test_settings_store_roundtrip(tmp_path) -> None:
    path = tmp_path / "settings.json"
    store = SettingsStore(path)
    values = {
        "mix_style": "Club",
        "preload_window": "4",
        "volume": 87.5,
        "auto_muq_sort": True,
    }
    store.save(values)
    assert store.load() == values
    assert not path.with_suffix(".json.tmp").exists()


def test_sliding_window_warms_tracks_after_hot_next(monkeypatch) -> None:
    engine = AutoDJEngine(
        EngineConfig(preload_window_tracks=4, preload_memory_mb=256)
    )
    tracks = [_track("A", 120), _track("B", 122), _track("C", 124), _track("D", 126)]

    def fake_prepare(item, target_bpm=None):
        return SimpleNamespace(
            original_bpm=float(item.bpm),
            audio=np.zeros((128, 2), dtype=np.float32),
            analysis=item,
        )

    plan = SimpleNamespace(next_end=64)
    monkeypatch.setattr(engine, "prepare_track", fake_prepare)
    monkeypatch.setattr(engine, "_calculate_plan", lambda *args, **kwargs: plan)
    monkeypatch.setattr(engine, "_apply_tempo_restore", lambda track, end: track)
    monkeypatch.setattr(engine, "_render_advanced_transition", lambda p, a, b: p)

    assert engine.preload_pair(tracks, 0)
    deadline = time.time() + 2.0
    while engine._prime_loading and time.time() < deadline:
        time.sleep(0.01)

    assert engine._prime_current.analysis.path == "A.wav"
    assert engine._prime_next.analysis.path == "B.wav"
    warm_paths = {key[1] for key in engine._warm_cache.keys()}
    assert warm_paths == {"C.wav", "D.wav"}
