from __future__ import annotations

from types import SimpleNamespace

from app import AutoDJApp, _rank_tracks_snapshot
from autodj.models import MuQProfile, TrackAnalysis


def _track(name: str, bpm: float) -> TrackAnalysis:
    beat = 60.0 / bpm
    beats = tuple(index * beat for index in range(64))
    return TrackAnalysis(
        path=f"{name}.wav",
        title=name,
        duration=max(120.0, beats[-1] + beat),
        bpm=bpm,
        beats_per_bar=4,
        beat_times=beats,
        beat_numbers=tuple((index % 4) + 1 for index in range(len(beats))),
        downbeat_times=beats[::4],
    )


def _var(value):
    return SimpleNamespace(get=lambda: value)


def test_rank_snapshot_reorders_with_bpm_when_muq_is_missing() -> None:
    tracks = [_track("A", 120.0), _track("Far", 174.0), _track("Near", 122.0)]

    ordered, pair_map, average, worst = _rank_tracks_snapshot(
        tracks,
        profile_map={},
        structure_map={},
        selected_index=0,
    )

    assert [track.title for track in ordered] == ["A", "Near", "Far"]
    assert set(pair_map) == {"A.wav", "Near.wav"}
    assert 0.0 <= worst <= average <= 1.0


def test_rank_snapshot_preserves_audible_transition_prefix() -> None:
    tracks = [
        _track("Played", 100.0),
        _track("A", 120.0),
        _track("B", 170.0),
        _track("C", 122.0),
    ]

    ordered, _, _, _ = _rank_tracks_snapshot(
        tracks,
        profile_map={track.path: MuQProfile() for track in tracks},
        structure_map={},
        playing=True,
        current_path="A.wav",
        next_path="B.wav",
        transitioning=True,
    )

    # Played history and already-audible B remain fixed; only B and later tracks
    # are eligible for graph ordering.
    assert [track.title for track in ordered[:3]] == ["Played", "A", "B"]
    assert sorted(track.title for track in ordered) == ["A", "B", "C", "Played"]


def test_analysis_completion_uses_fallback_sort_when_muq_is_disabled() -> None:
    app = AutoDJApp.__new__(AutoDJApp)
    app._pending_auto_sort = True
    app.auto_muq_sort_var = _var(True)
    app.muq_enabled_var = _var(False)
    app.tracks = [_track("A", 120.0), _track("B", 122.0)]
    calls: list[str] = []
    app._log = lambda text: calls.append(f"log:{text}")
    app._rank_without_muq = lambda reason: calls.append(f"rank:{reason}")
    app._finish_analysis_progress = lambda text: calls.append(f"finish:{text}")
    app._schedule_preload = lambda *args, **kwargs: calls.append("preload")

    AutoDJApp._continue_after_cuedetr(app)

    assert "rank:MuQ 未启用" in calls
    assert app._pending_auto_sort is True
    assert not any(item.startswith("finish:") for item in calls)


def test_automatic_sort_request_is_not_lost_while_analysis_is_busy() -> None:
    app = AutoDJApp.__new__(AutoDJApp)
    app.muq_ranking_active = False
    app.analysis_active = True
    app.allin1_analysis_active = False
    app.cuedetr_analysis_active = False
    app._pending_auto_sort = False
    messages: list[str] = []
    app._log = messages.append

    AutoDJApp._rank_with_muq(app, automatic=True)

    assert app._pending_auto_sort is True
    assert any("已排队" in message for message in messages)
