from __future__ import annotations

import numpy as np

from autodj.models import MuQProfile, TrackAnalysis
from autodj.playlist_ranker import rank_playlist, transition_compatibility
from autodj.transition_matcher import estimate_micro_alignment
from tests.test_smart_transition import make_track


def _analysis(name: str, bpm: float) -> TrackAnalysis:
    return TrackAnalysis(
        path=f"{name}.wav",
        title=name,
        duration=120.0,
        bpm=bpm,
        beats_per_bar=4,
        beat_times=tuple(np.arange(0.0, 120.0, 60.0 / bpm)),
        beat_numbers=tuple((i % 4) + 1 for i in range(int(120.0 / (60.0 / bpm)))),
        downbeat_times=tuple(np.arange(0.0, 120.0, 4 * 60.0 / bpm)),
    )


def _profile(global_v, intro_v, outro_v) -> MuQProfile:
    vectors = np.asarray([intro_v, global_v, outro_v], dtype=np.float32)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-8)
    return MuQProfile(
        global_embedding=vectors[1],
        intro_embedding=vectors[0],
        outro_embedding=vectors[2],
        timeline_embeddings=vectors,
        timeline_positions=np.asarray([0.1, 0.5, 0.9], dtype=np.float32),
        acoustic_features=np.zeros(6, dtype=np.float32),
        backend="test",
    )


def test_muq_directional_compatibility_and_graph_order() -> None:
    tracks = [_analysis("A", 120.0), _analysis("B", 122.0), _analysis("C", 128.0)]
    profiles = [
        _profile([1, 0, 0], [1, 0, 0], [0, 1, 0]),
        _profile([0.8, 0.6, 0], [0, 1, 0], [0, 0, 1]),
        _profile([0, 0.8, 0.6], [0, 0, 1], [1, 0, 0]),
    ]
    ab = transition_compatibility(tracks[0], tracks[1], profiles[0], profiles[1])
    ba = transition_compatibility(tracks[1], tracks[0], profiles[1], profiles[0])
    assert ab.directional > ba.directional
    order, _ = rank_playlist(tracks, profiles, start_index=0)
    assert order == [0, 1, 2]


def test_micro_alignment_finds_delayed_incoming_kick() -> None:
    a = make_track("A", np.full(16, 0.4), 0)
    b = make_track("B", np.full(16, 0.4), 0)
    # sample rate in make_track is 1000 Hz. B kick is 30 ms later.
    a.audio[10, :] = 1.0
    b.audio[40, :] = 1.0
    offset, confidence = estimate_micro_alignment(a, b, 0, 0, max_offset_ms=80)
    assert 20 <= offset <= 40
    assert confidence > 0.0
