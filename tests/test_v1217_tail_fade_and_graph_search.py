from __future__ import annotations

import numpy as np
import pytest

from autodj.audio_engine import AutoDJEngine, EngineConfig
from autodj.playlist_ranker import PairScore, _improve_path, _path_objective
from autodj.transition_matcher import find_best_transition
from tests.test_smart_transition import make_track


def _only_cue(track, index: int, *, incoming: bool = False) -> None:
    track.structure.cue_score[:] = 0.0
    track.structure.mix_in_score[:] = 0.0
    track.structure.mix_out_score[:] = 0.0
    track.structure.cue_score[index] = 1.0
    if incoming:
        track.structure.mix_in_score[index] = 1.0
    else:
        track.structure.mix_out_score[index] = 1.0


def test_late_last_cue_uses_physical_tail_crossfade() -> None:
    current = make_track("tail-A", np.linspace(0.6, 0.2, 16), 0)
    incoming = make_track("tail-B", np.linspace(0.2, 0.6, 16), 7)
    _only_cue(current, 8)
    _only_cue(incoming, 0, incoming=True)

    # Playback is already beyond the final available CUE-DETR OUT point.
    earliest = current.total_samples - 1_000
    base = find_best_transition(
        current,
        incoming,
        earliest_start=earliest,
        requested_bars=4,
    )

    assert base.metrics["late_cue_fallback"] == 1.0
    assert base.metrics["tail_fade_fallback"] == 1.0
    assert base.current_start >= earliest
    assert base.current_end == current.total_samples
    assert base.switch_position == pytest.approx(1.0)
    assert base.next_start == base.next_cue_sample

    engine = AutoDJEngine(EngineConfig(sample_rate=current.sample_rate))
    plan = engine._apply_automix_policy(current, incoming, earliest, base)
    assert plan.transition_mode == "Tail Crossfade"
    assert plan.policy_mode == "Endpoint-safe tail fade"
    assert plan.fade_out[0] == pytest.approx(1.0, abs=1e-6)
    assert plan.fade_out[-1] == pytest.approx(0.0, abs=1e-6)
    assert plan.fade_in[-1] == pytest.approx(1.0, abs=1e-6)


def test_tail_crossfade_promotes_without_zero_gap() -> None:
    current = make_track("tail-callback-A", np.linspace(0.6, 0.2, 16), 0)
    incoming = make_track("tail-callback-B", np.linspace(0.2, 0.6, 16), 7)
    _only_cue(current, 8)
    _only_cue(incoming, 0, incoming=True)

    current.audio[:] = 0.10
    current.low_audio[:] = 0.10
    current.mid_audio[:] = 0.0
    current.high_audio[:] = 0.0
    incoming.audio[:] = 0.20
    incoming.low_audio[:] = 0.20
    incoming.mid_audio[:] = 0.0
    incoming.high_audio[:] = 0.0

    earliest = current.total_samples - 1_000
    base = find_best_transition(current, incoming, earliest_start=earliest)
    engine = AutoDJEngine(EngineConfig(sample_rate=current.sample_rate))
    plan = engine._apply_automix_policy(current, incoming, earliest, base)

    engine._current = current
    engine._next = incoming
    engine._plan = plan
    engine._current_pos = plan.current_start
    engine._next_pos = 0
    engine._transition_pos = 0
    engine._transitioning = False
    engine._skip_plan_after_seek = False
    engine._playing = True
    engine._paused = False
    engine._auto_mix = True
    engine._index = 0

    post_samples = 128
    out = np.zeros((plan.length + post_samples, 2), dtype=np.float32)
    engine._audio_callback(out, len(out), None, 0)

    boundary = plan.length
    seam = out[boundary - 32 : boundary + 32]
    assert np.isfinite(seam).all()
    assert float(np.min(np.abs(seam))) > 0.17
    assert float(np.max(np.abs(np.diff(seam[:, 0])))) < 0.01
    assert engine._current is incoming


def _pair(total: float) -> PairScore:
    return PairScore(
        total=total,
        style=total,
        directional=total,
        trajectory=total,
        tempo=total,
        acoustic=total,
    )


def test_graph_local_polish_removes_weak_internal_edge() -> None:
    scores: dict[tuple[int, int], PairScore] = {
        (a, b): _pair(0.50) for a in range(4) for b in range(4) if a != b
    }
    scores[0, 1] = _pair(0.60)
    scores[1, 2] = _pair(0.20)
    scores[2, 3] = _pair(0.60)
    scores[0, 2] = _pair(0.92)
    scores[2, 1] = _pair(0.90)
    scores[1, 3] = _pair(0.91)
    energies = [0.0, 0.0, 0.0, 0.0]

    original = [0, 1, 2, 3]
    improved = _improve_path(original, scores, energies)
    original_value, original_weakest = _path_objective(original, scores, energies)
    improved_value, improved_weakest = _path_objective(improved, scores, energies)

    assert improved[0] == 0
    assert sorted(improved) == [0, 1, 2, 3]
    assert improved == [0, 2, 1, 3]
    assert improved_value > original_value
    assert improved_weakest > original_weakest
