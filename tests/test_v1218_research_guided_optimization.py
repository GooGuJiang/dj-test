from __future__ import annotations

import numpy as np

from autodj.audio_engine import (
    _correlation_compensated_pair,
    _normalized_audio_correlation,
)
from autodj.playlist_ranker import (
    PairScore,
    _energy_arc_term,
    _exact_rank_small,
)
from autodj.transition_matcher import find_best_transition
from tests.test_smart_transition import make_track


def _pair(total: float) -> PairScore:
    return PairScore(
        total=total,
        style=total,
        directional=total,
        trajectory=total,
        tempo=total,
        acoustic=total,
    )


def test_positive_correlation_compensation_removes_center_power_bump() -> None:
    length = 2_000
    phase = np.linspace(0.0, np.pi / 2.0, length, dtype=np.float32)
    fade_out = np.cos(phase).astype(np.float32)
    fade_in = np.sin(phase).astype(np.float32)
    wave = np.sin(np.linspace(0.0, 40.0 * np.pi, length, dtype=np.float32))
    stereo = np.column_stack([wave, wave]).astype(np.float32)

    correlation = _normalized_audio_correlation(stereo, stereo.copy())
    adapted_out, adapted_in, reduction_db = _correlation_compensated_pair(
        fade_out, fade_in, correlation
    )

    raw_gain = fade_out + fade_in
    adapted_gain = adapted_out + adapted_in
    center = length // 2
    assert correlation > 0.99
    assert raw_gain[center] > 1.40
    assert adapted_gain[center] < 1.02
    assert reduction_db > 2.8
    assert adapted_out[0] == fade_out[0]
    assert adapted_in[-1] == fade_in[-1]


def test_energy_context_rewards_controlled_reversal() -> None:
    alternating = _energy_arc_term(0.30, -0.30)
    same_direction = _energy_arc_term(0.30, 0.30)
    assert alternating > same_direction
    assert alternating > 0.0
    assert same_direction < 0.0


def test_exact_small_playlist_search_avoids_greedy_dead_end() -> None:
    scores = {
        (a, b): _pair(0.20)
        for a in range(4)
        for b in range(4)
        if a != b
    }
    scores[0, 1] = _pair(0.99)  # tempting first edge, bad continuation
    scores[1, 2] = _pair(0.08)
    scores[1, 3] = _pair(0.85)
    scores[0, 2] = _pair(0.86)
    scores[2, 1] = _pair(0.87)
    scores[2, 3] = _pair(0.12)

    order = _exact_rank_small(4, 0, scores, [0.0, 0.0, 0.0, 0.0])
    assert order == [0, 2, 1, 3]


def test_late_tail_prefers_in_cue_that_can_stand_alone() -> None:
    current = make_track("tail-research-A", np.linspace(0.6, 0.2, 16), 0)
    incoming_energy = np.concatenate(
        [np.full(12, 0.55), np.full(12, 0.30), np.full(8, 0.015)]
    )
    incoming = make_track("tail-research-B", incoming_energy, 0)

    current.structure.cue_score[:] = 0.0
    current.structure.mix_out_score[:] = 0.0
    current.structure.cue_score[8] = 1.0
    current.structure.mix_out_score[8] = 1.0

    incoming.structure.cue_score[:] = 0.0
    incoming.structure.mix_in_score[:] = 0.0
    incoming.structure.cue_score[0] = 0.75
    incoming.structure.mix_in_score[0] = 0.75
    incoming.structure.cue_score[24] = 1.0
    incoming.structure.mix_in_score[24] = 1.0

    plan = find_best_transition(
        current,
        incoming,
        earliest_start=current.total_samples - 1_000,
        requested_bars=4,
    )

    assert plan.metrics["late_cue_fallback"] == 1.0
    assert plan.next_bar_index == 0
    assert plan.metrics["tail_entry_quality"] > 0.70
