from __future__ import annotations

import numpy as np
import pytest

from autodj.human_transition import _render_archetype
from autodj.transition_matcher import find_best_transition
from tests.test_smart_transition import make_track


def _roles(sample_rate: int = 8_000, seconds: float = 2.0):
    length = int(sample_rate * seconds)
    t = np.arange(length, dtype=np.float32) / sample_rate
    low_a = np.column_stack([0.15 * np.sin(2 * np.pi * 55 * t)] * 2).astype(np.float32)
    low_b = np.column_stack([0.15 * np.sin(2 * np.pi * 65 * t)] * 2).astype(np.float32)
    harm_a = np.column_stack([0.08 * np.sin(2 * np.pi * 220 * t)] * 2).astype(np.float32)
    harm_b = np.column_stack([0.08 * np.sin(2 * np.pi * 262 * t)] * 2).astype(np.float32)
    pulse = np.zeros(length, dtype=np.float32)
    pulse[:: sample_rate // 2] = 0.24
    perc = np.column_stack([pulse, pulse]).astype(np.float32)
    return low_a, low_b, harm_a, harm_b, perc, perc.copy()


def test_cuedetr_point_is_handoff_not_transition_start() -> None:
    a = make_track("cue-A", np.linspace(0.6, 0.2, 64), 0)
    b = make_track("cue-B", np.linspace(0.2, 0.6, 64), 7)
    plan = find_best_transition(a, b, requested_bars=8)

    beat = int(round(60.0 / a.playback_bpm * a.sample_rate))
    assert plan.current_cue_sample - plan.current_start == 2 * beat
    assert plan.next_cue_sample - plan.next_start == 2 * beat
    assert plan.length == 4 * beat
    assert plan.next_resume_sample - plan.next_cue_sample == 2 * beat
    assert plan.switch_position == pytest.approx(0.5)
    assert plan.switch_sample_a == plan.current_cue_sample
    assert plan.switch_sample_b == plan.next_cue_sample


def test_cue_handoff_is_balanced_and_not_a_hard_cut() -> None:
    roles = _roles()
    _, deck_a, deck_b, controls = _render_archetype(
        "Short Blend",
        a_low=roles[0],
        b_low=roles[1],
        a_harm=roles[2],
        b_harm=roles[3],
        a_perc=roles[4],
        b_perc=roles[5],
        sample_rate=8_000,
        bpm=120.0,
        beats_per_bar=4,
        bars=1,
        local_gain=1.0,
        effect_strength=0.65,
        vocal_risk=0.2,
        handoff_phase=0.5,
        transition_beats=4.0,
    )
    length = len(deck_a)
    cue = int(0.5 * (length - 1))
    teaser = int(0.20 * (length - 1))
    overlap = int(0.40 * (length - 1))
    just_after = int(0.60 * (length - 1))
    released = int(0.78 * (length - 1))

    assert 0.15 < controls["drum_b"][teaser] < 0.25
    # The new one-bar window has a clearly audible pre-cue crossfade.
    assert controls["drum_a"][overlap] > 0.70
    assert controls["drum_b"][overlap] > 0.45
    assert controls["harm_a"][overlap] > 0.75
    assert controls["harm_b"][overlap] > 0.40
    # CUE-DETR is still the ownership switch, but A is not abruptly muted.
    assert controls["drum_b"][cue] > controls["drum_a"][cue] > 0.30
    assert controls["harm_b"][cue] > controls["harm_a"][cue] > 0.40
    assert controls["bass_b"][just_after] > 0.98
    assert controls["bass_a"][just_after] < 0.02
    assert controls["harm_a"][just_after] > 0.15
    assert controls["drum_a"][released] < 0.01
    assert controls["harm_a"][released] < 0.01
    assert np.mean(np.abs(deck_b[just_after])) > np.mean(np.abs(deck_a[just_after]))

    # Curves cross over finite samples rather than jumping from 1 to 0.
    for name in ("bass_a", "bass_b", "drum_a", "drum_b", "harm_a", "harm_b"):
        curve = controls[name]
        assert np.isfinite(curve).all()
        assert float(np.max(np.abs(np.diff(curve)))) < 0.01, name


def test_fallback_window_shortens_only_when_track_boundary_requires_it() -> None:
    a = make_track("short-A", np.linspace(0.6, 0.2, 16), 0)
    b = make_track("short-B", np.linspace(0.2, 0.6, 16), 7)
    # Force the only usable B cue near the beginning. The cue remains neural;
    # only its unavailable pre-roll is shortened.
    b.structure.cue_score[:] = 0.0
    b.structure.mix_in_score[:] = 0.0
    b.structure.cue_score[0] = 1.0
    b.structure.mix_in_score[0] = 1.0
    plan = find_best_transition(a, b, requested_bars=4)
    assert plan.next_bar_index == 0
    assert plan.next_start == plan.next_cue_sample == 0
    assert 0.0 <= plan.switch_position < 0.5
    assert plan.length <= 2 * int(round(60.0 / a.playback_bpm * a.sample_rate))
