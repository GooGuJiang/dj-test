from __future__ import annotations

import numpy as np
import pytest

from autodj.audio_engine import AutoDJEngine
from autodj.human_transition import _render_archetype
from autodj.transition_matcher import MatcherConfig, find_best_transition
from tests.test_smart_transition import make_track


def _constant_roles(length: int = 16_000) -> tuple[np.ndarray, ...]:
    a_low = np.full((length, 2), 0.10, dtype=np.float32)
    b_low = np.full((length, 2), 0.10, dtype=np.float32)
    a_harm = np.full((length, 2), 0.08, dtype=np.float32)
    b_harm = np.full((length, 2), 0.08, dtype=np.float32)
    a_perc = np.full((length, 2), 0.06, dtype=np.float32)
    b_perc = np.full((length, 2), 0.06, dtype=np.float32)
    return a_low, b_low, a_harm, b_harm, a_perc, b_perc


def test_default_matcher_reserves_four_beats_for_drum_bridge() -> None:
    assert MatcherConfig().pre_roll_beats == pytest.approx(4.0)
    assert MatcherConfig().release_beats == pytest.approx(2.0)

    a = make_track("balanced-A", np.linspace(0.6, 0.2, 64), 0)
    b = make_track("balanced-B", np.linspace(0.2, 0.6, 64), 7)
    plan = find_best_transition(a, b, requested_bars=8)
    beat = int(round(60.0 / a.playback_bpm * a.sample_rate))

    assert plan.handoff_offset_samples == 4 * beat
    assert plan.length == 6 * beat
    assert plan.metrics["pre_roll_beats"] == pytest.approx(4.0, abs=0.01)
    assert plan.metrics["release_beats"] == pytest.approx(2.0, abs=0.01)
    assert plan.switch_position == pytest.approx(4.0 / 6.0)


def test_advanced_renderer_has_audible_overlap_without_level_hole() -> None:
    roles = _constant_roles()
    _, _, _, controls = _render_archetype(
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
        effect_strength=0.60,
        vocal_risk=0.2,
        handoff_phase=0.5,
        transition_beats=4.0,
    )
    n = len(controls["harm_a"])
    overlap = slice(int(0.38 * n), int(0.62 * n))
    cue = int(0.50 * (n - 1))
    post = int(0.60 * (n - 1))

    # Both musical beds remain present over almost one beat around the cue.
    assert float(np.min(controls["harm_a"][overlap])) > 0.14
    assert float(np.min(controls["harm_b"][overlap])) > 0.39
    # Equal-power content never creates an envelope hole.
    power = np.sqrt(controls["harm_a"] ** 2 + controls["harm_b"] ** 2)
    assert float(np.min(power[overlap])) > 0.95
    assert controls["harm_b"][cue] > controls["harm_a"][cue]
    assert controls["harm_a"][post] > 0.15


def test_fallback_crossfade_matches_balanced_timing_and_reaches_unity() -> None:
    curves = AutoDJEngine._cue_centered_curves(
        length=8_000,
        gain=0.80,
        switch_position=0.5,
        transition_beats=4.0,
    )
    fade_out, fade_in, bass_out, bass_in, _, _ = curves
    before = int(0.40 * (len(fade_out) - 1))
    cue = int(0.50 * (len(fade_out) - 1))
    post = int(0.60 * (len(fade_out) - 1))

    assert fade_out[before] > 0.70
    assert fade_in[before] > 0.35
    assert fade_in[cue] > fade_out[cue] > 0.35
    assert fade_out[post] > 0.15
    assert bass_in[cue] > bass_out[cue]
    assert fade_out[-1] == pytest.approx(0.0, abs=1e-6)
    assert fade_in[-1] == pytest.approx(1.0, abs=1e-6)
    assert bass_out[-1] == pytest.approx(0.0, abs=1e-6)
    assert bass_in[-1] == pytest.approx(1.0, abs=1e-6)
