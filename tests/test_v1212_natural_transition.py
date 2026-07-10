from __future__ import annotations

import numpy as np
import pytest

from autodj.audio_engine import AutoDJEngine
from autodj.human_transition import (
    HumanTransitionConfig,
    SUPPORTED_ARCHETYPES,
    _render_archetype,
    render_human_transition,
)


def _roles(sample_rate: int = 8_000, seconds: float = 8.0):
    length = int(sample_rate * seconds)
    t = np.arange(length, dtype=np.float32) / sample_rate
    low_a = np.column_stack([0.16 * np.sin(2 * np.pi * 55 * t)] * 2).astype(np.float32)
    low_b = np.column_stack([0.16 * np.sin(2 * np.pi * 65 * t)] * 2).astype(np.float32)
    harm_a = np.column_stack([0.08 * np.sin(2 * np.pi * 220 * t)] * 2).astype(np.float32)
    harm_b = np.column_stack([0.08 * np.sin(2 * np.pi * 262 * t)] * 2).astype(np.float32)
    pulse = np.zeros(length, dtype=np.float32)
    pulse[:: sample_rate // 2] = 0.28
    perc = np.column_stack([pulse, pulse]).astype(np.float32)
    return low_a, low_b, harm_a, harm_b, perc, perc.copy()


def test_only_conservative_transition_modes_are_exposed() -> None:
    assert SUPPORTED_ARCHETYPES == ("Short Blend", "Bass Swap", "Echo Out")
    engine = AutoDJEngine()
    engine.set_human_style_mode("Adaptive Human")
    assert engine.config.human_style_mode == "Natural Auto"
    engine.set_human_style_mode("Long Blend")
    assert engine.config.human_style_mode == "Short Blend"


@pytest.mark.parametrize("removed", [
    "Drop Swap",
    "Double Drop",
    "Loop Out",
    "Filter Ride",
    "Post-Drop Relay",
    "Breakdown Lift",
])
def test_removed_impact_modes_cannot_render(removed: str) -> None:
    roles = _roles(seconds=2.0)
    with pytest.raises(ValueError, match="unsupported natural transition"):
        _render_archetype(
            removed,
            a_low=roles[0],
            b_low=roles[1],
            a_harm=roles[2],
            b_harm=roles[3],
            a_perc=roles[4],
            b_perc=roles[5],
            sample_rate=8_000,
            bpm=120.0,
            beats_per_bar=4,
            bars=4,
            local_gain=1.0,
            effect_strength=0.7,
            vocal_risk=0.1,
        )


def test_clean_pair_uses_two_natural_candidates_without_echo() -> None:
    roles = _roles()
    result = render_human_transition(
        a_low=roles[0],
        b_low=roles[1],
        a_harm=roles[2],
        b_harm=roles[3],
        a_perc=roles[4],
        b_perc=roles[5],
        sample_rate=8_000,
        bpm=120.0,
        beats_per_bar=4,
        bars=8,
        local_gain=1.0,
        effect_strength=0.65,
        plan_metrics={
            "vocal_clean": 0.90,
            "edm_confidence": 0.85,
            "harmonic": 0.80,
            "bass_clean": 0.75,
            "cue_alignment": 0.90,
            "phrase_alignment": 0.95,
        },
        current_label="SECTION",
        next_label="INTRO",
        config=HumanTransitionConfig(mode="Natural Auto", max_candidates=3, evaluation_sample_rate=8_000),
    )
    assert result.archetype in {"Short Blend", "Bass Swap"}
    assert result.quality["human_candidate_count"] == 2.0
    assert float(np.max(np.abs(result.audio))) <= 0.90


def test_risky_vocal_pair_adds_echo_as_safety_candidate() -> None:
    roles = _roles()
    result = render_human_transition(
        a_low=roles[0],
        b_low=roles[1],
        a_harm=roles[2],
        b_harm=roles[3],
        a_perc=roles[4],
        b_perc=roles[5],
        sample_rate=8_000,
        bpm=120.0,
        beats_per_bar=4,
        bars=8,
        local_gain=1.0,
        effect_strength=0.55,
        plan_metrics={
            "vocal_clean": 0.30,
            "edm_confidence": 0.70,
            "harmonic": 0.30,
            "bass_clean": 0.65,
            "cue_alignment": 0.88,
            "phrase_alignment": 0.90,
            "dj_intent_code": 6.0,
        },
        current_label="VOCAL",
        next_label="INTRO",
        config=HumanTransitionConfig(mode="Natural Auto", max_candidates=3, evaluation_sample_rate=8_000),
    )
    assert result.quality["human_candidate_count"] == 3.0
    assert result.archetype in SUPPORTED_ARCHETYPES


def test_natural_control_curves_are_continuous_and_end_owned() -> None:
    roles = _roles()
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
        bars=8,
        local_gain=1.0,
        effect_strength=0.65,
        vocal_risk=0.2,
    )
    assert controls["bass_a"][0] == pytest.approx(1.0)
    assert controls["bass_b"][0] == pytest.approx(0.0)
    assert controls["bass_a"][-1] == pytest.approx(0.0)
    assert controls["bass_b"][-1] == pytest.approx(1.0)
    assert controls["drum_a"][-1] == pytest.approx(0.0)
    assert controls["drum_b"][-1] == pytest.approx(1.0)
    for name, curve in controls.items():
        assert np.isfinite(curve).all(), name
        assert float(np.max(np.abs(np.diff(curve)))) < 0.01, name
