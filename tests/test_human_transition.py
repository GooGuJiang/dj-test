from __future__ import annotations

import numpy as np

from autodj.human_transition import (
    HumanTransitionConfig,
    evaluate_transition,
    render_human_transition,
)


def _roles(sample_rate: int = 8_000, seconds: float = 4.0):
    n = int(sample_rate * seconds)
    t = np.arange(n, dtype=np.float32) / sample_rate
    kick = np.zeros(n, dtype=np.float32)
    beat = int(sample_rate * 0.5)
    for start in range(0, n, beat):
        length = min(int(0.08 * sample_rate), n - start)
        if length > 0:
            env = np.exp(-np.arange(length) / max(1.0, 0.018 * sample_rate))
            kick[start : start + length] += (
                np.sin(2 * np.pi * 68.0 * np.arange(length) / sample_rate) * env
            ).astype(np.float32)
    low_a = np.column_stack([0.18 * np.sin(2 * np.pi * 55 * t)] * 2).astype(np.float32)
    low_b = np.column_stack([0.18 * np.sin(2 * np.pi * 65.4 * t)] * 2).astype(np.float32)
    perc_a = np.column_stack([0.25 * kick] * 2).astype(np.float32)
    perc_b = np.column_stack([0.25 * np.roll(kick, 2)] * 2).astype(np.float32)
    harm_a = np.column_stack([0.08 * np.sin(2 * np.pi * 220 * t)] * 2).astype(np.float32)
    harm_b = np.column_stack([0.08 * np.sin(2 * np.pi * 261.6 * t)] * 2).astype(np.float32)
    return low_a, low_b, harm_a, harm_b, perc_a, perc_b


def test_human_transition_renders_and_limits_peak() -> None:
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
        bars=2,
        local_gain=1.0,
        effect_strength=0.72,
        plan_metrics={
            "vocal_clean": 0.75,
            "edm_confidence": 0.9,
            "harmonic": 0.8,
            "cue_alignment": 0.8,
            "phrase_alignment": 0.9,
        },
        current_label="OUTRO",
        next_label="DROP",
        config=HumanTransitionConfig(max_candidates=4, evaluation_sample_rate=8_000),
    )
    assert result.audio.shape == roles[0].shape
    assert np.isfinite(result.audio).all()
    assert float(np.max(np.abs(result.audio))) <= 0.986
    assert result.archetype
    assert 0.0 <= result.score <= 1.0
    assert result.quality["human_candidate_count"] >= 1


def test_natural_transition_selection_is_deterministic() -> None:
    roles = _roles(seconds=2.0)
    kwargs = dict(
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
        effect_strength=0.7,
        plan_metrics={
            "vocal_clean": 0.8,
            "edm_confidence": 0.8,
            "harmonic": 0.75,
            "bass_clean": 0.7,
            "cue_alignment": 0.8,
            "phrase_alignment": 0.8,
        },
        current_label="OUTRO",
        next_label="INTRO",
        config=HumanTransitionConfig(
            mode="Natural Auto", max_candidates=3, evaluation_sample_rate=8_000
        ),
    )
    first = render_human_transition(**kwargs)
    second = render_human_transition(**kwargs)
    assert first.archetype == second.archetype
    assert first.score == second.score
    assert np.array_equal(first.audio, second.audio)


def test_collision_metric_penalizes_two_full_low_bands() -> None:
    roles = _roles(seconds=2.0)
    clean_a = roles[0] * np.linspace(1, 0, len(roles[0]))[:, None]
    clean_b = roles[1] * np.linspace(0, 1, len(roles[1]))[:, None]
    clean_mix = clean_a + clean_b
    bad_a = roles[0]
    bad_b = roles[1]
    bad_mix = bad_a + bad_b
    controls = {
        "harm_a": np.linspace(1, 0, len(clean_mix), dtype=np.float32),
        "harm_b": np.linspace(0, 1, len(clean_mix), dtype=np.float32),
    }
    clean = evaluate_transition(
        mixed=clean_mix,
        deck_a=clean_a,
        deck_b=clean_b,
        sample_rate=8_000,
        controls=controls,
        archetype="Bass Swap",
        context_fit=0.8,
        vocal_risk=0.0,
        evaluation_sample_rate=8_000,
    )
    bad = evaluate_transition(
        mixed=bad_mix,
        deck_a=bad_a,
        deck_b=bad_b,
        sample_rate=8_000,
        controls=controls,
        archetype="Bass Swap",
        context_fit=0.8,
        vocal_risk=0.0,
        evaluation_sample_rate=8_000,
    )
    assert clean["quality_collision"] > bad["quality_collision"]
