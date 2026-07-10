from __future__ import annotations

import numpy as np

from autodj.dj_phrase_policy import evaluate_phrase_policy
from autodj.human_transition import HumanTransitionConfig, render_human_transition
from autodj.models import BarFeatures, EDMStructure, PreparedTrack, TrackAnalysis
from autodj.transition_matcher import find_best_transition


def make_structured_track(title: str, roles: list[str], energy: np.ndarray) -> PreparedTrack:
    sr = 1000
    bars = len(roles)
    starts = np.arange(bars, dtype=np.int64) * sr
    ends = starts + sr
    total = bars * sr
    audio = np.zeros((total, 2), dtype=np.float32)
    chroma = np.zeros((bars, 12), dtype=np.float32)
    chroma[:, 0] = 1.0
    features = BarFeatures(
        start_samples=starts,
        end_samples=ends,
        rms=np.asarray(energy, dtype=np.float32),
        low_ratio=np.full(bars, 0.24, dtype=np.float32),
        onset=np.full(bars, 0.55, dtype=np.float32),
        brightness=np.full(bars, 0.40, dtype=np.float32),
        vocal_proxy=np.full(bars, 0.16, dtype=np.float32),
        chroma=chroma,
    )
    analysis = TrackAnalysis(
        path=f"{title}.wav",
        title=title,
        duration=float(bars),
        bpm=120.0,
        beats_per_bar=4,
        beat_times=tuple(i * 0.25 for i in range(bars * 4)),
        beat_numbers=tuple((i % 4) + 1 for i in range(bars * 4)),
        downbeat_times=tuple(float(i) for i in range(bars)),
    )
    beat_samples = np.arange(bars * 4, dtype=np.int64) * 250
    beat_numbers = np.tile(np.arange(1, 5), bars)
    boundaries = np.zeros(bars, dtype=np.float32)
    for index in range(bars):
        if index == 0 or roles[index] != roles[index - 1]:
            boundaries[index] = 1.0
    structure = EDMStructure(
        cue_score=np.maximum(boundaries, 0.4),
        mix_in_score=np.maximum(boundaries, 0.4),
        mix_out_score=np.maximum(boundaries, 0.4),
        phrase_mask=np.where(np.arange(bars) % 8 == 0, 1.0, 0.55).astype(np.float32),
        labels=tuple(roles),
        functional_labels=tuple("INST" for _ in roles),
        allin1_boundary_score=boundaries,
        edm_confidence=0.95,
    )
    return PreparedTrack(
        analysis=analysis,
        audio=audio,
        low_audio=audio.copy(),
        mid_audio=audio.copy(),
        high_audio=audio.copy(),
        source_audio=audio.copy(),
        sample_rate=sr,
        playback_bpm=120.0,
        original_bpm=120.0,
        stretch_rate=1.0,
        beat_samples=beat_samples,
        beat_numbers=beat_numbers,
        downbeat_samples=starts,
        source_beat_samples=beat_samples.copy(),
        source_beat_numbers=beat_numbers.copy(),
        source_downbeat_samples=starts.copy(),
        cue_sample=0,
        waveform_envelope=np.zeros(64, dtype=np.float32),
        bar_features=features,
        structure=structure,
    )


def test_post_drop_relay_is_preferred_over_mid_drop_blend() -> None:
    a_roles = ["PHRASE"] * 20 + ["DROP"] * 12 + ["COOLDOWN"] * 8 + ["OUTRO"] * 8
    b_roles = ["INTRO"] * 8 + ["BUILDUP"] * 8 + ["DROP"] * 16 + ["COOLDOWN"] * 16
    a_energy = np.asarray([0.45] * 20 + [0.90] * 12 + [0.62] * 8 + [0.32] * 8)
    b_energy = np.asarray([0.24] * 8 + list(np.linspace(0.30, 0.78, 8)) + [0.92] * 16 + [0.55] * 16)
    a = make_structured_track("A", a_roles, a_energy)
    b = make_structured_track("B", b_roles, b_energy)

    post = evaluate_phrase_policy(a, b, current_index=32, next_index=0, bars=16, harmonic=0.9, bass_clean=0.9)
    mid = evaluate_phrase_policy(a, b, current_index=24, next_index=0, bars=16, harmonic=0.9, bass_clean=0.9)

    assert post.intent == "Post-Drop Relay"
    assert post.next_landing_role == "DROP"
    assert post.score > mid.score
    assert post.drop_guard_score > mid.drop_guard_score


def test_matcher_lands_incoming_phrase_on_drop() -> None:
    a_roles = ["PHRASE"] * 20 + ["DROP"] * 12 + ["COOLDOWN"] * 8 + ["OUTRO"] * 8
    b_roles = ["INTRO"] * 8 + ["BUILDUP"] * 8 + ["DROP"] * 16 + ["COOLDOWN"] * 16
    a = make_structured_track("A", a_roles, np.asarray([0.5] * 20 + [0.9] * 12 + [0.58] * 8 + [0.3] * 8))
    b = make_structured_track("B", b_roles, np.asarray([0.25] * 8 + list(np.linspace(0.3, 0.8, 8)) + [0.9] * 16 + [0.5] * 16))

    plan = find_best_transition(a, b, requested_bars=0)
    assert plan.dj_intent in {"Post-Drop Relay", "Phrase-to-Drop", "Outro-Intro Blend"}
    assert plan.structure_policy_score >= 0.55
    assert plan.next_landing_role in {"DROP", "CHORUS", "COOLDOWN"}


def test_new_phrase_archetype_renders() -> None:
    sample_rate = 8000
    length = sample_rate * 4
    t = np.arange(length, dtype=np.float32) / sample_rate
    low_a = np.column_stack([0.15 * np.sin(2 * np.pi * 55 * t)] * 2).astype(np.float32)
    low_b = np.column_stack([0.15 * np.sin(2 * np.pi * 65 * t)] * 2).astype(np.float32)
    harm_a = np.column_stack([0.06 * np.sin(2 * np.pi * 220 * t)] * 2).astype(np.float32)
    harm_b = np.column_stack([0.06 * np.sin(2 * np.pi * 260 * t)] * 2).astype(np.float32)
    pulse = np.zeros(length, dtype=np.float32)
    pulse[:: sample_rate // 2] = 1.0
    perc = np.column_stack([pulse, pulse]).astype(np.float32)

    result = render_human_transition(
        a_low=low_a,
        b_low=low_b,
        a_harm=harm_a,
        b_harm=harm_b,
        a_perc=perc,
        b_perc=perc,
        sample_rate=sample_rate,
        bpm=120.0,
        beats_per_bar=4,
        bars=4,
        local_gain=1.0,
        effect_strength=0.7,
        plan_metrics={
            "vocal_clean": 0.9,
            "edm_confidence": 0.95,
            "harmonic": 0.85,
            "bass_clean": 0.9,
            "cue_alignment": 0.9,
            "phrase_alignment": 0.95,
            "dj_intent_code": 2.0,
            "dj_post_drop": 1.0,
            "dj_drop_landing": 1.0,
            "dj_energy_arc": 0.9,
        },
        current_label="COOLDOWN",
        next_label="INTRO",
        config=HumanTransitionConfig(
            mode="Post-Drop Relay",
            max_candidates=1,
            evaluation_sample_rate=8000,
        ),
    )
    assert result.archetype == "Post-Drop Relay"
    assert result.audio.shape == low_a.shape
    assert np.isfinite(result.audio).all()
    assert float(np.max(np.abs(result.audio))) <= 0.90
