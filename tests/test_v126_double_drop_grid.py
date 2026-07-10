from __future__ import annotations

import numpy as np

from autodj.dj_phrase_policy import evaluate_phrase_policy
from autodj.human_transition import _render_archetype, align_percussive_beat_grid
from tests.test_dj_phrase_policy import make_structured_track


def _pulse_track(length: int, positions: list[int], sr: int) -> np.ndarray:
    mono = np.zeros(length, dtype=np.float32)
    for pos in positions:
        if pos < 0 or pos >= length:
            continue
        size = min(int(0.035 * sr), length - pos)
        env = np.exp(-np.arange(size, dtype=np.float32) / max(1.0, 0.007 * sr))
        mono[pos : pos + size] += env
    return np.column_stack([mono, mono]).astype(np.float32)


def _peak_near(audio: np.ndarray, center: int, radius: int) -> int:
    mono = np.mean(audio, axis=1)
    onset = np.abs(np.diff(mono, prepend=mono[0]))
    lo, hi = max(0, center - radius), min(len(onset), center + radius + 1)
    return lo + int(np.argmax(onset[lo:hi]))


def test_percussive_grid_alignment_corrects_multiple_beats() -> None:
    sr = 8_000
    bpm = 120.0
    beat = int(sr * 60 / bpm)
    length = beat * 8
    nominal = [i * beat for i in range(8)]
    delays = [0, 96, 72, 112, 64, 104, 80, 0]
    a = _pulse_track(length, nominal, sr)
    b = _pulse_track(length, [p + d for p, d in zip(nominal, delays)], sr)
    zeros = np.zeros_like(a)

    _, aligned, metrics = align_percussive_beat_grid(
        a_low=zeros,
        b_low=zeros,
        a_perc=a,
        b_perc=b,
        sample_rate=sr,
        bpm=bpm,
        beats_per_bar=4,
        bars=2,
    )

    before = []
    after = []
    for center in nominal[1:-1]:
        pa = _peak_near(a, center, int(0.08 * sr))
        before.append(abs(_peak_near(b, center, int(0.08 * sr)) - pa))
        after.append(abs(_peak_near(aligned, center, int(0.08 * sr)) - pa))
    assert metrics["beat_grid_aligned_beats"] >= 4
    assert np.mean(after) < np.mean(before) * 0.45
    assert np.isfinite(aligned).all()


def test_double_drop_overlaps_drums_before_releasing_deck_a() -> None:
    sr = 8_000
    bpm = 120.0
    length = sr * 8
    zeros = np.zeros((length, 2), dtype=np.float32)
    percussion = _pulse_track(length, list(range(0, length, sr // 2)), sr)
    landing = 0.72
    _, _, _, controls = _render_archetype(
        "Double Drop",
        a_low=zeros,
        b_low=zeros,
        a_harm=zeros,
        b_harm=zeros,
        a_perc=percussion,
        b_perc=percussion,
        sample_rate=sr,
        bpm=bpm,
        beats_per_bar=4,
        bars=4,
        local_gain=1.0,
        effect_strength=0.7,
        vocal_risk=0.0,
        drop_landing_phase=landing,
    )
    beat_phase = (60.0 / bpm * sr) / length
    at_landing = int(landing * (length - 1))
    after_release = int(min(0.99, landing + 1.65 * beat_phase) * (length - 1))
    assert controls["drum_a"][at_landing] > 0.78
    assert controls["drum_b"][at_landing] > 0.70
    assert controls["drum_overlap"][at_landing] > 0.70
    assert controls["drum_a"][after_release] < 0.35
    assert controls["drum_b"][after_release] > 0.92


def test_double_drop_policy_uses_moderate_compatibility_thresholds() -> None:
    roles_a = ["BUILDUP"] * 4 + ["DROP"] * 8
    roles_b = ["BUILDUP"] * 4 + ["DROP"] * 8
    energy = np.asarray([0.4] * 4 + [0.9] * 8, dtype=np.float32)
    a = make_structured_track("A-double", roles_a, energy)
    b = make_structured_track("B-double", roles_b, energy)
    result = evaluate_phrase_policy(
        a,
        b,
        current_index=0,
        next_index=0,
        bars=4,
        harmonic=0.62,
        bass_clean=0.55,
    )
    assert result.intent == "Double Drop"
    assert result.recommended_archetypes[0] == "Double Drop"
