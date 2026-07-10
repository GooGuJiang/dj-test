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


def test_bass_swap_has_single_low_frequency_owner() -> None:
    sr = 8_000
    length = sr * 8
    zeros = np.zeros((length, 2), dtype=np.float32)
    percussion = _pulse_track(length, list(range(0, length, sr // 2)), sr)
    _, _, _, controls = _render_archetype(
        "Bass Swap",
        a_low=zeros,
        b_low=zeros,
        a_harm=zeros,
        b_harm=zeros,
        a_perc=percussion,
        b_perc=percussion,
        sample_rate=sr,
        bpm=120.0,
        beats_per_bar=4,
        bars=8,
        local_gain=1.0,
        effect_strength=0.7,
        vocal_risk=0.0,
    )
    assert np.allclose(controls["bass_a"] + controls["bass_b"], 1.0, atol=1e-6)
    assert controls["bass_b"][int(0.35 * length)] < 0.02
    assert controls["bass_a"][int(0.65 * length)] < 0.02
    # Before the cue, B is only a quiet drum teaser; at the cue it already
    # owns the rhythm, and A releases smoothly shortly afterwards.
    assert 0.10 < controls["drum_b"][int(0.20 * length)] < 0.25
    assert controls["drum_a"][int(0.20 * length)] > 0.95
    assert controls["drum_b"][int(0.50 * length)] > controls["drum_a"][int(0.50 * length)]
    assert controls["drum_a"][int(0.65 * length)] < 0.02


def test_drop_landing_policy_maps_to_safe_bass_handover() -> None:
    roles_a = ["BUILDUP"] * 4 + ["DROP"] * 8
    roles_b = ["BUILDUP"] * 4 + ["DROP"] * 8
    energy = np.asarray([0.4] * 4 + [0.9] * 8, dtype=np.float32)
    a = make_structured_track("A-handover", roles_a, energy)
    b = make_structured_track("B-handover", roles_b, energy)
    result = evaluate_phrase_policy(
        a,
        b,
        current_index=0,
        next_index=0,
        bars=4,
        harmonic=0.62,
        bass_clean=0.55,
    )
    assert result.intent == "Bass Handover"
    assert result.recommended_archetypes[0] == "Bass Swap"

