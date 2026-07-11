from __future__ import annotations

import numpy as np
import pytest

from autodj.human_transition import _cue_quantized_drum_loop, _render_archetype
from autodj.transition_matcher import MatcherConfig, find_best_transition
from tests.test_smart_transition import make_track


def test_default_transition_reserves_four_pre_cue_beats_for_drum_loop() -> None:
    config = MatcherConfig()
    assert config.pre_roll_beats == pytest.approx(4.0)
    assert config.release_beats == pytest.approx(2.0)

    a = make_track("loop-bridge-A", np.linspace(0.65, 0.20, 64), 0)
    b = make_track("loop-bridge-B", np.linspace(0.20, 0.65, 64), 7)
    plan = find_best_transition(a, b, requested_bars=8)
    beat = int(round(60.0 / a.playback_bpm * a.sample_rate))

    assert plan.handoff_offset_samples == 4 * beat
    assert plan.length == 6 * beat
    assert plan.switch_position == pytest.approx(4.0 / 6.0)
    assert plan.next_resume_sample - plan.next_cue_sample == 2 * beat


def test_cue_quantized_drum_loop_repeats_two_beat_phrase_before_cue() -> None:
    sample_rate = 8_000
    bpm = 120.0
    beat = int(round(60.0 / bpm * sample_rate))
    handoff = 4 * beat
    length = 6 * beat

    audio = np.zeros((length, 2), dtype=np.float32)
    pattern = np.linspace(-0.4, 0.4, 2 * beat, dtype=np.float32)
    pattern[:: beat // 4] += 0.18
    audio[handoff : handoff + 2 * beat] = pattern[:, None]

    looped = _cue_quantized_drum_loop(
        audio,
        sample_rate=sample_rate,
        bpm=bpm,
        handoff_index=handoff,
        loop_beats=2.0,
    )

    # Four pre-cue beats are two identical, grid-aligned repetitions.
    assert np.allclose(looped[: 2 * beat], looped[2 * beat : 4 * beat], atol=1e-7)
    # Outside the tiny de-click bridge, the live incoming stem is untouched.
    assert np.allclose(looped[handoff + 64 :], audio[handoff + 64 :], atol=1e-7)
    # Even a deliberately non-periodic source is smoothed at the loop wraps.
    for boundary in (2 * beat, handoff):
        jump = float(np.max(np.abs(looped[boundary] - looped[boundary - 1])))
        assert jump < 0.06


def test_early_bridge_is_percussion_only_then_live_deck_takes_cue() -> None:
    sample_rate = 8_000
    bpm = 120.0
    beat = int(round(60.0 / bpm * sample_rate))
    length = 6 * beat
    t = np.arange(length, dtype=np.float32) / sample_rate

    low_a = np.column_stack([0.14 * np.sin(2 * np.pi * 55.0 * t)] * 2).astype(np.float32)
    low_b = np.column_stack([0.14 * np.sin(2 * np.pi * 65.0 * t)] * 2).astype(np.float32)
    harm_a = np.column_stack([0.08 * np.sin(2 * np.pi * 220.0 * t)] * 2).astype(np.float32)
    harm_b = np.column_stack([0.08 * np.sin(2 * np.pi * 262.0 * t)] * 2).astype(np.float32)
    perc_a = np.column_stack([0.08 * np.sin(2 * np.pi * 8.0 * t)] * 2).astype(np.float32)
    perc_b = np.zeros_like(perc_a)
    cue = 4 * beat
    cue_pattern = np.column_stack(
        [0.10 * np.sin(2 * np.pi * 10.0 * np.arange(2 * beat) / sample_rate)] * 2
    ).astype(np.float32)
    perc_b[cue : cue + 2 * beat] = cue_pattern

    _, _, deck_b, controls = _render_archetype(
        "Short Blend",
        a_low=low_a,
        b_low=low_b,
        a_harm=harm_a,
        b_harm=harm_b,
        a_perc=perc_a,
        b_perc=perc_b,
        sample_rate=sample_rate,
        bpm=bpm,
        beats_per_bar=4,
        bars=1,
        local_gain=1.0,
        effect_strength=0.65,
        vocal_risk=0.2,
        handoff_phase=4.0 / 6.0,
        transition_beats=6.0,
    )

    early = beat
    assert 0.18 <= controls["drum_b"][early] <= 0.25
    assert controls["drum_a"][early] > 0.98
    assert controls["harm_b"][early] < 0.01
    assert controls["bass_b"][early] < 0.01
    assert float(np.mean(np.abs(deck_b[early : early + beat]))) > 0.005

    # The cue is still the ownership switch, not merely the start of a blend.
    assert controls["drum_b"][cue] > controls["drum_a"][cue]
    assert controls["harm_b"][cue] > controls["harm_a"][cue]
    assert controls["bass_b"][cue] > controls["bass_a"][cue]
    assert controls["drum_a"][cue + beat] < 0.01
    assert controls["harm_a"][cue + int(1.25 * beat)] < 0.01


def test_drum_loop_gain_fades_into_live_crossfade_without_level_hole() -> None:
    sample_rate = 8_000
    bpm = 120.0
    beat = int(round(60.0 / bpm * sample_rate))
    length = 6 * beat
    roles = [np.full((length, 2), value, dtype=np.float32) for value in (0.10, 0.10, 0.08, 0.08, 0.06, 0.06)]

    _, _, _, controls = _render_archetype(
        "Bass Swap",
        a_low=roles[0],
        b_low=roles[1],
        a_harm=roles[2],
        b_harm=roles[3],
        a_perc=roles[4],
        b_perc=roles[5],
        sample_rate=sample_rate,
        bpm=bpm,
        beats_per_bar=4,
        bars=1,
        local_gain=1.0,
        effect_strength=0.70,
        vocal_risk=0.1,
        handoff_phase=4.0 / 6.0,
        transition_beats=6.0,
    )

    bridge = slice(beat // 2, 3 * beat)
    assert float(np.min(controls["drum_b"][bridge])) >= 0.20
    assert float(np.min(controls["drum_a"][bridge])) > 0.95
    # As the loop gain releases, the live drum crossfade has already risen.
    combined = controls["drum_b"]
    approach = slice(int(2.5 * beat), 4 * beat)
    assert float(np.min(combined[approach])) > 0.20
    assert float(np.max(np.abs(np.diff(combined)))) < 0.01
    assert np.allclose(controls["bass_a"] + controls["bass_b"], 1.0, atol=1e-6)
