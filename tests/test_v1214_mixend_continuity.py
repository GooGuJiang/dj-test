from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import autodj.audio_engine as audio_engine_module
from autodj.audio_engine import AutoDJEngine, EngineConfig
from autodj.transition_matcher import find_best_transition
from tests.test_smart_transition import make_track


def test_fallback_curves_reach_unity_at_mixend() -> None:
    curves = AutoDJEngine._cue_centered_curves(
        length=4000,
        gain=0.68,
        switch_position=0.5,
    )
    fade_out, fade_in, bass_out, bass_in, high_out, high_in = curves
    assert fade_out[-1] == pytest.approx(0.0, abs=1e-6)
    assert bass_out[-1] == pytest.approx(0.0, abs=1e-6)
    assert fade_in[-1] == pytest.approx(1.0, abs=1e-6)
    assert bass_in[-1] == pytest.approx(1.0, abs=1e-6)
    assert high_in[-1] == pytest.approx(1.0, abs=1e-6)


def test_simple_transition_resumes_at_exact_next_sample() -> None:
    current = make_track("simple-A", np.linspace(0.6, 0.2, 64), 0)
    incoming = make_track("simple-B", np.linspace(0.2, 0.6, 64), 7)
    base = find_best_transition(current, incoming, requested_bars=8)
    base.metrics["local_gain"] = 0.72
    # Deliberately emulate an advanced phrase-warp resume point. A simple
    # transition must ignore it because its callback reads B linearly.
    base.next_resume_sample = base.next_start + base.length + 321

    engine = AutoDJEngine(EngineConfig(sample_rate=current.sample_rate))
    plan = engine._simple_transition_plan(
        current,
        incoming,
        earliest_start=0,
        base_plan=base,
        mode="Simple Crossfade",
    )

    assert plan.next_resume_sample == plan.next_start + plan.length
    assert plan.fade_in[-1] == pytest.approx(1.0, abs=1e-6)
    assert plan.bass_in[-1] == pytest.approx(1.0, abs=1e-6)


def test_rendered_transition_tail_is_stitched_to_live_next_buffer() -> None:
    current = make_track("seam-A", np.linspace(0.6, 0.2, 64), 0)
    incoming = make_track("seam-B", np.linspace(0.2, 0.6, 64), 7)
    sr = incoming.sample_rate
    t = np.arange(incoming.total_samples, dtype=np.float32) / sr
    wave = np.column_stack(
        [0.25 * np.sin(2.0 * np.pi * 37.0 * t)] * 2
    ).astype(np.float32)
    incoming.audio = wave

    plan = find_best_transition(current, incoming, requested_bars=8)
    plan.next_resume_sample = min(12_000, incoming.total_samples - 2)
    plan.rendered_audio = np.full((plan.length, 2), 0.40, dtype=np.float32)

    engine = AutoDJEngine(EngineConfig(sample_rate=sr))
    result = engine._stitch_rendered_transition_to_next(plan, incoming)

    assert result.rendered_audio is not None
    resume = result.next_resume_sample
    np.testing.assert_allclose(
        result.rendered_audio[-1], incoming.audio[resume - 1], atol=1e-6
    )
    assert result.metrics["mixend_seam_ms"] >= 10.0
    assert result.metrics["mixend_jump_after"] < result.metrics["mixend_jump_before"]
    natural_step = float(np.max(np.abs(incoming.audio[resume] - incoming.audio[resume - 1])))
    assert result.metrics["mixend_jump_after"] == pytest.approx(natural_step, abs=1e-6)


def test_tempo_restore_holds_one_bar_after_mixend_and_uses_dense_map(monkeypatch) -> None:
    synced = make_track(
        "tempo-hold",
        np.full(64, 0.5),
        0,
        playback_bpm=128.0,
        original_bpm=120.0,
    )
    # Use an 8 kHz sample grid so integer target coordinates do not dominate
    # the local-rate continuity measurement.
    scale = 8
    synced.sample_rate *= scale
    synced.audio = np.repeat(synced.audio, scale, axis=0)
    synced.low_audio = np.repeat(synced.low_audio, scale, axis=0)
    synced.mid_audio = np.repeat(synced.mid_audio, scale, axis=0)
    synced.high_audio = np.repeat(synced.high_audio, scale, axis=0)
    synced.source_audio = np.repeat(synced.source_audio, scale, axis=0)
    synced.beat_samples = synced.beat_samples * scale
    synced.downbeat_samples = synced.downbeat_samples * scale
    synced.source_beat_samples = synced.source_beat_samples * scale
    synced.source_downbeat_samples = synced.source_downbeat_samples * scale
    engine = AutoDJEngine(
        EngineConfig(sample_rate=synced.sample_rate, tempo_restore_bars=4)
    )
    captured: dict[str, object] = {}

    def fake_time_map(audio, sample_rate, keyframes, backend):
        del audio, sample_rate, backend
        captured["keyframes"] = list(keyframes)
        source = np.asarray([item[0] for item in keyframes], dtype=np.int64)
        target = np.asarray([item[1] for item in keyframes], dtype=np.int64)
        return (
            np.zeros((int(target[-1]), 2), dtype=np.float32),
            "fake continuous map",
            source,
            target,
        )

    monkeypatch.setattr(audio_engine_module, "stretch_stereo_time_map", fake_time_map)
    monkeypatch.setattr(
        engine,
        "_finish_prepared_track",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )

    mix_end = 8_000 * scale
    result = engine._apply_tempo_restore(synced, mix_end)
    keyframes = captured["keyframes"]
    assert isinstance(keyframes, list)

    initial_rate = synced.stretch_rate
    source_mix_end = int(round(mix_end * initial_rate))
    mix_index = next(i for i, item in enumerate(keyframes) if item[0] == source_mix_end)
    hold_source, hold_target = keyframes[mix_index + 1]

    source_bars = np.diff(synced.source_downbeat_samples)
    source_bar = int(round(float(np.median(source_bars[source_bars > 0]))))
    assert hold_source - source_mix_end >= source_bar
    assert hold_target > mix_end
    held_rate = (hold_source - source_mix_end) / (hold_target - mix_end)
    assert held_rate == pytest.approx(initial_rate, rel=2e-3)
    assert result.tempo_restore_start == hold_target

    # Four restore bars x 16 subdivisions, plus origin/hold/tail keyframes.
    assert len(keyframes) >= 4 * 16 + 3
    dense = np.asarray(keyframes[mix_index + 1 :], dtype=np.float64)
    dense_sources = dense[:-1, 0]
    assert np.all(np.diff(dense_sources) > 0)
    rates = np.diff(dense[:, 0]) / np.diff(dense[:, 1])
    relative_steps = np.abs(np.diff(rates) / np.maximum(rates[:-1], 1e-9))
    assert float(np.max(relative_steps)) < 0.005


def test_phrase_length_shortfall_never_pads_silence(monkeypatch) -> None:
    engine = AutoDJEngine(EngineConfig(sample_rate=8_000))
    source = np.full((100, 2), 0.25, dtype=np.float32)

    monkeypatch.setattr(
        audio_engine_module,
        "quality_time_stretch",
        lambda *args, **kwargs: (np.full((91, 2), 0.25, dtype=np.float32), "fake"),
    )
    rendered, _ = engine._fit_phrase_length(source, 100)
    assert rendered.shape == (100, 2)
    assert float(np.min(np.abs(rendered[-9:]))) > 0.20


def test_audio_callback_crosses_mixend_without_zero_gap() -> None:
    current = make_track("callback-A", np.linspace(0.6, 0.2, 64), 0)
    incoming = make_track("callback-B", np.linspace(0.2, 0.6, 64), 7)

    current.audio[:] = 0.10
    current.low_audio[:] = 0.10
    current.mid_audio[:] = 0.0
    current.high_audio[:] = 0.0
    incoming.audio[:] = 0.20
    incoming.low_audio[:] = 0.20
    incoming.mid_audio[:] = 0.0
    incoming.high_audio[:] = 0.0

    base = find_best_transition(current, incoming, requested_bars=8)
    base.metrics["local_gain"] = 0.70
    engine = AutoDJEngine(EngineConfig(sample_rate=current.sample_rate))
    plan = engine._simple_transition_plan(
        current,
        incoming,
        earliest_start=0,
        base_plan=base,
        mode="Simple Crossfade",
    )

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
    # The final transition samples and the first promoted-track samples are all
    # B at unity. There must be no inserted zero block or gain reset at MIX END.
    seam = out[boundary - 32 : boundary + 32]
    assert np.isfinite(seam).all()
    assert float(np.min(np.abs(seam))) > 0.17
    assert float(np.max(np.abs(np.diff(seam[:, 0])))) < 0.01
    assert engine._current is incoming
    assert engine._current_pos == plan.next_resume_sample + post_samples
