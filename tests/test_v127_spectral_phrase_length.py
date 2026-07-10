from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import autodj.audio_engine as audio_engine_module
from autodj.audio_engine import AutoDJEngine, EngineConfig
from autodj.models import TransitionPlan
from tests.test_dj_phrase_policy import make_structured_track


def test_spectral_seam_uses_locked_phrase_length_without_fallback(monkeypatch) -> None:
    current = make_structured_track("spectral-a", ["OUTRO"] * 8, np.full(8, 0.4))
    incoming = make_structured_track("spectral-b", ["INTRO"] * 8, np.full(8, 0.4))
    phrase_length = 2_000
    plan = TransitionPlan(
        current_start=1_000,
        next_start=0,
        length=phrase_length,
        bars=2,
        current_bar_index=1,
        next_bar_index=0,
        score=0.9,
        fade_out=np.ones(phrase_length, dtype=np.float32),
        fade_in=np.ones(phrase_length, dtype=np.float32),
        bass_out=np.ones(phrase_length, dtype=np.float32),
        bass_in=np.ones(phrase_length, dtype=np.float32),
        high_out=np.ones(phrase_length, dtype=np.float32),
        high_in=np.ones(phrase_length, dtype=np.float32),
        echo_audio=np.zeros((phrase_length, 2), dtype=np.float32),
        transition_mode="Paper EQ/Fader",
        next_resume_sample=phrase_length,
    )
    engine = AutoDJEngine(
        EngineConfig(sample_rate=1_000, transition_engine="Spectral Seam")
    )

    monkeypatch.setattr(
        engine,
        "_render_echo",
        lambda _plan, _current: np.zeros((phrase_length, 2), dtype=np.float32),
    )
    monkeypatch.setattr(
        engine,
        "_fit_phrase_length",
        lambda audio, target_length: (
            np.ascontiguousarray(audio[:target_length], dtype=np.float32),
            "Rubber Band R3",
        ),
    )

    def fake_seam(outgoing, incoming, sample_rate):
        assert len(outgoing) == phrase_length
        assert len(incoming) == phrase_length
        assert sample_rate == 1_000
        return SimpleNamespace(
            audio=np.ascontiguousarray((outgoing + incoming) * 0.5, dtype=np.float32),
            seam_seconds_mean=0.7,
            seam_seconds_std=0.1,
            flow_value=1.2,
        )

    monkeypatch.setattr(audio_engine_module, "spectral_seam_crossfade", fake_seam)

    rendered = engine._render_advanced_transition(plan, current, incoming)

    assert rendered.transition_mode == "Phrase-Locked Spectral Graph-Cut"
    assert rendered.rendered_audio is not None
    assert rendered.rendered_audio.shape == (phrase_length, 2)
    assert rendered.metrics["phrase_warp_ratio"] == 1.0
    assert not any("谱缝合回退" in message for message in engine.drain_events())
