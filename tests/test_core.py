from __future__ import annotations

import numpy as np

from autodj.audio_engine import _normalize_audio, _split_three_bands, related_bpm


def test_related_bpm() -> None:
    assert abs(related_bpm(64.0, 128.0) - 128.0) < 1e-6
    assert abs(related_bpm(140.0, 70.0) - 70.0) < 1e-6
    assert abs(related_bpm(124.0, 128.0) - 124.0) < 1e-6


def test_normalize_audio_is_finite() -> None:
    audio = np.zeros((1000, 2), dtype=np.float32)
    result = _normalize_audio(audio, -18.0)
    assert result.shape == audio.shape
    assert np.isfinite(result).all()


def test_three_band_split_recombines() -> None:
    rng = np.random.default_rng(7)
    audio = rng.normal(0.0, 0.05, (8000, 2)).astype(np.float32)
    low, mid, high = _split_three_bands(audio, 8000, 180.0, 2800.0)
    rebuilt = low + mid + high
    assert np.max(np.abs(rebuilt - audio)) < 1e-5
