from __future__ import annotations

import numpy as np

from autodj.edm_structure import analyze_edm_structure, checkerboard_novelty
from autodj.models import BarFeatures
from autodj.spectral_seam import spectral_seam_crossfade
from autodj.time_stretch import stretch_stereo


def _features(count: int = 32) -> BarFeatures:
    starts = np.arange(count, dtype=np.int64) * 1000
    ends = starts + 1000
    rms = np.r_[np.full(count // 2, 0.12), np.full(count - count // 2, 0.45)].astype(np.float32)
    onset = np.r_[np.full(count // 2, 0.2), np.full(count - count // 2, 0.8)].astype(np.float32)
    low = np.r_[np.full(count // 2, 0.3), np.full(count - count // 2, 0.75)].astype(np.float32)
    bright = np.r_[np.full(count // 2, 0.2), np.full(count - count // 2, 0.65)].astype(np.float32)
    vocal = np.full(count, 0.18, dtype=np.float32)
    chroma = np.zeros((count, 12), dtype=np.float32)
    chroma[:, 0] = 1.0
    chroma[count // 2 :, 7] = 0.7
    return BarFeatures(starts, ends, rms, low, onset, bright, vocal, chroma)


def test_checkerboard_finds_section_change() -> None:
    values = np.r_[np.zeros((16, 3)), np.ones((16, 3))]
    novelty = checkerboard_novelty(values)
    assert abs(int(np.argmax(novelty)) - 16) <= 2


def test_edm_structure_shapes_and_cue_scores() -> None:
    features = _features()
    t = np.arange(32000, dtype=np.float32) / 8000.0
    audio = np.column_stack((0.1 * np.sin(2 * np.pi * 120 * t),) * 2).astype(np.float32)
    result = analyze_edm_structure(audio, 8000, features, beats_per_bar=4)
    assert result.cue_score.shape == (features.count,)
    assert result.mix_in_score.shape == (features.count,)
    assert result.mix_out_score.shape == (features.count,)
    assert len(result.labels) == features.count
    assert 0.0 <= result.edm_confidence <= 1.0
    assert result.camelot != ""


def test_spectral_graph_cut_transition_is_finite_and_anchored() -> None:
    sr = 8000
    n = sr * 2
    t = np.arange(n, dtype=np.float32) / sr
    outgoing = np.column_stack((0.2 * np.sin(2 * np.pi * 100 * t),) * 2).astype(np.float32)
    incoming = np.column_stack((0.2 * np.sin(2 * np.pi * 500 * t),) * 2).astype(np.float32)
    result = spectral_seam_crossfade(
        outgoing,
        incoming,
        sr,
        n_fft=512,
        hop_length=128,
        graph_freq_bins=18,
        graph_time_bins=28,
    )
    assert result.audio.shape == outgoing.shape
    assert np.isfinite(result.audio).all()
    assert np.max(np.abs(result.audio)) <= 1.0
    assert np.mean(np.abs(result.audio[:64] - outgoing[:64])) < 0.08
    assert np.mean(np.abs(result.audio[-64:] - incoming[-64:])) < 0.08


def test_time_stretch_fallback_preserves_stereo() -> None:
    sr = 8000
    t = np.arange(sr, dtype=np.float32) / sr
    audio = np.column_stack((np.sin(2 * np.pi * 220 * t), np.sin(2 * np.pi * 330 * t))).astype(np.float32) * 0.1
    result, backend = stretch_stereo(audio, sr, rate=1.05, backend="librosa")
    assert result.ndim == 2 and result.shape[1] == 2
    assert len(result) < len(audio)
    assert "librosa" in backend.lower()
    assert np.isfinite(result).all()
