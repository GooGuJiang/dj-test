from __future__ import annotations

import numpy as np

from autodj.beat_this_analyzer import (
    _estimate_bpm,
    _infer_numbers_fallback,
    _regularize_steady_grid,
)
from autodj.time_stretch import stretch_stereo_time_map


def test_beat_this_grid_repairs_single_missing_beat() -> None:
    beats = np.arange(0.0, 16.0, 0.5)
    beats = np.delete(beats, 13)
    downbeats = np.arange(0.0, 16.0, 2.0)
    repaired, snapped = _regularize_steady_grid(beats, downbeats)
    assert len(repaired) == 32
    assert np.min(np.abs(repaired - 6.5)) < 1e-6
    assert len(snapped) == len(downbeats)
    assert abs(_estimate_bpm(repaired) - 120.0) < 0.5


def test_fallback_numbers_are_bar_aligned() -> None:
    beats = np.arange(0.0, 8.0, 0.5)
    downbeats = np.arange(0.0, 8.0, 2.0)
    numbers = _infer_numbers_fallback(beats, downbeats)
    assert np.array_equal(numbers[:8], np.asarray([1, 2, 3, 4, 1, 2, 3, 4]))


def test_continuous_time_map_has_requested_duration_and_stereo() -> None:
    sr = 8000
    seconds = 4
    t = np.arange(sr * seconds, dtype=np.float32) / sr
    audio = np.column_stack(
        (
            0.12 * np.sin(2 * np.pi * 110 * t),
            0.12 * np.sin(2 * np.pi * 220 * t),
        )
    ).astype(np.float32)

    # 前半段 1.1x，后半段连续恢复到原速。
    keyframes = [
        (0, 0),
        (sr * 2, int(round(sr * 2 / 1.1))),
        (sr * 3, int(round(sr * 2 / 1.1 + sr / 1.05))),
        (sr * 4, int(round(sr * 2 / 1.1 + sr / 1.05 + sr))),
    ]
    result, backend, sources, targets = stretch_stereo_time_map(
        audio,
        sample_rate=sr,
        keyframes=keyframes,
        backend="librosa",
    )
    assert result.shape == (targets[-1], 2)
    assert np.isfinite(result).all()
    assert np.all(np.diff(sources) > 0)
    assert np.all(np.diff(targets) > 0)
    assert "continuous" in backend.lower()


def test_analyzer_accepts_beat_this_output_and_writes_cache(tmp_path) -> None:
    import soundfile as sf

    from autodj.beat_this_analyzer import BeatThisAnalyzer

    sr = 8000
    audio_path = tmp_path / "grid.wav"
    sf.write(audio_path, np.zeros(sr * 5, dtype=np.float32), sr)
    beats = np.arange(0.0, 5.0, 0.5, dtype=np.float64)
    downbeats = np.asarray([0.0, 2.0, 4.0], dtype=np.float64)

    analyzer = BeatThisAnalyzer(
        checkpoint="small0",
        device="cpu",
        cache_path=tmp_path / "cache.json",
    )
    analyzer._get_model = lambda: (lambda _path: (beats, downbeats))  # type: ignore[method-assign]
    result = analyzer.analyze(audio_path, force=True)

    assert abs(result.bpm - 120.0) < 0.5
    assert result.beats_per_bar == 4
    assert result.beat_numbers[:8] == (1, 2, 3, 4, 1, 2, 3, 4)
    assert (tmp_path / "cache.json").exists()
