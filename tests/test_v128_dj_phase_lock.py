from __future__ import annotations

import numpy as np

from autodj.human_transition import lock_incoming_deck_phase


def _pulse_track(length: int, positions: np.ndarray, sr: int) -> np.ndarray:
    mono = np.zeros(length, dtype=np.float32)
    for pos in positions.astype(int):
        if pos < 0 or pos >= length:
            continue
        size = min(int(0.035 * sr), length - pos)
        env = np.exp(-np.arange(size, dtype=np.float32) / max(1.0, 0.007 * sr))
        mono[pos : pos + size] += env
    return np.column_stack([mono, mono]).astype(np.float32)


def _peak_error(audio: np.ndarray, centers: np.ndarray, radius: int) -> np.ndarray:
    onset = np.abs(np.diff(audio[:, 0], prepend=audio[0, 0]))
    errors: list[int] = []
    for center in centers.astype(int):
        lo = max(0, center - radius)
        hi = min(len(onset), center + radius + 1)
        errors.append(lo + int(np.argmax(onset[lo:hi])) - center)
    return np.asarray(errors, dtype=np.int64)


def test_whole_incoming_deck_uses_one_phase_map() -> None:
    sr = 8_000
    bpm = 120.0
    beat = int(sr * 60.0 / bpm)
    length = beat * 8
    a_beats = np.arange(8, dtype=np.float64) * beat
    b_beats = a_beats + int(round(0.024 * sr))
    a = _pulse_track(length, a_beats, sr)
    b = _pulse_track(length, b_beats, sr)

    out_low, out_harm, out_perc, metrics = lock_incoming_deck_phase(
        a_low=a,
        b_low=b,
        a_harm=a,
        b_harm=b,
        a_perc=a,
        b_perc=b,
        sample_rate=sr,
        bpm=bpm,
        beats_per_bar=4,
        bars=2,
        a_beat_positions=a_beats,
        b_beat_positions=b_beats,
        a_downbeat_positions=a_beats[::4],
        b_downbeat_positions=b_beats[::4],
    )

    centers = a_beats[1:-2]
    low_error = _peak_error(out_low, centers, int(0.07 * sr))
    harm_error = _peak_error(out_harm, centers, int(0.07 * sr))
    perc_error = _peak_error(out_perc, centers, int(0.07 * sr))
    assert np.median(np.abs(perc_error)) <= int(0.003 * sr)
    assert np.array_equal(low_error, harm_error)
    assert np.array_equal(harm_error, perc_error)
    assert metrics["beat_lock_post_error_ms"] < 3.0
    assert metrics["beat_lock_terminal_shift_samples"] == 0.0


def test_one_bad_kick_does_not_warp_every_other_beat() -> None:
    sr = 8_000
    bpm = 120.0
    beat = int(sr * 60.0 / bpm)
    length = beat * 12
    a_beats = np.arange(12, dtype=np.float64) * beat
    steady_delay = int(round(0.020 * sr))
    b_audio_beats = a_beats + steady_delay
    # One intentionally syncopated/bad transient should not drag its neighbours.
    b_audio_beats[5] -= int(round(0.055 * sr))
    a = _pulse_track(length, a_beats, sr)
    b = _pulse_track(length, b_audio_beats, sr)
    b_grid = a_beats + steady_delay

    _, _, out_perc, metrics = lock_incoming_deck_phase(
        a_low=a,
        b_low=b,
        a_harm=a,
        b_harm=b,
        a_perc=a,
        b_perc=b,
        sample_rate=sr,
        bpm=bpm,
        beats_per_bar=4,
        bars=3,
        a_beat_positions=a_beats,
        b_beat_positions=b_grid,
        a_downbeat_positions=a_beats[::4],
        b_downbeat_positions=b_grid[::4],
    )

    errors = _peak_error(out_perc, a_beats[1:-2], int(0.08 * sr))
    normal_errors = np.delete(errors, 4)  # the deliberately displaced beat
    assert np.median(np.abs(normal_errors)) <= int(0.003 * sr)
    assert metrics["beat_lock_bar_nudges"] == 3.0
    assert metrics["beat_lock_confidence"] > 0.65
