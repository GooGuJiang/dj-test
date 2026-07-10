from __future__ import annotations

"""
Human-style automatic DJ transition generation and reference-aware scoring.

The module deliberately separates two concerns:
1. Generate several phrase-quantized transition archetypes that resemble common
   DJ techniques (blend, bass swap, echo out, drop swap, loop out, filter ride).
2. Rank the rendered candidates with interpretable engineering/perceptual
   metrics: loudness/headroom, spectral collision, spectral continuity, gain
   smoothness, stereo stability, beat consistency, and context/style fit.

This is an engineering implementation inspired by public DJ-transition and
transition-evaluation research. It does not claim to reproduce a proprietary DJ
system or a trained human-DJ policy model.
"""

from dataclasses import dataclass
from typing import Iterable, Sequence

import librosa
import numpy as np
from scipy import signal


SUPPORTED_ARCHETYPES = (
    "Long Blend",
    "Bass Swap",
    "Echo Out",
    "Drop Swap",
    "Loop Out",
    "Filter Ride",
    "Post-Drop Relay",
    "Breakdown Lift",
    "Double Drop",
)


@dataclass(frozen=True)
class HumanTransitionConfig:
    mode: str = "Adaptive Human"
    variation: float = 0.58
    max_candidates: int = 5
    avoid_recent: int = 2
    evaluation_sample_rate: int = 16_000


@dataclass
class HumanTransitionResult:
    audio: np.ndarray
    archetype: str
    score: float
    quality: dict[str, float]
    deck_a: np.ndarray
    deck_b: np.ndarray


def _smoothstep(value: np.ndarray | float) -> np.ndarray:
    x = np.clip(np.asarray(value, dtype=np.float32), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _smootherstep(value: np.ndarray | float) -> np.ndarray:
    x = np.clip(np.asarray(value, dtype=np.float32), 0.0, 1.0)
    return x**3 * (x * (x * 6.0 - 15.0) + 10.0)


def _ramp(phase: np.ndarray, start: float, end: float, smoother: bool = True) -> np.ndarray:
    if end <= start + 1e-6:
        return (phase >= start).astype(np.float32)
    x = (phase - float(start)) / float(end - start)
    return _smootherstep(x) if smoother else _smoothstep(x)


def _bar_quantized(value: float, bars: int, subdivision: int = 2) -> float:
    steps = max(1, int(bars) * max(1, int(subdivision)))
    return float(np.clip(round(float(value) * steps) / steps, 0.0, 1.0))


def _equal_power(progress: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.clip(progress, 0.0, 1.0)
    return (
        np.cos(p * np.pi / 2.0).astype(np.float32),
        np.sin(p * np.pi / 2.0).astype(np.float32),
    )


def _frame_rms(audio: np.ndarray, frame: int = 2048, hop: int = 512) -> np.ndarray:
    mono = np.mean(audio, axis=1, dtype=np.float64)
    if mono.size < frame:
        mono = np.pad(mono, (0, frame - mono.size))
    return librosa.feature.rms(
        y=mono.astype(np.float32), frame_length=frame, hop_length=hop, center=True
    )[0].astype(np.float64)


def _db(values: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(np.asarray(values, dtype=np.float64), 1e-8))


def _soft_limit(audio: np.ndarray, threshold: float = 0.86, ceiling: float = 0.89) -> np.ndarray:
    result = np.asarray(audio, dtype=np.float32).copy()
    peak = float(np.max(np.abs(result)) + 1e-12)
    if peak > threshold:
        drive = 1.16
        result = np.tanh(result * drive) / np.tanh(drive)
    peak = float(np.max(np.abs(result)) + 1e-12)
    if peak > ceiling:
        result *= np.float32(ceiling / peak)
    return np.ascontiguousarray(result, dtype=np.float32)


def _lowpass_envelope(audio: np.ndarray, sample_rate: int, cutoff: float = 180.0) -> np.ndarray:
    sos = signal.butter(4, cutoff, btype="lowpass", fs=sample_rate, output="sos")
    return signal.sosfiltfilt(sos, audio, axis=0).astype(np.float32)


def _resample_for_eval(audio: np.ndarray, sample_rate: int, target: int) -> np.ndarray:
    if sample_rate == target:
        return np.asarray(audio, dtype=np.float32)
    # scipy.resample_poly avoids an optional resampy dependency and is adequate
    # for low-rate candidate evaluation.
    from math import gcd

    divisor = gcd(int(sample_rate), int(target))
    up = int(target) // divisor
    down = int(sample_rate) // divisor
    rendered = signal.resample_poly(
        np.asarray(audio, dtype=np.float32), up=up, down=down, axis=0
    )
    return np.ascontiguousarray(rendered, dtype=np.float32)


def _band_energy(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    mono = np.mean(audio, axis=1, dtype=np.float64)
    n_fft = 2048
    hop = 512
    if mono.size < n_fft:
        mono = np.pad(mono, (0, n_fft - mono.size))
    spectrum = np.abs(librosa.stft(mono.astype(np.float32), n_fft=n_fft, hop_length=hop)) ** 2
    frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
    bands = ((20.0, 180.0), (180.0, 1_000.0), (1_000.0, 6_000.0), (6_000.0, sample_rate / 2.0))
    output = []
    for low, high in bands:
        mask = (frequencies >= low) & (frequencies < high)
        if not np.any(mask):
            output.append(np.zeros(spectrum.shape[1], dtype=np.float64))
        else:
            output.append(np.sum(spectrum[mask], axis=0))
    return np.asarray(output, dtype=np.float64).T


def _stereo_ratio(audio: np.ndarray, frame: int = 2048, hop: int = 512) -> np.ndarray:
    left = audio[:, 0].astype(np.float64)
    right = audio[:, 1].astype(np.float64)
    mid = (left + right) / np.sqrt(2.0)
    side = (left - right) / np.sqrt(2.0)
    if len(mid) < frame:
        pad = frame - len(mid)
        mid = np.pad(mid, (0, pad))
        side = np.pad(side, (0, pad))
    ratios = []
    for start in range(0, len(mid) - frame + 1, hop):
        m = np.sqrt(np.mean(mid[start : start + frame] ** 2) + 1e-10)
        s = np.sqrt(np.mean(side[start : start + frame] ** 2) + 1e-10)
        ratios.append(s / m)
    return np.asarray(ratios, dtype=np.float64)


def _onset_similarity(a: np.ndarray, b: np.ndarray, sample_rate: int) -> float:
    a_mono = np.mean(a, axis=1, dtype=np.float64).astype(np.float32)
    b_mono = np.mean(b, axis=1, dtype=np.float64).astype(np.float32)
    hop = 512
    oa = librosa.onset.onset_strength(y=a_mono, sr=sample_rate, hop_length=hop)
    ob = librosa.onset.onset_strength(y=b_mono, sr=sample_rate, hop_length=hop)
    length = min(len(oa), len(ob))
    if length < 4:
        return 0.55
    oa = oa[:length]
    ob = ob[:length]
    oa = (oa - np.mean(oa)) / max(float(np.std(oa)), 1e-6)
    ob = (ob - np.mean(ob)) / max(float(np.std(ob)), 1e-6)
    correlation = float(np.mean(oa * ob))
    return float(np.clip(0.5 + 0.5 * correlation, 0.0, 1.0))




def _sanitize_positions(values: np.ndarray | Sequence[float] | None, length: int) -> np.ndarray:
    if values is None:
        return np.zeros(0, dtype=np.float64)
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    arr = np.unique(arr[(arr >= 0.0) & (arr < float(length))])
    return np.ascontiguousarray(arr, dtype=np.float64)


def _best_beat_pairing(
    a_positions: np.ndarray,
    b_positions: np.ndarray,
    max_index_shift: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Pair beat grids while tolerating a tracker starting one beat early/late.

    Human DJs listen for the same count position rather than forcing the first
    detected transient to be "beat one".  Searching a tiny index neighbourhood
    handles a downbeat label that is off by one without accepting a full-bar
    phase error.
    """
    if a_positions.size == 0 or b_positions.size == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    for shift in range(-max_index_shift, max_index_shift + 1):
        a_start = max(0, shift)
        b_start = max(0, -shift)
        count = min(a_positions.size - a_start, b_positions.size - b_start)
        if count < 2:
            continue
        aa = a_positions[a_start : a_start + count]
        bb = b_positions[b_start : b_start + count]
        residual = bb - aa
        # Prefer a pairing that already has a small absolute phase error and a
        # stable interval. A one-beat-wrong pairing is approximately 500 ms off.
        score = float(np.median(np.abs(residual))) + 0.35 * float(
            np.median(np.abs(np.diff(residual))) if residual.size > 1 else 0.0
        )
        if best is None or score < best[0]:
            best = (score, aa, bb)
    if best is None:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    return best[1], best[2]


def _kick_onset_envelope(low: np.ndarray, perc: np.ndarray, sample_rate: int) -> np.ndarray:
    """Low-frequency kick attack envelope with a small broadband transient cue."""
    length = min(len(low), len(perc))
    if length == 0:
        return np.zeros(0, dtype=np.float64)
    low_mono = np.mean(low[:length], axis=1, dtype=np.float64)
    perc_mono = np.mean(perc[:length], axis=1, dtype=np.float64)
    nyquist = sample_rate / 2.0
    try:
        lo = max(20.0 / nyquist, 1e-4)
        hi = min(190.0 / nyquist, 0.96)
        if hi > lo:
            sos = signal.butter(3, [lo, hi], btype="bandpass", output="sos")
            low_mono = signal.sosfiltfilt(sos, low_mono)
    except ValueError:
        pass
    energy = low_mono * low_mono
    energy_smooth = max(1, int(round(0.009 * sample_rate)))
    kernel = np.ones(energy_smooth, dtype=np.float64) / energy_smooth
    energy = np.convolve(energy, kernel, mode="same")
    kick_attack = np.maximum(np.diff(energy, prepend=energy[0]), 0.0)

    perc_attack = np.abs(np.diff(perc_mono, prepend=perc_mono[0]))
    perc_smooth = max(1, int(round(0.003 * sample_rate)))
    p_kernel = np.ones(perc_smooth, dtype=np.float64) / perc_smooth
    perc_attack = np.convolve(perc_attack, p_kernel, mode="same")

    def robust_norm(x: np.ndarray) -> np.ndarray:
        scale = float(np.percentile(x, 95.0) + 1e-12)
        return np.clip(x / scale, 0.0, 4.0)

    return 0.82 * robust_norm(kick_attack) + 0.18 * robust_norm(perc_attack)


def _weighted_line_fit(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Small robust linear fit used as a deck pitch/nudge model."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = np.maximum(np.asarray(w, dtype=np.float64), 1e-6)
    if x.size == 0:
        return 0.0, 0.0, np.zeros(0, dtype=np.float64)
    if x.size == 1 or float(np.ptp(x)) < 1e-9:
        return float(y[0]), 0.0, np.ones_like(w)
    design = np.column_stack([np.ones_like(x), x])
    robust = w.copy()
    beta = np.zeros(2, dtype=np.float64)
    for _ in range(4):
        root = np.sqrt(np.maximum(robust, 1e-9))
        beta, *_ = np.linalg.lstsq(design * root[:, None], y * root, rcond=None)
        resid = y - design @ beta
        mad = float(np.median(np.abs(resid - np.median(resid))) + 1e-9)
        cutoff = 2.5 * 1.4826 * mad + 1.0
        huber = np.ones_like(resid)
        mask = np.abs(resid) > cutoff
        huber[mask] = cutoff / np.maximum(np.abs(resid[mask]), 1e-9)
        robust = w * huber
    return float(beta[0]), float(beta[1]), robust


def _apply_source_map(audio: np.ndarray, source: np.ndarray, length: int) -> np.ndarray:
    out = np.empty((length, audio.shape[1]), dtype=np.float32)
    x = np.arange(length, dtype=np.float64)
    for channel in range(audio.shape[1]):
        out[:, channel] = np.interp(source, x, audio[:length, channel]).astype(np.float32)
    return np.ascontiguousarray(out, dtype=np.float32)


def lock_incoming_deck_phase(
    *,
    a_low: np.ndarray,
    b_low: np.ndarray,
    a_harm: np.ndarray,
    b_harm: np.ndarray,
    a_perc: np.ndarray,
    b_perc: np.ndarray,
    sample_rate: int,
    bpm: float,
    beats_per_bar: int,
    bars: int,
    a_beat_positions: np.ndarray | Sequence[float] | None = None,
    b_beat_positions: np.ndarray | Sequence[float] | None = None,
    a_downbeat_positions: np.ndarray | Sequence[float] | None = None,
    b_downbeat_positions: np.ndarray | Sequence[float] | None = None,
    initial_offset_samples: float = 0.0,
    max_nudge_ms: float = 48.0,
    max_bar_step_ms: float = 14.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """DJ-style deck phase lock using phrase beats and smooth bar-level nudges.

    Real beatmatching changes the phase/tempo of the *whole incoming deck*.
    It does not independently drag every kick transient.  This function therefore
    estimates a robust phase trajectory from the detected beat grids and kick
    attacks, then applies the same monotonic source map to bass, harmonic and
    percussion layers.  Corrections are only allowed to change at bar scale.
    """
    length = min(len(a_low), len(b_low), len(a_harm), len(b_harm), len(a_perc), len(b_perc))
    empty_metrics = {
        "beat_grid_alignment_score": 0.5,
        "beat_grid_align_ms": 0.0,
        "beat_grid_aligned_beats": 0.0,
        "beat_lock_pre_error_ms": 0.0,
        "beat_lock_post_error_ms": 0.0,
        "beat_lock_confidence": 0.0,
        "beat_lock_bar_nudges": 0.0,
        "beat_lock_terminal_shift_samples": 0.0,
    }
    if length < 512 or bpm <= 1.0:
        return b_low, b_harm, b_perc, empty_metrics

    beat_samples = float(sample_rate) * 60.0 / float(bpm)
    nominal_count = min(
        max(2, int(bars) * max(1, int(beats_per_bar)) + 1),
        int(length / beat_samples) + 2,
    )
    nominal = np.arange(nominal_count, dtype=np.float64) * beat_samples
    nominal = nominal[nominal < length]

    a_grid = _sanitize_positions(a_beat_positions, length)
    b_grid = _sanitize_positions(b_beat_positions, length)
    if a_grid.size < 2:
        a_grid = nominal
    if b_grid.size < 2:
        b_grid = nominal + float(initial_offset_samples)
        b_grid = b_grid[(b_grid >= 0.0) & (b_grid < length)]
    aa, bb = _best_beat_pairing(a_grid, b_grid)
    if aa.size < 2:
        return b_low, b_harm, b_perc, empty_metrics

    a_down = _sanitize_positions(a_downbeat_positions, length)
    b_down = _sanitize_positions(b_downbeat_positions, length)
    onset_a = _kick_onset_envelope(a_low[:length], a_perc[:length], sample_rate)
    onset_b = _kick_onset_envelope(b_low[:length], b_perc[:length], sample_rate)
    search = max(8, int(round(0.060 * sample_rate)))
    maximum = max(1, int(round(max_nudge_ms * sample_rate / 1000.0)))

    obs_x: list[float] = []
    obs_y: list[float] = []
    obs_w: list[float] = []
    acoustic_residuals: list[float] = []
    acoustic_weights: list[float] = []

    def is_downbeat(pos: float, grid: np.ndarray) -> bool:
        return bool(grid.size and np.min(np.abs(grid - pos)) <= max(0.035 * sample_rate, 1.0))

    for pa, pb in zip(aa, bb):
        center = float(0.5 * (pa + pb))
        grid_residual = float(np.clip(pb - pa, -maximum, maximum))
        downbeat_weight = 1.8 if is_downbeat(pa, a_down) and is_downbeat(pb, b_down) else 1.0
        obs_x.append(center)
        obs_y.append(grid_residual)
        obs_w.append(0.38 * downbeat_weight)

        ia = int(round(pa))
        ib = int(round(pb))
        alo, ahi = max(0, ia - search), min(length, ia + search + 1)
        blo, bhi = max(0, ib - search), min(length, ib + search + 1)
        if ahi - alo < 8 or bhi - blo < 8:
            continue
        peak_a = alo + int(np.argmax(onset_a[alo:ahi]))
        peak_b = blo + int(np.argmax(onset_b[blo:bhi]))
        floor_a = float(np.median(onset_a[alo:ahi]) + 1e-9)
        floor_b = float(np.median(onset_b[blo:bhi]) + 1e-9)
        prominence = float(np.clip(min(onset_a[peak_a] / floor_a, onset_b[peak_b] / floor_b) / 12.0, 0.0, 1.0))
        residual = float(np.clip(peak_b - peak_a, -maximum, maximum))
        if prominence >= 0.16:
            weight = (0.75 + 1.45 * prominence) * downbeat_weight
            obs_x.append(center)
            obs_y.append(residual)
            obs_w.append(weight)
            acoustic_residuals.append(residual)
            acoustic_weights.append(weight)

    if len(obs_y) < 2:
        return b_low, b_harm, b_perc, empty_metrics

    x = np.asarray(obs_x, dtype=np.float64)
    y = np.asarray(obs_y, dtype=np.float64)
    w = np.asarray(obs_w, dtype=np.float64)
    intercept, slope, robust_w = _weighted_line_fit(x, y, w)
    intercept = float(np.clip(intercept, -maximum, maximum))
    total_drift_limit = max(1.0, int(round(0.040 * sample_rate)))
    end_value = intercept + slope * max(length - 1, 1)
    end_value = float(np.clip(end_value, -maximum - total_drift_limit, maximum + total_drift_limit))
    slope = (end_value - intercept) / max(length - 1, 1)

    # Manual DJs nudge the deck occasionally, not on every kick.  Estimate one
    # correction per bar, robustly smooth it, then interpolate between bars.
    beats_per_bar = max(1, int(beats_per_bar))
    bar_samples = beat_samples * beats_per_bar
    bar_count = max(1, int(np.ceil(length / max(bar_samples, 1.0))))
    bar_x = [0.0]
    bar_y = [intercept]
    for bar_index in range(1, bar_count + 1):
        center = min(float(length - 1), bar_index * bar_samples)
        mask = np.abs(x - center) <= 0.72 * bar_samples
        if np.any(mask):
            local_y = y[mask]
            local_w = robust_w[mask]
            order = np.argsort(local_y)
            ly = local_y[order]
            lw = local_w[order]
            cumulative = np.cumsum(lw)
            target_w = 0.5 * cumulative[-1]
            value = float(ly[min(int(np.searchsorted(cumulative, target_w)), len(ly) - 1)])
            model_value = intercept + slope * center
            value = 0.72 * value + 0.28 * model_value
        else:
            value = intercept + slope * center
        bar_x.append(center)
        bar_y.append(float(np.clip(value, -maximum - total_drift_limit, maximum + total_drift_limit)))

    bar_y_arr = np.asarray(bar_y, dtype=np.float64)
    if bar_y_arr.size >= 3:
        bar_y_arr = signal.medfilt(bar_y_arr, kernel_size=3)
    max_bar_step = max(1.0, max_bar_step_ms * sample_rate / 1000.0)
    for i in range(1, len(bar_y_arr)):
        bar_y_arr[i] = np.clip(bar_y_arr[i], bar_y_arr[i - 1] - max_bar_step, bar_y_arr[i - 1] + max_bar_step)
    for i in range(len(bar_y_arr) - 2, -1, -1):
        bar_y_arr[i] = np.clip(bar_y_arr[i], bar_y_arr[i + 1] - max_bar_step, bar_y_arr[i + 1] + max_bar_step)

    target = np.arange(length, dtype=np.float64)
    correction = np.interp(target, np.asarray(bar_x, dtype=np.float64), bar_y_arr)
    correction = np.clip(correction, -maximum - total_drift_limit, maximum + total_drift_limit)

    # The renderer must reconnect to the unmodified incoming track. Keep the
    # deck locked while both tracks are audible, then release the nudge over the
    # final ~1.25 beats after deck A is normally almost gone. This avoids the
    # frozen-edge artifact caused by clipping a non-zero terminal source offset.
    release = int(round(min(1.25 * beat_samples, 0.18 * length)))
    release = max(32, min(release, max(32, length - 1)))
    release_start = max(0, length - release)
    if release_start < length - 1:
        release_phase = np.clip(
            (target - release_start) / max(length - 1 - release_start, 1), 0.0, 1.0
        )
        release_curve = 1.0 - (
            release_phase**3 * (release_phase * (release_phase * 6.0 - 15.0) + 10.0)
        )
        correction *= release_curve
    correction[-1] = 0.0
    source = np.clip(target + correction, 0.0, length - 1.0)
    # Monotonic source positions are required for a click-free deck nudge.
    source = np.maximum.accumulate(source)
    source[-1] = length - 1.0

    active_obs = x <= max(0.0, release_start - 0.25 * beat_samples)
    if not np.any(active_obs):
        active_obs = np.ones_like(x, dtype=bool)
    before_error = np.abs(y[active_obs])
    after_error = np.abs(
        y[active_obs] - np.interp(x[active_obs], target, correction)
    )
    active_weights = np.maximum(robust_w[active_obs], 1e-9)

    def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
        order = np.argsort(values)
        sorted_values = values[order]
        sorted_weights = weights[order]
        cumulative = np.cumsum(sorted_weights)
        index = min(
            int(np.searchsorted(cumulative, 0.5 * cumulative[-1])),
            len(sorted_values) - 1,
        )
        return float(sorted_values[index])

    pre_ms = 1000.0 * weighted_median(before_error, active_weights) / sample_rate
    post_ms = 1000.0 * weighted_median(after_error, active_weights) / sample_rate
    if acoustic_residuals:
        spread = float(np.median(np.abs(np.asarray(acoustic_residuals) - np.median(acoustic_residuals))))
        prominence_conf = float(np.average(np.clip(np.asarray(acoustic_weights) / 3.0, 0.0, 1.0)))
    else:
        spread = maximum
        prominence_conf = 0.0
    consistency = float(np.clip(1.0 - spread / max(maximum, 1), 0.0, 1.0))
    confidence = float(np.clip(0.35 + 0.40 * consistency + 0.25 * prominence_conf, 0.0, 1.0))
    score = float(np.clip(0.55 + 0.30 * confidence + 0.15 * (1.0 - min(post_ms / 35.0, 1.0)), 0.0, 1.0))
    weighted_ms = 1000.0 * float(np.average(np.abs(correction), weights=np.linspace(1.2, 0.8, length))) / sample_rate

    metrics = {
        "beat_grid_alignment_score": score,
        "beat_grid_align_ms": weighted_ms,
        "beat_grid_aligned_beats": float(len(aa)),
        "beat_lock_pre_error_ms": pre_ms,
        "beat_lock_post_error_ms": post_ms,
        "beat_lock_confidence": confidence,
        "beat_lock_bar_nudges": float(max(0, len(bar_x) - 1)),
        "beat_lock_terminal_shift_samples": 0.0,
        "beat_lock_release_start": float(release_start),
    }
    return (
        _apply_source_map(b_low, source, length),
        _apply_source_map(b_harm, source, length),
        _apply_source_map(b_perc, source, length),
        metrics,
    )


def align_percussive_beat_grid(
    *,
    a_low: np.ndarray,
    b_low: np.ndarray,
    a_perc: np.ndarray,
    b_perc: np.ndarray,
    sample_rate: int,
    bpm: float,
    beats_per_bar: int,
    bars: int,
    max_shift_ms: float = 42.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Backward-compatible wrapper around the whole-deck phase lock.

    Existing callers/tests receive low and percussion outputs, while the real
    engine uses :func:`lock_incoming_deck_phase` so harmonic content follows the
    exact same nudge map.
    """
    zeros_a = np.zeros_like(a_perc, dtype=np.float32)
    zeros_b = np.zeros_like(b_perc, dtype=np.float32)
    out_low, _, out_perc, metrics = lock_incoming_deck_phase(
        a_low=a_low,
        b_low=b_low,
        a_harm=zeros_a,
        b_harm=zeros_b,
        a_perc=a_perc,
        b_perc=b_perc,
        sample_rate=sample_rate,
        bpm=bpm,
        beats_per_bar=beats_per_bar,
        bars=bars,
        max_nudge_ms=max_shift_ms,
    )
    return out_low, out_perc, metrics

def _delay_echo(
    audio: np.ndarray,
    sample_rate: int,
    bpm: float,
    send_curve: np.ndarray,
    feedback: float,
) -> np.ndarray:
    result = np.zeros_like(audio, dtype=np.float32)
    if audio.size == 0:
        return result
    # One-beat and half-beat taps; the second tap prevents a sterile single delay.
    beat = max(1, int(round(60.0 / max(float(bpm), 1.0) * sample_rate)))
    dry_send = audio * send_curve[:, None]
    for tap, gain in ((beat // 2, feedback * 0.72), (beat, feedback), (beat * 2, feedback**2 * 0.72)):
        if tap <= 0 or tap >= len(audio):
            continue
        result[tap:] += dry_send[:-tap] * np.float32(gain)
    # Remove low end from echo tails, as a DJ would usually avoid feeding bass into delay.
    try:
        sos = signal.butter(3, 260.0, btype="highpass", fs=sample_rate, output="sos")
        result = signal.sosfilt(sos, result, axis=0).astype(np.float32)
    except Exception:
        pass
    return result


def _loop_percussion(
    percussive: np.ndarray,
    sample_rate: int,
    bpm: float,
    beats_per_bar: int,
    phase: np.ndarray,
    start: float,
    end: float,
) -> np.ndarray:
    output = np.zeros_like(percussive, dtype=np.float32)
    bar = max(1, int(round(60.0 / max(float(bpm), 1.0) * beats_per_bar * sample_rate)))
    region_start = int(np.clip(start, 0.0, 1.0) * len(percussive))
    region_end = int(np.clip(end, 0.0, 1.0) * len(percussive))
    if region_end <= region_start or region_start < bar:
        return output
    source = percussive[max(0, region_start - bar) : region_start]
    if len(source) < max(32, bar // 2):
        return output
    cursor = region_start
    while cursor < region_end:
        count = min(len(source), region_end - cursor)
        output[cursor : cursor + count] += source[:count]
        cursor += count
    decay = 1.0 - 0.68 * _smootherstep((phase - start) / max(end - start, 1e-4))
    gate = (_ramp(phase, start, start + 0.02) * (1.0 - _ramp(phase, end - 0.02, end))).astype(np.float32)
    return output * (decay * gate)[:, None]


def _intent_bonus(archetype: str, metrics: dict[str, float]) -> float:
    code = int(round(float(metrics.get("dj_intent_code", 0.0))))
    preferred = {
        1: ("Long Blend", "Bass Swap", "Filter Ride"),
        2: ("Post-Drop Relay", "Bass Swap", "Long Blend"),
        3: ("Breakdown Lift", "Filter Ride", "Drop Swap"),
        4: ("Breakdown Lift", "Drop Swap", "Filter Ride"),
        5: ("Double Drop", "Drop Swap", "Bass Swap"),
        6: ("Echo Out", "Loop Out", "Filter Ride"),
    }.get(code, ("Long Blend", "Bass Swap", "Echo Out"))
    if archetype == preferred[0]:
        return 0.36 if archetype == "Double Drop" else 0.24
    if archetype in preferred[1:]:
        return 0.12
    return 0.0


def _context_style_fit(
    archetype: str,
    current_label: str,
    next_label: str,
    metrics: dict[str, float],
) -> float:
    edm = float(np.clip(metrics.get("edm_confidence", 0.5), 0.0, 1.0))
    vocal_risk = float(np.clip(1.0 - metrics.get("vocal_clean", 0.5), 0.0, 1.0))
    harmonic = float(np.clip(metrics.get("harmonic", 0.5), 0.0, 1.0))
    cue = float(np.clip(metrics.get("cue_alignment", 0.5), 0.0, 1.0))
    phrase = float(np.clip(metrics.get("phrase_alignment", 0.5), 0.0, 1.0))
    current_role = current_label.upper()
    next_role = next_label.upper()
    next_drop = next_role in {"DROP", "CHORUS"}
    current_break = current_role in {"BREAK", "BREAKDOWN", "BRIDGE", "OUTRO"}
    current_vocal = current_role in {"VOCAL", "VERSE", "CHORUS", "SOLO"}
    clean_intro = next_role in {"INTRO", "PHRASE", "SECTION"}

    intent = _intent_bonus(archetype, metrics)
    if archetype == "Long Blend":
        return float(np.clip(0.25 + 0.28 * harmonic + 0.18 * phrase + 0.16 * (1.0 - vocal_risk) + 0.13 * float(clean_intro) + intent, 0.0, 1.0))
    if archetype == "Bass Swap":
        return float(np.clip(0.16 + 0.42 * edm + 0.16 * cue + 0.14 * phrase + 0.12 * float(clean_intro) + intent, 0.0, 1.0))
    if archetype == "Echo Out":
        return float(np.clip(0.22 + 0.25 * vocal_risk + 0.18 * cue + 0.16 * (1.0 - harmonic) + 0.19 * float(current_vocal) + intent, 0.0, 1.0))
    if archetype == "Drop Swap":
        return float(np.clip(0.10 + 0.46 * float(next_drop) + 0.20 * edm + 0.15 * phrase + 0.09 * float(current_break) + intent, 0.0, 1.0))
    if archetype == "Loop Out":
        return float(np.clip(0.17 + 0.28 * edm + 0.21 * cue + 0.25 * float(current_break) + 0.09 * float(clean_intro) + intent, 0.0, 1.0))
    if archetype == "Filter Ride":
        return float(np.clip(0.20 + 0.24 * edm + 0.22 * harmonic + 0.20 * cue + 0.14 * float(current_vocal) + intent, 0.0, 1.0))
    if archetype == "Post-Drop Relay":
        return float(np.clip(0.28 + 0.28 * metrics.get("dj_post_drop", 0.0) + 0.24 * metrics.get("dj_drop_landing", 0.0) + 0.12 * phrase + 0.08 * edm + intent, 0.0, 1.0))
    if archetype == "Breakdown Lift":
        return float(np.clip(0.24 + 0.30 * metrics.get("dj_drop_landing", 0.0) + 0.20 * metrics.get("dj_energy_arc", 0.0) + 0.14 * phrase + 0.12 * edm + intent, 0.0, 1.0))
    if archetype == "Double Drop":
        return float(np.clip(0.12 + 0.28 * metrics.get("dj_drop_landing", 0.0) + 0.22 * harmonic + 0.18 * metrics.get("bass_clean", 0.0) + 0.12 * phrase + 0.08 * edm + intent, 0.0, 1.0))
    return 0.5


def _candidate_order(
    mode: str,
    current_label: str,
    next_label: str,
    metrics: dict[str, float],
    history: Sequence[str],
    maximum: int,
) -> list[str]:
    if mode in SUPPORTED_ARCHETYPES:
        return [mode]

    scored = []
    for archetype in SUPPORTED_ARCHETYPES:
        fit = _context_style_fit(archetype, current_label, next_label, metrics)
        # Recent repetition is not forbidden, but receives a clear penalty. This
        # addresses the recognisability problem of repeatedly applying one rule.
        recent_penalty = 0.0
        if archetype in tuple(history)[-2:]:
            recent_penalty = 0.16 if history and archetype == history[-1] else 0.08
        if int(round(float(metrics.get("dj_intent_code", 0.0)))) == 5 and archetype == "Double Drop":
            fit += 0.16
            recent_penalty *= 0.35
        scored.append((fit - recent_penalty, archetype))
    scored.sort(reverse=True)
    return [name for _, name in scored[: max(1, int(maximum))]]


def _render_archetype(
    archetype: str,
    *,
    a_low: np.ndarray,
    b_low: np.ndarray,
    a_harm: np.ndarray,
    b_harm: np.ndarray,
    a_perc: np.ndarray,
    b_perc: np.ndarray,
    sample_rate: int,
    bpm: float,
    beats_per_bar: int,
    bars: int,
    local_gain: float,
    effect_strength: float,
    vocal_risk: float,
    drop_landing_phase: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    length = min(len(a_low), len(b_low), len(a_harm), len(b_harm), len(a_perc), len(b_perc))
    phase = np.linspace(0.0, 1.0, length, endpoint=True, dtype=np.float32)
    strength = float(np.clip(effect_strength, 0.0, 1.0))
    bars = max(1, int(bars))

    swap = _bar_quantized(0.54, bars, subdivision=2)
    bass_width = max(0.5 / bars, 0.025)
    drum_in_start = _bar_quantized(0.02, bars, subdivision=4)
    harm_in_start = _bar_quantized(0.18 + 0.22 * vocal_risk, bars, subdivision=2)
    harm_out_end = _bar_quantized(0.78 - 0.12 * vocal_risk, bars, subdivision=2)
    echo_send = np.zeros(length, dtype=np.float32)
    loop_audio = np.zeros_like(a_perc, dtype=np.float32)
    impact_duck = np.ones(length, dtype=np.float32)

    if archetype == "Long Blend":
        swap = _bar_quantized(0.58, bars, subdivision=2)
        bass_width = max(2.0 / bars, 0.08)
        drum_in_start = 0.0
        harm_in_start = _bar_quantized(0.10 + 0.16 * vocal_risk, bars, 2)
        harm_out_end = _bar_quantized(0.92 - 0.10 * vocal_risk, bars, 2)
    elif archetype == "Bass Swap":
        swap = _bar_quantized(0.50, bars, subdivision=2)
        bass_width = max(0.55 / bars, 0.025)
        drum_in_start = 0.0
        harm_in_start = _bar_quantized(0.28 + 0.16 * vocal_risk, bars, 2)
        harm_out_end = _bar_quantized(0.78, bars, 2)
    elif archetype == "Echo Out":
        swap = _bar_quantized(0.56, bars, subdivision=2)
        bass_width = max(1.0 / bars, 0.05)
        drum_in_start = _bar_quantized(0.08, bars, 2)
        harm_in_start = _bar_quantized(0.55 + 0.12 * vocal_risk, bars, 2)
        harm_out_end = _bar_quantized(0.70, bars, 2)
        echo_start = _bar_quantized(0.48, bars, 2)
        echo_send = _ramp(phase, echo_start, min(0.92, echo_start + 0.30))
    elif archetype == "Drop Swap":
        swap = _bar_quantized(0.72, bars, subdivision=2)
        bass_width = max(0.22 / bars, 0.012)
        drum_in_start = _bar_quantized(0.42, bars, 2)
        harm_in_start = max(0.0, swap - max(0.15 / bars, 0.008))
        harm_out_end = swap
        beat_fraction = 1.0 / max(bars * beats_per_bar, 1)
        gap_start = max(0.0, swap - 0.45 * beat_fraction)
        gap_end = min(1.0, swap + 0.12 * beat_fraction)
        impact_duck = 1.0 - (0.56 + 0.18 * strength) * (
            _ramp(phase, gap_start, swap, smoother=False)
            * (1.0 - _ramp(phase, swap, gap_end, smoother=False))
        )
    elif archetype == "Loop Out":
        swap = _bar_quantized(0.58, bars, subdivision=2)
        bass_width = max(0.75 / bars, 0.035)
        drum_in_start = 0.0
        harm_in_start = _bar_quantized(0.42 + 0.10 * vocal_risk, bars, 2)
        harm_out_end = _bar_quantized(0.56, bars, 2)
        loop_start = _bar_quantized(0.50, bars, 2)
        loop_end = _bar_quantized(0.88, bars, 2)
        loop_audio = _loop_percussion(
            a_perc, sample_rate, bpm, beats_per_bar, phase, loop_start, loop_end
        )
    elif archetype == "Filter Ride":
        swap = _bar_quantized(0.62, bars, subdivision=2)
        bass_width = max(1.0 / bars, 0.045)
        drum_in_start = 0.0
        harm_in_start = _bar_quantized(0.30 + 0.12 * vocal_risk, bars, 2)
        harm_out_end = _bar_quantized(0.80, bars, 2)
    elif archetype == "Post-Drop Relay":
        # The outgoing drop has finished: keep its groove briefly, introduce the
        # incoming drums early, then hand bass ownership over around the final
        # third so the incoming drop/chorus can arrive cleanly.
        swap = _bar_quantized(0.66, bars, subdivision=2)
        bass_width = max(0.65 / bars, 0.025)
        drum_in_start = 0.0
        harm_in_start = _bar_quantized(0.46 + 0.10 * vocal_risk, bars, 2)
        harm_out_end = _bar_quantized(0.64, bars, 2)
    elif archetype == "Breakdown Lift":
        # Preserve the breakdown atmosphere while the incoming buildup rises.
        # A short pre-drop hole creates a perceptual arrival instead of a mushy
        # full-level overlap.
        swap = _bar_quantized(0.84, bars, subdivision=4)
        bass_width = max(0.30 / bars, 0.012)
        drum_in_start = _bar_quantized(0.26, bars, 2)
        harm_in_start = _bar_quantized(0.20, bars, 2)
        harm_out_end = _bar_quantized(0.84, bars, 2)
        beat_fraction = 1.0 / max(bars * beats_per_bar, 1)
        hole_start = max(0.0, swap - 0.72 * beat_fraction)
        hole_end = min(1.0, swap + 0.10 * beat_fraction)
        impact_duck = 1.0 - (0.46 + 0.22 * strength) * (
            _ramp(phase, hole_start, swap, smoother=False)
            * (1.0 - _ramp(phase, swap, hole_end, smoother=False))
        )
    elif archetype == "Double Drop":
        # Land both drops on the same downbeat. Incoming drums become audible
        # first; the outgoing drums stay present for roughly three quarters of
        # a beat, then release over the following beat like a human deck handoff.
        beat_fraction = float(np.clip((60.0 / max(bpm, 1.0) * sample_rate) / max(length, 1), 1e-4, 0.25))
        swap = float(np.clip(drop_landing_phase, 0.50, 0.94))
        bass_width = max(0.24 * beat_fraction, 0.006)
        drum_in_start = max(0.0, swap - 0.55 * beat_fraction)
        harm_in_start = max(0.0, swap - 0.30 * beat_fraction)
        harm_out_end = min(1.0, swap + 1.55 * beat_fraction)
        gap_start = max(0.0, swap - 0.28 * beat_fraction)
        gap_end = min(1.0, swap + 0.06 * beat_fraction)
        # Only a small pre-impact pocket; do not erase the intended double hit.
        impact_duck = 1.0 - (0.10 + 0.08 * strength) * (
            _ramp(phase, gap_start, swap, smoother=False)
            * (1.0 - _ramp(phase, swap, gap_end, smoother=False))
        )

    bass_progress = _ramp(phase, swap - bass_width / 2.0, swap + bass_width / 2.0)
    bass_a, bass_b = _equal_power(bass_progress)

    drum_progress = _ramp(phase, drum_in_start, min(0.82, drum_in_start + 0.42))
    # Let deck B establish a groove before reducing deck A.
    drum_a_fade_start = max(0.30, drum_in_start + 0.22, swap - 0.10)
    drum_a_progress = _ramp(phase, drum_a_fade_start, min(1.0, drum_a_fade_start + 0.34))
    drum_a = np.sqrt(np.clip(1.0 - drum_a_progress, 0.0, 1.0)).astype(np.float32)
    drum_b = np.sqrt(np.clip(drum_progress, 0.0, 1.0)).astype(np.float32)
    if archetype == "Post-Drop Relay":
        drum_a = np.sqrt(np.clip(1.0 - _ramp(phase, 0.30, 0.74), 0.0, 1.0)).astype(np.float32)
        drum_b = np.sqrt(np.clip(_ramp(phase, 0.0, 0.44), 0.0, 1.0)).astype(np.float32)
    elif archetype == "Breakdown Lift":
        drum_a *= (1.0 - 0.34 * _ramp(phase, 0.55, 0.90)).astype(np.float32)
        drum_b = np.sqrt(np.clip(_ramp(phase, drum_in_start, swap), 0.0, 1.0)).astype(np.float32)
    elif archetype == "Double Drop":
        beat_fraction = float(np.clip((60.0 / max(bpm, 1.0) * sample_rate) / max(length, 1), 1e-4, 0.25))
        b_entry = _ramp(phase, swap - 0.55 * beat_fraction, swap, smoother=False)
        b_full = _ramp(phase, swap + 0.45 * beat_fraction, swap + 1.05 * beat_fraction)
        drum_b = (0.78 * b_entry + 0.22 * b_full).astype(np.float32)
        a_pretrim = 1.0 - 0.12 * _ramp(phase, swap - 0.55 * beat_fraction, swap, smoother=False)
        a_release = _ramp(phase, swap + 0.72 * beat_fraction, swap + 1.72 * beat_fraction)
        drum_a = (a_pretrim * np.sqrt(np.clip(1.0 - a_release, 0.0, 1.0))).astype(np.float32)

    harm_b_progress = _ramp(phase, harm_in_start, min(1.0, harm_in_start + 0.48))
    harm_a_progress = _ramp(phase, max(0.0, harm_out_end - 0.48), harm_out_end)
    harm_a, _ = _equal_power(harm_a_progress)
    _, harm_b = _equal_power(harm_b_progress)

    if archetype == "Filter Ride":
        # Approximate a long filter ride by progressively transferring harmonic
        # energy while keeping percussion stable. The offline renderer already
        # works on role-separated signals, so this avoids time-varying IIR state.
        ride = _smootherstep((phase - 0.10) / 0.78)
        harm_a *= 1.0 - (0.48 + 0.32 * strength) * ride
        harm_b *= 0.24 + 0.76 * _smootherstep((phase - 0.12) / 0.58)

    gain_release = float(local_gain) + (1.0 - float(local_gain)) * _smootherstep(phase)
    b_gain = gain_release.astype(np.float32)

    a_contribution = (
        a_low[:length] * bass_a[:, None]
        + a_harm[:length] * harm_a[:, None]
        + a_perc[:length] * drum_a[:, None]
    )
    b_contribution = (
        b_low[:length] * bass_b[:, None]
        + b_harm[:length] * harm_b[:, None]
        + b_perc[:length] * drum_b[:, None]
    ) * b_gain[:, None]

    if archetype == "Loop Out":
        # Replace, rather than simply stack, some outgoing percussion with the
        # loop to keep the groove controlled.
        loop_presence = np.max(np.abs(loop_audio), axis=1) > 1e-8
        a_contribution[loop_presence] *= np.float32(0.45)
        a_contribution += loop_audio * np.float32(0.72 + 0.18 * strength)

    echo = np.zeros_like(a_contribution, dtype=np.float32)
    if archetype == "Echo Out":
        echo = _delay_echo(
            a_harm[:length],
            sample_rate=sample_rate,
            bpm=bpm,
            send_curve=echo_send,
            feedback=0.34 + 0.18 * strength,
        )

    # Deliberate center space. Humans usually lower one deck before introducing
    # the other, rather than summing two full-level sources at equal power.
    center_space = 1.0 - (0.055 + 0.095 * strength) * np.sin(np.pi * phase) ** 2
    mixed = (a_contribution + b_contribution) * center_space[:, None]
    mixed = mixed * impact_duck[:, None] + echo
    mixed = _soft_limit(mixed)

    controls = {
        "bass_a": bass_a,
        "bass_b": bass_b,
        "drum_a": drum_a,
        "drum_b": drum_b,
        "drum_overlap": np.minimum(drum_a, drum_b).astype(np.float32),
        "harm_a": harm_a,
        "harm_b": harm_b,
        "impact_duck": impact_duck,
    }
    return (
        mixed,
        np.ascontiguousarray(a_contribution, dtype=np.float32),
        np.ascontiguousarray(b_contribution, dtype=np.float32),
        controls,
    )


def evaluate_transition(
    *,
    mixed: np.ndarray,
    deck_a: np.ndarray,
    deck_b: np.ndarray,
    sample_rate: int,
    controls: dict[str, np.ndarray],
    archetype: str,
    context_fit: float,
    recent_history: Sequence[str],
    vocal_risk: float,
    evaluation_sample_rate: int = 16_000,
) -> dict[str, float]:
    """Return interpretable [0,1] scores; higher is better."""
    target_sr = int(min(sample_rate, max(8_000, evaluation_sample_rate)))
    m = _resample_for_eval(mixed, sample_rate, target_sr)
    a = _resample_for_eval(deck_a, sample_rate, target_sr)
    b = _resample_for_eval(deck_b, sample_rate, target_sr)
    length = min(len(m), len(a), len(b))
    m, a, b = m[:length], a[:length], b[:length]

    # 1. Loudness and conservative true-peak headroom.
    rms_m = _frame_rms(m)
    rms_a = _frame_rms(a)
    rms_b = _frame_rms(b)
    frames = min(len(rms_m), len(rms_a), len(rms_b))
    rms_m, rms_a, rms_b = rms_m[:frames], rms_a[:frames], rms_b[:frames]
    delta_db = _db(rms_m) - _db(np.maximum(rms_a, rms_b))
    excess = np.maximum(delta_db - 1.25, 0.0)
    hole = np.maximum(-4.5 - delta_db, 0.0)
    peak_db = 20.0 * np.log10(float(np.max(np.abs(m))) + 1e-9)
    peak_penalty = max(0.0, peak_db + 1.0)
    loudness = float(np.exp(-(np.mean(excess) + 0.35 * np.mean(hole) + 1.8 * peak_penalty) / 2.2))

    # 2. Spectral collision, double-weighting sub/low bands.
    ea = _band_energy(a, target_sr)
    eb = _band_energy(b, target_sr)
    band_frames = min(len(ea), len(eb))
    ea, eb = ea[:band_frames], eb[:band_frames]
    qa = ea / np.maximum(np.sum(ea, axis=1, keepdims=True), 1e-10)
    qb = eb / np.maximum(np.sum(eb, axis=1, keepdims=True), 1e-10)
    activity = np.minimum(
        np.sum(ea, axis=1) / np.maximum(np.percentile(np.sum(ea, axis=1), 85), 1e-10),
        np.sum(eb, axis=1) / np.maximum(np.percentile(np.sum(eb, axis=1), 85), 1e-10),
    )
    weights = np.asarray([2.0, 1.35, 0.75, 0.45], dtype=np.float64)
    collision_penalty = np.mean(activity[:, None] * np.minimum(qa, qb) * weights[None, :])
    collision = float(np.exp(-4.6 * collision_penalty))

    # 3. Spectral continuity relative to a smooth A->B source-envelope path.
    em = _band_energy(m, target_sr)
    spec_frames = min(len(em), len(ea), len(eb))
    em, ea2, eb2 = em[:spec_frames], ea[:spec_frames], eb[:spec_frames]
    contribution_a = np.sum(ea2, axis=1)
    contribution_b = np.sum(eb2, axis=1)
    alpha = contribution_b / np.maximum(contribution_a + contribution_b, 1e-10)
    ua = np.log(np.maximum(ea2, 1e-10))
    ub = np.log(np.maximum(eb2, 1e-10))
    expected = (1.0 - alpha[:, None]) * ua + alpha[:, None] * ub
    observed = np.log(np.maximum(em, 1e-10))
    continuity_error = np.mean(np.abs(observed - expected))
    continuity = float(np.exp(-continuity_error / 2.0))

    # 4. Gain/control smoothness; deliberate drop/cut has a relaxed target.
    control_penalties = []
    for value in controls.values():
        curve = np.asarray(value, dtype=np.float64)
        if curve.size < 5:
            continue
        second = np.diff(curve, n=2)
        control_penalties.append(float(np.mean(np.abs(second))))
    smooth_penalty = float(np.mean(control_penalties)) if control_penalties else 0.0
    scale = 0.012 if archetype in {"Drop Swap", "Double Drop", "Breakdown Lift"} else 0.0045
    smoothness = float(np.exp(-smooth_penalty / scale))

    # 5. Stereo width should follow the weighted source image, not collapse or explode.
    rm = _stereo_ratio(m)
    ra = _stereo_ratio(a)
    rb = _stereo_ratio(b)
    stereo_frames = min(len(rm), len(ra), len(rb), len(alpha))
    if stereo_frames:
        expected_width = (1.0 - alpha[:stereo_frames]) * ra[:stereo_frames] + alpha[:stereo_frames] * rb[:stereo_frames]
        stereo_error = float(np.mean(np.abs(rm[:stereo_frames] - expected_width)))
        stereo = float(np.exp(-stereo_error / 0.34))
    else:
        stereo = 0.70

    # 6. Beat consistency from coincident percussive onset envelopes.
    beat = _onset_similarity(a, b, target_sr)

    # Drum handover: reward a short aligned overlap followed by a clear deck-A release.
    drum_overlap_curve = np.asarray(controls.get("drum_overlap", np.zeros(1)), dtype=np.float64)
    if drum_overlap_curve.size > 4:
        overlap_ratio = float(np.mean(drum_overlap_curve > 0.42))
        overlap_shape = float(np.exp(-abs(overlap_ratio - 0.13) / 0.16))
    else:
        overlap_ratio = 0.0
        overlap_shape = 0.55
    drum_handover = float(np.clip((0.58 + 0.42 * beat) * overlap_shape, 0.0, 1.0))

    # Additional human/style terms.
    harm_a = np.asarray(controls.get("harm_a", np.zeros(1)), dtype=np.float64)
    harm_b = np.asarray(controls.get("harm_b", np.zeros(1)), dtype=np.float64)
    overlap = float(np.mean(np.minimum(harm_a, harm_b))) if harm_a.size == harm_b.size else 0.5
    vocal_score = float(np.clip(1.0 - vocal_risk * overlap * 1.45, 0.0, 1.0))
    repetition = 1.0
    if recent_history and archetype == recent_history[-1]:
        repetition = 0.72
    elif archetype in tuple(recent_history)[-2:]:
        repetition = 0.86

    # Suggested engineering weights from the public transition-evaluation
    # framework, plus a small context/human-variation term.
    engineering = (
        0.25 * loudness
        + 0.25 * collision
        + 0.20 * continuity
        + 0.15 * smoothness
        + 0.10 * stereo
        + 0.05 * beat
    )
    total = (0.72 * engineering + 0.14 * context_fit + 0.10 * vocal_score + 0.04 * drum_handover) * repetition
    return {
        "human_quality": float(np.clip(total, 0.0, 1.0)),
        "quality_loudness": loudness,
        "quality_collision": collision,
        "quality_continuity": continuity,
        "quality_smoothness": smoothness,
        "quality_stereo": stereo,
        "quality_beat": beat,
        "quality_drum_handover": drum_handover,
        "drum_overlap_ratio": overlap_ratio,
        "quality_vocal": vocal_score,
        "quality_context": float(np.clip(context_fit, 0.0, 1.0)),
        "quality_repetition": repetition,
        "peak_dbfs": peak_db,
    }


def render_human_transition(
    *,
    a_low: np.ndarray,
    b_low: np.ndarray,
    a_harm: np.ndarray,
    b_harm: np.ndarray,
    a_perc: np.ndarray,
    b_perc: np.ndarray,
    sample_rate: int,
    bpm: float,
    beats_per_bar: int,
    bars: int,
    local_gain: float,
    effect_strength: float,
    plan_metrics: dict[str, float],
    current_label: str,
    next_label: str,
    history: Sequence[str] = (),
    config: HumanTransitionConfig | None = None,
) -> HumanTransitionResult:
    config = config or HumanTransitionConfig()
    vocal_risk = float(np.clip(1.0 - plan_metrics.get("vocal_clean", 0.5), 0.0, 1.0))
    candidates = _candidate_order(
        config.mode,
        current_label,
        next_label,
        plan_metrics,
        history,
        maximum=config.max_candidates,
    )

    results: list[HumanTransitionResult] = []
    for archetype in candidates:
        mixed, deck_a, deck_b, controls = _render_archetype(
            archetype,
            a_low=a_low,
            b_low=b_low,
            a_harm=a_harm,
            b_harm=b_harm,
            a_perc=a_perc,
            b_perc=b_perc,
            sample_rate=sample_rate,
            bpm=bpm,
            beats_per_bar=beats_per_bar,
            bars=bars,
            local_gain=local_gain,
            effect_strength=effect_strength,
            vocal_risk=vocal_risk,
            drop_landing_phase=float(plan_metrics.get("drop_landing_phase", 1.0)),
        )
        context_fit = _context_style_fit(archetype, current_label, next_label, plan_metrics)
        quality = evaluate_transition(
            mixed=mixed,
            deck_a=deck_a,
            deck_b=deck_b,
            sample_rate=sample_rate,
            controls=controls,
            archetype=archetype,
            context_fit=context_fit,
            recent_history=history,
            vocal_risk=vocal_risk,
            evaluation_sample_rate=config.evaluation_sample_rate,
        )
        # Variation is deterministic: it relaxes the best-score gap and rewards a
        # contextually valid alternative, without randomising reproducibility.
        diversity_bonus = 0.0
        if archetype not in tuple(history)[-max(1, config.avoid_recent) :]:
            diversity_bonus = 0.025 * float(np.clip(config.variation, 0.0, 1.0))
        score = float(np.clip(quality["human_quality"] + diversity_bonus, 0.0, 1.0))
        quality["human_diversity_bonus"] = diversity_bonus
        results.append(
            HumanTransitionResult(
                audio=mixed,
                archetype=archetype,
                score=score,
                quality=quality,
                deck_a=deck_a,
                deck_b=deck_b,
            )
        )

    if not results:
        raise RuntimeError("没有生成可用的真人 DJ 过渡候选。")
    results.sort(key=lambda item: item.score, reverse=True)
    best = results[0]
    if int(round(float(plan_metrics.get("dj_intent_code", 0.0)))) == 5:
        double_drop = next((item for item in results if item.archetype == "Double Drop"), None)
        if double_drop is not None and double_drop.score >= best.score - 0.10:
            best = double_drop
            best.quality["double_drop_preference_applied"] = 1.0
    best.quality["human_candidate_count"] = float(len(results))
    # Persist compact candidate scores for diagnostics without storing audio.
    for index, item in enumerate(results[:6]):
        best.quality[f"candidate_{index}_score"] = float(item.score)
    return best
