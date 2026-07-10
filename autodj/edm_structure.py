from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.spatial.distance import cdist

from .models import BarFeatures, EDMStructure


_MAJOR_PROFILE = np.asarray(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
    dtype=np.float64,
)
_MINOR_PROFILE = np.asarray(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
    dtype=np.float64,
)
# pitch class C=0 ... B=11 -> Camelot notation
_MAJOR_CAMELOT = ("8B", "3B", "10B", "5B", "12B", "7B", "2B", "9B", "4B", "11B", "6B", "1B")
_MINOR_CAMELOT = ("5A", "12A", "7A", "2A", "9A", "4A", "11A", "6A", "1A", "8A", "3A", "10A")


def _unit(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values.astype(np.float32)
    low, high = np.percentile(values, [5.0, 95.0])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low + 1e-12:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - low) / (high - low), 0.0, 1.0).astype(np.float32)


def _as_matrix(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    mean = np.mean(values, axis=0, keepdims=True)
    std = np.std(values, axis=0, keepdims=True)
    return (values - mean) / np.maximum(std, 1e-6)


def checkerboard_novelty(values: np.ndarray, half_window_bars: int = 4) -> np.ndarray:
    """
    Foote-style self-similarity novelty.

    The paper uses an eight-bar checkerboard kernel: four bars before and four
    bars after each candidate boundary. This implementation computes the same
    contrast as mean cross-region distance minus within-region distance.
    """
    matrix = _as_matrix(values)
    count = matrix.shape[0]
    novelty = np.zeros(count, dtype=np.float64)
    if count < 2 * half_window_bars + 1:
        return novelty.astype(np.float32)

    distances = cdist(matrix, matrix, metric="euclidean")
    half = int(max(1, half_window_bars))
    for index in range(half, count - half):
        before = slice(index - half, index)
        after = slice(index, index + half)
        cross = float(np.mean(distances[before, after]))
        within_a = float(np.mean(distances[before, before]))
        within_b = float(np.mean(distances[after, after]))
        novelty[index] = max(0.0, cross - 0.5 * (within_a + within_b))
    return _unit(novelty)


def _peak_mask(novelty: np.ndarray, radius_bars: int = 4, threshold_ratio: float = 0.30) -> np.ndarray:
    novelty = np.asarray(novelty, dtype=np.float64)
    if novelty.size == 0 or float(np.max(novelty)) <= 1e-9:
        return np.zeros_like(novelty, dtype=bool)
    radius = max(1, int(radius_bars))
    maximum = ndimage.maximum_filter1d(novelty, size=2 * radius + 1, mode="nearest")
    return (novelty >= maximum - 1e-8) & (novelty >= threshold_ratio * float(np.max(novelty)))


def _period_phase(novelty: np.ndarray, period_bars: int = 4) -> int:
    scores: list[float] = []
    for phase in range(period_bars):
        values = novelty[phase::period_bars]
        scores.append(float(np.sqrt(np.mean(np.square(values)))) if values.size else 0.0)
    return int(np.argmax(scores)) if scores else 0


def _phrase_mask(count: int, phase: int) -> np.ndarray:
    result = np.zeros(count, dtype=np.float32)
    for index in range(count):
        delta = (index - phase) % 16
        if delta == 0:
            result[index] = 1.0
        elif delta == 8:
            result[index] = 0.88
        elif delta % 4 == 0:
            result[index] = 0.64
    return result


def _estimate_key(features: BarFeatures) -> tuple[int, str, str, float]:
    if features.chroma.size == 0:
        return -1, "unknown", "—", 0.0
    weights = np.maximum(features.rms.astype(np.float64), 1e-5)
    chroma = np.average(features.chroma.astype(np.float64), axis=0, weights=weights)
    if np.linalg.norm(chroma) <= 1e-9:
        return -1, "unknown", "—", 0.0
    chroma = (chroma - np.mean(chroma)) / max(float(np.std(chroma)), 1e-6)

    scores: list[tuple[float, int, str]] = []
    for root in range(12):
        major = np.roll(_MAJOR_PROFILE, root)
        minor = np.roll(_MINOR_PROFILE, root)
        major = (major - np.mean(major)) / np.std(major)
        minor = (minor - np.mean(minor)) / np.std(minor)
        scores.append((float(np.dot(chroma, major) / 12.0), root, "major"))
        scores.append((float(np.dot(chroma, minor) / 12.0), root, "minor"))
    scores.sort(reverse=True, key=lambda item: item[0])
    best, root, mode = scores[0]
    second = scores[1][0] if len(scores) > 1 else -1.0
    confidence = float(np.clip((best - second) * 2.5 + 0.45, 0.0, 1.0))
    camelot = _MAJOR_CAMELOT[root] if mode == "major" else _MINOR_CAMELOT[root]
    return root, mode, camelot, confidence


def _silence_bounds(audio: np.ndarray, sample_rate: int) -> tuple[int, int]:
    if audio.size == 0:
        return 0, 0
    mono = np.mean(audio, axis=1, dtype=np.float64)
    frame = max(256, int(round(0.050 * sample_rate)))
    hop = max(128, frame // 2)
    if len(mono) <= frame:
        return 0, len(mono)
    rms = np.asarray(
        [
            np.sqrt(np.mean(np.square(mono[start : start + frame])) + 1e-12)
            for start in range(0, len(mono) - frame + 1, hop)
        ],
        dtype=np.float64,
    )
    reference = float(np.percentile(rms, 90.0))
    threshold = max(1e-5, reference * 10.0 ** (-42.0 / 20.0))
    active = np.flatnonzero(rms >= threshold)
    if active.size == 0:
        return 0, len(mono)
    start = max(0, int(active[0] * hop - 0.02 * sample_rate))
    end = min(len(mono), int(active[-1] * hop + frame + 0.02 * sample_rate))
    return start, max(start + 1, end)


def analyze_edm_structure(
    audio: np.ndarray,
    sample_rate: int,
    features: BarFeatures,
    beats_per_bar: int = 4,
) -> EDMStructure:
    """Extract supporting EDM roles, key and energy features only.

    Since v1.2.9 this function is deliberately *not* a cue detector.  The
    previous checkerboard/period/salience cue proposal has been removed from
    the playback decision path.  Every transition candidate is supplied by
    CUE-DETR and later quantized by Beat This!.  These local descriptors remain
    useful for role naming, Camelot compatibility and transition rendering.
    """
    count = features.count
    if count == 0:
        return EDMStructure()

    onset = _unit(features.onset)
    rms = _unit(features.rms)
    low = _unit(features.low_ratio)
    brightness = _unit(features.brightness)
    vocal = _unit(features.vocal_proxy)

    # A lightweight diagnostic boundary curve is retained solely for role
    # labelling.  It cannot become a cue because all cue/mix arrays returned
    # below are zero and are replaced only by apply_cuedetr_cues().
    descriptor = np.column_stack((rms, onset, low, brightness, vocal))
    delta = np.zeros(count, dtype=np.float32)
    if count > 1:
        delta[1:] = np.linalg.norm(np.diff(descriptor, axis=0), axis=1).astype(np.float32)
    combined = _unit(ndimage.gaussian_filter1d(delta.astype(np.float64), sigma=0.7))

    harmonic_energy = _unit(features.rms * (1.0 - 0.25 * onset))
    percussive_energy = _unit(features.rms * (0.15 + 0.85 * onset))
    salience = np.zeros(count, dtype=np.float32)
    for index in range(count):
        end = min(count, index + 4)
        harmonic_future = float(np.mean(harmonic_energy[index:end]))
        percussive_future = float(np.mean(percussive_energy[index:end]))
        salience[index] = max(harmonic_future, 0.86 * percussive_future)
    salience = _unit(salience)

    labels: list[str] = []
    energy = _unit(features.rms)
    for index in range(count):
        before_slice = energy[max(0, index - 4) : index]
        after_slice = energy[index : min(count, index + 4)]
        before = float(np.mean(before_slice)) if before_slice.size else 0.0
        after = float(np.mean(after_slice)) if after_slice.size else 0.0
        local = float(energy[index])
        vocal_here = float(np.mean(vocal[index : min(count, index + 4)]))
        percussion_here = float(np.mean(onset[index : min(count, index + 4)]))
        low_here = float(np.mean(low[index : min(count, index + 4)]))
        boundary = float(combined[index])
        rise = after - before
        fall = before - after

        if index <= max(2, int(0.10 * count)):
            label = "INTRO"
        elif index >= int(0.86 * count):
            label = "OUTRO"
        elif rise > 0.16 and after > 0.55 and (boundary > 0.30 or salience[index] > 0.55):
            label = "BUILDUP"
        elif fall > 0.17 and after < 0.55 and (boundary > 0.25 or percussion_here < 0.45):
            label = "BREAKDOWN"
        elif after > 0.62 and percussion_here > 0.48 and low_here > 0.38 and (rise > 0.10 or boundary > 0.42):
            label = "DROP"
        elif local > after + 0.10 and before > 0.55 and after > 0.34:
            label = "COOLDOWN"
        elif vocal_here > 0.64:
            label = "VOCAL"
        else:
            label = "SECTION"
        labels.append(label)

    key_index, mode, camelot, key_confidence = _estimate_key(features)
    silence_start, silence_end = _silence_bounds(audio, sample_rate)
    four_floor = np.clip(onset * low * 1.45, 0.0, 1.0)
    meter_score = 1.0 if beats_per_bar == 4 else 0.20
    drum_score = float(np.clip(np.mean(four_floor) * 1.45, 0.0, 1.0))
    dynamics_score = float(np.clip(np.std(energy) * 2.2 + np.mean(combined) * 0.8, 0.0, 1.0))
    edm_confidence = float(
        np.clip(
            0.48 * meter_score
            + 0.34 * drum_score
            + 0.10 * dynamics_score
            + 0.08 * key_confidence,
            0.0,
            1.0,
        )
    )

    zeros = np.zeros(count, dtype=np.float32)
    return EDMStructure(
        novelty=np.ascontiguousarray(combined, dtype=np.float32),
        cue_score=zeros.copy(),
        mix_in_score=zeros.copy(),
        mix_out_score=zeros.copy(),
        salience=np.ascontiguousarray(salience, dtype=np.float32),
        phrase_mask=zeros.copy(),
        labels=tuple(labels),
        structure_source="local roles only; CUE-DETR supplies all cue points",
        phase_offset=0,
        key_index=key_index,
        mode=mode,
        camelot=camelot,
        edm_confidence=edm_confidence,
        silence_start_sample=silence_start,
        silence_end_sample=silence_end,
    )


def camelot_compatibility(a: EDMStructure, b: EDMStructure) -> float:
    """Camelot-compatible key score with fifth and relative-key tolerance."""
    if a.key_index < 0 or b.key_index < 0:
        return 0.55
    delta = (b.key_index - a.key_index) % 12
    if delta == 0 and a.mode == b.mode:
        return 1.0
    if delta in (5, 7) and a.mode == b.mode:
        return 0.90
    # relative major/minor: major root and minor root differ by 9/3 semitones.
    if a.mode != b.mode and delta in (3, 9):
        return 0.92
    if delta in (2, 10):
        return 0.70
    return 0.42
