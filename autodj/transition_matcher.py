from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import librosa
import numpy as np
from scipy import signal

from .edm_structure import camelot_compatibility
from .models import BarFeatures, PreparedTrack, TransitionPlan
from .muq_analyzer import cosine_similarity


@dataclass(frozen=True)
class MatcherConfig:
    allowed_bars: tuple[int, ...] = (4, 8, 16, 32)
    max_exit_candidates: int = 48
    max_entry_candidates: int = 40
    min_auto_exit_position: float = 0.42
    max_auto_entry_position: float = 0.45
    feature_sample_rate: int = 22_050
    waveform_bins: int = 640


@dataclass(frozen=True)
class TransitionFXConfig:
    """专业过渡渲染参数。style 支持 Smooth / Club / Filter / Echo。"""

    style: str = "Club"
    strength: float = 0.72
    bass_swap_center: float = 0.53
    bass_swap_width: float = 0.18


def _safe_unit(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values.astype(np.float32)
    low, high = np.percentile(values, [10.0, 90.0])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low + 1e-10:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - low) / (high - low), 0.0, 1.0).astype(np.float32)


def waveform_envelope(audio: np.ndarray, bins: int = 640) -> np.ndarray:
    mono = np.mean(audio, axis=1, dtype=np.float64)
    if mono.size == 0:
        return np.zeros(bins, dtype=np.float32)
    edges = np.linspace(0, mono.size, bins + 1, dtype=np.int64)
    envelope = np.zeros(bins, dtype=np.float32)
    for index in range(bins):
        segment = mono[edges[index] : edges[index + 1]]
        if segment.size:
            envelope[index] = float(
                np.sqrt(np.mean(np.square(segment), dtype=np.float64) + 1e-12)
            )
    maximum = float(np.percentile(envelope, 98.0)) if envelope.size else 0.0
    if maximum > 1e-9:
        envelope = np.clip(envelope / maximum, 0.0, 1.0)
    return envelope


def _bar_boundaries(
    downbeat_samples: np.ndarray,
    total_samples: int,
    sample_rate: int,
    bpm: float,
    beats_per_bar: int,
) -> tuple[np.ndarray, np.ndarray]:
    starts = np.unique(
        np.asarray(downbeat_samples, dtype=np.int64)[
            (downbeat_samples >= 0) & (downbeat_samples < total_samples)
        ]
    )
    if starts.size < 2:
        bar_length = int(
            round(beats_per_bar * 60.0 / max(bpm, 1.0) * sample_rate)
        )
        bar_length = max(bar_length, sample_rate)
        starts = np.arange(0, total_samples, bar_length, dtype=np.int64)

    if starts.size == 0:
        starts = np.asarray([0], dtype=np.int64)

    if starts.size >= 2:
        median_bar = int(np.median(np.diff(starts)))
    else:
        median_bar = int(
            round(beats_per_bar * 60.0 / max(bpm, 1.0) * sample_rate)
        )
    median_bar = max(median_bar, int(0.5 * sample_rate))
    ends = np.concatenate(
        [starts[1:], np.asarray([min(total_samples, starts[-1] + median_bar)])]
    )
    valid = ends > starts
    return starts[valid], ends[valid]


def extract_bar_features(
    audio: np.ndarray,
    sample_rate: int,
    downbeat_samples: np.ndarray,
    bpm: float,
    beats_per_bar: int,
    feature_sample_rate: int = 22_050,
) -> BarFeatures:
    """提取用于 DJ 片段匹配的逐小节特征。"""
    starts, ends = _bar_boundaries(
        downbeat_samples,
        audio.shape[0],
        sample_rate,
        bpm,
        beats_per_bar,
    )
    mono = np.mean(audio, axis=1, dtype=np.float32)
    if sample_rate != feature_sample_rate:
        y = librosa.resample(
            mono,
            orig_sr=sample_rate,
            target_sr=feature_sample_rate,
            res_type="soxr_hq",
        )
    else:
        y = mono

    hop_length = 512
    n_fft = 2048
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, center=True)
    magnitude = np.abs(stft).astype(np.float32)
    power = np.square(magnitude, dtype=np.float32)
    frame_rms = librosa.feature.rms(S=magnitude, frame_length=n_fft)[0]
    frame_onset = librosa.onset.onset_strength(
        y=y,
        sr=feature_sample_rate,
        hop_length=hop_length,
    )
    frame_flatness = librosa.feature.spectral_flatness(S=magnitude)[0]
    frame_centroid = librosa.feature.spectral_centroid(
        S=magnitude,
        sr=feature_sample_rate,
    )[0]
    chroma = librosa.feature.chroma_stft(
        S=magnitude,
        sr=feature_sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
    )

    frequencies = librosa.fft_frequencies(sr=feature_sample_rate, n_fft=n_fft)
    low_mask = frequencies <= 180.0
    mid_mask = (frequencies >= 220.0) & (frequencies <= 4_500.0)
    total_power = np.sum(power, axis=0) + 1e-12
    low_ratio_frames = np.sum(power[low_mask], axis=0) / total_power
    mid_ratio_frames = np.sum(power[mid_mask], axis=0) / total_power

    frame_times = librosa.frames_to_time(
        np.arange(magnitude.shape[1]),
        sr=feature_sample_rate,
        hop_length=hop_length,
    )
    start_times = starts / float(sample_rate)
    end_times = ends / float(sample_rate)

    count = starts.size
    rms = np.zeros(count, dtype=np.float32)
    low_ratio = np.zeros(count, dtype=np.float32)
    onset = np.zeros(count, dtype=np.float32)
    brightness = np.zeros(count, dtype=np.float32)
    vocal = np.zeros(count, dtype=np.float32)
    bar_chroma = np.zeros((count, 12), dtype=np.float32)

    for index, (start, end) in enumerate(zip(start_times, end_times)):
        mask = (frame_times >= start) & (frame_times < end)
        frame_indices = np.flatnonzero(mask)
        if frame_indices.size == 0:
            nearest = int(np.argmin(np.abs(frame_times - start)))
            frame_indices = np.asarray([nearest], dtype=np.int64)

        rms[index] = float(np.mean(frame_rms[frame_indices]))
        low_ratio[index] = float(np.mean(low_ratio_frames[frame_indices]))
        onset[index] = float(np.mean(frame_onset[frame_indices]))
        brightness[index] = float(
            np.mean(frame_centroid[frame_indices]) / (feature_sample_rate / 2.0)
        )

        harmonic_mid = mid_ratio_frames[frame_indices] * (
            1.0 - np.clip(frame_flatness[frame_indices], 0.0, 1.0)
        )
        vocal[index] = float(np.mean(harmonic_mid))
        vector = np.mean(chroma[:, frame_indices], axis=1)
        norm = float(np.linalg.norm(vector))
        if norm > 1e-9:
            vector = vector / norm
        bar_chroma[index] = vector.astype(np.float32)

    onset_unit = _safe_unit(onset)
    vocal_unit = _safe_unit(vocal) * (1.0 - 0.25 * onset_unit)
    return BarFeatures(
        start_samples=starts,
        end_samples=ends,
        rms=rms,
        low_ratio=np.clip(low_ratio, 0.0, 1.0),
        onset=onset_unit,
        brightness=np.clip(brightness, 0.0, 1.0),
        vocal_proxy=np.clip(vocal_unit, 0.0, 1.0).astype(np.float32),
        chroma=bar_chroma,
    )


def _cosine_rows(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    numerator = np.sum(a * b, axis=1)
    denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    valid = denominator > 1e-9
    if not np.any(valid):
        return 0.0
    values = numerator[valid] / denominator[valid]
    return float(np.clip(np.mean(values), 0.0, 1.0))


def _harmonic_compatibility(a: np.ndarray, b: np.ndarray) -> float:
    direct = _cosine_rows(a, b)
    fifth_up = 0.88 * _cosine_rows(a, np.roll(b, 7, axis=1))
    fifth_down = 0.88 * _cosine_rows(a, np.roll(b, 5, axis=1))
    return max(direct, fifth_up, fifth_down)


def _muq_sequence_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Small monotonic alignment score for local MuQ semantic trajectories."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if (
        a.ndim != 2
        or b.ndim != 2
        or not len(a)
        or not len(b)
        or a.shape[1] != b.shape[1]
    ):
        return 0.5
    cost = np.full((len(a) + 1, len(b) + 1), np.inf, dtype=np.float64)
    cost[0, 0] = 0.0
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            local = 1.0 - cosine_similarity(a[i - 1], b[j - 1])
            cost[i, j] = local + min(
                cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1]
            )
    normalized = cost[-1, -1] / max(len(a), len(b))
    return float(np.clip(np.exp(-2.4 * normalized), 0.0, 1.0))


def _muq_candidate_metrics(
    current: PreparedTrack,
    next_track: PreparedTrack,
    current_index: int,
    next_index: int,
    bars: int,
) -> tuple[float, float, float]:
    pa = current.muq_profile
    pb = next_track.muq_profile
    if not pa.available or not pb.available:
        return 0.5, 0.5, 0.5
    a_position = (current_index + 0.5 * bars) / max(current.bar_features.count, 1)
    b_position = (next_index + 0.5 * bars) / max(next_track.bar_features.count, 1)
    style = cosine_similarity(pa.global_embedding, pb.global_embedding)
    segment = cosine_similarity(pa.embedding_at(a_position), pb.embedding_at(b_position))
    trajectory = _muq_sequence_similarity(
        pa.sequence_around(a_position, radius=1),
        pb.sequence_around(b_position, radius=1),
    )
    return style, segment, trajectory


def estimate_micro_alignment(
    current: PreparedTrack,
    next_track: PreparedTrack,
    current_start: int,
    next_start: int,
    max_offset_ms: float = 80.0,
) -> tuple[int, float]:
    """Fine-align downbeat transients to reduce kick flam after model beat tracking."""
    sr = current.sample_rate
    window = int(round(min(0.42, 60.0 / max(current.playback_bpm, 1.0)) * sr))
    if window < 128:
        return 0, 0.0
    a = current.audio[current_start : current_start + window]
    b = next_track.audio[next_start : next_start + window]
    length = min(len(a), len(b))
    if length < 128:
        return 0, 0.0
    a_mono = np.mean(a[:length], axis=1, dtype=np.float64)
    b_mono = np.mean(b[:length], axis=1, dtype=np.float64)
    try:
        sos = signal.butter(3, min(220.0 / (sr / 2.0), 0.95), btype="lowpass", output="sos")
        a_mono = signal.sosfiltfilt(sos, a_mono)
        b_mono = signal.sosfiltfilt(sos, b_mono)
    except ValueError:
        pass
    a_onset = np.abs(np.diff(a_mono, prepend=a_mono[0]))
    b_onset = np.abs(np.diff(b_mono, prepend=b_mono[0]))
    smooth = max(1, int(round(0.004 * sr)))
    kernel = np.ones(smooth, dtype=np.float64) / smooth
    a_onset = np.convolve(a_onset, kernel, mode="same")
    b_onset = np.convolve(b_onset, kernel, mode="same")
    search = min(length, int(round(0.18 * sr)))
    a_peak = int(np.argmax(a_onset[:search]))
    b_peak = int(np.argmax(b_onset[:search]))
    maximum = int(round(max_offset_ms * sr / 1000.0))
    offset = int(np.clip(b_peak - a_peak, -maximum, maximum))
    strength_a = float(a_onset[a_peak] / (np.mean(a_onset[:search]) + 1e-9))
    strength_b = float(b_onset[b_peak] / (np.mean(b_onset[:search]) + 1e-9))
    confidence = float(np.clip(min(strength_a, strength_b) / 8.0, 0.0, 1.0))
    if confidence < 0.18:
        return 0, confidence
    return offset, confidence


def _smoothstep(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 0.0, 1.0)
    return values * values * (3.0 - 2.0 * values)


def _allin1_role_compatibility(
    current_role: str,
    next_role: str,
    current_function: str,
    next_function: str,
) -> float:
    a = current_role.upper()
    b = next_role.upper()
    fa = current_function.upper()
    fb = next_function.upper()
    preferred = {
        ("OUTRO", "INTRO"): 1.00,
        ("BREAKDOWN", "DROP"): 0.98,
        ("BUILDUP", "DROP"): 0.96,
        ("VERSE", "INTRO"): 0.88,
        ("VOCAL", "INTRO"): 0.86,
        ("CHORUS", "BREAKDOWN"): 0.84,
        ("DROP", "BREAKDOWN"): 0.82,
        ("SOLO", "INTRO"): 0.82,
        ("PHRASE", "INTRO"): 0.80,
        ("OUTRO", "DROP"): 0.78,
        ("BREAKDOWN", "INTRO"): 0.78,
    }
    if (a, b) in preferred:
        return preferred[(a, b)]
    # Harmonix functional labels provide a useful fallback even when the local
    # EDM role conversion is uncertain.
    if fa in {"OUTRO", "END"} and fb in {"INTRO", "START", "INST"}:
        return 0.96
    if fa in {"BREAK", "BRIDGE"} and fb in {"CHORUS", "INST"}:
        return 0.86
    if fa in {"VERSE", "CHORUS", "SOLO"} and fb in {"INTRO", "BREAK", "INST"}:
        return 0.78
    if a in {"VOCAL", "VERSE", "CHORUS"} and b in {"VOCAL", "VERSE", "CHORUS"}:
        return 0.34
    if a == b and a in {"DROP", "CHORUS"}:
        return 0.52
    return 0.62


def _score_candidate(
    current: PreparedTrack,
    next_track: PreparedTrack,
    current_index: int,
    next_index: int,
    bars: int,
) -> tuple[float, dict[str, float]]:
    a = current.bar_features
    b = next_track.bar_features
    a_slice = slice(current_index, current_index + bars)
    b_slice = slice(next_index, next_index + bars)

    a_rms = np.maximum(a.rms[a_slice].astype(np.float64), 1e-8)
    b_rms = np.maximum(b.rms[b_slice].astype(np.float64), 1e-8)
    x = (np.arange(bars, dtype=np.float64) + 0.5) / bars
    fade_out = np.cos(x * np.pi / 2.0)
    fade_in = np.sin(x * np.pi / 2.0)

    local_gain = np.clip(
        np.median(a_rms) / max(float(np.median(b_rms)), 1e-8),
        0.65,
        1.55,
    )
    b_matched = b_rms * local_gain
    mixed_rms = np.sqrt(
        np.square(a_rms * fade_out) + np.square(b_matched * fade_in)
    )
    log_curve = np.log(mixed_rms + 1e-8)
    continuity = float(np.exp(-3.8 * np.std(log_curve)))
    endpoint_jump = abs(math.log((mixed_rms[-1] + 1e-8) / (mixed_rms[0] + 1e-8)))
    continuity *= float(np.exp(-0.7 * endpoint_jump))

    harmonic = _harmonic_compatibility(a.chroma[a_slice], b.chroma[b_slice])
    rhythm = float(
        np.exp(
            -2.2
            * abs(
                float(np.mean(a.onset[a_slice]))
                - float(np.mean(b.onset[b_slice]))
            )
        )
    )
    brightness = float(
        np.exp(
            -2.0
            * abs(
                float(np.mean(a.brightness[a_slice]))
                - float(np.mean(b.brightness[b_slice]))
            )
        )
    )

    overlap_weight = fade_out * fade_in * 2.0
    bass_collision = float(
        np.mean(a.low_ratio[a_slice] * b.low_ratio[b_slice] * overlap_weight)
    )
    vocal_overlap = float(
        np.mean(a.vocal_proxy[a_slice] * b.vocal_proxy[b_slice] * overlap_weight)
    )
    bass_clean = float(np.clip(1.0 - 1.55 * bass_collision, 0.0, 1.0))
    vocal_clean = float(np.clip(1.0 - 1.70 * vocal_overlap, 0.0, 1.0))

    a_position = current_index / max(a.count - 1, 1)
    b_position = next_index / max(b.count - 1, 1)
    exit_structure = float(np.exp(-((a_position - 0.76) / 0.27) ** 2))
    entry_structure = float(np.exp(-((b_position - 0.11) / 0.20) ** 2))
    structure = 0.58 * exit_structure + 0.42 * entry_structure

    third = max(1, bars // 3)
    a_decay = float(np.mean(a.rms[a_slice][:third]) - np.mean(a.rms[a_slice][-third:]))
    b_growth = float(np.mean(b.rms[b_slice][-third:]) - np.mean(b.rms[b_slice][:third]))
    scale = max(float(np.mean(a.rms[a_slice]) + np.mean(b.rms[b_slice])), 1e-8)
    dynamics = float(np.clip(0.5 + 2.2 * (a_decay + b_growth) / scale, 0.0, 1.0))

    average_vocal = max(
        float(np.mean(a.vocal_proxy[a_slice])),
        float(np.mean(b.vocal_proxy[b_slice])),
    )
    average_energy = 0.5 * (
        float(np.mean(_safe_unit(a.rms)[a_slice]))
        + float(np.mean(_safe_unit(b.rms)[b_slice]))
    )
    if average_vocal > 0.66:
        preferred_bars = 8
    elif average_vocal < 0.28 and average_energy < 0.58:
        preferred_bars = 32
    else:
        preferred_bars = 16
    length_fit = float(
        np.exp(-0.22 * abs(math.log2(max(bars, 1) / preferred_bars)))
    )

    cue_alignment = 0.50
    phrase_alignment = 0.50
    if current.structure.mix_out_score.size > current_index:
        cue_a = float(current.structure.mix_out_score[current_index])
    else:
        cue_a = 0.50
    if next_track.structure.mix_in_score.size > next_index:
        cue_b = float(next_track.structure.mix_in_score[next_index])
    else:
        cue_b = 0.50
    cue_alignment = float(np.sqrt(max(cue_a, 0.0) * max(cue_b, 0.0)))

    if current.structure.phrase_mask.size > current_index:
        phrase_a = float(current.structure.phrase_mask[current_index])
    else:
        phrase_a = 0.50
    if next_track.structure.phrase_mask.size > next_index:
        phrase_b = float(next_track.structure.phrase_mask[next_index])
    else:
        phrase_b = 0.50
    phrase_alignment = 0.5 * (phrase_a + phrase_b)
    key_score = camelot_compatibility(current.structure, next_track.structure)
    edm_confidence = float(
        np.sqrt(
            max(current.structure.edm_confidence, 0.0)
            * max(next_track.structure.edm_confidence, 0.0)
        )
    )
    muq_style, muq_segment, muq_trajectory = _muq_candidate_metrics(
        current, next_track, current_index, next_index, bars
    )
    current_role = (
        current.structure.labels[current_index]
        if current_index < len(current.structure.labels)
        else "SECTION"
    )
    next_role = (
        next_track.structure.labels[next_index]
        if next_index < len(next_track.structure.labels)
        else "SECTION"
    )
    current_function = (
        current.structure.functional_labels[current_index]
        if current_index < len(current.structure.functional_labels)
        else "UNKNOWN"
    )
    next_function = (
        next_track.structure.functional_labels[next_index]
        if next_index < len(next_track.structure.functional_labels)
        else "UNKNOWN"
    )
    role_compatibility = _allin1_role_compatibility(
        current_role, next_role, current_function, next_function
    )
    boundary_a = (
        float(current.structure.allin1_boundary_score[current_index])
        if current.structure.allin1_boundary_score.size > current_index
        else 0.50
    )
    boundary_b = (
        float(next_track.structure.allin1_boundary_score[next_index])
        if next_track.structure.allin1_boundary_score.size > next_index
        else 0.50
    )
    allin1_boundary = float(np.sqrt(max(boundary_a, 0.0) * max(boundary_b, 0.0)))

    # MuQ is used as a bounded semantic/style term, not as the sole compatibility
    # judge: recent mashup work shows general-purpose embeddings are insufficient
    # and that A->B compatibility is directional.
    # arbitrary position priors. Parametric EQ/fader terms follow the
    # differentiable-DSP transition literature.
    score = (
        0.14 * continuity
        + 0.10 * harmonic
        + 0.06 * key_score
        + 0.06 * rhythm
        + 0.02 * brightness
        + 0.09 * bass_clean
        + 0.09 * vocal_clean
        + 0.03 * structure
        + 0.03 * dynamics
        + 0.02 * length_fit
        + 0.08 * cue_alignment
        + 0.04 * phrase_alignment
        + 0.01 * edm_confidence
        + 0.05 * muq_style
        + 0.06 * muq_segment
        + 0.02 * muq_trajectory
        + 0.05 * allin1_boundary
        + 0.05 * role_compatibility
    )
    metrics = {
        "continuity": continuity,
        "harmonic": harmonic,
        "key_score": key_score,
        "rhythm": rhythm,
        "brightness": brightness,
        "bass_clean": bass_clean,
        "vocal_clean": vocal_clean,
        "structure": structure,
        "dynamics": dynamics,
        "length_fit": length_fit,
        "cue_alignment": cue_alignment,
        "phrase_alignment": phrase_alignment,
        "edm_confidence": edm_confidence,
        "muq_style": muq_style,
        "muq_segment": muq_segment,
        "muq_trajectory": muq_trajectory,
        "allin1_boundary": allin1_boundary,
        "allin1_role_compatibility": role_compatibility,
        "current_functional_role": float(
            {"INTRO": 0, "VERSE": 1, "CHORUS": 2, "BREAK": 3, "BRIDGE": 4, "INST": 5, "SOLO": 6, "OUTRO": 7, "START": 8, "END": 9}.get(current_function, -1)
        ),
        "next_functional_role": float(
            {"INTRO": 0, "VERSE": 1, "CHORUS": 2, "BREAK": 3, "BRIDGE": 4, "INST": 5, "SOLO": 6, "OUTRO": 7, "START": 8, "END": 9}.get(next_function, -1)
        ),
        "local_gain": float(local_gain),
    }
    return float(np.clip(score, 0.0, 1.0)), metrics


def _candidate_indices(
    features: BarFeatures,
    earliest_sample: int,
    is_exit: bool,
    config: MatcherConfig,
) -> np.ndarray:
    indices = np.arange(features.count, dtype=np.int64)
    indices = indices[features.start_samples >= earliest_sample]
    if indices.size == 0:
        return indices

    positions = indices / max(features.count - 1, 1)
    if is_exit:
        preferred = indices[positions >= config.min_auto_exit_position]
        if preferred.size:
            indices = preferred
        indices = indices[-config.max_exit_candidates :]
    else:
        preferred = indices[positions <= config.max_auto_entry_position]
        if preferred.size:
            indices = preferred
        indices = indices[: config.max_entry_candidates]
    return indices


def _rank_by_cue_scores(
    indices: np.ndarray,
    scores: np.ndarray,
    maximum: int,
) -> np.ndarray:
    if indices.size == 0 or scores.size == 0:
        return indices[:maximum]
    valid = indices[indices < scores.size]
    if valid.size == 0:
        return indices[:maximum]
    ranked = sorted(
        (int(index) for index in valid),
        key=lambda index: (float(scores[index]), -index),
        reverse=True,
    )[:maximum]
    # Search order does not affect the optimum; sorting restores chronology for
    # deterministic tie handling and readable debugging.
    return np.asarray(sorted(ranked), dtype=np.int64)


def _make_curves(
    length: int,
    local_gain: float,
    fx: TransitionFXConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    phase = np.linspace(0.0, 1.0, length, endpoint=True, dtype=np.float32)
    strength = float(np.clip(fx.strength, 0.0, 1.0))
    style = fx.style.lower()

    fade_out = np.cos(phase * np.pi / 2.0).astype(np.float32)
    fade_in = np.sin(phase * np.pi / 2.0).astype(np.float32)
    gain_curve = local_gain + (1.0 - local_gain) * _smoothstep(phase)
    fade_in = fade_in * gain_curve

    # 避免两首歌主体在中点同时满功率：强效果时增加约 1.2 dB 中点空间。
    center_duck = 1.0 - (0.13 * strength) * np.sin(np.pi * phase) ** 2
    fade_out *= center_duck
    fade_in *= center_duck

    center = float(np.clip(fx.bass_swap_center, 0.35, 0.70))
    width = float(np.clip(fx.bass_swap_width, 0.08, 0.35))
    bass_progress = _smoothstep((phase - (center - width / 2.0)) / width)
    bass_out = np.sqrt(np.clip(1.0 - bass_progress, 0.0, 1.0))
    bass_in = np.sqrt(np.clip(bass_progress, 0.0, 1.0))

    # 用三段分频增益模拟 DJ 高通/低通扫频。不是简单整轨音量淡化。
    if style == "smooth":
        high_out = fade_out.copy()
        high_in = fade_in.copy()
    elif style == "filter":
        outgoing_filter = 1.0 - strength * 0.92 * _smoothstep((phase - 0.10) / 0.70)
        incoming_open = 0.12 + 0.88 * _smoothstep((phase - 0.04) / 0.62)
        high_out = fade_out * outgoing_filter
        high_in = fade_in * ((1.0 - strength) + strength * incoming_open)
    elif style == "echo":
        outgoing_filter = 1.0 - strength * 0.68 * _smoothstep((phase - 0.34) / 0.50)
        incoming_open = 0.22 + 0.78 * _smoothstep((phase - 0.02) / 0.48)
        high_out = fade_out * outgoing_filter
        high_in = fade_in * ((1.0 - strength) + strength * incoming_open)
    else:  # Club
        outgoing_filter = 1.0 - strength * 0.78 * _smoothstep((phase - 0.27) / 0.56)
        incoming_open = 0.18 + 0.82 * _smoothstep((phase - 0.03) / 0.52)
        high_out = fade_out * outgoing_filter
        high_in = fade_in * ((1.0 - strength) + strength * incoming_open)

    return tuple(
        np.ascontiguousarray(array, dtype=np.float32)
        for array in (fade_out, fade_in, bass_out, bass_in, high_out, high_in)
    )  # type: ignore[return-value]


def find_best_transition(
    current: PreparedTrack,
    next_track: PreparedTrack,
    earliest_start: int = 0,
    requested_bars: int = 0,
    force_current_start: int | None = None,
    config: MatcherConfig | None = None,
    fx_config: TransitionFXConfig | None = None,
) -> TransitionPlan:
    """搜索最佳重拍片段、自动长度，并生成专业三段 EQ 过渡曲线。"""
    config = config or MatcherConfig()
    fx_config = fx_config or TransitionFXConfig()
    a = current.bar_features
    b = next_track.bar_features
    if a.count < 2 or b.count < 2:
        raise RuntimeError("逐小节特征不足，无法自动匹配切歌片段。")

    if requested_bars > 0:
        allowed_bars: Iterable[int] = (requested_bars,)
    else:
        allowed_bars = config.allowed_bars

    if force_current_start is None:
        exit_indices = _candidate_indices(
            a,
            earliest_sample=earliest_start,
            is_exit=True,
            config=config,
        )
    else:
        future = np.flatnonzero(a.start_samples >= force_current_start)
        if future.size:
            exit_indices = np.asarray([int(future[0])], dtype=np.int64)
        else:
            exit_indices = np.asarray([a.count - 1], dtype=np.int64)

    entry_indices = _candidate_indices(
        b,
        earliest_sample=0,
        is_exit=False,
        config=config,
    )
    if force_current_start is None:
        exit_indices = _rank_by_cue_scores(
            exit_indices,
            current.structure.mix_out_score,
            config.max_exit_candidates,
        )
    entry_indices = _rank_by_cue_scores(
        entry_indices,
        next_track.structure.mix_in_score,
        config.max_entry_candidates,
    )
    if exit_indices.size == 0 or entry_indices.size == 0:
        raise RuntimeError("没有足够的重拍候选点。")

    best: tuple[float, int, int, int, dict[str, float]] | None = None
    for bars in allowed_bars:
        bars = int(bars)
        if bars <= 0:
            continue
        for current_index in exit_indices:
            if current_index + bars > a.count:
                continue
            for next_index in entry_indices:
                if next_index + bars > b.count:
                    continue
                score, metrics = _score_candidate(
                    current,
                    next_track,
                    int(current_index),
                    int(next_index),
                    bars,
                )
                if best is None or score > best[0]:
                    best = (score, int(current_index), int(next_index), bars, metrics)

    if best is None:
        current_index = int(exit_indices[-1])
        next_index = int(entry_indices[0])
        maximum = min(a.count - current_index, b.count - next_index)
        bars = max(1, min(maximum, requested_bars or 4))
        score, metrics = _score_candidate(
            current,
            next_track,
            current_index,
            next_index,
            bars,
        )
        best = (score, current_index, next_index, bars, metrics)

    score, current_index, next_index, bars, metrics = best
    current_start = int(a.start_samples[current_index])
    next_start = int(b.start_samples[next_index])
    current_end = int(a.end_samples[current_index + bars - 1])
    next_end = int(b.end_samples[next_index + bars - 1])
    micro_offset, micro_confidence = estimate_micro_alignment(
        current, next_track, current_start, next_start
    )
    adjusted_next_start = int(
        np.clip(next_start + micro_offset, 0, max(0, next_end - 1))
    )
    metrics["micro_align_ms"] = 1000.0 * micro_offset / current.sample_rate
    metrics["micro_align_confidence"] = micro_confidence
    next_start = adjusted_next_start
    # The outgoing phrase defines wall-clock transition duration. The incoming
    # phrase is rendered to this exact duration later, then resumes at next_end.
    length = min(
        current_end - current_start,
        current.total_samples - current_start,
    )
    length = max(1, int(length))
    fade_out, fade_in, bass_out, bass_in, high_out, high_in = _make_curves(
        length,
        metrics.get("local_gain", 1.0),
        fx_config,
    )
    return TransitionPlan(
        current_start=current_start,
        next_start=next_start,
        length=length,
        bars=bars,
        current_bar_index=current_index,
        next_bar_index=next_index,
        score=score,
        fade_out=fade_out,
        fade_in=fade_in,
        bass_out=bass_out,
        bass_in=bass_in,
        high_out=high_out,
        high_in=high_in,
        echo_audio=np.zeros((length, 2), dtype=np.float32),
        metrics=metrics,
        automatic=(requested_bars == 0 and force_current_start is None),
        style=fx_config.style,
        effect_strength=float(np.clip(fx_config.strength, 0.0, 1.0)),
        transition_mode="Paper EQ/Fader",
        switch_position=(
            0.64
            if next_index < len(next_track.structure.labels)
            and next_track.structure.labels[next_index] == "DROP"
            else 0.55
        ),
        next_resume_sample=int(next_end),
        micro_offset_samples=int(micro_offset),
    )
