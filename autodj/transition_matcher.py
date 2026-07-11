from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import librosa
import numpy as np
from scipy import signal

from .edm_structure import camelot_compatibility
from .dj_phrase_policy import evaluate_phrase_policy
from .models import BarFeatures, PreparedTrack, TransitionPlan
from .muq_analyzer import cosine_similarity


@dataclass(frozen=True)
class MatcherConfig:
    # These bars are analysis context only. The audible transition remains
    # centered on the selected CUE-DETR switch point. Four beats before the cue
    # are reserved for a beat-quantized incoming drum-loop bridge, while the
    # outgoing deck is released over two beats after the cue. Only percussion is
    # exposed during the early bridge; harmonic and bass ownership still change
    # close to the neural cue, so this does not become a long two-song blend.
    allowed_bars: tuple[int, ...] = (4, 8, 16)
    pre_roll_beats: float = 4.0
    release_beats: float = 2.0
    max_exit_candidates: int = 32
    max_entry_candidates: int = 24
    # CUE-DETR remains the only cue source. Positional tail/head windows are
    # intentionally not used: every neural cue may compete on phrase, beat,
    # vocal, bass and energy compatibility.
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


def _kick_phase_envelope(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Kick-focused onset envelope used like a DJ headphone phase check."""
    if audio.size == 0:
        return np.zeros(0, dtype=np.float64)
    mono = np.mean(audio, axis=1, dtype=np.float64)
    nyquist = sample_rate / 2.0
    try:
        lo = max(25.0 / nyquist, 1e-4)
        hi = min(190.0 / nyquist, 0.96)
        if hi > lo:
            sos = signal.butter(3, [lo, hi], btype="bandpass", output="sos")
            mono = signal.sosfiltfilt(sos, mono)
    except ValueError:
        pass
    energy = mono * mono
    smooth = max(1, int(round(0.008 * sample_rate)))
    kernel = np.ones(smooth, dtype=np.float64) / smooth
    energy = np.convolve(energy, kernel, mode="same")
    onset = np.maximum(np.diff(energy, prepend=energy[0]), 0.0)
    scale = float(np.percentile(onset, 95.0) + 1e-12)
    return np.clip(onset / scale, 0.0, 5.0)


def estimate_micro_alignment(
    current: PreparedTrack,
    next_track: PreparedTrack,
    current_start: int,
    next_start: int,
    max_offset_ms: float = 80.0,
) -> tuple[int, float]:
    """Robust launch nudge from several beats instead of one transient.

    A human DJ cues the incoming deck, listens to a few kicks, and nudges the
    whole deck until the phase stops flamming.  Looking at only the first peak
    can confuse a hi-hat/snare or a weak pickup for the kick, so this estimator
    correlates a kick-focused envelope across roughly four beats.
    """
    sr = current.sample_rate
    beat_seconds = 60.0 / max(current.playback_bpm, 1.0)
    window = int(round(min(4.25 * beat_seconds, 3.2) * sr))
    if window < 128:
        return 0, 0.0
    a = current.audio[current_start : current_start + window]
    b = next_track.audio[next_start : next_start + window]
    length = min(len(a), len(b))
    if length < 128:
        return 0, 0.0
    ea = _kick_phase_envelope(a[:length], sr)
    eb = _kick_phase_envelope(b[:length], sr)
    maximum = min(int(round(max_offset_ms * sr / 1000.0)), max(1, length // 5))
    if maximum < 1 or float(np.max(ea)) < 1e-6 or float(np.max(eb)) < 1e-6:
        return 0, 0.0

    scores: list[float] = []
    lags = np.arange(-maximum, maximum + 1, dtype=np.int64)
    for lag in lags:
        if lag >= 0:
            xa = ea[: length - lag]
            xb = eb[lag:length]
        else:
            xa = ea[-lag:length]
            xb = eb[: length + lag]
        if xa.size < 64:
            scores.append(-1.0)
            continue
        # Emphasize the first two bars, as DJs lock the phase before opening EQ.
        weights = np.linspace(1.35, 0.75, xa.size, dtype=np.float64)
        numerator = float(np.sum(weights * xa * xb))
        denominator = float(
            np.sqrt(np.sum(weights * xa * xa) * np.sum(weights * xb * xb)) + 1e-12
        )
        scores.append(numerator / denominator)
    score_array = np.asarray(scores, dtype=np.float64)
    best_index = int(np.argmax(score_array))
    best_lag = int(lags[best_index])
    best_score = float(score_array[best_index])

    # Peak uniqueness and signal strength prevent a repetitive hi-hat pattern
    # from being treated as a confident kick lock.
    guard = max(1, int(round(0.008 * sr)))
    masked = score_array.copy()
    masked[max(0, best_index - guard) : min(len(masked), best_index + guard + 1)] = -1.0
    second = float(np.max(masked)) if masked.size else -1.0
    uniqueness = float(np.clip((best_score - second) / 0.18, 0.0, 1.0))
    strength = float(np.clip(best_score, 0.0, 1.0))
    confidence = float(np.clip(0.65 * strength + 0.35 * uniqueness, 0.0, 1.0))
    if confidence < 0.18:
        return 0, confidence
    return int(np.clip(best_lag, -maximum, maximum)), confidence

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

    # Positions are retained only for diagnostics. They do not influence the
    # score, so a musically valid middle-section cue can beat a weak tail/head
    # pair.
    a_position = current_index / max(a.count - 1, 1)
    b_position = next_index / max(b.count - 1, 1)

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
    phrase_policy = evaluate_phrase_policy(
        current,
        next_track,
        current_index,
        next_index,
        bars,
        harmonic=harmonic,
        bass_clean=bass_clean,
    )

    # MuQ is used as a bounded semantic/style term, not as the sole compatibility
    # judge: recent mashup work shows general-purpose embeddings are insufficient
    # and that A->B compatibility is directional.
    # arbitrary position priors. Parametric EQ/fader terms follow the
    # differentiable-DSP transition literature.
    score = (
        0.12 * continuity
        + 0.08 * harmonic
        + 0.04 * key_score
        + 0.05 * rhythm
        + 0.01 * brightness
        + 0.10 * bass_clean
        + 0.10 * vocal_clean
        + 0.04 * dynamics
        + 0.02 * length_fit
        + 0.08 * cue_alignment
        + 0.07 * phrase_alignment
        + 0.01 * edm_confidence
        + 0.04 * muq_style
        + 0.04 * muq_segment
        + 0.02 * muq_trajectory
        + 0.04 * allin1_boundary
        + 0.04 * role_compatibility
        + 0.10 * phrase_policy.score
    )
    # Starting a long blend in the middle of a drop should lose even when its
    # low-level similarity metrics happen to look good. The phrase guard steers
    # unsafe drop material toward a conservative bass handover instead.
    score *= 0.74 + 0.26 * phrase_policy.drop_guard_score
    metrics = {
        "continuity": continuity,
        "harmonic": harmonic,
        "key_score": key_score,
        "rhythm": rhythm,
        "brightness": brightness,
        "bass_clean": bass_clean,
        "vocal_clean": vocal_clean,
        "out_position": float(a_position),
        "in_position": float(b_position),
        "position_bias_applied": 0.0,
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
        "dj_phrase_policy": phrase_policy.score,
        "dj_post_drop": phrase_policy.post_drop_score,
        "dj_drop_landing": phrase_policy.drop_landing_score,
        "dj_role_path": phrase_policy.role_path_score,
        "dj_energy_arc": phrase_policy.energy_arc_score,
        "dj_boundary": phrase_policy.boundary_score,
        "dj_drop_guard": phrase_policy.drop_guard_score,
        "dj_intent_code": float(phrase_policy.intent_code),
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
    track: PreparedTrack,
    earliest_sample: int,
) -> np.ndarray:
    """Return every valid CUE-DETR candidate after the time floor.

    v1.2.11 forced exits into the final 42%/48 bars and entries into the first
    35%/32 bars. That produced predictable tail-to-head mixes but rejected
    stronger phrase boundaries elsewhere. Candidate direction is now expressed
    by CUE-DETR's IN/OUT scores and the pair scorer, not by song-position gates.
    """
    features = track.bar_features
    indices = np.asarray(track.structure.cue_indices, dtype=np.int64)
    indices = indices[(indices >= 0) & (indices < features.count)]
    if indices.size == 0:
        return indices
    indices = indices[features.start_samples[indices] >= earliest_sample]
    return np.unique(indices)


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


def _cue_centered_window(
    *,
    current: PreparedTrack,
    next_track: PreparedTrack,
    current_cue: int,
    next_cue: int,
    earliest_start: int,
    pre_roll_beats: float,
    release_beats: float,
) -> tuple[int, int, int, int, int]:
    """Return a common drum-bridge window around the two neural cue points.

    The selected CUE-DETR points remain the exact ownership-change instant.
    Normally four beats are exposed before the cue for a quantized percussion
    loop and two beats afterwards for the outgoing release. Near a track
    boundary or playback deadline either side is shortened without moving the
    neural cue or inventing another switch point.
    """
    sr = int(current.sample_rate)
    beat_samples = max(1, int(round(60.0 / max(current.playback_bpm, 1.0) * sr)))
    desired_pre = max(0, int(round(float(pre_roll_beats) * beat_samples)))
    desired_post = max(1, int(round(float(release_beats) * beat_samples)))

    available_pre = min(
        max(0, int(current_cue) - int(max(0, earliest_start))),
        max(0, int(next_cue)),
        desired_pre,
    )
    available_post = min(
        max(0, current.total_samples - int(current_cue)),
        max(0, next_track.total_samples - int(next_cue)),
        desired_post,
    )
    if available_post < 1:
        raise RuntimeError("CUE-DETR cue 点之后没有足够音频完成平滑交接。")

    current_start = int(current_cue) - int(available_pre)
    next_start = int(next_cue) - int(available_pre)
    length = int(available_pre + available_post)
    handoff_offset = int(available_pre)
    next_resume = int(next_cue) + int(available_post)
    return current_start, next_start, length, handoff_offset, next_resume


def find_best_transition(
    current: PreparedTrack,
    next_track: PreparedTrack,
    earliest_start: int = 0,
    requested_bars: int = 0,
    force_current_start: int | None = None,
    config: MatcherConfig | None = None,
    fx_config: TransitionFXConfig | None = None,
) -> TransitionPlan:
    """Search only CUE-DETR neural cues, then rank compatible cue pairs."""
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
            current,
            earliest_sample=earliest_start,
        )
    else:
        neural = np.asarray(current.structure.cue_indices, dtype=np.int64)
        neural = neural[(neural >= 0) & (neural < a.count)]
        future = neural[a.start_samples[neural] >= force_current_start]
        if future.size:
            exit_indices = np.asarray([int(future[0])], dtype=np.int64)
        else:
            raise RuntimeError("CUE-DETR 在请求位置之后没有可用 cue 点。")

    entry_indices = _candidate_indices(
        next_track,
        earliest_sample=0,
    )

    beat_samples = max(
        1,
        int(round(60.0 / max(current.playback_bpm, 1.0) * current.sample_rate)),
    )
    required_pre = int(round(config.pre_roll_beats * beat_samples))
    required_post = int(round(config.release_beats * beat_samples))

    def feasible(
        indices: np.ndarray,
        track: PreparedTrack,
        floor: int,
    ) -> np.ndarray:
        starts = track.bar_features.start_samples[indices]
        mask = (
            (starts - required_pre >= int(floor))
            & (starts + required_post < track.total_samples)
        )
        preferred = indices[mask]
        # Preserve CUE-DETR-only behavior for unusually short tracks: if no cue
        # can hold the full drum-bridge window, keep the neural cues and let the
        # window helper shorten symmetrically instead of inventing a new point.
        return preferred if preferred.size else indices

    exit_indices = feasible(exit_indices, current, earliest_start)
    entry_indices = feasible(entry_indices, next_track, 0)
    exit_scores = np.asarray(current.structure.mix_out_score, dtype=np.float32)
    entry_scores = np.asarray(next_track.structure.mix_in_score, dtype=np.float32)
    if force_current_start is None:
        exit_indices = _rank_by_cue_scores(
            exit_indices,
            exit_scores,
            config.max_exit_candidates,
        )
    entry_indices = _rank_by_cue_scores(
        entry_indices,
        entry_scores,
        config.max_entry_candidates,
    )
    if exit_indices.size == 0 or entry_indices.size == 0:
        raise RuntimeError("CUE-DETR 没有提供足够的 IN/OUT cue 点。")

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

    score, current_index, next_index, context_bars, metrics = best
    current_cue = int(a.start_samples[current_index])
    next_cue = int(b.start_samples[next_index])
    micro_offset, micro_confidence = estimate_micro_alignment(
        current, next_track, current_cue, next_cue
    )
    adjusted_next_cue = int(
        np.clip(next_cue + micro_offset, 0, max(0, next_track.total_samples - 1))
    )
    metrics["micro_align_ms"] = 1000.0 * micro_offset / current.sample_rate
    metrics["micro_align_confidence"] = micro_confidence

    current_start, next_start, length, handoff_offset, next_resume = _cue_centered_window(
        current=current,
        next_track=next_track,
        current_cue=current_cue,
        next_cue=adjusted_next_cue,
        earliest_start=earliest_start,
        pre_roll_beats=config.pre_roll_beats,
        release_beats=config.release_beats,
    )
    handoff_phase = float(handoff_offset / max(length, 1))
    transition_beats = float(
        length * max(current.playback_bpm, 1.0) / (60.0 * current.sample_rate)
    )
    metrics["matching_context_bars"] = float(context_bars)
    metrics["transition_beats"] = transition_beats
    metrics["pre_roll_beats"] = float(
        handoff_offset * max(current.playback_bpm, 1.0) / (60.0 * current.sample_rate)
    )
    metrics["release_beats"] = max(0.0, transition_beats - metrics["pre_roll_beats"])
    metrics["drum_bridge_beats"] = metrics["pre_roll_beats"]
    metrics["drum_loop_beats"] = min(2.0, max(1.0, metrics["pre_roll_beats"]))
    metrics["cue_handoff_phase"] = handoff_phase
    metrics["cue_handoff_sample_a"] = float(current_cue)
    metrics["cue_handoff_sample_b"] = float(adjusted_next_cue)

    fade_out, fade_in, bass_out, bass_in, high_out, high_in = _make_curves(
        length,
        metrics.get("local_gain", 1.0),
        fx_config,
    )
    phrase_policy = evaluate_phrase_policy(
        current,
        next_track,
        current_index,
        next_index,
        context_bars,
        harmonic=float(metrics.get("harmonic", 0.5)),
        bass_clean=float(metrics.get("bass_clean", 0.5)),
    )
    return TransitionPlan(
        current_start=current_start,
        next_start=next_start,
        length=length,
        bars=1,
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
        transition_mode="Cue-Centered EQ/Fader",
        switch_position=handoff_phase,
        next_resume_sample=next_resume,
        micro_offset_samples=int(micro_offset),
        current_cue_sample=current_cue,
        next_cue_sample=adjusted_next_cue,
        handoff_offset_samples=handoff_offset,
        dj_intent=phrase_policy.intent,
        current_role=phrase_policy.current_role,
        current_landing_role=phrase_policy.current_landing_role,
        next_role=phrase_policy.next_role,
        next_landing_role=phrase_policy.next_landing_role,
        structure_policy_score=phrase_policy.score,
        recommended_archetypes=phrase_policy.recommended_archetypes,
    )
