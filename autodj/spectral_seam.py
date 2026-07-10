from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import librosa
import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.csgraph import maximum_flow


@dataclass(frozen=True)
class SpectralSeamResult:
    audio: np.ndarray
    seam_seconds_mean: float
    seam_seconds_std: float
    flow_value: float


def _graph_cut_seam(cost: np.ndarray) -> tuple[np.ndarray, float]:
    """Find a left-to-right frequency-dependent seam using an s/t min-cut."""
    freq_bins, time_bins = cost.shape
    node_count = freq_bins * time_bins
    source = node_count
    sink = node_count + 1
    rows: list[int] = []
    cols: list[int] = []
    capacities: list[int] = []

    normalized = np.asarray(cost, dtype=np.float64)
    normalized -= float(np.min(normalized))
    maximum = float(np.max(normalized))
    if maximum > 1e-12:
        normalized /= maximum

    def add_edge(a: int, b: int, capacity: int) -> None:
        rows.append(a)
        cols.append(b)
        capacities.append(max(1, int(capacity)))

    def node(frequency: int, time: int) -> int:
        return frequency * time_bins + time

    # Low spectral difference should be cheap to cut. Vertical edges carry a
    # slightly larger smoothness term to discourage jagged frequency seams.
    for f in range(freq_bins):
        for t in range(time_bins):
            here = node(f, t)
            if t + 1 < time_bins:
                other = node(f, t + 1)
                cap = 1 + int(2200.0 * (normalized[f, t] + normalized[f, t + 1]) / 2.0)
                add_edge(here, other, cap)
                add_edge(other, here, cap)
            if f + 1 < freq_bins:
                other = node(f + 1, t)
                cap = 40 + int(3100.0 * (normalized[f, t] + normalized[f + 1, t]) / 2.0)
                add_edge(here, other, cap)
                add_edge(other, here, cap)

    infinite = 10_000_000
    for f in range(freq_bins):
        add_edge(source, node(f, 0), infinite)
        add_edge(node(f, time_bins - 1), sink, infinite)

    graph = sparse.coo_array(
        (np.asarray(capacities, dtype=np.int64), (rows, cols)),
        shape=(node_count + 2, node_count + 2),
        dtype=np.int64,
    ).tocsr()
    result = maximum_flow(graph, source, sink, method="dinic")
    residual = (graph - result.flow).tocsr()

    reachable = np.zeros(node_count + 2, dtype=bool)
    reachable[source] = True
    queue: deque[int] = deque([source])
    while queue:
        current = queue.popleft()
        start, end = residual.indptr[current], residual.indptr[current + 1]
        for neighbor, capacity in zip(
            residual.indices[start:end], residual.data[start:end]
        ):
            if capacity > 0 and not reachable[neighbor]:
                reachable[neighbor] = True
                queue.append(int(neighbor))

    source_side = reachable[:node_count].reshape(freq_bins, time_bins)
    seam = np.zeros(freq_bins, dtype=np.float64)
    for f in range(freq_bins):
        sink_positions = np.flatnonzero(~source_side[f])
        seam[f] = float(sink_positions[0]) if sink_positions.size else float(time_bins - 1)
    seam = ndimage.median_filter(seam, size=5, mode="nearest")
    seam = ndimage.gaussian_filter1d(seam, sigma=1.1, mode="nearest")
    return seam, float(result.flow_value)


def spectral_seam_crossfade(
    outgoing: np.ndarray,
    incoming: np.ndarray,
    sample_rate: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    graph_freq_bins: int = 56,
    graph_time_bins: int = 112,
) -> SpectralSeamResult:
    """
    Beat-aligned time-frequency transition inspired by Robinson & Brown.

    The graph cut chooses a different switch time for each frequency region,
    allowing bass, mids and highs to change ownership at different moments.
    """
    length = min(len(outgoing), len(incoming))
    if length < n_fft * 2:
        phase = np.linspace(0.0, np.pi / 2.0, max(length, 1), dtype=np.float32)
        audio = (
            outgoing[:length] * np.cos(phase)[:, None]
            + incoming[:length] * np.sin(phase)[:, None]
        )
        return SpectralSeamResult(
            audio=np.ascontiguousarray(audio, dtype=np.float32),
            seam_seconds_mean=length / max(sample_rate, 1) / 2.0,
            seam_seconds_std=0.0,
            flow_value=0.0,
        )

    a = np.ascontiguousarray(outgoing[:length, :2], dtype=np.float32)
    b = np.ascontiguousarray(incoming[:length, :2], dtype=np.float32)
    stft_a = np.stack(
        [librosa.stft(a[:, channel], n_fft=n_fft, hop_length=hop_length) for channel in range(2)],
        axis=0,
    )
    stft_b = np.stack(
        [librosa.stft(b[:, channel], n_fft=n_fft, hop_length=hop_length) for channel in range(2)],
        axis=0,
    )
    magnitude_a = np.mean(np.abs(stft_a), axis=0)
    magnitude_b = np.mean(np.abs(stft_b), axis=0)
    db_a = librosa.amplitude_to_db(magnitude_a + 1e-7, ref=np.max)
    db_b = librosa.amplitude_to_db(magnitude_b + 1e-7, ref=np.max)
    difference = np.abs(db_a - db_b)

    full_f, full_t = difference.shape
    coarse_f = min(graph_freq_bins, full_f)
    coarse_t = min(graph_time_bins, full_t)
    zoom = (coarse_f / full_f, coarse_t / full_t)
    coarse_cost = ndimage.zoom(difference, zoom=zoom, order=1, mode="nearest")
    coarse_cost = coarse_cost[:coarse_f, :coarse_t]

    try:
        seam_coarse, flow_value = _graph_cut_seam(coarse_cost)
    except Exception:
        # Monotonic low-cost fallback if scipy's max-flow is unavailable.
        seam_coarse = np.argmin(coarse_cost, axis=1).astype(np.float64)
        seam_coarse = ndimage.median_filter(seam_coarse, size=7, mode="nearest")
        flow_value = 0.0

    coarse_freq_axis = np.linspace(0.0, 1.0, len(seam_coarse))
    full_freq_axis = np.linspace(0.0, 1.0, full_f)
    seam_frames = np.interp(full_freq_axis, coarse_freq_axis, seam_coarse)
    seam_frames *= (full_t - 1) / max(coarse_t - 1, 1)

    time_axis = np.arange(full_t, dtype=np.float64)[None, :]
    softness = max(1.3, full_t * 0.014)
    local_progress = 0.5 + 0.5 * np.tanh((time_axis - seam_frames[:, None]) / softness)
    global_progress = np.linspace(0.0, 1.0, full_t, dtype=np.float64)[None, :]
    progress = np.clip(0.78 * local_progress + 0.22 * global_progress, 0.0, 1.0)
    progress[:, 0] = 0.0
    progress[:, -1] = 1.0

    gain_a = np.cos(progress * np.pi / 2.0).astype(np.float32)
    gain_b = np.sin(progress * np.pi / 2.0).astype(np.float32)
    mixed_stft = stft_a * gain_a[None, :, :] + stft_b * gain_b[None, :, :]

    channels = [
        librosa.istft(mixed_stft[channel], hop_length=hop_length, length=length)
        for channel in range(2)
    ]
    audio = np.column_stack(channels).astype(np.float32, copy=False)

    # Keep the first and last few milliseconds exactly anchored to the source
    # tracks, reducing edge clicks when the rendered buffer joins live audio.
    edge = min(length // 8, max(32, int(0.012 * sample_rate)))
    if edge > 1:
        fade = np.linspace(0.0, 1.0, edge, dtype=np.float32)
        audio[:edge] = a[:edge] * (1.0 - fade[:, None]) + audio[:edge] * fade[:, None]
        audio[-edge:] = audio[-edge:] * (1.0 - fade[:, None]) + b[-edge:] * fade[:, None]

    peak = float(np.max(np.abs(audio)) + 1e-9)
    if peak > 0.98:
        audio *= np.float32(0.98 / peak)

    seam_seconds = seam_frames * hop_length / float(sample_rate)
    return SpectralSeamResult(
        audio=np.ascontiguousarray(audio, dtype=np.float32),
        seam_seconds_mean=float(np.mean(seam_seconds)),
        seam_seconds_std=float(np.std(seam_seconds)),
        flow_value=flow_value,
    )
