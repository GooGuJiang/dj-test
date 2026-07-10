from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .models import SongFormerProfile, MuQProfile, TrackAnalysis
from .muq_analyzer import cosine_similarity


@dataclass(frozen=True)
class PairScore:
    total: float
    style: float
    directional: float
    trajectory: float
    tempo: float
    acoustic: float
    energy: float = 0.5
    structure: float = 0.5
    jump_penalty: float = 0.0


_IGNORE_LABELS = {"start", "end", "unknown", ""}


def _related_bpm(source: float, target: float) -> float:
    candidates = np.asarray([source * 0.5, source, source * 2.0], dtype=np.float64)
    distance = np.abs(np.log2(np.maximum(candidates, 1e-6) / max(target, 1e-6)))
    return float(candidates[int(np.argmin(distance))])


def _sequence_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Small monotonic-DTW similarity for MuQ segment trajectories."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.ndim != 2 or b.ndim != 2 or not len(a) or not len(b) or a.shape[1] != b.shape[1]:
        return 0.0
    cost = np.full((len(a) + 1, len(b) + 1), np.inf, dtype=np.float64)
    cost[0, 0] = 0.0
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            similarity = cosine_similarity(a[i - 1], b[j - 1])
            local = 1.0 - similarity
            cost[i, j] = local + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    normalized = cost[-1, -1] / max(len(a), len(b))
    return float(np.clip(np.exp(-2.4 * normalized), 0.0, 1.0))


def _profile_energy(profile: MuQProfile) -> float:
    if profile.acoustic_features.size < 1:
        return 0.0
    # MuQAnalyzer stores log10 mean RMS as the first feature.
    return float(profile.acoustic_features[0])


def _acoustic_subscores(a: MuQProfile, b: MuQProfile) -> tuple[float, float]:
    if (
        not a.acoustic_features.size
        or a.acoustic_features.shape != b.acoustic_features.shape
        or a.acoustic_features.size < 6
    ):
        return 0.5, 0.5
    av = np.asarray(a.acoustic_features, dtype=np.float64)
    bv = np.asarray(b.acoustic_features, dtype=np.float64)
    energy = float(np.exp(-2.8 * abs(av[0] - bv[0])))
    # onset, centroid, flatness, low-frequency ratio and dynamics.
    weights = np.asarray([1.2, 1.0, 0.8, 1.2, 0.8], dtype=np.float64)
    distance = float(np.sqrt(np.mean(weights * np.square(av[1:6] - bv[1:6]))))
    acoustic = float(np.exp(-2.0 * distance))
    return float(np.clip(acoustic, 0.0, 1.0)), float(np.clip(energy, 0.0, 1.0))


def _edge_labels(profile: SongFormerProfile | None) -> tuple[str, str]:
    if profile is None or not profile.available:
        return "unknown", "unknown"
    labels = [segment.label.lower() for segment in profile.segments]
    labels = [label for label in labels if label not in _IGNORE_LABELS]
    if not labels:
        return "unknown", "unknown"
    return labels[0], labels[-1]


def _structure_score(
    outgoing: SongFormerProfile | None,
    incoming: SongFormerProfile | None,
) -> float:
    in_a, out_a = _edge_labels(outgoing)
    in_b, out_b = _edge_labels(incoming)
    del in_a, out_b
    if out_a == "unknown" or in_b == "unknown":
        return 0.5

    # Functional directions that commonly produce controllable DJ transitions.
    preferred: dict[tuple[str, str], float] = {
        ("outro", "intro"): 1.00,
        ("outro", "verse"): 0.88,
        ("outro", "chorus"): 0.76,
        ("break", "chorus"): 0.92,
        ("bridge", "chorus"): 0.90,
        ("inst", "intro"): 0.90,
        ("solo", "intro"): 0.70,
        ("verse", "intro"): 0.80,
        ("chorus", "intro"): 0.82,
        ("chorus", "verse"): 0.75,
        ("break", "intro"): 0.86,
    }
    if (out_a, in_b) in preferred:
        return preferred[(out_a, in_b)]
    if out_a == in_b:
        return 0.70
    if out_a in {"outro", "inst", "break", "bridge"}:
        return 0.72
    if in_b in {"intro", "inst", "verse"}:
        return 0.68
    # Vocal-dense to vocal-dense jumps are possible, but less forgiving.
    if out_a in {"verse", "chorus", "solo"} and in_b in {"verse", "chorus", "solo"}:
        return 0.48
    return 0.58


def transition_compatibility(
    outgoing: TrackAnalysis,
    incoming: TrackAnalysis,
    profile_a: MuQProfile,
    profile_b: MuQProfile,
    structure_a: SongFormerProfile | None = None,
    structure_b: SongFormerProfile | None = None,
) -> PairScore:
    """Directional A→B compatibility for playlist ordering."""
    if profile_a.available and profile_b.available:
        style = cosine_similarity(profile_a.global_embedding, profile_b.global_embedding)
        directional = cosine_similarity(profile_a.outro_embedding, profile_b.intro_embedding)
        seq_a = profile_a.timeline_embeddings[-min(3, len(profile_a.timeline_embeddings)) :]
        seq_b = profile_b.timeline_embeddings[: min(3, len(profile_b.timeline_embeddings))]
        trajectory = _sequence_similarity(seq_a, seq_b)
    else:
        style = directional = trajectory = 0.5

    interpreted = _related_bpm(incoming.bpm, outgoing.bpm)
    ratio = max(interpreted, 1e-6) / max(outgoing.bpm, 1e-6)
    tempo = float(np.exp(-9.0 * abs(math.log(ratio))))
    acoustic, energy = _acoustic_subscores(profile_a, profile_b)
    structure = _structure_score(structure_a, structure_b)

    # Explicitly punish a single very large stylistic/tempo/energy leap. This is
    # important because a pure sum can hide one bad edge behind several good ones.
    jump_penalty = 0.0
    jump_penalty += 0.22 * max(0.0, 0.52 - style) / 0.52
    jump_penalty += 0.24 * max(0.0, 0.58 - directional) / 0.58
    jump_penalty += 0.20 * max(0.0, 0.58 - tempo) / 0.58
    jump_penalty += 0.14 * max(0.0, 0.50 - energy) / 0.50
    jump_penalty += 0.10 * max(0.0, 0.48 - structure) / 0.48

    total = (
        0.18 * style
        + 0.25 * directional
        + 0.16 * trajectory
        + 0.15 * tempo
        + 0.08 * acoustic
        + 0.09 * energy
        + 0.09 * structure
        - jump_penalty
    )
    return PairScore(
        total=float(np.clip(total, 0.0, 1.0)),
        style=float(style),
        directional=float(directional),
        trajectory=float(trajectory),
        tempo=float(tempo),
        acoustic=float(acoustic),
        energy=float(energy),
        structure=float(structure),
        jump_penalty=float(np.clip(jump_penalty, 0.0, 1.0)),
    )


def rank_playlist(
    tracks: Sequence[TrackAnalysis],
    profiles: Sequence[MuQProfile],
    start_index: int = 0,
    beam_width: int = 192,
    structures: Sequence[SongFormerProfile] | None = None,
) -> tuple[list[int], dict[tuple[int, int], PairScore]]:
    """Beam-search ordering with weak-edge and energy-zigzag penalties."""
    count = len(tracks)
    if count != len(profiles):
        raise ValueError("tracks 和 profiles 数量不一致。")
    if structures is not None and len(structures) != count:
        raise ValueError("tracks 和 structures 数量不一致。")
    if count <= 1:
        return list(range(count)), {}
    if not 0 <= start_index < count:
        raise IndexError("start_index 超出范围。")

    structure_list = list(structures) if structures is not None else [None] * count
    scores: dict[tuple[int, int], PairScore] = {}
    for i in range(count):
        for j in range(count):
            if i != j:
                scores[i, j] = transition_compatibility(
                    tracks[i], tracks[j], profiles[i], profiles[j],
                    structure_list[i], structure_list[j],
                )

    energies = [_profile_energy(profile) for profile in profiles]
    # State: objective, path, used, weakest edge so far.
    states: list[tuple[float, tuple[int, ...], frozenset[int], float]] = [
        (0.0, (start_index,), frozenset({start_index}), 1.0)
    ]
    width = max(12, int(beam_width))
    for _ in range(count - 1):
        candidates: list[tuple[float, tuple[int, ...], frozenset[int], float]] = []
        for value, path, used, weakest in states:
            last = path[-1]
            for nxt in range(count):
                if nxt in used:
                    continue
                pair = scores[last, nxt]
                new_weakest = min(weakest, pair.total)
                # Log objective makes one weak edge visible. The extra bottleneck
                # term directly optimizes the worst transition in the sequence.
                new_value = value + math.log(max(pair.total, 1e-4))
                new_value += 0.28 * math.log(max(new_weakest, 1e-4))

                # Avoid machine-like energy zigzags (hard up, hard down, hard up).
                if len(path) >= 2:
                    prev = path[-2]
                    previous_delta = energies[last] - energies[prev]
                    next_delta = energies[nxt] - energies[last]
                    if previous_delta * next_delta < 0.0:
                        new_value -= 0.18 * min(
                            1.0, abs(previous_delta - next_delta) / 0.45
                        )
                    acceleration = abs(next_delta - previous_delta)
                    new_value -= 0.08 * min(1.0, acceleration / 0.55)

                candidates.append(
                    (new_value, path + (nxt,), used | {nxt}, new_weakest)
                )
        candidates.sort(key=lambda item: (item[0], item[3], item[1]), reverse=True)
        states = candidates[:width]
    best = max(states, key=lambda item: (item[0], item[3]))
    return list(best[1]), scores


def style_clusters(profiles: Sequence[MuQProfile]) -> list[int]:
    """Small deterministic k-means for UI style groups, not genre labels."""
    valid = [profile.global_embedding for profile in profiles if profile.available]
    if not valid:
        return [0] * len(profiles)
    dimension = valid[0].size
    matrix = np.stack(
        [
            profile.global_embedding
            if profile.available and profile.global_embedding.size == dimension
            else np.zeros(dimension, dtype=np.float32)
            for profile in profiles
        ]
    ).astype(np.float32)
    count = len(matrix)
    k = int(np.clip(round(np.sqrt(count)), 1, min(5, count)))
    if k == 1:
        return [1] * count

    centers = [matrix[0]]
    for _ in range(1, k):
        distances = np.min(
            np.stack([1.0 - matrix @ center for center in centers]), axis=0
        )
        centers.append(matrix[int(np.argmax(distances))])
    centers_array = np.stack(centers)
    labels = np.zeros(count, dtype=np.int32)
    for iteration in range(20):
        similarity = matrix @ centers_array.T
        new_labels = np.argmax(similarity, axis=1).astype(np.int32)
        if np.array_equal(new_labels, labels) and iteration > 0:
            break
        labels = new_labels
        for index in range(k):
            members = matrix[labels == index]
            if len(members):
                center = np.mean(members, axis=0)
                norm = np.linalg.norm(center)
                if norm > 1e-9:
                    centers_array[index] = center / norm
    return [int(value) + 1 for value in labels]
