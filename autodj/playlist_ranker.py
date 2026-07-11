from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .models import AllInOneProfile, MuQProfile, TrackAnalysis
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


def _edge_labels(profile: AllInOneProfile | None) -> tuple[str, str]:
    if profile is None or not profile.available:
        return "unknown", "unknown"
    labels = [segment.label.lower() for segment in profile.segments]
    labels = [label for label in labels if label not in _IGNORE_LABELS]
    if not labels:
        return "unknown", "unknown"
    return labels[0], labels[-1]


def _structure_score(
    outgoing: AllInOneProfile | None,
    incoming: AllInOneProfile | None,
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
    structure_a: AllInOneProfile | None = None,
    structure_b: AllInOneProfile | None = None,
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


def _energy_arc_term(previous_delta: float, next_delta: float) -> float:
    """Second-order energy context for a three-track window.

    Curated sequencing studies report frequent alternation between rising and
    falling energy.  The old implementation accidentally penalized exactly that
    pattern.  This term mildly rewards a controlled reversal, penalizes two
    consecutive large moves in the same direction, and still limits abrupt
    acceleration so the result does not become a saw-tooth sequence.
    """
    previous = float(previous_delta)
    following = float(next_delta)
    acceleration = abs(following - previous)
    value = -0.055 * min(1.0, acceleration / 0.60)

    quiet = 0.035
    if abs(previous) < quiet or abs(following) < quiet:
        return float(value)
    if previous * following < 0.0:
        controlled = min(abs(previous), abs(following))
        value += 0.075 * min(1.0, controlled / 0.28)
    else:
        sustained = min(abs(previous), abs(following))
        value -= 0.065 * min(1.0, sustained / 0.28)
    return float(value)


def _path_objective(
    path: Sequence[int],
    scores: dict[tuple[int, int], PairScore],
    energies: Sequence[float],
) -> tuple[float, float]:
    """Return the exact beam objective and weakest edge for a complete path."""
    if len(path) <= 1:
        return 0.0, 1.0
    value = 0.0
    weakest = 1.0
    for index in range(1, len(path)):
        prev = int(path[index - 1])
        current = int(path[index])
        pair = scores[prev, current]
        weakest = min(weakest, pair.total)
        value += math.log(max(pair.total, 1e-4))
        value += 0.28 * math.log(max(weakest, 1e-4))
        if index >= 2:
            before = int(path[index - 2])
            previous_delta = energies[prev] - energies[before]
            next_delta = energies[current] - energies[prev]
            value += _energy_arc_term(previous_delta, next_delta)
    return float(value), float(weakest)


def _future_potential(
    node: int,
    used_mask: int,
    count: int,
    scores: dict[tuple[int, int], PairScore],
) -> float:
    """Two-step continuation estimate for ranking incomplete beam states.

    The estimate is deliberately optimistic and never enters the final path
    objective.  Looking one edge farther prevents the beam from over-valuing a
    node with one attractive incoming edge but no usable continuation.
    """
    remaining = [
        nxt for nxt in range(count)
        if not used_mask & (1 << nxt) and nxt != node
    ]
    if not remaining:
        return 0.0

    route_values: list[float] = []
    direct_values: list[float] = []
    for nxt in remaining:
        direct = math.log(max(scores[node, nxt].total, 1e-4))
        direct_values.append(direct)
        onward = [
            math.log(max(scores[nxt, after].total, 1e-4))
            for after in remaining
            if after != nxt
        ]
        best_onward = max(onward) if onward else direct
        route_values.append(direct + 0.52 * best_onward)

    route_values.sort(reverse=True)
    direct_values.sort(reverse=True)
    best_route = route_values[0]
    second_route = route_values[1] if len(route_values) > 1 else best_route
    best_direct = direct_values[0]
    return 0.105 * best_route + 0.035 * second_route + 0.025 * best_direct


def _prune_dominated_states(
    states: Sequence[tuple[float, tuple[int, ...], int, float]],
    max_frontier: int | None = 2,
) -> list[tuple[float, tuple[int, ...], int, float]]:
    """Deduplicate equivalent search states while preserving bottleneck Pareto points."""
    kept: dict[
        tuple[int, int, int],
        list[tuple[float, tuple[int, ...], int, float]],
    ] = {}
    for state in states:
        value, path, used_mask, weakest = state
        last = path[-1]
        previous = path[-2] if len(path) >= 2 else -1
        key = (used_mask, previous, last)
        frontier = kept.setdefault(key, [])
        if any(old[0] >= value and old[3] >= weakest for old in frontier):
            continue
        frontier[:] = [
            old for old in frontier
            if not (value >= old[0] and weakest >= old[3])
        ]
        frontier.append(state)
        frontier.sort(
            key=lambda item: (
                item[0] + 0.12 * math.log(max(item[3], 1e-4)),
                item[3],
                item[1],
            ),
            reverse=True,
        )
        if max_frontier is not None:
            del frontier[max(1, int(max_frontier)) :]
    return [state for frontier in kept.values() for state in frontier]


def _state_rank(
    state: tuple[float, tuple[int, ...], int, float],
    count: int,
    scores: dict[tuple[int, int], PairScore],
) -> tuple[float, float, float, tuple[int, ...]]:
    value, path, used_mask, weakest = state
    return (
        value + _future_potential(path[-1], used_mask, count, scores),
        weakest,
        value,
        path,
    )


def _select_diverse_beam(
    candidates: Sequence[tuple[float, tuple[int, ...], int, float]],
    width: int,
    count: int,
    scores: dict[tuple[int, int], PairScore],
) -> list[tuple[float, tuple[int, ...], int, float]]:
    """Keep a small endpoint-diverse reserve before filling by global rank.

    Standard beam search often spends most of its budget on near-duplicate
    prefixes.  Reserving only a fraction of slots for distinct final nodes keeps
    alternative graph regions alive without changing the deterministic result.
    """
    ordered = sorted(
        candidates,
        key=lambda item: _state_rank(item, count, scores),
        reverse=True,
    )
    target = max(1, int(width))
    reserve = min(len(ordered), count, max(1, target // 5))
    selected: list[tuple[float, tuple[int, ...], int, float]] = []
    selected_paths: set[tuple[int, ...]] = set()
    seen_last: set[int] = set()

    for state in ordered:
        last = state[1][-1]
        if last in seen_last:
            continue
        selected.append(state)
        selected_paths.add(state[1])
        seen_last.add(last)
        if len(selected) >= reserve:
            break

    for state in ordered:
        if len(selected) >= target:
            break
        if state[1] in selected_paths:
            continue
        selected.append(state)
        selected_paths.add(state[1])
    return selected


def _exact_rank_small(
    count: int,
    start_index: int,
    scores: dict[tuple[int, int], PairScore],
    energies: Sequence[float],
) -> list[int]:
    """Exact subset-DP search for small playlists.

    States are keyed by used set and the final two tracks because the objective
    contains a three-track energy-context term.  All Pareto-nondominated
    accumulated-score/weakest-edge variants are retained, making this exact for
    the implemented objective while the playlist remains small.
    """
    states: list[tuple[float, tuple[int, ...], int, float]] = [
        (0.0, (start_index,), 1 << start_index, 1.0)
    ]
    for _ in range(count - 1):
        candidates: list[tuple[float, tuple[int, ...], int, float]] = []
        for value, path, used_mask, weakest in states:
            last = path[-1]
            for nxt in range(count):
                bit = 1 << nxt
                if used_mask & bit:
                    continue
                pair = scores[last, nxt]
                new_weakest = min(weakest, pair.total)
                new_value = value + math.log(max(pair.total, 1e-4))
                new_value += 0.28 * math.log(max(new_weakest, 1e-4))
                if len(path) >= 2:
                    prev = path[-2]
                    new_value += _energy_arc_term(
                        energies[last] - energies[prev],
                        energies[nxt] - energies[last],
                    )
                candidates.append(
                    (new_value, path + (nxt,), used_mask | bit, new_weakest)
                )
        states = _prune_dominated_states(candidates, max_frontier=None)
    best = max(states, key=lambda item: (item[0], item[3], item[1]))
    return list(best[1])


def _improve_path(
    path: Sequence[int],
    scores: dict[tuple[int, int], PairScore],
    energies: Sequence[float],
    max_passes: int = 2,
) -> list[int]:
    """Deterministic swap/relocate polish that keeps the requested first track."""
    best = list(path)
    best_value, best_weakest = _path_objective(best, scores, energies)
    count = len(best)
    # Full local search is intentionally bounded; larger playlists already gain
    # most of the benefit from lookahead and transposition pruning.
    if count > 36:
        return best

    for _ in range(max(0, int(max_passes))):
        improved = False
        candidate_best = best
        candidate_value = best_value
        candidate_weakest = best_weakest

        for left in range(1, count - 1):
            for right in range(left + 1, count):
                candidate = best.copy()
                candidate[left], candidate[right] = candidate[right], candidate[left]
                value, weakest = _path_objective(candidate, scores, energies)
                if (value, weakest, tuple(candidate)) > (
                    candidate_value, candidate_weakest, tuple(candidate_best)
                ):
                    candidate_best = candidate
                    candidate_value = value
                    candidate_weakest = weakest

        for source in range(1, count):
            item = best[source]
            reduced = best[:source] + best[source + 1 :]
            for target in range(1, count):
                candidate = reduced[:target] + [item] + reduced[target:]
                value, weakest = _path_objective(candidate, scores, energies)
                if (value, weakest, tuple(candidate)) > (
                    candidate_value, candidate_weakest, tuple(candidate_best)
                ):
                    candidate_best = candidate
                    candidate_value = value
                    candidate_weakest = weakest

        if (candidate_value, candidate_weakest) > (best_value, best_weakest):
            best = candidate_best
            best_value = candidate_value
            best_weakest = candidate_weakest
            improved = True
        if not improved:
            break
    return best


def rank_playlist(
    tracks: Sequence[TrackAnalysis],
    profiles: Sequence[MuQProfile],
    start_index: int = 0,
    beam_width: int = 192,
    structures: Sequence[AllInOneProfile] | None = None,
) -> tuple[list[int], dict[tuple[int, int], PairScore]]:
    """Directional graph search with lookahead, state pruning and local polish."""
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
    neighbors = {
        source: sorted(
            (target for target in range(count) if target != source),
            key=lambda target: (scores[source, target].total, -target),
            reverse=True,
        )
        for source in range(count)
    }

    # Subset dynamic programming is exact and inexpensive for small queues.
    # Larger queues retain bounded beam search and local path polish.
    if count <= 10:
        return _exact_rank_small(count, start_index, scores, energies), scores

    # State: exact objective, path, bit-mask, weakest edge so far.
    states: list[tuple[float, tuple[int, ...], int, float]] = [
        (0.0, (start_index,), 1 << start_index, 1.0)
    ]
    width = max(12, int(beam_width))
    for _ in range(count - 1):
        candidates: list[tuple[float, tuple[int, ...], int, float]] = []
        for value, path, used_mask, weakest in states:
            last = path[-1]
            for nxt in neighbors[last]:
                bit = 1 << nxt
                if used_mask & bit:
                    continue
                pair = scores[last, nxt]
                new_weakest = min(weakest, pair.total)
                new_value = value + math.log(max(pair.total, 1e-4))
                new_value += 0.28 * math.log(max(new_weakest, 1e-4))

                if len(path) >= 2:
                    prev = path[-2]
                    new_value += _energy_arc_term(
                        energies[last] - energies[prev],
                        energies[nxt] - energies[last],
                    )

                candidates.append(
                    (new_value, path + (nxt,), used_mask | bit, new_weakest)
                )

        candidates = _prune_dominated_states(candidates)
        states = _select_diverse_beam(candidates, width, count, scores)

    best_state = max(states, key=lambda item: (item[0], item[3], item[1]))
    polished = _improve_path(best_state[1], scores, energies)
    return polished, scores

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
