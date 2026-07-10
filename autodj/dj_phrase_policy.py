from __future__ import annotations

"""Structure-aware DJ phrase policy.

The matcher in this project already estimates local audio compatibility.  This
module adds a second layer that models *when a human DJ would actually make the
move*. It scores phrase boundaries, energy arcs, vocal overlap and bass safety,
then maps every structural situation onto one of three conservative renderers:
long blend, bass handover, or vocal echo exit.

The implementation is deterministic and bar-synchronous.  It is inspired by
public cue-point, phrasing, Raveform and real-DJ-mix analysis research, but does
not claim to reproduce a proprietary DJ policy.
"""

from dataclasses import dataclass

import numpy as np

from .models import PreparedTrack


INTENT_CODES = {
    "Safe Phrase Blend": 0,
    "Outro-Intro Blend": 1,
    "Energy Relay Blend": 2,
    "Breakdown Blend": 3,
    "Phrase Landing Blend": 4,
    "Bass Handover": 5,
    "Vocal Echo Exit": 6,
}


@dataclass(frozen=True)
class PhrasePolicyEvaluation:
    score: float
    intent: str
    current_role: str
    current_landing_role: str
    next_role: str
    next_landing_role: str
    post_drop_score: float
    drop_landing_score: float
    role_path_score: float
    energy_arc_score: float
    boundary_score: float
    drop_guard_score: float
    recommended_archetypes: tuple[str, ...]

    @property
    def intent_code(self) -> int:
        return int(INTENT_CODES.get(self.intent, 0))


def _unit(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values.astype(np.float32)
    low, high = np.percentile(values, [5.0, 95.0])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low + 1e-10:
        return np.full(values.shape, 0.5, dtype=np.float32)
    return np.clip((values - low) / (high - low), 0.0, 1.0).astype(np.float32)


def _role(track: PreparedTrack, index: int) -> str:
    labels = track.structure.labels
    if not labels:
        return "SECTION"
    index = int(np.clip(index, 0, len(labels) - 1))
    return str(labels[index]).upper()


def _functional(track: PreparedTrack, index: int) -> str:
    labels = track.structure.functional_labels
    if not labels:
        return "UNKNOWN"
    index = int(np.clip(index, 0, len(labels) - 1))
    return str(labels[index]).upper()


def _is_boundary(track: PreparedTrack, index: int) -> float:
    index = int(max(0, index))
    role = _role(track, index)
    functional = _functional(track, index)
    changed = 1.0 if index == 0 else float(
        role != _role(track, index - 1)
        or functional != _functional(track, index - 1)
    )
    neural = (
        float(track.structure.allin1_boundary_score[index])
        if track.structure.allin1_boundary_score.size > index
        else 0.5
    )
    phrase = (
        float(track.structure.phrase_mask[index])
        if track.structure.phrase_mask.size > index
        else 0.5
    )
    return float(np.clip(0.50 * changed + 0.28 * neural + 0.22 * phrase, 0.0, 1.0))


def _previous_drop_distance(track: PreparedTrack, index: int, maximum: int = 8) -> int | None:
    """Return bars since the closest previous DROP, excluding a current DROP."""
    if _role(track, index) == "DROP":
        return None
    for distance in range(1, maximum + 1):
        candidate = index - distance
        if candidate < 0:
            break
        if _role(track, candidate) == "DROP":
            return distance
    return None


def _bars_until_role_change(track: PreparedTrack, index: int, maximum: int = 16) -> int:
    role = _role(track, index)
    count = track.bar_features.count
    for distance in range(1, maximum + 1):
        candidate = index + distance
        if candidate >= count:
            return distance
        if _role(track, candidate) != role:
            return distance
    return maximum + 1


def _energy_arc(
    current: PreparedTrack,
    next_track: PreparedTrack,
    current_index: int,
    next_index: int,
    bars: int,
) -> float:
    a = _unit(current.bar_features.rms)
    b = _unit(next_track.bar_features.rms)
    a_seg = a[current_index : current_index + bars]
    b_seg = b[next_index : next_index + bars]
    if not len(a_seg) or not len(b_seg):
        return 0.5
    third = max(1, min(len(a_seg), len(b_seg)) // 3)
    a_fall = float(np.mean(a_seg[:third]) - np.mean(a_seg[-third:]))
    b_rise = float(np.mean(b_seg[-third:]) - np.mean(b_seg[:third]))
    landing_index = min(next_index + bars, len(b) - 1)
    landing_energy = float(b[landing_index]) if len(b) else 0.5
    # A gentle decline plus an incoming rise/arrival is usually more natural
    # than two simultaneous peaks.  Flat techno grooves still receive a usable
    # neutral score instead of being rejected.
    return float(
        np.clip(
            0.48
            + 0.24 * np.tanh(2.8 * a_fall)
            + 0.20 * np.tanh(2.8 * b_rise)
            + 0.08 * landing_energy,
            0.0,
            1.0,
        )
    )


def _role_path_score(
    current_role: str,
    current_landing: str,
    next_role: str,
    next_landing: str,
    post_drop: float,
) -> float:
    current_role = current_role.upper()
    current_landing = current_landing.upper()
    next_role = next_role.upper()
    next_landing = next_landing.upper()

    if post_drop >= 0.65 and next_role in {"INTRO", "BUILDUP", "PHRASE", "BREAKDOWN"} and next_landing in {"DROP", "CHORUS"}:
        return 1.0
    if current_role in {"BREAKDOWN", "BREAK", "BRIDGE"} and next_role in {"INTRO", "BUILDUP", "BREAKDOWN", "PHRASE"} and next_landing in {"DROP", "CHORUS"}:
        return 0.98
    if current_landing in {"DROP", "CHORUS"} and next_landing in {"DROP", "CHORUS"}:
        return 0.91
    if current_role == "OUTRO" and next_role in {"INTRO", "PHRASE", "SECTION"}:
        return 0.96
    if current_role in {"VOCAL", "VERSE", "CHORUS", "SOLO"} and next_role in {"INTRO", "PHRASE", "SECTION", "BREAKDOWN"}:
        return 0.82
    if next_landing in {"DROP", "CHORUS"} and next_role in {"INTRO", "BUILDUP", "BREAKDOWN", "PHRASE"}:
        return 0.88
    if current_role in {"VOCAL", "VERSE", "CHORUS"} and next_role in {"VOCAL", "VERSE", "CHORUS"}:
        return 0.18
    if current_role == "DROP" and next_role == "DROP":
        return 0.40
    if current_role in {"COOLDOWN", "OUTRO", "BREAKDOWN", "PHRASE"} and next_role in {"INTRO", "PHRASE", "SECTION"}:
        return 0.82
    return 0.58


def _recommended(intent: str) -> tuple[str, ...]:
    mapping = {
        "Energy Relay Blend": ("Bass Swap", "Short Blend", "Echo Out"),
        "Breakdown Blend": ("Short Blend", "Bass Swap", "Echo Out"),
        "Phrase Landing Blend": ("Short Blend", "Bass Swap", "Echo Out"),
        "Bass Handover": ("Bass Swap", "Short Blend", "Echo Out"),
        "Outro-Intro Blend": ("Short Blend", "Bass Swap", "Echo Out"),
        "Vocal Echo Exit": ("Echo Out", "Short Blend", "Bass Swap"),
        "Safe Phrase Blend": ("Short Blend", "Bass Swap", "Echo Out"),
    }
    return mapping.get(intent, mapping["Safe Phrase Blend"])


def _choose_intent(
    current_role: str,
    current_landing: str,
    next_role: str,
    next_landing: str,
    post_drop: float,
    harmonic: float,
    bass_clean: float,
) -> str:
    if (
        current_landing in {"DROP", "CHORUS"}
        and next_landing in {"DROP", "CHORUS"}
        and harmonic >= 0.58
        and bass_clean >= 0.50
    ):
        return "Bass Handover"
    if (
        post_drop >= 0.65
        and next_role in {"INTRO", "BUILDUP", "PHRASE", "BREAKDOWN"}
        and next_landing in {"DROP", "CHORUS"}
    ):
        return "Energy Relay Blend"
    if (
        current_role in {"BREAKDOWN", "BREAK", "BRIDGE"}
        and next_role in {"INTRO", "BUILDUP", "BREAKDOWN", "PHRASE"}
        and next_landing in {"DROP", "CHORUS"}
    ):
        return "Breakdown Blend"
    if current_role == "OUTRO" and next_role in {"INTRO", "PHRASE", "SECTION"}:
        return "Outro-Intro Blend"
    if (
        current_role in {"VOCAL", "VERSE", "CHORUS", "SOLO"}
        and next_role in {"INTRO", "PHRASE", "SECTION", "BREAKDOWN"}
    ):
        return "Vocal Echo Exit"
    if next_landing in {"DROP", "CHORUS"}:
        return "Phrase Landing Blend"
    return "Safe Phrase Blend"


def evaluate_phrase_policy(
    current: PreparedTrack,
    next_track: PreparedTrack,
    current_index: int,
    next_index: int,
    bars: int,
    *,
    harmonic: float = 0.5,
    bass_clean: float = 0.5,
) -> PhrasePolicyEvaluation:
    current_landing_index = min(
        current_index + bars, current.bar_features.count - 1
    )
    next_landing_index = min(
        next_index + bars, next_track.bar_features.count - 1
    )
    current_role = _role(current, current_index)
    current_landing = _role(current, current_landing_index)
    next_role = _role(next_track, next_index)
    next_landing = _role(next_track, next_landing_index)

    distance = _previous_drop_distance(current, current_index)
    post_drop = 0.0 if distance is None else float(np.exp(-0.34 * max(0, distance - 1)))
    if current_role in {"COOLDOWN", "BREAKDOWN", "OUTRO"} and distance is not None:
        post_drop = min(1.0, post_drop + 0.15)

    landing_boundary = _is_boundary(next_track, next_landing_index)
    if next_landing == "DROP":
        drop_landing = 0.70 + 0.30 * landing_boundary
    elif next_landing == "CHORUS":
        drop_landing = 0.58 + 0.27 * landing_boundary
    elif next_role == "DROP" and bars <= 4:
        drop_landing = 0.62
    else:
        drop_landing = 0.22 + 0.28 * landing_boundary
    drop_landing = float(np.clip(drop_landing, 0.0, 1.0))

    start_boundary = 0.5 * (
        _is_boundary(current, current_index) + _is_boundary(next_track, next_index)
    )
    boundary = float(np.clip(0.72 * start_boundary + 0.28 * landing_boundary, 0.0, 1.0))

    if current_role == "DROP":
        bars_until_change = _bars_until_role_change(current, current_index)
        # Starting a long blend halfway through a drop is strongly discouraged.
        # A short, phrase-aligned bass handover remains possible.
        drop_guard = 0.78 if bars_until_change <= 2 or bars <= 4 else 0.16
    else:
        drop_guard = 1.0
    if next_role in {"VOCAL", "VERSE", "CHORUS"} and current_role in {"VOCAL", "VERSE", "CHORUS"}:
        drop_guard *= 0.55

    role_path = _role_path_score(
        current_role, current_landing, next_role, next_landing, post_drop
    )
    energy_arc = _energy_arc(current, next_track, current_index, next_index, bars)

    intent = _choose_intent(
        current_role,
        current_landing,
        next_role,
        next_landing,
        post_drop,
        harmonic,
        bass_clean,
    )

    score = (
        0.25 * role_path
        + 0.22 * drop_landing
        + 0.16 * post_drop
        + 0.15 * energy_arc
        + 0.12 * boundary
        + 0.10 * drop_guard
    )
    # Intent-specific safeguards.
    if intent == "Bass Handover":
        score *= float(np.clip(0.60 + 0.22 * harmonic + 0.18 * bass_clean, 0.0, 1.0))
    if current_role == "DROP" and intent != "Bass Handover":
        score *= drop_guard

    return PhrasePolicyEvaluation(
        score=float(np.clip(score, 0.0, 1.0)),
        intent=intent,
        current_role=current_role,
        current_landing_role=current_landing,
        next_role=next_role,
        next_landing_role=next_landing,
        post_drop_score=float(np.clip(post_drop, 0.0, 1.0)),
        drop_landing_score=drop_landing,
        role_path_score=float(np.clip(role_path, 0.0, 1.0)),
        energy_arc_score=energy_arc,
        boundary_score=boundary,
        drop_guard_score=float(np.clip(drop_guard, 0.0, 1.0)),
        recommended_archetypes=_recommended(intent),
    )


def structural_candidate_prior(track: PreparedTrack, is_exit: bool) -> np.ndarray:
    """Return role-aware priors for candidate ranking before pairwise search."""
    count = track.bar_features.count
    output = np.full(count, 0.45, dtype=np.float32)
    for index in range(count):
        role = _role(track, index)
        boundary = _is_boundary(track, index)
        if is_exit:
            base = {
                "OUTRO": 1.00,
                "COOLDOWN": 0.96,
                "BREAKDOWN": 0.90,
                "BREAK": 0.88,
                "PHRASE": 0.72,
                "SECTION": 0.58,
                "BUILDUP": 0.34,
                "VOCAL": 0.30,
                "VERSE": 0.28,
                "CHORUS": 0.26,
                "DROP": 0.16,
            }.get(role, 0.48)
            distance = _previous_drop_distance(track, index)
            if distance is not None:
                base += 0.24 * float(np.exp(-0.35 * max(0, distance - 1)))
            if role == "DROP" and _bars_until_role_change(track, index) <= 2:
                base = max(base, 0.58)
        else:
            base = {
                "INTRO": 1.00,
                "BUILDUP": 0.96,
                "BREAKDOWN": 0.80,
                "BREAK": 0.78,
                "PHRASE": 0.75,
                "SECTION": 0.60,
                "DROP": 0.62,
                "CHORUS": 0.48,
                "VOCAL": 0.22,
                "VERSE": 0.20,
                "OUTRO": 0.08,
            }.get(role, 0.48)
        output[index] = float(np.clip(0.78 * base + 0.22 * boundary, 0.0, 1.0))
    return output
