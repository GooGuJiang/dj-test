from __future__ import annotations

import numpy as np

from .models import AllInOneProfile, BarFeatures, EDMStructure


def _unit(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values.astype(np.float32)
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    if maximum <= minimum + 1e-9:
        return np.zeros_like(values, dtype=np.float32)
    return np.asarray((values - minimum) / (maximum - minimum), dtype=np.float32)


def _role_for_label(
    label: str,
    index: int,
    features: BarFeatures,
    base_role: str,
) -> str:
    label = label.lower().strip()
    energy = _unit(features.rms)
    onset = _unit(features.onset)
    low = _unit(features.low_ratio)
    vocal = _unit(features.vocal_proxy)
    local_energy = float(energy[index]) if index < len(energy) else 0.5
    local_onset = float(onset[index]) if index < len(onset) else 0.5
    local_low = float(low[index]) if index < len(low) else 0.5
    local_vocal = float(vocal[index]) if index < len(vocal) else 0.5

    if label in {"start", "intro"}:
        return "INTRO"
    if label in {"end", "outro"}:
        return "OUTRO"
    if label == "break":
        return "BREAKDOWN"
    if label == "bridge":
        before = float(np.mean(energy[max(0, index - 3) : index])) if index else 0.0
        after = float(np.mean(energy[index : min(len(energy), index + 4)]))
        return "BUILDUP" if after > before + 0.10 else "BREAKDOWN"
    if label == "chorus":
        if local_energy > 0.58 and local_onset > 0.46 and local_low > 0.30:
            return "DROP"
        return "CHORUS"
    if label == "verse":
        return "VOCAL" if local_vocal > 0.48 else "VERSE"
    if label == "solo":
        return "SOLO"
    if label == "inst":
        return "PHRASE" if local_onset > 0.42 else "SECTION"
    return base_role


def fuse_allinone_structure(
    base: EDMStructure,
    profile: AllInOneProfile,
    features: BarFeatures,
    source_downbeat_samples: np.ndarray,
    sample_rate: int,
) -> EDMStructure:
    """Fuse All-In-One functional segments with local EDM role estimates.

    Beat This! remains the timing authority. All-In-One labels are sampled on the
    corresponding original-audio downbeats, then converted into DJ-oriented roles.
    Its boundaries boost cue, mix-in and mix-out probabilities instead of blindly
    replacing local energy/phrase evidence.
    """
    count = features.count
    if count == 0 or not profile.available:
        return base

    source_downbeats = np.asarray(source_downbeat_samples, dtype=np.int64)
    if source_downbeats.size >= count:
        times = source_downbeats[:count] / float(sample_rate)
    else:
        duration = max(profile.duration, 1e-6)
        times = np.linspace(0.0, duration, count, endpoint=False, dtype=np.float64)

    functional: list[str] = []
    roles: list[str] = []
    boundary_score = np.zeros(count, dtype=np.float32)
    boundary_labels: list[str] = []

    bar_seconds = float(np.median(np.diff(times))) if len(times) >= 2 else 2.0
    boundary_radius = max(0.20, 0.55 * bar_seconds)
    starts = np.asarray([segment.start for segment in profile.segments], dtype=np.float64)

    for index, time_value in enumerate(times):
        label = profile.label_at(float(time_value))
        functional.append(label.upper() if label else "UNKNOWN")
        base_role = base.labels[index] if index < len(base.labels) else "SECTION"
        roles.append(_role_for_label(label, index, features, base_role))
        if starts.size:
            nearest = int(np.argmin(np.abs(starts - time_value)))
            distance = abs(float(starts[nearest] - time_value))
            if distance <= boundary_radius:
                # A neural segment boundary is not guaranteed to land exactly on
                # Beat This!'s downbeat. Snapping it to the nearest bar should
                # retain high confidence instead of being punished by sub-beat
                # offsets between the two models.
                boundary_score[index] = float(
                    0.75 + 0.25 * np.clip(1.0 - distance / boundary_radius, 0.0, 1.0)
                )
            else:
                boundary_score[index] = float(
                    0.35 * np.exp(-0.5 * (distance / max(boundary_radius, 1e-6)) ** 2)
                )
            boundary_labels.append(profile.segments[nearest].label.upper())
        else:
            boundary_labels.append("")

    cue = np.asarray(base.cue_score, dtype=np.float32).copy()
    mix_in = np.asarray(base.mix_in_score, dtype=np.float32).copy()
    mix_out = np.asarray(base.mix_out_score, dtype=np.float32).copy()
    phrase = np.asarray(base.phrase_mask, dtype=np.float32).copy()

    for index, (functional_label, role) in enumerate(zip(functional, roles)):
        boundary = float(boundary_score[index])
        cue[index] = max(cue[index], 0.70 * boundary)
        phrase[index] = max(phrase[index], 0.86 * boundary)
        if functional_label in {"INTRO", "START", "INST"}:
            mix_in[index] = max(mix_in[index], 0.72 * boundary + 0.18)
        elif functional_label in {"CHORUS", "BRIDGE", "SOLO"}:
            mix_in[index] = max(mix_in[index], 0.62 * boundary + 0.12)
        if functional_label in {"OUTRO", "END", "BREAK", "BRIDGE"}:
            mix_out[index] = max(mix_out[index], 0.74 * boundary + 0.16)
        elif functional_label in {"VERSE", "CHORUS", "SOLO"}:
            mix_out[index] = max(mix_out[index], 0.54 * boundary + 0.08)
        if role == "DROP":
            mix_in[index] = max(mix_in[index], 0.78 * boundary + 0.12)

    base.cue_score = np.clip(cue, 0.0, 1.0).astype(np.float32)
    base.mix_in_score = np.clip(mix_in, 0.0, 1.0).astype(np.float32)
    base.mix_out_score = np.clip(mix_out, 0.0, 1.0).astype(np.float32)
    base.phrase_mask = np.clip(phrase, 0.0, 1.0).astype(np.float32)
    base.labels = tuple(roles)
    base.functional_labels = tuple(functional)
    base.allin1_boundary_score = np.ascontiguousarray(boundary_score, dtype=np.float32)
    base.structure_source = f"All-In-One/{profile.model_name} + local EDM"
    base.allin1_backend = profile.backend
    # Functional segmentation evidence increases confidence but cannot force an
    # EDM classification on non-EDM songs.
    base.edm_confidence = float(
        np.clip(0.88 * base.edm_confidence + 0.12 * np.mean(boundary_score > 0.25), 0.0, 1.0)
    )
    return base
