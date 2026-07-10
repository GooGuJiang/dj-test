from __future__ import annotations

import numpy as np

from .models import BarFeatures, CueDETRProfile, EDMStructure


def apply_cuedetr_cues(
    base: EDMStructure,
    profile: CueDETRProfile,
    features: BarFeatures,
    source_downbeat_samples: np.ndarray,
    sample_rate: int,
) -> EDMStructure:
    """Replace every legacy cue prior with CUE-DETR predictions.

    All-In-One/local roles may influence whether a neural cue is better as IN or
    OUT, but they cannot create a cue at a bar that CUE-DETR did not predict.
    """
    count = features.count
    cue = np.zeros(count, dtype=np.float32)
    mix_in = np.zeros(count, dtype=np.float32)
    mix_out = np.zeros(count, dtype=np.float32)
    phrase = np.zeros(count, dtype=np.float32)
    if count == 0:
        base.cue_score = cue
        base.mix_in_score = mix_in
        base.mix_out_score = mix_out
        base.phrase_mask = phrase
        base.structure_source = "CUE-DETR unavailable"
        return base

    source_downbeats = np.asarray(source_downbeat_samples, dtype=np.int64)
    if source_downbeats.size >= count:
        bar_times = source_downbeats[:count] / float(sample_rate)
    else:
        bar_times = np.linspace(0.0, max(1.0, float(count)), count, endpoint=False)

    for time_value, score_value in zip(profile.cue_times, profile.cue_scores):
        index = int(np.argmin(np.abs(bar_times - float(time_value))))
        score = float(np.clip(score_value, 0.02, 1.0))
        cue[index] = max(cue[index], score)
        phrase[index] = max(phrase[index], score)
        position = index / max(count - 1, 1)
        functional = (
            base.functional_labels[index].upper()
            if index < len(base.functional_labels)
            else ""
        )
        role = base.labels[index].upper() if index < len(base.labels) else "SECTION"

        # Directional ranking only, not candidate generation.
        entry = score * (1.04 - 0.48 * position)
        exit_score = score * (0.56 + 0.56 * position)
        if functional in {"INTRO", "START", "INST", "BREAK", "BRIDGE"}:
            entry *= 1.18
        if role in {"BUILDUP", "BREAKDOWN", "INTRO"}:
            entry *= 1.10
        if functional in {"OUTRO", "END", "BREAK", "BRIDGE"}:
            exit_score *= 1.20
        if role in {"COOLDOWN", "BREAKDOWN", "OUTRO"}:
            exit_score *= 1.12
        if role == "DROP":
            entry *= 1.10
            exit_score *= 0.84
        mix_in[index] = float(np.clip(entry, 0.0, 1.0))
        mix_out[index] = float(np.clip(exit_score, 0.0, 1.0))

    base.cue_score = cue
    base.mix_in_score = mix_in
    base.mix_out_score = mix_out
    base.phrase_mask = phrase
    base.structure_source = (
        f"{profile.model_name} + Beat This! downbeat quantization + All-In-One roles"
        if profile.available
        else "CUE-DETR endpoint safety only"
    )
    return base
