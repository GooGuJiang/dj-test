"""Offline smoke test for the v1.2.18 cue-centered drum-loop bridge."""

from __future__ import annotations

import sys

import numpy as np

from autodj.human_transition import SUPPORTED_ARCHETYPES, _render_archetype


def _roles(sample_rate: int = 8_000, seconds: float = 3.0):
    length = int(sample_rate * seconds)
    t = np.arange(length, dtype=np.float32) / sample_rate
    low_a = np.column_stack([0.15 * np.sin(2 * np.pi * 55 * t)] * 2).astype(np.float32)
    low_b = np.column_stack([0.15 * np.sin(2 * np.pi * 65 * t)] * 2).astype(np.float32)
    harm_a = np.column_stack([0.07 * np.sin(2 * np.pi * 220 * t)] * 2).astype(np.float32)
    harm_b = np.column_stack([0.07 * np.sin(2 * np.pi * 262 * t)] * 2).astype(np.float32)
    pulse = np.zeros(length, dtype=np.float32)
    pulse[:: sample_rate // 2] = 0.24
    perc = np.column_stack([pulse, pulse]).astype(np.float32)
    return low_a, low_b, harm_a, harm_b, perc, perc.copy()


def main() -> int:
    sample_rate = 8_000
    bpm = 120.0
    beat = int(round(60.0 / bpm * sample_rate))
    roles = _roles(sample_rate)
    failures: list[str] = []

    for archetype in SUPPORTED_ARCHETYPES:
        mixed, deck_a, deck_b, controls = _render_archetype(
            archetype,
            a_low=roles[0],
            b_low=roles[1],
            a_harm=roles[2],
            b_harm=roles[3],
            a_perc=roles[4],
            b_perc=roles[5],
            sample_rate=sample_rate,
            bpm=bpm,
            beats_per_bar=4,
            bars=1,
            local_gain=1.0,
            effect_strength=0.65,
            vocal_risk=0.2,
            handoff_phase=4.0 / 6.0,
            transition_beats=6.0,
        )
        peak = float(np.max(np.abs(mixed)))
        bass_error = float(np.max(np.abs(controls["bass_a"] + controls["bass_b"] - 1.0)))
        early = beat
        approach = 3 * beat
        cue = 4 * beat
        after = cue + beat // 2
        released = cue + beat
        max_step = max(
            float(np.max(np.abs(np.diff(controls[name]))))
            for name in (
                "bass_a",
                "bass_b",
                "drum_a",
                "drum_b",
                "harm_a",
                "harm_b",
                "drum_loop_gain",
            )
        )
        print(
            f"{archetype:11s} peak={peak:.4f} bass_error={bass_error:.2e} "
            f"loop_B={controls['drum_b'][early]:.3f} "
            f"approach_B={controls['drum_b'][approach]:.3f} "
            f"cue_drums={controls['drum_a'][cue]:.3f}->{controls['drum_b'][cue]:.3f} "
            f"cue_harm={controls['harm_a'][cue]:.3f}->{controls['harm_b'][cue]:.3f} "
            f"post_bass={controls['bass_a'][after]:.3f}->{controls['bass_b'][after]:.3f} "
            f"A@+1beat={controls['drum_a'][released]:.3f} max_step={max_step:.4f}"
        )
        if not np.isfinite(mixed).all():
            failures.append(f"{archetype}: non-finite audio")
        if peak > 0.90:
            failures.append(f"{archetype}: peak {peak:.4f} > 0.90")
        if bass_error > 1e-6:
            failures.append(f"{archetype}: overlapping bass ownership")
        if not (0.18 <= controls["drum_b"][early] <= 0.25):
            failures.append(f"{archetype}: quantized drum loop is not audible/conservative")
        if controls["harm_b"][early] > 0.01 or controls["bass_b"][early] > 0.01:
            failures.append(f"{archetype}: early bridge exposes harmonic or bass content")
        if controls["drum_b"][approach] < 0.19:
            failures.append(f"{archetype}: drum bridge has a level hole before cue")
        if controls["drum_b"][cue] <= controls["drum_a"][cue]:
            failures.append(f"{archetype}: incoming drums do not own the cue")
        if controls["harm_b"][cue] <= controls["harm_a"][cue]:
            failures.append(f"{archetype}: incoming musical body does not own the cue")
        if controls["bass_b"][after] < 0.98 or controls["bass_a"][after] > 0.02:
            failures.append(f"{archetype}: bass handoff is too slow after cue")
        if controls["drum_a"][released] > 0.02:
            failures.append(f"{archetype}: outgoing drums remain too long")
        if np.mean(np.abs(deck_b[after])) <= np.mean(np.abs(deck_a[after])):
            failures.append(f"{archetype}: deck B is not dominant after cue")
        if max_step >= 0.01:
            failures.append(f"{archetype}: control curve has a hard step")

    if failures:
        print("FAILED")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("PASS: quantized drum-loop transition invariants satisfied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
