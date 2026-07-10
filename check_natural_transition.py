"""Offline smoke test for the cue-centered v1.2.13 transition renderer."""

from __future__ import annotations

import sys

import numpy as np

from autodj.human_transition import SUPPORTED_ARCHETYPES, _render_archetype


def _roles(sample_rate: int = 8_000, seconds: float = 2.0):
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
            bpm=120.0,
            beats_per_bar=4,
            bars=1,
            local_gain=1.0,
            effect_strength=0.65,
            vocal_risk=0.2,
            handoff_phase=0.5,
            transition_beats=2.0,
        )
        peak = float(np.max(np.abs(mixed)))
        bass_error = float(np.max(np.abs(controls["bass_a"] + controls["bass_b"] - 1.0)))
        pre = int(0.20 * (len(mixed) - 1))
        cue = int(0.50 * (len(mixed) - 1))
        after = int(0.55 * (len(mixed) - 1))
        released = int(0.70 * (len(mixed) - 1))
        max_step = max(
            float(np.max(np.abs(np.diff(controls[name]))))
            for name in ("bass_a", "bass_b", "drum_a", "drum_b", "harm_a", "harm_b")
        )
        print(
            f"{archetype:11s} peak={peak:.4f} bass_error={bass_error:.2e} "
            f"pre_B={controls['drum_b'][pre]:.3f} "
            f"cue_drums={controls['drum_a'][cue]:.3f}->{controls['drum_b'][cue]:.3f} "
            f"post_bass={controls['bass_a'][after]:.3f}->{controls['bass_b'][after]:.3f} "
            f"A@70%={controls['drum_a'][released]:.3f} max_step={max_step:.4f}"
        )
        if not np.isfinite(mixed).all():
            failures.append(f"{archetype}: non-finite audio")
        if peak > 0.90:
            failures.append(f"{archetype}: peak {peak:.4f} > 0.90")
        if bass_error > 1e-6:
            failures.append(f"{archetype}: overlapping bass ownership")
        if not (0.10 <= controls["drum_b"][pre] <= 0.25):
            failures.append(f"{archetype}: pre-cue drum teaser is not conservative")
        if controls["drum_b"][cue] <= controls["drum_a"][cue]:
            failures.append(f"{archetype}: incoming drums do not own the cue")
        if controls["bass_b"][after] < 0.90 or controls["bass_a"][after] > 0.10:
            failures.append(f"{archetype}: bass handoff is too slow after cue")
        if controls["drum_a"][released] > 0.02:
            failures.append(f"{archetype}: outgoing drums release too late")
        if np.mean(np.abs(deck_b[after])) <= np.mean(np.abs(deck_a[after])):
            failures.append(f"{archetype}: deck B is not dominant after cue")
        if max_step >= 0.01:
            failures.append(f"{archetype}: control curve has a hard step")

    if failures:
        print("FAILED")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("PASS: cue-centered transition invariants satisfied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
