"""Offline smoke test for the conservative v1.2.12 transition renderer."""

from __future__ import annotations

import sys

import numpy as np

from autodj.human_transition import SUPPORTED_ARCHETYPES, _render_archetype


def _roles(sample_rate: int = 8_000, seconds: float = 8.0):
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
        mixed, _, _, controls = _render_archetype(
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
            bars=8,
            local_gain=1.0,
            effect_strength=0.65,
            vocal_risk=0.2,
        )
        peak = float(np.max(np.abs(mixed)))
        bass_error = float(np.max(np.abs(controls["bass_a"] + controls["bass_b"] - 1.0)))
        early = int(0.20 * len(mixed))
        late = int(0.95 * len(mixed))
        print(
            f"{archetype:10s} peak={peak:.4f} bass_sum_error={bass_error:.2e} "
            f"drum_B@20%={controls['drum_b'][early]:.3f} "
            f"drum_A@95%={controls['drum_a'][late]:.3f}"
        )
        if not np.isfinite(mixed).all():
            failures.append(f"{archetype}: non-finite audio")
        if peak > 0.90:
            failures.append(f"{archetype}: peak {peak:.4f} > 0.90")
        if bass_error > 1e-6:
            failures.append(f"{archetype}: overlapping bass ownership")
        if controls["drum_b"][early] < 0.50:
            failures.append(f"{archetype}: incoming groove enters too late")
        if controls["drum_a"][late] > 0.10:
            failures.append(f"{archetype}: outgoing drums release too late")

    if failures:
        print("FAILED")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("PASS: natural transition renderer invariants satisfied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
