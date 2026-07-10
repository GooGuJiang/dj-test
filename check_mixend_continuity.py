"""Offline smoke test for v1.2.14 MIX END continuity invariants."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np

from autodj.audio_engine import AutoDJEngine, EngineConfig


def main() -> int:
    sr = 8_000
    engine = AutoDJEngine(EngineConfig(sample_rate=sr))
    failures: list[str] = []

    curves = engine._cue_centered_curves(8_000, gain=0.67, switch_position=0.5)
    fade_out, fade_in, bass_out, bass_in, _, high_in = curves
    print(
        "fallback endpoints "
        f"A={fade_out[-1]:.6f}/{bass_out[-1]:.6f} "
        f"B={fade_in[-1]:.6f}/{bass_in[-1]:.6f}/{high_in[-1]:.6f}"
    )
    if max(abs(float(fade_in[-1]) - 1.0), abs(float(bass_in[-1]) - 1.0)) > 1e-6:
        failures.append("incoming fallback gain does not reach unity")

    length = sr * 3
    t = np.arange(length, dtype=np.float32) / sr
    incoming = np.column_stack([0.25 * np.sin(2 * np.pi * 61.0 * t)] * 2).astype(np.float32)
    resume = sr * 2
    plan = SimpleNamespace(
        rendered_audio=np.full((sr, 2), 0.4, dtype=np.float32),
        next_resume_sample=resume,
        next_start=resume - sr,
        metrics={},
    )
    track = SimpleNamespace(audio=incoming, total_samples=len(incoming))
    plan = engine._stitch_rendered_transition_to_next(plan, track)
    natural_step = float(np.max(np.abs(incoming[resume] - incoming[resume - 1])))
    print(
        f"render seam={plan.metrics['mixend_seam_ms']:.2f} ms "
        f"jump={plan.metrics['mixend_jump_before']:.6f}->"
        f"{plan.metrics['mixend_jump_after']:.6f} natural={natural_step:.6f}"
    )
    if plan.metrics["mixend_jump_after"] >= plan.metrics["mixend_jump_before"]:
        failures.append("render seam did not reduce the promotion discontinuity")
    if not np.allclose(plan.rendered_audio[-1], incoming[resume - 1], atol=1e-6):
        failures.append("rendered tail does not end on contiguous incoming audio")

    if failures:
        print("FAILED")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("PASS: MIX END continuity invariants satisfied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
