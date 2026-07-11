"""Standalone test for the beat-quantized incoming percussion loop."""

from __future__ import annotations

import sys

import numpy as np

from autodj.human_transition import _cue_quantized_drum_loop


def main() -> int:
    sample_rate = 8_000
    bpm = 120.0
    beat = int(round(60.0 / bpm * sample_rate))
    handoff = 4 * beat
    length = 6 * beat
    audio = np.zeros((length, 2), dtype=np.float32)

    pattern = np.linspace(-0.4, 0.4, 2 * beat, dtype=np.float32)
    pattern[:: beat // 4] += 0.18
    audio[handoff : handoff + 2 * beat] = pattern[:, None]

    looped = _cue_quantized_drum_loop(
        audio,
        sample_rate=sample_rate,
        bpm=bpm,
        handoff_index=handoff,
        loop_beats=2.0,
    )
    repeat_error = float(np.max(np.abs(looped[: 2 * beat] - looped[2 * beat : handoff])))
    wrap_jumps = [
        float(np.max(np.abs(looped[index] - looped[index - 1])))
        for index in (2 * beat, handoff)
    ]
    live_error = float(np.max(np.abs(looped[handoff + 64 :] - audio[handoff + 64 :])))
    print(
        f"repeat_error={repeat_error:.3e} "
        f"wrap_jump={max(wrap_jumps):.4f} live_error={live_error:.3e}"
    )
    if repeat_error > 1e-7:
        print("FAILED: pre-cue loop repetitions are not identical")
        return 1
    if max(wrap_jumps) >= 0.06:
        print("FAILED: drum-loop seam is too abrupt")
        return 1
    if live_error > 1e-7:
        print("FAILED: live incoming percussion was altered after the de-click seam")
        return 1
    print("PASS: beat-quantized drum-loop bridge is continuous")
    return 0


if __name__ == "__main__":
    sys.exit(main())
