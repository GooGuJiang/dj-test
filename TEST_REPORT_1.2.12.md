# Test report — Auto DJ 1.2.12 natural transition

## Scope

This release changes cue-pair selection and the final transition renderer. The regression suite covers matcher behavior, CUE-DETR-only cue provenance, phrase policy, preload/seek behavior, BPM recovery, beat-grid phase lock, time stretching, natural transition controls and removed impact modes.

## Baseline

Before the 1.2.12 changes:

```text
61 passed in 9.94s
```

## Final full suite

Command:

```bash
pytest -q
```

Result:

```text
71 passed in 12.36s
```

## Standalone renderer invariant check

Command:

```bash
python check_natural_transition.py
```

Result:

```text
Long Blend peak=0.4417 bass_sum_error=0.00e+00 drum_B@20%=0.866 drum_A@95%=0.000
Bass Swap  peak=0.4417 bass_sum_error=0.00e+00 drum_B@20%=0.866 drum_A@95%=0.000
Echo Out   peak=0.4417 bass_sum_error=0.00e+00 drum_B@20%=0.866 drum_A@95%=0.000
PASS: natural transition renderer invariants satisfied
```

## Import/version smoke test

```text
import_ok 1.2.12 Natural Auto ('Long Blend', 'Bass Swap', 'Echo Out')
```

## Assertions added for 1.2.12

- No 42%/48-bar outgoing or 35%/32-bar incoming position gate remains.
- Song position is diagnostic only (`position_bias_applied == 0`).
- Removed impact modes cannot be rendered or selected in the GUI.
- Incoming drums establish the groove before the low-frequency handover.
- `bass_A + bass_B == 1` across the transition.
- Curves are finite, continuous and have correct start/end ownership.
- Echo is omitted for clean compatible pairs and considered only for vocal/harmonic risk.
- Selection is deterministic for identical audio, analysis and settings.

## Limitation

These tests verify code behavior and DSP invariants. Perceived naturalness on a real library still depends on accurate BPM, beat/downbeat grids, phrase labels and cue predictions. Tracks with incorrect analysis should be corrected or rejected rather than masked with stronger effects.
