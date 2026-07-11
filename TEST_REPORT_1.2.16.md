# Test report — Auto DJ 1.2.16 quantized drum-loop bridge

## Goal

Extend the pre-cue rhythmic setup without returning to a long full-song overlap. The selected CUE-DETR point remains the ownership switch. The incoming percussion is allowed to repeat for several beats, while incoming bass, harmonic and vocal content remain gated until close to the cue.

## Implemented timing

- Pre-cue window: 4 beats.
- Post-cue release: 2 beats.
- Incoming percussion loop: 2 beats, repeated twice before the cue.
- Loop level: approximately 24% after its short fade-in.
- Live drum takeover: begins approximately 1.5 beats before the cue.
- Bass ownership: complementary (`bass_A + bass_B = 1`).
- Outgoing drums: released within approximately 1 beat after the cue.
- MIX END continuity: unchanged from 1.2.14/1.2.15.

## Code-level coverage

New 1.2.16 tests verify:

1. the matcher reserves four beats before the cue and two beats after it;
2. a two-beat cue-start percussion phrase repeats exactly twice in the pre-roll;
3. loop wraps remain below the discontinuity threshold;
4. the live incoming percussion is unchanged after the short de-click bridge;
5. the early bridge contains percussion but not incoming harmonic or bass material;
6. deck B owns drums, harmonic content and bass at the CUE-DETR point;
7. the loop-gain release and live-drum crossfade do not create a level hole;
8. low-frequency ownership remains complementary and control curves remain continuous.

## Full regression

```text
87 passed in 12.39s
```

## Standalone drum-loop check

```text
repeat_error=0.000e+00 wrap_jump=0.0232 live_error=0.000e+00
PASS: beat-quantized drum-loop bridge is continuous
```

Interpretation:

- the two pre-cue repetitions are sample-identical;
- the deliberately difficult loop boundary remains below the click threshold;
- after the tiny join seam, the renderer returns to the unmodified live B stem.

## Natural-transition DSP check

```text
Short Blend peak=0.2976 bass_error=0.00e+00 loop_B=0.240 approach_B=0.227 cue_drums=0.388->0.922 cue_harm=0.559->0.805 post_bass=0.000->1.000 A@+1beat=0.000 max_step=0.0005
Bass Swap   peak=0.2976 bass_error=0.00e+00 loop_B=0.240 approach_B=0.227 cue_drums=0.388->0.922 cue_harm=0.559->0.805 post_bass=0.000->1.000 A@+1beat=0.000 max_step=0.0007
Echo Out    peak=0.2976 bass_error=0.00e+00 loop_B=0.240 approach_B=0.227 cue_drums=0.388->0.922 cue_harm=0.559->0.805 post_bass=0.000->1.000 A@+1beat=0.000 max_step=0.0006
PASS: quantized drum-loop transition invariants satisfied
```

Interpretation:

- incoming loop is audible but conservative;
- there is no pre-cue level hole as the synthetic loop hands over to live B drums;
- B clearly owns drums and harmonic content on the cue;
- bass has completely changed owner shortly after the cue;
- outgoing drums are gone one beat after the cue;
- all control changes remain far below the hard-step threshold.

## MIX END continuity check

```text
fallback endpoints A=-0.000000/0.000000 B=1.000000/1.000000/1.000000
render seam=24.00 ms jump=0.400006->0.011974 natural=0.011974
PASS: MIX END continuity invariants satisfied
```

The 1.2.16 drum-loop bridge does not regress the previous tempo-hold, gain-unity, sample-position or rendered-buffer seam fixes.

## Remaining listening dependency

Automated tests validate timing, phase ownership, envelopes, loop repetition and sample continuity. Final musical quality still depends on accurate BeatGrid/downbeat and CUE-DETR analysis for the user's real tracks. A wrong grid can make any quantized loop repeat the wrong rhythmic phase and should be corrected at analysis level rather than masked with effects.
