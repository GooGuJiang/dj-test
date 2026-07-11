# Test report — Auto DJ 1.2.15 balanced CUE handoff

## Scope

Version 1.2.15 addresses the report that beat-driven transitions felt too aggressive and had almost no audible crossfade. The selected CUE-DETR cue remains the ownership handoff point, but the default audible window is extended from 2 beats to 4 beats.

## Implementation verified

- Default matcher window: 2 beats before cue + 2 beats after cue.
- CUE-DETR cue is unchanged and remains centered when both tracks have enough audio.
- Incoming drums start as a 20% teaser, then cross with outgoing drums over roughly 2 beats.
- Harmonic/main content uses an approximately 2–2.5 beat equal-power crossfade.
- At the cue, the incoming deck is dominant while the outgoing deck is still audible.
- Outgoing drums remain for roughly 0.75 beat and outgoing harmonic content for roughly 1.1 beats after cue.
- Bass ownership remains complementary (`bass_A + bass_B = 1`) and changes over approximately 0.65–0.9 beat.
- Simple Crossfade and Gapless fallback paths use the same balanced timing.
- Existing MIX END tempo hold, dense tempo recovery, unity gain, no-zero-padding and 24 ms render seam behavior remain active.

## Automated test suite

Executed from a clean extraction of the delivery ZIP:

```text
83 passed in 12.59s
```

New 1.2.15 coverage includes:

1. Default matcher uses two beats on each side of the CUE-DETR point.
2. Advanced renderer has a measurable pre-cue musical overlap.
3. Incoming deck dominates at cue without instantly muting the outgoing deck.
4. Outgoing harmonic content remains audible after cue.
5. Equal-power musical curves do not create an envelope hole.
6. Fallback curves follow the same balanced timing and reach unity at MIX END.

## Natural transition smoke test

```text
Short Blend peak=0.2890 bass_error=0.00e+00 pre_B=0.200 cue_drums=0.419->0.908 overlap=0.869/0.468 post_bass=0.000->1.000 A@78%=0.000 max_step=0.0005
Bass Swap   peak=0.2890 bass_error=0.00e+00 pre_B=0.200 cue_drums=0.419->0.908 overlap=0.869/0.468 post_bass=0.000->1.000 A@78%=0.000 max_step=0.0007
Echo Out    peak=0.2893 bass_error=0.00e+00 pre_B=0.200 cue_drums=0.419->0.908 overlap=0.869/0.468 post_bass=0.000->1.000 A@78%=0.000 max_step=0.0006
PASS: cue-centered transition invariants satisfied
```

Interpretation:

- Incoming drum teaser before the main cross: `0.200`.
- At cue, outgoing/incoming drum gains: `0.419 -> 0.908`.
- Before cue, outgoing/incoming harmonic gains in the audible overlap: `0.869 / 0.468`.
- Bass is fully owned by the incoming deck shortly after cue.
- Maximum curve step is below `0.001`, far below the hard-step guard of `0.01`.

## MIX END continuity smoke test

```text
fallback endpoints A=-0.000000/0.000000 B=1.000000/1.000000/1.000000
render seam=24.00 ms jump=0.400006->0.011974 natural=0.011974
PASS: MIX END continuity invariants satisfied
```

The longer handoff did not regress the 1.2.14 MIX END fixes.

## Result

All automated and standalone checks passed. Code-level verification confirms a longer audible transition, preserved beat/cue ownership, complementary bass control, continuous envelopes and unchanged MIX END continuity.
