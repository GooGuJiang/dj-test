# Engineering notes — 1.2.12

## Research conclusion

Official DJ software documentation consistently treats an accurate beat grid as the basis for sync, loops and tempo-synchronised effects. Phrase analysis and editable phrase/grid positions are also exposed because automatic analysis can be wrong. Smooth transitions therefore must first satisfy beat/downbeat and phrase alignment; effects cannot repair a structurally wrong cue pair.

The renderer follows four conservative mixer actions:

1. establish the incoming percussion while the outgoing groove remains stable;
2. keep the incoming low band muted;
3. hand low-frequency ownership to the incoming deck at a phrase-quantized point;
4. release outgoing harmonic/percussive content smoothly, using a filtered echo only when overlapping vocals or harmony are unsafe.

## Cue search

CUE-DETR remains the only cue generator. Unlike 1.2.11, no song-position window is applied. Every neural cue after the outgoing time floor is ranked by:

- CUE-DETR IN/OUT confidence;
- phrase and All-In-One boundary confidence;
- beat/onset compatibility;
- low-frequency and vocal collision estimates;
- RMS continuity and energy trajectory;
- harmonic, Camelot and MuQ compatibility;
- structure role compatibility and drop guard.

`out_position` and `in_position` remain available for debugging. `position_bias_applied` is always `0.0`.

## Natural renderer

The low-band curves are complementary linear/smootherstep ownership curves rather than equal-power curves. Equal-power is useful for uncorrelated full-band crossfades, but two coherent kick/bass layers can create a hump or phase cancellation. Complementary ownership keeps the summed control gain at unity and shortens the period where both bass layers are active.

Percussion enters before bass. Outgoing drums are not released until the incoming groove has reached a useful level. Harmonic layers have a longer, vocal-risk-dependent transition window. Echo uses only the outgoing harmonic layer, so sub and kick energy do not feed the delay.

## Determinism

Transition history, variation bonuses and impact-mode preferences were removed. The same analyzed pair and settings now produce the same cue pair, curves and rendered audio.

## Verification

- 71 pytest cases pass.
- The standalone natural-transition smoke test checks finite audio, peak ceiling, bass ownership, early incoming drums and final outgoing release.
- Existing preload, seek, phase-lock, tempo recovery, time-stretch and CUE-DETR-only tests remain active.
