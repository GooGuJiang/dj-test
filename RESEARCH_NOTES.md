# Engineering notes — 1.2.13

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

## Cue-centered renderer

CUE-DETR cue points are treated as temporal transition boundaries. The audible transition is no longer a multi-bar phrase beginning at the cue. A two-beat window is built around the selected pair: one beat of pre-roll and one beat of release. The exact cue is stored separately from the window start and is used as the ownership switch.

Before the cue, incoming percussion is limited to a quiet teaser. At the cue, incoming drums are already stronger than outgoing drums and the complementary low-band curves cross. Outgoing harmonic and percussive content then reaches zero within less than one beat. This creates a decisive switch without a discontinuity.

The low-band curves remain complementary rather than equal-power, so `bass_A + bass_B = 1`. Full-band/harmonic curves remain continuous equal-power curves. Echo is restricted to the outgoing harmonic layer and decays inside the short release window.

If either cue is too close to a boundary, the pre/post window is shortened while preserving the neural cue. The matcher never invents a replacement location.

## Determinism

Transition history, variation bonuses and impact-mode preferences were removed. The same analyzed pair and settings now produce the same cue pair, curves and rendered audio.

## Verification

- 74 pytest cases pass.
- The standalone natural-transition smoke test checks finite audio, peak ceiling, bass ownership, quiet pre-cue drums, cue-time ownership, sub-beat release and continuous curves.
- Existing preload, seek, phase-lock, tempo recovery, time-stretch and CUE-DETR-only tests remain active.
