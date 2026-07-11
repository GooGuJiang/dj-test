# Engineering notes — 1.2.15

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

CUE-DETR cue points are treated as temporal ownership boundaries. The audible transition is not a multi-bar phrase beginning at the cue, but the earlier two-beat window proved too abrupt in listening. Version 1.2.15 uses a balanced four-beat window: two beats before the cue and two beats after it. The exact cue is still stored separately from the window start and remains the ownership switch.

Incoming percussion begins as a quiet teaser, then drums cross for about two beats. Harmonic/main content uses a wider 2–2.5 beat equal-power crossfade. At the cue, B is already dominant, but A remains audible for roughly 0.75 beat in drums and about 1.1 beats in harmonic content. This preserves a clear handoff while making the transition perceptible rather than instantaneous.

The low-band curves remain complementary, so `bass_A + bass_B = 1`, but their exchange is widened to roughly 0.65–0.9 beat. Echo stays restricted to the outgoing harmonic layer. All curves are continuous and the fallback renderer mirrors the same timing.

If either cue is too close to a boundary, the pre/post window is shortened while preserving the neural cue. The matcher never invents a replacement location.

## Determinism

Transition history, variation bonuses and impact-mode preferences were removed. The same analyzed pair and settings now produce the same cue pair, curves and rendered audio.

## Verification

- 83 pytest cases pass.
- The standalone natural-transition smoke test checks finite audio, peak ceiling, bass ownership, quiet pre-cue drums, audible overlap, cue-time ownership, post-cue tail and continuous curves.
- Existing preload, seek, phase-lock, tempo recovery, time-stretch and CUE-DETR-only tests remain active.


## MIX END tempo and buffer continuity

The incoming deck remains at the synchronized BPM for at least one complete bar after MIX END. Tempo restoration then uses 16 source-to-target map subdivisions per bar with a smootherstep/geometric rate trajectory. This keeps the derivative near zero at both endpoints and avoids a bar-level staircase.

A second failure mode was independent of tempo: fallback transitions consumed B linearly but resumed from the advanced phrase-warp endpoint, and advanced rendered buffers were promoted without a final waveform seam. The fallback now resumes at `next_start + length`; local gain reaches unity at the boundary; advanced renders crossfade their final ~24 ms to the exact B samples immediately preceding `next_resume_sample`. Time-stretch length shortfalls no longer receive zero padding.

These changes follow the general behavior exposed by DJ/audio tools: tempo changes should retain key when key lock is enabled, and adjacent audio clips use short crossfades to avoid edge discontinuities.
