# Test report — Auto DJ 1.2.13 cue-centered handoff

## Goal

Use the selected CUE-DETR OUT/IN points as the actual ownership handoff, not as the start of a long overlap. Keep beat/downbeat phase locked, allow only a short pre-entry, make deck B dominant immediately after the cue, and release deck A smoothly instead of hard-cutting it.

## Implemented timing

- Default audible window: 2 beats total.
- Pre-entry: approximately 1 beat before the cue.
- Handoff: exact CUE-DETR cue sample.
- Release: approximately 1 beat after the cue.
- Requested 4/8/16/32 bars: matching context only; it no longer increases audible overlap.
- Near track boundaries: shorten the available window while preserving the neural cue.

## DSP invariants

- Incoming percussion before cue is limited to a low-level teaser.
- Incoming percussion is stronger than outgoing percussion at the cue.
- Low-frequency ownership is complementary: `bass_A + bass_B = 1`.
- Bass B exceeds 90% shortly after the cue.
- Outgoing drums and main content reach near zero within less than one beat.
- All gain curves are continuous; no hard-step transition is used.
- Echo Out sends only the outgoing harmonic layer and remains inside the short release window.

## Full test suite

Command:

```bash
pytest -q
```

Result:

```text
74 passed in 11.87s
```

## Independent transition smoke test

Command:

```bash
python check_natural_transition.py
```

Result:

```text
Short Blend peak=0.2871 bass_error=0.00e+00 pre_B=0.183 cue_drums=0.324->0.946 post_bass=0.049->0.951 A@70%=0.000 max_step=0.0007
Bass Swap   peak=0.2871 bass_error=0.00e+00 pre_B=0.183 cue_drums=0.324->0.946 post_bass=0.000->1.000 A@70%=0.000 max_step=0.0012
Echo Out    peak=0.2871 bass_error=0.00e+00 pre_B=0.183 cue_drums=0.324->0.946 post_bass=0.005->0.995 A@70%=0.000 max_step=0.0010
PASS: cue-centered transition invariants satisfied
```

## New regression coverage

- CUE-DETR cue equals `switch_sample_a` and `switch_sample_b`.
- Normal cues receive one beat of pre-roll and one beat of release.
- Cue-time incoming drums already own the rhythm.
- Deck B becomes full-band dominant immediately after the cue.
- Bass ownership completes quickly without double-bass gain.
- Deck A is released rapidly but with finite continuous ramps.
- Boundary-limited cues keep the neural location and shorten only unavailable audio.
- Simple Crossfade and Gapless fallbacks preserve the same cue-centered window.
- Old `Long Blend` settings migrate to `Short Blend`.

## Research basis

- CUE-DETR describes cue points as temporal boundaries used in DJ transitions and reports close adherence to human cue placement and phrasing.
- Serato documents accurate Beatgrids as the basis for precise Sync and beat-synchronized effects.
- The implementation therefore treats CUE-DETR as the structural switch authority and Beat This!/local phase lock as the timing authority.
