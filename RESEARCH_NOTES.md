# Engineering notes — 1.2.18

## Network research translated into implementation

The 1.2.18 pass used public papers, official repositories and expert-annotation guidelines rather than changing weights by trial and error. CUE-DETR remains the only source of cue locations. M-DJCUE's annotation rules motivate a separate endpoint entry-quality term: an IN cue should have enough meaningful material after it to carry the mix and should not be excessively quiet. The differentiable-DSP DJ transition literature motivates keeping the transition controls explicit as EQ-band gains plus fader curves.

Playlist ordering is treated as a directed Hamiltonian-path problem whose edge values already encode style, tempo, structure and transition quality. Small queues now use exact subset dynamic programming. Larger queues keep beam search but add two-step continuation estimates and a small endpoint-diversity reserve, following the general result that near-duplicate beams waste search budget.

## Correlation-aware fallback fade

Equal-power curves assume the two sources are effectively uncorrelated. Aligned kicks, repeated loops or similar source material can have positive correlation, adding a cross term to expected overlap power:

```text
P = a² + b² + 2ρab
```

For each low/mid/high band, 1.2.18 estimates zero-lag normalized correlation over the planned overlap. Positive correlation scales both curves by `sqrt((a²+b²)/P)`. This leaves both endpoints unchanged and reduces the centre by at most approximately 3 dB. Negative correlation is measured for diagnostics but never boosted because cancellation estimates are not stable enough to justify added gain.

## Energy context correction

The old path objective subtracted a penalty whenever consecutive energy deltas changed sign. That meant a rise followed by a release was treated as worse than an uninterrupted rise, contrary to the sequencing regularities reported in curated albums. The new second-order term mildly rewards controlled direction reversal, penalizes two large moves in the same direction, and retains an acceleration penalty to prevent exaggerated saw-tooth ordering.

## Verification

- 94 pytest cases pass.
- Exact DP remains below 0.13 seconds for a synthetic 10-track queue in the test environment.
- A 20-track beam-search queue completes in about 0.40 seconds in the same smoke benchmark.
- Natural transition, MIX END and drum-loop continuity checks pass.

---

# Engineering notes — 1.2.16

## Research conclusion

Official DJ software documentation treats an accurate BeatGrid as the basis for Sync, quantized cue triggering, loops and tempo-synchronised effects. A loop used as a transition bridge must therefore begin and wrap on musical beat positions. Crossfading should then control which deck owns percussion, harmonic content and bass rather than allowing both full mixes to run at equal level.

The renderer follows five conservative mixer actions:

1. keep the outgoing groove and bass stable;
2. introduce a beat-quantized loop of the incoming percussion only;
3. start the real incoming drum/harmonic crossfade near the CUE-DETR point;
4. hand low-frequency ownership to the incoming deck at the cue;
5. release outgoing content smoothly, using filtered Echo only when vocal/harmonic overlap is unsafe.

## Cue search

CUE-DETR remains the only cue generator. No song-position window is applied. Every neural cue after the outgoing time floor is ranked by:

- CUE-DETR IN/OUT confidence;
- phrase and All-In-One boundary confidence;
- beat/onset compatibility;
- low-frequency and vocal collision estimates;
- RMS continuity and energy trajectory;
- harmonic, Camelot and MuQ compatibility;
- structure role compatibility and drop guard.

`out_position` and `in_position` remain available for debugging. `position_bias_applied` is always `0.0`.

## Quantized drum-loop bridge

Version 1.2.16 uses a six-beat audible window: four beats before the selected cue and two beats after it. The exact CUE-DETR point remains the ownership boundary.

The early four-beat region is not a full mix. The renderer extracts two beats of the incoming percussive stem beginning exactly on the IN cue and repeats that phrase twice before the cue. The repeat is counted backwards from the cue, so every loop wrap lands on the same BeatGrid phase even when the available start window is shortened.

A short cyclic bridge smooths the loop's end-to-start waveform, and a second short bridge joins the final synthetic repetition to the live incoming stem at the cue. Both bridges are approximately 4 ms and do not change the loop duration.

The incoming loop fades up to about 24% while the outgoing deck remains near unity. The live drum crossfade starts about 1.5 beats before the cue. Harmonic/main content still uses an approximately 2–2.5 beat equal-power crossfade; bass remains complementary and changes within about 0.7–0.9 beat around the cue. Thus the longer lead-in creates rhythmic expectation without extending double-vocal, double-harmonic or double-bass overlap.

If the cue is too close to a track boundary, the available window is shortened while preserving the neural cue. The matcher never invents a replacement location.

## Bass ownership and determinism

The low-band curves remain complementary, so `bass_A + bass_B = 1`. Echo stays restricted to the outgoing harmonic layer. Transition history, variation bonuses and impact-mode preferences remain removed, so identical audio and settings produce identical cue pairs, loops, curves and rendered audio.

## MIX END tempo and buffer continuity

The incoming deck remains at synchronized BPM for at least one complete bar after MIX END. Tempo restoration then uses 16 source-to-target map subdivisions per bar with a smootherstep/geometric rate trajectory.

Fallback transitions resume at `next_start + length`; local gain reaches unity at the boundary; advanced renders crossfade their final ~24 ms to the exact B samples immediately preceding `next_resume_sample`. Time-stretch length shortfalls do not receive zero padding.

## Verification

- 87 pytest cases pass.
- `check_drum_loop_bridge.py` verifies exact two-beat repetition, loop-wrap continuity and return to the live B stem.
- `check_natural_transition.py` verifies percussion-only lead-in, cue-time ownership, bass complementarity, continuous controls and peak headroom.
- `check_mixend_continuity.py` verifies tempo hold, dense recovery mapping and sample-continuous promotion.
- Existing preload, seek, phase-lock, time-stretch and CUE-DETR-only tests remain active.
