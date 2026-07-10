# Test report — Auto DJ 1.2.14 MIX END continuity

## Scope

This release fixes two related playback defects around `MIX END`:

1. the incoming song began restoring its original BPM immediately after the transition, producing an audible speed/pitch-rise impression;
2. the transition renderer and the promoted incoming-track buffer could use different resume coordinates or gains, producing a short interruption, skipped/repeated samples, or a waveform discontinuity.

## Implementation checks

### Tempo restoration

- `MIX END` is now treated as the end of the DJ handoff, not the start of tempo restoration.
- The incoming track remains at synchronized BPM for at least one complete source bar after `MIX END`.
- Restoration uses 16 source-to-target time-map subdivisions per bar.
- Local speed follows smootherstep-shaped geometric interpolation.
- `tempo_restore_start` metadata points to the actual post-hold restore start, so GUI BPM reporting matches rendered audio.

### Playback continuity

- Simple Crossfade and Gapless resume from `next_start + length`, the exact sample after the linearly consumed incoming buffer.
- Incoming local loudness compensation returns to unity at `MIX END`.
- Advanced phrase-warp/phase-lock rendering crossfades its final ~24 ms to the exact incoming samples immediately before `next_resume_sample`.
- Time-stretch length shortfalls use edge continuation rather than zero padding.
- The real-time audio callback is tested across the promotion boundary in one output buffer.

## Automated tests

Command:

```bash
python -m pytest -q
```

Result:

```text
........................................................................ [ 90%]
........                                                                 [100%]
80 passed in 11.92s
```

New v1.2.14 tests: 6

1. fallback curves reach unity at `MIX END`;
2. simple transitions resume at the exact next sample;
3. rendered transition tails are stitched to the live incoming buffer;
4. tempo restoration holds at least one bar and uses a dense map;
5. phrase-length correction never pads a silence gap;
6. the real-time callback crosses `MIX END` without zero samples or a position jump.

## Standalone DSP checks

### Cue-centered transition

```text
Short Blend peak=0.2871 bass_error=0.00e+00 pre_B=0.183 cue_drums=0.324->0.946 post_bass=0.049->0.951 A@70%=0.000 max_step=0.0007
Bass Swap   peak=0.2871 bass_error=0.00e+00 pre_B=0.183 cue_drums=0.324->0.946 post_bass=0.000->1.000 A@70%=0.000 max_step=0.0012
Echo Out    peak=0.2871 bass_error=0.00e+00 pre_B=0.183 cue_drums=0.324->0.946 post_bass=0.005->0.995 A@70%=0.000 max_step=0.0010
PASS: cue-centered transition invariants satisfied
```

### MIX END continuity

```text
fallback endpoints A=-0.000000/0.000000 B=1.000000/1.000000/1.000000
render seam=24.00 ms jump=0.400006->0.011974 natural=0.011974
PASS: MIX END continuity invariants satisfied
```

The post-stitch discontinuity equals the incoming audio's normal adjacent-sample step; it is no longer an artificial renderer-to-player jump.

## Static checks

```bash
python -m compileall -q autodj app.py check_natural_transition.py check_mixend_continuity.py
git diff --check
```

Both completed successfully.

## Limitations

The code can guarantee continuous coordinates, envelopes, and buffers. Final listening quality still depends on accurate Beat This! beat/downbeat analysis, CUE-DETR cue placement, and the installed pitch-preserving time-stretch backend. Rubber Band R3 remains the recommended backend for production use.
