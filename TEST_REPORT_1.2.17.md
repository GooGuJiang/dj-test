# Test report — Auto DJ 1.2.17 endpoint-safe fade and graph search

## New endpoint behavior

When playback has passed the final neural OUT cue, or the selected cue has less than a quarter beat of post-cue audio, the matcher creates a physical-tail window instead of failing. The current track reaches zero at its real end while the incoming track reaches unity. The callback promotes B without inserting silence.

## Graph-search behavior

Playlist ordering now uses pre-sorted directed neighbors, bit-mask states, one-step continuation potential, Pareto transposition pruning, and deterministic swap/relocate polishing. The requested first track remains fixed.

## Regression result

```text
90 passed in 15.84s
```

New tests cover:

1. playback already beyond the final CUE-DETR OUT;
2. endpoint tail crossfade reaching A=0 and B=1;
3. realtime callback crossing the boundary without a zero block;
4. local graph-path polishing replacing a weak internal edge while preserving the first track.
