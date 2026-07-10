# Engineering notes — 1.2.10

## Cue source

The official `disco-eth/cue-detr` checkpoint is the sole transition-candidate generator. The adapter follows the public example: 22.05 kHz Mel power spectrogram, viridis RGB conversion, width-355 windows, 75% overlap, DETR object predictions, merged overlapping detections and sensitivity peak filtering.

Predicted times are subsequently snapped to Beat This! downbeats. All-In-One labels, MuQ embeddings, Camelot/key, energy and stem collision scores can rank a CUE-DETR candidate but cannot create an additional candidate.

The old checkerboard/period/salience cue proposal is no longer executed by `analyze_edm_structure`; that module now returns role/key/energy support features with zero cue arrays.

## Read-ahead architecture

The architecture is inspired by Mixxx's separation between cached reading, read-ahead management, worker scheduling and real-time engine buffers. No Mixxx source code is copied.

There are three cache levels:

- analysis profiles keyed by source file and model configuration;
- warm prepared tracks keyed by outgoing/incoming/target BPM/DSP configuration;
- fully rendered adjacent pairs containing synchronized base track, final restored incoming track and rendered TransitionPlan.

The audio stream is not started until the immediate pair is fully rendered. Additional adjacent pairs are rendered within the configured sliding window and evicted with an LRU-like memory budget.

## Safety behavior

If a CUE-DETR profile is missing, the engine raises an explicit error instead of invoking a local cue heuristic. Endpoint safety cues are only created inside the CUE-DETR adapter when the neural output contains no usable prediction after filtering; they are tagged with low confidence.
