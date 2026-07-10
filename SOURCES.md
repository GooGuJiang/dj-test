# Sources

## CUE-DETR

- Official repository: https://github.com/ETH-DISCO/cue-detr
- Official checkpoint: https://huggingface.co/disco-eth/cue-detr
- EDM-CUE dataset metadata: https://huggingface.co/datasets/disco-eth/edm-cue

The adapter follows the public `cue_points.py` inference constants and processing sequence. CUE-DETR code/checkpoint licensing must be reviewed from the official repository/model card before redistribution.

## Mixxx architecture reference

- Mixxx repository: https://github.com/mixxxdj/mixxx
- EngineBuffer implementation: `src/engine/enginebuffer.cpp`

Only architectural principles were used: cached reading, read-ahead, worker scheduling, quantization and separation of offline work from the real-time callback. No Mixxx GPL source code was copied into this project.

## Other models

- Beat This!: https://github.com/CPJKU/beat_this
- MuQ: https://huggingface.co/OpenMuQ/MuQ-large-msd-iter
- All-In-One: https://github.com/mir-aidj/all-in-one
- Rubber Band: https://breakfastquay.com/rubberband/
