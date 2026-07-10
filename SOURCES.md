# Sources

## Beat, phrase and mixer behavior

- rekordbox 7 overview — BPM/Grid Analysis, Key/Phrase Analysis, and manual grid correction: https://rekordbox.com/en/feature/overview/
- rekordbox 7 manual and FAQ — track analysis plus editable BeatGrid/Phrase positions: https://cdn.rekordbox.com/files/20250313121537/rekordbox7.1.0_manual_EN.pdf
- Serato — Introduction to Beatgrids: https://support.serato.com/hc/en-us/articles/202523390-Introduction-to-Beatgrids
- Serato — Beatgrids in Serato DJ Pro: https://support.serato.com/hc/en-us/articles/202856014-Beatgrids-in-Serato-DJ-Pro
- Ableton Live 12 manual — mixer crossfader and smooth transitions: https://www.ableton.com/en/manual/mixing/
- Ableton Live 12 manual — tempo-synchronised delay: https://www.ableton.com/en/manual/live-audio-effect-reference/
- Mixxx manual — Beatmatching and Mixing: https://manual.mixxx.org/2.5/en/chapters/djing_with_mixxx.html#beatmatching-and-mixing

These sources support the implementation principles, not a claim that the project reproduces any proprietary Auto DJ algorithm.

## CUE-DETR

- ISMIR 2024 paper — “Cue Point Estimation using Object Detection”: https://arxiv.org/abs/2407.06823

- Official repository: https://github.com/ETH-DISCO/cue-detr
- Official checkpoint: https://huggingface.co/disco-eth/cue-detr
- Official example: `cue_points.py` uses 22.05 kHz Mel spectrograms, width-355 windows, 75% overlap and peak filtering.

CUE-DETR remains the sole source of transition cue locations. Beat This! only quantizes those predictions to downbeats.

## Models and DSP

- Beat This!: https://github.com/CPJKU/beat_this
- MuQ: https://huggingface.co/OpenMuQ/MuQ-large-msd-iter
- All-In-One: https://github.com/mir-aidj/all-in-one
- Rubber Band: https://breakfastquay.com/rubberband/
