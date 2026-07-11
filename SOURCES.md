# Sources

## BeatGrid, sync, loops and mixer behavior

- rekordbox 7 overview — BPM/Grid Analysis, Key/Phrase Analysis, and manual grid correction: https://rekordbox.com/en/feature/overview/
- rekordbox 7 manual and FAQ — track analysis plus editable BeatGrid/Phrase positions: https://cdn.rekordbox.com/files/20250313121537/rekordbox7.1.0_manual_EN.pdf
- Serato — Introduction to Beatgrids: https://support.serato.com/hc/en-us/articles/202523390-Introduction-to-Beatgrids
- Serato — Auto Looping: https://support.serato.com/hc/en-us/articles/226382007-Auto-Looping
- Serato — Quantize: https://support.serato.com/hc/en-us/articles/225122628-Quantize
- Serato — Sample Player Repeat / loop Beat Sync requirements: https://support.serato.com/hc/en-us/articles/227564847-Sample-Player-Repeat
- Serato — Sync with BeatGrid: https://support.serato.com/hc/en-us/articles/203056994-SYNC-with-Serato-DJ
- Serato — Mixing / Keylock keeps musical key while tempo changes: https://support.serato.com/hc/en-us/articles/360000067636-Mixing
- Ableton Live 12 manual — mixer crossfader creates smooth transitions between clips: https://www.ableton.com/en/manual/mixing/
- Ableton Live 12 manual — adjacent clip-edge crossfades avoid discontinuities: https://www.ableton.com/en/manual/comping/
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

## 1.2.18 research-guided sequencing and transition updates

- CUE-DETR paper — cue points align closely with expert annotations and underlying musical structure: https://arxiv.org/abs/2407.06823
- M-DJCUE expert annotation guidelines — cue-ins should be able to stand alone and should not be too quiet; cue-outs may define a fade duration: https://github.com/MZehren/M-DJCUE
- Automatic DJ Transitions with Differentiable Audio Effects — real-world DJ transition modelling with explicit EQ and fader controls: https://arxiv.org/abs/2110.06525
- The algorithmic nature of song-sequencing — curated sequences often alternate increases and decreases in energy/valence: https://arxiv.org/abs/2408.04383
- The Importance of Song Context and Song Order in Automated Music Playlist Generation — playlist continuation benefits from song context: https://arxiv.org/abs/1807.04690
- Diverse Beam Search — preserving diverse beam hypotheses can improve exploration/exploitation and top-1 solutions: https://arxiv.org/abs/1610.02424
- Bellman–Held–Karp subset dynamic programming is the classical exact approach for small Hamiltonian/TSP-like sequencing problems: https://en.wikipedia.org/wiki/Held%E2%80%93Karp_algorithm
