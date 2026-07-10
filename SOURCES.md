# Sources

## Active models

- Beat This!: https://github.com/CPJKU/beat_this
- MuQ-large-msd-iter: https://huggingface.co/OpenMuQ/MuQ-large-msd-iter
- SongFormer model: https://huggingface.co/ASLP-lab/SongFormer
- SongFormer official implementation: https://github.com/ASLP-lab/SongFormer

## Transition research used by the project

- Automatic DJ cue-point detection: https://arxiv.org/abs/2007.08411
- CUE-DETR: https://arxiv.org/abs/2407.06823
- Graph-Cut Crossfading: https://arxiv.org/abs/2301.13380
- DJtransGAN: https://arxiv.org/abs/2110.06525
- Raveform dataset/taxonomy: https://mir-aidj.github.io/raveform/

## Runtime notes

- SongFormer runs in an isolated Python environment.
- Beat This! remains the beat/downbeat timing authority.
- MuQ embeddings are used for directional playlist ranking.
- No EDMFormer or All-In-One inference path is included in version 1.2.1.

## 1.2.3 CUDA runtime

- PyTorch official Start Locally / Previous Versions documentation: CUDA 12.8 wheels for Windows and Linux, including PyTorch 2.7.1.
- NVIDIA GeForce RTX 5070 official product page: RTX 50 series / Blackwell GPU.
- ASLP-lab SongFormer Hugging Face model card: official example moves the model to `cuda:0` before inference.
