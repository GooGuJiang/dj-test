# Primary Research and Model Sources

本项目 0.6.1 的算法选择主要参考以下公开的一手来源：

1. MuQ official model card — https://huggingface.co/OpenMuQ/MuQ-large-msd-iter
2. MuQ official repository — https://github.com/tencent-ailab/MuQ
3. MuQ paper — https://arxiv.org/abs/2501.01108
4. Beat This! official implementation — https://github.com/CPJKU/beat_this
5. Beat This! paper — https://arxiv.org/abs/2407.21658
6. AutoMashup — https://arxiv.org/abs/2508.06516
7. Efficient Feature Aggregation / MuQ-token — https://arxiv.org/abs/2604.20847
8. SELEBI percussion-aware time stretching — https://arxiv.org/abs/2602.16421
9. DJ-AI playlist alignment — https://doi.org/10.1145/3771594.3771640

## 实现与论文的边界

- MuQ 使用官方公开推理接口和模型名。
- MuQ-token 仅启发本项目的多层聚合；未复现其推荐训练框架。
- SELEBI 尚未有成熟官方 Python 实现被集成；本项目的 HPSS+WSOLA 是工程替代方案，不是逐公式复现。
- MuQ 权重为 CC-BY-NC 4.0，不适合未经授权的商业使用。

## All-In-One 0.9.0

10. All-In-One official repository — https://github.com/mir-aidj/all-in-one
11. All-In-One paper — https://arxiv.org/abs/2307.16425
12. All-In-One PyPI package — https://pypi.org/project/allin1/
13. NATTEN official repository — https://github.com/SHI-Labs/NATTEN
14. All-In-One NATTEN compatibility issue — https://github.com/mir-aidj/all-in-one/issues/33

### All-In-One 实现边界

- 使用官方 `allin1.analyze()` 和 `harmonix-all` 权重。
- Beat This! 的 beat/downbeat 不被 All-In-One 覆盖，避免两个节拍网格互相漂移。
- All-In-One 的 Harmonix 功能段会映射成 DJ 角色，但 GUI 同时保留原始功能标签。
- 没有宣称 All-In-One 能直接识别 Raveform 的 buildup/drop/cooldown；DROP 等 DJ 角色由 All-In-One 功能标签与局部能量/鼓组特征融合得到。
- 纯 PyTorch NATTEN 是兼容回退，不是官方 CUDA 内核，速度较慢。

