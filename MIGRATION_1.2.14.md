# 1.2.14 升级说明：MIX END 速度与连续性修复

## 修复内容

1. MIX END 后不再立即恢复原 BPM。下一首先保持至少一个完整小节的同步速度。
2. BPM 恢复从每小节一个速度段改为每小节 16 个连续 time-map 分段，并使用五次缓动与几何速度插值。
3. Simple Crossfade / Gapless 的继续播放位置改为 `next_start + length`，不再错误沿用高级 phrase-warp 的 resume 点。
4. 下一首局部响度补偿会在 MIX END 前平滑回到 unity，避免切换 current deck 时发生增益跳变。
5. 高级预渲染尾端增加约 24 ms equal-power 去点击接缝，精确接入下一首实时缓冲区。
6. time-stretch 输出少量短于目标时不再补零，避免 MIX END 前出现静音槽。

## 不变内容

- CUE-DETR 仍是唯一 IN/OUT cue 来源。
- cue 前约一拍预入、cue 当拍交接、cue 后不到一拍释放旧歌。
- 鼓拍同步、downbeat 对齐和低频单一所有权保持不变。
- 自动手法仍只保留 Short Blend、Bass Swap、Echo Out。

## 验证

```bash
pytest -q
python check_natural_transition.py
python check_mixend_continuity.py
```

本版本包含 80 项测试，其中 6 项专门覆盖 MIX END 速度保持、密集 tempo map、恢复位置、增益 unity、预渲染接缝、零填充与实时 callback 连续性。
