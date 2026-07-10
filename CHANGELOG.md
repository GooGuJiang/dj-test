# Changelog

## 0.5.0

- 离线 beat/downbeat 分析从 BeatNet 替换为 Beat This! 1.1.0。
- 使用 `File2Beats`、`infer_beat_numbers()` 和 final0/small0 模型。
- 删除运行时对 madmom、PyAudio 和 BeatNet 的要求。
- 新增稳定电子节奏漏拍修复与重复峰合并。
- BPM 恢复从逐小节拼接改为整轨连续 source-target time map。
- Rubber Band R3 使用 `--timemap` 和 centre-focus。
- 新增单次 STFT 连续可变 phase-vocoder 回退。
- BPM 曲线改为五次缓动和几何插值。
- BPM 恢复后重新渲染 transition endpoint，减少 MIX END 接缝。
- GUI 增加 Beat This! final0/final1/small0 选择。
- 新增 Beat This! 安装和验证脚本。

## 0.4.0

- 论文 cue/phrase 分析、Graph-Cut Spectral Seam 和 AutoMix-like 策略。
