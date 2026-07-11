# 1.2.17 升级说明：尾端安全淡化与图搜索优化

## 是否需要重新安装模型

不需要。CUE-DETR、Beat This!、All-In-One、MuQ 与 Rubber Band 环境保持兼容。

## 行为变化

- 最后一个 CUE 已经过期或贴近歌曲末端时，播放器会从当前实际尾部渐变到下一首，不再因“cue 后没有足够音频”停止。
- 端点安全转场固定使用 `Tail Crossfade`，不会调用鼓循环、HPSS 或 Spectral Seam。
- MuQ 智能排序仍固定当前首，但搜索增加前瞻、去重和局部路径修复，较大的队列通常会减少单条异常弱衔接。

## 验证

```bash
pytest -q
python check_drum_loop_bridge.py
python check_natural_transition.py
python check_mixend_continuity.py
```

预期完整测试：`90 passed`。
