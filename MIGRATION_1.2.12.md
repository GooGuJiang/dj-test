# 1.2.12 升级说明：自由 cue 配对与自然接歌

## 是否需要重新安装模型

不需要。1.2.12 没有更换 Beat This!、CUE-DETR、MuQ 或 All-In-One 权重。

## 设置迁移

- `Adaptive Human` 自动转换为 `Natural Auto`。
- 已删除的 Drop Swap、Double Drop、Loop Out、Filter Ride、Post-Drop Relay、Breakdown Lift 不再出现在 GUI 中。
- “真人变化度”设置已删除。
- 候选手法数最大值改为 3。

## 切点行为

旧版尾部/头部硬窗口已删除。CUE-DETR 仍然是唯一 cue 来源，但所有 neural cue 都可以参加配对。`earliest_start` 仍会保护已播放区域，强制 Seek 后只使用当前位置之后的 OUT cue。

## 转场行为

默认执行乐句长混或低频交接：下一首鼓组先建立、低频在乐句中段交换、上一首再退出。Echo 仅作为人声/调性冲突时的安全回退。

## 验证

```powershell
conda activate beatthis-auto-dj
cd beatthis_muq_cuedetr_auto_dj_gui_v1212
pytest -q
python check_natural_transition.py
python app.py
```
