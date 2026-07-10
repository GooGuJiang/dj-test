# 1.2.13 升级说明：CUE-DETR cue 居中短交接

## 是否需要重新安装模型？

不需要。Beat This!、CUE-DETR、MuQ 和 All-In-One 的模型与缓存格式没有变化。

## 行为变化

1.2.12 把 cue 作为自然融合片段的开始；1.2.13 把 cue 作为真正的主导权交接点。

默认时间线：

```text
cue - 1 beat   B 的鼓组低电平预入
cue            B 鼓组占优，bass 快速换手
cue + <1 beat  A 鼓组/主体平滑归零
```

实际双歌窗口约 2 拍。GUI 的 4/8/16/32 小节选项现在只控制 cue 对的结构评分上下文，不再控制可听重叠长度。

## 旧设置兼容

- `Adaptive Human` 继续自动迁移到 `Natural Auto`。
- 旧设置中的 `Long Blend` 会自动迁移为 `Short Blend`；新界面不再显示长混模式。
- 保存的“过渡长度”数值仍可读取，但语义变为“Cue 配对上下文”。

## 验证

```bash
pytest -q
python check_natural_transition.py
```

预期：74 项测试通过，独立检查输出 `PASS: cue-centered transition invariants satisfied`。
