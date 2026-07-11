# 1.2.16 升级说明：量化鼓循环桥

## 为什么修改

1.2.15 已将硬切放宽为 4 拍平滑交接，但下一首鼓组只在约 2 拍内正式交叉。试听反馈希望先让下一首节奏重复几拍，为主体进入留出时间，同时避免把两首完整歌曲叠得更久。

## 新时序

```text
cue 前 4 拍：B 的 2 拍 percussion loop 重复两次，A 保持主体和 bass
cue 前约 1.5 拍：真实 B 鼓组与 harmonic 开始进入
cue 当拍：B 占主导，bass 换手
cue 后约 1 拍：A 鼓组退出
cue 后第 2 拍：MIX END
```

## 兼容性

- 不改变 CUE-DETR cue。
- 不恢复尾部接头部硬约束。
- 不循环 bass、旋律或人声。
- Short Blend、Bass Swap、Echo Out 三种手法继续保留。
- 1.2.14 的 MIX END 速度与缓冲连续性修复继续有效。
- 旧设置文件可以直接使用，无需迁移字段。

## 验证

```bash
pytest -q
python check_drum_loop_bridge.py
python check_natural_transition.py
python check_mixend_continuity.py
```

预期：87 项测试通过，三个独立检查均输出 `PASS`。
