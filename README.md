# Beat This! + CUE-DETR + MuQ + All-In-One Auto DJ 1.2.12

这是一个面向本地音乐文件的自动 DJ 实验程序。1.2.12 重点解决“切点被尾部→头部硬规则锁死”和“自动效果过多导致衔接突兀”两个问题。

## 模型分工

- **CUE-DETR (`disco-eth/cue-detr`)**：唯一的 IN/OUT cue 候选来源。
- **Beat This!**：beat/downbeat 网格，并把 CUE-DETR 预测吸附到真实 downbeat。
- **All-In-One**：intro、verse、chorus、break、outro 等结构角色，只参与排序。
- **MuQ**：播放顺序、风格连续性和 cue 对兼容度。
- **本地 DSP**：BPM 同步、逐拍相位锁定、低频所有权、HPSS 和最终转场渲染。

## 1.2.12 的核心变化

### 1. 取消“尾部接头部”硬约束

1.2.11 曾强制：

```text
A OUT：后 42% / 最后 48 小节
B IN ：前 35% / 前 32 小节
```

1.2.12 已完全移除这些位置门槛和 `tail_to_head` 加分。现在所有有效 CUE-DETR cue 都可以参加配对，候选只根据以下因素排序：

- downbeat 与乐句边界；
- CUE-DETR IN/OUT 置信度；
- BPM/鼓组相位稳定性；
- 低频碰撞和双人声风险；
- 音量、能量弧线和频谱连续性；
- 调性、All-In-One 结构与 MuQ 局部兼容度。

歌曲中的位置仍会写入诊断指标，但权重为 0，不会淘汰中段或后段的优秀 cue。

### 2. 自动转场收敛为三种保守手法

保留：

- **Long Blend**：适合调性、人声和能量兼容的长乐句融合。
- **Bass Swap**：下一首鼓组先建立，在乐句中点附近交换低频所有权。
- **Echo Out**：只在双人声风险高或调性兼容较低时作为安全退出候选。

删除自动渲染分支：

- Drop Swap
- Double Drop
- Loop Out
- Filter Ride
- Post-Drop Relay
- Breakdown Lift

旧设置中的 `Adaptive Human` 会自动迁移为 `Natural Auto`。被删除的手法无法再直接渲染。

### 3. 按真实接歌公式渲染

自动转场统一执行：

```text
1. 两首歌按 BPM、beat 和 downbeat 对齐
2. 在乐句边界开始重叠
3. B 的鼓组先平滑进入，B 的低频保持关闭
4. A 的鼓组继续维持 groove，避免提前出现能量洞
5. 乐句中段交换 bass 所有权
6. A 的和声/人声与鼓组平滑退出
7. B 在过渡终点完全接管
```

低频使用互补增益：

```text
bass_A + bass_B = 1
```

不再使用会在交接中点造成双低频增益的 equal-power bass 叠加。Echo 仅处理旧歌的和声层，低频不会送入 delay，反馈也限制在保守范围。

### 4. 去除“为了变化而变化”

删除真人变化度、最近手法惩罚和随机化式候选奖励。相同音频、相同分析和相同设置会得到完全相同的转场结果，便于试听、复现和调试。

## 安装与升级

已有 1.2.11 环境不需要重新安装模型或 CUDA：

```powershell
conda activate beatthis-auto-dj
cd beatthis_muq_cuedetr_auto_dj_gui_v1212
python verify_cuedetr.py
python app.py
```

首次安装：

```powershell
python install_cuedetr.py
python verify_cuedetr.py --download --device cuda
python app.py
```

## 推荐设置

```text
过渡长度：自动（8 / 16 / 32 小节）
自然接歌策略：Natural Auto
候选手法数：3
效果强度：45%–70%
AutoMix 策略：AutoMix-like
时间拉伸：Rubber Band R3
CUE-DETR 设备：cuda 或 auto
CUE-DETR 灵敏度：0.88–0.93
最小 cue 间隔：8 或 16 小节
```

人声密集的流行音乐建议把效果强度控制在 45%–60%；纯器乐 House/Techno 可使用 60%–70%。

## 使用流程

1. 添加歌曲并等待 Beat This!、All-In-One、CUE-DETR 和 MuQ 分析完成。
2. 等待当前歌曲与下一首的热预加载完成。
3. 点击播放；播放线程会复用预渲染结果。
4. 时间轴显示 OUT、IN、乐句长度、结构角色和最终自然接歌手法。
5. 如果自动分析的 BeatGrid 或 phrase 明显错误，应先修正分析数据，而不是增加效果强度掩盖错拍。

## 测试

完整测试：

```bash
pytest -q
```

自然转场独立检查：

```bash
python check_natural_transition.py
```

当前结果：

```text
71 passed
PASS: natural transition renderer invariants satisfied
```

专项覆盖：

- 所有 CUE-DETR cue 均可参与，不存在 42%/48 小节与 35%/32 小节门槛；
- 歌曲位置不参与评分；
- 下一首鼓组先建立，再释放上一首；
- bass 所有权之和恒为 1；
- 控制曲线连续，首尾所有权正确；
- 干净的歌曲配对不会滥用 Echo；
- 高人声/低调性兼容时才加入 Echo 候选；
- 已删除的冲击式手法会明确拒绝；
- 完整预加载、Seek、BPM 恢复、相位锁定和 CUE-DETR-only 回归测试继续通过。

## 研究与实现依据

参见 `SOURCES.md` 与 `RESEARCH_NOTES.md`。核心原则是：准确 BeatGrid 是 Sync、循环和节拍效果的前提；phrase/downbeat 决定接歌结构；平滑 crossfade 与明确低频所有权优先于堆叠效果。
