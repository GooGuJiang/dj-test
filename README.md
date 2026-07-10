# Beat This! + CUE-DETR + MuQ + All-In-One Auto DJ 1.2.13

这是一个面向本地音乐文件的自动 DJ 实验程序。1.2.13 把 CUE-DETR 标记从“长过渡起点”改为真正的换曲交接点：切点前只做一拍低电平预入，切点当拍完成鼓组与低频主导权交接，切点后约一拍内平滑释放旧歌。

## 模型分工

- **CUE-DETR (`disco-eth/cue-detr`)**：唯一的 IN/OUT cue 候选来源。
- **Beat This!**：beat/downbeat 网格，并把 CUE-DETR 预测吸附到真实 downbeat。
- **All-In-One**：intro、verse、chorus、break、outro 等结构角色，只参与排序。
- **MuQ**：播放顺序、风格连续性和 cue 对兼容度。
- **本地 DSP**：BPM 同步、逐拍相位锁定、低频所有权、HPSS 和最终转场渲染。

## 1.2.13 的核心变化

### 1. 取消“尾部接头部”硬约束

1.2.11 曾强制：

```text
A OUT：后 42% / 最后 48 小节
B IN ：前 35% / 前 32 小节
```

1.2.13 已完全移除这些位置门槛和 `tail_to_head` 加分。现在所有有效 CUE-DETR cue 都可以参加配对，候选只根据以下因素排序：

- downbeat 与乐句边界；
- CUE-DETR IN/OUT 置信度；
- BPM/鼓组相位稳定性；
- 低频碰撞和双人声风险；
- 音量、能量弧线和频谱连续性；
- 调性、All-In-One 结构与 MuQ 局部兼容度。

歌曲中的位置仍会写入诊断指标，但权重为 0，不会淘汰中段或后段的优秀 cue。

### 2. CUE-DETR cue 是主交接点

每个选中的 OUT/IN cue 都保留为精确交接锚点，而不是把 cue 当成长淡化的开始。默认可听窗口为 **2 拍**：

```text
cue 前 1 拍：B 只做低电平鼓组预入，A 仍保持主体
cue 当拍   ：B 的鼓组已经占优，bass 所有权快速交换
cue 后 <1 拍：A 的鼓组与主体平滑归零，B 完全接管
```

如果 cue 靠近歌曲边界，窗口只会按可用音频缩短，不会另造切点，也不会移动 CUE-DETR 结果。

### 3. 自动转场只保留三种短手法

保留：

- **Short Blend**：默认短融合，适合大多数兼容 cue 对。
- **Bass Swap**：在 cue 附近更快交换低频所有权。
- **Echo Out**：仅在人声重叠或调性风险较高时，给旧歌和声层增加极短尾音。

删除自动渲染分支：Drop Swap、Double Drop、Loop Out、Filter Ride、Post-Drop Relay、Breakdown Lift。

统一执行：

```text
1. BPM、beat、downbeat 和局部 kick 相位对齐
2. cue 前一拍只轻量预入 B 的鼓组
3. cue 当拍让 B 的鼓组强于 A，并交换 bass 所有权
4. cue 后不到一拍释放 A 的鼓组/人声/和声
5. B 在窗口结束时以正常电平无缝继续播放
```

低频使用互补增益 `bass_A + bass_B = 1`。主包络使用连续 equal-power 曲线，因此切换明确但不会像突然停止播放。Echo 只处理旧歌的和声层，不把 kick 和 sub 送进 delay。

### 4. 去除“为了变化而变化”

删除真人变化度、最近手法惩罚和随机化式候选奖励。相同音频、相同分析和相同设置会得到完全相同的转场结果，便于试听、复现和调试。

## 安装与升级

已有 1.2.11 环境不需要重新安装模型或 CUDA：

```powershell
conda activate beatthis-auto-dj
cd beatthis_muq_cuedetr_auto_dj_gui_v1213
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
Cue 配对上下文：自动（4 / 8 / 16 小节；实际重叠固定约 2 拍）
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
74 passed
PASS: cue-centered transition invariants satisfied
```

专项覆盖：

- 所有 CUE-DETR cue 均可参与，不存在 42%/48 小节与 35%/32 小节门槛；
- 歌曲位置不参与评分；
- cue 前只允许低电平鼓组预入，cue 当拍下一首鼓组占优；
- cue 后不到一拍释放上一首，且不存在硬切；
- bass 所有权之和恒为 1；
- 控制曲线连续，首尾所有权正确；
- 干净的歌曲配对不会滥用 Echo；
- 高人声/低调性兼容时才加入 Echo 候选；
- 已删除的冲击式手法会明确拒绝；
- 完整预加载、Seek、BPM 恢复、相位锁定和 CUE-DETR-only 回归测试继续通过。

## 研究与实现依据

参见 `SOURCES.md` 与 `RESEARCH_NOTES.md`。核心原则是：准确 BeatGrid 是 Sync、循环和节拍效果的前提；phrase/downbeat 决定接歌结构；平滑 crossfade 与明确低频所有权优先于堆叠效果。
