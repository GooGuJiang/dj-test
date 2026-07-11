# Beat This! + CUE-DETR + MuQ + All-In-One Auto DJ 1.2.15

这是一个面向本地音乐文件的自动 DJ 实验程序。1.2.15 将过于激进的 2 拍交接扩展为 cue 居中的 4 拍平滑交替：切点前逐步建立下一首的鼓组和主体，切点时完成低频与主导权交换，切点后保留约一拍旧歌尾巴，同时继续保留 1.2.14 的 MIX END 变速与缓冲区连续性修复。

## 模型分工

- **CUE-DETR (`disco-eth/cue-detr`)**：唯一的 IN/OUT cue 候选来源。
- **Beat This!**：beat/downbeat 网格，并把 CUE-DETR 预测吸附到真实 downbeat。
- **All-In-One**：intro、verse、chorus、break、outro 等结构角色，只参与排序。
- **MuQ**：播放顺序、风格连续性和 cue 对兼容度。
- **本地 DSP**：BPM 同步、逐拍相位锁定、低频所有权、HPSS 和最终转场渲染。

## 1.2.15 的核心变化

### 1. 取消“尾部接头部”硬约束

1.2.11 曾强制：

```text
A OUT：后 42% / 最后 48 小节
B IN ：前 35% / 前 32 小节
```

1.2.15 继续完全移除这些位置门槛和 `tail_to_head` 加分。现在所有有效 CUE-DETR cue 都可以参加配对，候选只根据以下因素排序：

- downbeat 与乐句边界；
- CUE-DETR IN/OUT 置信度；
- BPM/鼓组相位稳定性；
- 低频碰撞和双人声风险；
- 音量、能量弧线和频谱连续性；
- 调性、All-In-One 结构与 MuQ 局部兼容度。

歌曲中的位置仍会写入诊断指标，但权重为 0，不会淘汰中段或后段的优秀 cue。

### 2. CUE-DETR cue 是主交接点

每个选中的 OUT/IN cue 都保留为精确交接锚点，而不是把 cue 当成长淡化的开始。默认可听窗口为 **4 拍**：

```text
cue 前 2 拍：B 从低电平鼓组预入逐步进入主融合，A 保持稳定 groove
cue 当拍   ：B 的鼓组与主体已占优，bass 所有权平滑交换
cue 后约 1 拍：A 继续保留可听尾巴并逐步释放
cue 后第 2 拍：只剩 B，以正常电平进入 MIX END
```

主旋律/和声的实际交叉区约 2–2.5 拍，鼓组交叉约 2 拍，bass 交换约 0.7–0.9 拍。这样能听到明确过渡，又不会回到 8/16/32 小节长混。如果 cue 靠近歌曲边界，窗口只按可用音频缩短，不另造切点，也不移动 CUE-DETR 结果。

### 3. 自动转场只保留三种短手法

保留：

- **Short Blend**：默认一小节内的平滑融合，适合大多数兼容 cue 对。
- **Bass Swap**：在 cue 附近更快交换低频所有权。
- **Echo Out**：仅在人声重叠或调性风险较高时，给旧歌和声层增加极短尾音。

删除自动渲染分支：Drop Swap、Double Drop、Loop Out、Filter Ride、Post-Drop Relay、Breakdown Lift。

统一执行：

```text
1. BPM、beat、downbeat 和局部 kick 相位对齐
2. cue 前两拍先轻量预入 B，再建立可听见的鼓组/主体融合
3. cue 当拍让 B 的鼓组和主体占优，并平滑交换 bass 所有权
4. cue 后保留约一拍 A 的鼓组/人声/和声尾巴，再完全释放
5. 第四拍末 B 以正常电平无缝继续播放
```

低频使用互补增益 `bass_A + bass_B = 1`。主包络使用连续 equal-power 曲线，因此切换明确但不会像突然停止播放。Echo 只处理旧歌的和声层，不把 kick 和 sub 送进 delay。


### 4. MIX END 后不立即变速

下一首在接歌窗口内保持同步 BPM。MIX END 后先至少保持一个完整小节，再按每小节 16 个连续 time-map 分段，用五次缓动逐渐恢复原 BPM。音高保持由 Rubber Band R3 或连续可变相位声码器负责。

```text
MIX END → 保持同步 BPM ≥ 1 小节 → 连续恢复 → 原 BPM
```

速度恢复不会再贴着低频交接点开始，也不会按“一小节一个速度台阶”跳变。

### 5. MIX END 连续接缝

- Simple Crossfade / Gapless 从最后实际播放的 B sample 紧接继续，不再沿用 phrase-warp 的恢复点。
- 下一首局部响度补偿在 MIX END 前平滑回到 unity，避免 promote 时音量突跳。
- 高级预渲染尾端用约 24 ms equal-power seam 接到下一首实时缓冲区。
- time-stretch 后端长度少量不足时使用连续边缘 sample，不再补零制造静音缝。

这段 seam 只用于去点击，不延长两首歌的音乐重叠。

### 6. 去除“为了变化而变化”

删除真人变化度、最近手法惩罚和随机化式候选奖励。相同音频、相同分析和相同设置会得到完全相同的转场结果，便于试听、复现和调试。

## 安装与升级

已有 1.2.11 环境不需要重新安装模型或 CUDA：

```powershell
conda activate beatthis-auto-dj
cd beatthis_muq_cuedetr_auto_dj_gui_v1215
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
Cue 配对上下文：自动（4 / 8 / 16 小节；实际窗口固定约 4 拍）
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
python check_mixend_continuity.py
```

当前结果：

```text
83 passed
PASS: cue-centered transition invariants satisfied
PASS: MIX END continuity invariants satisfied
```

专项覆盖：

- 所有 CUE-DETR cue 均可参与，不存在 42%/48 小节与 35%/32 小节门槛；
- 歌曲位置不参与评分；
- cue 前两拍逐步建立下一首，cue 前已有可听鼓组与主体交叉；
- cue 当拍下一首占优，但上一首仍保留约一拍平滑尾巴；
- 交叉区不存在音量包络空洞，也不存在硬切；
- bass 所有权之和恒为 1；
- 控制曲线连续，首尾所有权正确；
- 干净的歌曲配对不会滥用 Echo；
- 高人声/低调性兼容时才加入 Echo 候选；
- 已删除的冲击式手法会明确拒绝；
- MIX END 后至少一小节速度保持、每小节 16 段连续 BPM 恢复；
- Simple/Gapless 播放位置连续、下一首增益在 MIX END 达到 unity；
- 高级渲染 24 ms 接缝、time-stretch 不补零、实时 callback 跨 MIX END 无零样本；
- 完整预加载、Seek、BPM 恢复、相位锁定和 CUE-DETR-only 回归测试继续通过。

## 研究与实现依据

参见 `SOURCES.md` 与 `RESEARCH_NOTES.md`。核心原则是：准确 BeatGrid 是 Sync、循环和节拍效果的前提；phrase/downbeat 决定接歌结构；平滑 crossfade 与明确低频所有权优先于堆叠效果。
