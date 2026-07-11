# Beat This! + CUE-DETR + MuQ + All-In-One Auto DJ 1.2.16

这是一个面向本地音乐文件的自动 DJ 实验程序。1.2.16 在 1.2.15 的自然短接基础上加入 **量化鼓循环桥**：CUE-DETR 仍然决定真正换曲点，但在切点前先重复下一首 CUE 附近的鼓组，让听众提前进入下一首节奏；旋律、人声和低频仍然只在 CUE 附近交接，因此不会变成多拍双歌堆叠。

## 模型分工

- **CUE-DETR (`disco-eth/cue-detr`)**：唯一的 IN/OUT cue 候选来源。
- **Beat This!**：beat/downbeat 网格，并把 CUE-DETR 预测吸附到真实 downbeat。
- **All-In-One**：intro、verse、chorus、break、outro 等结构角色，只参与排序。
- **MuQ**：播放顺序、风格连续性和 cue 对兼容度。
- **本地 DSP**：BPM 同步、逐拍相位锁定、三段分频、鼓循环、低频所有权和最终转场渲染。

## 1.2.16 的核心变化

### 1. CUE-DETR 仍是主交接点

所有有效 CUE-DETR cue 都可以参加配对，不存在以下旧版硬约束：

```text
A OUT：后 42% / 最后 48 小节
B IN ：前 35% / 前 32 小节
```

歌曲位置只写入诊断指标，权重为 0。候选根据 downbeat、乐句、CUE-DETR 置信度、鼓组相位、低频/人声碰撞、能量、调性、结构角色和 MuQ 兼容度排序。

### 2. 默认使用“4 拍鼓循环桥 + 2 拍释放”

```text
CUE 前第 4～2 拍
├─ 从 B 的 CUE 当拍提取 2 拍 percussion loop
├─ 按 BeatGrid 重复两次
├─ B 鼓循环保持约 24% 电平
└─ A 继续保持 bass、主体和 groove

CUE 前约 1.5 拍
├─ B 的真实鼓组开始接管
├─ B 的旋律/和声开始进入
└─ 仍不开放 B 的完整低频

CUE 当拍
├─ B 鼓组与主体占优
├─ bass 所有权平滑换手
└─ A 保留短尾，不会硬断

CUE 后约 1 拍
├─ A 鼓组基本归零
└─ B 独立稳定播放

CUE 后第 2 拍
└─ MIX END，以正常增益继续下一首
```

新增的 2 拍不是把两首完整歌曲叠得更久，而是只让下一首的 percussion loop 提前出现。主旋律/和声实际交叉仍约 2–2.5 拍，bass 交换约 0.7–0.9 拍。

### 3. 鼓循环实现

- 循环素材来自下一首 **CUE 当拍之后的 2 拍 percussion stem**。
- 预入区有 4 拍，因此该 2 拍 loop 重复两次。
- 循环点严格按 BeatGrid 整拍对齐，不改变 CUE-DETR 的位置。
- 循环边界与 loop→live B 接点使用约 4 ms 去点击桥。
- 不循环 bass，也不循环 harmonic/vocal，避免双低频和双人声。
- 如果 cue 太靠近边界或没有足够音频，只缩短可用窗口，不创造新 cue。

### 4. 自动手法保持精简

保留：

- **Short Blend**：默认自然融合。
- **Bass Swap**：在 CUE 附近更明确地交换低频。
- **Echo Out**：只在人声重叠或调性风险较高时，对旧歌和声层添加极短 Echo。

已删除自动分支：Drop Swap、Double Drop、Loop Out、Filter Ride、Post-Drop Relay、Breakdown Lift。

统一原则：

```text
1. BPM、beat、downbeat 和局部 kick 相位对齐
2. 先用量化鼓循环建立下一首律动
3. CUE 附近再引入下一首主体与低频
4. CUE 当拍让 B 占优
5. A 用连续包络短尾退出
```

低频使用互补增益：

```text
bass_A + bass_B = 1
```

### 5. MIX END 连续性

保留 1.2.14/1.2.15 修复：

- MIX END 后至少保持一个完整小节的同步 BPM。
- BPM 恢复使用每小节 16 个 time-map 分段、五次缓动和几何速度插值。
- Simple Crossfade / Gapless 从最后实际播放的 B sample 连续继续。
- 下一首增益在 MIX END 前回到 unity。
- 高级预渲染尾端用约 24 ms equal-power seam 接到实时缓冲区。
- time-stretch 长度不足不再补零。

## 安装与升级

已有环境通常不需要重新安装模型或 CUDA：

```powershell
conda activate beatthis-auto-dj
cd beatthis_muq_cuedetr_auto_dj_gui_v1216
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
Cue 配对上下文：自动（4 / 8 / 16 小节）
实际过渡：CUE 前约 4 拍鼓循环 + CUE 后约 2 拍释放
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
3. 点击播放；播放线程复用预渲染结果。
4. 时间轴显示 OUT、IN、结构角色和最终手法。
5. BeatGrid 或 phrase 明显错误时，应修正分析结果，而不是增大效果强度掩盖错拍。

## 测试

完整测试：

```bash
pytest -q
```

独立检查：

```bash
python check_drum_loop_bridge.py
python check_natural_transition.py
python check_mixend_continuity.py
```

当前结果：

```text
87 passed
PASS: beat-quantized drum-loop bridge is continuous
PASS: quantized drum-loop transition invariants satisfied
PASS: MIX END continuity invariants satisfied
```

专项覆盖：

- CUE 前 4 拍窗口与 CUE 后 2 拍释放；
- 2 拍鼓片段在预入区重复两次；
- 循环边界无明显点击；
- CUE 后恢复真实 B percussion，不继续播放合成循环；
- 早段只出现 percussion，不提前叠加 B harmonic/bass；
- CUE 当拍 B 鼓组、主体和低频占优；
- bass 所有权之和恒为 1；
- 控制曲线连续且没有音量洞；
- MIX END 速度、增益和缓冲区连续；
- 预加载、Seek、相位锁定、BPM 恢复和 CUE-DETR-only 回归继续通过。

## 研究与实现依据

参见 `SOURCES.md` 与 `RESEARCH_NOTES.md`。核心原则：准确 BeatGrid 是 Sync 与循环的前提；循环起止点应量化到拍网格；CUE/phrase 决定结构交接位置；增加铺垫时优先重复鼓组，而不是延长双人声、双旋律和双低频重叠。
