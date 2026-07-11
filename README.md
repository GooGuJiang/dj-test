# Beat This! + CUE-DETR + MuQ + All-In-One Auto DJ 1.2.19

这是一个面向本地音乐文件的自动 DJ 实验程序。1.2.19 修复了导入歌曲完成分析后队列可能不重排的问题，并保留 1.2.18 的 **研究驱动自适应淡化** 与 **上下文序列搜索**：尾端切歌会按实际低/中/高频相关性修正交叉增益，避免对齐 kick 或相似素材在中点突然变响；播放列表则对小队列使用精确子集动态规划，对大队列使用两步前瞻、端点多样性 beam search，并修正能量轨迹方向。

## 模型分工

- **CUE-DETR (`disco-eth/cue-detr`)**：唯一的 IN/OUT cue 候选来源。
- **Beat This!**：beat/downbeat 网格，并把 CUE-DETR 预测吸附到真实 downbeat。
- **All-In-One**：intro、verse、chorus、break、outro 等结构角色，只参与排序。
- **MuQ**：播放顺序、风格连续性和 cue 对兼容度。
- **本地 DSP**：BPM 同步、逐拍相位锁定、三段分频、鼓循环、低频所有权和最终转场渲染。


## 1.2.19 的核心变化

- 导入或重新分析歌曲时创建“必达”的自动排序请求；如果请求发生在 Beat This!、All-In-One 或 CUE-DETR 仍运行期间，会排队到阶段结束，不再静默丢失。
- MuQ 未启用或自动加载失败时，自动降级为 BPM、All-In-One 段落结构和已有 MuQ 缓存驱动的图搜索排序，列表不再停留在文件选择顺序。
- 排序完成后保留当前选中歌曲；播放中保留已播放历史、当前歌曲，以及已经开始转场时的下一首。
- 状态栏明确显示“队列顺序已更新”或“当前顺序已是最优，无需移动”，便于区分没有执行和无需改变。

## 1.2.18 的核心变化

### 1. 相关性感知的尾端/简化转场

- 在规划阶段分别测量低频、中频和高频重叠区的归一化相关性。
- 对正相关素材使用功率交叉项补偿：保留原始 fade 形状和首尾端点，只削减中点额外能量。
- 完全相关的等功率交叉最多约削减 3 dB；负相关不做自动增益，避免不可靠的抵消补偿导致削波。
- 诊断指标新增 `fade_correlation_low/mid/high` 与 `correlation_compensation_db`。

### 2. 尾端 IN cue 独立可播放评分

- 尾端安全路径不再要求下一首 cue 前必须存在 4 拍 pre-roll，因此位于样本 0/第一拍的有效 IN cue 不会被错误过滤。
- 保持 CUE-DETR 为唯一 cue 来源，但在多个 neural IN cue 之间加入后续空间、局部 RMS、onset、vocal 风险和 cue 置信度评分。
- 高置信度但位于安静尾部的 IN cue 会输给后续内容充足、可以独立承接播放的入口。

### 3. 播放列表搜索升级

- 10 首及以下使用精确子集动态规划，状态包含已用集合、最后两首、累计分与最弱边 Pareto 前沿。
- 大队列 beam search 从一层前瞻升级为两步连通性前瞻，并预留部分槽位给不同末端节点，减少近重复路径挤占搜索预算。
- 修复旧能量项：旧代码错误惩罚“升后降/降后升”；现在轻度奖励受控反转，惩罚连续大幅同向移动，同时限制过强锯齿。
- 仍保留确定性 swap/relocate 局部改良，首曲固定不变。

### 4. 验证

- 完整 pytest：94 项通过。
- 新增完全相关素材中点功率补偿、能量反转、精确小队列搜索和尾端 IN cue 可播放性测试。
- 自然转场、MIX END 连续性和量化鼓循环三个独立检查脚本全部通过。

## 1.2.17 的核心变化

### 1. 末端 CUE 不再导致断歌

- 当前播放位置已经越过最后一个 CUE-DETR OUT 时，保留该 CUE 作为语义上下文，但把实际渲染窗口移动到歌曲物理尾部。
- CUE 后不足 1/4 拍时，不再进入需要 post-cue release 的鼓循环/HPSS 复杂渲染器。
- 改用最长约 2 拍的 equal-power `Tail Crossfade`：A 在真实文件末尾降到 0，B 在同一点回到 unity。
- 如果下一首的 IN cue 本身错误地落在末端，则只在端点安全路径中回退到第一个有效 downbeat/样本 0，保证播放器继续。
- 即使 GUI 选择 `Always DJ`，端点安全仍优先，避免为坚持复杂效果而断歌。

### 2. 播放列表图搜索继续优化

- 邻接边按方向兼容度预排序，搜索状态改用 bit-mask，降低集合复制开销。
- 加入一层未来连通性前瞻，避免当前边很高但下一步无路可走的“死端”节点。
- 对相同 `已用集合 + 前一首 + 当前首` 做 Pareto 状态去重，同时保留累计分更高与最弱边更强的方案。
- Beam 完成后执行确定性 swap/relocate 局部改良，修复内部单条弱边；首曲固定不变。
- 评分仍保持方向性 MuQ、BPM、能量、结构与最弱边惩罚，不引入随机结果。

### 3. 1.2.16 鼓循环桥继续保留

#### 3.1 CUE-DETR 仍是主交接点

所有有效 CUE-DETR cue 都可以参加配对，不存在以下旧版硬约束：

```text
A OUT：后 42% / 最后 48 小节
B IN ：前 35% / 前 32 小节
```

歌曲位置只写入诊断指标，权重为 0。候选根据 downbeat、乐句、CUE-DETR 置信度、鼓组相位、低频/人声碰撞、能量、调性、结构角色和 MuQ 兼容度排序。

#### 3.2 默认使用“4 拍鼓循环桥 + 2 拍释放”

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

#### 3.3 鼓循环实现

- 循环素材来自下一首 **CUE 当拍之后的 2 拍 percussion stem**。
- 预入区有 4 拍，因此该 2 拍 loop 重复两次。
- 循环点严格按 BeatGrid 整拍对齐，不改变 CUE-DETR 的位置。
- 循环边界与 loop→live B 接点使用约 4 ms 去点击桥。
- 不循环 bass，也不循环 harmonic/vocal，避免双低频和双人声。
- 如果 cue 太靠近边界或没有足够音频，只缩短可用窗口，不创造新 cue。

#### 3.4 自动手法保持精简

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

#### 3.5 MIX END 连续性

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
cd dj-test-main
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
90 passed
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
