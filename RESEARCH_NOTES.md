# Research Notes 0.6.1

## 采用的公开研究方向

### MuQ (2025)

MuQ 使用 Mel Residual Vector Quantization 进行自监督音乐表示学习。项目使用官方 `MuQ-large-msd-iter` 的最后四层隐藏状态平均，再做时间池化，用于全局风格和局部段落表示。

### AutoMashup (2025)

AutoMashup 的结果强调：兼容度具有方向性，而且 CLAP/MERT 等通用 embedding 不能单独代替感知兼容模型。因此本项目：

- 区分 `A outro → B intro`
- MuQ 只占组合评分的一部分
- 保留 beat、phrase、key、energy、bass 和 vocal 冲突项

### DJ-AI / graph playlist ordering (2025)

播放列表被建模成有方向的图。边权为 MuQ 段落兼容度、tempo 和声学连续性，采用 beam search 求路径，并保留用户选择的起始歌曲。

### SELEBI (2026) 与 transient-aware TSM

SELEBI 指出传统 phase vocoder 对打击乐会产生 smearing。项目当前优先使用 Rubber Band R3，并在过渡层将低频、percussive 和 harmonic 分开处理。这里不是 SELEBI 的精确复现，因为截至开发时未找到成熟的官方实现。

## 为什么 Adaptive 改用 HPSS

Graph-cut 能为不同频率寻找不同切换时刻，但在密集电子鼓、强 sidechain 和宽频瞬态中，频率相关 seam 也可能让 transient 所有权不稳定。新默认策略采用更稳定的三角色所有权：

- low：一次 bass swap
- percussive：鼓组较早交接
- harmonic：根据人声风险延迟交接

Graph-cut 仍作为显式实验模式。

## 0.6.1 新增实现

### MuQ 多层与多窗口聚合

官方 MuQ 支持输出全部隐藏层。本项目对最后四层求平均，再对时间维做池化；全曲使用多个窗口，分别保留 intro、outro 与局部时间轨迹。该实现受 2026 年 MuQ-token 多层特征聚合研究启发，但没有复制其面向推荐系统的离散 token 训练流程。

### 有方向的语义排序

AutoMashup 指出 mashup 兼容度取决于 stem/角色方向，且通用 embedding 不能单独代替感知兼容模型。因此 MuQ 只作为组合评分的一部分；排序边权同时包含 `outro→intro`、局部 DTW、BPM 与声学连续性。

### Hybrid HPSS + WSOLA

SELEBI 研究聚焦传统 phase vocoder 的 percussion smearing。本项目尚未实现其非平稳 Gabor 变换，而是提供可运行的工程回退：谐波使用相位声码器、打击乐使用立体声 WSOLA。Rubber Band R3 仍是优先后端。

## 0.9.0 All-In-One functional structure model

The application now invokes the official `allin1.analyze()` API with the
`harmonix-all` ensemble. Beat This! remains the single metrical clock. All-In-One
segments are sampled at the original-audio downbeats and mapped to the prepared
playback timeline, including tempo-sync and tempo-recovery warps.

Original Harmonix labels are retained (`intro`, `verse`, `chorus`, `break`,
`bridge`, `inst`, `solo`, `outro`). A separate DJ-role fusion maps those labels
with local energy/percussion evidence, e.g. a high-energy chorus can become
`DROP`, while bridge/break can become `BUILDUP` or `BREAKDOWN`.

The project does not claim that All-In-One is a Raveform model. Raveform remains
a taxonomy/dataset reference; All-In-One is the actual structure inference
backend in 0.9.0.
