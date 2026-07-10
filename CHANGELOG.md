# Changelog

## 1.2.12

- 删除上一首后 42%/最后 48 小节、下一首先 35%/前 32 小节的自动切点硬约束。
- 删除 `tail_to_head` 位置加分；歌曲位置仅保留为诊断指标。
- 所有有效 CUE-DETR cue 统一按乐句、边界、低频、人声、能量和相位质量评分。
- 自动渲染只保留 Long Blend、Bass Swap 和必要的 Echo Out。
- 删除 Drop Swap、Double Drop、Loop Out、Filter Ride、Post-Drop Relay、Breakdown Lift 渲染代码和 GUI 选项。
- 删除 Double Drop 的额外两拍尾部、落点偏好和特殊恢复点逻辑。
- 低频交接改为互补所有权，`bass_A + bass_B = 1`，避免双 bass 增益和相位碰撞。
- 下一首鼓组先进入，上一首鼓组后释放；和声/人声按 vocal risk 缩短重叠。
- Echo 只在高人声风险或低调性兼容时进入自动候选，并降低 feedback。
- 删除真人变化度、最近手法惩罚和候选多样性奖励，结果完全确定性。
- 新增 `check_natural_transition.py` 独立检查脚本。
- 新增取消位置门槛、低频所有权、控制曲线、Echo 门控和移除旧手法等测试；共 71 项测试通过。

## 1.2.11

- 将 CUE-DETR 切点方向改为硬约束：上一首后 42%/最后 48 小节 OUT，下一首前 35%/前 32 小节 IN。
- 禁止下一首后半段高分 cue 成为入口，避免“后段接后段”。
- 自动过渡最短改为 8 小节，手动 4 小节设置仍可使用。
- Drop Swap 和 Double Drop 在过渡前段加入低音量 incoming 鼓组/和声床。
- 滑动窗口改为分级调度：热 A→B 完整渲染，未来 B→C 只做解码、同步和 cue 规划。
- 未来窗口只规划最近一个 pair，不再完整渲染整个队列。
- 点击播放等待并接管正在运行的热 pair，消除重复 BPM 恢复和相位锁定。
- 新增播放启动互斥状态，阻止 GUI 定时器在启动期间再次创建预加载任务。
- 窗口跨代时保留已完成的同步音轨，避免在歌曲切换边界丢弃昂贵结果。
- 新增尾部→头部、早期 incoming、分级窗口和启动互斥测试；共 61 项测试通过。

## 1.2.10

- 接入官方 `disco-eth/cue-detr`，并将其设为唯一 IN/OUT cue 来源。
- 移除旧 checkerboard novelty、周期、salience 和静音尾部规则切点。
- Beat This! 只负责把 neural cue 吸附到 downbeat。
- All-In-One 和 MuQ 只对 CUE-DETR cue 排序，不允许新增候选。
- CUE-DETR 使用官方 75% 重叠 Mel 频谱滑窗流程，并增加批量 GPU 推理与磁盘缓存。
- 滑动窗口升级为完整相邻 pair 预渲染：解码、变速、cue 配对、BPM 恢复和最终过渡全部提前完成。
- 播放流启动前强制准备好当前→下一首的最终过渡。
- 简单淡化、Gapless 和截止保护不再移动已选中的 neural cue。
- 新增 CUE-DETR 安装、验证、GUI 进度和 4 项专项回归测试。

## 1.2.7

- 修复 Spectral Seam 渲染分支未定义 `phrase_length` 导致的 NameError。
- 谱缝合、echo 缓冲区与 phrase warp 统一使用不可变的锁定乐句长度。
- 新增谱缝合不回退回归测试。

## 1.2.6

- 新增逐拍鼓组微对齐：Phrase-Lock 后按每个 beat 检测 kick/percussion 瞬态，并用端点固定的连续小幅 time-map 校正下一首。
- 同时微调下一首低频 kick 与打击乐层，保持完整乐句终点不变，切歌后无需重新计算或跳拍。
- Double Drop 结构兼容阈值适度放宽，并在候选评分接近时优先选择 Double Drop。
- Double Drop 增加最多 2 拍的落点后交接窗口：两边鼓组先重叠约 3/4 拍，再平滑降低上一首鼓组。
- Double Drop 低频仍在落点附近快速交换，避免两条 bass 长时间重叠。
- 普通过渡也改为下一首鼓组先建立 groove，再开始衰减上一首。
- GUI 新增首拍微调、逐拍校正拍数与双鼓交接评分。
- 新增逐拍校正、双鼓重叠后释放及 Double Drop 优先回归测试；共 47 项测试通过。

## 1.2.5

- 修复播放前 pair preload 与播放中 next loader 同时处理同一歌曲的竞争条件。
- 相同文件、目标 BPM 和 DSP 配置的音轨准备任务现在全局单实例执行，其他调用等待并复用结果。
- 点击播放会复用仍在进行的当前轨准备，不再取消后重新解码。
- 取消的播放前预加载会在当前轨完成后立即停止，不再继续处理下一轨。
- 重复 GUI 预加载调度不再重复输出启动日志。
- 新增“基础准备、切点搜索、BPM 恢复、最终渲染、提交播放器”阶段日志。
- 节拍 Seek 不再重新规划切歌，不再调用模型、Rubber Band 或过渡渲染。
- Seek 进入已渲染过渡区时按对应 offset 继续；越过原切点时跳过本次计划。
- 新增并发准备、预加载取消和无重算 Seek 回归测试；共 44 项测试通过。

## 1.2.2

- GUI 底部新增歌曲分析进度条、阶段名称、当前歌曲、完成数量和百分比。
- 模型下载/加载阶段使用动态进度条，避免长时间停在 0% 造成卡死误判。
- MuQ 进度细化到语义窗口，缓存命中也会正确推进。
- SongFormer worker 改为实时流式输出，不再使用 `capture_output` 等待整批结束。
- 命令行日志新增时间、级别和线程名称，并透传 SongFormer 下载、加载、推理及错误日志。
- 新增 worker 结构化进度消息解析回归测试。

## 1.2.1

- 修复启动时仍调用旧 `set_allin1_model()` 校验导致 `ASLP-lab/SongFormer` 被拒绝的问题。
- 引擎配置和 GUI 全部改用 `songformer_*` 接口。
- 删除残留的 All-In-One 模型校验、变量名和运行时分支。
- SongFormer 官方模型名称现在进行明确校验，并兼容旧缓存字段读取。

## 1.1.0 — Structure-Aware Drop/Phrase DJ Policy

- 新增 `dj_phrase_policy.py`，把切歌决策从静态相似度改为结构意图规划
- 新增 Post-Drop Relay、Breakdown Lift、Double Drop 三种渲染手法
- 优先在 Drop 结束后的 cooldown/breakdown/outro 边界退出
- 反向规划 B 的 intro/buildup，使混音终点精确落在 Drop/Chorus
- Drop 中段长混音加入硬惩罚，仅保留短 Drop Swap 和高兼容 Double Drop
- 匹配分新增角色路径、Drop 落点、Drop 后退出、能量弧线、边界和 guard
- AutoMix 复杂过渡置信度纳入结构策略评分
- GUI 显示 DJ 意图、角色路径和 DROP LAND 时间轴标记
- 新增结构策略与三种新手法测试；项目共 35 项测试通过

## 0.9.0 — All-In-One Functional Structure Integration

- Beat This! 保持 beat/downbeat 时间权威，All-In-One 提供功能段标签
- 新增 intro、verse、chorus、break、bridge、inst、solo、outro 融合
- All-In-One 边界和角色兼容度进入 OUT/IN 候选评分
- 真人化过渡根据功能段角色调整 Long Blend、Echo、Drop Swap 和 Loop Out
- GUI 时间轴显示彩色功能段，队列表格显示结构摘要
- 播放前批量分析并缓存，不在切歌时启动 Demucs
- 新增原生 NATTEN 兼容检测和纯 PyTorch legacy-op 回退
- 新增 `install_songformer.py` 与 `verify_songformer.py`

## 0.7.0 — Seek + Pair Preload + Live MuQ Reordering

- 双轨时间轴上方 A 轨支持点击和拖动 Seek
- Seek 吸附最近 beat，并使用 30ms 淡入抑制 click
- Seek 后取消旧 transition，按新位置后台重算 OUT/IN
- 播放前后台完整预加载当前歌曲和下一首
- 预加载包含结构特征、BPM 同步、Phrase-Lock、恢复桥和过渡渲染
- 队列表格显示“预加载中 / 已预载 / 下一首预载”
- MuQ 分析完成后默认自动重排队列
- 播放中只重排未播放后缀；正在过渡时锁定 incoming track
- GUI 排序结果同步到音频引擎真实播放列表
- MuQ profile 注入音频引擎，避免下一首重复模型推理
- 修复旧 next loader 被队列更新失效后可能覆盖 loading 状态的竞争条件
- 新增 Seek、队列热更新和 pair preload 测试；共 22 项测试通过

## 0.6.1 — MuQ Ranking + Natural Tempo/Transient Update

- MuQ 多窗口全局、intro、outro 与局部语义轨迹分析
- 有方向的 `A outro → B intro` 图/beam 播放列表排序
- Phrase-Locked：B 完整乐句精确映射到 A 乐句时长
- ±80 ms 低频瞬态微对齐，减少双 kick flam
- 混音结束从 B 完整小节终点继续，再进入连续 BPM Recovery
- 无 Rubber Band 时改用 Hybrid HPSS + stereo WSOLA
- GUI 新增 `Hybrid HPSS` 时间拉伸后端
- 共 17 项自动测试和完整过渡冒烟测试通过

## 0.6 — Beat This! + MuQ Natural Transition

- 接入 `OpenMuQ/MuQ-large-msd-iter`
- 多窗口、多层隐藏状态聚合和本地 float16 压缩缓存
- 新增有方向的 MuQ `outro → intro` 播放列表兼容度
- 新增图/beam search 智能排序按钮
- GUI 新增 MuQ 风格组和匹配子评分
- 切点评分新增全局风格、局部片段和语义轨迹
- 新增 ±80 ms 低频瞬态微对齐
- B 候选乐句精确拉伸到 A 的乐句时长
- 过渡完成后从 B 完整乐句终点继续，消除尾部半拍偏移
- Adaptive 渲染升级为 Phrase-Locked HPSS
- 低频、鼓组、和声/人声采用不同所有权曲线
- 显式 Spectral Seam 仍可选，Adaptive 不再默认 graph-cut
- 延迟 HPSS 渲染到 BPM Recovery 完成后，避免重复计算
- 新增 MuQ 排序和微对齐测试

## 0.5

- Beat This! 替代 BeatNet
- 连续 BPM time-map 恢复桥
