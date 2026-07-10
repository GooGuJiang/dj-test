# Changelog

## 1.2.3

- 修复 SongFormer worker 在 RTX 5070 上加载到 CPU 的问题。
- GUI 默认计算设备由 `cpu` 改为 `auto`，并迁移 1.2.2 的旧默认设置。
- SongFormer 独立环境改用 PyTorch 2.7.1 + CUDA 12.8 wheel。
- 新增真实 CUDA 张量 smoke test、GPU 名称、compute capability、CUDA build 和驱动诊断。
- 显式 `cuda` 与 NVIDIA 环境下的 `auto` 不再静默回退 CPU。
- 新增 `repair_songformer_cuda.py` 原地修复脚本。
- PyTorch 从 SongFormer 普通依赖文件中分离，避免后续 pip 安装覆盖 CUDA wheel。
- 新增 RTX 50 系列设备选择与禁止 CPU 回退测试。

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
