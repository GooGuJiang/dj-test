# Changelog

## 1.0.0 — MuQ Smooth Ordering + Sliding Preload + Auto-Save

- MuQ 排序升级为平均兼容度与最差相邻边双目标
- 排序新增 Outro→Intro、局部语义 DTW、BPM、能量、音色和 All-In-One 角色评分
- 对单次大幅风格、BPM、能量跨度加入显式 jump penalty
- Beam search 新增能量曲线加速度与上下锯齿惩罚
- GUI 列表显示每首歌曲与下一首的相邻兼容分
- 播放中只重排未播放后缀，已播放、当前轨和正在混入的轨道保持锁定
- 新增当前轨 + 热下一轨 + 暖未来轨的滑动窗口预加载
- 暖轨提前完成解码、模型特征读取和 BPM 同步时间拉伸
- 新增预加载内存上限、窗口轨数和截止保护时间设置
- 接近截止点仍未完成时自动使用短节拍淡化，避免错过切歌点
- 所有 GUI 参数和窗口尺寸自动保存为原子 JSON，并在下次启动恢复
- 新增 3 项排序、配置和滑动窗口测试；项目共 32 项测试通过

## 0.9.0 — All-In-One Functional Structure Integration

- 接入 `mir-aidj/all-in-one` 1.1.0 / `harmonix-all`
- Beat This! 保持 beat/downbeat 时间权威，All-In-One 提供功能段标签
- 新增 intro、verse、chorus、break、bridge、inst、solo、outro 融合
- All-In-One 边界和角色兼容度进入 OUT/IN 候选评分
- 真人化过渡根据功能段角色调整 Long Blend、Echo、Drop Swap 和 Loop Out
- GUI 时间轴显示彩色功能段，队列表格显示结构摘要
- 播放前批量分析并缓存，不在切歌时启动 Demucs
- 新增原生 NATTEN 兼容检测和纯 PyTorch legacy-op 回退
- 新增 `install_allinone.py` 与 `verify_allinone.py`
- 新增 4 项 All-In-One/NATTEN 测试；项目共 29 项测试通过

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
