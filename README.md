# Beat This! + CUE-DETR + MuQ + All-In-One Auto DJ 1.2.10

这是一个面向本地音乐文件的自动 DJ 实验程序。1.2.9 将切歌候选改为只来自官方 CUE-DETR；1.2.10 修复 CUE-DETR 与 All-In-One 的 NATTEN 兼容层在同一进程中的导入冲突。

## 模型分工

- **CUE-DETR (`disco-eth/cue-detr`)**：唯一的 IN/OUT cue 候选来源。
- **Beat This!**：beat/downbeat 网格，并将 CUE-DETR 的时间预测量化到最近 downbeat。
- **All-In-One**：`intro / verse / chorus / break / bridge / outro` 等结构角色，只给 neural cue 排序，不新增 cue。
- **MuQ**：歌曲风格、Outro→Intro 语义关系、播放顺序和 cue 配对兼容度。
- **本地 DSP 特征**：调性、能量、低频、人声和鼓组冲突评分，只影响候选分数和渲染方式。

## 1.2.10 的主要变化

### CUE-DETR 与 All-In-One 共存修复

- 为纯 PyTorch `natten` 和 `natten.functional` 兼容模块补充合法 `ModuleSpec`。
- Transformers 的可选依赖检查不再报 `natten.__spec__ is None`。
- CUE-DETR 检测会直接导入 `DetrImageProcessor` 与 `DetrForObjectDetection`。
- GUI 会区分缺少依赖与导入冲突。
- 安装器和验证器会模拟真实的 All-In-One → CUE-DETR 导入顺序。

### CUE-DETR-only 切点（1.2.9 起）

旧版的以下切点来源已经停用：

- checkerboard novelty 自动边界
- 固定 4/8/16 小节周期候选
- 能量阈值自动造点
- 静音尾部重新选择切点
- 截止保护临时移动到任意位置

新的流程：

```text
整曲音频
  ↓
22.05 kHz Mel 频谱
  ↓
CUE-DETR 75% 重叠滑窗预测
  ↓
合并重叠窗口候选并做峰值筛选
  ↓
吸附到 Beat This! downbeat
  ↓
All-In-One / MuQ / BPM / 调性只做排序
```

即使 AutoMix 降级为短淡化，也只能缩短已经选中的 CUE-DETR cue 对，不能移动到新的规则位置。

### Mixxx 风格的三级预读

本项目没有复制 Mixxx 源码，而是采用其 `CachingReader / ReadAheadManager / worker scheduler` 所体现的实时播放原则：耗时工作不能等到音频回调或 OUT 点附近才开始。

滑动窗口现在分为：

1. **分析窗口**：提前完成 Beat This!、All-In-One、CUE-DETR 和 MuQ 缓存。
2. **计划窗口**：提前解码、BPM 同步并搜索相邻歌曲的 CUE-DETR cue 对。
3. **渲染窗口**：提前生成 Phrase-Lock、Deck Phase Lock、BPM 恢复桥和最终过渡音频。

播放启动前，当前歌曲与紧邻下一首的完整过渡必须已经就绪。窗口后方的相邻歌曲对继续后台渲染，并受内存上限控制。

日志示例：

```text
CUE-DETR 完成：Track B.flac · 7 个 cue
滑动窗口提前渲染 2/8：Track A → Track B
滑动窗口完整过渡就绪：Track A → Track B · CUE-DETR 184.0s
开始播放：Track A
滑动窗口命中完整过渡：Track A → Track B
```

## 安装

建议解压到新目录：

```powershell
conda activate beatthis-auto-dj
cd beatthis_muq_cuedetr_auto_dj_gui_v1210

python install_cuedetr.py
python verify_cuedetr.py
python app.py
```

`install_cuedetr.py` 不会重装 PyTorch，避免覆盖已有 CUDA 构建。首次分析会下载：

- `disco-eth/cue-detr`
- `facebook/detr-resnet-50` 图像处理配置

## 推荐设置

```text
CUE-DETR 设备：cuda 或 auto
灵敏度：0.88–0.93
最小 cue 间隔：8 或 16 小节
滑动窗口：3–4 轨
预加载内存：1024–2048 MB
截止保护：60–90 秒
Beat This! / MuQ：auto 或 cuda
All-In-One：cpu
Rubber Band：R3
```

灵敏度过高会减少候选，过低会产生过多候选。电子音乐建议先从 `0.90 / 8 小节` 开始。

## 使用流程

1. 添加歌曲。
2. 等待 Beat This!、All-In-One、CUE-DETR 和 MuQ 完成。
3. MuQ 完成后自动调整尚未播放的顺序。
4. 等待状态显示“热下一轨预加载完成”。
5. 点击播放。

如果任何待播放歌曲没有 CUE-DETR profile，程序会阻止播放并启动 cue 分析，不会静默使用旧算法。

## 缓存

CUE-DETR 缓存目录：

```text
~/.cache/beatthis_auto_dj/cue_detr
```

缓存键包含文件大小、修改时间、模型、灵敏度、最小间隔和 Beat This! BPM 网格信息。

## 验证

- 56 项自动测试通过
- CUE-DETR downbeat 吸附测试通过
- 最小小节距离测试通过
- 旧 cue 数组清零测试通过
- 匹配器只使用 CUE-DETR cue 测试通过
- Gapless/截止降级不移动 neural cue 测试通过
- 完整相邻歌曲过渡滑动缓存测试通过
- GUI 无头启动测试通过

## CUE-DETR 显示 `natten.__spec__ is None`

这是 1.2.9 中 All-In-One 的 NATTEN 兼容模块与 Transformers 的导入检测冲突，
不是 CUE-DETR 依赖没有安装。1.2.10 已修复，已有环境无需重装：

```powershell
python verify_cuedetr.py
python app.py
```

