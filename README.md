# Beat This! + MuQ + All-In-One Auto DJ 1.2.4

本版本把结构分析从 SongFormer 恢复为 `mir-aidj/all-in-one`，并按要求取消独立 worker。All-In-One 直接在主程序进程的后台线程中运行。

## 模型分工

- Beat This!：beat、downbeat 和播放时间网格
- All-In-One：intro、verse、chorus、break、bridge、solo、outro 等功能段
- MuQ-large-msd-iter：风格语义、播放顺序和 Outro→Intro 兼容度
- 本地 EDM 融合：将结构标签、能量、鼓组、低频和新颖度融合为 BUILDUP、DROP、BREAKDOWN 等 DJ 角色

## 资源策略

All-In-One 有独立的运行设备设置，默认是 `cpu`。这不会改变 Beat This! 和 MuQ 的设备选择；它们仍可使用 `auto` 或 `cuda`。

分析顺序为：

```text
Beat This!
  → All-In-One 主进程后台线程
  → MuQ 风格分析与重排
  → 滑动窗口预加载
```

分析期间会暂停 MuQ 排序和音频预加载，避免 Demucs、结构模型、MuQ 和时间拉伸同时争用内存。All-In-One 调用被全局锁串行化，不会同时运行多个结构分析。

## 安装

在原来的主环境中安装，不创建独立 Conda worker：

```powershell
conda activate beatthis-auto-dj
python install_allinone.py
python verify_allinone.py
python app.py
```

进行一次真实分析验证：

```powershell
python verify_allinone.py demo_tracks/demo_01_120bpm.wav --device cpu
```

## GUI 设置

建议 RTX 5070 用户使用：

```text
Beat This! / MuQ 计算设备：auto 或 cuda
All-In-One 运行设备：cpu
All-In-One 模型：harmonix-all
```

All-In-One 可选 `harmonix-all` 或 `harmonix-fold0` 至 `harmonix-fold7`。`harmonix-all` 精度优先，fold 模型速度和内存略低。

## 进度与日志

GUI 底部继续显示分析阶段、歌曲数量和百分比。All-In-One 加载 Demucs 与结构模型时使用动态进度条；完成批量推理后逐首更新结果。

PowerShell 会实时显示主进程线程日志以及 All-In-One 自带的 Demucs、频谱提取和推理进度：

```text
All-In-One 主进程后台线程分析 6 首 · harmonix-all · cpu
=> Found 6 tracks to analyze.
Separating tracks: ...
Extracting spectrograms: ...
Analyzing Track01.flac: ...
All-In-One 完成：Track01.flac · intro, verse, chorus, outro
```

## 缓存

All-In-One 缓存：

```text
~/.beatthis_muq_allinone_cache_v3.json
```

缓存键包含文件路径、大小、修改时间、模型名称和分析器版本。

## 运行

```powershell
conda activate beatthis-auto-dj
python app.py
```
