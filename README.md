# Beat This! + MuQ + All-In-One Auto DJ 1.0.0

## 1.0.0 新增：平滑排序、滑动预加载、配置自动保存

### MuQ 自动重排

MuQ 分析完成后会自动重排尚未播放的队列。排序不是只看整曲 embedding，
而是同时计算：

- MuQ 全局风格相似度
- 有方向的 `A Outro → B Intro` 语义兼容度
- 局部语义轨迹 DTW
- BPM/半拍双拍同步难度
- 平均能量、节奏密度、亮度、低频和动态差
- All-In-One 的 `outro→intro`、`break→chorus` 等段落方向
- 单次大跨度 jump penalty
- 整个播放顺序的能量锯齿和加速度惩罚

列表中的“与下一首”列显示最终相邻兼容分。播放期间只修改尚未播放部分；
当前歌曲、已播放歌曲和已经开始混入的下一首不会被打乱。

### 滑动窗口预加载

默认窗口为 3 轨：

```text
当前轨：实时播放
下一轨：完整热预载（切点、BPM、过渡音频全部就绪）
再下一轨：暖预载（解码、特征、BPM 同步时间拉伸提前完成）
```

GUI 可以设置窗口轨数、暖轨内存上限和截止保护时间。若机器处理速度不足并
接近当前歌曲末尾，系统会提升下一轨优先级；仍来不及时会采用短节拍淡化，
避免越过计划 OUT 点后硬切。

### 配置自动保存

所有混音、模型、预加载、输出设备和窗口尺寸配置会自动保存到：

```text
~/.config/beatthis-muq-auto-dj/settings.json
```

Windows 下该路径位于当前用户主目录。也可以通过环境变量
`AUTODJ_SETTINGS_PATH` 指定其他位置。写入使用临时文件替换，避免程序异常退出
留下半个 JSON。

这是一个面向电子音乐和连续播放场景的 Python 桌面自动 DJ 原型：

- **Beat This!**：beat/downbeat 与小节网格
- **MuQ-large-msd-iter**：全局风格、局部段落语义和播放列表排序
- **论文驱动 Cue/Phrase 搜索**：寻找 A 的 OUT、B 的 IN 和自动过渡长度
- **Phrase-Locked HPSS 渲染**：低频、鼓组、旋律/人声采用不同所有权曲线
- **连续 BPM Recovery**：混音结束后按小节平滑恢复下一首原 BPM
- **Tkinter + sounddevice**：GUI、队列和实时播放


## 1.0.0：引入 All-In-One 功能段模型

本版本正式接入 `mir-aidj/all-in-one`，不再把 Raveform-compatible 标签描述成模型输出。

分析职责现在分离为：

```text
Beat This!
  └── beat / downbeat / 小节网格（时间权威）

All-In-One harmonix-all
  └── start / intro / verse / chorus / break / bridge / inst / solo / outro / end

MuQ-large-msd-iter
  └── 风格语义、局部轨迹和有方向的播放列表排序
```

All-In-One 结果会真正参与：

- OUT/IN cue 候选排序
- `OUTRO→INTRO`、`BREAK/BRIDGE→CHORUS/DROP` 等结构角色兼容度
- Long Blend、Echo Out、Drop Swap、Loop Out 等真人手法选择
- GUI 双轨时间轴顶部的功能段彩色条
- 队列表格中的结构摘要
- 播放前预加载和切歌规划

All-In-One 使用 HTDemucs 生成分轨频谱，因此首次分析比 Beat This! 慢。程序会批量处理整个队列并缓存结果，播放和切歌阶段不会临时运行模型。

### 安装 All-In-One

在现有环境中执行：

```bash
conda activate beatthis-auto-dj
python install_allinone.py
python verify_allinone.py
```

完整模型验证：

```bash
python verify_allinone.py demo_tracks/demo_01_120bpm.wav
```

All-In-One 1.1.0 使用旧版 NATTEN 函数名。项目会：

1. 保留已安装且兼容的原生 NATTEN 0.15–0.19；
2. 新版或缺失 NATTEN 时注入纯 PyTorch 兼容实现；
3. 不强制替换当前 PyTorch、Beat This! 和 MuQ 环境。

纯 PyTorch 兼容层速度比原生 CUDA NATTEN 慢，但结构结果会缓存。可设置：

```bash
set AUTODJ_FORCE_TORCH_NATTEN=1       # Windows CMD
$env:AUTODJ_FORCE_TORCH_NATTEN="1"    # PowerShell
```

### GUI 操作

右侧滚动设置区新增：

- 启用 All-In-One 功能段模型
- `harmonix-all` / `harmonix-fold0..7`
- 检测模型
- 分析全部结构

歌曲添加后的默认流水线为：

```text
Beat This! → All-In-One → MuQ 排序 → 当前/下一首预加载 → 播放
```


## 0.7.0：可 Seek、播放前预加载与 MuQ 自动重排

### 点击/拖动时间轴跳转

上方 A 轨波形现在可以直接操作：

1. 在当前歌曲波形上点击，直接跳转到对应时间。
2. 按住鼠标拖动时只显示蓝色预览线。
3. 松开鼠标后才真正执行跳转，避免拖动过程中频繁重建音频。
4. 实际位置会吸附到最近的 Beat This! 节拍。
5. 跳转后自动加入约 30ms 淡入，减少从非零采样点起播的 click。
6. 原来的 OUT/IN 规划会失效，程序会从新的播放位置后台重新搜索切歌点。

为保证行为明确，Seek 操作作用于时间轴上方的当前歌曲 A；下方 B 轨仍用于显示下一首的 IN、MIX END 与 BPM RESTORE。

### 播放前完整预加载当前歌曲和下一首

默认开启：

```text
播放前后台预加载当前歌曲和下一首
```

预加载不只是读取文件，而是提前完成：

```text
MuQ 缓存读取
音频解码与响度归一
结构与 Cue 特征
下一首 BPM 同步
OUT / IN 搜索
Phrase-Lock
BPM Recovery Bridge
HPSS / EQ / Echo 过渡渲染
```

预加载完成后，队列表格会显示“已预载”和“下一首预载”；点击播放时直接复用结果。播放过程中引擎仍会立即准备新的下一首，因此不会等到切换完成后才开始分析。

为了控制内存，播放前只完整保留当前歌曲和下一首，不会把整个大型音乐库全部解码到内存。

### MuQ 完成后自动调整列表和真实播放顺序

默认开启：

```text
MuQ 分析完成后自动重排未播放队列
```

行为规则：

- 未播放时：以当前选中歌曲作为第一首，重排整个队列。
- 播放中：保留已经播放的歌曲和当前歌曲，只重排尚未播放部分。
- 已经进入 A→B 过渡时：锁定正在混入的 B，只重排 B 后面的歌曲。
- 排序完成后同时调用音频引擎更新内部 playlist，不只是改变 GUI 表格顺序。
- 如果新的下一首发生变化，旧的预加载和切歌计划会被撤销，并立即后台准备新的下一首。
- GUI 已完成的 MuQ profile 会直接注入音频引擎，避免为下一首重复运行 MuQ-large。

排序依旧使用有方向的 `A outro → B intro` 图/beam search，而不是简单按文件名、BPM 或全曲余弦相似度排列。

## 0.6.1 重点改进

### MuQ 风格分析与图排序

使用官方模型：

```text
OpenMuQ/MuQ-large-msd-iter
```

程序会提取：

- 全曲风格/语义 embedding
- intro embedding
- outro embedding
- 沿歌曲时间轴的多个局部 embedding
- 基础声学轮廓

队列中的“MuQ 智能排序”使用有方向的 `A outro → B intro` 兼容度和图搜索重新排列歌曲，而不是只比较两个整曲 embedding。

排序综合：

```text
MuQ 全局风格
MuQ outro→intro
MuQ 局部轨迹 DTW
BPM 可同步性
基础声学连续性
```

MuQ 是表示模型，不是校准后的流派分类器，所以 GUI 显示“风格组”而不是伪造具体 genre 名称。

### 更自然的切歌点匹配

候选 OUT/IN 评分新增：

```text
MuQ 风格相似度
MuQ 候选片段相似度
MuQ 局部语义轨迹连续性
Beat This! Cue/Phrase
Camelot 调性
能量连续性
低频冲突
疑似人声冲突
节奏密度
```

MuQ 只占有限权重。公开的 AutoMashup 研究指出，通用音频 embedding 不能单独复现人类感知的 mashup 兼容度，而且兼容度具有方向性。

### 微节拍对齐

即使两个点都被标记为 downbeat，kick 瞬态仍可能相差几十毫秒。程序会在 ±80 ms 范围内分析低频瞬态，并微调 B 的起点，减少双 kick 的 flam 感。

GUI 匹配详情显示：

```text
微对齐 +18.0 ms
```

### Phrase-Locked 过渡

旧版直接取 A/B 过渡片段的较短长度，局部 BPM 波动可能导致 8/16 小节结束位置不一致。

新版：

1. A 的乐句长度定义实际过渡时长。
2. B 的完整候选乐句只做一次高质量局部 time-stretch。
3. B 在过渡结束后从其原始完整小节终点继续。
4. BPM Recovery 从该终点开始。

因此不会在过渡尾部丢掉半拍，或从 B 的小节中间继续播放。


### 打击乐保真的时间拉伸回退

当系统未安装 Rubber Band R3 时，`auto` 不再回退到整轨普通 phase vocoder，而是：

```text
HPSS 软分离
├── harmonic → phase vocoder
└── percussive → stereo WSOLA
            ↓
      对齐长度并重组
```

这不是 SELEBI 的逐公式复现；它采用相同的工程目标——减少传统变速对电子鼓瞬态的涂抹——并提供无需外部二进制的可运行回退。GUI 也可以显式选择 `Hybrid HPSS`。

### HPSS 角色感知混音

复杂过渡会把非低频部分软分解为：

```text
harmonic：旋律、和声和主要人声倾向
percussive：kick 之外的鼓组与瞬态倾向
low：bass/sub/kick 低频所有权
```

不同角色使用不同曲线：

- B drums 较早进入
- A drums 保留到主要 bass swap 附近
- 两首低频在中段交换，不长期叠加
- 疑似双人声时延迟 B harmonic
- 中点自动留出约 1–2 dB 空间
- 过渡预渲染后只在实时回调中读取数组

默认 `Adaptive` 使用 **MuQ Phrase-Locked HPSS**。`Spectral Seam` 保留为实验选项。

## 安装

推荐新环境：

```bash
conda env create -f environment.yml
conda activate beatthis-auto-dj
python install_beat_this.py
python install_muq.py
python verify_install.py
python app.py
```

CUDA 用户可先按 PyTorch 官方方式安装对应版本：

```bash
python install_beat_this.py --skip-torch
python install_muq.py --skip-torch
```

MuQ 模型首次使用时会从 Hugging Face 下载并缓存。它约有 3 亿参数，CPU 分析会比较慢，GPU/MPS 更适合大队列。

## 许可提醒

- MuQ 官方代码：MIT
- `OpenMuQ/MuQ-large-msd-iter` 权重：**CC-BY-NC 4.0**

因此此模型权重不能直接用于未获授权的商业产品。商业落地需要替换成允许商用的模型或取得许可。

## 使用步骤

1. 添加歌曲，等待 Beat This! 分析。
2. 点击 **MuQ 智能排序**。
3. 选择希望作为第一首的歌曲；排序会保留它作为起点。
4. 推荐设置：

```text
过渡长度：自动
混音风格：Club
效果强度：65%–75%
BPM 回归：自动或 16 小节
AutoMix：AutoMix-like
过渡引擎：Adaptive
时间拉伸：auto（优先 Rubber Band R3，否则 Hybrid HPSS + WSOLA）
MuQ：开启
```

5. 播放后 GUI 会标注 OUT、IN、MIX END、SWAP 和 BPM RESTORE。

## 文件结构

```text
app.py                         GUI、MuQ 排序按钮和风格组

autodj/beat_this_analyzer.py  beat/downbeat 分析
autodj/muq_analyzer.py        MuQ 推理、窗口池化和压缩缓存
autodj/playlist_ranker.py     有方向的兼容度与图/beam 排序
autodj/transition_matcher.py  Cue/Phrase/MuQ 候选评分和微对齐
autodj/audio_engine.py        HPSS 角色混音、phrase warp、实时播放
autodj/time_stretch.py        Rubber Band 与连续 tempo map
autodj/edm_structure.py       EDM 结构和 cue 特征
autodj/spectral_seam.py       实验性频时域 graph-cut
autodj/timeline.py            GUI 时间轴
```

## 当前边界

- 当前 HPSS 是软角色分离，不等同于 Demucs/BS-RoFormer 真正 stems。
- 任意两首速度、调性、主唱和编曲都完全不同的歌曲，无法保证绝对无感。
- MuQ embedding 用于风格和语义连续性，但不应替代调性、节拍、结构与 stem 冲突分析。
- 最好的电子音乐效果仍建议安装 Rubber Band R3；未安装时会自动使用 Hybrid HPSS + WSOLA，分别处理谐波和打击乐，减少 kick/hi-hat 涂抹。

## Rubber Band 已安装但检测不到

0.6.2 起程序按以下顺序查找 CLI：

1. GUI 中手动选择的路径
2. `AUTODJ_RUBBERBAND`
3. `RUBBERBAND_EXE`
4. `RUBBERBAND_PATH`
5. 当前 Python 进程的 `PATH`
6. 当前 Conda 环境的 `Scripts`、`Library/bin` 和 `bin`
7. Windows 常见安装目录

先在**激活项目环境后的同一个终端**运行：

```bash
python check_quality_backend.py
```

Windows 还可以检查：

```bat
where rubberband
rubberband -h
python -c "import shutil; print(shutil.which('rubberband'))"
```

注意：系统 `PATH` 中应该加入 `rubberband.exe` 所在的**目录**，而不是把
`rubberband.exe` 文件路径本身作为一条 PATH 目录。修改系统环境变量后，需要
关闭并重新打开 Anaconda Prompt、PowerShell、CMD 和 GUI 程序。

最稳定的显式配置方式：

```bat
set AUTODJ_RUBBERBAND=C:\tools\rubberband\rubberband.exe
python check_quality_backend.py
python app.py
```

PowerShell：

```powershell
$env:AUTODJ_RUBBERBAND = "C:\tools\rubberband\rubberband.exe"
python check_quality_backend.py
python app.py
```

也可以在 GUI 的“Rubber Band CLI 路径”处直接选择文件。