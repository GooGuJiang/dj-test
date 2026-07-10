# Beat This! Research Auto DJ 0.5

这是一个使用 **Beat This! + 论文 cue/phrase 匹配 + 连续变速时间映射 + Tkinter + sounddevice** 实现的桌面自动 DJ 原型。

## 0.5 版本核心升级

### Beat This! 替代 BeatNet 离线分析

- 使用 `beat_this.inference.File2Beats`
- 默认使用 `final0` 高精度模型
- 可选择 `small0` 快速模型
- 使用 Beat This! 1.1.0 的 `infer_beat_numbers()` 生成 beat number
- 不再依赖 madmom、PyAudio 或 BeatNet
- 针对稳定电子节奏保守修复偶发漏拍，但不会对速度变化明显的歌曲强制规则化
- 新缓存与旧 BeatNet 缓存隔离

### 连续 BPM 恢复桥

旧版本将恢复区拆成多个小节分别拉伸，再用短 crossfade 拼接，电子音乐容易在每个小节边界出现：

- kick 瞬态变软
- 低频相位变化
- hi-hat 抖动
- BPM 分段感

新版使用一条连续时间映射：

```text
B 进入混音时的同步 BPM
          ↓
保持同步到 MIX END
          ↓
五次缓动 + 几何 BPM 插值
          ↓
恢复到 B 原始 BPM
          ↓
原速尾段
```

整个过程一次性渲染，不再逐小节重新启动 time-stretch。

- 安装 Rubber Band 时：使用 R3 `--timemap`
- 未安装时：使用单次 STFT 连续可变 phase-vocoder
- 恢复开始和结束的一阶、二阶速度变化均趋近于 0
- 恢复完成后自动重新渲染过渡终点，避免 MIX END 与下一轨首样本不一致
- GUI BPM 显示使用与渲染器一致的几何缓动

## 安装

推荐全新环境：

```bash
conda env create -f environment.yml
conda activate beatthis-auto-dj
python install_beat_this.py
python verify_install.py
python app.py
```

### CUDA 用户

先按 PyTorch 官方方式安装与你 CUDA 对应的 `torch` 和 `torchaudio`，然后：

```bash
python install_beat_this.py --skip-torch
```

### 旧环境

旧的 `beatnet-auto-dj` 环境使用 Python 3.9 和较旧 NumPy。建议新建 `beatthis-auto-dj`，不要直接覆盖旧环境。

## Rubber Band R3

高质量连续变速建议安装 Rubber Band 4.x。程序会自动寻找：

```text
rubberband-r3
rubberband
```

检查：

```bash
python check_quality_backend.py
```

没有 Rubber Band 时程序仍可运行，但电子鼓和复杂混音建议使用 R3。

## GUI 操作

1. 点击“添加歌曲”。
2. 等待 Beat This! 完成 beat/downbeat 分析。
3. 选择起始歌曲并播放。
4. 过渡长度选择“自动”。
5. BPM 回归建议选择 8 或 16 小节。
6. 时间拉伸质量选择 `auto` 或 `Rubber Band R3`。
7. 时间轴会显示 `OUT / IN / MIX END / BPM RESTORE`。

## 推荐电子音乐设置

```text
Beat This! 模型：final0
过渡长度：自动
混音风格：Club
效果强度：65%～78%
BPM 回归：16 小节（BPM 差 > 4）
BPM 回归：8 小节（BPM 差 <= 4）
最大变速：±8%～10%
AutoMix 策略：AutoMix-like
过渡渲染引擎：Adaptive
时间拉伸质量：Rubber Band R3
```

## 主要文件

```text
app.py                         GUI

autodj/beat_this_analyzer.py  Beat This! 分析、缓存、稳定网格修复
autodj/audio_engine.py        实时引擎、连续 BPM 恢复
autodj/time_stretch.py        定速与 time-map 连续变速
autodj/transition_matcher.py  自动 OUT/IN/过渡长度搜索
autodj/edm_structure.py       EDM cue/phrase/结构特征
autodj/spectral_seam.py       频时域谱缝合
autodj/timeline.py            GUI 时间轴
```

## 技术边界

- Beat This! 是离线分析器，不在声卡回调中运行。
- 声卡回调只读取预渲染数组并进行混合，避免模型推理导致掉音。
- 任何时间拉伸都不是 bit-perfect；这里的目标是无空白、无 click、速度连续和尽量保留瞬态。
- 两首速度、调性和编曲差异极大的歌曲不可能保证完全听不出切换。
- 高质量时间拉伸和 stem 分离仍是进一步提升电子音乐效果的关键。
