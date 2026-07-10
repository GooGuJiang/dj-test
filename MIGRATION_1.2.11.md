# 1.2.11 升级说明：尾部→头部接歌与分级预渲染

## 是否需要重新安装模型

不需要。1.2.11 没有更换 CUE-DETR、Beat This!、MuQ 或 All-In-One 权重，也不会修改现有 PyTorch/CUDA 环境。

```powershell
conda activate beatthis-auto-dj
cd beatthis_muq_cuedetr_auto_dj_gui_v1211
python verify_cuedetr.py
python app.py
```

## 行为变化

### 切点方向

- A 的 OUT 只允许位于后 42% / 最后 48 小节窗口。
- B 的 IN 只允许位于前 35% / 前 32 小节窗口。
- 仍然只使用 CUE-DETR cue，不恢复旧规则候选。
- 自动过渡最短提高为 8 小节；手动固定 4 小节不受影响。

### 渲染调度

- 播放前只完整渲染 A→B。
- 播放后只轻量规划最近的 B→C。
- B 成为当前轨时，B→C 升级为高优先级最终渲染。
- 不再完整渲染 C→D、D→E 等整个窗口。
- 点击播放会等待并复用正在进行的热 pair；即使当前轨尚未解码完成，也不会另起重复任务。

### 听感

Drop Swap 和 Double Drop 会从过渡前段引入下一首的低音量鼓组/和声床，bass 仍在中后段交换，减少“第二首最后才突然出现”的感觉。

## 观察日志

正常情况下，每个 pair 应只有一次：

```text
连续 BPM 恢复
DJ 相位锁定
下一首已提交到播放器
```

未来 pair 应先显示：

```text
滑动窗口轻量规划
```

而不是立即显示完整 BPM 恢复和最终渲染。
