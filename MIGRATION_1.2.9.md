# 从 1.2.8 升级到 1.2.9

1. 解压到新目录，不要覆盖旧目录。
2. 激活原主环境：

```powershell
conda activate beatthis-auto-dj
```

3. 安装 CUE-DETR 额外依赖：

```powershell
python install_cuedetr.py
python verify_cuedetr.py
```

4. 启动：

```powershell
python app.py
```

旧的 Beat This!、MuQ、All-In-One、Rubber Band 与设置缓存可以继续使用。首次运行需要为歌曲生成 CUE-DETR cue 缓存。

重要行为变化：

- CUE-DETR 固定启用，并成为唯一切点来源。
- 未完成 CUE-DETR 分析时不会开始播放。
- 旧的 novelty/规则 cue 不再作为备用。
- 播放前必须完成当前→下一首的最终过渡渲染。
