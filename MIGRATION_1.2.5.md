# 从 1.2.4 升级到 1.2.5

本版本修复下一首重复同步后无法提交到播放器的问题，并改变时间轴跳转行为。

## 升级

建议解压到新目录后直接运行：

```powershell
conda activate beatthis-auto-dj
cd beatthis_muq_allinone_mainprocess_auto_dj_gui_v125
python app.py
```

无需重新安装 Beat This!、MuQ、All-In-One 或 Rubber Band，也无需删除模型缓存。

## 正常日志

相同下一首只应出现一次“已同步”，最终必须出现：

```text
下一首已提交到播放器：歌曲名称
```

如果播放前预加载尚未完成就点击播放，可能出现：

```text
复用正在准备的音轨：歌曲名称
```

这是正常的任务复用，不是第二次处理。

## Seek 行为

时间轴点击/拖动后不再显示“切歌规划正在更新”。它只使用已有分析和预渲染结果，不会重新运行模型或时间拉伸。
