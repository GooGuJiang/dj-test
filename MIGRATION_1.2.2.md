# 1.2.2 升级说明

建议把压缩包解压到新的 `beatthis_muq_songformer_auto_dj_gui_v122` 目录后运行，不要只复制 `app.py`。

现有环境和缓存均可复用：

```powershell
conda activate beatthis-auto-dj
cd beatthis_muq_songformer_auto_dj_gui_v122
python verify_songformer.py
python app.py
```

无需重新安装 SongFormer worker。首次没有缓存的实际结构分析仍会下载官方权重。

新增的窗口底部进度条会依次显示 Beat This!、SongFormer、MuQ 三个阶段；详细日志保留在启动 `python app.py` 的 PowerShell 中。
