# 从 1.2.5 升级到 1.2.6

无需重装 Beat This!、MuQ、All-In-One 或 Rubber Band。解压到新目录后直接运行：

```powershell
conda activate beatthis-auto-dj
cd beatthis_muq_allinone_mainprocess_auto_dj_gui_v126
python app.py
```

新版沿用原配置和模型缓存。主要变化是逐拍鼓组微对齐、双鼓短重叠后交接，以及结构条件允许时优先 Double Drop。
