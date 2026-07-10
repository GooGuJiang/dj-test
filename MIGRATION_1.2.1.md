# 1.2.1 升级说明

1. 删除或移走旧的 1.2.0 目录。
2. 解压完整的 1.2.1 压缩包到新目录。
3. 激活原主环境：`conda activate beatthis-auto-dj`。
4. 运行 `python verify_songformer.py`。
5. 运行 `python app.py`。

不需要重新创建 SongFormer worker 环境，也不需要重新下载已经缓存的模型权重。
