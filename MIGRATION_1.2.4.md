# 从 1.2.3 升级到 1.2.4

1. 解压到新目录，不要覆盖旧 SongFormer 目录。
2. 激活原来的主环境：

```powershell
conda activate beatthis-auto-dj
```

3. 在主环境安装 All-In-One：

```powershell
python install_allinone.py
python verify_allinone.py
```

4. 启动：

```powershell
python app.py
```

新版不再使用 `songformer-auto-dj` 环境。该旧环境可以自行删除，但删除不是运行新版的必要条件。

推荐配置：

```text
Beat This! / MuQ：auto
All-In-One：cpu
```
