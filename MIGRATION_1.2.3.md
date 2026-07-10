# 1.2.3 RTX 5070 / CUDA 升级说明

建议解压到新目录：

```text
beatthis_muq_songformer_auto_dj_gui_v123
```

旧的 SongFormer 模型缓存、MuQ 缓存和配置文件可以继续使用，但需要修复独立 worker 中的 PyTorch：

```powershell
conda activate beatthis-auto-dj
cd beatthis_muq_songformer_auto_dj_gui_v123
python repair_songformer_cuda.py
python verify_songformer.py --require-cuda
python app.py
```

修复脚本只修改 `songformer-auto-dj` 独立环境，不覆盖主程序的 Beat This!、MuQ 或音频环境。

成功验证的关键字段：

```text
cuda_available = true
cuda_usable = true
device_name = NVIDIA GeForce RTX 5070
cuda_build = 12.8
```

如果仍失败：

```powershell
nvidia-smi
conda run -n songformer-auto-dj python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

明确选择 `cuda` 后，新版不会回退 CPU；错误日志会给出 PyTorch、CUDA build、GPU 名称和修复命令。
