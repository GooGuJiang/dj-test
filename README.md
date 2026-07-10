# Beat This! + MuQ + SongFormer Auto DJ 1.2.3

这是 RTX 50 系列 / SongFormer CUDA 修复版。结构分析仍只使用官方 `ASLP-lab/SongFormer`；Beat This! 负责节拍网格，MuQ 负责风格语义与播放列表排序。

## 1.2.3：RTX 5070 CUDA 修复

1.2.2 的 SongFormer 独立环境默认从普通 PyPI 安装 `torch==2.4.0`，在 Windows 上可能得到 CPU 构建；GUI 的旧默认计算设备也是 `cpu`。这会导致日志显示：

```text
[SongFormer] SongFormer 模型已加载 · cpu
```

1.2.3 已改为：

- GUI 默认计算设备为 `auto`，优先 CUDA
- 自动迁移 1.2.2 保存的默认 `cpu` 设置为 `auto`
- RTX 50 系列使用 PyTorch `2.7.1` + CUDA `12.8` wheel
- worker 不只检查 `torch.cuda.is_available()`，还会真正执行一次 CUDA 张量运算
- 日志显示 GPU 名称、计算能力、PyTorch 版本和 CUDA build
- 明确选择 CUDA 时，失败会直接报错，不再静默回退 CPU
- `auto` 检测到 NVIDIA 驱动但 PyTorch 无法用 CUDA时，也不会偷偷转 CPU
- 新增 `repair_songformer_cuda.py`，可原地修复已有 worker 环境

成功日志应类似：

```text
[SongFormer] SongFormer 模型已加载 · cuda:0 · NVIDIA GeForce RTX 5070 · PyTorch 2.7.1+cu128 / CUDA 12.8
```

## RTX 5070 修复已有环境

在项目目录、主环境中执行：

```powershell
conda activate beatthis-auto-dj
python repair_songformer_cuda.py
python verify_songformer.py --require-cuda
python app.py
```

验证输出应包含：

```json
{
  "cuda_available": true,
  "cuda_usable": true,
  "device_name": "NVIDIA GeForce RTX 5070",
  "cuda_build": "12.8"
}
```

若驱动版本过旧，请先更新 NVIDIA 驱动，再重新执行修复脚本。

## 全新安装 SongFormer worker

```powershell
conda activate beatthis-auto-dj
python install_songformer.py --compute cuda
python verify_songformer.py --require-cuda
python app.py
```

安装脚本会创建或修复：

```text
songformer-auto-dj
```

并安装：

```text
PyTorch 2.7.1
Torchaudio 2.7.1
CUDA wheel: cu128
```

SongFormer 其他依赖不会再把 PyTorch 固定回 CPU 版本。

## 计算设备选项

GUI 的“计算设备”支持：

- `auto`：优先 CUDA，其次 MPS，最后 CPU；检测到 NVIDIA 显卡但 CUDA 环境损坏时直接提示修复
- `cuda`：强制 GPU，无法使用时直接报错
- `cpu`：明确使用 CPU
- `mps`：Apple Silicon

RTX 5070 推荐使用 `auto` 或 `cuda`。

## 模型分工

```text
Beat This!
  └── beat、downbeat、小节网格和播放时间权威

SongFormer
  └── intro、verse、pre-chorus、chorus、bridge、instrumental、outro

MuQ-large-msd-iter
  └── 风格语义、Outro→Intro 兼容度和播放列表排序

本地 EDM 结构融合
  └── 根据能量、鼓组、低频和结构边界推断 buildup、drop、breakdown
```

## 分析进度与日志

GUI 底部显示模型加载、当前歌曲、完成数量和百分比；PowerShell 实时显示 Beat This!、SongFormer、MuQ、排序和预加载日志。

SongFormer CUDA 诊断会输出：

```text
GPU 名称
CUDA 可用性
实际 CUDA kernel 测试
GPU compute capability
PyTorch CUDA build
当前选择的 cuda:N
```

## 运行

```powershell
conda activate beatthis-auto-dj
python app.py
```

首次实际分析会从 Hugging Face 下载官方 SongFormer 权重。

## 测试

- CUDA 自动选择测试
- NVIDIA 环境禁止静默 CPU 回退测试
- 显式 CUDA 错误诊断测试
- SongFormer worker 依赖隔离测试
- GUI/引擎 `auto` 设备设置测试
- 全项目 43 项自动测试
