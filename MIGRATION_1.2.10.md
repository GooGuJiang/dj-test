# 1.2.10 升级说明：CUE-DETR / All-In-One 共存修复

1.2.9 中，All-In-One 的纯 PyTorch NATTEN 兼容层会向 `sys.modules` 注入临时模块。
该模块没有合法的 `__spec__`，Transformers 在检测可选依赖时会触发：

```text
ValueError: natten.__spec__ is None
```

GUI 随后把这个导入冲突误报为“CUE-DETR 未安装”。

1.2.10 为 `natten` 和 `natten.functional` 注入完整 `ModuleSpec`，并让 CUE-DETR
检测直接导入实际使用的 `DetrImageProcessor` 与 `DetrForObjectDetection`。

已有环境不需要重新安装依赖：

```powershell
conda activate beatthis-auto-dj
python verify_cuedetr.py
python app.py
```

验证成功后应显示：

```text
CUE-DETR 依赖已就绪 · Transformers 4.x · CUDA 可用
```
