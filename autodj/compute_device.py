from __future__ import annotations

from typing import Any


def resolve_torch_device(requested: str | None, *, strict_cuda: bool = False) -> str:
    """Resolve auto/cuda/mps/cpu against the current Python environment.

    `strict_cuda=True` is used for an explicitly requested CUDA device so the
    application does not silently run a large model on CPU.
    """
    import torch

    value = str(requested or "auto").strip().lower()
    if value in {"", "auto", "system", "default"}:
        if torch.cuda.is_available():
            return "cuda"
        if bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
            return "mps"
        return "cpu"
    if value.startswith("cuda"):
        if torch.cuda.is_available():
            return value
        if strict_cuda:
            raise RuntimeError(
                "已明确选择 CUDA，但当前 Python 环境的 PyTorch 无法使用 CUDA。"
            )
        return "cpu"
    if value == "mps":
        available = bool(
            getattr(torch.backends, "mps", None)
            and torch.backends.mps.is_available()
        )
        return "mps" if available else "cpu"
    if value == "cpu":
        return "cpu"
    raise ValueError(f"未知计算设备：{requested}")


def torch_cuda_diagnostics(torch_module: Any) -> dict[str, Any]:
    """Return CUDA build/device details and perform a real tensor smoke test."""
    info: dict[str, Any] = {
        "torch": str(torch_module.__version__),
        "cuda_build": str(getattr(torch_module.version, "cuda", None) or "none"),
        "cuda_available": bool(torch_module.cuda.is_available()),
        "cuda_usable": False,
        "device_count": 0,
        "device_name": "",
        "capability": "",
        "arch_list": [],
        "cuda_error": "",
    }
    if not info["cuda_available"]:
        return info
    try:
        count = int(torch_module.cuda.device_count())
        index = int(torch_module.cuda.current_device()) if count else 0
        info["device_count"] = count
        info["device_name"] = str(torch_module.cuda.get_device_name(index))
        major, minor = torch_module.cuda.get_device_capability(index)
        info["capability"] = f"sm_{major}{minor}"
        if hasattr(torch_module.cuda, "get_arch_list"):
            info["arch_list"] = list(torch_module.cuda.get_arch_list())
        # is_available() alone is insufficient for a brand-new GPU architecture.
        # Force a tiny kernel launch and synchronize to catch unsupported builds.
        probe = torch_module.ones(8, device=f"cuda:{index}")
        probe = probe.square().sum()
        _ = float(probe.item())
        torch_module.cuda.synchronize(index)
        info["cuda_usable"] = True
    except Exception as exc:  # depends on local driver/build
        info["cuda_error"] = f"{type(exc).__name__}: {exc}"
    return info
