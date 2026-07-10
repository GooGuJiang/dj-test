from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any


PROGRESS_PREFIX = "AUTODJ_PROGRESS "


def _emit_progress(current: int, total: int, message: str, stage: str) -> None:
    payload = {
        "current": int(current),
        "total": int(total),
        "message": str(message),
        "stage": str(stage),
    }
    print(PROGRESS_PREFIX + json.dumps(payload, ensure_ascii=False), flush=True)


def _write_json(path: str | Path | None, payload: dict[str, Any]) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_suffix(target.suffix + ".tmp")
        temp.write_text(text, encoding="utf-8")
        temp.replace(target)
    else:
        print(text)



def _nvidia_smi_info() -> dict[str, Any]:
    command = shutil.which("nvidia-smi")
    if not command:
        return {"nvidia_driver_detected": False, "nvidia_smi_name": "", "driver_version": ""}
    process = subprocess.run(
        [command, "--query-gpu=name,driver_version", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    if process.returncode != 0 or not process.stdout.strip():
        return {"nvidia_driver_detected": False, "nvidia_smi_name": "", "driver_version": ""}
    first = process.stdout.strip().splitlines()[0]
    parts = [part.strip() for part in first.split(",", 1)]
    return {
        "nvidia_driver_detected": True,
        "nvidia_smi_name": parts[0],
        "driver_version": parts[1] if len(parts) > 1 else "",
    }


def _cuda_diagnostics(torch_module: Any) -> dict[str, Any]:
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
        probe = torch_module.ones(8, device=f"cuda:{index}")
        probe = probe.square().sum()
        _ = float(probe.item())
        torch_module.cuda.synchronize(index)
        info["cuda_usable"] = True
    except Exception as exc:  # pragma: no cover - depends on local CUDA stack
        info["cuda_error"] = f"{type(exc).__name__}: {exc}"
    return info


def _resolve_device(torch_module: Any, requested: str) -> tuple[str, dict[str, Any]]:
    value = str(requested or "auto").strip().lower()
    cuda = _cuda_diagnostics(torch_module)
    nvidia = _nvidia_smi_info()
    cuda.update(nvidia)

    if value in {"", "auto", "system", "default"}:
        if cuda["cuda_usable"]:
            return "cuda:0", cuda
        if cuda.get("nvidia_driver_detected"):
            raise RuntimeError(
                "系统检测到 NVIDIA 显卡，但 SongFormer worker 的 PyTorch 无法使用 CUDA。"
                f" GPU={cuda.get('nvidia_smi_name') or 'NVIDIA GPU'}，"
                f"driver={cuda.get('driver_version') or 'unknown'}，"
                f"PyTorch={cuda.get('torch')}，CUDA build={cuda.get('cuda_build')}。"
                " 请运行 python repair_songformer_cuda.py，并确认 NVIDIA 驱动为最新版；不会自动退回 CPU。"
            )
        available = bool(
            hasattr(torch_module.backends, "mps")
            and torch_module.backends.mps.is_available()
        )
        return ("mps" if available else "cpu"), cuda

    if value.startswith("cuda"):
        if not cuda["cuda_available"]:
            raise RuntimeError(
                "已选择 CUDA，但 SongFormer worker 的 PyTorch 是 CPU 构建或无法连接 NVIDIA 驱动。"
                f" torch={cuda['torch']}，torch.version.cuda={cuda['cuda_build']}。"
                " 请运行 python repair_songformer_cuda.py。"
            )
        if not cuda["cuda_usable"]:
            raise RuntimeError(
                "SongFormer worker 检测到 CUDA，但无法在显卡上执行张量运算。"
                f" GPU={cuda['device_name'] or 'unknown'}，能力={cuda['capability'] or 'unknown'}，"
                f"PyTorch={cuda['torch']}，CUDA build={cuda['cuda_build']}，"
                f"错误={cuda['cuda_error'] or 'unknown'}。"
                " RTX 5070 请使用 PyTorch CUDA 12.8 构建；运行 python repair_songformer_cuda.py。"
            )
        if value == "cuda":
            value = "cuda:0"
        index_text = value.split(":", 1)[1] if ":" in value else "0"
        try:
            index = int(index_text)
        except ValueError as exc:
            raise ValueError(f"CUDA 设备格式无效：{requested}") from exc
        if index < 0 or index >= int(cuda["device_count"]):
            raise RuntimeError(
                f"CUDA 设备 {index} 不存在；当前检测到 {cuda['device_count']} 张显卡。"
            )
        torch_module.cuda.set_device(index)
        return f"cuda:{index}", cuda

    if value == "mps":
        available = bool(
            hasattr(torch_module.backends, "mps")
            and torch_module.backends.mps.is_available()
        )
        if not available:
            raise RuntimeError("已选择 MPS，但当前 SongFormer worker 不支持 MPS。")
        return "mps", cuda
    if value == "cpu":
        return "cpu", cuda
    raise ValueError(f"未知 SongFormer 设备：{requested}")


def _normalise_segments(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, dict):
        for key in ("segments", "result", "predictions"):
            if key in raw:
                raw = raw[key]
                break
    if hasattr(raw, "segments"):
        raw = getattr(raw, "segments")
    if not isinstance(raw, (list, tuple)):
        raise TypeError(f"SongFormer 返回类型无法识别：{type(raw)!r}")

    segments: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, dict):
            start = item.get("start", 0.0)
            end = item.get("end", start)
            label = item.get("label", "unknown")
            confidence = item.get("confidence", item.get("score", 1.0))
        else:
            start = getattr(item, "start", 0.0)
            end = getattr(item, "end", start)
            label = getattr(item, "label", "unknown")
            confidence = getattr(item, "confidence", getattr(item, "score", 1.0))
        start_f = float(start)
        end_f = float(end)
        if end_f <= start_f:
            continue
        segments.append(
            {
                "start": start_f,
                "end": end_f,
                "label": str(label).strip().lower() or "unknown",
                "confidence": float(confidence),
            }
        )
    segments.sort(key=lambda item: (item["start"], item["end"]))
    return segments


def _probe() -> dict[str, Any]:
    import torch
    import transformers
    import huggingface_hub

    optional: dict[str, str] = {}
    for module_name in ("muq", "librosa", "safetensors", "accelerate"):
        try:
            module = __import__(module_name)
            optional[module_name] = str(getattr(module, "__version__", "installed"))
        except Exception as exc:  # pragma: no cover - depends on local environment
            optional[module_name] = f"missing: {exc}"

    cuda = _cuda_diagnostics(torch)
    cuda.update(_nvidia_smi_info())
    return {
        "ok": True,
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "transformers": str(transformers.__version__),
        "huggingface_hub": str(huggingface_hub.__version__),
        **cuda,
        "mps_available": bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ),
        "optional": optional,
    }


def _load_model(model_name: str, requested_device: str):
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModel

    # Validate the requested accelerator before downloading the multi-GB model.
    device, cuda = _resolve_device(torch, requested_device)

    # This follows the official ASLP-lab/SongFormer Hugging Face QuickStart.
    local_dir = snapshot_download(
        repo_id=model_name,
        repo_type="model",
        local_dir_use_symlinks=False,
        resume_download=True,
        allow_patterns="*",
        ignore_patterns=["SongFormer.pt", "SongFormer.safetensors"],
    )
    local_dir = str(Path(local_dir).resolve())
    if local_dir not in sys.path:
        sys.path.insert(0, local_dir)
    os.environ["SONGFORMER_LOCAL_DIR"] = local_dir

    model = AutoModel.from_pretrained(
        local_dir,
        trust_remote_code=True,
        low_cpu_mem_usage=False,
    )
    model.to(device)
    model.eval()
    return model, device, local_dir, cuda


def _analyze_manifest(manifest_path: Path, output_path: Path) -> int:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    paths = [str(Path(value).expanduser().resolve()) for value in manifest.get("paths", [])]
    model_name = str(manifest.get("model_name", "ASLP-lab/SongFormer"))
    requested_device = str(manifest.get("device", "auto"))

    payload: dict[str, Any] = {
        "ok": False,
        "model_name": model_name,
        "requested_device": requested_device,
        "results": {},
        "errors": {},
    }
    try:
        import torch

        total = len(paths)
        _emit_progress(-1, total, "SongFormer 正在下载或加载官方权重", "model")
        model, device, local_dir, cuda = _load_model(model_name, requested_device)
        if device.startswith("cuda"):
            label = (
                f"{device} · {cuda.get('device_name', 'NVIDIA GPU')} · "
                f"PyTorch {cuda.get('torch')} / CUDA {cuda.get('cuda_build')}"
            )
        else:
            label = device
        _emit_progress(0, total, f"SongFormer 模型已加载 · {label}", "model_ready")
        payload["device"] = device
        payload["cuda"] = cuda
        payload["local_dir"] = local_dir
        with torch.inference_mode():
            for index, path in enumerate(paths):
                name = Path(path).name
                _emit_progress(index, total, f"SongFormer 正在分析：{name}", "track")
                try:
                    raw = model(path)
                    payload["results"][path] = {
                        "segments": _normalise_segments(raw),
                        "backend": "songformer-hf-official",
                    }
                except Exception as exc:
                    payload["errors"][path] = f"{type(exc).__name__}: {exc}"
                payload["completed"] = index + 1
                _write_json(output_path, payload)
                if path in payload["errors"]:
                    message = f"SongFormer 失败：{name} · {payload['errors'][path]}"
                else:
                    message = f"SongFormer 完成：{name}"
                _emit_progress(index + 1, total, message, "track_done")
        payload["ok"] = bool(payload["results"])
        _write_json(output_path, payload)
        _emit_progress(total, total, "SongFormer 批量结构分析完成", "done")
        return 0 if payload["ok"] else 2
    except Exception as exc:
        payload["fatal_error"] = f"{type(exc).__name__}: {exc}"
        payload["traceback"] = traceback.format_exc()
        _write_json(output_path, payload)
        print(payload["traceback"], file=sys.stderr, flush=True)
        return 2


def main() -> int:
    parser = argparse.ArgumentParser(description="SongFormer isolated inference worker")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.probe:
        try:
            _write_json(None, _probe())
            return 0
        except Exception as exc:
            _write_json(
                None,
                {
                    "ok": False,
                    "python": sys.version.split()[0],
                    "python_executable": sys.executable,
                    "message": f"{type(exc).__name__}: {exc}",
                },
            )
            return 2

    if args.manifest is None or args.output is None:
        parser.error("--manifest and --output are required for inference")
    return _analyze_manifest(args.manifest, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
