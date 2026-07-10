from __future__ import annotations

from pathlib import Path

import pytest

import songformer_worker
from autodj.audio_engine import AutoDJEngine, EngineConfig


class _Cuda:
    def __init__(self) -> None:
        self.selected = None

    def set_device(self, index: int) -> None:
        self.selected = index


class _Backends:
    class mps:
        @staticmethod
        def is_available() -> bool:
            return False


class _Torch:
    cuda = _Cuda()
    backends = _Backends()


def _diag(*, available: bool, usable: bool) -> dict:
    return {
        "torch": "2.7.1+cu128",
        "cuda_build": "12.8",
        "cuda_available": available,
        "cuda_usable": usable,
        "device_count": 1 if available else 0,
        "device_name": "NVIDIA GeForce RTX 5070" if available else "",
        "capability": "sm_120" if available else "",
        "arch_list": ["sm_120"] if usable else [],
        "cuda_error": "unsupported" if available and not usable else "",
    }


def test_auto_prefers_usable_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(songformer_worker, "_cuda_diagnostics", lambda _torch: _diag(available=True, usable=True))
    monkeypatch.setattr(songformer_worker, "_nvidia_smi_info", lambda: {"nvidia_driver_detected": True, "nvidia_smi_name": "NVIDIA GeForce RTX 5070", "driver_version": "999.0"})
    device, info = songformer_worker._resolve_device(_Torch(), "auto")
    assert device == "cuda:0"
    assert info["device_name"].endswith("5070")


def test_explicit_cuda_never_silently_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(songformer_worker, "_cuda_diagnostics", lambda _torch: _diag(available=False, usable=False))
    monkeypatch.setattr(songformer_worker, "_nvidia_smi_info", lambda: {"nvidia_driver_detected": True, "nvidia_smi_name": "NVIDIA GeForce RTX 5070", "driver_version": "999.0"})
    with pytest.raises(RuntimeError, match="repair_songformer_cuda.py"):
        songformer_worker._resolve_device(_Torch(), "cuda")



def test_auto_with_nvidia_driver_never_uses_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(songformer_worker, "_cuda_diagnostics", lambda _torch: _diag(available=False, usable=False))
    monkeypatch.setattr(songformer_worker, "_nvidia_smi_info", lambda: {"nvidia_driver_detected": True, "nvidia_smi_name": "NVIDIA GeForce RTX 5070", "driver_version": "999.0"})
    with pytest.raises(RuntimeError, match="不会自动退回 CPU"):
        songformer_worker._resolve_device(_Torch(), "auto")


def test_engine_accepts_auto_compute_device() -> None:
    engine = AutoDJEngine(EngineConfig())
    engine.set_muq_device("auto")
    engine.set_songformer_device("auto")
    assert engine.config.muq_device == "auto"
    assert engine.config.songformer_device == "auto"


def test_songformer_runtime_requirements_do_not_pin_cpu_torch() -> None:
    text = (Path(__file__).parents[1] / "requirements_songformer_runtime.txt").read_text(encoding="utf-8")
    package_names = [line.split("==", 1)[0].strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    assert "torch" not in package_names
    assert "torchaudio" not in package_names
