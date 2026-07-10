from __future__ import annotations

import collections
import collections.abc
import json
import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from .models import AllInOneProfile, FunctionalSegment
from .natten_compat import NattenCompatStatus, ensure_natten_compat

StatusCallback = Callable[[str], None]
ProgressCallback = Callable[[float, float, str], None]
LOGGER = logging.getLogger("autodj.allinone")
_ANALYSIS_LOCK = threading.Lock()

# Compatibility aliases required by older madmom releases.
if not hasattr(collections, "MutableSequence"):
    collections.MutableSequence = collections.abc.MutableSequence  # type: ignore[attr-defined]
if not hasattr(np, "float"):
    np.float = np.float64  # type: ignore[attr-defined]
if not hasattr(np, "int"):
    np.int = np.int_  # type: ignore[attr-defined]
if not hasattr(np, "bool"):
    np.bool = np.bool_  # type: ignore[attr-defined]


class AllInOneAnalyzer:
    """In-process cached adapter for ``mir-aidj/all-in-one``.

    Inference runs in the application's background Python thread, not in a
    subprocess or a separate Conda worker. Calls are serialized to keep Demucs,
    NATTEN, MuQ and preload jobs from competing for memory.
    """

    def __init__(
        self,
        model: str = "harmonix-all",
        device: str = "cpu",
        cache_path: str | Path | None = None,
        force_torch_natten: bool | None = None,
        cpu_threads: int | None = None,
    ) -> None:
        self.model = str(model)
        self.device = self._resolve_device(str(device))
        self.cache_path = Path(
            cache_path or Path.home() / ".beatthis_muq_allinone_cache_v3.json"
        )
        self.force_torch_natten = (
            bool(force_torch_natten)
            if force_torch_natten is not None
            else os.environ.get("AUTODJ_FORCE_TORCH_NATTEN", "").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        configured_threads = int(os.environ.get("AUTODJ_ALLIN1_CPU_THREADS", "4"))
        self.cpu_threads = max(1, int(cpu_threads or configured_threads))
        self._cache_lock = threading.RLock()
        self._cache = self._read_cache()
        self._natten_status: NattenCompatStatus | None = None

    @staticmethod
    def _resolve_device(device: str) -> str:
        value = device.strip().lower()
        if value == "auto":
            # CPU is intentionally the default for the structure model. It keeps
            # RTX memory available for MuQ, Beat This! and real-time preload.
            return "cpu"
        if value not in {"cpu", "cuda", "mps"}:
            raise ValueError(f"未知 All-In-One 设备：{device}")
        return value

    def _read_cache(self) -> dict:
        try:
            if self.cache_path.exists():
                data = json.loads(self.cache_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
        except (OSError, json.JSONDecodeError):
            pass
        return {"version": 3, "tracks": {}}

    def _write_cache(self) -> None:
        with self._cache_lock:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.cache_path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(self._cache, ensure_ascii=False), encoding="utf-8"
            )
            temporary.replace(self.cache_path)

    def _cache_key(self, path: Path) -> str:
        stat = path.stat()
        return (
            f"{path.resolve()}::{stat.st_size}::{stat.st_mtime_ns}::"
            f"{self.model}::allin1-1.1-main-thread-v3"
        )

    def _import_backend(self):
        self._natten_status = ensure_natten_compat(self.force_torch_natten)
        try:
            import torch

            if self.device == "cpu":
                torch.set_num_threads(self.cpu_threads)
                try:
                    torch.set_num_interop_threads(1)
                except RuntimeError:
                    pass
            elif self.device == "cuda" and not torch.cuda.is_available():
                raise RuntimeError(
                    "当前主环境的 PyTorch 无法使用 CUDA。请把 All-In-One 设备设为 CPU，"
                    "或修复主环境 CUDA。"
                )
            import allin1
        except Exception as exc:
            raise RuntimeError(
                "无法在当前主环境导入 All-In-One。请运行 python install_allinone.py。"
            ) from exc
        return allin1

    @staticmethod
    def _convert(
        result: object, model: str, natten_status: NattenCompatStatus
    ) -> AllInOneProfile:
        segments = tuple(
            FunctionalSegment(
                start=float(getattr(segment, "start")),
                end=float(getattr(segment, "end")),
                label=str(getattr(segment, "label")).lower(),
                confidence=1.0,
            )
            for segment in getattr(result, "segments", [])
        )
        return AllInOneProfile(
            bpm=float(getattr(result, "bpm", 0.0)),
            beats=tuple(float(value) for value in getattr(result, "beats", [])),
            downbeats=tuple(float(value) for value in getattr(result, "downbeats", [])),
            beat_positions=tuple(
                int(value) for value in getattr(result, "beat_positions", [])
            ),
            segments=segments,
            backend=(
                "allin1-native" if natten_status.native else "allin1-torch-natten"
            ),
            model_name=model,
            natten_backend=natten_status.backend,
        )

    def analyze_many(
        self,
        paths: Iterable[str | Path],
        status: StatusCallback | None = None,
        progress: ProgressCallback | None = None,
        force: bool = False,
    ) -> dict[str, AllInOneProfile]:
        resolved = [Path(path).expanduser().resolve() for path in paths]
        for path in resolved:
            if not path.is_file():
                raise FileNotFoundError(path)

        total = len(resolved)
        output: dict[str, AllInOneProfile] = {}
        todo: list[Path] = []
        cached_count = 0
        for path in resolved:
            key = self._cache_key(path)
            cached = None if force else self._cache.get("tracks", {}).get(key)
            if cached:
                output[str(path)] = AllInOneProfile.from_json_dict(cached)
                cached_count += 1
                message = f"All-In-One 缓存：{path.name}"
                LOGGER.info(message)
                if status:
                    status(message)
                if progress:
                    progress(cached_count, total, f"缓存 · {path.name}")
            else:
                todo.append(path)

        if not todo:
            return output

        if progress:
            progress(-1, total, f"等待结构分析锁 · {len(todo)} 首")

        with _ANALYSIS_LOCK:
            allin1 = self._import_backend()
            assert self._natten_status is not None
            message = (
                f"All-In-One 主进程后台线程分析 {len(todo)} 首 · {self.model} · "
                f"{self.device} · {self._natten_status.backend}"
            )
            LOGGER.info(message)
            if status:
                status(message)
            if progress:
                progress(-1, total, f"Demucs + 结构模型 · {self.device}")

            with tempfile.TemporaryDirectory(prefix="autodj_allin1_") as directory:
                root = Path(directory)
                try:
                    results = allin1.analyze(
                        [str(path) for path in todo],
                        model=self.model,
                        device=self.device,
                        include_activations=False,
                        include_embeddings=False,
                        demix_dir=root / "demix",
                        spec_dir=root / "spec",
                        keep_byproducts=False,
                        overwrite=True,
                        multiprocess=False,
                    )
                except Exception as exc:
                    raise RuntimeError(
                        "All-In-One 推理失败。请运行 python verify_allinone.py 查看诊断。"
                    ) from exc
                finally:
                    if self.device == "cuda":
                        try:
                            import torch

                            torch.cuda.empty_cache()
                        except Exception:
                            pass

        if not isinstance(results, list):
            results = [results]
        result_by_path = {
            str(Path(getattr(result, "path")).expanduser().resolve()): result
            for result in results
        }
        completed = cached_count
        with self._cache_lock:
            tracks_cache = self._cache.setdefault("tracks", {})
            for result_index, path in enumerate(todo):
                raw = result_by_path.get(str(path))
                if raw is None and result_index < len(results):
                    raw = results[result_index]
                if raw is None:
                    continue
                profile = self._convert(raw, self.model, self._natten_status)
                output[str(path)] = profile
                tracks_cache[self._cache_key(path)] = profile.to_json_dict()
                completed += 1
                labels = ", ".join(profile.unique_labels[:6]) or "无标签"
                message = f"All-In-One 完成：{path.name} · {labels}"
                LOGGER.info(message)
                if status:
                    status(message)
                if progress:
                    progress(completed, total, path.name)
        self._write_cache()
        return output

    def analyze(
        self,
        path: str | Path,
        status: StatusCallback | None = None,
        progress: ProgressCallback | None = None,
        force: bool = False,
    ) -> AllInOneProfile:
        resolved = str(Path(path).expanduser().resolve())
        return self.analyze_many(
            [resolved], status=status, progress=progress, force=force
        ).get(resolved, AllInOneProfile(backend="fallback"))


def probe_allinone(force_torch_natten: bool = False) -> dict[str, object]:
    try:
        status = ensure_natten_compat(force_torch_natten)
        import allin1

        version = getattr(allin1, "__version__", "1.1.x")
        return {
            "ok": True,
            "allin1_version": str(version),
            "natten_backend": status.backend,
            "natten_version": status.version,
            "message": status.message,
            "in_process": True,
        }
    except Exception as exc:
        return {
            "ok": False,
            "allin1_version": "",
            "natten_backend": "unavailable",
            "natten_version": "",
            "message": str(exc),
            "in_process": True,
        }
