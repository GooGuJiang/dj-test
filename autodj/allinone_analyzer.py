from __future__ import annotations

import collections
import collections.abc
import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from .models import AllInOneProfile, FunctionalSegment
from .natten_compat import NattenCompatStatus, ensure_natten_compat


StatusCallback = Callable[[str], None]

# Current madmom main fixes most NumPy removals, but these aliases keep older
# cached installations importable without forcing the DJ application to use an
# old NumPy globally.
if not hasattr(collections, "MutableSequence"):
    collections.MutableSequence = collections.abc.MutableSequence  # type: ignore[attr-defined]
if not hasattr(np, "float"):
    np.float = np.float64  # type: ignore[attr-defined]
if not hasattr(np, "int"):
    np.int = np.int_  # type: ignore[attr-defined]
if not hasattr(np, "bool"):
    np.bool = np.bool_  # type: ignore[attr-defined]


class AllInOneAnalyzer:
    """Cached adapter for ``mir-aidj/all-in-one`` functional structure analysis."""

    def __init__(
        self,
        model: str = "harmonix-all",
        device: str = "cpu",
        cache_path: str | Path | None = None,
        force_torch_natten: bool | None = None,
    ) -> None:
        self.model = str(model)
        self.device = str(device)
        self.cache_path = Path(
            cache_path or Path.home() / ".beatthis_muq_allinone_cache.json"
        )
        self.force_torch_natten = (
            bool(force_torch_natten)
            if force_torch_natten is not None
            else os.environ.get("AUTODJ_FORCE_TORCH_NATTEN", "").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        self._cache_lock = threading.RLock()
        self._cache = self._read_cache()
        self._natten_status: NattenCompatStatus | None = None

    def _read_cache(self) -> dict:
        try:
            if self.cache_path.exists():
                data = json.loads(self.cache_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
        except (OSError, json.JSONDecodeError):
            pass
        return {"version": 2, "tracks": {}}

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
            f"{self.model}::allin1-1.1"
        )

    def _import_backend(self):
        self._natten_status = ensure_natten_compat(self.force_torch_natten)
        try:
            import allin1
        except Exception as exc:
            raise RuntimeError(
                "无法导入 All-In-One。请运行 python install_allinone.py。"
            ) from exc
        return allin1

    @staticmethod
    def _convert(result: object, model: str, natten_status: NattenCompatStatus) -> AllInOneProfile:
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
            backend=("allin1-native" if natten_status.native else "allin1-torch-natten"),
            model_name=model,
            natten_backend=natten_status.backend,
        )

    def analyze_many(
        self,
        paths: Iterable[str | Path],
        status: StatusCallback | None = None,
        force: bool = False,
    ) -> dict[str, AllInOneProfile]:
        resolved = [Path(path).expanduser().resolve() for path in paths]
        for path in resolved:
            if not path.is_file():
                raise FileNotFoundError(path)

        output: dict[str, AllInOneProfile] = {}
        todo: list[Path] = []
        for path in resolved:
            key = self._cache_key(path)
            cached = None if force else self._cache.get("tracks", {}).get(key)
            if cached:
                output[str(path)] = AllInOneProfile.from_json_dict(cached)
                if status:
                    status(f"All-In-One 缓存：{path.name}")
            else:
                todo.append(path)

        if not todo:
            return output

        allin1 = self._import_backend()
        assert self._natten_status is not None
        if status:
            status(
                f"All-In-One 正在分析 {len(todo)} 首歌曲 · "
                f"{self.model} · {self._natten_status.backend}"
            )

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
                    "All-In-One 推理失败。Windows/Linux 若未安装兼容 NATTEN，"
                    "程序会自动使用较慢的纯 PyTorch 兼容层；仍失败时请运行 "
                    "python verify_allinone.py 查看诊断。"
                ) from exc

        if not isinstance(results, list):
            results = [results]
        result_by_path = {
            str(Path(getattr(result, "path")).expanduser().resolve()): result
            for result in results
        }
        with self._cache_lock:
            tracks_cache = self._cache.setdefault("tracks", {})
            for result_index, path in enumerate(todo):
                raw = result_by_path.get(str(path))
                if raw is None and result_index < len(results):
                    # Some decoder/backends normalise or resolve paths differently;
                    # All-In-One preserves input order, so use the positional result
                    # as a safe fallback.
                    raw = results[result_index]
                if raw is None:
                    continue
                profile = self._convert(raw, self.model, self._natten_status)
                output[str(path)] = profile
                tracks_cache[self._cache_key(path)] = profile.to_json_dict()
                if status:
                    labels = ", ".join(profile.unique_labels[:6]) or "无标签"
                    status(f"All-In-One 完成：{path.name} · {labels}")
        self._write_cache()
        return output

    def analyze(
        self,
        path: str | Path,
        status: StatusCallback | None = None,
        force: bool = False,
    ) -> AllInOneProfile:
        resolved = str(Path(path).expanduser().resolve())
        return self.analyze_many([resolved], status=status, force=force).get(
            resolved, AllInOneProfile(backend="fallback")
        )


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
        }
    except Exception as exc:
        return {
            "ok": False,
            "allin1_version": "",
            "natten_backend": "unavailable",
            "natten_version": "",
            "message": str(exc),
        }
