from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Callable, Iterable, Sequence

from .models import SongFormerProfile, FunctionalSegment


StatusCallback = Callable[[str], None]
ProgressCallback = Callable[[int, int, str], None]
PROGRESS_PREFIX = "AUTODJ_PROGRESS "


def _project_worker() -> Path:
    return Path(__file__).resolve().parent.parent / "songformer_worker.py"


def _normalise_explicit_python(value: str | Path | None) -> str:
    if value is None:
        return ""
    text = str(value).strip().strip('"')
    return text


def songformer_command(
    python_executable: str | Path | None = None,
    conda_env: str = "songformer-auto-dj",
) -> list[str]:
    """Return an isolated Python command for the official SongFormer runtime.

    SongFormer uses an isolated Python 3.10 runtime, while the main DJ
    application may use a different Python for Beat This!/audio compatibility.
    Keeping it in a worker process avoids dependency and CUDA-library conflicts.
    """
    explicit = _normalise_explicit_python(
        python_executable
        or os.environ.get("AUTODJ_SONGFORMER_PYTHON")
        or os.environ.get("SONGFORMER_PYTHON")
    )
    if explicit:
        path = Path(explicit).expanduser()
        if path.is_file():
            return [str(path.resolve())]
        # Also accept a command name already visible on PATH.
        resolved = shutil.which(explicit)
        if resolved:
            return [resolved]
        raise FileNotFoundError(f"SongFormer Python 不存在：{explicit}")

    conda = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if conda:
        # `conda run` works without activating the environment in the GUI process.
        command = [conda, "run", "--no-capture-output", "-n", conda_env, "python"]
        probe = subprocess.run(
            command + ["-c", "import sys; print(sys.executable)"],
            capture_output=True,
            text=True,
            timeout=25,
            check=False,
        )
        if probe.returncode == 0:
            return command

    return [sys.executable]


def _extract_json(text: str) -> dict:
    text = text.strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # conda may emit activation messages. Parse the final JSON object.
        start = text.rfind("\n{")
        candidate = text[start + 1 :] if start >= 0 else text[text.find("{") :]
        return json.loads(candidate)


def parse_worker_progress(line: str) -> dict:
    text = str(line).strip()
    if not text.startswith(PROGRESS_PREFIX):
        return {}
    try:
        value = json.loads(text[len(PROGRESS_PREFIX) :])
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


class SongFormerAnalyzer:
    """Cached subprocess adapter for the official ASLP-lab SongFormer model."""

    def __init__(
        self,
        model_name: str = "ASLP-lab/SongFormer",
        device: str = "auto",
        python_executable: str | Path | None = None,
        conda_env: str = "songformer-auto-dj",
        cache_path: str | Path | None = None,
        timeout_seconds: int = 7200,
    ) -> None:
        self.model_name = str(model_name)
        self.device = str(device)
        self.python_executable = _normalise_explicit_python(python_executable)
        self.conda_env = str(conda_env)
        self.timeout_seconds = int(timeout_seconds)
        self.cache_path = Path(
            cache_path or Path.home() / ".beatthis_muq_songformer_cache.json"
        )
        self._cache_lock = threading.RLock()
        self._cache = self._read_cache()

    def _read_cache(self) -> dict:
        try:
            if self.cache_path.exists():
                data = json.loads(self.cache_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
        except (OSError, json.JSONDecodeError):
            pass
        return {"version": 1, "tracks": {}}

    def _write_cache(self) -> None:
        with self._cache_lock:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            temp = self.cache_path.with_suffix(".tmp")
            temp.write_text(json.dumps(self._cache, ensure_ascii=False), encoding="utf-8")
            temp.replace(self.cache_path)

    def _cache_key(self, path: Path) -> str:
        stat = path.stat()
        return (
            f"{path.resolve()}::{stat.st_size}::{stat.st_mtime_ns}::"
            f"{self.model_name}::songformer-hf-v1"
        )

    @staticmethod
    def _convert(path: Path, payload: dict, model_name: str) -> SongFormerProfile:
        segments = tuple(
            FunctionalSegment.from_json_dict(item)
            for item in payload.get("segments", [])
            if float(item.get("end", 0.0)) > float(item.get("start", 0.0))
        )
        return SongFormerProfile(
            segments=segments,
            backend=str(payload.get("backend", "songformer-hf-official")),
            model_name=model_name,
            runtime_backend="isolated-worker",
        )

    def _command(self) -> list[str]:
        return songformer_command(self.python_executable, self.conda_env)

    def analyze_many(
        self,
        paths: Iterable[str | Path],
        status: StatusCallback | None = None,
        progress: ProgressCallback | None = None,
        force: bool = False,
    ) -> dict[str, SongFormerProfile]:
        resolved = [Path(path).expanduser().resolve() for path in paths]
        for path in resolved:
            if not path.is_file():
                raise FileNotFoundError(path)

        output: dict[str, SongFormerProfile] = {}
        todo: list[Path] = []
        total_count = len(resolved)
        completed_count = 0
        for path in resolved:
            cached = None if force else self._cache.get("tracks", {}).get(self._cache_key(path))
            if cached:
                output[str(path)] = SongFormerProfile.from_json_dict(cached)
                completed_count += 1
                if status:
                    status(f"SongFormer 缓存：{path.name}")
                if progress:
                    progress(completed_count, total_count, f"{path.name} · 缓存")
            else:
                todo.append(path)

        if not todo:
            return output

        command = self._command()
        worker = _project_worker()
        if not worker.is_file():
            raise FileNotFoundError(worker)
        if status:
            status(
                f"SongFormer 正在分析 {len(todo)} 首歌曲 · {self.device} · "
                f"{Path(command[-1]).name if len(command) == 1 else self.conda_env}"
            )

        with tempfile.TemporaryDirectory(prefix="autodj_songformer_") as directory:
            root = Path(directory)
            manifest_path = root / "manifest.json"
            output_path = root / "result.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "paths": [str(path) for path in todo],
                        "model_name": self.model_name,
                        "device": self.device,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            process = subprocess.Popen(
                command
                + [str(worker), "--manifest", str(manifest_path), "--output", str(output_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env={**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"},
            )
            recent_lines: list[str] = []
            timed_out = threading.Event()

            def terminate_on_timeout() -> None:
                timed_out.set()
                if process.poll() is None:
                    process.kill()

            watchdog = threading.Timer(self.timeout_seconds, terminate_on_timeout)
            watchdog.daemon = True
            watchdog.start()
            try:
                assert process.stdout is not None
                for raw_line in process.stdout:
                    line = raw_line.rstrip("\r\n")
                    if not line:
                        continue
                    event = parse_worker_progress(line)
                    if event:
                        message = str(event.get("message", "SongFormer 正在分析"))
                        worker_current = int(event.get("current", 0))
                        worker_total = max(int(event.get("total", len(todo))), 1)
                        mapped_current = (
                            -1
                            if worker_current < 0
                            else completed_count + min(worker_current, worker_total)
                        )
                        if status:
                            status(message)
                        if progress:
                            progress(mapped_current, total_count, message)
                        print(f"[SongFormer] {message}", flush=True)
                    else:
                        print(f"[SongFormer worker] {line}", flush=True)
                        recent_lines.append(line)
                        recent_lines = recent_lines[-30:]
                returncode = process.wait()
            finally:
                watchdog.cancel()
            if timed_out.is_set():
                raise RuntimeError(
                    f"SongFormer 推理超过 {self.timeout_seconds} 秒，已终止 worker。"
                )
            payload = (
                json.loads(output_path.read_text(encoding="utf-8"))
                if output_path.exists()
                else {}
            )
            if returncode != 0 and not payload.get("results"):
                detail = payload.get("fatal_error") or "\n".join(recent_lines[-12:])
                raise RuntimeError(
                    "SongFormer 推理失败。请运行 python verify_songformer.py。"
                    f"\n{detail}"
                )

        raw_results = payload.get("results", {})
        errors = payload.get("errors", {})
        with self._cache_lock:
            tracks_cache = self._cache.setdefault("tracks", {})
            for index, path in enumerate(todo):
                raw = raw_results.get(str(path))
                done = completed_count + index + 1
                if raw is None:
                    message = f"SongFormer 失败：{path.name} · {errors.get(str(path), '无结果')}"
                    if status:
                        status(message)
                    if progress:
                        progress(done, total_count, message)
                    continue
                profile = self._convert(path, raw, self.model_name)
                output[str(path)] = profile
                tracks_cache[self._cache_key(path)] = profile.to_json_dict()
                labels = ", ".join(profile.unique_labels[:6]) or "无标签"
                message = f"SongFormer 完成：{path.name} · {labels}"
                if status:
                    status(message)
                if progress:
                    progress(done, total_count, message)
        self._write_cache()
        return output

    def analyze(
        self,
        path: str | Path,
        status: StatusCallback | None = None,
        progress: ProgressCallback | None = None,
        force: bool = False,
    ) -> SongFormerProfile:
        resolved = str(Path(path).expanduser().resolve())
        return self.analyze_many(
            [resolved], status=status, progress=progress, force=force
        ).get(
            resolved, SongFormerProfile(backend="songformer-fallback")
        )


def probe_songformer(
    python_executable: str | Path | None = None,
    conda_env: str = "songformer-auto-dj",
) -> dict[str, object]:
    try:
        command = songformer_command(python_executable, conda_env)
        process = subprocess.run(
            command + [str(_project_worker()), "--probe"],
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
            env={**os.environ, "PYTHONUTF8": "1"},
        )
        payload = _extract_json(process.stdout)
        if process.returncode != 0:
            payload.setdefault("message", process.stderr.strip() or "probe failed")
            payload["ok"] = False
        payload["command"] = command
        return payload
    except Exception as exc:
        return {
            "ok": False,
            "message": str(exc),
            "command": [],
        }
