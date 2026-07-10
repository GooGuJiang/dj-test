from __future__ import annotations

import json
import threading
from collections import Counter
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf

from .models import TrackAnalysis

StatusCallback = Callable[[str], None]


def _estimate_bpm(beat_times: np.ndarray) -> float:
    intervals = np.diff(np.asarray(beat_times, dtype=np.float64))
    intervals = intervals[np.isfinite(intervals)]
    intervals = intervals[(intervals >= 0.22) & (intervals <= 1.8)]
    if intervals.size < 4:
        raise RuntimeError("Beat This! 检测到的有效节拍不足，无法估计 BPM。")

    # 截尾中位数对少量漏拍和额外峰值更稳定。
    center = float(np.median(intervals))
    keep = intervals[(intervals >= center * 0.72) & (intervals <= center * 1.35)]
    if keep.size >= 4:
        center = float(np.median(keep))
    bpm = 60.0 / center
    while bpm < 70.0:
        bpm *= 2.0
    while bpm > 190.0:
        bpm *= 0.5
    return float(bpm)


def _regularize_steady_grid(
    beats: np.ndarray,
    downbeats: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    对稳定节拍歌曲做保守的网格修复。

    Beat This! 不使用 DBN，优点是不会强行限制速度与拍号；但在长电子乐中偶发
    漏掉一个 beat 会让后续 phrase 对齐整体偏移。本函数只在局部间隔足够稳定时：
    1. 合并距离过近的重复峰；2. 填补明显的单个/少量漏拍；3. 将 downbeat 吸附
    到最近 beat。速度变化明显的歌曲不会触发填补。
    """
    beats = np.asarray(beats, dtype=np.float64)
    downbeats = np.asarray(downbeats, dtype=np.float64)
    beats = np.unique(beats[np.isfinite(beats) & (beats >= 0.0)])
    if beats.size < 8:
        return beats, downbeats

    diffs = np.diff(beats)
    median = float(np.median(diffs[(diffs > 0.18) & (diffs < 1.8)]))
    if not np.isfinite(median) or median <= 0:
        return beats, downbeats

    # 先去除极近重复峰。
    merged: list[float] = [float(beats[0])]
    for value in beats[1:]:
        if value - merged[-1] < 0.42 * median:
            merged[-1] = 0.5 * (merged[-1] + float(value))
        else:
            merged.append(float(value))
    beats = np.asarray(merged, dtype=np.float64)

    intervals = np.diff(beats)
    normalized = intervals / median
    close = normalized[(normalized >= 0.70) & (normalized <= 1.35)]
    cv = float(np.std(close)) if close.size else 1.0
    # 仅对 House/Techno/EDM 一类稳定网格做补拍。
    if close.size >= max(6, intervals.size // 2) and cv < 0.12:
        repaired: list[float] = [float(beats[0])]
        for left, right in zip(beats[:-1], beats[1:]):
            gap = float(right - left)
            multiple = int(round(gap / median))
            if 2 <= multiple <= 4 and abs(gap / multiple - median) <= 0.16 * median:
                for step in range(1, multiple):
                    repaired.append(float(left + step * gap / multiple))
            repaired.append(float(right))
        beats = np.asarray(repaired, dtype=np.float64)

    # Beat This! 的 downbeat 通常已经在 beat 上；修复后重新吸附可避免浮点误差。
    snapped: list[float] = []
    for value in downbeats:
        if not np.isfinite(value) or value < 0 or beats.size == 0:
            continue
        index = int(np.argmin(np.abs(beats - value)))
        if abs(float(beats[index] - value)) <= max(0.09, 0.28 * median):
            snapped.append(float(beats[index]))
    return np.unique(beats), np.unique(np.asarray(snapped, dtype=np.float64))


def _infer_numbers_fallback(
    beats: np.ndarray,
    downbeats: np.ndarray,
) -> np.ndarray:
    """Beat This! 1.0 或异常输出时的兼容 beat number 推断。"""
    if beats.size == 0:
        return np.zeros(0, dtype=np.int32)
    downbeat_indices = np.unique(np.searchsorted(beats, downbeats)).astype(int)
    downbeat_indices = downbeat_indices[
        (downbeat_indices >= 0) & (downbeat_indices < beats.size)
    ]
    intervals = np.diff(downbeat_indices)
    intervals = intervals[(intervals >= 2) & (intervals <= 8)]
    meter = int(round(float(np.median(intervals)))) if intervals.size else 4
    meter = int(np.clip(meter, 2, 8))

    numbers = np.empty(beats.size, dtype=np.int32)
    if downbeat_indices.size:
        first = int(downbeat_indices[0])
        for index in range(beats.size):
            previous = downbeat_indices[downbeat_indices <= index]
            if previous.size:
                numbers[index] = ((index - int(previous[-1])) % meter) + 1
            else:
                numbers[index] = ((index - first) % meter) + 1
    else:
        numbers[:] = (np.arange(beats.size) % meter) + 1
    return numbers


def _infer_meter(numbers: np.ndarray, downbeats: np.ndarray, beats: np.ndarray) -> int:
    if downbeats.size >= 2 and beats.size:
        indices = np.searchsorted(beats, downbeats)
        counts = np.diff(indices)
        counts = counts[(counts >= 2) & (counts <= 8)]
        if counts.size:
            counter = Counter(int(value) for value in counts)
            return int(counter.most_common(1)[0][0])
    positive = numbers[(numbers >= 1) & (numbers <= 8)]
    if positive.size:
        return int(np.clip(np.max(positive), 2, 8))
    return 4


class BeatThisAnalyzer:
    """使用 Beat This! 1.1+ 进行离线 beat/downbeat 分析并缓存结果。"""

    CACHE_VERSION = 2

    def __init__(
        self,
        checkpoint: str = "final0",
        device: str = "cpu",
        float16: bool | None = None,
        cache_path: str | Path | None = None,
    ) -> None:
        self.checkpoint = checkpoint.strip() or "final0"
        self.device = device.strip() or "cpu"
        self.float16 = bool(float16) if float16 is not None else self.device.startswith("cuda")
        self.cache_path = Path(
            cache_path or Path.home() / ".beat_this_auto_dj_cache_v2.json"
        )
        self._cache_lock = threading.Lock()
        self._cache = self._read_cache()
        self._model = None

    def _read_cache(self) -> dict:
        try:
            if self.cache_path.exists():
                data = json.loads(self.cache_path.read_text(encoding="utf-8"))
                if int(data.get("version", 0)) == self.CACHE_VERSION:
                    return data
        except (OSError, ValueError, json.JSONDecodeError):
            pass
        return {"version": self.CACHE_VERSION, "tracks": {}}

    def _write_cache(self) -> None:
        with self._cache_lock:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.cache_path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(self._cache, ensure_ascii=False),
                encoding="utf-8",
            )
            temporary.replace(self.cache_path)

    def _cache_key(self, path: Path) -> str:
        stat = path.stat()
        return (
            f"beat-this:{self.checkpoint}:{path.resolve()}::"
            f"{stat.st_size}::{stat.st_mtime_ns}"
        )

    @staticmethod
    def _duration(path: Path) -> float:
        try:
            return float(sf.info(str(path)).duration)
        except RuntimeError:
            import librosa

            return float(librosa.get_duration(path=str(path)))

    def _get_model(self):
        if self._model is not None:
            return self._model
        try:
            from beat_this.inference import File2Beats
        except Exception as exc:
            raise RuntimeError(
                "无法导入 Beat This!。请运行 python install_beat_this.py，"
                "然后执行 python verify_install.py。"
            ) from exc

        try:
            self._model = File2Beats(
                checkpoint_path=self.checkpoint,
                device=self.device,
                float16=self.float16,
                dbn=False,
            )
        except Exception as exc:
            raise RuntimeError(
                f"无法加载 Beat This! 模型 {self.checkpoint}。首次使用需要联网下载权重；"
                "也可以先运行 python verify_install.py 完成下载。"
            ) from exc
        return self._model

    def analyze(
        self,
        path: str | Path,
        status: StatusCallback | None = None,
        force: bool = False,
    ) -> TrackAnalysis:
        path = Path(path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)

        key = self._cache_key(path)
        if not force:
            cached = self._cache.get("tracks", {}).get(key)
            if cached:
                if status:
                    status(f"读取 Beat This! 缓存：{path.name}")
                return TrackAnalysis.from_json_dict(cached)

        if status:
            status(f"Beat This! 正在分析：{path.name}")
        model = self._get_model()

        output = model(str(path))
        if not isinstance(output, tuple) or len(output) != 2:
            raise RuntimeError("Beat This! 返回格式异常。")
        beats = np.asarray(output[0], dtype=np.float64).reshape(-1)
        downbeats = np.asarray(output[1], dtype=np.float64).reshape(-1)
        beats, downbeats = _regularize_steady_grid(beats, downbeats)

        if beats.size < 8:
            raise RuntimeError(f"Beat This! 未能从 {path.name} 提取足够节拍。")

        try:
            from beat_this.utils import infer_beat_numbers

            numbers = np.asarray(
                infer_beat_numbers(beats, downbeats), dtype=np.int32
            )
        except Exception:
            numbers = _infer_numbers_fallback(beats, downbeats)

        if numbers.size != beats.size:
            numbers = _infer_numbers_fallback(beats, downbeats)
        beats_per_bar = _infer_meter(numbers, downbeats, beats)

        # 当 downbeat 过少时，以推断出的第 1 拍补足，保证下游 phrase 搜索可用。
        if downbeats.size < 2:
            downbeats = beats[numbers == 1]
        if downbeats.size < 2:
            downbeats = beats[::beats_per_bar]
            numbers = (np.arange(beats.size, dtype=np.int32) % beats_per_bar) + 1

        bpm = _estimate_bpm(beats)
        result = TrackAnalysis(
            path=str(path),
            title=path.stem,
            duration=self._duration(path),
            bpm=bpm,
            beats_per_bar=beats_per_bar,
            beat_times=tuple(float(value) for value in beats),
            beat_numbers=tuple(int(value) for value in numbers),
            downbeat_times=tuple(float(value) for value in downbeats),
        )

        with self._cache_lock:
            self._cache.setdefault("tracks", {})[key] = result.to_json_dict()
        self._write_cache()

        if status:
            status(
                f"Beat This! 完成：{path.name} | {result.bpm:.1f} BPM | "
                f"{result.beats_per_bar}/4 | {self.checkpoint}"
            )
        return result
