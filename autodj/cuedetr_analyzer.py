from __future__ import annotations

import hashlib
import json
import logging
import threading
from pathlib import Path
from typing import Callable, Iterable

import librosa
import numpy as np
from scipy.signal import find_peaks

from .compute_device import resolve_torch_device
from .models import CueDETRProfile, TrackAnalysis

StatusCallback = Callable[[str], None]
ProgressCallback = Callable[[float, float, str], None]
LOGGER = logging.getLogger("autodj.cuedetr")

# Constants copied from the official ETH-DISCO example implementation.
_OVERLAP = 0.75
_WINDOW_WIDTH = 355
_PADDING = 266
_SAMPLE_RATE = 22_050
_HOP_LENGTH = 512
_ANALYSIS_LOCK = threading.Lock()


class CueDETRAnalyzer:
    """Cached adapter for the official ``disco-eth/cue-detr`` checkpoint.

    The neural model is the sole cue-point generator. Beat This! is used only to
    quantize predictions to the nearest downbeat and to enforce a minimum phrase
    distance. Local novelty/energy rules never create extra cue candidates.
    """

    def __init__(
        self,
        model_name: str = "disco-eth/cue-detr",
        device: str = "auto",
        sensitivity: float = 0.90,
        min_bars: int = 8,
        batch_size: int = 6,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.model_name = str(model_name).strip() or "disco-eth/cue-detr"
        requested = str(device).strip().lower() or "auto"
        self.device = resolve_torch_device(
            requested, strict_cuda=requested.startswith("cuda")
        )
        self.sensitivity = float(np.clip(sensitivity, 0.20, 0.995))
        self.min_bars = int(np.clip(min_bars, 1, 64))
        self.batch_size = int(np.clip(batch_size, 1, 32))
        self.cache_dir = Path(
            cache_dir
            or Path.home() / ".cache" / "beatthis_auto_dj" / "cue_detr"
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._processor = None
        self._model = None
        self._model_lock = threading.Lock()

    @staticmethod
    def _file_signature(path: Path) -> str:
        stat = path.stat()
        raw = f"{path.resolve()}::{stat.st_size}::{stat.st_mtime_ns}".encode()
        return hashlib.sha256(raw).hexdigest()

    def _cache_path(self, path: Path, analysis: TrackAnalysis) -> Path:
        config = (
            f"{self.model_name}::{self.sensitivity:.5f}::{self.min_bars}::"
            f"{analysis.bpm:.5f}::{analysis.beats_per_bar}::v3"
        )
        suffix = hashlib.sha1(config.encode()).hexdigest()[:12]
        return self.cache_dir / f"{self._file_signature(path)}_{suffix}.json"

    def _load_cache(
        self, path: Path, analysis: TrackAnalysis
    ) -> CueDETRProfile | None:
        cache = self._cache_path(path, analysis)
        try:
            if cache.exists():
                profile = CueDETRProfile.from_json_dict(
                    json.loads(cache.read_text(encoding="utf-8"))
                )
                if profile.model_name == self.model_name:
                    profile.backend = "CUE-DETR cache"
                    return profile
        except (OSError, ValueError, json.JSONDecodeError):
            pass
        return None

    def _write_cache(
        self, path: Path, analysis: TrackAnalysis, profile: CueDETRProfile
    ) -> None:
        cache = self._cache_path(path, analysis)
        temporary = cache.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(profile.to_json_dict(), ensure_ascii=False),
            encoding="utf-8",
        )
        temporary.replace(cache)

    def _get_model(self):
        with self._model_lock:
            if self._model is not None and self._processor is not None:
                return self._processor, self._model
            try:
                import torch
                from transformers import DetrForObjectDetection, DetrImageProcessor
            except Exception as exc:  # pragma: no cover - optional dependency
                raise RuntimeError(
                    "CUE-DETR 尚未安装。请运行 python install_cuedetr.py。"
                ) from exc

            processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
            model = DetrForObjectDetection.from_pretrained(self.model_name)
            model = model.to(self.device).eval()
            self._processor = processor
            self._model = model
            LOGGER.info("CUE-DETR 模型已加载 · %s · %s", self.device, self.model_name)
            return processor, model

    @staticmethod
    def _spectrogram_image(path: Path) -> tuple[np.ndarray, float]:
        # The official implementation uses librosa's 22.05 kHz default and a
        # power Mel spectrogram converted to a viridis RGB image.
        waveform, _ = librosa.load(path, sr=_SAMPLE_RATE, mono=True, dtype=np.float32)
        duration = len(waveform) / float(_SAMPLE_RATE)
        mel = librosa.feature.melspectrogram(
            y=waveform,
            sr=_SAMPLE_RATE,
            n_fft=2048,
            hop_length=_HOP_LENGTH,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        try:
            from matplotlib import colormaps

            rgba = colormaps["viridis"](
                np.clip((mel_db[::-1] - mel_db.min()) / max(np.ptp(mel_db), 1e-9), 0, 1),
                bytes=True,
            )
        except Exception:
            from matplotlib import cm

            mapper = cm.ScalarMappable(cmap="viridis")
            mapper.set_clim(float(mel_db.min()), float(mel_db.max()))
            rgba = mapper.to_rgba(mel_db[::-1], bytes=True)
        return np.ascontiguousarray(rgba[:, :, :3], dtype=np.uint8), duration

    @staticmethod
    def _windows(image: np.ndarray) -> tuple[list[np.ndarray], list[int]]:
        image_width = image.shape[1] + _PADDING
        count = max(
            1,
            int(np.floor(image_width / (_WINDOW_WIDTH * (1.0 - _OVERLAP)))),
        )
        images: list[np.ndarray] = []
        borders: list[int] = []
        for index in range(count):
            left = int(np.floor(index * _WINDOW_WIDTH * (1.0 - _OVERLAP))) - _PADDING
            right = left + _WINDOW_WIDTH
            borders.append(left)
            if left < 0:
                segment = image[:, : max(0, right)]
                segment = np.pad(
                    segment,
                    ((0, 0), (-left, 0), (0, 0)),
                    mode="linear_ramp",
                )
            elif right > image.shape[1]:
                segment = image[:, left:]
                segment = np.pad(
                    segment,
                    ((0, 0), (0, right - image.shape[1]), (0, 0)),
                    mode="linear_ramp",
                )
            else:
                segment = image[:, left:right]
            if segment.shape[1] != _WINDOW_WIDTH:
                segment = np.pad(
                    segment,
                    ((0, 0), (0, max(0, _WINDOW_WIDTH - segment.shape[1])), (0, 0)),
                    mode="linear_ramp",
                )[:, :_WINDOW_WIDTH]
            images.append(np.ascontiguousarray(segment, dtype=np.uint8))
        return images, borders

    def _predict_raw(
        self,
        path: Path,
        status: StatusCallback | None,
        progress: Callable[[float, str], None] | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        import torch

        processor, model = self._get_model()
        image, duration = self._spectrogram_image(path)
        images, borders = self._windows(image)
        positions: list[int] = []
        scores: list[float] = []
        total_batches = int(np.ceil(len(images) / self.batch_size))

        for batch_index, start in enumerate(range(0, len(images), self.batch_size)):
            batch = images[start : start + self.batch_size]
            batch_borders = borders[start : start + self.batch_size]
            if status:
                status(
                    f"CUE-DETR {path.name}：滑窗 {batch_index + 1}/{total_batches}"
                )
            encoding = processor.preprocess(batch, do_resize=False, return_tensors="pt")
            pixel_values = encoding["pixel_values"].to(self.device)
            pixel_mask = encoding.get("pixel_mask")
            if pixel_mask is not None:
                pixel_mask = pixel_mask.to(self.device)
            with torch.inference_mode():
                outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask)
            targets = [(128, _WINDOW_WIDTH)] * len(batch)
            predictions = processor.post_process_object_detection(
                outputs, threshold=0.0, target_sizes=targets
            )
            for prediction, left in zip(predictions, batch_borders):
                boxes = prediction["boxes"].detach().cpu().numpy()
                local_scores = prediction["scores"].detach().cpu().numpy()
                centers = np.floor((boxes[:, 0] + boxes[:, 2]) * 0.5).astype(np.int64)
                positions.extend((centers + int(left)).tolist())
                scores.extend(local_scores.astype(float).tolist())
            del pixel_values, outputs
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()
            if progress:
                progress(
                    0.18 + 0.72 * ((batch_index + 1) / max(total_batches, 1)),
                    f"滑窗 {batch_index + 1}/{total_batches}",
                )

        if not positions:
            return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
        positions_array = np.asarray(positions, dtype=np.int64)
        scores_array = np.asarray(scores, dtype=np.float64)
        valid = (
            (positions_array >= 0)
            & (positions_array < image.shape[1])
            & np.isfinite(scores_array)
        )
        positions_array = positions_array[valid]
        scores_array = scores_array[valid]
        if positions_array.size == 0:
            return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)

        # Overlapping windows produce duplicate detections. Keep the maximum score
        # for each spectrogram frame before reproducing the official peak filter.
        unique_positions = np.unique(positions_array)
        merged_scores = np.asarray(
            [np.max(scores_array[positions_array == position]) for position in unique_positions],
            dtype=np.float64,
        )
        low = float(np.min(merged_scores))
        high = float(np.max(merged_scores))
        if high > low + 1e-12:
            merged_scores = (merged_scores - low) / (high - low)
        else:
            merged_scores = np.zeros_like(merged_scores)
        peak_indices, properties = find_peaks(
            merged_scores,
            height=self.sensitivity,
            distance=16,
        )
        selected_positions = unique_positions[peak_indices]
        selected_scores = np.asarray(properties.get("peak_heights", []), dtype=np.float64)
        raw_times = librosa.frames_to_time(
            selected_positions,
            sr=_SAMPLE_RATE,
            hop_length=_HOP_LENGTH,
        )
        valid_times = (raw_times >= 0.0) & (raw_times <= duration + 0.25)
        return raw_times[valid_times], selected_scores[valid_times]

    def _snap_and_filter(
        self,
        raw_times: np.ndarray,
        raw_scores: np.ndarray,
        analysis: TrackAnalysis,
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        downbeats = np.asarray(analysis.downbeat_times, dtype=np.float64)
        if downbeats.size == 0:
            downbeats = np.asarray(analysis.beat_times, dtype=np.float64)
        if downbeats.size == 0:
            return (), ()

        by_index: dict[int, float] = {}
        for time_value, score in zip(raw_times, raw_scores):
            index = int(np.argmin(np.abs(downbeats - float(time_value))))
            by_index[index] = max(by_index.get(index, 0.0), float(score))

        # The official CLI exposes a minimum distance between cues. Enforce it in
        # actual Beat This! bars rather than in DETR query order.
        selected: list[tuple[int, float]] = []
        minimum = max(1, self.min_bars)
        for index, score in sorted(by_index.items(), key=lambda item: item[1], reverse=True):
            if all(abs(index - previous) >= minimum for previous, _ in selected):
                selected.append((index, score))
        selected.sort()

        # Endpoint safety is only used when the neural model returns no usable
        # cue. It does not reintroduce the removed novelty/energy cue detector.
        if not selected:
            first = 0
            last = max(0, len(downbeats) - max(2, self.min_bars))
            selected = [(first, 0.36)]
            if last > first:
                selected.append((last, 0.36))

        times = tuple(float(downbeats[index]) for index, _ in selected)
        scores = tuple(float(score) for _, score in selected)
        return times, scores

    def analyze(
        self,
        analysis: TrackAnalysis,
        status: StatusCallback | None = None,
        progress: Callable[[float, str], None] | None = None,
        force: bool = False,
    ) -> CueDETRProfile:
        path = Path(analysis.path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if not force:
            cached = self._load_cache(path, analysis)
            if cached is not None:
                if status:
                    status(f"CUE-DETR 缓存：{path.name} · {len(cached.cue_times)} 个 cue")
                if progress:
                    progress(1.0, f"缓存 · {path.name}")
                return cached

        if status:
            status(f"CUE-DETR 正在搜索专业 cue 点：{path.name}")
        if progress:
            progress(0.05, "Mel 频谱")
        raw_times, raw_scores = self._predict_raw(path, status, progress)
        cue_times, cue_scores = self._snap_and_filter(raw_times, raw_scores, analysis)
        profile = CueDETRProfile(
            cue_times=cue_times,
            cue_scores=cue_scores,
            raw_cue_times=tuple(float(x) for x in raw_times),
            raw_cue_scores=tuple(float(x) for x in raw_scores),
            backend=f"CUE-DETR/{self.device}",
            model_name=self.model_name,
            sensitivity=self.sensitivity,
            min_bars=self.min_bars,
        )
        self._write_cache(path, analysis, profile)
        if progress:
            progress(1.0, f"{len(profile.cue_times)} 个 cue")
        if status:
            status(f"CUE-DETR 完成：{path.name} · {len(profile.cue_times)} 个 cue")
        return profile

    def analyze_many(
        self,
        analyses: Iterable[TrackAnalysis],
        status: StatusCallback | None = None,
        progress: ProgressCallback | None = None,
        force: bool = False,
    ) -> dict[str, CueDETRProfile]:
        items = list(analyses)
        output: dict[str, CueDETRProfile] = {}
        total = len(items)
        with _ANALYSIS_LOCK:
            for index, analysis in enumerate(items):
                profile = self.analyze(
                    analysis,
                    status=status,
                    progress=(
                        lambda fraction, detail, i=index: progress(
                            i + float(fraction), total, detail
                        )
                        if progress
                        else None
                    ),
                    force=force,
                )
                output[analysis.path] = profile
        return output


def probe_cuedetr() -> dict[str, object]:
    """Validate the exact classes used by CUE-DETR.

    Importing the concrete DETR classes catches broken torchvision/Transformers
    installations early. The diagnostic deliberately distinguishes a missing
    package from an import conflict introduced by another optional backend.
    """
    try:
        import torch
        import transformers
        from PIL import Image  # noqa: F401
        from transformers import DetrForObjectDetection, DetrImageProcessor  # noqa: F401

        return {
            "ok": True,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": bool(torch.cuda.is_available()),
            "message": "CUE-DETR 依赖已安装；首次分析会下载官方权重。",
        }
    except ModuleNotFoundError as exc:
        return {
            "ok": False,
            "error_type": "missing_dependency",
            "message": f"缺少依赖：{exc.name or exc}",
        }
    except Exception as exc:
        return {
            "ok": False,
            "error_type": "import_conflict",
            "message": f"依赖导入冲突：{type(exc).__name__}: {exc}",
        }
