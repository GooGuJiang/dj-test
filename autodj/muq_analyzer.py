from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Callable

import librosa
import numpy as np

from .models import MuQProfile

StatusCallback = Callable[[str], None]


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1e-9:
        return np.zeros_like(vector, dtype=np.float32)
    return np.ascontiguousarray(vector / norm, dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return 0.0
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 1e-9:
        return 0.0
    # MuQ cosine can be negative. Map to a bounded perceptual-style score.
    cosine = float(np.dot(a, b) / denominator)
    return float(np.clip(0.5 + 0.5 * cosine, 0.0, 1.0))


class MuQAnalyzer:
    """
    Lazy MuQ feature extractor with compressed local cache.

    MuQ is a representation model rather than a calibrated genre classifier. This
    class therefore exposes global and time-local embeddings for style continuity,
    candidate ranking and graph-based playlist ordering. It deliberately does not
    invent genre labels.
    """

    def __init__(
        self,
        model_name: str = "OpenMuQ/MuQ-large-msd-iter",
        device: str = "cpu",
        cache_dir: str | Path | None = None,
        segment_seconds: float = 14.0,
        timeline_windows: int = 7,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.segment_seconds = float(np.clip(segment_seconds, 8.0, 30.0))
        self.timeline_windows = int(np.clip(timeline_windows, 3, 12))
        self.cache_dir = Path(
            cache_dir
            or Path.home() / ".cache" / "beatthis_auto_dj" / "muq_large_msd_iter"
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._model = None
        self._model_lock = threading.Lock()

    @staticmethod
    def _file_signature(path: Path) -> str:
        stat = path.stat()
        raw = f"{path.resolve()}::{stat.st_size}::{stat.st_mtime_ns}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def _cache_path(self, path: Path) -> Path:
        model_hash = hashlib.sha1(self.model_name.encode("utf-8")).hexdigest()[:10]
        return self.cache_dir / f"{self._file_signature(path)}_{model_hash}.npz"

    def _get_model(self):
        with self._model_lock:
            if self._model is not None:
                return self._model
            try:
                import torch
                from muq import MuQ
            except Exception as exc:  # pragma: no cover - depends on optional package
                raise RuntimeError(
                    "MuQ 尚未安装。请运行 python install_muq.py，或在 GUI 中关闭 MuQ。"
                ) from exc

            requested = self.device.lower().strip()
            if requested == "cuda" and not torch.cuda.is_available():
                requested = "cpu"
            if requested == "mps":
                available = bool(
                    getattr(torch.backends, "mps", None)
                    and torch.backends.mps.is_available()
                )
                if not available:
                    requested = "cpu"
            self.device = requested
            model = MuQ.from_pretrained(self.model_name)
            model = model.to(self.device).eval()
            self._model = model
            return model

    @staticmethod
    def _pool_output(output) -> np.ndarray:
        """
        Multi-layer MuQ-token aggregation inspired by recent large music encoder
        recommendation work: average the last four hidden layers, then perform
        temporal mean pooling and L2 normalization.
        """
        hidden_states = getattr(output, "hidden_states", None)
        if hidden_states:
            selected = hidden_states[-min(4, len(hidden_states)) :]
            hidden = sum(selected) / float(len(selected))
        else:
            hidden = output.last_hidden_state
        pooled = hidden.float().mean(dim=1)[0].detach().cpu().numpy()
        return _l2_normalize(pooled)

    def _embed_clip(self, waveform: np.ndarray) -> np.ndarray:
        import torch

        model = self._get_model()
        tensor = torch.from_numpy(np.ascontiguousarray(waveform, dtype=np.float32))
        tensor = tensor.unsqueeze(0).to(self.device)
        with torch.inference_mode():
            output = model(tensor, output_hidden_states=True)
        return self._pool_output(output)

    @staticmethod
    def _acoustic_features(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        if waveform.size == 0:
            return np.zeros(6, dtype=np.float32)
        hop = 512
        stft = np.abs(librosa.stft(waveform, n_fft=2048, hop_length=hop))
        rms = librosa.feature.rms(S=stft)[0]
        onset = librosa.onset.onset_strength(y=waveform, sr=sample_rate, hop_length=hop)
        centroid = librosa.feature.spectral_centroid(S=stft, sr=sample_rate)[0]
        flatness = librosa.feature.spectral_flatness(S=stft)[0]
        frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=2048)
        power = stft * stft
        low = np.sum(power[frequencies <= 180.0], axis=0) / (
            np.sum(power, axis=0) + 1e-9
        )
        dynamic = np.percentile(rms, 90) - np.percentile(rms, 10)
        values = np.asarray(
            [
                np.log10(float(np.mean(rms)) + 1e-8),
                float(np.mean(onset)) / 10.0,
                float(np.mean(centroid)) / (sample_rate / 2.0),
                float(np.mean(flatness)),
                float(np.mean(low)),
                float(dynamic),
            ],
            dtype=np.float32,
        )
        return np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=-1.0)

    def _load_cache(self, path: Path) -> MuQProfile | None:
        cache = self._cache_path(path)
        if not cache.exists():
            return None
        try:
            with np.load(cache, allow_pickle=False) as data:
                metadata = json.loads(str(data["metadata"].item()))
                if metadata.get("model_name") != self.model_name:
                    return None
                return MuQProfile(
                    global_embedding=np.asarray(data["global_embedding"], dtype=np.float32),
                    intro_embedding=np.asarray(data["intro_embedding"], dtype=np.float32),
                    outro_embedding=np.asarray(data["outro_embedding"], dtype=np.float32),
                    timeline_embeddings=np.asarray(data["timeline_embeddings"], dtype=np.float32),
                    timeline_positions=np.asarray(data["timeline_positions"], dtype=np.float32),
                    acoustic_features=np.asarray(data["acoustic_features"], dtype=np.float32),
                    backend=str(metadata.get("backend", "MuQ cache")),
                    model_name=self.model_name,
                )
        except Exception:
            return None

    def _write_cache(self, path: Path, profile: MuQProfile) -> None:
        cache = self._cache_path(path)
        temp = cache.with_suffix(".tmp.npz")
        metadata = json.dumps(
            {
                "model_name": self.model_name,
                "backend": profile.backend,
                "version": 2,
            },
            ensure_ascii=False,
        )
        np.savez_compressed(
            temp,
            metadata=np.asarray(metadata),
            global_embedding=profile.global_embedding.astype(np.float16),
            intro_embedding=profile.intro_embedding.astype(np.float16),
            outro_embedding=profile.outro_embedding.astype(np.float16),
            timeline_embeddings=profile.timeline_embeddings.astype(np.float16),
            timeline_positions=profile.timeline_positions.astype(np.float32),
            acoustic_features=profile.acoustic_features.astype(np.float32),
        )
        temp.replace(cache)

    def analyze(
        self,
        path: str | Path,
        status: StatusCallback | None = None,
        force: bool = False,
    ) -> MuQProfile:
        path = Path(path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if not force:
            cached = self._load_cache(path)
            if cached is not None:
                if status:
                    status(f"MuQ 缓存：{path.name}")
                return cached

        if status:
            status(f"MuQ 正在分析风格与段落语义：{path.name}")
        waveform, sample_rate = librosa.load(path, sr=24_000, mono=True, dtype=np.float32)
        if waveform.size < sample_rate:
            waveform = np.pad(waveform, (0, sample_rate - waveform.size))

        # Keep silent padding out of style windows while preserving path identity.
        non_silent, _ = librosa.effects.trim(waveform, top_db=48)
        if non_silent.size >= sample_rate:
            waveform = non_silent

        segment_length = min(
            len(waveform), max(sample_rate, int(round(self.segment_seconds * sample_rate)))
        )
        if len(waveform) <= segment_length:
            positions = np.asarray([0.5], dtype=np.float32)
            starts = np.asarray([0], dtype=np.int64)
        else:
            positions = np.linspace(0.05, 0.95, self.timeline_windows, dtype=np.float32)
            centers = positions * (len(waveform) - 1)
            starts = np.asarray(
                np.clip(centers - segment_length / 2.0, 0, len(waveform) - segment_length),
                dtype=np.int64,
            )
            starts = np.unique(starts)
            positions = ((starts + segment_length / 2.0) / len(waveform)).astype(np.float32)

        embeddings: list[np.ndarray] = []
        for index, start in enumerate(starts):
            if status:
                status(
                    f"MuQ {path.name}：语义窗口 {index + 1}/{len(starts)}"
                )
            clip = waveform[int(start) : int(start) + segment_length]
            embeddings.append(self._embed_clip(clip))

        timeline = np.stack(embeddings).astype(np.float32)
        global_embedding = _l2_normalize(np.mean(timeline, axis=0))
        profile = MuQProfile(
            global_embedding=global_embedding,
            intro_embedding=_l2_normalize(timeline[0]),
            outro_embedding=_l2_normalize(timeline[-1]),
            timeline_embeddings=timeline,
            timeline_positions=positions,
            acoustic_features=self._acoustic_features(waveform, sample_rate),
            backend=f"MuQ large / {self.device}",
            model_name=self.model_name,
        )
        self._write_cache(path, profile)
        if status:
            status(f"MuQ 分析完成：{path.name}")
        return profile
