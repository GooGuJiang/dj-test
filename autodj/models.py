from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TrackAnalysis:
    path: str
    title: str
    duration: float
    bpm: float
    beats_per_bar: int
    beat_times: tuple[float, ...]
    beat_numbers: tuple[int, ...]
    downbeat_times: tuple[float, ...]

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> "TrackAnalysis":
        return cls(
            path=str(data["path"]),
            title=str(data["title"]),
            duration=float(data["duration"]),
            bpm=float(data["bpm"]),
            beats_per_bar=int(data["beats_per_bar"]),
            beat_times=tuple(float(x) for x in data["beat_times"]),
            beat_numbers=tuple(int(x) for x in data["beat_numbers"]),
            downbeat_times=tuple(float(x) for x in data["downbeat_times"]),
        )


@dataclass(frozen=True)
class FunctionalSegment:
    start: float
    end: float
    label: str
    confidence: float = 1.0

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> "FunctionalSegment":
        return cls(
            start=float(data["start"]),
            end=float(data["end"]),
            label=str(data["label"]),
            confidence=float(data.get("confidence", 1.0)),
        )


@dataclass
class AllInOneProfile:
    """All-In-One functional structure analysis, aligned to original audio time."""

    bpm: float = 0.0
    beats: tuple[float, ...] = ()
    downbeats: tuple[float, ...] = ()
    beat_positions: tuple[int, ...] = ()
    segments: tuple[FunctionalSegment, ...] = ()
    backend: str = "disabled"
    model_name: str = "harmonix-all"
    natten_backend: str = "unknown"

    @property
    def available(self) -> bool:
        return bool(self.segments)

    @property
    def duration(self) -> float:
        if self.segments:
            return float(max(segment.end for segment in self.segments))
        if self.beats:
            return float(self.beats[-1])
        return 0.0

    @property
    def unique_labels(self) -> tuple[str, ...]:
        seen: list[str] = []
        for segment in self.segments:
            label = segment.label.lower()
            if label not in seen:
                seen.append(label)
        return tuple(seen)

    def label_at(self, seconds: float) -> str:
        value = float(max(0.0, seconds))
        for segment in self.segments:
            if segment.start <= value < segment.end:
                return segment.label.lower()
        if self.segments and value >= self.segments[-1].start:
            return self.segments[-1].label.lower()
        return "unknown"

    def boundaries(self) -> tuple[float, ...]:
        return tuple(float(segment.start) for segment in self.segments)

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "bpm": self.bpm,
            "beats": list(self.beats),
            "downbeats": list(self.downbeats),
            "beat_positions": list(self.beat_positions),
            "segments": [segment.to_json_dict() for segment in self.segments],
            "backend": self.backend,
            "model_name": self.model_name,
            "natten_backend": self.natten_backend,
        }

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> "AllInOneProfile":
        return cls(
            bpm=float(data.get("bpm", 0.0)),
            beats=tuple(float(value) for value in data.get("beats", [])),
            downbeats=tuple(float(value) for value in data.get("downbeats", [])),
            beat_positions=tuple(int(value) for value in data.get("beat_positions", [])),
            segments=tuple(
                FunctionalSegment.from_json_dict(item)
                for item in data.get("segments", [])
            ),
            backend=str(data.get("backend", "cache")),
            model_name=str(data.get("model_name", "harmonix-all")),
            natten_backend=str(data.get("natten_backend", "unknown")),
        )


@dataclass
class BarFeatures:
    """按重拍分割后的逐小节音乐特征。"""

    start_samples: np.ndarray
    end_samples: np.ndarray
    rms: np.ndarray
    low_ratio: np.ndarray
    onset: np.ndarray
    brightness: np.ndarray
    vocal_proxy: np.ndarray
    chroma: np.ndarray

    @property
    def count(self) -> int:
        return int(self.start_samples.size)


@dataclass
class EDMStructure:
    """论文驱动的 EDM 结构与 cue-point 分析结果。"""

    novelty: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    cue_score: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    mix_in_score: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    mix_out_score: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    salience: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    phrase_mask: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    labels: tuple[str, ...] = ()
    # Exact functional labels predicted by All-In-One, sampled per Beat This! bar.
    functional_labels: tuple[str, ...] = ()
    allin1_boundary_score: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float32)
    )
    structure_source: str = "local EDM heuristic"
    allin1_backend: str = "disabled"
    phase_offset: int = 0
    key_index: int = -1
    mode: str = "unknown"
    camelot: str = "—"
    edm_confidence: float = 0.0
    silence_start_sample: int = 0
    silence_end_sample: int = -1

    @property
    def cue_indices(self) -> np.ndarray:
        if self.cue_score.size == 0:
            return np.zeros(0, dtype=np.int64)
        return np.flatnonzero(self.cue_score >= 0.34).astype(np.int64)


@dataclass
class MuQProfile:
    """MuQ 音乐基础模型提取的全局与分段风格表示。"""

    global_embedding: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float32)
    )
    intro_embedding: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float32)
    )
    outro_embedding: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float32)
    )
    timeline_embeddings: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 0), dtype=np.float32)
    )
    timeline_positions: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float32)
    )
    acoustic_features: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float32)
    )
    backend: str = "disabled"
    model_name: str = "OpenMuQ/MuQ-large-msd-iter"

    @property
    def available(self) -> bool:
        return self.global_embedding.size > 0

    def embedding_at(self, relative_position: float) -> np.ndarray:
        if self.timeline_embeddings.size == 0 or self.timeline_positions.size == 0:
            return self.global_embedding
        index = int(
            np.argmin(
                np.abs(
                    self.timeline_positions
                    - float(np.clip(relative_position, 0.0, 1.0))
                )
            )
        )
        return self.timeline_embeddings[index]

    def sequence_around(self, relative_position: float, radius: int = 1) -> np.ndarray:
        if self.timeline_embeddings.size == 0 or self.timeline_positions.size == 0:
            if self.global_embedding.size:
                return self.global_embedding[None, :]
            return np.zeros((0, 0), dtype=np.float32)
        center = int(
            np.argmin(
                np.abs(
                    self.timeline_positions
                    - float(np.clip(relative_position, 0.0, 1.0))
                )
            )
        )
        start = max(0, center - max(0, int(radius)))
        end = min(len(self.timeline_embeddings), center + max(0, int(radius)) + 1)
        return self.timeline_embeddings[start:end]


@dataclass
class PreparedTrack:
    analysis: TrackAnalysis
    # 实际送入实时播放器的版本。可能包含同步段、BPM 恢复桥和原速尾段。
    audio: np.ndarray
    low_audio: np.ndarray
    mid_audio: np.ndarray
    high_audio: np.ndarray
    # 统一响度后的原速音频，用于构建 BPM 恢复桥。
    source_audio: np.ndarray
    sample_rate: int
    # 歌曲进入混音时的 BPM。
    playback_bpm: float
    # 恢复完成后的歌曲原 BPM（已处理半拍/双拍解释）。
    original_bpm: float
    stretch_rate: float
    beat_samples: np.ndarray
    beat_numbers: np.ndarray
    downbeat_samples: np.ndarray
    source_beat_samples: np.ndarray
    source_beat_numbers: np.ndarray
    source_downbeat_samples: np.ndarray
    cue_sample: int
    waveform_envelope: np.ndarray
    bar_features: BarFeatures
    tempo_restore_start: int = -1
    tempo_restore_end: int = -1
    tempo_restore_bars: int = 0
    structure: EDMStructure = field(default_factory=EDMStructure)
    muq_profile: MuQProfile = field(default_factory=MuQProfile)
    allin1_profile: AllInOneProfile = field(default_factory=AllInOneProfile)
    stretch_backend: str = "librosa"

    @property
    def total_samples(self) -> int:
        return int(self.audio.shape[0])

    @property
    def duration(self) -> float:
        return self.total_samples / float(self.sample_rate)

    @property
    def title(self) -> str:
        return self.analysis.title

    @property
    def has_tempo_restore(self) -> bool:
        return (
            self.tempo_restore_start >= 0
            and self.tempo_restore_end > self.tempo_restore_start
            and not np.isclose(self.playback_bpm, self.original_bpm, atol=0.05)
        )

    def bpm_at_sample(self, sample: int) -> float:
        """返回播放位置处的有效 BPM，用于 GUI 和后续歌曲同步。"""
        if not self.has_tempo_restore or sample <= self.tempo_restore_start:
            return float(self.playback_bpm)
        if sample >= self.tempo_restore_end:
            return float(self.original_bpm)
        progress = (sample - self.tempo_restore_start) / max(
            self.tempo_restore_end - self.tempo_restore_start,
            1,
        )
        progress = float(np.clip(progress, 0.0, 1.0))
        # 与渲染器一致的五次缓动 + 几何 BPM 插值。端点斜率为 0，
        # GUI 显示不会在恢复开始/结束时出现速度突跳。
        smooth = progress**3 * (progress * (progress * 6.0 - 15.0) + 10.0)
        if self.original_bpm <= 0 or self.playback_bpm <= 0:
            return float(self.original_bpm)
        ratio = self.playback_bpm / self.original_bpm
        return float(self.original_bpm * np.exp(np.log(ratio) * (1.0 - smooth)))


@dataclass
class TransitionPlan:
    current_start: int
    next_start: int
    length: int
    bars: int
    current_bar_index: int
    next_bar_index: int
    score: float
    # 中频/主体的等功率曲线。
    fade_out: np.ndarray
    fade_in: np.ndarray
    # 低频交换曲线。
    bass_out: np.ndarray
    bass_in: np.ndarray
    # 高频滤波扫频的近似增益曲线。
    high_out: np.ndarray
    high_in: np.ndarray
    # 预渲染的节拍同步 echo-out。
    echo_audio: np.ndarray
    metrics: dict[str, float] = field(default_factory=dict)
    automatic: bool = True
    style: str = "Club"
    effect_strength: float = 0.7
    # 频时域谱缝合结果；存在时实时回调直接播放该缓冲区。
    rendered_audio: np.ndarray | None = None
    transition_mode: str = "EQ/Fader"
    policy_mode: str = "AutoMix-like"
    # 过渡内部由 A 主导切换为 B 主导的位置，0~1。
    switch_position: float = 0.55
    # 当过渡片段经过精确小节时间扭曲时，过渡结束后应从 B 的原始
    # 同步时间轴此位置继续播放，而不是简单 next_start + length。
    next_resume_sample: int = -1
    micro_offset_samples: int = 0
    # Human-style candidate generation and reference-aware quality selection.
    human_archetype: str = ""
    human_quality_score: float = 0.0
    human_quality_metrics: dict[str, float] = field(default_factory=dict)

    @property
    def current_end(self) -> int:
        return self.current_start + self.length

    @property
    def next_end(self) -> int:
        if self.next_resume_sample >= 0:
            return self.next_resume_sample
        return self.next_start + self.length

    @property
    def switch_sample_a(self) -> int:
        return self.current_start + int(round(self.length * self.switch_position))

    @property
    def switch_sample_b(self) -> int:
        return self.next_start + int(round(self.length * self.switch_position))
