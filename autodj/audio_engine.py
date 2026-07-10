from __future__ import annotations

import math
import queue
import threading
from collections import OrderedDict
from dataclasses import dataclass, replace
from typing import Any, Sequence

import librosa
import numpy as np
try:
    import sounddevice as sd
except ImportError:  # 允许在无声卡/测试环境中导入匹配与 DSP 模块。
    sd = None  # type: ignore[assignment]
from scipy import signal

from .edm_structure import analyze_edm_structure
from .songformer_structure import fuse_songformer_structure
from .human_transition import (
    HumanTransitionConfig,
    SUPPORTED_ARCHETYPES,
    render_human_transition,
)
from .models import SongFormerProfile, MuQProfile, PreparedTrack, TrackAnalysis, TransitionPlan
from .muq_analyzer import MuQAnalyzer
from .spectral_seam import spectral_seam_crossfade
from .time_stretch import (
    stretch_stereo as quality_time_stretch,
    stretch_stereo_time_map,
)
from .transition_matcher import (
    MatcherConfig,
    TransitionFXConfig,
    extract_bar_features,
    find_best_transition,
    waveform_envelope,
)


@dataclass
class EngineConfig:
    sample_rate: int = 44_100
    channels: int = 2
    # 0 表示由匹配器在 4/8/16/32 小节中自动选择。
    crossfade_bars: int = 0
    max_stretch_percent: float = 12.0
    cue_on_first_downbeat: bool = True
    target_rms_db: float = -18.0
    bass_split_hz: float = 180.0
    high_split_hz: float = 6_000.0
    # 混音后用多少小节把下一首从同步 BPM 恢复到其原 BPM。0 表示关闭。
    # -1 表示根据 BPM 差和 EDM 置信度自动选择 4/8/16/32 小节。
    tempo_restore_bars: int = -1
    mix_style: str = "Club"
    effect_strength: float = 0.72
    limiter_drive: float = 1.18
    # Adaptive: high-confidence EDM uses graph-cut spectral seam, otherwise EQ/fader.
    transition_engine: str = "Adaptive"
    # AutoMix-like: complex DJ transition / simple crossfade / gapless trim are selected by score.
    automix_policy: str = "AutoMix-like"
    time_stretch_backend: str = "auto"
    min_complex_score: float = 0.60
    # MuQ semantic/style embeddings are optional but enabled by default.
    muq_enabled: bool = True
    muq_device: str = "auto"
    muq_model_name: str = "OpenMuQ/MuQ-large-msd-iter"
    # SongFormer provides functional structure boundaries/labels. Beat This!
    # remains the timing authority so both outputs share one stable beat grid.
    songformer_enabled: bool = True
    songformer_device: str = "auto"
    songformer_model_name: str = "ASLP-lab/SongFormer"
    # GUI performs batch analysis before playback. Keeping runtime inference off
    # prevents SongFormer inference from unexpectedly blocking next-track preload.
    songformer_runtime_inference: bool = False
    # Generate several context-aware DJ techniques and select the best with
    # reference-aware transition quality metrics.
    human_style_mode: str = "Adaptive Human"
    human_variation: float = 0.58
    human_max_candidates: int = 5
    # Sliding preload window: current + hot next + warm future tracks.
    preload_window_tracks: int = 3
    preload_memory_mb: int = 1024
    preload_deadline_seconds: float = 60.0
    latency: str = "high"


def related_bpm(source_bpm: float, target_bpm: float) -> float:
    candidates = np.asarray(
        [source_bpm * 0.5, source_bpm, source_bpm * 2.0],
        dtype=np.float64,
    )
    distance = np.abs(np.log2(candidates / target_bpm))
    return float(candidates[int(np.argmin(distance))])


def _smoothstep(value: float | np.ndarray) -> float | np.ndarray:
    clipped = np.clip(value, 0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def _smootherstep(value: float | np.ndarray) -> float | np.ndarray:
    """五次缓动：端点的一阶、二阶导数均为 0，适合 BPM 自动化。"""
    clipped = np.clip(value, 0.0, 1.0)
    return clipped**3 * (clipped * (clipped * 6.0 - 15.0) + 10.0)


def _to_stereo(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return np.column_stack((audio, audio)).astype(np.float32, copy=False)
    if audio.shape[0] <= 8:
        audio = audio[:2].T
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    elif audio.shape[1] > 2:
        audio = audio[:, :2]
    return np.ascontiguousarray(audio, dtype=np.float32)


def _normalize_audio(audio: np.ndarray, target_db: float) -> np.ndarray:
    rms = float(np.sqrt(np.mean(np.square(audio), dtype=np.float64) + 1e-12))
    target = 10.0 ** (target_db / 20.0)
    gain = float(np.clip(target / max(rms, 1e-8), 0.25, 4.0))
    audio = audio * gain
    peak = float(np.max(np.abs(audio)) + 1e-9)
    if peak > 0.98:
        audio = audio * (0.98 / peak)
    return np.ascontiguousarray(audio, dtype=np.float32)


def _nearest_zero_crossing(
    audio: np.ndarray, sample: int, radius: int
) -> int:
    """Snap a join point to the nearest mono zero crossing to reduce clicks."""
    if audio.size == 0:
        return 0
    sample = int(np.clip(sample, 0, len(audio) - 1))
    start = max(1, sample - max(1, radius))
    end = min(len(audio) - 1, sample + max(1, radius))
    mono = np.mean(audio[start - 1 : end + 1], axis=1, dtype=np.float64)
    crossings = np.flatnonzero(np.signbit(mono[:-1]) != np.signbit(mono[1:])) + start
    if crossings.size == 0:
        return sample
    return int(crossings[np.argmin(np.abs(crossings - sample))])


def _split_three_bands(
    audio: np.ndarray,
    sample_rate: int,
    bass_cutoff_hz: float,
    high_cutoff_hz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """互补式三段分频，三段相加严格还原原音频。"""
    nyquist = sample_rate / 2.0
    low_norm = float(np.clip(bass_cutoff_hz / nyquist, 0.001, 0.90))
    high_norm = float(np.clip(high_cutoff_hz / nyquist, low_norm + 0.01, 0.98))

    low_sos = signal.butter(4, low_norm, btype="lowpass", output="sos")
    low = signal.sosfilt(low_sos, audio, axis=0).astype(np.float32)
    remainder = np.ascontiguousarray(audio - low, dtype=np.float32)

    high_sos = signal.butter(4, high_norm, btype="highpass", output="sos")
    high = signal.sosfilt(high_sos, remainder, axis=0).astype(np.float32)
    mid = np.ascontiguousarray(remainder - high, dtype=np.float32)
    return np.ascontiguousarray(low), mid, np.ascontiguousarray(high)


def _stretch_stereo(
    audio: np.ndarray,
    rate: float,
    sample_rate: int = 44_100,
    backend: str = "auto",
) -> np.ndarray:
    stretched, _ = quality_time_stretch(
        audio,
        sample_rate=sample_rate,
        rate=rate,
        backend=backend,
    )
    return stretched


def _append_crossfade(
    base: np.ndarray,
    addition: np.ndarray,
    overlap: int,
) -> tuple[np.ndarray, int]:
    """拼接立体声音频，返回新数组和 addition 在结果中的起点。"""
    if base.size == 0:
        return np.ascontiguousarray(addition, dtype=np.float32), 0
    if addition.size == 0:
        return np.ascontiguousarray(base, dtype=np.float32), len(base)
    overlap = int(max(0, min(overlap, len(base), len(addition))))
    if overlap == 0:
        start = len(base)
        return np.ascontiguousarray(np.concatenate([base, addition], axis=0)), start

    phase = np.linspace(0.0, np.pi / 2.0, overlap, dtype=np.float32)
    mixed = (
        base[-overlap:] * np.cos(phase)[:, None]
        + addition[:overlap] * np.sin(phase)[:, None]
    )
    start = len(base) - overlap
    result = np.concatenate(
        [base[:-overlap], mixed, addition[overlap:]],
        axis=0,
    )
    return np.ascontiguousarray(result, dtype=np.float32), start


def _unique_grid(
    samples: list[int],
    numbers: list[int],
    total_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    pairs = sorted(
        {
            (int(sample), int(number))
            for sample, number in zip(samples, numbers)
            if 0 <= sample < total_samples
        },
        key=lambda pair: pair[0],
    )
    if not pairs:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int32)
    return (
        np.asarray([pair[0] for pair in pairs], dtype=np.int64),
        np.asarray([pair[1] for pair in pairs], dtype=np.int32),
    )


class AutoDJEngine:
    """Beat This! 重拍驱动、连续 BPM 回归与专业过渡的实时 DJ 引擎。"""

    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig()
        self.matcher_config = MatcherConfig()
        self._lock = threading.RLock()
        self._events: queue.SimpleQueue[str] = queue.SimpleQueue()

        self._playlist: list[TrackAnalysis] = []
        self._index = -1
        self._current: PreparedTrack | None = None
        self._next: PreparedTrack | None = None
        # 未加入 BPM 恢复桥的同步版，用于重新规划。
        self._next_synced_base: PreparedTrack | None = None
        self._plan: TransitionPlan | None = None

        self._current_pos = 0
        self._next_pos = 0
        self._transition_pos = 0
        self._transitioning = False

        self._stream: sd.OutputStream | None = None
        self._playing = False
        self._paused = False
        self._auto_mix = True
        self._master_volume = 0.9
        self._device: int | str | None = None

        self._next_loader: threading.Thread | None = None
        self._next_loading = False
        self._loader_generation = 0
        self._last_callback_status = ""
        self._muq_analyzer: MuQAnalyzer | None = None
        # Recent chosen techniques are used to avoid machine-like repetition.
        self._transition_history: list[str] = []

        # 播放前后台预加载的第一对歌曲。预加载只保存“当前 + 下一首”，
        # 因此不会随着队列长度线性占用内存。
        self._prime_generation = 0
        self._prime_loading = False
        self._prime_key: tuple[Any, ...] | None = None
        self._prime_current: PreparedTrack | None = None
        self._prime_next_synced_base: PreparedTrack | None = None
        self._prime_next: PreparedTrack | None = None
        self._prime_plan: TransitionPlan | None = None

        # 外部 MuQ 排序器已经分析过的 profile 可直接注入，避免音频引擎
        # 在准备下一首时再次运行大模型。
        self._preloaded_muq_profiles: dict[str, MuQProfile] = {}
        self._preloaded_songformer_profiles: dict[str, SongFormerProfile] = {}

        # Sliding-window preloader. The immediate next track is rendered hot in
        # self._next; later tracks are kept as synchronized warm PreparedTrack
        # objects so model inference, decoding and time-stretching do not start at
        # the last moment. OrderedDict gives deterministic memory-bounded LRU.
        self._warm_cache: OrderedDict[tuple[Any, ...], PreparedTrack] = OrderedDict()
        self._warm_cache_bytes = 0
        self._warm_loading_paths: set[str] = set()
        self._preload_urgent = False
        self._urgent_notice_index = -1

        # seek 后的短淡入，避免从非零采样位置突然起音产生 click。
        self._seek_fade_remaining = 0
        self._seek_fade_total = max(64, int(0.030 * self.config.sample_rate))

    def emit(self, message: str) -> None:
        self._events.put(message)

    def drain_events(self) -> list[str]:
        messages: list[str] = []
        while True:
            try:
                messages.append(self._events.get_nowait())
            except queue.Empty:
                break
        return messages

    @staticmethod
    def output_devices() -> list[tuple[int, str]]:
        if sd is None:
            return []
        devices = sd.query_devices()
        result: list[tuple[int, str]] = []
        for index, item in enumerate(devices):
            if int(item.get("max_output_channels", 0)) > 0:
                hostapi = sd.query_hostapis(int(item["hostapi"]))["name"]
                result.append((index, f"{item['name']} ({hostapi})"))
        return result

    def _get_muq_profile(self, analysis: TrackAnalysis) -> MuQProfile:
        if not self.config.muq_enabled:
            return MuQProfile()
        cached = self._preloaded_muq_profiles.get(analysis.path)
        if cached is not None and cached.available:
            return cached
        try:
            if self._muq_analyzer is None:
                self._muq_analyzer = MuQAnalyzer(
                    model_name=self.config.muq_model_name,
                    device=self.config.muq_device,
                )
            return self._muq_analyzer.analyze(analysis.path, status=self.emit)
        except Exception as exc:
            self.emit(f"MuQ 回退到传统特征：{analysis.title} · {exc}")
            return MuQProfile(backend="fallback")

    def _get_songformer_profile(self, analysis: TrackAnalysis) -> SongFormerProfile:
        if not self.config.songformer_enabled:
            return SongFormerProfile()
        cached = self._preloaded_songformer_profiles.get(analysis.path)
        if cached is not None and cached.available:
            return cached
        # SongFormer is deliberately batch-preloaded in its isolated worker.
        # Never launch a 0.7B structure model from the real-time preload path.
        self.emit(f"SongFormer 尚未预分析：{analysis.title}，使用本地结构估计")
        return SongFormerProfile(backend="not-preloaded")

    def _finish_prepared_track(
        self,
        *,
        analysis: TrackAnalysis,
        audio: np.ndarray,
        source_audio: np.ndarray,
        playback_bpm: float,
        original_bpm: float,
        stretch_rate: float,
        beat_samples: np.ndarray,
        beat_numbers: np.ndarray,
        downbeat_samples: np.ndarray,
        source_beat_samples: np.ndarray,
        source_beat_numbers: np.ndarray,
        source_downbeat_samples: np.ndarray,
        cue_sample: int,
        tempo_restore_start: int = -1,
        tempo_restore_end: int = -1,
        tempo_restore_bars: int = 0,
        muq_profile: MuQProfile | None = None,
        songformer_profile: SongFormerProfile | None = None,
        stretch_backend: str = "librosa",
    ) -> PreparedTrack:
        low, mid, high = _split_three_bands(
            audio,
            self.config.sample_rate,
            self.config.bass_split_hz,
            self.config.high_split_hz,
        )
        features = extract_bar_features(
            audio=audio,
            sample_rate=self.config.sample_rate,
            downbeat_samples=downbeat_samples,
            bpm=original_bpm if tempo_restore_end >= 0 else playback_bpm,
            beats_per_bar=max(2, analysis.beats_per_bar),
            feature_sample_rate=self.matcher_config.feature_sample_rate,
        )
        structure = analyze_edm_structure(
            audio=audio,
            sample_rate=self.config.sample_rate,
            features=features,
            beats_per_bar=max(2, analysis.beats_per_bar),
        )
        songformer_profile = songformer_profile or SongFormerProfile()
        if songformer_profile.available:
            structure = fuse_songformer_structure(
                base=structure,
                profile=songformer_profile,
                features=features,
                source_downbeat_samples=source_downbeat_samples,
                sample_rate=self.config.sample_rate,
            )
        envelope = waveform_envelope(audio, bins=self.matcher_config.waveform_bins)
        return PreparedTrack(
            analysis=analysis,
            audio=np.ascontiguousarray(audio, dtype=np.float32),
            low_audio=low,
            mid_audio=mid,
            high_audio=high,
            source_audio=np.ascontiguousarray(source_audio, dtype=np.float32),
            sample_rate=self.config.sample_rate,
            playback_bpm=float(playback_bpm),
            original_bpm=float(original_bpm),
            stretch_rate=float(stretch_rate),
            beat_samples=np.asarray(beat_samples, dtype=np.int64),
            beat_numbers=np.asarray(beat_numbers, dtype=np.int32),
            downbeat_samples=np.asarray(downbeat_samples, dtype=np.int64),
            source_beat_samples=np.asarray(source_beat_samples, dtype=np.int64),
            source_beat_numbers=np.asarray(source_beat_numbers, dtype=np.int32),
            source_downbeat_samples=np.asarray(source_downbeat_samples, dtype=np.int64),
            cue_sample=int(cue_sample),
            waveform_envelope=envelope,
            bar_features=features,
            tempo_restore_start=int(tempo_restore_start),
            tempo_restore_end=int(tempo_restore_end),
            tempo_restore_bars=int(tempo_restore_bars),
            structure=structure,
            muq_profile=muq_profile or MuQProfile(),
            songformer_profile=songformer_profile,
            stretch_backend=stretch_backend,
        )

    def prepare_track(
        self,
        analysis: TrackAnalysis,
        target_bpm: float | None = None,
    ) -> PreparedTrack:
        audio_cf, _ = librosa.load(
            analysis.path,
            sr=self.config.sample_rate,
            mono=False,
            dtype=np.float32,
        )
        source_audio = _normalize_audio(_to_stereo(audio_cf), self.config.target_rms_db)

        source_beat_samples_all = np.asarray(
            np.rint(np.asarray(analysis.beat_times) * self.config.sample_rate),
            dtype=np.int64,
        )
        source_beat_numbers_all = np.asarray(analysis.beat_numbers, dtype=np.int32)
        valid_source_beats = (
            (source_beat_samples_all >= 0)
            & (source_beat_samples_all < source_audio.shape[0])
        )
        source_beat_samples = source_beat_samples_all[valid_source_beats]
        if source_beat_numbers_all.size == source_beat_samples_all.size:
            source_beat_numbers = source_beat_numbers_all[valid_source_beats]
        else:
            source_beat_numbers = source_beat_numbers_all[: source_beat_samples.size]
            if source_beat_numbers.size < source_beat_samples.size:
                source_beat_numbers = np.pad(
                    source_beat_numbers,
                    (0, source_beat_samples.size - source_beat_numbers.size),
                    constant_values=0,
                )

        source_downbeats = np.asarray(
            np.rint(np.asarray(analysis.downbeat_times) * self.config.sample_rate),
            dtype=np.int64,
        )
        source_downbeats = source_downbeats[
            (source_downbeats >= 0) & (source_downbeats < source_audio.shape[0])
        ]

        stretch_rate = 1.0
        playback_bpm = float(analysis.bpm)
        original_bpm = float(analysis.bpm)
        if target_bpm and target_bpm > 0:
            interpreted = related_bpm(analysis.bpm, target_bpm)
            original_bpm = float(interpreted)
            proposed_rate = target_bpm / interpreted
            limit = self.config.max_stretch_percent / 100.0
            if abs(proposed_rate - 1.0) <= limit:
                stretch_rate = float(proposed_rate)
                playback_bpm = float(interpreted * stretch_rate)
            else:
                self.emit(
                    f"速度差过大，{analysis.title} 不强制同步："
                    f"{interpreted:.1f} → {target_bpm:.1f} BPM"
                )

        audio, stretch_backend = quality_time_stretch(
            source_audio,
            sample_rate=self.config.sample_rate,
            rate=stretch_rate,
            backend=self.config.time_stretch_backend,
        )
        if not math.isclose(stretch_rate, 1.0, abs_tol=1e-4):
            self.emit(
                f"已同步 {analysis.title}：{original_bpm:.1f} → "
                f"{playback_bpm:.1f} BPM · {stretch_backend}"
            )

        scale = 1.0 / stretch_rate
        beat_samples = np.asarray(np.rint(source_beat_samples * scale), dtype=np.int64)
        valid = (beat_samples >= 0) & (beat_samples < audio.shape[0])
        beat_samples = beat_samples[valid]
        beat_numbers = source_beat_numbers[valid]
        downbeat_samples = np.asarray(
            np.rint(source_downbeats * scale),
            dtype=np.int64,
        )
        downbeat_samples = downbeat_samples[
            (downbeat_samples >= 0) & (downbeat_samples < audio.shape[0])
        ]

        cue_sample = 0
        if self.config.cue_on_first_downbeat and downbeat_samples.size:
            candidates = downbeat_samples[
                downbeat_samples >= int(0.25 * self.config.sample_rate)
            ]
            if candidates.size:
                cue_sample = int(candidates[0])

        muq_profile = self._get_muq_profile(analysis)
        songformer_profile = self._get_songformer_profile(analysis)
        return self._finish_prepared_track(
            analysis=analysis,
            audio=audio,
            source_audio=source_audio,
            playback_bpm=playback_bpm,
            original_bpm=original_bpm,
            stretch_rate=stretch_rate,
            beat_samples=beat_samples,
            beat_numbers=beat_numbers,
            downbeat_samples=downbeat_samples,
            source_beat_samples=source_beat_samples,
            source_beat_numbers=source_beat_numbers,
            source_downbeat_samples=source_downbeats,
            cue_sample=cue_sample,
            muq_profile=muq_profile,
            songformer_profile=songformer_profile,
            stretch_backend=stretch_backend,
        )

    def _source_bar_boundaries(self, track: PreparedTrack) -> np.ndarray:
        downbeats = np.unique(track.source_downbeat_samples)
        if downbeats.size >= 2:
            return downbeats
        bar_length = int(
            round(
                max(2, track.analysis.beats_per_bar)
                * 60.0
                / max(track.original_bpm, 1.0)
                * track.sample_rate
            )
        )
        return np.arange(
            0,
            track.source_audio.shape[0],
            max(bar_length, track.sample_rate),
            dtype=np.int64,
        )

    def _apply_tempo_restore(
        self,
        synced: PreparedTrack,
        restore_start: int,
    ) -> PreparedTrack:
        """
        使用一条连续 source->target 时间映射恢复原 BPM。

        旧实现逐小节单独 time-stretch 后用 12ms crossfade 拼接，电子鼓和低频会
        在每个小节边界产生相位/瞬态变化。新实现一次渲染整首歌曲：恢复前保持
        同步速度，恢复区用五次缓动逐小节改变局部速度，恢复后保持原速。
        """
        bars_requested = int(self.config.tempo_restore_bars)
        if math.isclose(synced.stretch_rate, 1.0, abs_tol=1e-3):
            return synced
        if bars_requested < 0:
            difference = abs(float(synced.stretch_rate) - 1.0)
            if difference <= 0.008:
                bars_requested = 4
            elif difference <= 0.020:
                bars_requested = 8
            elif difference <= 0.040:
                bars_requested = 16
            else:
                bars_requested = 32
            # 稳定电子音乐对 kick/bass 的速度自动化更敏感，至少留 16 小节。
            if synced.structure.edm_confidence >= 0.55 and difference > 0.015:
                bars_requested = max(16, bars_requested)
        if (
            bars_requested <= 0
            or restore_start >= synced.total_samples - synced.sample_rate
        ):
            return synced

        source_length = int(synced.source_audio.shape[0])
        initial_rate = float(synced.stretch_rate)
        source_start = int(round(restore_start * initial_rate))
        source_start = int(np.clip(source_start, 0, max(0, source_length - 1)))

        boundaries = self._source_bar_boundaries(synced)
        boundaries = np.unique(boundaries[(boundaries > source_start) & (boundaries <= source_length)])
        if boundaries.size == 0:
            return synced

        endpoints = boundaries[:bars_requested]
        if endpoints.size == 0:
            return synced
        bars = int(endpoints.size)

        # 保证恢复开始前的时间坐标与原同步版本完全一致，因而已有 IN/MIX
        # 位置不需要重新计算。
        keyframes: list[tuple[int, int]] = [(0, 0)]
        if source_start > 0:
            keyframes.append((source_start, int(restore_start)))

        source_cursor = source_start
        target_cursor = float(restore_start)
        log_initial_rate = math.log(max(initial_rate, 1e-6))

        for index, source_end in enumerate(endpoints):
            source_end = int(source_end)
            if source_end <= source_cursor:
                continue
            midpoint = (index + 0.5) / max(bars, 1)
            eased = float(_smootherstep(midpoint))
            # BPM/速度用几何插值，比线性 rate 插值的听感更均匀。
            local_rate = math.exp(log_initial_rate * (1.0 - eased))
            target_cursor += (source_end - source_cursor) / max(local_rate, 1e-6)
            keyframes.append((source_end, int(round(target_cursor))))
            source_cursor = source_end

        if source_cursor >= source_length:
            return synced
        # 恢复结束后 1:1 映射，原速尾段与恢复桥在同一次渲染中连续生成。
        keyframes.append(
            (source_length, int(round(target_cursor + source_length - source_cursor)))
        )

        rendered, backend, source_map, target_map = stretch_stereo_time_map(
            synced.source_audio,
            sample_rate=synced.sample_rate,
            keyframes=keyframes,
            backend=self.config.time_stretch_backend,
        )
        if rendered.size == 0:
            return synced

        def map_samples(values: np.ndarray) -> np.ndarray:
            values = np.asarray(values, dtype=np.float64)
            mapped = np.interp(values, source_map, target_map)
            return np.asarray(np.rint(mapped), dtype=np.int64)

        beat_grid = map_samples(synced.source_beat_samples)
        valid_beats = (beat_grid >= 0) & (beat_grid < len(rendered))
        beat_grid = beat_grid[valid_beats]
        number_grid = synced.source_beat_numbers[: valid_beats.size]
        if synced.source_beat_numbers.size == valid_beats.size:
            number_grid = synced.source_beat_numbers[valid_beats]
        elif number_grid.size < beat_grid.size:
            number_grid = np.pad(
                number_grid,
                (0, beat_grid.size - number_grid.size),
                constant_values=0,
            )

        downbeat_grid = map_samples(synced.source_downbeat_samples)
        downbeat_grid = np.unique(
            downbeat_grid[(downbeat_grid >= 0) & (downbeat_grid < len(rendered))]
        )
        source_cue = int(round(synced.cue_sample * initial_rate))
        cue_sample = int(
            round(float(np.interp(source_cue, source_map, target_map)))
        )

        restore_end_source = int(endpoints[-1])
        restore_end = int(
            round(float(np.interp(restore_end_source, source_map, target_map)))
        )
        restore_start_mapped = int(
            round(float(np.interp(source_start, source_map, target_map)))
        )

        result = self._finish_prepared_track(
            analysis=synced.analysis,
            audio=rendered,
            source_audio=synced.source_audio,
            playback_bpm=synced.playback_bpm,
            original_bpm=synced.original_bpm,
            stretch_rate=synced.stretch_rate,
            beat_samples=beat_grid,
            beat_numbers=number_grid,
            downbeat_samples=downbeat_grid,
            source_beat_samples=synced.source_beat_samples,
            source_beat_numbers=synced.source_beat_numbers,
            source_downbeat_samples=synced.source_downbeat_samples,
            cue_sample=cue_sample,
            tempo_restore_start=restore_start_mapped,
            tempo_restore_end=restore_end,
            tempo_restore_bars=bars,
            muq_profile=synced.muq_profile,
            songformer_profile=synced.songformer_profile,
            stretch_backend=backend,
        )
        self.emit(
            f"连续 BPM 恢复：{synced.playback_bpm:.1f} → "
            f"{synced.original_bpm:.1f} BPM / {bars} 小节 · {backend}"
        )
        return result

    def _render_echo(self, plan: TransitionPlan, current: PreparedTrack) -> np.ndarray:
        strength = float(np.clip(self.config.effect_strength, 0.0, 1.0))
        style = self.config.mix_style.lower()
        style_amount = {
            "smooth": 0.08,
            "club": 0.20,
            "filter": 0.12,
            "echo": 0.42,
        }.get(style, 0.20)
        amount = style_amount * strength
        if amount <= 1e-4 or plan.length < 256:
            return np.zeros((plan.length, 2), dtype=np.float32)

        segment = (
            current.mid_audio[plan.current_start : plan.current_end]
            + current.high_audio[plan.current_start : plan.current_end]
        )
        if len(segment) < plan.length:
            segment = np.pad(segment, ((0, plan.length - len(segment)), (0, 0)))
        segment = segment[: plan.length]

        phase = np.linspace(0.0, 1.0, plan.length, dtype=np.float32)
        send_start = 0.58 if style == "echo" else 0.68
        send = np.asarray(_smoothstep((phase - send_start) / (1.0 - send_start)), dtype=np.float32)
        source = segment * send[:, None] * np.float32(amount)

        delay = int(
            round(
                current.sample_rate
                * 60.0
                / max(current.bpm_at_sample(plan.current_start), 1.0)
                * 0.5
            )
        )
        delay = max(64, delay)
        feedback = 0.43 if style == "echo" else 0.31
        echo = np.zeros((plan.length, 2), dtype=np.float32)
        for repeat in range(1, 6):
            shift = delay * repeat
            if shift >= plan.length:
                break
            contribution = source[: plan.length - shift] * np.float32(feedback ** (repeat - 1))
            if repeat % 2:
                contribution = contribution[:, ::-1]
            echo[shift:] += contribution

        tail_shape = np.sin(np.pi * np.clip((phase - 0.45) / 0.55, 0.0, 1.0))
        return np.ascontiguousarray(echo * tail_shape[:, None], dtype=np.float32)

    def _preload_config_signature(self) -> tuple[Any, ...]:
        return (
            self.config.sample_rate,
            self.config.crossfade_bars,
            round(self.config.max_stretch_percent, 3),
            self.config.tempo_restore_bars,
            self.config.mix_style,
            round(self.config.effect_strength, 3),
            self.config.transition_engine,
            self.config.automix_policy,
            self.config.time_stretch_backend,
            self.config.muq_enabled,
            self.config.muq_device,
            self.config.songformer_enabled,
            self.config.songformer_device,
            self.config.songformer_model_name,
            self.config.preload_window_tracks,
            self.config.preload_memory_mb,
        )

    def _pair_key(
        self, playlist: Sequence[TrackAnalysis], start_index: int
    ) -> tuple[Any, ...]:
        current_path = playlist[start_index].path
        next_path = (
            playlist[start_index + 1].path
            if start_index + 1 < len(playlist)
            else ""
        )
        return (
            current_path,
            next_path,
            self._preload_config_signature(),
        )

    @staticmethod
    def _prepared_track_bytes(track: PreparedTrack) -> int:
        """Approximate unique NumPy memory retained by a PreparedTrack."""
        total = 0
        seen: set[int] = set()
        for value in vars(track).values():
            if isinstance(value, np.ndarray):
                pointer = int(value.__array_interface__["data"][0]) if value.size else id(value)
                if pointer not in seen:
                    seen.add(pointer)
                    total += int(value.nbytes)
        return total

    def _warm_key(
        self,
        outgoing: TrackAnalysis,
        incoming: TrackAnalysis,
        target_bpm: float,
    ) -> tuple[Any, ...]:
        return (
            outgoing.path,
            incoming.path,
            round(float(target_bpm), 3),
            self._preload_config_signature(),
        )

    def _take_warm_track(
        self,
        outgoing: TrackAnalysis,
        incoming: TrackAnalysis,
        target_bpm: float,
    ) -> PreparedTrack | None:
        key = self._warm_key(outgoing, incoming, target_bpm)
        with self._lock:
            track = self._warm_cache.pop(key, None)
            if track is not None:
                self._warm_cache_bytes = max(
                    0, self._warm_cache_bytes - self._prepared_track_bytes(track)
                )
        if track is not None:
            self.emit(f"滑动窗口命中：{incoming.title} 已完成解码与变速预热")
        return track

    def _store_warm_track(
        self,
        outgoing: TrackAnalysis,
        incoming: TrackAnalysis,
        target_bpm: float,
        track: PreparedTrack,
    ) -> None:
        key = self._warm_key(outgoing, incoming, target_bpm)
        size = self._prepared_track_bytes(track)
        budget = max(256, int(self.config.preload_memory_mb)) * 1024 * 1024
        with self._lock:
            old = self._warm_cache.pop(key, None)
            if old is not None:
                self._warm_cache_bytes = max(
                    0, self._warm_cache_bytes - self._prepared_track_bytes(old)
                )
            # Keep at least one warm item when possible. Evict oldest pairs first.
            while self._warm_cache and self._warm_cache_bytes + size > budget:
                _, evicted = self._warm_cache.popitem(last=False)
                self._warm_cache_bytes = max(
                    0, self._warm_cache_bytes - self._prepared_track_bytes(evicted)
                )
            if size <= budget:
                self._warm_cache[key] = track
                self._warm_cache_bytes += size

    def _window_token_valid(self, kind: str, generation: int) -> bool:
        with self._lock:
            if kind == "prime":
                return generation == self._prime_generation and not self._playing
            return generation == self._loader_generation and self._playing

    def _warm_following_tracks(
        self,
        snapshot: Sequence[TrackAnalysis],
        first_target_index: int,
        source_bpm: float,
        generation: int,
        token_kind: str,
    ) -> None:
        """Prepare the warm portion of the sliding window in priority order."""
        # Window count includes current and immediate hot next; the remaining
        # slots are warm synchronized bases.
        warm_count = max(0, int(self.config.preload_window_tracks) - 2)
        stop = min(len(snapshot), first_target_index + warm_count)
        previous_bpm = float(source_bpm)
        for target_index in range(first_target_index, stop):
            if not self._window_token_valid(token_kind, generation):
                return
            outgoing = snapshot[target_index - 1]
            incoming = snapshot[target_index]
            key = self._warm_key(outgoing, incoming, previous_bpm)
            with self._lock:
                cached = self._warm_cache.get(key)
                if cached is not None:
                    self._warm_cache.move_to_end(key)
                    previous_bpm = cached.original_bpm
                    continue
                self._warm_loading_paths.add(incoming.path)
            try:
                self.emit(
                    f"滑动窗口预热 {target_index + 1}/{len(snapshot)}：{incoming.title}"
                )
                warmed = self.prepare_track(incoming, target_bpm=previous_bpm)
                if not self._window_token_valid(token_kind, generation):
                    return
                self._store_warm_track(outgoing, incoming, previous_bpm, warmed)
                previous_bpm = warmed.original_bpm
            except Exception as exc:
                self.emit(f"暖轨预加载失败：{incoming.title} · {exc}")
                return
            finally:
                with self._lock:
                    self._warm_loading_paths.discard(incoming.path)

    def set_preloaded_muq_profiles(
        self, profiles: dict[str, MuQProfile]
    ) -> None:
        """注入 GUI 已完成的 MuQ 结果，避免引擎重复推理。"""
        with self._lock:
            self._preloaded_muq_profiles = dict(profiles)

    def set_preloaded_songformer_profiles(
        self, profiles: dict[str, SongFormerProfile]
    ) -> None:
        """注入 GUI 已完成的 SongFormer 结构结果，避免播放时重复推理。"""
        with self._lock:
            self._preloaded_songformer_profiles = dict(profiles)

    def clear_preload(self) -> None:
        with self._lock:
            self._prime_generation += 1
            self._prime_loading = False
            self._prime_key = None
            self._prime_current = None
            self._prime_next_synced_base = None
            self._prime_next = None
            self._prime_plan = None
            self._warm_cache.clear()
            self._warm_cache_bytes = 0
            self._warm_loading_paths.clear()
            self._preload_urgent = False

    def preload_pair(
        self,
        playlist: Sequence[TrackAnalysis],
        start_index: int = 0,
    ) -> bool:
        """后台完整准备第一首和下一首。

        包括 MuQ 缓存读取、音频加载、结构特征、BPM 同步、切点搜索、
        BPM 恢复桥和过渡渲染。用户点击播放时可直接复用结果。
        """
        if not playlist or not 0 <= start_index < len(playlist):
            return False
        key = self._pair_key(playlist, start_index)
        with self._lock:
            if self._playing:
                return False
            if self._prime_key == key and (
                self._prime_loading or self._prime_current is not None
            ):
                return True
            self._prime_generation += 1
            generation = self._prime_generation
            self._prime_loading = True
            self._prime_key = key
            self._prime_current = None
            self._prime_next_synced_base = None
            self._prime_next = None
            self._prime_plan = None
            snapshot = list(playlist)

        def worker() -> None:
            try:
                self.emit(f"后台预加载：{snapshot[start_index].title}")
                current = self.prepare_track(snapshot[start_index])
                synced_base: PreparedTrack | None = None
                prepared: PreparedTrack | None = None
                plan: TransitionPlan | None = None
                if start_index + 1 < len(snapshot):
                    incoming = snapshot[start_index + 1]
                    self.emit(f"后台预分析下一首：{incoming.title}")
                    synced_base = self.prepare_track(
                        incoming, target_bpm=current.original_bpm
                    )
                    plan = self._calculate_plan(
                        current,
                        synced_base,
                        earliest_start=int(0.25 * self.config.sample_rate),
                    )
                    prepared = self._apply_tempo_restore(
                        synced_base, plan.next_end
                    )
                    plan = self._render_advanced_transition(
                        plan, current, prepared
                    )
                with self._lock:
                    if (
                        generation == self._prime_generation
                        and self._prime_key == key
                        and not self._playing
                    ):
                        self._prime_current = current
                        self._prime_next_synced_base = synced_base
                        self._prime_next = prepared
                        self._prime_plan = plan
                        self.emit("热下一轨预加载完成：点击播放可直接开始")
                if prepared is not None and start_index + 2 < len(snapshot):
                    self._warm_following_tracks(
                        snapshot=snapshot,
                        first_target_index=start_index + 2,
                        source_bpm=prepared.original_bpm,
                        generation=generation,
                        token_kind="prime",
                    )
            except Exception as exc:
                self.emit(f"后台预加载失败，将在播放时重试：{exc}")
            finally:
                with self._lock:
                    if generation == self._prime_generation:
                        self._prime_loading = False

        threading.Thread(
            target=worker,
            daemon=True,
            name="AutoDJ-PairPreloader",
        ).start()
        return True

    def start_playlist(
        self,
        playlist: Sequence[TrackAnalysis],
        start_index: int = 0,
        device: int | str | None = None,
    ) -> None:
        if not playlist:
            raise ValueError("播放列表为空。")
        if not 0 <= start_index < len(playlist):
            raise IndexError("start_index 超出播放列表范围。")

        if sd is None:
            raise RuntimeError(
                "缺少 sounddevice。请在 beatthis-auto-dj 环境中安装 "
                "python-sounddevice/portaudio 后再启动实时播放。"
            )

        self.stop(clear_preload=False)
        key = self._pair_key(playlist, start_index)
        with self._lock:
            use_prime = self._prime_key == key and self._prime_current is not None
            if use_prime:
                current = self._prime_current
                primed_synced = self._prime_next_synced_base
                primed_next = self._prime_next
                primed_plan = self._prime_plan
            else:
                current = None
                primed_synced = None
                primed_next = None
                primed_plan = None
            # 取消仍在进行的旧预加载，但保留刚刚复制出的对象。
            self._prime_generation += 1
            self._prime_loading = False
            self._prime_key = None
            self._prime_current = None
            self._prime_next_synced_base = None
            self._prime_next = None
            self._prime_plan = None

        if current is None:
            self.emit(f"正在提取当前歌曲的智能匹配特征：{playlist[start_index].title}")
            current = self.prepare_track(playlist[start_index])
        else:
            self.emit("使用后台预加载结果启动播放")

        with self._lock:
            self._playlist = list(playlist)
            self._index = start_index
            self._current = current
            self._next = primed_next
            self._next_synced_base = primed_synced
            self._plan = primed_plan
            self._current_pos = 0
            self._next_pos = 0
            self._transition_pos = 0
            self._transitioning = False
            self._playing = True
            self._paused = False
            self._device = device
            self._loader_generation += 1

        stream = sd.OutputStream(
            samplerate=self.config.sample_rate,
            blocksize=0,
            device=device,
            channels=self.config.channels,
            dtype="float32",
            latency=self.config.latency,
            callback=self._audio_callback,
        )
        stream.start()
        with self._lock:
            self._stream = stream

        self.emit(f"开始播放：{current.title}")
        self.service()

    def _fx_config(self) -> TransitionFXConfig:
        return TransitionFXConfig(
            style=self.config.mix_style,
            strength=self.config.effect_strength,
        )

    @staticmethod
    def _neutral_curves(length: int, gain: float = 1.0) -> tuple[np.ndarray, ...]:
        phase = np.linspace(0.0, np.pi / 2.0, max(1, length), dtype=np.float32)
        fade_out = np.cos(phase).astype(np.float32)
        fade_in = (np.sin(phase) * np.float32(gain)).astype(np.float32)
        return tuple(
            np.ascontiguousarray(value, dtype=np.float32)
            for value in (fade_out, fade_in, fade_out, fade_in, fade_out, fade_in)
        )

    def _simple_transition_plan(
        self,
        current: PreparedTrack,
        next_track: PreparedTrack,
        earliest_start: int,
        base_plan: TransitionPlan,
        mode: str,
    ) -> TransitionPlan:
        """Build a short silence-trim or beat-aligned equal-power transition."""
        sr = self.config.sample_rate
        if mode == "Gapless Trim":
            duration_seconds = 0.38
            next_start = int(max(0, next_track.structure.silence_start_sample))
            active_end = int(current.structure.silence_end_sample)
            if active_end <= 0:
                active_end = current.total_samples
            length = min(
                max(256, int(round(duration_seconds * sr))),
                max(1, current.total_samples - earliest_start),
                max(1, next_track.total_samples - next_start),
            )
            current_start = max(earliest_start, active_end - length)
            current_start = min(current_start, max(0, current.total_samples - length))
            radius = int(0.020 * sr)
            current_start = _nearest_zero_crossing(current.audio, current_start, radius)
            next_start = _nearest_zero_crossing(next_track.audio, next_start, radius)
            length = min(
                length,
                current.total_samples - current_start,
                next_track.total_samples - next_start,
            )
            length = max(1, int(length))
            bars = 0
            policy = "Silence-trim gapless"
        else:
            # Keep phrase-aligned paper cue points but use a conservative fader.
            current_start = max(earliest_start, base_plan.current_start)
            next_start = base_plan.next_start
            max_length = int(round(8 * max(2, current.analysis.beats_per_bar) * 60.0 /
                                   max(current.bpm_at_sample(current_start), 1.0) * sr))
            length = min(
                base_plan.length,
                max_length,
                current.total_samples - current_start,
                next_track.total_samples - next_start,
            )
            length = max(1, int(length))
            bars = min(base_plan.bars, 8)
            policy = "Simple beat crossfade"

        gain = float(base_plan.metrics.get("local_gain", 1.0))
        fade_out, fade_in, bass_out, bass_in, high_out, high_in = self._neutral_curves(length, gain)
        metrics = dict(base_plan.metrics)
        metrics["policy_complexity"] = 0.15 if mode == "Gapless Trim" else 0.42
        return TransitionPlan(
            current_start=int(current_start),
            next_start=int(next_start),
            length=int(length),
            bars=int(bars),
            current_bar_index=base_plan.current_bar_index,
            next_bar_index=base_plan.next_bar_index,
            score=base_plan.score,
            fade_out=fade_out,
            fade_in=fade_in,
            bass_out=bass_out,
            bass_in=bass_in,
            high_out=high_out,
            high_in=high_in,
            echo_audio=np.zeros((length, 2), dtype=np.float32),
            metrics=metrics,
            automatic=base_plan.automatic,
            style="Smooth",
            effect_strength=0.0,
            transition_mode=mode,
            policy_mode=policy,
            switch_position=0.50,
            dj_intent=base_plan.dj_intent,
            current_role=base_plan.current_role,
            current_landing_role=base_plan.current_landing_role,
            next_role=base_plan.next_role,
            next_landing_role=base_plan.next_landing_role,
            structure_policy_score=base_plan.structure_policy_score,
            recommended_archetypes=base_plan.recommended_archetypes,
        )

    def _apply_automix_policy(
        self,
        current: PreparedTrack,
        next_track: PreparedTrack,
        earliest_start: int,
        plan: TransitionPlan,
    ) -> TransitionPlan:
        policy = self.config.automix_policy.strip().lower()
        if policy == "always dj":
            plan.policy_mode = "Forced complex DJ"
            return plan
        if policy == "crossfade":
            return self._simple_transition_plan(
                current, next_track, earliest_start, plan, "Simple Crossfade"
            )

        edm_pair = float(np.sqrt(
            max(current.structure.edm_confidence, 0.0)
            * max(next_track.structure.edm_confidence, 0.0)
        ))
        cue = float(plan.metrics.get("cue_alignment", 0.0))
        phrase = float(plan.metrics.get("phrase_alignment", 0.0))
        structure_policy = float(np.clip(plan.structure_policy_score, 0.0, 1.0))
        complex_confidence = (
            0.43 * plan.score
            + 0.20 * edm_pair
            + 0.10 * cue
            + 0.07 * phrase
            + 0.20 * structure_policy
        )
        plan.metrics["automix_confidence"] = float(np.clip(complex_confidence, 0.0, 1.0))
        if (
            complex_confidence >= self.config.min_complex_score
            or (
                structure_policy >= 0.72
                and plan.dj_intent
                in {
                    "Post-Drop Relay",
                    "Breakdown Lift",
                    "Phrase-to-Drop",
                    "Double Drop",
                    "Outro-Intro Blend",
                    "Vocal Echo Exit",
                }
            )
        ):
            plan.policy_mode = f"Structure-aware · {plan.dj_intent}"
            return plan
        if plan.score >= 0.43 or cue >= 0.46:
            return self._simple_transition_plan(
                current, next_track, earliest_start, plan, "Simple Crossfade"
            )
        return self._simple_transition_plan(
            current, next_track, earliest_start, plan, "Gapless Trim"
        )

    def _fit_phrase_length(
        self, audio: np.ndarray, target_length: int
    ) -> tuple[np.ndarray, str]:
        """Pitch-preserving local phrase warp to the outgoing wall-clock length."""
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        target_length = max(1, int(target_length))
        if len(audio) == target_length:
            return audio, "none"
        rate = len(audio) / float(target_length)
        rendered, backend = quality_time_stretch(
            audio,
            sample_rate=self.config.sample_rate,
            rate=rate,
            backend=self.config.time_stretch_backend,
        )
        if len(rendered) > target_length:
            rendered = rendered[:target_length]
        elif len(rendered) < target_length:
            rendered = np.pad(
                rendered, ((0, target_length - len(rendered)), (0, 0))
            )
        return np.ascontiguousarray(rendered, dtype=np.float32), backend

    @staticmethod
    def _hpss_stereo(audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Soft harmonic/percussive roles; sum remains close to the input."""
        try:
            harmonic, percussive = librosa.effects.hpss(
                np.ascontiguousarray(audio.T),
                margin=(1.0, 1.35),
            )
            return (
                np.ascontiguousarray(harmonic.T, dtype=np.float32),
                np.ascontiguousarray(percussive.T, dtype=np.float32),
            )
        except Exception:
            return (
                np.ascontiguousarray(audio, dtype=np.float32),
                np.zeros_like(audio, dtype=np.float32),
            )

    def _render_phrase_locked_hpss(
        self,
        plan: TransitionPlan,
        current: PreparedTrack,
        next_track: PreparedTrack,
    ) -> TransitionPlan:
        """Generate several human-style phrase-locked transitions and rank them."""
        a = current.audio[plan.current_start : plan.current_start + plan.length]
        b_end = (
            plan.next_resume_sample
            if plan.next_resume_sample >= 0
            else plan.next_start + plan.length
        )
        b_raw = next_track.audio[plan.next_start : b_end]
        if len(a) < plan.length:
            a = np.pad(a, ((0, plan.length - len(a)), (0, 0)))
        a = np.ascontiguousarray(a[: plan.length], dtype=np.float32)
        b, phrase_backend = self._fit_phrase_length(b_raw, plan.length)

        a_low, _, _ = _split_three_bands(
            a, self.config.sample_rate, self.config.bass_split_hz, self.config.high_split_hz
        )
        b_low, _, _ = _split_three_bands(
            b, self.config.sample_rate, self.config.bass_split_hz, self.config.high_split_hz
        )
        a_body = np.ascontiguousarray(a - a_low, dtype=np.float32)
        b_body = np.ascontiguousarray(b - b_low, dtype=np.float32)
        a_harm, a_perc = self._hpss_stereo(a_body)
        b_harm, b_perc = self._hpss_stereo(b_body)

        current_label = plan.current_role or (
            current.structure.labels[plan.current_bar_index]
            if plan.current_bar_index < len(current.structure.labels)
            else "SECTION"
        )
        next_label = plan.next_role or (
            next_track.structure.labels[plan.next_bar_index]
            if plan.next_bar_index < len(next_track.structure.labels)
            else "SECTION"
        )
        human = render_human_transition(
            a_low=a_low,
            b_low=b_low,
            a_harm=a_harm,
            b_harm=b_harm,
            a_perc=a_perc,
            b_perc=b_perc,
            sample_rate=self.config.sample_rate,
            bpm=current.playback_bpm,
            beats_per_bar=max(2, current.analysis.beats_per_bar),
            bars=max(1, plan.bars),
            local_gain=float(plan.metrics.get("local_gain", 1.0)),
            effect_strength=plan.effect_strength,
            plan_metrics=plan.metrics,
            current_label=current_label,
            next_label=next_label,
            history=tuple(self._transition_history),
            config=HumanTransitionConfig(
                mode=self.config.human_style_mode,
                variation=self.config.human_variation,
                max_candidates=self.config.human_max_candidates,
            ),
        )

        plan.rendered_audio = np.ascontiguousarray(human.audio, dtype=np.float32)
        plan.human_archetype = human.archetype
        plan.human_quality_score = human.score
        plan.human_quality_metrics = dict(human.quality)
        plan.metrics.update(human.quality)
        plan.metrics["phrase_warp_ratio"] = len(b_raw) / max(plan.length, 1)
        plan.metrics["phrase_warp_backend"] = (
            1.0 if "rubber" in phrase_backend.lower() else 0.0
        )
        plan.metrics["human_current_role"] = float(
            {"INTRO": 0, "BUILDUP": 1, "SECTION": 2, "PHRASE": 3, "VOCAL": 4, "BREAK": 5, "BREAKDOWN": 5, "COOLDOWN": 6, "DROP": 7, "OUTRO": 8}.get(current_label, 1)
        )
        plan.metrics["human_next_role"] = float(
            {"INTRO": 0, "BUILDUP": 1, "SECTION": 2, "PHRASE": 3, "VOCAL": 4, "BREAK": 5, "BREAKDOWN": 5, "COOLDOWN": 6, "DROP": 7, "OUTRO": 8}.get(next_label, 1)
        )
        plan.transition_mode = f"Human Candidate · {human.archetype}"
        return plan

    def _render_advanced_transition(
        self,
        plan: TransitionPlan,
        current: PreparedTrack,
        next_track: PreparedTrack,
    ) -> TransitionPlan:
        """Pre-render role-aware or graph-cut transition outside audio callback."""
        is_complex = plan.transition_mode == "Paper EQ/Fader"
        plan.echo_audio = (
            self._render_echo(plan, current)
            if is_complex
            else np.zeros((plan.length, 2), dtype=np.float32)
        )
        if not is_complex:
            return plan

        engine = self.config.transition_engine.strip().lower()
        # Adaptive now defaults to phrase-locked role-aware rendering. Graph-cut is
        # retained as an explicit experimental choice because it can blur dense EDM
        # transients when the seam changes too frequently across frequency bins.
        use_spectral = engine == "spectral seam"
        if use_spectral and plan.length / self.config.sample_rate <= 24.0:
            a = current.audio[plan.current_start : plan.current_start + plan.length]
            b_end = plan.next_resume_sample if plan.next_resume_sample >= 0 else plan.next_start + plan.length
            b_raw = next_track.audio[plan.next_start : b_end]
            b, phrase_backend = self._fit_phrase_length(b_raw, plan.length)
            try:
                result = spectral_seam_crossfade(
                    outgoing=a[: plan.length],
                    incoming=b,
                    sample_rate=self.config.sample_rate,
                )
                rendered = result.audio[: plan.length] + plan.echo_audio[: plan.length]
                peak = float(np.max(np.abs(rendered)) + 1e-9)
                if peak > 0.98:
                    rendered *= np.float32(0.98 / peak)
                plan.rendered_audio = np.ascontiguousarray(rendered, dtype=np.float32)
                plan.transition_mode = "Phrase-Locked Spectral Graph-Cut"
                plan.metrics["seam_mean_seconds"] = result.seam_seconds_mean
                plan.metrics["seam_std_seconds"] = result.seam_seconds_std
                plan.metrics["graph_flow"] = result.flow_value
                plan.metrics["phrase_warp_ratio"] = len(b_raw) / max(plan.length, 1)
                plan.metrics["phrase_warp_backend"] = 1.0 if "rubber" in phrase_backend.lower() else 0.0
                return plan
            except Exception as exc:
                self.emit(f"谱缝合回退到 MuQ/HPSS：{exc}")

        return self._render_phrase_locked_hpss(plan, current, next_track)

    def _calculate_plan(
        self,
        current: PreparedTrack,
        next_track: PreparedTrack,
        earliest_start: int,
        force_current_start: int | None = None,
    ) -> TransitionPlan:
        plan = find_best_transition(
            current=current,
            next_track=next_track,
            earliest_start=earliest_start,
            requested_bars=self.config.crossfade_bars,
            force_current_start=force_current_start,
            config=self.matcher_config,
            fx_config=self._fx_config(),
        )
        plan = self._apply_automix_policy(current, next_track, earliest_start, plan)
        # Rendering is intentionally deferred until after the incoming track has
        # received its continuous BPM-recovery bridge. This avoids doing HPSS and
        # phrase warping twice in the background loader.
        plan.metrics["tempo_sync_bpm"] = next_track.playback_bpm
        plan.metrics["tempo_original_bpm"] = next_track.original_bpm
        plan.metrics["current_edm_confidence"] = current.structure.edm_confidence
        plan.metrics["next_edm_confidence"] = next_track.structure.edm_confidence
        return plan

    def _load_next_worker(
        self,
        generation: int,
        source_index: int,
        analysis: TrackAnalysis,
        target_bpm: float,
    ) -> None:
        try:
            with self._lock:
                snapshot = list(self._playlist)
                outgoing_analysis = snapshot[source_index]
            synced_base = self._take_warm_track(
                outgoing_analysis, analysis, target_bpm
            )
            if synced_base is None:
                self.emit(f"正在分析下一首匹配片段：{analysis.title}")
                synced_base = self.prepare_track(analysis, target_bpm=target_bpm)

            with self._lock:
                still_valid = (
                    generation == self._loader_generation
                    and source_index == self._index
                    and self._current is not None
                    and self._playing
                )
                current = self._current
                latest = self._current_pos
                earliest = latest + int(0.25 * self.config.sample_rate)
                remaining_seconds = (
                    (current.total_samples - latest) / self.config.sample_rate
                    if current is not None else 0.0
                )

            plan: TransitionPlan | None = None
            prepared = synced_base
            if still_valid and current is not None:
                plan = self._calculate_plan(current, synced_base, earliest)
                with self._lock:
                    latest = self._current_pos
                    remaining_seconds = (
                        current.total_samples - latest
                    ) / self.config.sample_rate
                if latest >= plan.current_start:
                    plan = self._calculate_plan(
                        current,
                        synced_base,
                        latest + int(0.10 * self.config.sample_rate),
                    )

                # Deadline guard: if heavy processing finished close to the end,
                # use a shorter beat-aligned transition rather than missing the
                # planned OUT point and falling off the end of the current track.
                if remaining_seconds <= max(8.0, self.config.preload_deadline_seconds * 0.35):
                    if plan.current_end > current.total_samples or (
                        plan.current_start - latest
                    ) / self.config.sample_rate < 1.5:
                        plan = self._simple_transition_plan(
                            current,
                            synced_base,
                            latest + int(0.08 * self.config.sample_rate),
                            plan,
                            "Simple Crossfade",
                        )
                        plan.metrics["preload_deadline_fallback"] = 1.0
                        self.emit("预加载接近截止点：已切换为短节拍淡化，避免错过切歌")

                prepared = self._apply_tempo_restore(synced_base, plan.next_end)
                plan = self._render_advanced_transition(plan, current, prepared)

            installed = False
            with self._lock:
                if (
                    generation == self._loader_generation
                    and source_index == self._index
                    and self._playing
                ):
                    self._next_synced_base = synced_base
                    self._next = prepared
                    self._plan = plan
                    self._preload_urgent = False
                    self._next_loading = False
                    installed = True
                    if plan is not None:
                        self.emit(
                            "专业切歌已规划："
                            f"A {plan.current_start / self.config.sample_rate:.1f}s → "
                            f"B {plan.next_start / self.config.sample_rate:.1f}s | "
                            f"{(str(plan.bars) + ' 小节') if plan.bars else '无缝裁切'} | {plan.dj_intent} | {plan.transition_mode} | "
                            f"匹配 {plan.score * 100:.0f}分"
                        )

            # The immediate next track is hot. Use the remaining background time
            # to prepare synchronized bases farther ahead in the queue.
            if installed and prepared is not None and source_index + 2 < len(snapshot):
                self._warm_following_tracks(
                    snapshot=snapshot,
                    first_target_index=source_index + 2,
                    source_bpm=prepared.original_bpm,
                    generation=generation,
                    token_kind="loader",
                )
        except Exception as exc:
            self.emit(f"准备下一首失败：{exc}")
        finally:
            with self._lock:
                if (
                    generation == self._loader_generation
                    and source_index == self._index
                ):
                    self._next_loading = False

    def service(self) -> None:
        with self._lock:
            if not self._playing or self._current is None:
                return
            remaining_seconds = (
                self._current.total_samples - self._current_pos
            ) / self.config.sample_rate
            if self._next is None and self._index + 1 < len(self._playlist):
                urgent = remaining_seconds <= self.config.preload_deadline_seconds
                self._preload_urgent = urgent
                if urgent and self._urgent_notice_index != self._index:
                    self._urgent_notice_index = self._index
                    self.emit(
                        f"预加载进入优先模式：距离当前轨结束约 {remaining_seconds:.0f} 秒"
                    )
            else:
                self._preload_urgent = False

            if (
                self._next is not None
                or self._next_loading
                or self._index + 1 >= len(self._playlist)
            ):
                return
            analysis = self._playlist[self._index + 1]
            target_bpm = self._current.original_bpm
            generation = self._loader_generation
            source_index = self._index
            self._next_loading = True

        worker = threading.Thread(
            target=self._load_next_worker,
            args=(generation, source_index, analysis, target_bpm),
            daemon=True,
            name="AutoDJ-SlidingWindowLoader",
        )
        self._next_loader = worker
        worker.start()
    def _replan(self, force_start: int | None = None) -> TransitionPlan | None:
        with self._lock:
            current = self._current
            synced_base = self._next_synced_base
            earliest = self._current_pos + int(0.15 * self.config.sample_rate)
        if current is None or synced_base is None:
            return None
        plan = self._calculate_plan(
            current,
            synced_base,
            earliest_start=earliest,
            force_current_start=force_start,
        )
        prepared = self._apply_tempo_restore(synced_base, plan.next_end)
        plan = self._render_advanced_transition(plan, current, prepared)
        with self._lock:
            if self._current is current and self._next_synced_base is synced_base:
                self._next = prepared
                self._plan = plan
        return plan

    def rebuild_transition(self) -> None:
        try:
            plan = self._replan()
            if plan is not None:
                self.emit(
                    f"切歌点已更新：{plan.bars} 小节，{plan.style}，"
                    f"匹配 {plan.score * 100:.0f}分"
                )
        except Exception as exc:
            self.emit(f"重新计算切歌点失败：{exc}")

    def request_next(self) -> bool:
        with self._lock:
            if self._current is None or self._next_synced_base is None:
                self.emit("下一首仍在准备，暂时不能切歌。")
                return False
            earliest = self._current_pos + int(0.08 * self.config.sample_rate)
        try:
            plan = self._replan(force_start=earliest)
        except Exception as exc:
            self.emit(f"安排立即切歌失败：{exc}")
            return False
        if plan is None:
            return False
        with self._lock:
            self._auto_mix = True
        self.emit(
            f"已安排在下一个重拍切歌：{plan.bars} 小节，"
            f"{plan.style}，匹配 {plan.score * 100:.0f}分"
        )
        return True

    def _begin_transition_locked(self) -> None:
        if self._plan is None or self._next is None:
            return
        self._transitioning = True
        self._transition_pos = max(0, self._current_pos - self._plan.current_start)
        self._next_pos = self._plan.next_start + self._transition_pos

    def _promote_next_locked(self) -> None:
        if self._next is None:
            self._playing = False
            return
        if self._plan is not None and self._plan.human_archetype:
            self._transition_history.append(self._plan.human_archetype)
            del self._transition_history[:-8]
        resume = self._next_pos
        if self._plan is not None and self._plan.next_resume_sample >= 0:
            resume = self._plan.next_resume_sample
        self._current = self._next
        self._current_pos = int(np.clip(resume, 0, max(0, self._current.total_samples - 1)))
        self._index += 1
        self._next = None
        self._next_synced_base = None
        self._plan = None
        self._transitioning = False
        self._transition_pos = 0
        self._next_pos = 0
        self._loader_generation += 1

    def _soft_limit(self, data: np.ndarray) -> None:
        drive = float(max(1.0, self.config.limiter_drive))
        peak = float(np.max(np.abs(data)))
        if peak > 0.86:
            data[:] = np.tanh(data * drive) / np.tanh(drive)
        np.clip(data, -1.0, 1.0, out=data)

    def _audio_callback(
        self,
        outdata: np.ndarray,
        frames: int,
        time_info: Any,
        status: sd.CallbackFlags,
    ) -> None:
        outdata.fill(0.0)
        if status:
            self._last_callback_status = str(status)

        with self._lock:
            if not self._playing or self._paused or self._current is None:
                return

            written = 0
            while written < frames and self._playing and self._current is not None:
                current = self._current
                if (
                    not self._transitioning
                    and self._auto_mix
                    and self._plan is not None
                    and self._next is not None
                    and self._current_pos >= self._plan.current_start
                ):
                    self._begin_transition_locked()

                if self._transitioning and self._plan is not None and self._next is not None:
                    plan = self._plan
                    next_track = self._next
                    remaining = plan.length - self._transition_pos
                    count = min(
                        frames - written,
                        remaining,
                        current.total_samples - self._current_pos,
                        next_track.total_samples - self._next_pos,
                    )
                    if count <= 0:
                        self._promote_next_locked()
                        continue

                    t0 = self._transition_pos
                    t1 = t0 + count
                    a0 = self._current_pos
                    a1 = a0 + count
                    b0 = self._next_pos
                    b1 = b0 + count

                    if plan.rendered_audio is not None:
                        mixed = plan.rendered_audio[t0:t1]
                    else:
                        # 三段 EQ：低频独立交换，中频等功率，高频执行滤波扫频。
                        mixed = (
                            current.low_audio[a0:a1] * plan.bass_out[t0:t1, None]
                            + next_track.low_audio[b0:b1] * plan.bass_in[t0:t1, None]
                            + current.mid_audio[a0:a1] * plan.fade_out[t0:t1, None]
                            + next_track.mid_audio[b0:b1] * plan.fade_in[t0:t1, None]
                            + current.high_audio[a0:a1] * plan.high_out[t0:t1, None]
                            + next_track.high_audio[b0:b1] * plan.high_in[t0:t1, None]
                            + plan.echo_audio[t0:t1]
                        )
                    outdata[written : written + count] = mixed
                    self._current_pos = a1
                    self._next_pos = b1
                    self._transition_pos = t1
                    written += count
                    if self._transition_pos >= plan.length:
                        self._promote_next_locked()
                    continue

                if self._current_pos >= current.total_samples:
                    if self._next is not None:
                        self._next_pos = self._next.cue_sample
                        self._promote_next_locked()
                        continue
                    self._playing = False
                    break

                stop_at = current.total_samples
                if self._auto_mix and self._plan is not None and self._next is not None:
                    stop_at = min(stop_at, self._plan.current_start)
                count = min(frames - written, max(0, stop_at - self._current_pos))
                if count <= 0:
                    if self._plan is not None and self._next is not None:
                        self._begin_transition_locked()
                        continue
                    self._playing = False
                    break

                a0 = self._current_pos
                a1 = a0 + count
                outdata[written : written + count] = current.audio[a0:a1]
                self._current_pos = a1
                written += count

            if self._master_volume != 1.0:
                outdata *= np.float32(self._master_volume)
            if self._seek_fade_remaining > 0:
                count = min(frames, self._seek_fade_remaining)
                completed = self._seek_fade_total - self._seek_fade_remaining
                ramp = np.linspace(
                    completed / max(self._seek_fade_total, 1),
                    (completed + count) / max(self._seek_fade_total, 1),
                    count,
                    endpoint=False,
                    dtype=np.float32,
                )
                outdata[:count] *= ramp[:, None]
                self._seek_fade_remaining -= count
            self._soft_limit(outdata)

    def pause(self) -> None:
        with self._lock:
            self._paused = True

    def resume(self) -> None:
        with self._lock:
            if self._current is not None:
                self._paused = False
                self._playing = True

    def seek(self, seconds: float, snap_to_beat: bool = True) -> float:
        """跳转当前歌曲位置，并在后台重新规划后续切歌。

        默认吸附到最近 beat，既便于 DJ 场景定位，也减少从瞬态中间起播的
        违和感。跳转后加入 30ms 淡入以抑制 click。
        """
        with self._lock:
            current = self._current
            if current is None:
                raise RuntimeError("当前没有可跳转的歌曲。")
            target = int(round(float(seconds) * self.config.sample_rate))
            target = int(np.clip(target, 0, max(0, current.total_samples - 1)))
            if snap_to_beat and current.beat_samples.size:
                index = int(np.argmin(np.abs(current.beat_samples - target)))
                target = int(current.beat_samples[index])

            self._current_pos = target
            self._transitioning = False
            self._transition_pos = 0
            self._next_pos = 0
            self._seek_fade_remaining = self._seek_fade_total

            should_replan = self._next_synced_base is not None
            if should_replan:
                # 旧 plan 的 OUT 位置可能已经在跳转点之前；先撤销已渲染版本，
                # 后台以新播放位置为 earliest_start 重新搜索。
                self._next = None
                self._plan = None
                self._next_loading = True
                replan_generation = self._loader_generation
                replan_index = self._index
            else:
                replan_generation = -1
                replan_index = -1
            actual = target / float(self.config.sample_rate)

        self.emit(f"已跳转到 {actual:.1f}s（吸附最近节拍）")
        if should_replan:
            def worker() -> None:
                try:
                    self._replan()
                    self.emit("跳转后的切歌位置已重新规划")
                except Exception as exc:
                    self.emit(f"跳转后重新规划失败：{exc}")
                finally:
                    with self._lock:
                        if (
                            replan_generation == self._loader_generation
                            and replan_index == self._index
                        ):
                            self._next_loading = False

            threading.Thread(
                target=worker,
                daemon=True,
                name="AutoDJ-SeekReplan",
            ).start()
        else:
            self.service()
        return actual

    def update_playlist_order(self, playlist: Sequence[TrackAnalysis]) -> None:
        """在播放中更新队列，保留当前歌曲并重载改变后的下一首。"""
        snapshot = list(playlist)
        if not snapshot:
            return
        reload_needed = False
        with self._lock:
            if self._current is None or not self._playing:
                self._playlist = snapshot
                return
            current_path = self._current.analysis.path
            positions = [
                index
                for index, item in enumerate(snapshot)
                if item.path == current_path
            ]
            if not positions:
                raise ValueError("新播放列表中缺少当前正在播放的歌曲。")
            new_index = positions[0]
            desired_next = (
                snapshot[new_index + 1].path
                if new_index + 1 < len(snapshot)
                else ""
            )
            loaded_next = self._next.analysis.path if self._next is not None else ""

            self._playlist = snapshot
            self._index = new_index
            self._loader_generation += 1
            self._next_loading = False
            self._warm_loading_paths.clear()
            self._preload_urgent = False

            # 已经进入过渡后不能替换 B；GUI 排序器会锁住该歌曲。
            if not self._transitioning and desired_next != loaded_next:
                self._next = None
                self._next_synced_base = None
                self._plan = None
                self._next_pos = 0
                self._transition_pos = 0
                reload_needed = bool(desired_next)

        if reload_needed:
            self.emit("MuQ 已更新尚未播放顺序，正在预加载新的下一首")
            self.service()

    def stop(self, clear_preload: bool = True) -> None:
        with self._lock:
            self._playing = False
            self._paused = False
            self._loader_generation += 1
        stream = self._stream
        self._stream = None
        if stream is not None:
            try:
                stream.abort()
            finally:
                stream.close()
        with self._lock:
            self._current = None
            self._next = None
            self._next_synced_base = None
            self._plan = None
            self._playlist = []
            self._index = -1
            self._current_pos = 0
            self._next_pos = 0
            self._transition_pos = 0
            self._transitioning = False
            self._seek_fade_remaining = 0
            self._next_loading = False
            self._preload_urgent = False
            self._warm_loading_paths.clear()
        if clear_preload:
            self.clear_preload()

    def close(self) -> None:
        self.stop()

    def set_auto_mix(self, enabled: bool) -> None:
        with self._lock:
            self._auto_mix = bool(enabled)

    def set_volume(self, value: float) -> None:
        with self._lock:
            self._master_volume = float(np.clip(value, 0.0, 1.25))

    def set_crossfade_bars(self, bars: int) -> None:
        self.config.crossfade_bars = int(max(0, bars))
        self.rebuild_transition()

    def set_max_stretch_percent(self, percent: float) -> None:
        self.config.max_stretch_percent = float(np.clip(percent, 0.0, 30.0))

    def set_tempo_restore_bars(self, bars: int) -> None:
        value = int(bars)
        self.config.tempo_restore_bars = -1 if value < 0 else max(0, value)
        self.rebuild_transition()

    def set_mix_style(self, style: str) -> None:
        if style not in {"Smooth", "Club", "Filter", "Echo"}:
            raise ValueError(f"未知混音风格：{style}")
        self.config.mix_style = style
        self.rebuild_transition()

    def set_effect_strength(self, strength: float) -> None:
        self.config.effect_strength = float(np.clip(strength, 0.0, 1.0))
        self.rebuild_transition()

    def set_human_style_mode(self, value: str) -> None:
        allowed = {"Adaptive Human", *SUPPORTED_ARCHETYPES}
        if value not in allowed:
            raise ValueError(f"未知真人 DJ 策略：{value}")
        self.config.human_style_mode = value
        self.rebuild_transition()

    def set_human_variation(self, value: float) -> None:
        self.config.human_variation = float(np.clip(value, 0.0, 1.0))
        self.rebuild_transition()

    def set_human_candidate_count(self, value: int) -> None:
        self.config.human_max_candidates = int(np.clip(value, 1, len(SUPPORTED_ARCHETYPES)))
        self.rebuild_transition()

    def set_transition_engine(self, value: str) -> None:
        if value not in {"Adaptive", "Spectral Seam", "EQ/Fader"}:
            raise ValueError(f"未知过渡引擎：{value}")
        self.config.transition_engine = value
        self.rebuild_transition()

    def set_automix_policy(self, value: str) -> None:
        if value not in {"AutoMix-like", "Always DJ", "Crossfade"}:
            raise ValueError(f"未知 AutoMix 策略：{value}")
        self.config.automix_policy = value
        self.rebuild_transition()

    def set_time_stretch_backend(self, value: str) -> None:
        if value not in {"auto", "Rubber Band R3", "Hybrid HPSS", "librosa"}:
            raise ValueError(f"未知时间拉伸后端：{value}")
        self.config.time_stretch_backend = value

    def set_muq_enabled(self, enabled: bool) -> None:
        self.config.muq_enabled = bool(enabled)
        self.rebuild_transition()

    def set_muq_device(self, device: str) -> None:
        if device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError(f"未知 MuQ 设备：{device}")
        self.config.muq_device = device
        self._muq_analyzer = None

    def set_songformer_enabled(self, enabled: bool) -> None:
        self.config.songformer_enabled = bool(enabled)
        self.rebuild_transition()

    def set_songformer_device(self, device: str) -> None:
        if device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError(f"未知 SongFormer 设备：{device}")
        self.config.songformer_device = device

    def set_songformer_model(self, model: str) -> None:
        model = str(model).strip()
        if not model:
            raise ValueError("SongFormer 模型名称不能为空。")
        if model != "ASLP-lab/SongFormer":
            raise ValueError(f"当前版本仅支持官方模型 ASLP-lab/SongFormer：{model}")
        if self.config.songformer_model_name == model:
            return
        self.config.songformer_model_name = model
        self.rebuild_transition()

    def set_preload_window_tracks(self, value: int) -> None:
        self.config.preload_window_tracks = int(np.clip(value, 2, 6))

    def set_preload_memory_mb(self, value: int) -> None:
        self.config.preload_memory_mb = int(np.clip(value, 256, 8192))

    def set_preload_deadline_seconds(self, value: float) -> None:
        self.config.preload_deadline_seconds = float(np.clip(value, 15.0, 180.0))

    def set_cue_on_first_downbeat(self, enabled: bool) -> None:
        self.config.cue_on_first_downbeat = bool(enabled)

    @staticmethod
    def _timeline_sections(track: PreparedTrack | None) -> tuple[tuple[float, float, str], ...]:
        if track is None or not track.songformer_profile.available:
            return ()
        source_grid = np.asarray(track.source_downbeat_samples, dtype=np.float64)
        target_grid = np.asarray(track.downbeat_samples, dtype=np.float64)
        count = min(source_grid.size, target_grid.size)
        if count >= 2:
            source_grid = source_grid[:count]
            target_grid = target_grid[:count]
            source_grid = np.concatenate(([0.0], source_grid, [float(len(track.source_audio))]))
            target_grid = np.concatenate(([0.0], target_grid, [float(track.total_samples)]))
            source_grid, unique = np.unique(source_grid, return_index=True)
            target_grid = target_grid[unique]

            def mapped(seconds: float) -> float:
                sample = float(seconds) * track.sample_rate
                value = float(np.interp(sample, source_grid, target_grid))
                return value / track.sample_rate
        else:
            def mapped(seconds: float) -> float:
                return float(seconds) / max(track.stretch_rate, 1e-6)

        return tuple(
            (mapped(segment.start), mapped(segment.end), segment.label.upper())
            for segment in track.songformer_profile.segments
            if segment.end > segment.start
        )

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            current = self._current
            next_track = self._next
            plan = self._plan
            stream = self._stream
            if current is None:
                return {
                    "playing": False,
                    "paused": False,
                    "current": "",
                    "current_path": "",
                    "next": "",
                    "next_path": "",
                    "position": 0.0,
                    "duration": 0.0,
                    "next_duration": 0.0,
                    "progress": 0.0,
                    "bpm": 0.0,
                    "sync_bpm": 0.0,
                    "original_bpm": 0.0,
                    "beat_number": 0,
                    "transitioning": False,
                    "transition_progress": 0.0,
                    "index": -1,
                    "next_loading": self._next_loading,
                    "next_ready": False,
                    "preload_loading": self._prime_loading,
                    "preload_ready": self._prime_current is not None,
                    "preload_current_path": self._prime_key[0] if self._prime_key else "",
                    "preload_next_path": self._prime_key[1] if self._prime_key else "",
                    "preload_window_tracks": self.config.preload_window_tracks,
                    "warm_ready_paths": tuple(key[1] for key in self._warm_cache.keys()),
                    "warm_loading_paths": tuple(sorted(self._warm_loading_paths)),
                    "warm_cache_mb": self._warm_cache_bytes / (1024.0 * 1024.0),
                    "preload_urgent": self._preload_urgent,
                    "seconds_to_transition": None,
                    "cpu_load": 0.0,
                    "callback_status": self._last_callback_status,
                    "current_waveform": None,
                    "next_waveform": None,
                    "transition_start": None,
                    "transition_end": None,
                    "next_entry": None,
                    "next_transition_end": None,
                    "tempo_restore_start": None,
                    "tempo_restore_end": None,
                    "next_tempo_restore_start": None,
                    "next_tempo_restore_end": None,
                    "tempo_restore_progress": 0.0,
                    "transition_bars": 0,
                    "tempo_restore_bars": 0,
                    "match_score": 0.0,
                    "match_metrics": {},
                    "mix_style": self.config.mix_style,
                    "effect_strength": self.config.effect_strength,
                    "human_style_mode": self.config.human_style_mode,
                    "human_variation": self.config.human_variation,
                    "human_archetype": "",
                    "human_quality_score": 0.0,
                    "human_quality_metrics": {},
                    "transition_mode": "",
                    "policy_mode": self.config.automix_policy,
                    "dj_intent": "",
                    "current_role": "",
                    "current_landing_role": "",
                    "next_role": "",
                    "next_landing_role": "",
                    "structure_policy_score": 0.0,
                    "stretch_backend": "",
                    "current_key": "—",
                    "next_key": "—",
                    "current_edm_confidence": 0.0,
                    "next_edm_confidence": 0.0,
                    "current_cues": (),
                    "next_cues": (),
                    "current_sections": (),
                    "next_sections": (),
                    "current_structure_source": "",
                    "next_structure_source": "",
                    "current_function_label": "",
                    "next_function_label": "",
                    "songformer_enabled": self.config.songformer_enabled,
                    "switch_time_a": None,
                    "switch_time_b": None,
                    "plan_signature": None,
                }

            current_pos = self._current_pos
            beat_number = 0
            if current.beat_samples.size:
                beat_index = int(
                    np.searchsorted(current.beat_samples, current_pos, side="right") - 1
                )
                if 0 <= beat_index < current.beat_numbers.size:
                    beat_number = int(current.beat_numbers[beat_index])

            duration = current.duration
            position = current_pos / float(self.config.sample_rate)
            progress = position / duration if duration > 0 else 0.0
            transition_progress = (
                self._transition_pos / plan.length
                if self._transitioning and plan and plan.length
                else 0.0
            )
            if current.has_tempo_restore:
                tempo_restore_progress = float(
                    np.clip(
                        (current_pos - current.tempo_restore_start)
                        / max(current.tempo_restore_end - current.tempo_restore_start, 1),
                        0.0,
                        1.0,
                    )
                )
            else:
                tempo_restore_progress = 0.0
            try:
                cpu_load = float(stream.cpu_load) if stream else 0.0
            except Exception:
                cpu_load = 0.0

            transition_start = plan.current_start / self.config.sample_rate if plan else None
            transition_end = plan.current_end / self.config.sample_rate if plan else None
            next_entry = plan.next_start / self.config.sample_rate if plan else None
            next_transition_end = plan.next_end / self.config.sample_rate if plan else None
            signature = (
                self._index,
                plan.current_start,
                plan.next_start,
                plan.length,
                plan.bars,
                plan.style,
                round(plan.effect_strength, 2),
                self.config.tempo_restore_bars,
                plan.transition_mode,
                plan.policy_mode,
                plan.human_archetype,
                plan.dj_intent,
                plan.next_landing_role,
                round(plan.human_quality_score, 3),
            ) if plan else None

            return {
                "playing": self._playing,
                "paused": self._paused,
                "current": current.title,
                "current_path": current.analysis.path,
                "next": next_track.title if next_track else "",
                "next_path": next_track.analysis.path if next_track else "",
                "position": position,
                "duration": duration,
                "next_duration": next_track.duration if next_track else 0.0,
                "progress": float(np.clip(progress, 0.0, 1.0)),
                "bpm": current.bpm_at_sample(current_pos),
                "sync_bpm": current.playback_bpm,
                "original_bpm": current.original_bpm,
                "next_sync_bpm": next_track.playback_bpm if next_track else 0.0,
                "next_original_bpm": next_track.original_bpm if next_track else 0.0,
                "beat_number": beat_number,
                "transitioning": self._transitioning,
                "transition_progress": float(np.clip(transition_progress, 0.0, 1.0)),
                "index": self._index,
                "next_loading": self._next_loading,
                "next_ready": next_track is not None and plan is not None,
                "preload_loading": self._prime_loading,
                "preload_ready": self._prime_current is not None,
                "preload_current_path": self._prime_key[0] if self._prime_key else "",
                "preload_next_path": self._prime_key[1] if self._prime_key else "",
                "preload_window_tracks": self.config.preload_window_tracks,
                "warm_ready_paths": tuple(key[1] for key in self._warm_cache.keys()),
                "warm_loading_paths": tuple(sorted(self._warm_loading_paths)),
                "warm_cache_mb": self._warm_cache_bytes / (1024.0 * 1024.0),
                "preload_urgent": self._preload_urgent,
                "seconds_to_transition": (
                    max(0.0, (plan.current_start - current_pos) / self.config.sample_rate)
                    if plan is not None else max(0.0, duration - position)
                ),
                "cpu_load": cpu_load,
                "callback_status": self._last_callback_status,
                "current_waveform": current.waveform_envelope,
                "next_waveform": next_track.waveform_envelope if next_track else None,
                "transition_start": transition_start,
                "transition_end": transition_end,
                "next_entry": next_entry,
                "next_transition_end": next_transition_end,
                "tempo_restore_start": (
                    current.tempo_restore_start / self.config.sample_rate
                    if current.has_tempo_restore else None
                ),
                "tempo_restore_end": (
                    current.tempo_restore_end / self.config.sample_rate
                    if current.has_tempo_restore else None
                ),
                "next_tempo_restore_start": (
                    next_track.tempo_restore_start / self.config.sample_rate
                    if next_track and next_track.has_tempo_restore else None
                ),
                "next_tempo_restore_end": (
                    next_track.tempo_restore_end / self.config.sample_rate
                    if next_track and next_track.has_tempo_restore else None
                ),
                "tempo_restore_progress": tempo_restore_progress,
                "transition_bars": plan.bars if plan else 0,
                "tempo_restore_bars": (
                    next_track.tempo_restore_bars
                    if next_track and next_track.has_tempo_restore
                    else current.tempo_restore_bars
                ),
                "match_score": plan.score if plan else 0.0,
                "match_metrics": dict(plan.metrics) if plan else {},
                "mix_style": plan.style if plan else self.config.mix_style,
                "effect_strength": (
                    plan.effect_strength if plan else self.config.effect_strength
                ),
                "human_style_mode": self.config.human_style_mode,
                "human_variation": self.config.human_variation,
                "human_archetype": plan.human_archetype if plan else "",
                "human_quality_score": plan.human_quality_score if plan else 0.0,
                "human_quality_metrics": dict(plan.human_quality_metrics) if plan else {},
                "transition_mode": plan.transition_mode if plan else "",
                "policy_mode": plan.policy_mode if plan else self.config.automix_policy,
                "dj_intent": plan.dj_intent if plan else "",
                "current_role": plan.current_role if plan else "",
                "current_landing_role": plan.current_landing_role if plan else "",
                "next_role": plan.next_role if plan else "",
                "next_landing_role": plan.next_landing_role if plan else "",
                "structure_policy_score": plan.structure_policy_score if plan else 0.0,
                "stretch_backend": next_track.stretch_backend if next_track else current.stretch_backend,
                "current_key": current.structure.camelot,
                "next_key": next_track.structure.camelot if next_track else "—",
                "current_edm_confidence": current.structure.edm_confidence,
                "next_edm_confidence": next_track.structure.edm_confidence if next_track else 0.0,
                "current_cues": tuple(
                    float(current.bar_features.start_samples[index] / self.config.sample_rate)
                    for index in current.structure.cue_indices[:48]
                    if index < current.bar_features.count
                ),
                "next_cues": tuple(
                    float(next_track.bar_features.start_samples[index] / self.config.sample_rate)
                    for index in next_track.structure.cue_indices[:48]
                    if next_track and index < next_track.bar_features.count
                ) if next_track else (),
                "current_sections": self._timeline_sections(current),
                "next_sections": self._timeline_sections(next_track),
                "current_structure_source": current.structure.structure_source,
                "next_structure_source": next_track.structure.structure_source if next_track else "",
                "current_function_label": (
                    current.structure.functional_labels[plan.current_bar_index]
                    if plan and plan.current_bar_index < len(current.structure.functional_labels)
                    else ""
                ),
                "next_function_label": (
                    next_track.structure.functional_labels[plan.next_bar_index]
                    if plan and next_track and plan.next_bar_index < len(next_track.structure.functional_labels)
                    else ""
                ),
                "songformer_enabled": self.config.songformer_enabled,
                "switch_time_a": plan.switch_sample_a / self.config.sample_rate if plan else None,
                "switch_time_b": plan.switch_sample_b / self.config.sample_rate if plan else None,
                "plan_signature": signature,
            }
