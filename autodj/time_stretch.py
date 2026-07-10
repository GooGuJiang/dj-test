from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Sequence

import librosa
import numpy as np
import soundfile as sf


def rubberband_executable() -> str | None:
    return shutil.which("rubberband-r3") or shutil.which("rubberband")


def _read_stereo(path: Path, sample_rate: int) -> np.ndarray:
    result, result_sr = sf.read(path, dtype="float32", always_2d=True)
    if int(result_sr) != int(sample_rate):
        result = librosa.resample(
            result.T,
            orig_sr=int(result_sr),
            target_sr=int(sample_rate),
            res_type="soxr_hq",
        ).T
    if result.shape[1] == 1:
        result = np.repeat(result, 2, axis=1)
    return np.ascontiguousarray(result[:, :2], dtype=np.float32)


def _rubberband_command(executable: str) -> list[str]:
    # R3/Fine 与 centre-focus 对整曲立体声的成像和瞬态更稳定。
    return [str(executable), "-q", "-3", "--centre-focus"]


def stretch_stereo(
    audio: np.ndarray,
    sample_rate: int,
    rate: float,
    backend: str = "auto",
) -> tuple[np.ndarray, str]:
    """
    保持音高的立体声定速时间拉伸。

    `rate` 与 librosa 语义一致：大于 1 表示更快、输出更短。auto/high
    优先使用 Rubber Band R3，缺失时回退到 librosa phase vocoder。
    """
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    if audio.size == 0 or math.isclose(rate, 1.0, abs_tol=1e-4):
        return audio, "none"

    requested = backend.strip().lower()
    executable = rubberband_executable()
    use_rubberband = requested in {
        "auto",
        "rubberband",
        "rubber band r3",
        "high",
    } and executable

    if use_rubberband:
        try:
            with tempfile.TemporaryDirectory(prefix="autodj_rb_") as temp_dir:
                input_path = Path(temp_dir) / "input.wav"
                output_path = Path(temp_dir) / "output.wav"
                sf.write(input_path, audio, sample_rate, subtype="FLOAT")
                command = _rubberband_command(str(executable)) + [
                    "-T",
                    f"{float(rate):.10f}",
                    str(input_path),
                    str(output_path),
                ]
                subprocess.run(
                    command,
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                return _read_stereo(output_path, sample_rate), "Rubber Band R3"
        except Exception:
            # 显式请求也保持可播放回退，而不是让后台线程直接退出。
            pass

    stretched = librosa.effects.time_stretch(
        np.ascontiguousarray(audio.T),
        rate=float(rate),
    )
    return (
        np.ascontiguousarray(stretched.T, dtype=np.float32),
        "librosa phase vocoder",
    )


def _validate_time_map(
    keyframes: Sequence[tuple[int, int]],
    source_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    if source_length <= 0:
        return np.asarray([0], dtype=np.int64), np.asarray([0], dtype=np.int64)

    pairs = sorted(
        {
            (int(source), int(target))
            for source, target in keyframes
            if 0 <= int(source) <= source_length and int(target) >= 0
        },
        key=lambda item: item[0],
    )
    if not pairs or pairs[0][0] != 0:
        pairs.insert(0, (0, 0))
    if pairs[-1][0] != source_length:
        if len(pairs) >= 2:
            ds = pairs[-1][0] - pairs[-2][0]
            dt = pairs[-1][1] - pairs[-2][1]
            ratio = dt / max(ds, 1)
        else:
            ratio = 1.0
        pairs.append(
            (source_length, pairs[-1][1] + int(round((source_length - pairs[-1][0]) * ratio)))
        )

    sources: list[int] = []
    targets: list[int] = []
    last_source = -1
    last_target = -1
    for source, target in pairs:
        if source <= last_source:
            continue
        target = max(target, last_target + 1 if sources else 0)
        sources.append(source)
        targets.append(target)
        last_source = source
        last_target = target
    return np.asarray(sources, dtype=np.int64), np.asarray(targets, dtype=np.int64)


def _variable_phase_vocoder(
    audio: np.ndarray,
    sample_rate: int,
    source_keyframes: np.ndarray,
    target_keyframes: np.ndarray,
) -> np.ndarray:
    """单次 STFT 的连续变速回退，避免逐小节独立拉伸产生接缝。"""
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    target_length = int(target_keyframes[-1])
    if audio.size == 0 or target_length <= 0:
        return np.zeros((0, 2), dtype=np.float32)

    # 较短窗口可减少电子鼓 kick/snare 的拖影，同时保持足够低频分辨率。
    n_fft = 2048 if sample_rate >= 32_000 else 1024
    hop_length = n_fft // 4
    spectrogram = librosa.stft(
        audio.T,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window="hann",
        center=True,
    )
    spectrogram = np.pad(spectrogram, ((0, 0), (0, 0), (0, 2)))

    target_positions = np.arange(
        0,
        target_length + hop_length,
        hop_length,
        dtype=np.float64,
    )
    target_positions = np.minimum(target_positions, target_length)
    source_positions = np.interp(
        target_positions,
        target_keyframes.astype(np.float64),
        source_keyframes.astype(np.float64),
    ) / hop_length
    source_positions = np.clip(source_positions, 0.0, spectrogram.shape[-1] - 2.001)

    channels, bins, _ = spectrogram.shape
    output = np.empty(
        (channels, bins, source_positions.size), dtype=np.complex64
    )
    phase_advance = np.linspace(0.0, np.pi * hop_length, bins, dtype=np.float64)
    phase_acc = np.angle(spectrogram[..., 0]).astype(np.float64)

    for output_index, step in enumerate(source_positions):
        frame = int(np.floor(step))
        alpha = float(step - frame)
        left = spectrogram[..., frame]
        right = spectrogram[..., frame + 1]
        magnitude = (1.0 - alpha) * np.abs(left) + alpha * np.abs(right)

        if output_index > 0:
            phase_delta = np.angle(right) - np.angle(left) - phase_advance[None, :]
            phase_delta -= 2.0 * np.pi * np.round(phase_delta / (2.0 * np.pi))
            phase_acc += phase_advance[None, :] + phase_delta
        else:
            phase_acc = np.angle((1.0 - alpha) * left + alpha * right)
        output[..., output_index] = magnitude * np.exp(1j * phase_acc)

    waveform = librosa.istft(
        output,
        hop_length=hop_length,
        win_length=n_fft,
        window="hann",
        center=True,
        length=target_length,
    )
    waveform = np.ascontiguousarray(waveform.T, dtype=np.float32)
    peak = float(np.max(np.abs(waveform)) + 1e-9)
    if peak > 0.995:
        waveform *= np.float32(0.995 / peak)
    return waveform


def stretch_stereo_time_map(
    audio: np.ndarray,
    sample_rate: int,
    keyframes: Sequence[tuple[int, int]],
    backend: str = "auto",
) -> tuple[np.ndarray, str, np.ndarray, np.ndarray]:
    """
    按 source->target sample keyframes 连续改变速度并保持音高。

    Rubber Band 可用时使用官方 `--timemap` 一次处理整个音频；这比逐小节
    独立拉伸再拼接更平滑。缺失时使用单次 STFT 可变相位声码器回退。
    """
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    sources, targets = _validate_time_map(keyframes, len(audio))
    if len(audio) == 0 or int(targets[-1]) <= 0:
        return audio, "none", sources, targets

    requested = backend.strip().lower()
    executable = rubberband_executable()
    use_rubberband = requested in {
        "auto",
        "rubberband",
        "rubber band r3",
        "high",
    } and executable

    if use_rubberband:
        try:
            with tempfile.TemporaryDirectory(prefix="autodj_rb_map_") as temp_dir:
                temp = Path(temp_dir)
                input_path = temp / "input.wav"
                output_path = temp / "output.wav"
                map_path = temp / "tempo.map"
                sf.write(input_path, audio, sample_rate, subtype="FLOAT")
                map_path.write_text(
                    "".join(
                        f"{int(source)} {int(target)}\n"
                        for source, target in zip(sources, targets)
                    ),
                    encoding="utf-8",
                )
                duration = float(targets[-1]) / float(sample_rate)
                command = _rubberband_command(str(executable)) + [
                    "-M",
                    str(map_path),
                    "-D",
                    f"{duration:.10f}",
                    str(input_path),
                    str(output_path),
                ]
                subprocess.run(
                    command,
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                result = _read_stereo(output_path, sample_rate)
                expected = int(targets[-1])
                if len(result) > expected:
                    result = result[:expected]
                elif len(result) < expected:
                    result = np.pad(result, ((0, expected - len(result)), (0, 0)))
                return (
                    np.ascontiguousarray(result, dtype=np.float32),
                    "Rubber Band R3 time-map",
                    sources,
                    targets,
                )
        except Exception:
            pass

    result = _variable_phase_vocoder(
        audio,
        sample_rate,
        sources,
        targets,
    )
    return result, "continuous variable phase vocoder", sources, targets
