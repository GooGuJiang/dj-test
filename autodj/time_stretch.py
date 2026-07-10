from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Sequence

import librosa
import numpy as np
import soundfile as sf


_RUBBERBAND_ENV_VARS = (
    "AUTODJ_RUBBERBAND",
    "RUBBERBAND_EXE",
    "RUBBERBAND_PATH",
)
_RUBBERBAND_NAMES = (
    "rubberband-r3",
    "rubberband",
    "rubberband-cli",
)


def _candidate_executables(value: str | Path) -> list[Path]:
    """把目录、可执行文件路径或带引号的环境变量展开为候选文件。"""
    raw = os.path.expandvars(os.path.expanduser(str(value).strip().strip('"').strip("'")))
    if not raw:
        return []
    path = Path(raw)
    if path.is_file():
        return [path]
    suffix = ".exe" if os.name == "nt" else ""
    if path.is_dir():
        return [path / f"{name}{suffix}" for name in _RUBBERBAND_NAMES]
    # Windows 用户有时把完整 exe 路径错误地放进 PATH；即使 Path.exists()
    # 因为当前盘符/引号问题返回 False，也保留一次直接候选。
    return [path]


def _is_runnable_file(path: Path) -> bool:
    try:
        return path.is_file() and (os.name == "nt" or os.access(path, os.X_OK))
    except OSError:
        return False


def rubberband_executable(explicit_path: str | Path | None = None) -> str | None:
    """
    查找 Rubber Band CLI。

    搜索顺序：
    1. 显式路径；
    2. AUTODJ_RUBBERBAND / RUBBERBAND_EXE / RUBBERBAND_PATH；
    3. 当前进程 PATH；
    4. 当前 Python/Conda 环境的 Scripts、Library/bin、bin；
    5. 常见 Windows 安装目录。

    R3 是 `rubberband -3` 的处理引擎；官方 CLI 通常仍叫
    `rubberband`，也可能以 `rubberband-r3` 名称安装。
    """
    checked: set[str] = set()

    def accept(candidate: Path) -> str | None:
        try:
            resolved = candidate.resolve(strict=False)
        except OSError:
            resolved = candidate
        key = os.path.normcase(str(resolved))
        if key in checked:
            return None
        checked.add(key)
        if _is_runnable_file(resolved):
            return str(resolved)
        return None

    if explicit_path:
        for candidate in _candidate_executables(explicit_path):
            found = accept(candidate)
            if found:
                return found

    for variable in _RUBBERBAND_ENV_VARS:
        value = os.environ.get(variable, "")
        if value:
            for candidate in _candidate_executables(value):
                found = accept(candidate)
                if found:
                    return found

    for name in _RUBBERBAND_NAMES:
        found = shutil.which(name)
        if found:
            accepted = accept(Path(found))
            if accepted:
                return accepted

    # 容忍 PATH 中误放了完整 exe 路径，而不是其所在目录。
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        if not entry:
            continue
        for candidate in _candidate_executables(entry):
            found = accept(candidate)
            if found:
                return found

    prefixes: list[Path] = []
    for value in (
        os.environ.get("CONDA_PREFIX"),
        sys.prefix,
        Path(sys.executable).parent,
    ):
        if value:
            prefixes.append(Path(value))

    search_dirs: list[Path] = []
    for prefix in prefixes:
        search_dirs.extend(
            [
                prefix,
                prefix / "bin",
                prefix / "Scripts",
                prefix / "Library" / "bin",
            ]
        )

    if os.name == "nt":
        for variable in ("ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA"):
            base = os.environ.get(variable)
            if not base:
                continue
            base_path = Path(base)
            search_dirs.extend(
                [
                    base_path / "Rubber Band",
                    base_path / "RubberBand",
                    base_path / "Programs" / "Rubber Band",
                    base_path / "Programs" / "RubberBand",
                ]
            )

    for directory in search_dirs:
        for candidate in _candidate_executables(directory):
            found = accept(candidate)
            if found:
                return found
    return None


def rubberband_probe(explicit_path: str | Path | None = None) -> dict[str, object]:
    """返回可用于 GUI/诊断脚本的检测与实际启动结果。"""
    executable = rubberband_executable(explicit_path)
    result: dict[str, object] = {
        "executable": executable,
        "python": sys.executable,
        "conda_prefix": os.environ.get("CONDA_PREFIX", ""),
        "environment": {
            name: os.environ.get(name, "") for name in _RUBBERBAND_ENV_VARS
        },
    }
    if not executable:
        result.update(ok=False, message="未找到 Rubber Band CLI")
        return result
    try:
        process = subprocess.run(
            [executable, "-h"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=8,
            check=False,
        )
        output = process.stdout or ""
        # 某些构建在 -h 时返回非零状态，但只要成功启动并输出帮助即有效。
        ok = "rubber" in output.lower() or process.returncode == 0
        result.update(
            ok=ok,
            returncode=process.returncode,
            message=(output.strip().splitlines() or ["命令已启动"])[0],
        )
    except Exception as exc:
        result.update(ok=False, message=f"找到文件但无法启动：{exc}")
    return result


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




def _match_length(audio: np.ndarray, length: int) -> np.ndarray:
    if len(audio) > length:
        return np.ascontiguousarray(audio[:length], dtype=np.float32)
    if len(audio) < length:
        return np.ascontiguousarray(
            np.pad(audio, ((0, length - len(audio)), (0, 0))), dtype=np.float32
        )
    return np.ascontiguousarray(audio, dtype=np.float32)


def _wsola_stereo(
    audio: np.ndarray,
    rate: float,
    frame_length: int = 2048,
    synthesis_hop: int = 512,
    search_radius: int = 256,
) -> np.ndarray:
    """Small-rate stereo WSOLA used for transient/percussive components."""
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    if audio.size == 0 or math.isclose(rate, 1.0, abs_tol=1e-4):
        return audio
    target_length = max(1, int(round(len(audio) / rate)))
    frame_length = int(min(frame_length, max(256, len(audio))))
    synthesis_hop = int(min(synthesis_hop, max(64, frame_length // 4)))
    overlap = frame_length - synthesis_hop
    analysis_hop = synthesis_hop * float(rate)
    window = np.sqrt(np.hanning(frame_length).astype(np.float32) + 1e-8)
    output = np.zeros((target_length + frame_length, audio.shape[1]), dtype=np.float32)
    weight = np.zeros(target_length + frame_length, dtype=np.float32)

    source_position = 0
    output_position = 0
    previous_frame = None
    while output_position < target_length:
        expected = int(round(source_position))
        expected = int(np.clip(expected, 0, max(0, len(audio) - frame_length)))
        best = expected
        if previous_frame is not None and overlap > 32:
            reference = np.mean(previous_frame[-overlap:], axis=1, dtype=np.float64)
            reference -= np.mean(reference)
            reference_norm = float(np.linalg.norm(reference)) + 1e-9
            start = max(0, expected - search_radius)
            end = min(max(0, len(audio) - frame_length), expected + search_radius)
            step = max(1, search_radius // 24)
            best_score = -np.inf
            for candidate in range(start, end + 1, step):
                probe = np.mean(audio[candidate : candidate + overlap], axis=1, dtype=np.float64)
                probe -= np.mean(probe)
                score = float(np.dot(reference, probe) / (reference_norm * (np.linalg.norm(probe) + 1e-9)))
                if score > best_score:
                    best_score = score
                    best = candidate

        frame = audio[best : best + frame_length]
        if len(frame) < frame_length:
            frame = np.pad(frame, ((0, frame_length - len(frame)), (0, 0)))
        end_out = output_position + frame_length
        output[output_position:end_out] += frame * window[:, None]
        weight[output_position:end_out] += window
        previous_frame = frame
        output_position += synthesis_hop
        source_position = best + analysis_hop
        if best >= len(audio) - frame_length and output_position >= target_length:
            break

    output = output[:target_length]
    weight = weight[:target_length]
    output /= np.maximum(weight[:, None], 1e-5)
    return np.ascontiguousarray(output, dtype=np.float32)


def _hybrid_hpss_stretch(
    audio: np.ndarray, sample_rate: int, rate: float
) -> np.ndarray:
    """
    Percussion-aware fallback inspired by selective-window TSM research.

    Harmonic material uses a phase vocoder while the percussive mask uses WSOLA,
    reducing kick/snare smearing without requiring an external binary.
    """
    target_length = max(1, int(round(len(audio) / rate)))
    try:
        harmonic, percussive = librosa.effects.hpss(
            np.ascontiguousarray(audio.T), margin=(1.0, 1.5)
        )
        harmonic_stretched = librosa.effects.time_stretch(harmonic, rate=float(rate)).T
        percussive_stretched = _wsola_stereo(percussive.T, rate=float(rate))
        harmonic_stretched = _match_length(harmonic_stretched, target_length)
        percussive_stretched = _match_length(percussive_stretched, target_length)
        result = harmonic_stretched + percussive_stretched
    except Exception:
        result = librosa.effects.time_stretch(
            np.ascontiguousarray(audio.T), rate=float(rate)
        ).T
        result = _match_length(result, target_length)
    peak = float(np.max(np.abs(result)) + 1e-9)
    if peak > 0.995:
        result *= np.float32(0.995 / peak)
    return np.ascontiguousarray(result, dtype=np.float32)


def stretch_stereo(
    audio: np.ndarray,
    sample_rate: int,
    rate: float,
    backend: str = "auto",
) -> tuple[np.ndarray, str]:
    """
    保持音高的立体声定速时间拉伸。

    `rate` 与 librosa 语义一致：大于 1 表示更快、输出更短。auto/high
    优先使用 Rubber Band R3，缺失时回退到 Hybrid HPSS + WSOLA。
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

    if requested in {"auto", "hybrid", "hybrid hpss", "percussion-aware"}:
        return (
            _hybrid_hpss_stretch(audio, sample_rate, float(rate)),
            "Hybrid HPSS + WSOLA",
        )

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
