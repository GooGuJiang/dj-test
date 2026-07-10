"""生成三首可用于检查界面和切歌流程的合成节拍 WAV。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def make_track(
    path: Path,
    bpm: float,
    duration: float,
    root_hz: float,
    sample_rate: int = 44_100,
) -> None:
    total = int(duration * sample_rate)
    t = np.arange(total, dtype=np.float64) / sample_rate

    # 低音和弦底色
    music = (
        0.08 * np.sin(2 * np.pi * root_hz * t)
        + 0.04 * np.sin(2 * np.pi * root_hz * 1.5 * t)
    )

    beat_interval = 60.0 / bpm
    click_length = int(0.05 * sample_rate)
    envelope = np.exp(-np.arange(click_length) / (0.008 * sample_rate))

    for beat_index, beat_time in enumerate(
        np.arange(0.5, duration - 0.1, beat_interval)
    ):
        start = int(beat_time * sample_rate)
        length = min(click_length, total - start)
        if length <= 0:
            continue
        frequency = 1100.0 if beat_index % 4 == 0 else 700.0
        click_t = np.arange(length) / sample_rate
        click = np.sin(2 * np.pi * frequency * click_t) * envelope[:length]
        music[start : start + length] += 0.55 * click

    # 轻微左右差异
    stereo = np.column_stack(
        (
            music,
            music * 0.96 + 0.01 * np.sin(2 * np.pi * (root_hz * 2) * t),
        )
    )
    peak = np.max(np.abs(stereo))
    stereo = (0.9 * stereo / max(peak, 1e-8)).astype(np.float32)
    sf.write(path, stereo, sample_rate, subtype="PCM_24")


def main() -> None:
    output = Path("demo_tracks")
    output.mkdir(exist_ok=True)
    specs = [
        ("demo_01_120bpm.wav", 120.0, 48.0, 110.0),
        ("demo_02_124bpm.wav", 124.0, 48.0, 130.81),
        ("demo_03_118bpm.wav", 118.0, 48.0, 146.83),
    ]
    for name, bpm, duration, root in specs:
        path = output / name
        make_track(path, bpm, duration, root)
        print(path)


if __name__ == "__main__":
    main()
