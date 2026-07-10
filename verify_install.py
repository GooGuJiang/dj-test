from __future__ import annotations

import platform
import sys

import librosa
import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
import torchaudio

from beat_this.inference import File2Beats
from beat_this.utils import infer_beat_numbers
from autodj.time_stretch import rubberband_probe
from autodj.allinone_analyzer import probe_allinone


def main() -> None:
    print("Python:", sys.version.split()[0])
    print("Platform:", platform.platform())
    print("NumPy:", np.__version__)
    print("librosa:", librosa.__version__)
    print("SoundFile:", sf.__version__)
    print("sounddevice:", sd.__version__)
    print("PyTorch:", torch.__version__)
    print("torchaudio:", torchaudio.__version__)

    # small0 仅约 8MB，适合作为首次安装验证。模型会自动下载并缓存。
    tracker = File2Beats(
        checkpoint_path="small0",
        device="cpu",
        float16=False,
        dbn=False,
    )
    assert callable(tracker)
    sample_numbers = infer_beat_numbers(
        np.asarray([0.0, 0.5, 1.0, 1.5, 2.0]),
        np.asarray([0.0, 2.0]),
    )
    print("Beat This!: small0 加载成功")
    print("Beat number helper:", sample_numbers.tolist())

    try:
        from muq import MuQ  # noqa: F401
        print("MuQ: 官方 Python 包可导入")
        print("MuQ model: OpenMuQ/MuQ-large-msd-iter（首次分析自动下载）")
    except Exception as exc:
        print("MuQ: 未安装或导入失败：", exc)
        print("运行 python install_muq.py 后可启用风格排序；Beat This! 播放仍可使用。")

    aio = probe_allinone()
    if aio.get("ok"):
        print(
            "All-In-One:", aio.get("allin1_version"),
            "· NATTEN", aio.get("natten_backend")
        )
    else:
        print("All-In-One: 未安装；运行 python install_allinone.py")
        print("All-In-One 诊断:", aio.get("message"))

    rb = rubberband_probe()
    if rb.get("ok"):
        print("Rubber Band R3 time-map:", rb.get("executable"))
    elif rb.get("executable"):
        print("Rubber Band: 找到但无法启动：", rb.get("message"))
    else:
        print("Rubber Band: 未检测到，将使用 Hybrid HPSS/连续变速回退")
        print("可在 GUI 中手动选择 rubberband.exe，或设置 AUTODJ_RUBBERBAND。")

    try:
        output_devices = [
            item["name"]
            for item in sd.query_devices()
            if int(item.get("max_output_channels", 0)) > 0
        ]
        print(f"音频输出设备数量: {len(output_devices)}")
    except Exception as exc:
        print("音频设备枚举警告:", exc)

    print("\n安装验证通过，可以运行 python app.py")


if __name__ == "__main__":
    main()
