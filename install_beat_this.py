from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys


def pip_install(*args: str) -> None:
    command = [sys.executable, "-m", "pip", "install", *args]
    print("\n>", " ".join(command))
    subprocess.check_call(command)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="安装 Beat This! 1.1.0 及其推理依赖"
    )
    parser.add_argument(
        "--skip-torch",
        action="store_true",
        help="已按显卡平台自行安装 PyTorch/torchaudio 时使用",
    )
    args = parser.parse_args()

    pip_install("--upgrade", "pip", "setuptools", "wheel")
    if not args.skip_torch and (
        importlib.util.find_spec("torch") is None
        or importlib.util.find_spec("torchaudio") is None
    ):
        # 默认安装官方 PyPI 构建。CUDA 用户可先按 pytorch.org 指令安装，
        # 然后执行本脚本 --skip-torch。
        pip_install("torch>=2.3", "torchaudio>=2.3")

    pip_install(
        "beat-this==1.1.0",
        "einops>=0.8",
        "rotary-embedding-torch>=0.6",
        "soxr>=0.3",
        "tqdm>=4.66",
    )
    print("\nBeat This! 安装完成。接下来运行：")
    print("  python verify_install.py")
    print("  python app.py")


if __name__ == "__main__":
    main()
