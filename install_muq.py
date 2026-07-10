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
        description="安装 OpenMuQ 官方推理库；权重首次使用时从 Hugging Face 下载"
    )
    parser.add_argument(
        "--skip-torch",
        action="store_true",
        help="已按显卡平台安装 PyTorch 时使用",
    )
    args = parser.parse_args()

    pip_install("--upgrade", "pip", "setuptools", "wheel")
    if not args.skip_torch and importlib.util.find_spec("torch") is None:
        pip_install("torch>=2.3")
    pip_install("muq", "huggingface-hub>=0.24", "safetensors>=0.4")

    print("\nMuQ 安装完成。模型 OpenMuQ/MuQ-large-msd-iter 会在首次分析时下载。")
    print("注意：官方模型权重采用 CC-BY-NC 4.0，仅限非商业用途。")
    print("接下来运行：python verify_install.py")


if __name__ == "__main__":
    main()
