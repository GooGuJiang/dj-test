from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


TORCH_VERSION = "2.7.1"
CUDA_INDEX = "https://download.pytorch.org/whl/cu128"


def run(command: list[str]) -> None:
    print("\n>", " ".join(command), flush=True)
    subprocess.check_call(command)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Repair SongFormer CUDA runtime for RTX 50-series / Blackwell GPUs."
    )
    parser.add_argument("--env-name", default="songformer-auto-dj")
    parser.add_argument("--torch-version", default=TORCH_VERSION)
    args = parser.parse_args()

    conda = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if not conda:
        raise SystemExit("未找到 conda。")
    base = [conda, "run", "--no-capture-output", "-n", args.env_name, "python", "-m", "pip"]
    run(base + ["install", "--upgrade", "pip"])
    run(
        base
        + [
            "install",
            "--upgrade",
            "--force-reinstall",
            f"torch=={args.torch_version}",
            f"torchaudio=={args.torch_version}",
            "--index-url",
            CUDA_INDEX,
        ]
    )
    verify = Path(__file__).resolve().parent / "verify_songformer.py"
    run([sys.executable, str(verify), "--require-cuda", "--env-name", args.env_name])
    print("\nCUDA 修复完成。重新启动 python app.py。")


if __name__ == "__main__":
    main()
