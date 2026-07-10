from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path


DEFAULT_TORCH_VERSION = "2.7.1"
DEFAULT_CUDA_INDEX = "https://download.pytorch.org/whl/cu128"
CPU_INDEX = "https://download.pytorch.org/whl/cpu"


def run(command: list[str]) -> None:
    print("\n>", " ".join(command), flush=True)
    subprocess.check_call(command)


def conda_env_exists(conda: str, env_name: str) -> bool:
    process = subprocess.run(
        [conda, "env", "list", "--json"],
        capture_output=True,
        text=True,
        check=False,
    )
    if process.returncode != 0:
        return False
    try:
        paths = json.loads(process.stdout).get("envs", [])
    except json.JSONDecodeError:
        return False
    return any(Path(path).name.lower() == env_name.lower() for path in paths)


def has_nvidia_driver() -> bool:
    command = shutil.which("nvidia-smi")
    if not command:
        return False
    return subprocess.run(
        [command, "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    ).returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create or repair the isolated official SongFormer runtime."
    )
    parser.add_argument("--env-name", default="songformer-auto-dj")
    parser.add_argument(
        "--compute",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="auto uses CUDA 12.8 when an NVIDIA driver is detected.",
    )
    parser.add_argument("--torch-version", default=DEFAULT_TORCH_VERSION)
    parser.add_argument("--torch-index-url", default="")
    parser.add_argument("--skip-torch", action="store_true")
    args = parser.parse_args()

    conda = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if not conda:
        raise SystemExit("未找到 conda。请安装 Miniconda/Anaconda 后重试。")

    env_name = args.env_name
    if not conda_env_exists(conda, env_name):
        run([conda, "create", "-n", env_name, "python=3.10", "pip", "-y"])
    else:
        print(f"检测到已有环境 {env_name}，将在原环境中修复/升级。", flush=True)

    base = [conda, "run", "--no-capture-output", "-n", env_name, "python", "-m", "pip"]
    run(base + ["install", "--upgrade", "pip", "setuptools", "wheel"])

    if not args.skip_torch:
        wants_cuda = args.compute == "cuda" or (
            args.compute == "auto"
            and platform.system() in {"Windows", "Linux"}
            and has_nvidia_driver()
        )
        index_url = args.torch_index_url or (DEFAULT_CUDA_INDEX if wants_cuda else CPU_INDEX)
        torch_version = args.torch_version
        print(
            f"安装 PyTorch {torch_version} · "
            f"{'CUDA 12.8' if 'cu128' in index_url else 'CPU'}",
            flush=True,
        )
        run(
            base
            + [
                "install",
                "--upgrade",
                "--force-reinstall",
                f"torch=={torch_version}",
                f"torchaudio=={torch_version}",
                "--index-url",
                index_url,
            ]
        )

    requirements = Path(__file__).resolve().parent / "requirements_songformer_runtime.txt"
    run(base + ["install", "-r", str(requirements)])

    print("\nSongFormer 独立环境已安装/修复。下一步：")
    print("  python verify_songformer.py --require-cuda")
    print("  python app.py")
    print("RTX 5070 应显示 CUDA 12.8、GPU 名称和 cuda_usable=true。")


if __name__ == "__main__":
    main()
