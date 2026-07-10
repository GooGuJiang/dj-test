from __future__ import annotations

import argparse
import subprocess
import sys


def run_pip(*args: str) -> None:
    command = [sys.executable, "-m", "pip", *args]
    print("\n>", " ".join(command))
    subprocess.check_call(command)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="安装 All-In-One 1.1.0 及 Auto DJ 所需兼容依赖"
    )
    parser.add_argument(
        "--skip-madmom",
        action="store_true",
        help="已有可用 madmom 时跳过安装",
    )
    args = parser.parse_args()

    run_pip("install", "--upgrade", "pip", "wheel", "setuptools<76", "Cython<3")
    run_pip(
        "install",
        "demucs>=4.0",
        "hydra-core>=1.3",
        "omegaconf>=2.3",
        "huggingface-hub>=0.24",
        "tqdm>=4.66",
    )
    if not args.skip_madmom:
        # The All-In-One README recommends current madmom from GitHub rather
        # than the old PyPI release.
        run_pip(
            "install",
            "--no-build-isolation",
            "git+https://github.com/CPJKU/madmom.git",
        )

    # Prevent pip from installing/upgrading NATTEN and PyTorch automatically.
    # The application keeps a compatible native NATTEN 0.15-0.19 if present,
    # otherwise it injects a pure-PyTorch legacy-op fallback.
    run_pip("install", "--no-deps", "allin1==1.1.0")

    print("\nAll-In-One 安装完成。下一步运行：")
    print("  python verify_allinone.py")
    print("  python app.py")


if __name__ == "__main__":
    main()
