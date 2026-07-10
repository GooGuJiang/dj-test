from __future__ import annotations

import argparse
import json

from autodj.songformer_analyzer import probe_songformer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default="")
    parser.add_argument("--env-name", default="songformer-auto-dj")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    result = probe_songformer(args.python or None, args.env_name)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result.get("ok"):
        raise SystemExit(2)
    if args.require_cuda and not result.get("cuda_usable"):
        print("\nCUDA 不可用。RTX 5070 请运行：python repair_songformer_cuda.py")
        raise SystemExit(3)
    if result.get("cuda_usable"):
        print(
            "\nSongFormer CUDA 可用："
            f"{result.get('device_name')} · {result.get('capability')} · "
            f"PyTorch {result.get('torch')} / CUDA {result.get('cuda_build')}"
        )
    else:
        print("\nSongFormer worker 环境可用，但当前只能使用 CPU。")
    print("首次实际分析时会下载官方权重。")


if __name__ == "__main__":
    main()
