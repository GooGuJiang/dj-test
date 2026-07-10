from __future__ import annotations

import argparse
from pathlib import Path

from autodj.allinone_analyzer import AllInOneAnalyzer, probe_allinone


def main() -> None:
    parser = argparse.ArgumentParser(description="验证 All-In-One Auto DJ 集成")
    parser.add_argument(
        "audio",
        nargs="?",
        help="可选：实际分析一首 WAV/FLAC/MP3，验证模型权重和 Demucs",
    )
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps"))
    parser.add_argument("--model", default="harmonix-all")
    parser.add_argument(
        "--force-torch-natten",
        action="store_true",
        help="强制测试纯 PyTorch NATTEN 兼容层",
    )
    args = parser.parse_args()

    result = probe_allinone(force_torch_natten=args.force_torch_natten)
    print("All-In-One probe:")
    for key, value in result.items():
        print(f"  {key}: {value}")
    if not result.get("ok"):
        raise SystemExit("All-In-One 导入失败；请运行 python install_allinone.py")

    if args.audio:
        path = Path(args.audio).expanduser().resolve()
        analyzer = AllInOneAnalyzer(
            model=args.model,
            device=args.device,
            force_torch_natten=args.force_torch_natten,
        )
        profile = analyzer.analyze(path, status=print, force=True)
        print(f"\nBPM: {profile.bpm:.1f}")
        print(f"Segments: {len(profile.segments)}")
        for segment in profile.segments:
            print(f"  {segment.start:7.2f} - {segment.end:7.2f}  {segment.label}")
    else:
        print("\n导入验证通过。可附加音频路径进行完整模型测试：")
        print("  python verify_allinone.py demo_tracks/demo_01_120bpm.wav")


if __name__ == "__main__":
    main()
