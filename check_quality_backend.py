from __future__ import annotations

from autodj.time_stretch import rubberband_executable


def main() -> None:
    executable = rubberband_executable()
    if executable:
        print("Rubber Band R3/CLI 已找到：", executable)
        print("GUI 中将时间拉伸质量设置为 auto 或 Rubber Band R3 即可使用。")
    else:
        print("没有找到 rubberband-r3 或 rubberband 可执行文件。")
        print("程序仍可运行：定速使用 librosa，BPM 回归使用单次 STFT 连续可变 phase-vocoder。")
        print("需要更高质量时，请安装 Rubber Band 命令行程序并加入 PATH。")


if __name__ == "__main__":
    main()
