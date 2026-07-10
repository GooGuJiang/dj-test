from __future__ import annotations

import json
import os
import sys

from autodj.time_stretch import rubberband_probe


def main() -> None:
    result = rubberband_probe()
    print("Python:", sys.executable)
    print("CONDA_PREFIX:", os.environ.get("CONDA_PREFIX", "<未设置>"))
    print("显式 Rubber Band 环境变量:")
    for name, value in result.get("environment", {}).items():
        print(f"  {name}={value or '<未设置>'}")
    print()
    if result.get("ok"):
        print("Rubber Band R3/CLI 已找到并可启动：")
        print(" ", result.get("executable"))
        print("探测信息：", result.get("message"))
        print("GUI 中选择 auto 或 Rubber Band R3 即可使用。")
    elif result.get("executable"):
        print("找到了文件，但进程无法正常启动：")
        print(" ", result.get("executable"))
        print("原因：", result.get("message"))
    else:
        print("没有找到 Rubber Band CLI。")
        print("Windows 请注意：PATH 中应加入 rubberband.exe 所在目录，而不是 exe 文件本身。")
        print("也可以设置：")
        print(r'  set AUTODJ_RUBBERBAND=C:\\tools\\rubberband\\rubberband.exe')
        print("或者在 GUI 中手动选择 rubberband.exe。")


if __name__ == "__main__":
    main()
