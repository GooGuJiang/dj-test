from __future__ import annotations

import subprocess
import sys


def run(*args: str) -> None:
    print("+", sys.executable, "-m", "pip", *args, flush=True)
    subprocess.check_call([sys.executable, "-m", "pip", *args])


if __name__ == "__main__":
    # Do not reinstall torch: the main environment may already contain a CUDA
    # build for RTX 5070. CUE-DETR only needs Transformers and Pillow on top.
    run("install", "transformers>=4.42.3,<5", "pillow>=10.4,<12", "timm>=1.0.7,<2")

    # Reproduce the real application import order: All-In-One may first install
    # its pure-PyTorch NATTEN compatibility module, then CUE-DETR imports
    # Transformers. This catches cross-backend import conflicts immediately.
    from autodj.natten_compat import ensure_natten_compat
    from autodj.cuedetr_analyzer import probe_cuedetr

    ensure_natten_compat(force_torch=True)
    result = probe_cuedetr()
    print("CUE-DETR coexistence check:", result, flush=True)
    if not result.get("ok"):
        raise SystemExit(1)
    print("CUE-DETR dependencies installed. The official checkpoint downloads on first analysis.")
