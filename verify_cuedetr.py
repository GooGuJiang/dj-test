from __future__ import annotations

import argparse

from autodj.cuedetr_analyzer import CueDETRAnalyzer, probe_cuedetr
from autodj.natten_compat import ensure_natten_compat


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true", help="also download and load the official checkpoint")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    # Match the GUI import order and verify coexistence with All-In-One's
    # compatibility backend, not just a clean standalone Transformers import.
    natten = ensure_natten_compat(force_torch=True)
    print("All-In-One compatibility backend:", natten)
    result = probe_cuedetr()
    print("CUE-DETR probe:", result)
    if not result.get("ok"):
        raise SystemExit(1)
    if args.download:
        analyzer = CueDETRAnalyzer(device=args.device)
        processor, model = analyzer._get_model()
        print("Loaded:", analyzer.model_name, "device=", analyzer.device)
        print("Processor:", processor.__class__.__name__)
        print("Model:", model.__class__.__name__)
