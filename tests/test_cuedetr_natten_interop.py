from __future__ import annotations

import importlib.util
import sys

from autodj.cuedetr_analyzer import probe_cuedetr
from autodj.natten_compat import ensure_natten_compat


def test_torch_natten_fallback_has_valid_module_specs() -> None:
    status = ensure_natten_compat(force_torch=True)
    assert status.backend == "torch-fallback"
    assert sys.modules["natten"].__spec__ is not None
    assert sys.modules["natten.functional"].__spec__ is not None
    assert importlib.util.find_spec("natten") is not None
    assert importlib.util.find_spec("natten.functional") is not None


def test_cuedetr_probe_after_allinone_natten_fallback() -> None:
    ensure_natten_compat(force_torch=True)
    result = probe_cuedetr()
    # In the packaged runtime Transformers may or may not be installed during
    # source-only tests, but the old ValueError must never be reported again.
    assert "__spec__ is None" not in str(result.get("message", ""))
    assert "natten.__spec__" not in str(result.get("message", ""))
