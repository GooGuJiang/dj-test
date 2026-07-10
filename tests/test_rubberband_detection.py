from __future__ import annotations

import os
from pathlib import Path

from autodj.time_stretch import rubberband_executable, rubberband_probe


def _make_fake_cli(path: Path) -> None:
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "print('Rubber Band test CLI')\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def test_explicit_rubberband_directory_is_supported(tmp_path: Path) -> None:
    name = "rubberband.exe" if os.name == "nt" else "rubberband"
    executable = tmp_path / name
    _make_fake_cli(executable)
    found = rubberband_executable(tmp_path)
    assert found is not None
    assert Path(found).resolve() == executable.resolve()


def test_rubberband_environment_variable_is_supported(tmp_path: Path, monkeypatch) -> None:
    name = "rubberband.exe" if os.name == "nt" else "rubberband"
    executable = tmp_path / name
    _make_fake_cli(executable)
    monkeypatch.setenv("AUTODJ_RUBBERBAND", str(executable))
    found = rubberband_executable()
    assert found is not None
    assert Path(found).resolve() == executable.resolve()
    probe = rubberband_probe()
    assert probe["ok"] is True
