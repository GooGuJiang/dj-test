from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from autodj.allinone_analyzer import AllInOneAnalyzer
from autodj.audio_engine import AutoDJEngine, EngineConfig
from autodj.models import AllInOneProfile
from autodj.natten_compat import NattenCompatStatus


def test_allinone_auto_device_resolves_to_cpu() -> None:
    analyzer = AllInOneAnalyzer(device="auto")
    assert analyzer.device == "cpu"


def test_allinone_runs_without_subprocess_adapter() -> None:
    source = inspect.getsource(AllInOneAnalyzer.analyze_many)
    assert "subprocess" not in source
    assert "conda" not in source.lower()


def test_allinone_profile_conversion() -> None:
    result = SimpleNamespace(
        bpm=124,
        beats=[0.0, 0.5, 1.0],
        downbeats=[0.0],
        beat_positions=[1, 2, 3],
        segments=[
            SimpleNamespace(start=0.0, end=8.0, label="intro"),
            SimpleNamespace(start=8.0, end=32.0, label="chorus"),
        ],
    )
    profile = AllInOneAnalyzer._convert(
        result,
        "harmonix-all",
        NattenCompatStatus("torch-fallback", "autodj", False, "test"),
    )
    assert isinstance(profile, AllInOneProfile)
    assert profile.available
    assert profile.unique_labels == ("intro", "chorus")
    assert profile.natten_backend == "torch-fallback"


def test_engine_accepts_harmonix_models_and_cpu_device() -> None:
    engine = AutoDJEngine(EngineConfig())
    engine.set_allin1_model("harmonix-all")
    engine.set_allin1_model("harmonix-fold3")
    engine.set_allin1_device("cpu")
    assert engine.config.allin1_model_name == "harmonix-fold3"
    assert engine.config.allin1_device == "cpu"


def test_engine_rejects_unknown_allinone_model() -> None:
    engine = AutoDJEngine(EngineConfig())
    with pytest.raises(ValueError, match="未知 All-In-One 模型"):
        engine.set_allin1_model("ASLP-lab/SongFormer")


def test_no_songformer_runtime_files_remain() -> None:
    root = Path(__file__).parents[1]
    forbidden = [
        root / "songformer_worker.py",
        root / "install_songformer.py",
        root / "repair_songformer_cuda.py",
        root / "autodj" / "songformer_analyzer.py",
    ]
    assert not any(path.exists() for path in forbidden)


def test_analyze_many_reports_progress_without_subprocess(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    audio = tmp_path / "demo.wav"
    audio.write_bytes(b"fake")
    result = SimpleNamespace(
        path=audio,
        bpm=120,
        beats=[0.0, 0.5],
        downbeats=[0.0],
        beat_positions=[1, 2],
        segments=[SimpleNamespace(start=0.0, end=4.0, label="intro")],
    )

    class FakeAllInOne:
        @staticmethod
        def analyze(paths, **kwargs):
            assert paths == [str(audio.resolve())]
            assert kwargs["multiprocess"] is False
            return [result]

    analyzer = AllInOneAnalyzer(cache_path=tmp_path / "cache.json", device="cpu")

    def fake_import():
        analyzer._natten_status = NattenCompatStatus(
            "torch-fallback", "autodj", False, "test"
        )
        return FakeAllInOne

    monkeypatch.setattr(analyzer, "_import_backend", fake_import)
    updates = []
    profiles = analyzer.analyze_many(
        [audio], progress=lambda current, total, detail: updates.append((current, total, detail))
    )
    assert profiles[str(audio.resolve())].available
    assert updates[-1][0] == 1
    assert updates[-1][1] == 1
