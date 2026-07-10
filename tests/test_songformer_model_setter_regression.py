from __future__ import annotations

import pytest

from autodj.audio_engine import AutoDJEngine, EngineConfig


def test_official_songformer_model_is_accepted() -> None:
    engine = AutoDJEngine(EngineConfig())
    engine.set_songformer_model("ASLP-lab/SongFormer")
    assert engine.config.songformer_model_name == "ASLP-lab/SongFormer"


def test_old_harmonix_model_name_is_rejected() -> None:
    engine = AutoDJEngine(EngineConfig())
    with pytest.raises(ValueError, match="仅支持官方模型"):
        engine.set_songformer_model("harmonix-all")


def test_old_allin1_setter_no_longer_exists() -> None:
    engine = AutoDJEngine(EngineConfig())
    assert not hasattr(engine, "set_allin1_model")
