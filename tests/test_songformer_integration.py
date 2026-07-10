from pathlib import Path

from autodj.models import SongFormerProfile
from autodj.songformer_analyzer import SongFormerAnalyzer
from songformer_worker import _normalise_segments


def test_songformer_normalises_official_output():
    raw = [
        {"start": 0.0, "end": 12.5, "label": "intro"},
        {"start": 12.5, "end": 44.0, "label": "chorus"},
    ]
    result = _normalise_segments(raw)
    assert result[0]["label"] == "intro"
    assert result[1]["end"] == 44.0


def test_songformer_profile_conversion():
    profile = SongFormerAnalyzer._convert(
        Path("track.wav"),
        {
            "backend": "songformer-hf-official",
            "segments": [
                {"start": 0.0, "end": 8.0, "label": "intro", "confidence": 1.0},
                {"start": 8.0, "end": 32.0, "label": "verse", "confidence": 1.0},
            ],
        },
        "ASLP-lab/SongFormer",
    )
    assert isinstance(profile, SongFormerProfile)
    assert profile.available
    assert profile.backend == "songformer-hf-official"
    assert profile.unique_labels == ("intro", "verse")
