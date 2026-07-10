from __future__ import annotations

import numpy as np
import torch

from autodj.allinone_structure import fuse_allinone_structure
from autodj.models import (
    AllInOneProfile,
    BarFeatures,
    EDMStructure,
    FunctionalSegment,
)
from autodj.natten_compat import (
    natten1dav,
    natten1dqkrpb,
    natten2dav,
    natten2dqkrpb,
)


def test_allinone_profile_labels() -> None:
    profile = AllInOneProfile(
        segments=(
            FunctionalSegment(0.0, 8.0, "intro"),
            FunctionalSegment(8.0, 16.0, "chorus"),
            FunctionalSegment(16.0, 24.0, "outro"),
        ),
        backend="test",
    )
    assert profile.available
    assert profile.label_at(2.0) == "intro"
    assert profile.label_at(10.0) == "chorus"
    assert profile.label_at(30.0) == "outro"
    assert profile.unique_labels == ("intro", "chorus", "outro")


def test_fuse_allinone_labels_and_boundaries() -> None:
    count = 12
    starts = np.arange(count, dtype=np.int64) * 2000
    features = BarFeatures(
        start_samples=starts,
        end_samples=starts + 2000,
        rms=np.linspace(0.2, 1.0, count, dtype=np.float32),
        low_ratio=np.linspace(0.2, 0.8, count, dtype=np.float32),
        onset=np.linspace(0.2, 0.9, count, dtype=np.float32),
        brightness=np.linspace(0.3, 0.7, count, dtype=np.float32),
        vocal_proxy=np.linspace(0.7, 0.2, count, dtype=np.float32),
        chroma=np.ones((count, 12), dtype=np.float32),
    )
    base = EDMStructure(
        cue_score=np.zeros(count, dtype=np.float32),
        mix_in_score=np.zeros(count, dtype=np.float32),
        mix_out_score=np.zeros(count, dtype=np.float32),
        phrase_mask=np.zeros(count, dtype=np.float32),
        labels=tuple("SECTION" for _ in range(count)),
        edm_confidence=0.7,
    )
    profile = AllInOneProfile(
        segments=(
            FunctionalSegment(0.0, 8.0, "intro"),
            FunctionalSegment(8.0, 16.0, "chorus"),
            FunctionalSegment(16.0, 24.0, "outro"),
        ),
        backend="allin1-test",
    )
    result = fuse_allinone_structure(
        base,
        profile,
        features,
        source_downbeat_samples=starts,
        sample_rate=1000,
    )
    assert result.structure_source.startswith("All-In-One")
    assert "INTRO" in result.labels
    assert "OUTRO" in result.labels
    assert result.allin1_boundary_score.max() > 0.8
    assert result.mix_in_score.max() > 0.5
    assert result.mix_out_score.max() > 0.5


def test_pure_torch_natten_1d_shapes() -> None:
    query = torch.randn(1, 2, 8, 4)
    key = torch.randn(1, 2, 8, 4)
    value = torch.randn(1, 2, 8, 4)
    rpb = torch.randn(2, 5)
    scores = natten1dqkrpb(query, key, rpb, 3, 1)
    assert scores.shape == (1, 2, 8, 3)
    attention = torch.softmax(scores, dim=-1)
    output = natten1dav(attention, value, 3, 1)
    assert output.shape == query.shape
    assert torch.isfinite(output).all()


def test_pure_torch_natten_2d_shapes() -> None:
    query = torch.randn(1, 2, 5, 6, 3)
    key = torch.randn(1, 2, 5, 6, 3)
    value = torch.randn(1, 2, 5, 6, 3)
    rpb = torch.randn(2, 5, 5)
    scores = natten2dqkrpb(query, key, rpb, 3, 1)
    assert scores.shape == (1, 2, 5, 6, 9)
    attention = torch.softmax(scores, dim=-1)
    output = natten2dav(attention, value, 3, 1)
    assert output.shape == query.shape
    assert torch.isfinite(output).all()
