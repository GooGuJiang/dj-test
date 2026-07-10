from __future__ import annotations

import numpy as np

from autodj.audio_engine import AutoDJEngine, EngineConfig
from autodj.cuedetr_analyzer import CueDETRAnalyzer
from autodj.cuedetr_structure import apply_cuedetr_cues
from autodj.edm_structure import analyze_edm_structure
from autodj.models import (
    BarFeatures,
    CueDETRProfile,
    EDMStructure,
    PreparedTrack,
    TrackAnalysis,
)
from autodj.transition_matcher import find_best_transition


def _features(count: int, bar_samples: int = 1000) -> BarFeatures:
    starts = np.arange(count, dtype=np.int64) * bar_samples
    ends = starts + bar_samples
    chroma = np.zeros((count, 12), dtype=np.float32)
    chroma[:, 0] = 1.0
    return BarFeatures(
        start_samples=starts,
        end_samples=ends,
        rms=np.linspace(0.25, 0.65, count, dtype=np.float32),
        low_ratio=np.full(count, 0.45, dtype=np.float32),
        onset=np.full(count, 0.65, dtype=np.float32),
        brightness=np.full(count, 0.40, dtype=np.float32),
        vocal_proxy=np.full(count, 0.18, dtype=np.float32),
        chroma=chroma,
    )


def _analysis(title: str, count: int) -> TrackAnalysis:
    return TrackAnalysis(
        path=f"{title}.wav",
        title=title,
        duration=float(count),
        bpm=120.0,
        beats_per_bar=4,
        beat_times=tuple(i * 0.25 for i in range(count * 4)),
        beat_numbers=tuple(i % 4 + 1 for i in range(count * 4)),
        downbeat_times=tuple(float(i) for i in range(count)),
    )


def _track(title: str, count: int, cue_indices: tuple[int, ...]) -> PreparedTrack:
    sr = 1000
    features = _features(count)
    total = count * 1000
    audio = np.zeros((total, 2), dtype=np.float32)
    cue = np.zeros(count, dtype=np.float32)
    for index in cue_indices:
        cue[index] = 0.95
    structure = EDMStructure(
        cue_score=cue,
        mix_in_score=cue.copy(),
        mix_out_score=cue.copy(),
        phrase_mask=cue.copy(),
        labels=tuple("SECTION" for _ in range(count)),
        functional_labels=tuple("unknown" for _ in range(count)),
    )
    beats = np.arange(count * 4, dtype=np.int64) * 250
    beat_numbers = np.tile(np.arange(1, 5, dtype=np.int64), count)
    downbeats = np.arange(count, dtype=np.int64) * 1000
    return PreparedTrack(
        analysis=_analysis(title, count),
        audio=audio,
        low_audio=audio.copy(),
        mid_audio=audio.copy(),
        high_audio=audio.copy(),
        source_audio=audio.copy(),
        sample_rate=sr,
        playback_bpm=120.0,
        original_bpm=120.0,
        stretch_rate=1.0,
        beat_samples=beats,
        beat_numbers=beat_numbers,
        downbeat_samples=downbeats,
        source_beat_samples=beats.copy(),
        source_beat_numbers=beat_numbers.copy(),
        source_downbeat_samples=downbeats.copy(),
        cue_sample=0,
        waveform_envelope=np.zeros(64, dtype=np.float32),
        bar_features=features,
        structure=structure,
        cuedetr_profile=CueDETRProfile(
            cue_times=tuple(float(i) for i in cue_indices),
            cue_scores=tuple(0.95 for _ in cue_indices),
            backend="test",
        ),
    )


def test_local_edm_analysis_no_longer_generates_transition_cues() -> None:
    features = _features(32)
    audio = np.zeros((32_000, 2), dtype=np.float32)
    structure = analyze_edm_structure(audio, 1000, features, beats_per_bar=4)
    assert np.count_nonzero(structure.cue_score) == 0
    assert np.count_nonzero(structure.mix_in_score) == 0
    assert np.count_nonzero(structure.mix_out_score) == 0
    assert "CUE-DETR" in structure.structure_source


def test_cuedetr_replaces_every_legacy_candidate() -> None:
    count = 24
    features = _features(count)
    legacy = np.ones(count, dtype=np.float32)
    base = EDMStructure(
        cue_score=legacy.copy(),
        mix_in_score=legacy.copy(),
        mix_out_score=legacy.copy(),
        phrase_mask=legacy.copy(),
        labels=tuple("SECTION" for _ in range(count)),
        functional_labels=tuple("unknown" for _ in range(count)),
    )
    profile = CueDETRProfile(
        cue_times=(4.1, 12.2),
        cue_scores=(0.92, 0.81),
        backend="test",
    )
    result = apply_cuedetr_cues(
        base,
        profile,
        features,
        np.arange(count, dtype=np.int64) * 1000,
        1000,
    )
    assert result.cue_indices.tolist() == [4, 12]
    assert np.count_nonzero(result.mix_in_score) == 2
    assert np.count_nonzero(result.mix_out_score) == 2


def test_cuedetr_predictions_snap_to_downbeats_and_minimum_bar_distance(tmp_path) -> None:
    analyzer = CueDETRAnalyzer(
        device="cpu",
        min_bars=8,
        cache_dir=tmp_path,
    )
    analysis = _analysis("snap", 32)
    cue_times, cue_scores = analyzer._snap_and_filter(  # noqa: SLF001
        np.asarray([1.1, 5.2, 10.1, 18.1]),
        np.asarray([0.70, 0.99, 0.90, 0.80]),
        analysis,
    )
    assert cue_times == (5.0, 18.0)
    assert cue_scores == (0.99, 0.8)


def test_matcher_and_simple_fallback_preserve_cuedetr_pair() -> None:
    current = _track("A", 64, (24, 40, 48))
    incoming = _track("B", 64, (0, 8, 16))
    plan = find_best_transition(current, incoming, requested_bars=8)
    assert plan.current_bar_index in {24, 40, 48}
    assert plan.next_bar_index in {0, 8, 16}

    engine = AutoDJEngine(EngineConfig(sample_rate=1000, automix_policy="Crossfade"))
    fallback = engine._simple_transition_plan(  # noqa: SLF001
        current,
        incoming,
        earliest_start=plan.current_start + 5000,
        base_plan=plan,
        mode="Gapless Trim",
    )
    assert fallback.current_start == plan.current_start
    assert fallback.next_start == plan.next_start
    assert fallback.policy_mode == "CUE-DETR short gapless"
