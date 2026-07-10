from __future__ import annotations

import numpy as np

from autodj.models import BarFeatures, EDMStructure, PreparedTrack, TrackAnalysis
from autodj.transition_matcher import TransitionFXConfig, find_best_transition


def make_track(
    title: str,
    energy: np.ndarray,
    chroma_pitch: int,
    playback_bpm: float = 120.0,
    original_bpm: float = 120.0,
) -> PreparedTrack:
    sr = 1000
    bars = len(energy)
    bar_length = 1000
    total = bars * bar_length
    audio = np.zeros((total, 2), dtype=np.float32)
    starts = np.arange(bars, dtype=np.int64) * bar_length
    ends = starts + bar_length
    chroma = np.zeros((bars, 12), dtype=np.float32)
    chroma[:, chroma_pitch] = 1.0
    features = BarFeatures(
        start_samples=starts,
        end_samples=ends,
        rms=np.asarray(energy, dtype=np.float32),
        low_ratio=np.full(bars, 0.25, dtype=np.float32),
        onset=np.full(bars, 0.45, dtype=np.float32),
        brightness=np.full(bars, 0.40, dtype=np.float32),
        vocal_proxy=np.full(bars, 0.20, dtype=np.float32),
        chroma=chroma,
    )
    analysis = TrackAnalysis(
        path=f"{title}.wav",
        title=title,
        duration=float(bars),
        bpm=original_bpm,
        beats_per_bar=4,
        beat_times=tuple(float(i) * 0.25 for i in range(bars * 4)),
        beat_numbers=tuple((i % 4) + 1 for i in range(bars * 4)),
        downbeat_times=tuple(float(i) for i in range(bars)),
    )
    beat_samples = np.arange(bars * 4, dtype=np.int64) * 250
    beat_numbers = np.tile(np.arange(1, 5), bars)
    cue = np.zeros(bars, dtype=np.float32)
    cue[np.arange(0, bars, 8)] = 0.95
    cue[max(0, bars - 16)] = 0.95
    structure = EDMStructure(
        cue_score=cue, mix_in_score=cue.copy(), mix_out_score=cue.copy(),
        phrase_mask=cue.copy(), labels=tuple("SECTION" for _ in range(bars)),
    )
    return PreparedTrack(
        analysis=analysis,
        audio=audio,
        low_audio=audio.copy(),
        mid_audio=audio.copy(),
        high_audio=audio.copy(),
        source_audio=audio.copy(),
        sample_rate=sr,
        playback_bpm=playback_bpm,
        original_bpm=original_bpm,
        stretch_rate=playback_bpm / original_bpm,
        beat_samples=beat_samples,
        beat_numbers=beat_numbers,
        downbeat_samples=starts,
        source_beat_samples=beat_samples.copy(),
        source_beat_numbers=beat_numbers.copy(),
        source_downbeat_samples=starts.copy(),
        cue_sample=0,
        waveform_envelope=np.zeros(64, dtype=np.float32),
        bar_features=features,
        structure=structure,
    )


def test_auto_transition_returns_professional_curves() -> None:
    a_energy = np.concatenate([np.full(32, 0.5), np.linspace(0.55, 0.18, 32)])
    b_energy = np.concatenate([np.linspace(0.18, 0.55, 32), np.full(32, 0.5)])
    a = make_track("A", a_energy, 0)
    b = make_track("B", b_energy, 0)

    plan = find_best_transition(
        a,
        b,
        requested_bars=0,
        fx_config=TransitionFXConfig(style="Club", strength=0.8),
    )

    assert plan.bars in (8, 16, 32)
    assert plan.current_start >= int(a.total_samples * 0.35)
    assert plan.next_start <= int(b.total_samples * 0.5)
    assert 0.0 <= plan.score <= 1.0
    for curve in (
        plan.fade_out,
        plan.fade_in,
        plan.bass_out,
        plan.bass_in,
        plan.high_out,
        plan.high_in,
    ):
        assert len(curve) == plan.length
        assert np.isfinite(curve).all()
    assert plan.echo_audio.shape == (plan.length, 2)
    assert plan.fade_out[0] > 0.99
    assert plan.fade_out[-1] < 0.01
    assert plan.bass_out[0] > 0.99
    assert plan.bass_in[-1] > 0.99
    assert plan.style == "Club"


def test_fixed_transition_length_is_respected() -> None:
    a = make_track("A", np.linspace(0.6, 0.2, 64), 0)
    b = make_track("B", np.linspace(0.2, 0.6, 64), 7)
    plan = find_best_transition(a, b, requested_bars=8)
    assert plan.bars == 8


def test_bpm_at_sample_returns_to_original() -> None:
    track = make_track(
        "tempo",
        np.full(64, 0.5),
        0,
        playback_bpm=128.0,
        original_bpm=124.0,
    )
    track.tempo_restore_start = 10_000
    track.tempo_restore_end = 20_000
    track.tempo_restore_bars = 8

    assert track.bpm_at_sample(5_000) == 128.0
    assert 124.0 < track.bpm_at_sample(15_000) < 128.0
    assert track.bpm_at_sample(25_000) == 124.0
