"""和弦识别：谐波抑制 chroma 与序列平滑的单测。"""

from __future__ import annotations

import numpy as np
import pytest

from aitsm.pipeline import chords as chords_mod


def _state(symbol: str, confidence: float) -> dict:
    root = {
        "C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5,
        "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11,
    }[symbol[0] if len(symbol) == 1 or symbol[1] != "#" else symbol[:2]]
    return {
        "symbol": symbol,
        "score": confidence,
        "slot_confidence": confidence,
        "root_pc": root,
        "quality": "maj" if "m" not in symbol[1:] else "min",
    }


# --- 序列平滑 ----------------------------------------------------------------


def test_smooth_chord_sequence_removes_isolated_low_confidence():
    states = [
        _state("C", 0.8),
        _state("G", 0.40),  # 只占一槽、前后都是 C、置信度低 → 抖动
        _state("C", 0.8),
    ]
    smoothed, changed = chords_mod._smooth_chord_sequence(states)
    assert changed == 1
    assert [item["symbol"] for item in smoothed] == ["C", "C", "C"]
    # 入参不被就地修改
    assert states[1]["symbol"] == "G"


def test_smooth_chord_sequence_keeps_confident_passing_chord():
    """高置信度的孤立和弦可能是真实的一小节过门，必须保留。"""
    states = [_state("C", 0.8), _state("G", 0.85), _state("C", 0.8)]
    smoothed, changed = chords_mod._smooth_chord_sequence(states)
    assert changed == 0
    assert [item["symbol"] for item in smoothed] == ["C", "G", "C"]


def test_smooth_chord_sequence_needs_matching_neighbours():
    states = [_state("C", 0.8), _state("G", 0.4), _state("Am", 0.8)]
    smoothed, changed = chords_mod._smooth_chord_sequence(states)
    assert changed == 0
    assert smoothed[1]["symbol"] == "G"


def test_smooth_chord_sequence_handles_short_sequences():
    for length in (0, 1, 2):
        states = [_state("C", 0.8)] * length
        smoothed, changed = chords_mod._smooth_chord_sequence(states)
        assert changed == 0
        assert len(smoothed) == length


# --- 谐波抑制 chroma ---------------------------------------------------------


def test_harmonic_chroma_emphasises_fundamental_pitch_class():
    """440 Hz（A4）正弦：加权后 A 音级应明显压过其他音级。"""
    librosa = pytest.importorskip("librosa")
    sr = 22050
    times = np.arange(sr, dtype=float) / sr
    tone = 0.6 * np.sin(2 * np.pi * 440.0 * times)
    chroma = chords_mod._harmonic_chroma(librosa, tone, sr, np, 0.0)
    assert chroma.shape[0] == 12
    assert chroma.shape[1] > 0
    per_frame = np.linalg.norm(chroma, axis=0)
    assert np.allclose(per_frame, 1.0, atol=1e-6)  # 逐帧 L2 归一化
    dominant = chroma.mean(axis=1)
    assert int(np.argmax(dominant)) == 9  # A
    assert dominant[9] > 0.5


def test_harmonic_chroma_suppresses_high_octave_energy_vs_plain_chroma():
    """只喂高八度（1760 Hz）时，加权 chroma 的总能量应低于等权 CQT chroma。

    这里比的是「同一音级但高八度」被压低的程度：权重表给了 C1 0.10、C6 0.42，
    而 C7（3520 Hz）只有 0.20，所以纯高音输入不会像等权折叠那样点亮满值。
    """
    librosa = pytest.importorskip("librosa")
    sr = 22050
    times = np.arange(sr, dtype=float) / sr
    tone = 0.5 * np.sin(2 * np.pi * 1760.0 * times)
    weighted = chords_mod._harmonic_chroma(librosa, tone, sr, np, 0.0)
    plain = librosa.feature.chroma_cqt(
        y=tone, sr=sr, hop_length=chords_mod.HOP_LENGTH, bins_per_octave=36, n_chroma=12
    )
    peak_weighted = float(weighted.max())
    peak_plain = float(plain.max())
    assert peak_weighted < peak_plain  # 高八度被权重压下去


def test_harmonic_octave_weights_length():
    assert len(chords_mod._HARMONIC_OCTAVE_WEIGHTS) == 7
    assert chords_mod._HARMONIC_OCTAVE_WEIGHTS[2] == max(chords_mod._HARMONIC_OCTAVE_WEIGHTS)
