"""节奏量化（metrical HMM）单测：docstring 里承诺的每条性质都要有断言。"""

from __future__ import annotations

import math

import pytest

from aitsm.pipeline import quantize as qz


def _beats(bpm: float = 120.0, beats_per_bar: int = 4, count: int = 40) -> dict:
    period = 60.0 / bpm
    beats = [round(index * period, 6) for index in range(count)]
    bar = period * beats_per_bar
    downbeats = [round(index * bar, 6) for index in range(count // beats_per_bar + 1)]
    return {
        "bpm": bpm,
        "beats_per_bar": beats_per_bar,
        "beats": beats,
        "downbeats": downbeats,
    }


def _note(start: float, end: float, midi: int = 60) -> dict:
    return {"start": start, "end": end, "midi": midi, "confidence": 0.8}


# --- 重音等级与可记谱时值 ---------------------------------------------------


def test_metrical_strength_levels():
    assert qz.metrical_strength(0.0) == 3  # 正拍
    assert qz.metrical_strength(4.0) == 3  # 下一小节正拍
    assert qz.metrical_strength(1.0) == 2  # 拍
    assert qz.metrical_strength(2.5) == 1  # 半拍
    assert qz.metrical_strength(1.25) == 0  # 十六分位置


def test_duration_cost_prefers_legal_values():
    for legal in qz.LEGAL_DURATIONS:
        assert qz.duration_cost(legal) == pytest.approx(0.0, abs=1e-9)
    # 1/6 拍（三连音网格漂移出来的时值）必须比合法值贵
    assert qz.duration_cost(1.0 / 6.0) > 0.2
    # 太短（1/32 拍）额外罚
    assert qz.duration_cost(0.125) > qz.duration_cost(0.25)
    assert math.isinf(qz.duration_cost(0.0))


# --- HMM 量化 ----------------------------------------------------------------


def test_quantize_produces_only_notatable_durations():
    """核心承诺：输出时值全部可记谱——这是不再需要导出层「强行吸附」的前提。

    两个音相隔 0.06 s（120 BPM 下 0.12 拍）：就近吸附会得到 1/8 拍（32 分音符），
    HMM 必须给出 ≥1/16 的合法时值。
    """
    beats = _beats(bpm=120.0)
    notes = [_note(0.02, 0.08), _note(0.08, 0.52), _note(0.53, 1.01)]
    quantized, stats = qz.quantize_notes_hmm(notes, beats)
    assert len(quantized) == len(notes)
    period = 0.5
    for note in quantized:
        duration = (note["end"] - note["start"]) / period
        nearest = min(qz.LEGAL_DURATIONS, key=lambda legal: abs(legal - duration))
        assert abs(duration - nearest) < 1e-6, f"时值 {duration:.4f} 拍不可记谱"
    assert stats["legal_duration_ratio"] >= 0.99


def test_quantize_starts_are_monotonic_and_raw_kept():
    beats = _beats(bpm=100.0)
    notes = [_note(0.05, 0.4), _note(0.42, 0.9), _note(0.91, 1.4)]
    quantized, _stats = qz.quantize_notes_hmm(notes, beats)
    starts = [note["start"] for note in quantized]
    assert starts == sorted(starts)
    assert len(set(starts)) == len(starts)
    for before, after in zip(notes, quantized):
        assert after["start_raw"] == pytest.approx(before["start"], abs=1e-3)
        assert after["end_raw"] == pytest.approx(before["end"], abs=1e-3)


def test_quantize_snaps_offbeat_notes_to_beats():
    """120 BPM、拍长 0.5 s：起音偏 20 ms 的音应被吸到拍点上。"""
    beats = _beats(bpm=120.0)
    notes = [_note(0.02, 0.48), _note(0.52, 0.98), _note(1.02, 1.48)]
    quantized, stats = qz.quantize_notes_hmm(notes, beats)
    assert [note["start"] for note in quantized] == [0.0, 0.5, 1.0]
    assert stats["mean_shift_ms"] > 0.0
    assert stats["snapped"] == 3


def test_quantize_harmony_anchor_wins_tie():
    """和弦变化点作为「多声部一致」证据：距离相同时把起音拉到换和弦的位置。

    起音 0.3125 到 0.25（半拍位，便宜）与 0.375（十六分位，贵）等距。
    没有锚点时按节拍重音选 0.25；把和弦变化放在 0.375，锚点奖励应当翻盘。
    """
    beats = _beats(bpm=120.0)
    notes = [_note(0.3125, 0.49), _note(0.52, 0.98)]
    plain, _ = qz.quantize_notes_hmm(notes, beats)
    anchored, stats = qz.quantize_notes_hmm(notes, beats, harmony_onsets=[0.0, 0.375])
    assert plain[0]["start"] == pytest.approx(0.25)
    assert anchored[0]["start"] == pytest.approx(0.375)
    assert stats["harmony_anchored"] >= 1


def test_quantize_without_beats_is_passthrough():
    notes = [_note(0.11, 0.4)]
    quantized, stats = qz.quantize_notes_hmm(notes, None)
    assert quantized[0]["start"] == pytest.approx(0.11, abs=1e-6)
    assert stats["engine"] == "metrical-hmm"
    assert stats["grid_points"] == 0


def test_quantize_empty_and_unvoiced_inputs():
    quantized, stats = qz.quantize_notes_hmm([], _beats())
    assert quantized == []
    assert stats["legal_duration_ratio"] == 0.0


def test_quantize_rest_after_long_gap_is_kept_as_rest():
    """长间隔（>0.5 拍）不被强制连奏：休止后的音时值吸到可记谱值。"""
    beats = _beats(bpm=120.0)
    notes = [_note(0.02, 0.30), _note(1.02, 1.48)]
    quantized, stats = qz.quantize_notes_hmm(notes, beats)
    assert stats["rests"] >= 1
    first = quantized[0]
    duration = (first["end"] - first["start"]) / 0.5
    nearest = min(qz.LEGAL_DURATIONS, key=lambda legal: abs(legal - duration))
    assert abs(duration - nearest) < 1e-6


def test_quantizer_enabled_env_switch(monkeypatch):
    monkeypatch.delenv("AITSM_RHYTHM_QUANTIZER", raising=False)
    assert qz.quantizer_enabled() is True
    monkeypatch.setenv("AITSM_RHYTHM_QUANTIZER", "grid")
    assert qz.quantizer_enabled() is False


def test_melody_quantize_or_fallback_uses_hmm(monkeypatch):
    """melody 里的接线：默认走 HMM，AITSM_RHYTHM_QUANTIZER=grid 时退回就近吸附。"""
    from aitsm.pipeline import melody as melody_mod

    beats = _beats(bpm=120.0)
    notes = [_note(0.02, 0.48), _note(0.52, 0.98)]
    monkeypatch.delenv("AITSM_RHYTHM_QUANTIZER", raising=False)
    hmm, stats = melody_mod._quantize_or_fallback(
        notes, beats, until=2.0, harmony_onsets=None, warnings=[]
    )
    assert stats["engine"] == "metrical-hmm"
    monkeypatch.setenv("AITSM_RHYTHM_QUANTIZER", "grid")
    legacy, stats2 = melody_mod._quantize_or_fallback(
        notes, beats, until=2.0, harmony_onsets=None, warnings=[]
    )
    assert "engine" not in stats2 or stats2.get("engine") != "metrical-hmm"
    assert len(legacy) == len(hmm) == 2
